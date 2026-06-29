"""Phase 4 – Evaluate: load trained estimators, compute offloading metrics."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ApproachConfig, EstimatorConfig, PipelineConfig

from .. import log
from ..models import ESTIMATOR_REGISTRY
from ..offloader import (ConfiguredOffloader, OffloadContext, OffloadDecision,
                         classify_metric)
from ._shared import resolve_target
from .eval_cache import (
    _EVAL_CACHE_DIR,
    _eval_cache_dir_for,
    _eval_cache_signature,
    _eval_cache_valid,
    _load_eval_cache,
    _sanitize_name,
    _save_eval_cache,
    load_baseline_cache,
    save_baseline_cache,
)
from .eval_runner import (
    DETECTION_MODEL_SPECS,
    _build_evaluation_tasks,
    _clone_cfg_for_evaluation_seed,
    _compute_random_baseline,
    _constant_curve,
    _detector_cost_profile,
    _load_prepared,
    _load_raw_test_boxes,
    _load_train_data,
    _measure_detector_latency,
    _ratio_errors,
    _resolve_evaluation_workers,
    _resolve_trace_source_ratio,
    _write_optional_json,
    compute_offloading,
    compute_offloading_combined,
    compute_offloading_for_offloader,
    evaluate_one,
)
from .eval_metrics import compute_ranking_metrics
from .evaluation_profiles import (DefaultEvaluationProfile, EvaluationRun,
                                  get_evaluation_profile)
from .paper_eval import (
    aggregate_seeded_rows,
    build_trace_rows,
    cache_key_for_run,
    collect_proxy_metric_rows,
    collect_qualitative_rows,
    compute_component_diagnostic_rows,
    compute_prediction_diagnostics,
    compute_resource_tradeoff_row,
    compute_selection_diagnostics,
    compute_slice_opportunity_rows,
    compute_slice_rows,
    compute_statistical_summary,
    resolve_evaluation_seeds,
    summarize_offloading_curve,
)

# Names of virtual estimators (zero-cost baselines)
VIRTUAL_NAMES = {"weak_model", "strong_model"}


def _compute_offloading_combined_compat(*args, **kwargs) -> tuple[dict, dict, Optional[dict], Optional[OffloadDecision]]:
    """Normalize legacy multi-strategy helper payloads into one-offloader output."""
    import inspect
    try:
        result = compute_offloading_combined(*args, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        signature = inspect.signature(compute_offloading_combined)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            raise
        filtered_kwargs = {
            key: value for key, value in kwargs.items()
            if key in signature.parameters
        }
        result = compute_offloading_combined(*args, **filtered_kwargs)
    if not isinstance(result, tuple):
        raise TypeError("compute_offloading_combined must return a tuple")
    if len(result) == 4:
        return result

    offloader = args[3] if len(args) >= 4 else kwargs.get("offloader")
    policy_id = getattr(offloader, "policy_id", "")
    legacy_map = {
        16: {
            "native_threshold": (0, 1, None, None),
            "online_ecdf_calibrated": (2, 3, None, None),
            "sequential_csr": (4, 5, None, None),
            "sequential_csr_utility": (8, 9, None, None),
            "fixed_classifier": (None, None, 10, 11),
            "online_sqt": (12, 13, None, None),
            "online_lvq": (14, 15, None, None),
        },
        9: {
            "native_threshold": (1, 2, None, None),
            "online_ecdf_calibrated": (3, 4, None, None),
            "sequential_csr": (5, 6, None, None),
            "fixed_classifier": (None, None, 7, 8),
        },
    }
    if len(result) not in legacy_map or policy_id not in legacy_map[len(result)]:
        raise ValueError(
            "Unsupported compute_offloading_combined payload for policy "
            f"'{policy_id}': size={len(result)}"
        )

    curve_idx, decisions_idx, point_idx, point_dec_idx = legacy_map[len(result)][policy_id]
    curve = result[curve_idx] if isinstance(curve_idx, int) else {}
    decisions = result[decisions_idx] if isinstance(decisions_idx, int) else {}
    point = result[point_idx] if isinstance(point_idx, int) else None
    point_decision = result[point_dec_idx] if isinstance(point_dec_idx, int) else None
    return curve or {}, decisions or {}, point, point_decision


def _execute_evaluation_task(task: dict, cfg: PipelineConfig,
                             data: dict, train_data: dict | None,
                             *, raw_boxes: list, stream_order: np.ndarray | None,
                             has_raw: bool, n_test: int) -> dict:
    pcfg = task["pcfg"]
    seed = int(task["seed"])
    profile = task["profile"]
    est_cls = task["est_cls"]
    eval_run = task["eval_run"]
    local_cfg = _clone_cfg_for_evaluation_seed(cfg, seed)
    scenario_name = eval_run.name_suffix
    scenario_weights = eval_run.predict_kwargs.get("scenario_weights")

    result = evaluate_one(
        pcfg,
        local_cfg,
        data,
        train_data=train_data,
        scenario_name=scenario_name,
        scenario_weights=scenario_weights,
    )
    result["seed"] = seed
    payload = {"title": task["title"], "result": result}

    if result["status"] != "PASS":
        return payload

    result_name = result["estimator"]
    run_key = cache_key_for_run(result_name, seed, len(getattr(cfg, "evaluation_seeds", []) or [cfg.seed]))
    result["_run_key"] = run_key
    run_meta = {
        "estimator": result_name,
        "base_model": pcfg.registry_key,
        "stage": pcfg.stage,
        "offloader_id": pcfg.offloader_name,
        "policy_id": pcfg.policy_id,
        "seed": seed,
        "scenario": eval_run.row_metadata.get("scenario", result.get("scenario", "")),
        "scenario_type": eval_run.row_metadata.get("scenario_type", result.get("scenario_type", "")),
    }
    result["offloader_id"] = pcfg.offloader_name
    result["policy_id"] = pcfg.policy_id

    preds = result.pop("_predictions")
    predict_outputs = result.pop("_predict_outputs", None)
    train_preds = result.pop("_train_predictions", None)
    train_predict_outputs = result.pop("_train_predict_outputs", None)

    y_test = profile.resolve_reporting_target(
        pcfg, data, "test", est_cls, eval_run
    )
    y_test = np.asarray(y_test, dtype=float)
    y_train = None
    if train_data is not None and train_preds is not None:
        try:
            y_train = profile.resolve_reporting_target(
                pcfg, train_data, "train", est_cls, eval_run
            )
        except Exception:
            y_train = None

    ranking = compute_ranking_metrics(
        y_test,
        preds,
        top_k=[max(1, int(round(n_test * frac))) for frac in (0.01, 0.05, 0.10, 0.20)],
    )
    for key, value in ranking.items():
        if key not in result:
            result[key] = value

    prediction_diag = compute_prediction_diagnostics(
        y_test,
        preds,
        bins=cfg.calibration_bins or 10,
        proxy_metric=pcfg.proxy_metric,
        y_train_true=y_train,
        y_train_pred=train_preds,
    )
    slice_rows = compute_slice_rows(
        result_name, seed, y_test, preds, data,
        scenario=run_meta["scenario"],
        scenario_type=run_meta["scenario_type"],
    )
    qualitative_rows = collect_qualitative_rows(
        result_name, pcfg.registry_key, seed, y_test, preds, data,
        scenario=run_meta["scenario"],
        scenario_type=run_meta["scenario_type"],
    )
    try:
        target_bundle = resolve_target(pcfg, data, "test")
    except Exception:
        target_bundle = None
    component_rows = compute_component_diagnostic_rows(
        result_name, pcfg.registry_key, target_bundle,
        predict_outputs, seed,
        scenario=run_meta["scenario"],
        scenario_type=run_meta["scenario_type"],
    )

    curve_res: dict[float, dict[str, float]] = {}
    decisions: dict[float, OffloadDecision] = {}
    fixed_res = None
    fixed_decision = None
    peak_map = None
    peak_map_coco = None
    peak_map_coco50 = None

    if has_raw and pcfg.registry_key not in ("weak_model", "strong_model"):
        runtime_offloader = ConfiguredOffloader(
            pcfg.offloader_name,
            pcfg.policy_id,
            params=pcfg.offloader_params,
        )
        context = OffloadContext(
            predictions=np.asarray(preds, dtype=float),
            proxy_metric=pcfg.proxy_metric,
            train_predictions=(
                np.asarray(train_preds, dtype=float)
                if train_preds is not None else None
            ),
            predict_outputs=(
                predict_outputs if isinstance(predict_outputs, dict) else None
            ),
            train_predict_outputs=(
                train_predict_outputs if isinstance(train_predict_outputs, dict) else None
            ),
            stream_order=stream_order,
        )
        (curve_res, decisions,
         fixed_res, fixed_decision) = _compute_offloading_combined_compat(
            preds,
            raw_boxes,
            cfg.offload_ratios,
            runtime_offloader,
            train_predictions=train_preds,
            predict_outputs=context.predict_outputs,
            train_predict_outputs=context.train_predict_outputs,
            stream_order=context.stream_order,
            proxy_metric=pcfg.proxy_metric,
        )
        if fixed_res is not None:
            peak_map = fixed_res["ap50"]
            peak_map_coco = fixed_res["ap_coco"]
            peak_map_coco50 = fixed_res["ap50_allpoint"]
        elif curve_res:
            peak_map = max(res["ap50"] for res in curve_res.values())
            peak_map_coco = max(res["ap_coco"] for res in curve_res.values())
            peak_map_coco50 = max(res["ap50_allpoint"] for res in curve_res.values())

        ratio_errors = _ratio_errors(decisions)
        if ratio_errors:
            result["mean_ratio_error"] = float(np.mean(ratio_errors))
            result["max_ratio_error"] = float(np.max(ratio_errors))
        result["ratio_compatible"] = pcfg.policy_id == "native_threshold"

    payload.update({
        "run_key": run_key,
        "run_meta": run_meta,
        "predictions": preds,
        "predict_outputs": predict_outputs,
        "train_predictions": train_preds,
        "train_predict_outputs": train_predict_outputs,
        "reporting_target": y_test,
        "proxy_metric": pcfg.proxy_metric or "gain_11pt",
        "ranking": ranking,
        "prediction_diag": prediction_diag,
        "slice_rows": slice_rows,
        "qualitative_rows": qualitative_rows,
        "component_rows": component_rows,
        "curve_res": curve_res,
        "decisions": decisions,
        "fixed_res": fixed_res,
        "fixed_decision": fixed_decision,
        "peak_map": peak_map,
        "peak_map_coco": peak_map_coco,
        "peak_map_coco50": peak_map_coco50,
    })
    return payload


# ---- public API -------------------------------------------------------

def run(cfg: PipelineConfig) -> None:
    """Execute the evaluation phase."""
    _run_evaluate(cfg)


def _write_scenario_eval_data(
    cfg: PipelineConfig,
    metrics_df: pd.DataFrame,
    offload_summary_df: pd.DataFrame,
    offload_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    component_df: pd.DataFrame,
    scenario_profiles_df: pd.DataFrame,
) -> None:
    """Write scenario-specific evaluation data to scenario_eval_dir."""
    out = cfg.output.scenario_eval_dir

    def _scenario_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "scenario" not in df.columns:
            return pd.DataFrame()
        return df[df["scenario"].fillna("").astype(str) != ""].copy()

    def _save(df: pd.DataFrame, path: Path) -> None:
        if not df.empty:
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        elif path.exists():
            path.unlink()

    _save(_scenario_rows(metrics_df),         out / "scenario_metrics.csv")
    _save(_scenario_rows(offload_summary_df), out / "scenario_offloading_summary.csv")
    _save(_scenario_rows(offload_df),         out / "scenario_offloading.csv")
    _save(_scenario_rows(ranking_df),         out / "scenario_ranking.csv")
    _save(_scenario_rows(component_df),       out / "scenario_component_diagnostics.csv")
    _save(scenario_profiles_df,               out / "scenario_profiles.csv")


def _run_evaluate(cfg: PipelineConfig) -> None:
    with log.phase_timer(4):
        _run_evaluate_inner(cfg)


def _run_evaluate_inner(cfg: PipelineConfig) -> None:
    if not (cfg.output.prepared_dir / "data.npz").exists():
        raise FileNotFoundError("Prepared data not found. Run 'prepare' first.")

    def _write_optional_csv(path: Path, rows: list[dict], group_cols: list[str] | None = None,
                            raw_path: Path | None = None) -> pd.DataFrame:
        if not rows:
            if path.exists():
                path.unlink()
            if raw_path is not None and raw_path.exists():
                raw_path.unlink()
            return pd.DataFrame()
        raw_df = pd.DataFrame(rows)
        if raw_path is not None:
            raw_df.to_csv(raw_path, index=False)
        out_df = aggregate_seeded_rows(raw_df, group_cols or []) if group_cols else raw_df
        out_df.to_csv(path, index=False)
        return out_df

    data = _load_prepared(cfg)
    n_test = len(data["X_test"])
    log.kv("Test frames", log.fmt_count(n_test))
    seeds = resolve_evaluation_seeds(cfg)
    cfg._use_seeded_checkpoints = len(seeds) > 1
    if cfg._use_seeded_checkpoints:
        log.kv("Evaluation seeds", ", ".join(str(seed) for seed in seeds))

    all_approaches = list(cfg.enabled_approaches())
    all_approaches = [p for p in all_approaches if p.registry_key != "oracle"]

    virtual = [
        ApproachConfig("weak_model",
                       estimator=EstimatorConfig("weak_model", stage="pre",
                                                  proxy_metric="gain_11pt")),
        ApproachConfig("strong_model",
                       estimator=EstimatorConfig("strong_model", stage="pre",
                                                  proxy_metric="gain_11pt")),
    ]
    virtual += [p for p in all_approaches if p.registry_key in VIRTUAL_NAMES]
    real = [p for p in all_approaches if p.registry_key not in VIRTUAL_NAMES]

    # --- Eval cache: check which approaches can be loaded from cache ---
    _cached_payloads: list[dict] = []
    _new_payload_buf: dict[str, list[dict]] = {}  # approach_name → payloads
    _cache_sigs: dict[str, str] = {}  # approach_name → signature
    uncached_virtual: list[ApproachConfig] = []
    uncached_real: list[ApproachConfig] = []
    eval_cache_count = 0

    for pcfg in virtual + real:
        sig = _eval_cache_signature(pcfg, cfg)
        _cache_sigs[pcfg.name] = sig
        adir = _eval_cache_dir_for(cfg, pcfg)
        if not cfg.force_retrain and _eval_cache_valid(adir, sig):
            loaded = _load_eval_cache(adir)
            if loaded is not None:
                for p in loaded:
                    p["_from_cache"] = True
                _cached_payloads.extend(loaded)
                eval_cache_count += 1
                continue
        if pcfg.registry_key in VIRTUAL_NAMES:
            uncached_virtual.append(pcfg)
        else:
            uncached_real.append(pcfg)

    if eval_cache_count:
        log.kv("Eval-cached approaches", str(eval_cache_count))

    approaches = uncached_virtual + uncached_real
    needs_train = bool(uncached_real)
    train_data = _load_train_data(cfg) if needs_train else None

    edge_test = data["edge_test"]
    cloud_test = data["cloud_test"]
    has_coco = "edge_test_coco" in data
    edge_test_coco = data.get("edge_test_coco", edge_test)
    cloud_test_coco = data.get("cloud_test_coco", cloud_test)
    has_coco50 = "edge_test_coco50" in data
    edge_test_coco50 = data.get("edge_test_coco50", edge_test)
    cloud_test_coco50 = data.get("cloud_test_coco50", cloud_test)

    edge_mean = float(edge_test.mean())
    cloud_mean = float(cloud_test.mean())
    edge_mean_coco = float(edge_test_coco.mean())
    cloud_mean_coco = float(cloud_test_coco.mean())
    edge_mean_coco50 = float(edge_test_coco50.mean())
    cloud_mean_coco50 = float(cloud_test_coco50.mean())
    log.section("Baseline Detection AP")
    log.table(
        ["Model", "mAP@0.5(11pt)", "AP@0.5(COCO)", "mAP@COCO"],
        [
            ["Edge", f"{edge_mean:.4f}", f"{edge_mean_coco50:.4f}", f"{edge_mean_coco:.4f}"],
            ["Cloud", f"{cloud_mean:.4f}", f"{cloud_mean_coco50:.4f}", f"{cloud_mean_coco:.4f}"],
        ],
        col_widths=[8, 15, 14, 10],
    )

    results_list: list[dict] = []
    ranking_rows: list[dict] = []
    prediction_diagnostic_rows: list[dict] = []
    component_rows: list[dict] = []
    slice_rows: list[dict] = []
    slice_opportunity_rows: list[dict] = compute_slice_opportunity_rows(data)
    qualitative_rows: list[dict] = []
    selection_rows: list[dict] = []
    resource_rows: list[dict] = []
    trace_rows: list[dict] = []
    offload_rows: list[dict] = []
    threshold_rows: list[dict] = []
    offload_summary_rows: list[dict] = []
    proxy_metric_rows = collect_proxy_metric_rows(data)
    scenario_profile_rows: list[dict] = []

    all_predictions: dict[str, np.ndarray] = {}
    all_train_predictions: dict[str, np.ndarray] = {}
    reporting_targets_by_key: Dict[str, np.ndarray] = {}
    estimator_proxy_metrics: Dict[str, str] = {}
    run_meta_by_key: Dict[str, dict] = {}
    peak_maps: Dict[str, float] = {}
    peak_maps_coco: Dict[str, float] = {}
    peak_maps_coco50: Dict[str, float] = {}
    offload_curves_by_run: dict[str, dict[float, dict[str, float]]] = {}
    offload_decisions_by_run: dict[str, dict[float, OffloadDecision]] = {}
    offload_points_by_run: dict[str, dict[str, float]] = {}
    offload_point_decisions_by_run: dict[str, OffloadDecision] = {}
    forced_offload_curves_by_run: dict[str, dict[float, dict[str, float]]] = {}

    raw_boxes = _load_raw_test_boxes(
        cfg, data.get("video_name_test", []), data.get("frame_id_test", []),
    )
    stream_order = None
    if "video_name_test" in data and "frame_id_test" in data:
        video_names = np.asarray(data["video_name_test"]).astype(str)
        frame_ids = np.asarray(data["frame_id_test"]).astype(int)
        stream_order = np.lexsort((frame_ids, video_names))
    has_raw = bool(raw_boxes) and any(b is not None for b in raw_boxes)
    if not has_raw:
        log.warn("No raw boxes loaded, offloading dataset-wide AP will be skipped.")

    tasks, scenario_profile_rows = _build_evaluation_tasks(cfg, approaches, seeds)
    eval_workers = _resolve_evaluation_workers(cfg, len(tasks))
    log.section(f"Evaluating {len(approaches) + eval_cache_count} Approaches"
                f" ({eval_cache_count} cached)" if eval_cache_count else
                f"Evaluating {len(approaches)} Approaches")
    log.kv("Evaluation runs", log.fmt_count(len(tasks) + len(_cached_payloads)))
    if eval_workers > 1:
        log.kv("Evaluation workers", str(eval_workers))

    def _iter_payloads():
        # Yield cached payloads first (no computation needed)
        yield from _cached_payloads

        if eval_workers <= 1:
            for task in tasks:
                yield _execute_evaluation_task(
                    task, cfg, data, train_data,
                    raw_boxes=raw_boxes,
                    stream_order=stream_order,
                    has_raw=has_raw,
                    n_test=n_test,
                )
            return

        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            for payload in executor.map(
                lambda task: _execute_evaluation_task(
                    task, cfg, data, train_data,
                    raw_boxes=raw_boxes,
                    stream_order=stream_order,
                    has_raw=has_raw,
                    n_test=n_test,
                ),
                tasks,
            ):
                yield payload

    for payload in _iter_payloads():
        result = payload["result"]
        is_cached = payload.get("_from_cache", False)
        if not is_cached:
            log.subsection(payload["title"])
        if result["status"] != "PASS":
            if not is_cached:
                log.fail(f"{result.get('error')}", indent=6)
            results_list.append(result)
            continue

        run_key = payload["run_key"]
        run_meta = payload["run_meta"]
        run_meta_by_key[run_key] = run_meta
        all_predictions[run_key] = payload["predictions"]
        if payload["train_predictions"] is not None:
            all_train_predictions[run_key] = np.asarray(payload["train_predictions"])
        estimator_proxy_metrics[run_key] = payload["proxy_metric"]
        reporting_targets_by_key[run_key] = payload["reporting_target"]

        ranking_rows.append({**run_meta, **payload["ranking"]})
        if payload["prediction_diag"]:
            prediction_diagnostic_rows.append({**run_meta, **payload["prediction_diag"]})
        slice_rows.extend(payload["slice_rows"])
        qualitative_rows.extend(payload["qualitative_rows"])
        component_rows.extend(payload["component_rows"])

        r2_val = result.get('r2', float('nan'))
        mae_val = result.get('mae', float('nan'))
        infer_val = result.get('inference_time_ms', 0)
        log.info(f"R²={log.fmt_metric(r2_val)}  "
                 f"MAE={log.fmt_metric(mae_val)}  "
                 f"Infer={infer_val:.3f}ms", indent=6)

        curve_res = payload["curve_res"]
        decisions = payload["decisions"]
        fixed_res = payload["fixed_res"]
        fixed_decision = payload["fixed_decision"]
        if has_raw and result.get("base_model") not in ("weak_model", "strong_model"):
            if fixed_res is not None:
                log.info(
                    f"Offloading AP (fixed point, offloader={run_meta['offloader_id']}, policy={run_meta['policy_id']})",
                    indent=6,
                )
            elif curve_res:
                log.info(
                    f"Offloading AP ({len(cfg.offload_ratios)} ratios, offloader={run_meta['offloader_id']}, policy={run_meta['policy_id']})",
                    indent=6,
                )
            if decisions:
                offload_decisions_by_run[run_key] = decisions
            if curve_res:
                offload_curves_by_run[run_key] = curve_res
            if fixed_res is not None and fixed_decision is not None:
                offload_points_by_run[run_key] = fixed_res
                offload_point_decisions_by_run[run_key] = fixed_decision

            if payload["peak_map"] is not None:
                peak_maps[run_key] = float(payload["peak_map"])
            if has_coco and payload["peak_map_coco"] is not None:
                peak_maps_coco[run_key] = float(payload["peak_map_coco"])
            if has_coco50 and payload["peak_map_coco50"] is not None:
                peak_maps_coco50[run_key] = float(payload["peak_map_coco50"])
            if run_key in peak_maps:
                log.success(
                    f"Peak AP@0.5 [{run_meta['offloader_id']}] = {peak_maps[run_key]:.4f}",
                    indent=6,
                )

        result["peak_map"] = peak_maps.get(run_key, edge_mean)
        if has_coco:
            result["peak_map_coco"] = peak_maps_coco.get(run_key, result["peak_map"])
        if has_coco50:
            result["peak_map_coco50"] = peak_maps_coco50.get(run_key, result["peak_map"])
        results_list.append(result)

        # Buffer new (non-cached) payloads for saving to eval cache
        if not is_cached and result["status"] == "PASS":
            est_name = run_meta.get("estimator", result.get("estimator", ""))
            # Find the approach config that owns this estimator
            for _pcfg in uncached_virtual + uncached_real:
                if _pcfg.name == est_name or est_name.startswith(_pcfg.name):
                    _new_payload_buf.setdefault(_pcfg.name, []).append(payload)
                    break

    # --- Save eval cache for newly evaluated approaches ---
    for pcfg in uncached_virtual + uncached_real:
        payloads_to_cache = _new_payload_buf.get(pcfg.name, [])
        if payloads_to_cache:
            sig = _cache_sigs.get(pcfg.name, "")
            adir = _eval_cache_dir_for(cfg, pcfg)
            try:
                _save_eval_cache(adir, payloads_to_cache, sig)
            except Exception as e:
                log.warn(f"Failed to cache eval results for {pcfg.name}: {e}")

    # Clean up stale cache dirs for approaches no longer in config
    cache_root = cfg.output.metrics_dir / _EVAL_CACHE_DIR
    if cache_root.exists():
        active_names = {_sanitize_name(p.name) for p in virtual + real}
        for child in cache_root.iterdir():
            if child.is_dir() and child.name not in active_names:
                shutil.rmtree(child, ignore_errors=True)

    if has_raw:
        for run_key, preds in all_predictions.items():
            meta = run_meta_by_key.get(run_key, {})
            if str(meta.get("estimator", "")) in {"weak_model", "strong_model"}:
                continue
            forced_offload_curves_by_run[run_key] = compute_offloading(
                np.asarray(preds, dtype=float),
                raw_boxes,
                cfg.offload_ratios,
            )

    # Compute random + oracle baselines (or load from cache)
    random_offload: Dict[float, Dict[str, float]] = {}
    oracle_offload: Dict[float, Dict[str, float]] = {}
    _baseline_hit = load_baseline_cache(cfg) if has_raw else None
    if _baseline_hit is not None:
        random_offload, oracle_offload = _baseline_hit
        log.cached("Random + oracle baselines")
    elif has_raw:
        log.subsection("Random baseline")
        random_offload = _compute_random_baseline(raw_boxes, cfg.offload_ratios)

    if random_offload and 0.0 in random_offload and 1.0 in random_offload:
        edge_dataset_ap = float(random_offload[0.0]["ap50"])
        cloud_dataset_ap = float(random_offload[1.0]["ap50"])
        edge_dataset_ap_coco = float(random_offload[0.0].get("ap_coco", edge_mean_coco))
        cloud_dataset_ap_coco = float(random_offload[1.0].get("ap_coco", cloud_mean_coco))
        edge_dataset_ap_coco50 = float(random_offload[0.0].get("ap50_allpoint", edge_mean_coco50))
        cloud_dataset_ap_coco50 = float(random_offload[1.0].get("ap50_allpoint", cloud_mean_coco50))
    else:
        edge_dataset_ap, cloud_dataset_ap = edge_mean, cloud_mean
        edge_dataset_ap_coco, cloud_dataset_ap_coco = edge_mean_coco, cloud_mean_coco
        edge_dataset_ap_coco50, cloud_dataset_ap_coco50 = edge_mean_coco50, cloud_mean_coco50

    weak_curve = _constant_curve(cfg.offload_ratios, edge_dataset_ap, edge_dataset_ap_coco, edge_dataset_ap_coco50)
    strong_curve = _constant_curve(cfg.offload_ratios, cloud_dataset_ap, cloud_dataset_ap_coco, cloud_dataset_ap_coco50)

    for r in results_list:
        run_key = r.get("_run_key")
        if r["estimator"] == "weak_model":
            peak_maps.setdefault(run_key, edge_dataset_ap)
            if has_coco:
                peak_maps_coco.setdefault(run_key, edge_dataset_ap_coco)
            if has_coco50:
                peak_maps_coco50.setdefault(run_key, edge_dataset_ap_coco50)
        elif r["estimator"] == "strong_model":
            peak_maps.setdefault(run_key, cloud_dataset_ap)
            if has_coco:
                peak_maps_coco.setdefault(run_key, cloud_dataset_ap_coco)
            if has_coco50:
                peak_maps_coco50.setdefault(run_key, cloud_dataset_ap_coco50)
        r["peak_map"] = peak_maps.get(run_key, edge_dataset_ap)
        if has_coco:
            r["peak_map_coco"] = peak_maps_coco.get(run_key, r["peak_map"])
        if has_coco50:
            r["peak_map_coco50"] = peak_maps_coco50.get(run_key, r["peak_map"])
        if "inference_time_ms" not in r:
            r["inference_time_ms"] = 0.0

    if _baseline_hit is None and has_raw:
        has_ds_oric = "y_test_dataset_oric_11pt" in data
        log.subsection("Oracle")
        if has_ds_oric:
            log.info("Using dataset-wide ORIC (set-wise AP impact)")
        else:
            log.info("Dataset-wide ORIC not found, falling back to per-frame AP gain")
        oracle_gains = {
            "ap50": (data["y_test_dataset_oric_11pt"]
                     if has_ds_oric
                     else data.get("y_test_gain_11pt", data["y_test"])),
            "ap50_allpoint": (data["y_test_dataset_oric_allpoint"]
                              if has_ds_oric
                              else data.get("y_test_gain_allpoint", data["y_test"])),
            "ap_coco": (data["y_test_dataset_oric_coco"]
                        if has_ds_oric
                        else data.get("y_test_gain_coco", data["y_test"])),
        }
        oracle_runs = {}
        for key, gains in oracle_gains.items():
            log.info(f"Oracle offloading ({key})...")
            oracle_runs[key] = compute_offloading(gains, raw_boxes, cfg.offload_ratios)
        for ratio in cfg.offload_ratios:
            oracle_offload[ratio] = {
                "ap50": oracle_runs["ap50"][ratio]["ap50"],
                "ap50_allpoint": oracle_runs["ap50_allpoint"][ratio]["ap50_allpoint"],
                "ap_coco": oracle_runs["ap_coco"][ratio]["ap_coco"],
                "n_offload": oracle_runs["ap50"][ratio]["n_offload"],
            }
        try:
            save_baseline_cache(cfg, random_offload, oracle_offload)
        except Exception as e:
            log.warn(f"Failed to cache baselines: {e}")

    if image_paths := data.get("paths_test", []):
        for r in results_list:
            det_model = None
            if r["estimator"] == "weak_model":
                det_model = cfg.dataset.edge_model
            elif r["estimator"] == "strong_model":
                det_model = cfg.dataset.cloud_model
            if det_model:
                log.info(f"Measuring per-frame latency: {det_model}")
                latency = _measure_detector_latency(
                    det_model, image_paths,
                    n_warmup=cfg.latency_warmup,
                    n_samples=cfg.latency_samples,
                    device=cfg.device,
                )
                r["detector_time_ms"] = latency
                log.kv(det_model, f"{latency:.2f} ms/frame", indent=6)

    result_by_run_key = {
        r.get("_run_key"): r for r in results_list
        if r.get("_run_key") is not None
    }
    detector_costs = _detector_cost_profile(cfg, results_list)
    trace_ratio_refs = [float(r) for r in (cfg.fixed_ratio_points or [])]
    forced_offload_rows: list[dict] = []
    forced_offload_summary_rows: list[dict] = []

    for run_key, curve in offload_curves_by_run.items():
        meta = run_meta_by_key[run_key]
        decisions = offload_decisions_by_run.get(run_key, {})
        policy_id = str(meta.get("policy_id", "native_threshold"))
        y_true = reporting_targets_by_key.get(run_key)
        result_row = result_by_run_key.get(run_key, {})
        metric_type = classify_metric(
            estimator_proxy_metrics.get(run_key, "gain_11pt")
        ).value
        for ratio, res in curve.items():
            row = {
                **meta,
                "ratio": ratio,
                "strategy": policy_id,
                "mAP": res["ap50"],
            }
            if has_coco:
                row["mAP_coco"] = res.get("ap_coco")
            if has_coco50:
                row["mAP_coco50"] = res.get("ap50_allpoint")
            offload_rows.append(row)

            decision = decisions.get(ratio)
            threshold_row = {
                **meta,
                "proxy_metric": estimator_proxy_metrics.get(run_key, "gain_11pt"),
                "metric_type": metric_type,
                "strategy": policy_id,
                "threshold_source": policy_id,
                "target_ratio": ratio,
                "actual_ratio": decision.actual_ratio if decision else 0.0,
                "ratio_error": decision.ratio_error if decision else 0.0,
                "threshold": decision.threshold if decision else float("nan"),
                "lambda_final": decision.lambda_final if decision else float("nan"),
                "lambda_mean": decision.lambda_mean if decision else float("nan"),
                "mAP": res["ap50"],
            }
            if has_coco:
                threshold_row["mAP_coco"] = res.get("ap_coco")
            if has_coco50:
                threshold_row["mAP_coco50"] = res.get("ap50_allpoint")
            threshold_rows.append(threshold_row)

            if y_true is not None and decision is not None:
                selection = compute_selection_diagnostics(
                    meta["estimator"], meta["base_model"], meta["seed"],
                    y_true, decision.mask, ratio, policy_id,
                    scenario=meta.get("scenario", ""),
                    scenario_type=meta.get("scenario_type", ""),
                )
                if selection:
                    selection_rows.append({**meta, **selection})
            resource_rows.append(
                {**meta, **compute_resource_tradeoff_row(
                    estimator_name=meta["estimator"],
                    base_model=meta["base_model"],
                    stage=meta.get("stage", "other"),
                    seed=meta["seed"],
                    strategy=policy_id,
                    ratio=ratio,
                    actual_ratio=threshold_row["actual_ratio"],
                    inference_time_ms=float(result_row.get("inference_time_ms", 0.0) or 0.0),
                    estimator_gflops=float(result_row.get("gflops", 0.0) or 0.0),
                    map_0_5=float(res["ap50"]),
                    map_coco=float(res.get("ap_coco", float("nan"))) if has_coco else float("nan"),
                    **detector_costs,
                    scenario=meta.get("scenario", ""),
                    scenario_type=meta.get("scenario_type", ""),
                )}
            )
        if y_true is not None and decisions and trace_ratio_refs:
            available_trace_ratios = sorted(float(r) for r in decisions.keys())
            for focus_ratio in trace_ratio_refs:
                source_ratio = _resolve_trace_source_ratio(
                    focus_ratio,
                    available_trace_ratios,
                )
                trace_decision = decisions.get(source_ratio)
                if trace_decision is None:
                    continue
                trace_rows.extend(
                    {**meta, **row} for row in build_trace_rows(
                        meta["estimator"],
                        meta["base_model"],
                        meta.get("stage", "other"),
                        meta["seed"],
                        policy_id,
                        focus_ratio,
                        trace_decision,
                        y_true,
                        source_target_ratio=source_ratio,
                        scenario=meta.get("scenario", ""),
                        scenario_type=meta.get("scenario_type", ""),
                    )
                )

        summary = summarize_offloading_curve(
            curve,
            weak_curve=weak_curve,
            oracle_curve=oracle_offload or None,
            fixed_ratio_points=cfg.fixed_ratio_points,
            ratio_errors=_ratio_errors(decisions),
        )
        offload_summary_rows.append({**meta, "strategy": policy_id, **summary})

    for run_key, curve in forced_offload_curves_by_run.items():
        meta = run_meta_by_key[run_key]
        for ratio, res in curve.items():
            row = {
                **meta,
                "ratio": ratio,
                "strategy": "topk",
                "mAP": res["ap50"],
            }
            if has_coco:
                row["mAP_coco"] = res.get("ap_coco")
            if has_coco50:
                row["mAP_coco50"] = res.get("ap50_allpoint")
            forced_offload_rows.append(row)

        forced_summary = summarize_offloading_curve(
            curve,
            weak_curve=weak_curve,
            oracle_curve=oracle_offload or None,
            fixed_ratio_points=cfg.fixed_ratio_points,
        )
        forced_offload_summary_rows.append({**meta, "strategy": "topk", **forced_summary})

    for run_key, res in offload_points_by_run.items():
        meta = run_meta_by_key[run_key]
        dec = offload_point_decisions_by_run.get(run_key)
        result_row = result_by_run_key.get(run_key, {})
        y_true = reporting_targets_by_key.get(run_key)
        policy_id = str(meta.get("policy_id", "fixed_classifier"))
        metric_type = classify_metric(
            estimator_proxy_metrics.get(run_key, "gain_11pt")
        ).value
        threshold_rows.append({
            **meta,
            "proxy_metric": estimator_proxy_metrics.get(run_key, "gain_11pt"),
            "metric_type": metric_type,
            "strategy": policy_id,
            "threshold_source": policy_id,
            "target_ratio": float("nan"),
            "actual_ratio": dec.actual_ratio if dec else 0.0,
            "ratio_error": float("nan"),
            "threshold": dec.threshold if dec else 0.5,
            "lambda_final": float("nan"),
            "lambda_mean": float("nan"),
            "mAP": res["ap50"],
            **({"mAP_coco": res.get("ap_coco")} if has_coco else {}),
            **({"mAP_coco50": res.get("ap50_allpoint")} if has_coco50 else {}),
        })
        offload_summary_rows.append({
            **meta,
            "strategy": policy_id,
            "peak_map": res["ap50"],
            "peak_map_coco": res.get("ap_coco", float("nan")),
            "peak_map_coco50": res.get("ap50_allpoint", float("nan")),
            "auc_0_5": float("nan"),
            "auc_coco": float("nan"),
            "auc_coco50": float("nan"),
            "mean_ratio_error": float(dec.ratio_error) if dec else float("nan"),
        })
        resource_rows.append(
            {**meta, **compute_resource_tradeoff_row(
                estimator_name=meta["estimator"],
                base_model=meta["base_model"],
                stage=meta.get("stage", "other"),
                seed=meta["seed"],
                strategy=policy_id,
                ratio=float(dec.actual_ratio) if dec else float("nan"),
                actual_ratio=float(dec.actual_ratio) if dec else 0.0,
                inference_time_ms=float(result_row.get("inference_time_ms", 0.0) or 0.0),
                estimator_gflops=float(result_row.get("gflops", 0.0) or 0.0),
                map_0_5=float(res["ap50"]),
                map_coco=float(res.get("ap_coco", float("nan"))) if has_coco else float("nan"),
                **detector_costs,
                scenario=meta.get("scenario", ""),
                scenario_type=meta.get("scenario_type", ""),
            )}
        )
        if y_true is not None and dec is not None:
            selection = compute_selection_diagnostics(
                meta["estimator"], meta["base_model"], meta["seed"],
                y_true, dec.mask, float(dec.actual_ratio), policy_id,
                scenario=meta.get("scenario", ""),
                scenario_type=meta.get("scenario_type", ""),
            )
            if selection:
                selection_rows.append({**meta, **selection})
            trace_rows.extend(
                {**meta, **row} for row in build_trace_rows(
                    meta["estimator"],
                    meta["base_model"],
                    meta.get("stage", "other"),
                    meta["seed"],
                    policy_id,
                    float(dec.actual_ratio),
                    dec,
                    y_true,
                    source_target_ratio=float(dec.actual_ratio),
                    scenario=meta.get("scenario", ""),
                    scenario_type=meta.get("scenario_type", ""),
                )
            )

    if random_offload:
        for ratio, res in random_offload.items():
            row = {"estimator": "random", "base_model": "random", "seed": np.nan,
                   "stage": "other", "offloader_id": "random", "policy_id": "random",
                   "scenario": "", "scenario_type": "", "ratio": ratio,
                   "strategy": "random", "mAP": res["ap50"]}
            if has_coco:
                row["mAP_coco"] = res.get("ap_coco")
            if has_coco50:
                row["mAP_coco50"] = res.get("ap50_allpoint")
            offload_rows.append(row)
            forced_offload_rows.append(dict(row))
        offload_summary_rows.append({
            "estimator": "random", "base_model": "random", "stage": "other", "seed": np.nan,
            "offloader_id": "random", "policy_id": "random",
            "scenario": "", "scenario_type": "", "strategy": "random",
            **summarize_offloading_curve(random_offload, weak_curve=weak_curve,
                                         oracle_curve=oracle_offload or None,
                                         fixed_ratio_points=cfg.fixed_ratio_points),
        })
        forced_offload_summary_rows.append({
            "estimator": "random", "base_model": "random", "stage": "other", "seed": np.nan,
            "offloader_id": "random", "policy_id": "random",
            "scenario": "", "scenario_type": "", "strategy": "random",
            **summarize_offloading_curve(random_offload, weak_curve=weak_curve,
                                         oracle_curve=oracle_offload or None,
                                         fixed_ratio_points=cfg.fixed_ratio_points),
        })

    if oracle_offload:
        for ratio, res in oracle_offload.items():
            row = {"estimator": "oracle", "base_model": "oracle", "seed": np.nan,
                   "stage": "other", "offloader_id": "oracle", "policy_id": "oracle",
                   "scenario": "", "scenario_type": "", "ratio": ratio,
                   "strategy": "oracle", "mAP": res["ap50"]}
            if has_coco:
                row["mAP_coco"] = res.get("ap_coco")
            if has_coco50:
                row["mAP_coco50"] = res.get("ap50_allpoint")
            offload_rows.append(row)
            forced_offload_rows.append(dict(row))
        offload_summary_rows.append({
            "estimator": "oracle", "base_model": "oracle", "stage": "other", "seed": np.nan,
            "offloader_id": "oracle", "policy_id": "oracle",
            "scenario": "", "scenario_type": "", "strategy": "oracle",
            **summarize_offloading_curve(oracle_offload, weak_curve=weak_curve,
                                         oracle_curve=oracle_offload,
                                         fixed_ratio_points=cfg.fixed_ratio_points),
        })
        forced_offload_summary_rows.append({
            "estimator": "oracle", "base_model": "oracle", "stage": "other", "seed": np.nan,
            "offloader_id": "oracle", "policy_id": "oracle",
            "scenario": "", "scenario_type": "", "strategy": "oracle",
            **summarize_offloading_curve(oracle_offload, weak_curve=weak_curve,
                                         oracle_curve=oracle_offload,
                                         fixed_ratio_points=cfg.fixed_ratio_points),
        })

    for r in results_list:
        meta = {
            "estimator": r["estimator"],
            "base_model": r.get("base_model", r["estimator"]),
            "stage": r.get("stage", "other"),
            "offloader_id": r.get("offloader_id", "constant"),
            "policy_id": r.get("policy_id", "constant"),
            "seed": r.get("seed", np.nan),
            "scenario": r.get("scenario", ""),
            "scenario_type": r.get("scenario_type", ""),
        }
        if r["estimator"] == "weak_model":
            const_meta = {**meta, "offloader_id": "constant", "policy_id": "constant"}
            offload_summary_rows.append({
                **const_meta,
                "strategy": "constant",
                **summarize_offloading_curve(weak_curve, weak_curve=weak_curve,
                                             oracle_curve=oracle_offload or None,
                                             fixed_ratio_points=cfg.fixed_ratio_points),
            })
            resource_rows.append(
                {**const_meta, **compute_resource_tradeoff_row(
                    estimator_name=const_meta["estimator"],
                    base_model=const_meta["base_model"],
                    stage=const_meta.get("stage", "other"),
                    seed=const_meta.get("seed", np.nan),
                    strategy="constant",
                    ratio=0.0,
                    actual_ratio=0.0,
                    inference_time_ms=0.0,
                    estimator_gflops=0.0,
                    map_0_5=float(edge_dataset_ap),
                    map_coco=float(edge_dataset_ap_coco) if has_coco else float("nan"),
                    **detector_costs,
                    scenario=const_meta.get("scenario", ""),
                    scenario_type=const_meta.get("scenario_type", ""),
                )}
            )
        elif r["estimator"] == "strong_model":
            const_meta = {**meta, "offloader_id": "constant", "policy_id": "constant"}
            offload_summary_rows.append({
                **const_meta,
                "strategy": "constant",
                **summarize_offloading_curve(strong_curve, weak_curve=weak_curve,
                                             oracle_curve=oracle_offload or None,
                                             fixed_ratio_points=cfg.fixed_ratio_points),
            })
            resource_rows.append(
                {**const_meta, **compute_resource_tradeoff_row(
                    estimator_name=const_meta["estimator"],
                    base_model=const_meta["base_model"],
                    stage=const_meta.get("stage", "other"),
                    seed=const_meta.get("seed", np.nan),
                    strategy="constant",
                    ratio=1.0,
                    actual_ratio=1.0,
                    inference_time_ms=0.0,
                    estimator_gflops=0.0,
                    map_0_5=float(cloud_dataset_ap),
                    map_coco=float(cloud_dataset_ap_coco) if has_coco else float("nan"),
                    **detector_costs,
                    scenario=const_meta.get("scenario", ""),
                    scenario_type=const_meta.get("scenario_type", ""),
                )}
            )

    out = cfg.output.metrics_dir
    out.mkdir(parents=True, exist_ok=True)

    metrics_runs_df = pd.DataFrame(results_list)
    for col in ("_predictions", "_run_key"):
        if col in metrics_runs_df.columns:
            metrics_runs_df.drop(columns=[col], inplace=True)
    metrics_runs_path = out / "estimator_metrics_runs.csv"
    metrics_runs_df.to_csv(metrics_runs_path, index=False)
    metrics_df = aggregate_seeded_rows(
        metrics_runs_df,
        ["estimator", "base_model", "stage", "scenario", "scenario_type"],
    )
    metrics_path = out / "estimator_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    if "base_model" in metrics_df.columns:
        summary_rows = []
        for bm, grp in metrics_df.groupby("base_model"):
            passed = grp[grp["status"] != "FAIL"] if "status" in grp.columns else grp
            row = {"base_model": bm, "stage": grp["stage"].iloc[0] if "stage" in grp.columns else "other"}
            row["approaches"] = ", ".join(grp["estimator"].astype(str).tolist())
            if not passed.empty:
                for col in ("r2", "spearman_rho", "ndcg", "mae", "rmse"):
                    if col in passed.columns and passed[col].notna().any():
                        row[f"best_{col}"] = float(
                            passed[col].max() if col in ("r2", "spearman_rho", "ndcg") else passed[col].min()
                        )
                if "inference_time_ms" in passed.columns:
                    row["inference_time_ms"] = float(passed["inference_time_ms"].iloc[0])
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out / "estimator_summary.csv"
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df = pd.DataFrame()
        summary_path = out / "estimator_summary.csv"

    scenario_profiles_df = pd.DataFrame(scenario_profile_rows)
    scenario_profiles_csv = out / "scenario_profiles.csv"
    scenario_profiles_json = out / "scenario_profiles.json"
    if not scenario_profiles_df.empty:
        scenario_profiles_df.to_csv(scenario_profiles_csv, index=False)
    elif scenario_profiles_csv.exists():
        scenario_profiles_csv.unlink()
    _write_optional_json(scenario_profiles_json, scenario_profile_rows)

    offload_df = _write_optional_csv(
        out / "offloading_results.csv",
        offload_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "ratio", "scenario", "scenario_type"],
        raw_path=out / "offloading_results_runs.csv",
    )
    forced_offload_df = _write_optional_csv(
        out / "offloading_forced_results.csv",
        forced_offload_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "ratio", "scenario", "scenario_type"],
        raw_path=out / "offloading_forced_results_runs.csv",
    )
    threshold_df = _write_optional_csv(
        out / "offloading_threshold_results.csv",
        threshold_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "proxy_metric", "metric_type",
         "threshold_source", "target_ratio", "scenario", "scenario_type"],
        raw_path=out / "offloading_threshold_results_runs.csv",
    )
    ranking_df = _write_optional_csv(
        out / "ranking_metrics.csv",
        ranking_rows,
        ["estimator", "base_model", "stage", "scenario", "scenario_type"],
        raw_path=out / "ranking_metrics_runs.csv",
    )
    prediction_diag_df = _write_optional_csv(
        out / "prediction_diagnostics.csv",
        prediction_diagnostic_rows,
        ["estimator", "base_model", "stage", "scenario", "scenario_type"],
        raw_path=out / "prediction_diagnostics_runs.csv",
    )
    component_df = _write_optional_csv(
        out / "component_diagnostics.csv",
        component_rows,
        ["estimator", "base_model", "diagnostic_family", "component", "scenario", "scenario_type"],
        raw_path=out / "component_diagnostics_runs.csv",
    )
    slice_df = _write_optional_csv(
        out / "slice_metrics.csv",
        slice_rows,
        ["estimator", "slice_name", "slice_value", "scenario", "scenario_type"],
        raw_path=out / "slice_metrics_runs.csv",
    )
    slice_opp_df = _write_optional_csv(
        out / "slice_opportunity.csv",
        slice_opportunity_rows,
        ["slice_name", "slice_value"],
    )
    qualitative_df = _write_optional_csv(
        out / "qualitative_examples.csv",
        qualitative_rows,
        ["estimator", "example_type", "example_rank", "scenario", "scenario_type"],
        raw_path=out / "qualitative_examples_runs.csv",
    )
    selection_df = _write_optional_csv(
        out / "selection_diagnostics.csv",
        selection_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "target_ratio", "scenario", "scenario_type"],
        raw_path=out / "selection_diagnostics_runs.csv",
    )
    resource_df = _write_optional_csv(
        out / "resource_tradeoff.csv",
        resource_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "target_ratio", "scenario", "scenario_type"],
        raw_path=out / "resource_tradeoff_runs.csv",
    )
    trace_df = _write_optional_csv(
        out / "trace_diagnostics.csv",
        trace_rows,
        ["estimator", "base_model", "offloader_id", "policy_id", "target_ratio", "step", "scenario", "scenario_type"],
        raw_path=out / "trace_diagnostics_runs.csv",
    )
    offload_summary_df = _write_optional_csv(
        out / "offloading_summary.csv",
        offload_summary_rows,
        ["estimator", "base_model", "stage", "offloader_id", "policy_id", "scenario", "scenario_type"],
        raw_path=out / "offloading_summary_runs.csv",
    )
    forced_offload_summary_df = _write_optional_csv(
        out / "offloading_forced_summary.csv",
        forced_offload_summary_rows,
        ["estimator", "base_model", "stage", "offloader_id", "policy_id", "scenario", "scenario_type"],
        raw_path=out / "offloading_forced_summary_runs.csv",
    )
    proxy_df = pd.DataFrame(proxy_metric_rows)
    if not proxy_df.empty:
        proxy_df.to_csv(out / "proxy_metric_stats.csv", index=False)
    elif (out / "proxy_metric_stats.csv").exists():
        (out / "proxy_metric_stats.csv").unlink()

    stats_df = compute_statistical_summary(
        metrics_runs_df[metrics_runs_df["status"] == "PASS"] if "status" in metrics_runs_df.columns else metrics_runs_df,
        pd.DataFrame(offload_summary_rows),
        bootstrap_samples=cfg.bootstrap_samples or 1000,
    )
    if not stats_df.empty:
        stats_df.to_csv(out / "statistical_summary.csv", index=False)
    elif (out / "statistical_summary.csv").exists():
        (out / "statistical_summary.csv").unlink()

    pred_arrays = {}
    for run_key, preds in all_predictions.items():
        if run_meta_by_key.get(run_key, {}).get("estimator") in VIRTUAL_NAMES:
            continue
        pred_arrays[f"pred_{run_key}"] = preds
        train_preds = all_train_predictions.get(run_key)
        if train_preds is not None:
            pred_arrays[f"pred_train_{run_key}"] = np.asarray(train_preds)
        pred_arrays[f"proxy_{run_key}"] = estimator_proxy_metrics.get(run_key, "gain_11pt")
    if pred_arrays:
        for key in data:
            if key.startswith("y_test"):
                pred_arrays[key] = data[key]
        for key, value in (train_data or {}).items():
            if key.startswith("y_train"):
                pred_arrays[key] = value
        np.savez_compressed(out / "proxy-metric_distributions.npz", **pred_arrays)

    for model_dir in cfg.derived_dir.iterdir() if cfg.derived_dir.exists() else []:
        ds_json = model_dir / "dataset_summary.json" if model_dir.is_dir() else None
        if ds_json and ds_json.exists():
            shutil.copy2(ds_json, out / "dataset_summary.json")
            break
    else:
        ds_json = cfg.derived_dir / "dataset_summary.json"
        if ds_json.exists():
            shutil.copy2(ds_json, out / "dataset_summary.json")

    sweep_path = out / "sweep_results.csv"
    if sweep_path.exists():
        try:
            sweep_df = pd.read_csv(sweep_path)
            metrics_merge_cols = [col for col in ("estimator", "seed", "inference_time_ms", "spearman_rho", "r2") if col in metrics_runs_df.columns]
            merged = sweep_df.merge(metrics_runs_df[metrics_merge_cols], on=[col for col in ("estimator", "seed") if col in sweep_df.columns], how="left")
            if not offload_summary_df.empty:
                primary_summary = offload_summary_df.copy()
                merge_cols = [col for col in ("estimator", "seed") if col in primary_summary.columns and col in merged.columns]
                if merge_cols:
                    merged = merged.merge(
                        primary_summary[[*merge_cols, *[col for col in ("auc_0_5", "oracle_regret_auc_0_5") if col in primary_summary.columns]]],
                        on=merge_cols, how="left",
                    )
                    if "oracle_regret_auc_0_5" in merged.columns:
                        merged["oracle_regret"] = merged["oracle_regret_auc_0_5"]
                    if "inference_time_ms" in merged.columns:
                        merged["latency"] = merged["inference_time_ms"]
            merged.to_csv(sweep_path, index=False)
        except Exception:
            pass

    _write_scenario_eval_data(
        cfg, metrics_df, offload_summary_df, offload_df,
        ranking_df, component_df, scenario_profiles_df,
    )

    log.section("Saved Outputs")
    for path in (
        metrics_path,
        metrics_runs_path,
        summary_path,
        out / "scenario_profiles.csv",
        out / "scenario_profiles.json",
        out / "offloading_results.csv",
        out / "offloading_forced_results.csv",
        out / "offloading_threshold_results.csv",
        out / "ranking_metrics.csv",
        out / "prediction_diagnostics.csv",
        out / "proxy_metric_stats.csv",
        out / "slice_metrics.csv",
        out / "slice_opportunity.csv",
        out / "component_diagnostics.csv",
        out / "statistical_summary.csv",
        out / "offloading_summary.csv",
        out / "offloading_forced_summary.csv",
        out / "qualitative_examples.csv",
        out / "selection_diagnostics.csv",
        out / "resource_tradeoff.csv",
        out / "trace_diagnostics.csv",
        out / "proxy-metric_distributions.npz",
    ):
        if path.exists():
            log.arrow(str(path))

    scenario_eval_out = cfg.output.scenario_eval_dir
    for path in (
        scenario_eval_out / "scenario_metrics.csv",
        scenario_eval_out / "scenario_offloading_summary.csv",
        scenario_eval_out / "scenario_offloading.csv",
        scenario_eval_out / "scenario_ranking.csv",
        scenario_eval_out / "scenario_component_diagnostics.csv",
        scenario_eval_out / "scenario_profiles.csv",
    ):
        if path.exists():
            log.arrow(str(path))

    trapz = getattr(np, 'trapezoid', None) or np.trapz
    aucs = {}
    for _, row in metrics_df.iterrows():
        name = row["estimator"]
        run_rows = offload_df[(offload_df["estimator"].astype(str) == str(name))]
        aucs[name] = {"ap50": 0.0, "ap_coco": 0.0, "ap_coco50": 0.0}
        if not run_rows.empty and "ratio" in run_rows.columns:
            run_rows = run_rows.sort_values("ratio")
            ratios = run_rows["ratio"].astype(float).to_numpy()
            aucs[name]["ap50"] = float(trapz(run_rows["mAP"].astype(float).to_numpy(), ratios))
            if has_coco and "mAP_coco" in run_rows.columns:
                aucs[name]["ap_coco"] = float(trapz(run_rows["mAP_coco"].astype(float).to_numpy(), ratios))
            if has_coco50 and "mAP_coco50" in run_rows.columns:
                aucs[name]["ap_coco50"] = float(trapz(run_rows["mAP_coco50"].astype(float).to_numpy(), ratios))
        elif name in ("weak_model", "strong_model"):
            aucs[name]["ap50"] = float(row.get("peak_map", 0.0))
            if has_coco:
                aucs[name]["ap_coco"] = float(row.get("peak_map_coco", 0.0))
            if has_coco50:
                aucs[name]["ap_coco50"] = float(row.get("peak_map_coco50", 0.0))

    log.section("Evaluation Summary")
    headers = ["Estimator", "Type", "Infer(ms)", "R²", "Peak mAP@0.5", "Peak COCO", "AUC 0.5", "AUC C50", "AUC COCO"]
    rows = []
    for _, row in metrics_df.iterrows():
        name = row["estimator"]
        ptype = "virtual" if (name in VIRTUAL_NAMES or name == "oracle") else row.get("stage", "-")
        auc_vals = aucs.get(name, {"ap50": 0.0, "ap_coco": 0.0, "ap_coco50": 0.0})
        rows.append([
            name,
            ptype,
            f"{float(row.get('inference_time_ms', 0.0)):.3f}",
            log.fmt_metric(row.get("r2", float("nan"))),
            f"{float(row.get('peak_map', 0.0)):.4f}",
            f"{float(row.get('peak_map_coco', row.get('peak_map', 0.0))):.4f}",
            f"{auc_vals['ap50']:.4f}",
            f"{auc_vals['ap_coco50']:.4f}",
            f"{auc_vals['ap_coco']:.4f}",
        ])
    log.table(
        headers, rows,
        col_widths=[25, 9, 10, 8, 13, 11, 9, 9, 9],
        alignments=["<", "<", ">", ">", ">", ">", ">", ">", ">"],
    )
