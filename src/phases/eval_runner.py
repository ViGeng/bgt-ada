"""Per-approach evaluation: data loading, latency, evaluate_one(), offloading."""

import copy
import inspect
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import ApproachConfig, PipelineConfig

from .. import log
from ..models import ESTIMATOR_REGISTRY
from ..models.virtual import OracleEstimator, VirtualEstimator
from ..offloader import (ConfiguredOffloader, OffloadContext, OffloadDecision,
                         Offloader, classify_metric)
from ._shared import (checkpoint_path as _checkpoint_path,
                      resolve_paths, resolve_target, select_input)
from .eval_cache import _eval_cache_dir_for, _eval_cache_signature, _eval_cache_valid, _load_eval_cache, _save_eval_cache
from .eval_metrics import compute_ranking_metrics, compute_regression_metrics
from .evaluation_profiles import (DefaultEvaluationProfile, EvaluationRun,
                                  get_evaluation_profile)
from .offloading import compute_map_at_ratio
from .paper_eval import (cache_key_for_run,
                         build_trace_rows,
                         compute_component_diagnostic_rows,
                         compute_prediction_diagnostics,
                         compute_resource_tradeoff_row,
                         compute_selection_diagnostics,
                         compute_slice_rows,
                         collect_qualitative_rows,
                         resolve_evaluation_seeds)

# Names of virtual estimators (zero-cost baselines) — imported from evaluate
# to avoid circular dependency; re-defined here for local use.
VIRTUAL_NAMES = {"weak_model", "strong_model"}

# Detection model specs: GFLOPs and params (M).
DETECTION_MODEL_SPECS = {
    # Torchvision
    "fasterrcnn_mobilenet_v3_large_fpn":     {"gflops": 4.49,   "params": 19.39},
    "fasterrcnn_mobilenet_v3_large_320_fpn": {"gflops": 0.72,   "params": 19.39},
    "fasterrcnn_resnet50_fpn_v2":            {"gflops": 280.37, "params": 43.71},
    "retinanet_resnet50_fpn_v2":             {"gflops": 152.24, "params": 38.20},
    "ssd300_vgg16":                          {"gflops": 34.86,  "params": 35.64},
    # YOLOv7
    "yolov7n":                               {"gflops": 4.20,   "params": 6.20},
    "yolov7l":                               {"gflops": 104.70, "params": 36.90},
    "yolov7":                                {"gflops": 104.70, "params": 36.90},
    "yolov7x":                               {"gflops": 189.90, "params": 71.30},
    # YOLO11
    "yolo11n":                               {"gflops": 6.50,   "params": 2.60},
    "yolo11s":                               {"gflops": 21.50,  "params": 9.40},
    "yolo11m":                               {"gflops": 68.00,  "params": 20.10},
    "yolo11l":                               {"gflops": 86.90,  "params": 25.30},
    "yolo11x":                               {"gflops": 194.90, "params": 56.90},
    # RT-DETR (transformer-based)
    "rtdetr-l":                              {"gflops": 110.00, "params": 32.00},
    "rtdetr-x":                              {"gflops": 232.00, "params": 65.00},
}


def _load_prepared(cfg: PipelineConfig):
    d = np.load(cfg.output.prepared_dir / "data.npz", allow_pickle=True)
    paths_test = resolve_paths(
        (cfg.output.prepared_dir / "paths_test.txt").read_text().splitlines())

    result = {
        "X_test": d["X_test"],
        "y_test": d["y_test"],
        "edge_test": d["edge_test"],
        "cloud_test": d["cloud_test"],
        "paths_test": paths_test,
    }
    # Optional COCO-style and per-frame keys
    for opt_key in ("edge_test_coco", "cloud_test_coco",
                    "edge_test_coco50", "cloud_test_coco50",
                    "video_name_test", "frame_id_test"):
        if opt_key in d:
            result[opt_key] = d[opt_key]
    for key in d.keys():
        if (key.startswith("y_test_") or key.startswith("X_test_")
                or key.startswith("meta_")):
            result[key] = d[key]

    return result


def _load_train_data(cfg: PipelineConfig) -> dict:
    """Load training-split features/paths for empirical threshold derivation."""
    d = np.load(cfg.output.prepared_dir / "data.npz", allow_pickle=True)
    paths_train = resolve_paths(
        (cfg.output.prepared_dir / "paths_train.txt").read_text().splitlines())
    result = {"X_train": d["X_train"], "paths_train": paths_train}
    for key in d.keys():
        if key.startswith(("X_train_", "y_train_", "meta_")):
            result[key] = d[key]
    return result


def _load_raw_test_boxes(cfg: PipelineConfig, video_names: np.ndarray,
                         frame_ids: np.ndarray) -> list:
    """Load raw bounding boxes for test-set frames from .pkl files."""
    import pickle

    raw_boxes = [None] * len(video_names)
    vid_to_frames: Dict[str, Dict] = {}
    for i, (v, f) in enumerate(zip(video_names, frame_ids)):
        if v not in vid_to_frames:
            vid_to_frames[v] = {}
        vid_to_frames[v][f] = i

    for v, f_dict in vid_to_frames.items():
        pkl_path = cfg.derived_dir / f"{v}_boxes.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Missing derived raw-box sidecar: {pkl_path}. Re-run prepare."
            )
        with open(pkl_path, "rb") as fh:
            v_boxes = pickle.load(fh)
        for frame_data in v_boxes:
            fid = frame_data["frame_id"]
            if fid in f_dict:
                raw_boxes[f_dict[fid]] = frame_data

    missing = sum(1 for b in raw_boxes if b is None)
    if missing:
        raise FileNotFoundError(
            f"Missing raw-box payloads for {missing}/{len(raw_boxes)} test frames. "
            "Re-run prepare to rebuild a complete derived cache."
        )
    return raw_boxes


def _test_input(pcfg: ApproachConfig, data: dict):
    return select_input(pcfg, data, "test")


def _train_input(pcfg: ApproachConfig, train_data: dict):
    return select_input(pcfg, train_data, "train")


def _try_cuda_sync(estimator):
    """Call torch.cuda.synchronize() if the estimator uses a CUDA model."""
    try:
        import torch
        if hasattr(estimator, 'model') and estimator.model is not None:
            device = next(estimator.model.parameters()).device
            if device.type == 'cuda':
                torch.cuda.synchronize()
    except (StopIteration, AttributeError, RuntimeError):
        pass


def _measure_latency(estimator, X_test, feature_type: str,
                     n_warmup: int = 5, n_samples: int = 50,
                     predict_kwargs: dict | None = None):
    """Measure single-sample inference latency in ms."""
    predict_kwargs = predict_kwargs or {}
    if hasattr(estimator, "measure_pure_latency"):
        return estimator.measure_pure_latency(X_test, n_warmup=n_warmup,
                                              n_samples=n_samples)
    n = min(len(X_test), n_samples)
    for _ in range(n_warmup):
        _ = _call_prediction_method(estimator, "predict", X_test[:1], predict_kwargs)
    _try_cuda_sync(estimator)
    times = []
    for i in range(n):
        sample = X_test[i:i + 1]
        _try_cuda_sync(estimator)
        t0 = time.perf_counter()
        _ = _call_prediction_method(estimator, "predict", sample, predict_kwargs)
        _try_cuda_sync(estimator)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times) * 1000)


def _active_evaluation_seed(cfg: PipelineConfig) -> int | None:
    seed = getattr(cfg, "_active_evaluation_seed", None)
    return None if seed is None else int(seed)


def _use_seeded_checkpoints(cfg: PipelineConfig) -> bool:
    return bool(getattr(cfg, "_use_seeded_checkpoints", False))


def _evaluation_checkpoint_path(cfg: PipelineConfig, pcfg: ApproachConfig) -> Path:
    seed = _active_evaluation_seed(cfg)
    if _use_seeded_checkpoints(cfg) and seed is not None:
        seeded = _checkpoint_path(cfg, pcfg, seed)
        if seeded.exists():
            return seeded
        legacy = _checkpoint_path(cfg, pcfg)
        if legacy.exists():
            return legacy
        return seeded
    return _checkpoint_path(cfg, pcfg)


def _clone_cfg_for_evaluation_seed(cfg: PipelineConfig, seed: int) -> PipelineConfig:
    """Shallow-clone cfg so worker tasks never mutate shared evaluation state."""
    cloned = copy.copy(cfg)
    cloned._active_evaluation_seed = int(seed)
    cloned._use_seeded_checkpoints = _use_seeded_checkpoints(cfg)
    return cloned


def _evaluation_uses_cuda(cfg: PipelineConfig) -> bool:
    device = str(getattr(cfg, "device", "") or "").strip().lower()
    if device and device != "auto":
        return device.startswith("cuda")
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _resolve_evaluation_workers(cfg: PipelineConfig, total_runs: int) -> int:
    total_runs = max(0, int(total_runs))
    if total_runs <= 1:
        return 1

    requested = getattr(cfg, "evaluation_num_workers", 1)
    if requested is None:
        requested = 1
    requested = int(requested)

    if _evaluation_uses_cuda(cfg):
        if requested != 1:
            log.info(
                "Evaluation parallelism disabled on CUDA; running sequentially.",
                indent=6,
            )
        return 1

    if requested == 0:
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, total_runs, 8))
    return max(1, min(requested, total_runs))


def _ratio_errors(decisions: dict) -> list[float]:
    if not decisions:
        return []
    return [
        float(decision.ratio_error)
        for decision in decisions.values()
        if getattr(decision, "target_ratio", float("nan")) == getattr(decision, "target_ratio", float("nan"))
        and 0.0 < float(getattr(decision, "target_ratio", 0.0)) < 1.0
    ]


def _nearest_ratio(target: float, available: list[float]) -> float:
    return min(
        (float(x) for x in available),
        key=lambda x: (abs(x - float(target)), -x),
    )


def _resolve_trace_source_ratio(target: float, available: list[float]) -> float:
    target = float(target)
    for ratio in available:
        ratio = float(ratio)
        if np.isclose(ratio, target):
            return ratio
    return _nearest_ratio(target, available)


def _detector_cost_profile(cfg: PipelineConfig, results_list: list[dict]) -> dict[str, float]:
    weak_meta = DETECTION_MODEL_SPECS.get(cfg.dataset.edge_model, {})
    cloud_meta = DETECTION_MODEL_SPECS.get(cfg.dataset.cloud_model, {})
    weak_result = next((row for row in results_list if row.get("estimator") == "weak_model"), {})
    cloud_result = next((row for row in results_list if row.get("estimator") == "strong_model"), {})
    return {
        "weak_detector_time_ms": float(weak_result.get("detector_time_ms", 0.0) or 0.0),
        "cloud_detector_time_ms": float(cloud_result.get("detector_time_ms", 0.0) or 0.0),
        "weak_detector_gflops": float(weak_meta.get("gflops", 0.0) or 0.0),
        "cloud_detector_gflops": float(cloud_meta.get("gflops", 0.0) or 0.0),
    }


def _fixed_threshold_trace(mask: np.ndarray, actual_ratio: float,
                           threshold: float) -> dict[str, np.ndarray]:
    order = np.arange(len(mask), dtype=int)
    trace = Offloader._build_trace(
        np.asarray(mask, dtype=bool),
        float(actual_ratio),
        order,
        control_trace=np.full(len(mask), float(threshold), dtype=float),
    )
    trace["threshold_trace"] = trace.pop("threshold")
    return trace


def _constant_curve(ratios: list[float], ap50: float, ap_coco: float, ap_coco50: float) -> dict[float, dict[str, float]]:
    return {
        float(ratio): {
            "ap50": float(ap50),
            "ap_coco": float(ap_coco),
            "ap50_allpoint": float(ap_coco50),
            "n_offload": int(round(float(ratio) * len(ratios))) if ratios else 0,
        }
        for ratio in ratios
    }


def _build_evaluation_run(scenario_name: str = None,
                          scenario_weights=None) -> EvaluationRun:
    predict_kwargs = {}
    row_metadata = {}
    if scenario_name is not None:
        row_metadata["scenario"] = scenario_name
    if scenario_weights is not None:
        predict_kwargs["scenario_weights"] = scenario_weights
    return EvaluationRun(
        name_suffix=scenario_name,
        predict_kwargs=predict_kwargs,
        row_metadata=row_metadata,
    )


def _predict_kwargs(run: EvaluationRun) -> dict:
    return dict(run.predict_kwargs)


def _call_prediction_method(estimator, method_name: str, X, predict_kwargs: dict):
    method = getattr(estimator, method_name)
    if not predict_kwargs:
        return method(X)
    try:
        params = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return method(X, **predict_kwargs)
    supports_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)
    accepted = {param.name for param in params}
    if supports_kwargs or all(key in accepted for key in predict_kwargs):
        return method(X, **predict_kwargs)
    return method(X)


def _apply_estimator_metadata(row: dict, estimator_name: str,
                              metadata_by_name: Dict[str, dict]) -> None:
    row.update(metadata_by_name.get(estimator_name, {}))


def _predict_output_keys(predict_outputs: object) -> set[str]:
    if not isinstance(predict_outputs, dict):
        return set()
    return {str(key) for key in predict_outputs}


def _write_optional_json(path: Path, rows: list[dict]) -> None:
    if rows:
        path.write_text(json.dumps(rows, indent=2))
    elif path.exists():
        path.unlink()


def _measure_detector_latency(model_name: str, image_paths: list,
                               n_warmup: int = 5, n_samples: int = 20,
                               device: str = None) -> float:
    """Measure per-frame detection latency for a real detection model (ms).

    Args:
        device: Device to run on (e.g. "cuda:0"). If None, auto-detects.
    """
    import time

    import torch
    from PIL import Image

    from .. import detector as _det_module
    from ..detector import Detector

    n = min(len(image_paths), n_samples)
    if n == 0:
        return 0.0

    # Temporarily override FORCE_DEVICE so Detector uses the requested device
    old_force = _det_module.FORCE_DEVICE
    if device:
        _det_module.FORCE_DEVICE = device
    try:
        det = Detector(model_name, conf_threshold=0.0, vehicle_only=False)
    finally:
        _det_module.FORCE_DEVICE = old_force

    use_cuda = "cuda" in det.device
    sample_paths = image_paths[:n]

    if det._backend == "ultralytics":
        # YOLO models handle their own preprocessing; pass file paths directly
        for _ in range(n_warmup):
            det.model(sample_paths[:1], verbose=False)
        if use_cuda:
            torch.cuda.synchronize()

        times = []
        for i in range(n):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            det.model([sample_paths[i]], verbose=False)
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    else:
        # Torchvision backend: pre-load tensors to avoid IO in timing
        tensors = []
        for path in sample_paths:
            img = Image.open(path).convert("RGB")
            tensors.append(det.transform(img).to(det.device))

        for _ in range(n_warmup):
            with torch.no_grad():
                _ = det.model([tensors[0]])
        if use_cuda:
            torch.cuda.synchronize()

        times = []
        for i in range(n):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = det.model([tensors[i]])
            if use_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return float(np.mean(times) * 1000)


def evaluate_one(pcfg: ApproachConfig, cfg: PipelineConfig,
                 data: dict, train_data: dict = None,
                 scenario_name: str = None,
                 scenario_weights=None) -> Dict:
    """Load or create estimator, predict, compute metrics."""
    base_name = "oracle" if pcfg.registry_key.startswith("oracle") else pcfg.registry_key
    cls = ESTIMATOR_REGISTRY.get(base_name)
    eval_run = _build_evaluation_run(scenario_name, scenario_weights)
    profile = get_evaluation_profile(cls) if cls is not None else DefaultEvaluationProfile()
    estimator_name = profile.format_estimator_name(pcfg, eval_run)
    result = {"estimator": estimator_name, "base_model": pcfg.registry_key,
              "stage": pcfg.stage, "status": "FAIL"}
    profile.apply_result_metadata(result, eval_run)

    if cls is None:
        result["error"] = f"Unknown estimator: {pcfg.name}"
        return result

    # --- Virtual estimators (zero-cost baselines) -------------------------
    if issubclass(cls, VirtualEstimator):
        estimator = cls(device=getattr(cfg, "device", "auto"))
        if isinstance(estimator, OracleEstimator):
            estimator.set_ground_truth(data["y_test"])
        estimator.is_fitted = True

        predict_kwargs = _predict_kwargs(eval_run)
        X_test = _test_input(pcfg, data)
        y_test = profile.resolve_reporting_target(
            pcfg, data, "test", cls, eval_run
        )

        preds = _call_prediction_method(estimator, "predict", X_test, predict_kwargs)

        if pcfg.registry_key.startswith("oracle"):
            preds = y_test
            # Optional: ensure oracle latency is recorded as 0
            if "latency_mean" not in result:
                result["latency_mean"] = 0.0
                result["latency_std"] = 0.0

        metrics = compute_regression_metrics(y_test, preds)
        result.update(metrics)
        result.update(estimator.get_info())
        result["inference_time_ms"] = 0.0
        result["status"] = "PASS"
        result["_predictions"] = preds

        # For weak/strong: inject real detection model specs
        det_model = None
        if pcfg.registry_key == "weak_model":
            det_model = cfg.dataset.edge_model if cfg else None
        elif pcfg.registry_key == "strong_model":
            det_model = cfg.dataset.cloud_model if cfg else None

        if det_model and det_model in DETECTION_MODEL_SPECS:
            specs = DETECTION_MODEL_SPECS[det_model]
            result["gflops"] = specs["gflops"]
            result["params"] = specs["params"]
            result["description"] = (
                f"{result.get('description', '')} [{det_model}]"
            )

        return result

    # --- Real estimators (require checkpoint) -----------------------------
    ckpt = _evaluation_checkpoint_path(cfg, pcfg)

    try:
        if not ckpt.exists():
            result["error"] = f"Checkpoint not found: {ckpt}"
            return result

        estimator = cls.load(ckpt, device=getattr(cfg, "device", "auto"))
        predict_kwargs = _predict_kwargs(eval_run)
        X_test = _test_input(pcfg, data)
        y_test = profile.resolve_reporting_target(
            pcfg, data, "test", cls, eval_run
        )

        result["inference_time_ms"] = _measure_latency(
            estimator, X_test, pcfg.feature_type,
            n_warmup=cfg.latency_warmup, n_samples=cfg.latency_samples,
            predict_kwargs=predict_kwargs,
        )
        if hasattr(estimator, "get_info"):
            result.update(estimator.get_info())

        predict_outputs = None
        if hasattr(estimator, "predict_outputs"):
            predict_outputs = _call_prediction_method(
                estimator, "predict_outputs", X_test, predict_kwargs
            )
        if isinstance(predict_outputs, dict):
            preds = predict_outputs.get("primary")
            if preds is None:
                preds = predict_outputs.get("score")
        else:
            preds = None
        if preds is None:
            preds = _call_prediction_method(estimator, "predict", X_test, predict_kwargs)
        metrics = compute_regression_metrics(y_test, preds)
        result.update(metrics)

        # Binary classification metrics for classifiers (e.g. DCSB)
        if hasattr(estimator, 'predict_proba') and estimator.predict_proba(X_test[:1]) is not None:
            from .eval_metrics import compute_classification_metrics
            y_bin = (y_test > 0).astype(int)
            pred_bin = (preds >= 0.5).astype(int)
            proba = estimator.predict_proba(X_test)
            cls_metrics = compute_classification_metrics(y_bin, pred_bin, proba)
            result.update({f"cls_{k}": v for k, v in cls_metrics.items()})

        result["status"] = "PASS"
        result["_predictions"] = preds
        if predict_outputs is not None:
            result["_predict_outputs"] = predict_outputs

        # Train predictions are saved for thresholding diagnostics and train/test shift charts.
        if train_data is not None:
            X_train = _train_input(pcfg, train_data)
            result["_train_predictions"] = _call_prediction_method(
                estimator, "predict", X_train, predict_kwargs
            )
            if hasattr(estimator, "predict_outputs"):
                result["_train_predict_outputs"] = _call_prediction_method(
                    estimator, "predict_outputs", X_train, predict_kwargs
                )

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def compute_offloading(predictions: np.ndarray, raw_boxes: list,
                       ratios: List[float]) -> Dict[float, Dict[str, float]]:
    from .offloading import compute_dataset_map_batch
    log.info("Precomputing IoU for all detections ...", indent=6)
    results = compute_dataset_map_batch(raw_boxes, predictions, ratios)
    return results


def compute_offloading_for_offloader(
    offloader,
    context,
    raw_boxes: list,
    ratios: List[float],
) -> tuple[dict[float, dict[str, float]], dict[float, OffloadDecision], Optional[dict], Optional[OffloadDecision]]:
    """Evaluate one configured offloader for one approach."""
    from .offloading import _eval_offload_set, _prepare_detection_data

    prep = _prepare_detection_data(raw_boxes)
    if prep is None:
        empty = {'ap50': 0.0, 'ap50_allpoint': 0.0, 'ap_coco': 0.0, 'n_offload': 0}
        if offloader.mode == "fixed":
            return {}, {}, empty, None
        return ({float(r): empty.copy() for r in ratios}, {}, None, None)

    (valid_indices, precomputed, det_source, det_frame,
     frame_to_valid_idx, gt_by_frame, total_gt) = prep
    valid_set = set(valid_indices)

    def _evaluate_decision(decision: OffloadDecision) -> dict[str, float]:
        selected = {
            idx for idx in range(len(decision.mask))
            if decision.mask[idx] and idx in valid_set
        }
        return _eval_offload_set(
            selected,
            precomputed,
            det_source,
            det_frame,
            frame_to_valid_idx,
            gt_by_frame,
            total_gt,
        )

    log.info("Precomputing IoU for all detections ...", indent=6)

    if offloader.mode == "fixed":
        decision = offloader.decide(context, None)
        if decision.trace is None:
            decision.trace = _fixed_threshold_trace(
                decision.mask,
                decision.actual_ratio,
                decision.threshold,
            )
        result = _evaluate_decision(decision)
        log.info(
            f"Fixed policy {offloader.name}: AP50={result['ap50']:.4f}  "
            f"actual_ratio={decision.actual_ratio:.3f}",
            indent=6,
        )
        return {}, {}, result, decision

    curve: dict[float, dict[str, float]] = {}
    decisions: dict[float, OffloadDecision] = {}
    for ratio in ratios:
        decision = offloader.decide(context, float(ratio))
        decisions[float(ratio)] = decision
        curve[float(ratio)] = _evaluate_decision(decision)

    return curve, decisions, None, None


def compute_offloading_combined(
    predictions: np.ndarray,
    raw_boxes: list,
    ratios: List[float],
    offloader,
    *,
    train_predictions: np.ndarray = None,
    predict_outputs: Optional[dict] = None,
    train_predict_outputs: Optional[dict] = None,
    stream_order: Optional[np.ndarray] = None,
    proxy_metric: Optional[str] = None,
) -> tuple[dict[float, dict[str, float]], dict[float, OffloadDecision], Optional[dict], Optional[OffloadDecision]]:
    """Compatibility wrapper around the one-offloader evaluation path."""
    context = OffloadContext(
        predictions=np.asarray(predictions, dtype=float),
        proxy_metric=proxy_metric,
        train_predictions=(
            np.asarray(train_predictions, dtype=float)
            if train_predictions is not None else None
        ),
        predict_outputs=predict_outputs if isinstance(predict_outputs, dict) else None,
        train_predict_outputs=(
            train_predict_outputs if isinstance(train_predict_outputs, dict) else None
        ),
        stream_order=stream_order,
    )
    return compute_offloading_for_offloader(offloader, context, raw_boxes, ratios)


def _compute_offloading_combined_compat(*args, **kwargs) -> tuple[dict, dict, Optional[dict], Optional[OffloadDecision]]:
    """Normalize legacy multi-strategy helper payloads into one-offloader output."""
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


def _compute_random_baseline(raw_boxes: list,
                             ratios: List[float]) -> Dict[float, Dict[str, float]]:
    """Random offloading baseline."""
    from .offloading import compute_dataset_map_batch
    rng = np.random.RandomState(42)
    rand_preds = rng.rand(len(raw_boxes))
    log.info("Computing random offloading AP (batch)...", indent=6)
    return compute_dataset_map_batch(raw_boxes, rand_preds, ratios)


def _build_evaluation_tasks(cfg: PipelineConfig,
                            approaches: list[ApproachConfig],
                            seeds: list[int]) -> tuple[list[dict], list[dict]]:
    tasks: list[dict] = []
    scenario_profile_rows: list[dict] = []
    for pcfg in approaches:
        base_key = "oracle" if pcfg.registry_key.startswith("oracle") else pcfg.registry_key
        est_cls = ESTIMATOR_REGISTRY.get(base_key)
        profile = (get_evaluation_profile(est_cls)
                   if est_cls is not None else DefaultEvaluationProfile())
        if est_cls is not None:
            scenario_profile_rows.extend(
                profile.describe_scenario_profiles(cfg, pcfg, est_cls)
            )
        for seed in seeds:
            eval_runs = (profile.iter_runs(cfg, pcfg, est_cls)
                         if est_cls is not None else [EvaluationRun()])
            for eval_run in eval_runs:
                scenario_name = eval_run.name_suffix
                title = pcfg.name if scenario_name is None else f"{pcfg.name} [{scenario_name}]"
                if len(seeds) > 1:
                    title = f"{title} [seed={int(seed)}]"
                tasks.append({
                    "seed": int(seed),
                    "pcfg": pcfg,
                    "profile": profile,
                    "est_cls": est_cls,
                    "eval_run": eval_run,
                    "title": title,
                })
    return tasks, scenario_profile_rows


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
