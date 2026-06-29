"""Paper-oriented evaluation helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..error_decomposition import LCER_ERROR_TYPES
from ..features import EDGE_FEATURES
from ..offloader import classify_metric
from .eval_metrics import compute_ranking_metrics

STATIC_PROPOSED_BASE_MODELS = {
    "adaptive_scenario_hybrid",
    "adaptive_scenario_lcer6",
    "adaptive_scenario_lcerfg",
    "hybrid_lcer_csr",
    "hybrid_lcer_csr_utility",
}

_LEGACY_POLICY_LABELS = {
    "native_threshold": "threshold",
    "online_ecdf_calibrated": "calibrated",
    "fixed_classifier": "fixed",
}


def _ensure_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "policy_id" not in out.columns and "strategy" in out.columns:
        out["policy_id"] = out["strategy"].astype(str)
    if "offloader_id" not in out.columns:
        if "policy_id" in out.columns:
            out["offloader_id"] = out["policy_id"].astype(str)
        elif "strategy" in out.columns:
            out["offloader_id"] = out["strategy"].astype(str)
    if "strategy" not in out.columns and "policy_id" in out.columns:
        policy = out["policy_id"].astype(str)
        out["strategy"] = policy.map(_LEGACY_POLICY_LABELS).fillna(policy)
    return out


def _proposed_base_models(metrics_runs: pd.DataFrame,
                          offload_summary: pd.DataFrame) -> set[str]:
    proposed = set(STATIC_PROPOSED_BASE_MODELS)
    for df in (metrics_runs, offload_summary):
        if df.empty or "base_model" not in df.columns or "scenario" not in df.columns:
            continue
        scenario_rows = df["scenario"].fillna("").astype(str) != ""
        proposed.update(df.loc[scenario_rows, "base_model"].astype(str).tolist())
    return proposed


def resolve_evaluation_seeds(cfg) -> list[int]:
    seeds = getattr(cfg, "evaluation_seeds", None) or [cfg.seed]
    return [int(seed) for seed in seeds]


def cache_key_for_run(estimator_name: str, seed: int, n_seeds: int) -> str:
    if n_seeds <= 1:
        return estimator_name
    return f"{estimator_name}__seed{int(seed)}"


def bootstrap_ci(values: Iterable[float], samples: int = 1000,
                 seed: int = 42, alpha: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=float)
    for idx in range(int(samples)):
        means[idx] = float(rng.choice(arr, size=arr.size, replace=True).mean())
    lower = float(np.percentile(means, 100 * (1 - alpha) / 2.0))
    upper = float(np.percentile(means, 100 * (1 + alpha) / 2.0))
    return lower, upper


def _has_variation(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < 2:
        return False
    return not np.allclose(arr, arr[0])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if not _has_variation(x) or not _has_variation(y):
        return 0.0
    corr, _ = spearmanr(x, y)
    return float(corr) if np.isfinite(corr) else 0.0


def _score_boundary(proxy_metric: str | None) -> tuple[float, float]:
    proxy_metric = str(proxy_metric or "")
    if proxy_metric.startswith("moric_"):
        return 0.5, 0.1
    metric_type = classify_metric(proxy_metric).value
    if metric_type == "moric":
        return 0.5, 0.1
    if metric_type == "moric_plus":
        return 0.0, 0.1
    return 0.0, 0.1


def compute_prediction_diagnostics(y_true: np.ndarray, y_pred: np.ndarray,
                                   bins: int = 10,
                                   proxy_metric: str | None = None,
                                   y_train_true: Optional[np.ndarray] = None,
                                   y_train_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
    from scipy.stats import ks_2samp

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return {}

    residual = y_pred - y_true
    boundary_center, boundary_width = _score_boundary(proxy_metric)
    diag = {
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_true": float(np.std(y_true)),
        "std_pred": float(np.std(y_pred)),
        "spread_ratio": float(np.std(y_pred) / max(np.std(y_true), 1e-10)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_residual": float(np.mean(residual)),
        "overprediction_rate": float(np.mean(residual > 0)),
        "underprediction_rate": float(np.mean(residual < 0)),
        "ks_distance": float(ks_2samp(y_true, y_pred).statistic),
        "near_boundary_rate_pred": float(np.mean(np.abs(y_pred - boundary_center) <= boundary_width)),
        "near_boundary_rate_true": float(np.mean(np.abs(y_true - boundary_center) <= boundary_width)),
    }

    if len(np.unique(y_pred)) > 1:
        slope, intercept = np.polyfit(y_pred, y_true, deg=1)
        diag["calibration_slope"] = float(slope)
        diag["calibration_intercept"] = float(intercept)
    else:
        diag["calibration_slope"] = float("nan")
        diag["calibration_intercept"] = float(np.mean(y_true) - np.mean(y_pred))

    order = np.argsort(y_pred)
    pred_sorted = y_pred[order]
    true_sorted = y_true[order]
    bin_edges = np.linspace(0, len(y_pred), num=max(int(bins), 1) + 1, dtype=int)
    gaps = []
    quantile_mae = []
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        if end <= start:
            continue
        pred_bin = pred_sorted[start:end]
        true_bin = true_sorted[start:end]
        gaps.append(abs(float(pred_bin.mean()) - float(true_bin.mean())))
        quantile_mae.append(float(np.mean(np.abs(pred_bin - true_bin))))
    diag["calibration_gap"] = float(np.mean(gaps)) if gaps else float("nan")
    diag["quantile_mae"] = float(np.mean(quantile_mae)) if quantile_mae else float("nan")

    if y_train_true is not None and y_train_pred is not None:
        y_train_true = np.asarray(y_train_true, dtype=float).reshape(-1)
        y_train_pred = np.asarray(y_train_pred, dtype=float).reshape(-1)
        if len(y_train_true) == len(y_train_pred) and len(y_train_true) > 0:
            diag["train_mean_true"] = float(np.mean(y_train_true))
            diag["train_mean_pred"] = float(np.mean(y_train_pred))
            diag["train_std_true"] = float(np.std(y_train_true))
            diag["train_std_pred"] = float(np.std(y_train_pred))
            diag["train_test_pred_ks"] = float(ks_2samp(y_train_pred, y_pred).statistic)
            diag["train_test_pred_mean_shift"] = float(np.mean(y_pred) - np.mean(y_train_pred))
            diag["train_test_pred_std_ratio"] = float(np.std(y_pred) / max(np.std(y_train_pred), 1e-10))
    return diag


def compute_proxy_metric_stats(name: str, values: np.ndarray,
                               utility_reference: Optional[np.ndarray] = None,
                               oracle_reference: Optional[np.ndarray] = None) -> Dict[str, float]:
    from scipy.stats import kurtosis, skew

    arr = np.asarray(values, dtype=float).reshape(-1)
    row = {
        "proxy_metric": name,
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr)) if arr.size else float("nan"),
        "p05": float(np.percentile(arr, 5)) if arr.size else float("nan"),
        "p50": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p95": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "skew": float(skew(arr)) if arr.size > 2 and _has_variation(arr) else float("nan"),
        "kurtosis": float(kurtosis(arr)) if arr.size > 3 and _has_variation(arr) else float("nan"),
        "zero_rate": float(np.mean(np.isclose(arr, 0.0))) if arr.size else float("nan"),
        "positive_rate": float(np.mean(arr > 0.0)) if arr.size else float("nan"),
    }
    if utility_reference is not None and len(utility_reference) == len(arr):
        row["utility_spearman"] = _safe_spearman(arr, utility_reference)
    if oracle_reference is not None and len(oracle_reference) == len(arr):
        row["oracle_spearman"] = _safe_spearman(arr, oracle_reference)
    return row


def collect_proxy_metric_rows(data: dict) -> list[dict]:
    utility_ref = np.asarray(data.get("y_test", []), dtype=float).reshape(-1)
    oracle_ref = np.asarray(
        data.get("y_test_dataset_oric_11pt", data.get("y_test", [])),
        dtype=float,
    ).reshape(-1)
    rows: list[dict] = []
    for key, value in data.items():
        if not key.startswith("y_test_"):
            continue
        arr = np.asarray(value)
        name = key.removeprefix("y_test_")
        if arr.ndim == 1:
            if not np.issubdtype(arr.dtype, np.number):
                continue
            rows.append(
                compute_proxy_metric_stats(
                    name,
                    arr,
                    utility_reference=utility_ref if len(utility_ref) == len(arr) else None,
                    oracle_reference=oracle_ref if len(oracle_ref) == len(arr) else None,
                )
            )
            continue
        if arr.ndim == 2 and (
            name.startswith("finegrained_vec_") or name.startswith("lcer_vec_")
        ):
            if name.startswith("finegrained_vec_"):
                suffix = name.removeprefix("finegrained_vec_")
                comp_names = np.asarray(
                    data.get(f"meta_scenario_component_names_{suffix}", []), dtype=object
                ).reshape(-1)
            else:
                comp_names = np.asarray(LCER_ERROR_TYPES, dtype=object)
            if len(comp_names) != arr.shape[1]:
                comp_names = np.asarray(
                    [f"component_{idx}" for idx in range(arr.shape[1])], dtype=object
                )
            for idx, comp_name in enumerate(comp_names):
                rows.append(
                    compute_proxy_metric_stats(
                        f"{name}::{comp_name}",
                        arr[:, idx],
                        utility_reference=utility_ref if len(utility_ref) == len(arr) else None,
                        oracle_reference=oracle_ref if len(oracle_ref) == len(arr) else None,
                    )
                )
    return rows


def compute_component_diagnostic_rows(estimator_name: str, base_model: str,
                                      target_bundle: object, predict_outputs: object,
                                      seed: int, scenario: str | None = None,
                                      scenario_type: str | None = None) -> list[dict]:
    if not isinstance(target_bundle, dict) or not isinstance(predict_outputs, dict):
        return []

    rows: list[dict] = []
    common = {
        "estimator": estimator_name,
        "base_model": base_model,
        "seed": int(seed),
        "scenario": scenario or "",
        "scenario_type": scenario_type or "",
    }

    if "vector" in target_bundle and "vector" in predict_outputs:
        true = np.asarray(target_bundle["vector"], dtype=float)
        pred = np.asarray(predict_outputs["vector"], dtype=float)
        names = target_bundle.get("meta_component_names", predict_outputs.get("component_names"))
        if names is None:
            names = [f"component_{idx}" for idx in range(pred.shape[1])]
        flat_names = [str(name) for name in np.asarray(names, dtype=object).reshape(-1)]
        for idx, name in enumerate(flat_names[: pred.shape[1]]):
            rows.append({
                **common,
                "diagnostic_family": "vector",
                "component": name,
                "mae": float(np.mean(np.abs(pred[:, idx] - true[:, idx]))),
                "rmse": float(np.sqrt(np.mean(np.square(pred[:, idx] - true[:, idx])))),
                "spearman_rho": _safe_spearman(pred[:, idx], true[:, idx]),
            })

    if "survival" in target_bundle and "survival" in predict_outputs:
        true = np.asarray(target_bundle["survival"], dtype=float)
        pred = np.asarray(predict_outputs["survival"], dtype=float)
        tau = np.asarray(target_bundle.get("meta_tau", predict_outputs.get("tau", [])), dtype=float).reshape(-1)
        for idx in range(min(pred.shape[1], true.shape[1])):
            rows.append({
                **common,
                "diagnostic_family": "survival",
                "component": f"tau_{tau[idx]:.4f}" if idx < len(tau) else f"tau_{idx}",
                "mae": float(np.mean(np.abs(pred[:, idx] - true[:, idx]))),
                "rmse": float(np.sqrt(np.mean(np.square(pred[:, idx] - true[:, idx])))),
                "brier": float(np.mean(np.square(pred[:, idx] - true[:, idx]))),
                "accuracy": float(np.mean((pred[:, idx] >= 0.5) == (true[:, idx] >= 0.5))),
            })

    if "ordinal" in target_bundle and "ordinal_probs" in predict_outputs:
        true = np.asarray(target_bundle["ordinal"], dtype=float)
        pred = np.asarray(predict_outputs["ordinal_probs"], dtype=float)
        true_bucket = true.sum(axis=1)
        pred_bucket = (pred >= 0.5).sum(axis=1)
        rows.append({
            **common,
            "diagnostic_family": "ordinal",
            "component": "overall",
            "mae": float(np.mean(np.abs(pred_bucket - true_bucket))),
            "rmse": float(np.sqrt(np.mean(np.square(pred_bucket - true_bucket)))),
            "exact_bucket_accuracy": float(np.mean(pred_bucket == true_bucket)),
            "under_budget_rate": float(np.mean(pred_bucket < true_bucket)),
            "over_budget_rate": float(np.mean(pred_bucket > true_bucket)),
        })

    if "spatial" in target_bundle and "spatial" in predict_outputs:
        true = np.asarray(target_bundle["spatial"], dtype=float)
        pred = np.asarray(predict_outputs["spatial"], dtype=float)
        flat_true = true.reshape(len(true), -1)
        flat_pred = pred.reshape(len(pred), -1)
        top_k = max(1, int(flat_true.shape[1] * 0.1))
        hotspot_overlap = []
        for true_row, pred_row in zip(flat_true, flat_pred):
            true_idx = set(np.argsort(true_row)[-top_k:])
            pred_idx = set(np.argsort(pred_row)[-top_k:])
            hotspot_overlap.append(len(true_idx & pred_idx) / top_k)
        rows.append({
            **common,
            "diagnostic_family": "spatial",
            "component": "grid",
            "mae": float(np.mean(np.abs(flat_pred - flat_true))),
            "rmse": float(np.sqrt(np.mean(np.square(flat_pred - flat_true)))),
            "hotspot_recall": float(np.mean(hotspot_overlap)),
        })

    return rows


def build_slice_sources(data: dict) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = {}
    if "video_name_test" in data:
        sources["video_name"] = np.asarray(data["video_name_test"], dtype=object).reshape(-1)
        if "frame_id_test" in data:
            video_names = np.asarray(data["video_name_test"], dtype=object).reshape(-1)
            frame_ids = np.asarray(data["frame_id_test"], dtype=int).reshape(-1)
            quartiles = np.empty(len(video_names), dtype=object)
            for video_name in np.unique(video_names):
                idx = np.where(video_names == video_name)[0]
                order = idx[np.argsort(frame_ids[idx])]
                bins = np.array_split(order, 4)
                for quartile_idx, members in enumerate(bins, start=1):
                    quartiles[members] = f"Q{quartile_idx}"
            sources["frame_quartile"] = quartiles

    X_test = np.asarray(data.get("X_test", np.empty((0, 0))), dtype=float)
    if X_test.ndim == 2 and X_test.shape[1] >= len(EDGE_FEATURES):
        edge_idx = {name: idx for idx, name in enumerate(EDGE_FEATURES)}
        if "edge_det_count" in edge_idx:
            sources["edge_det_count"] = X_test[:, edge_idx["edge_det_count"]]
        if "edge_conf_mean" in edge_idx:
            sources["edge_conf_mean"] = X_test[:, edge_idx["edge_conf_mean"]]

    for key, name in (("y_test_entropy", "entropy"), ("y_test_img_complexity", "img_complexity")):
        if key in data:
            sources[name] = np.asarray(data[key], dtype=float).reshape(-1)
    return sources


def _quantile_labels(values: np.ndarray, bins: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return np.asarray([], dtype=object)
    try:
        labels = pd.qcut(values, q=bins, labels=[f"Q{i}" for i in range(1, bins + 1)], duplicates="drop")
        return np.asarray(labels.astype(str), dtype=object)
    except ValueError:
        return np.asarray(["Q1"] * len(values), dtype=object)


def _slice_label_sets(data: dict) -> list[tuple[str, np.ndarray]]:
    labeled_slices: list[tuple[str, np.ndarray]] = []
    for slice_name, raw_values in build_slice_sources(data).items():
        values = np.asarray(raw_values)
        if values.dtype.kind in {"U", "S", "O"}:
            labels = values.astype(object)
        else:
            labels = _quantile_labels(values)
        labeled_slices.append((slice_name, labels))
    return labeled_slices


def compute_slice_rows(estimator_name: str, seed: int, y_true: np.ndarray,
                       y_pred: np.ndarray, data: dict,
                       scenario: str | None = None,
                       scenario_type: str | None = None) -> list[dict]:
    rows: list[dict] = []
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    for slice_name, labels in _slice_label_sets(data):
        for label in pd.unique(labels):
            mask = labels == label
            count = int(np.sum(mask))
            if count < 3:
                continue
            metrics = compute_ranking_metrics(
                y_true[mask], y_pred[mask],
                top_k=[max(1, int(round(count * frac))) for frac in (0.1, 0.2)],
            )
            k25 = max(1, int(round(count * 0.25)))
            pred_top = np.argsort(y_pred[mask])[-k25:]
            oracle_top = np.argsort(y_true[mask])[-k25:]
            pred_gain = float(np.mean(y_true[mask][pred_top]))
            oracle_gain = float(np.mean(y_true[mask][oracle_top]))
            rows.append({
                "estimator": estimator_name,
                "seed": int(seed),
                "scenario": scenario or "",
                "scenario_type": scenario_type or "",
                "slice_name": slice_name,
                "slice_value": str(label),
                "count": count,
                "mae": float(np.mean(np.abs(y_pred[mask] - y_true[mask]))),
                "rmse": float(np.sqrt(np.mean(np.square(y_pred[mask] - y_true[mask])))),
                "spearman_rho": float(metrics.get("spearman_rho", np.nan)),
                "ndcg": float(metrics.get("ndcg", np.nan)),
                "pred_gain_at_25": pred_gain,
                "oracle_gain_at_25": oracle_gain,
                "regret_at_25": float(oracle_gain - pred_gain),
            })
    return rows


def compute_slice_opportunity_rows(data: dict) -> list[dict]:
    y_true = np.asarray(data.get("y_test", []), dtype=float).reshape(-1)
    edge = np.asarray(data.get("edge_test", np.zeros(len(y_true))), dtype=float).reshape(-1)
    cloud = np.asarray(data.get("cloud_test", edge), dtype=float).reshape(-1)
    if len(y_true) == 0:
        return []

    rows: list[dict] = []
    for slice_name, labels in _slice_label_sets(data):
        for label in pd.unique(labels):
            mask = labels == label
            count = int(np.sum(mask))
            if count < 3:
                continue
            values = y_true[mask]
            rows.append({
                "slice_name": slice_name,
                "slice_value": str(label),
                "count": count,
                "mean_true_gain": float(np.mean(values)),
                "median_true_gain": float(np.median(values)),
                "beneficial_rate": float(np.mean(values > 0.0)),
                "harmful_rate": float(np.mean(values < 0.0)),
                "neutral_rate": float(np.mean(np.isclose(values, 0.0))),
                "edge_map_mean": float(np.mean(edge[mask])) if len(edge) == len(y_true) else float("nan"),
                "cloud_map_mean": float(np.mean(cloud[mask])) if len(cloud) == len(y_true) else float("nan"),
                "headroom_mean": float(np.mean(cloud[mask] - edge[mask])) if len(edge) == len(y_true) and len(cloud) == len(y_true) else float("nan"),
            })
    return rows


def compute_selection_diagnostics(estimator_name: str, base_model: str, seed: int,
                                  y_true: np.ndarray, selection_mask: np.ndarray,
                                  target_ratio: float, strategy: str,
                                  scenario: str | None = None,
                                  scenario_type: str | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    selection_mask = np.asarray(selection_mask, dtype=bool).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(selection_mask):
        return {}

    selected_idx = np.flatnonzero(selection_mask)
    n_selected = int(len(selected_idx))
    actual_ratio = float(n_selected / len(y_true)) if len(y_true) > 0 else 0.0
    oracle_k = min(n_selected, len(y_true))
    oracle_idx = (
        np.argsort(y_true)[-oracle_k:] if oracle_k > 0 else np.asarray([], dtype=int)
    )
    selected_set = set(selected_idx.tolist())
    oracle_set = set(oracle_idx.tolist())
    beneficial_mask = y_true > 0.0
    harmful_mask = y_true < 0.0
    selected_gain_sum = float(np.sum(y_true[selected_idx])) if n_selected > 0 else 0.0
    oracle_gain_sum = float(np.sum(y_true[oracle_idx])) if oracle_k > 0 else 0.0
    selected_beneficial = int(np.sum(beneficial_mask[selected_idx])) if n_selected > 0 else 0
    intersection = len(selected_set & oracle_set)
    union = len(selected_set | oracle_set)

    return {
        "estimator": estimator_name,
        "base_model": base_model,
        "seed": int(seed),
        "scenario": scenario or "",
        "scenario_type": scenario_type or "",
        "strategy": strategy,
        "target_ratio": float(target_ratio),
        "actual_ratio": actual_ratio,
        "selected_count": n_selected,
        "beneficial_selection_precision": (
            float(selected_beneficial / n_selected) if n_selected > 0 else 1.0
        ),
        "beneficial_selection_recall": (
            float(selected_beneficial / np.sum(beneficial_mask))
            if np.sum(beneficial_mask) > 0 else 1.0
        ),
        "harmful_offload_rate": (
            float(np.mean(harmful_mask[selected_idx])) if n_selected > 0 else 0.0
        ),
        "oracle_overlap_jaccard": float(intersection / union) if union > 0 else 1.0,
        "oracle_overlap_recall": float(intersection / len(oracle_set)) if len(oracle_set) > 0 else 1.0,
        "selected_true_gain_sum": selected_gain_sum,
        "oracle_true_gain_sum": oracle_gain_sum,
        "scenario_gain": float(selected_gain_sum / len(y_true)) if len(y_true) > 0 else 0.0,
        "oracle_scenario_gain": float(oracle_gain_sum / len(y_true)) if len(y_true) > 0 else 0.0,
        "gain_capture_ratio_vs_oracle": (
            float(selected_gain_sum / oracle_gain_sum) if oracle_gain_sum > 0 else float("nan")
        ),
        "selection_regret_vs_oracle": float(oracle_gain_sum - selected_gain_sum),
    }


def build_trace_rows(estimator_name: str, base_model: str, stage: str, seed: int,
                     strategy: str, target_ratio: float, decision,
                     y_true: np.ndarray,
                     source_target_ratio: float | None = None,
                     scenario: str | None = None,
                     scenario_type: str | None = None) -> list[dict]:
    if decision is None:
        return []
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    mask = np.asarray(decision.mask, dtype=bool).reshape(-1)
    if len(mask) == 0 or len(mask) != len(y_true):
        return []

    trace = dict(decision.trace or {})
    order = np.asarray(trace.get("order", np.arange(len(mask))), dtype=int).reshape(-1)
    ordered_mask = mask[order]
    ordered_true = y_true[order]
    steps = np.asarray(trace.get("step", np.arange(1, len(order) + 1)), dtype=int).reshape(-1)
    cumulative_offload = np.cumsum(ordered_mask.astype(np.int32))
    cumulative_gain = np.cumsum(ordered_true * ordered_mask.astype(float))
    cumulative_ratio = np.asarray(
        trace.get("cumulative_ratio", cumulative_offload / np.maximum(steps, 1)),
        dtype=float,
    ).reshape(-1)
    budget_debt = np.asarray(
        trace.get("budget_debt", cumulative_offload.astype(float) - float(target_ratio) * steps),
        dtype=float,
    ).reshape(-1)

    control_name = ""
    control_values = None
    for candidate in ("threshold_trace", "lambda_trace", "queue_trace", "threshold", "lambda", "queue"):
        if candidate in trace:
            control_name = candidate
            control_values = np.asarray(trace[candidate], dtype=float).reshape(-1)
            break

    rows: list[dict] = []
    for idx, step in enumerate(steps):
        prefix_values = ordered_true[: step]
        k = int(cumulative_offload[idx])
        oracle_prefix_gain = float(np.sum(np.sort(prefix_values)[-k:])) if k > 0 else 0.0
        rows.append({
            "estimator": estimator_name,
            "base_model": base_model,
            "stage": stage,
            "seed": int(seed),
            "scenario": scenario or "",
            "scenario_type": scenario_type or "",
            "strategy": strategy,
            "target_ratio": float(target_ratio),
            "source_target_ratio": (
                float(source_target_ratio)
                if source_target_ratio is not None else float(target_ratio)
            ),
            "step": int(step),
            "offload": int(ordered_mask[idx]),
            "cumulative_ratio": float(cumulative_ratio[idx]),
            "cumulative_gain": float(cumulative_gain[idx]),
            "cumulative_regret": float(oracle_prefix_gain - cumulative_gain[idx]),
            "budget_debt": float(budget_debt[idx]),
            "control_name": control_name,
            "control_value": float(control_values[idx]) if control_values is not None and idx < len(control_values) else float("nan"),
        })
    return rows


def compute_resource_tradeoff_row(estimator_name: str, base_model: str, stage: str,
                                  seed: int, strategy: str, ratio: float,
                                  actual_ratio: float, inference_time_ms: float,
                                  estimator_gflops: float,
                                  weak_detector_time_ms: float,
                                  cloud_detector_time_ms: float,
                                  weak_detector_gflops: float,
                                  cloud_detector_gflops: float,
                                  map_0_5: float,
                                  map_coco: float = float("nan"),
                                  scenario: str | None = None,
                                  scenario_type: str | None = None) -> dict:
    actual_ratio = float(actual_ratio)
    stage = str(stage or "other")
    if estimator_name == "weak_model":
        estimator_calls = 0.0
        weak_calls = 1.0
        cloud_calls = 0.0
    elif estimator_name == "strong_model":
        estimator_calls = 0.0
        weak_calls = 0.0
        cloud_calls = 1.0
    elif stage == "post":
        estimator_calls = 1.0
        weak_calls = 1.0
        cloud_calls = actual_ratio
    else:
        estimator_calls = 1.0
        weak_calls = max(0.0, 1.0 - actual_ratio)
        cloud_calls = actual_ratio

    est_ms = estimator_calls * float(inference_time_ms or 0.0)
    weak_ms = weak_calls * float(weak_detector_time_ms or 0.0)
    cloud_ms = cloud_calls * float(cloud_detector_time_ms or 0.0)
    total_ms = est_ms + weak_ms + cloud_ms

    est_gflops = estimator_calls * float(estimator_gflops or 0.0)
    weak_gflops = weak_calls * float(weak_detector_gflops or 0.0)
    cloud_gflops = cloud_calls * float(cloud_detector_gflops or 0.0)
    total_gflops = est_gflops + weak_gflops + cloud_gflops

    return {
        "estimator": estimator_name,
        "base_model": base_model,
        "stage": stage,
        "seed": int(seed),
        "scenario": scenario or "",
        "scenario_type": scenario_type or "",
        "strategy": strategy,
        "target_ratio": float(ratio),
        "actual_ratio": actual_ratio,
        "mAP": float(map_0_5),
        "mAP_coco": float(map_coco),
        "estimator_calls_per_frame": estimator_calls,
        "weak_calls_per_frame": weak_calls,
        "cloud_calls_per_frame": cloud_calls,
        "estimated_estimator_ms_per_frame": est_ms,
        "estimated_weak_ms_per_frame": weak_ms,
        "estimated_cloud_ms_per_frame": cloud_ms,
        "estimated_end_to_end_ms_per_frame": total_ms,
        "estimated_estimator_gflops_per_frame": est_gflops,
        "estimated_weak_gflops_per_frame": weak_gflops,
        "estimated_cloud_gflops_per_frame": cloud_gflops,
        "estimated_end_to_end_gflops_per_frame": total_gflops,
        "estimator_overhead_share": float(est_ms / total_ms) if total_ms > 0 else 0.0,
    }


def collect_qualitative_rows(estimator_name: str, base_model: str, seed: int,
                             y_true: np.ndarray, y_pred: np.ndarray, data: dict,
                             scenario: str | None = None,
                             scenario_type: str | None = None,
                             top_n: int = 5) -> list[dict]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return []
    errors = np.abs(y_pred - y_true)
    top_n = max(1, min(int(top_n), len(y_true)))
    rows: list[dict] = []
    names = np.asarray(data.get("video_name_test", [""] * len(y_true)), dtype=object).reshape(-1)
    frame_ids = np.asarray(data.get("frame_id_test", np.arange(len(y_true))), dtype=object).reshape(-1)
    paths = np.asarray(data.get("paths_test", [""] * len(y_true)), dtype=object).reshape(-1)
    selections = {
        "most_accurate": np.argsort(errors)[:top_n],
        "largest_error": np.argsort(errors)[-top_n:][::-1],
    }
    for label, indices in selections.items():
        for rank, idx in enumerate(indices, start=1):
            rows.append({
                "estimator": estimator_name,
                "base_model": base_model,
                "seed": int(seed),
                "scenario": scenario or "",
                "scenario_type": scenario_type or "",
                "example_type": label,
                "example_rank": rank,
                "sample_index": int(idx),
                "video_name": str(names[idx]) if idx < len(names) else "",
                "frame_id": str(frame_ids[idx]) if idx < len(frame_ids) else "",
                "path": str(paths[idx]) if idx < len(paths) else "",
                "y_true": float(y_true[idx]),
                "y_pred": float(y_pred[idx]),
                "abs_error": float(errors[idx]),
            })
    return rows


def summarize_offloading_curve(curve: dict[float, dict[str, float]],
                               weak_curve: Optional[dict[float, dict[str, float]]] = None,
                               oracle_curve: Optional[dict[float, dict[str, float]]] = None,
                               fixed_ratio_points: Optional[Iterable[float]] = None,
                               ratio_errors: Optional[Iterable[float]] = None) -> dict:
    if not curve:
        return {}
    trapz = getattr(np, "trapezoid", None) or np.trapz
    ratios = sorted(float(r) for r in curve.keys())
    row = {
        "peak_map": float(max(curve[r]["ap50"] for r in ratios)),
        "peak_map_coco": float(max(curve[r].get("ap_coco", np.nan) for r in ratios)),
        "peak_map_coco50": float(max(curve[r].get("ap50_allpoint", np.nan) for r in ratios)),
    }
    if len(ratios) >= 2:
        row["auc_0_5"] = float(trapz([curve[r]["ap50"] for r in ratios], ratios))
        if all("ap_coco" in curve[r] for r in ratios):
            row["auc_coco"] = float(trapz([curve[r]["ap_coco"] for r in ratios], ratios))
        if all("ap50_allpoint" in curve[r] for r in ratios):
            row["auc_coco50"] = float(trapz([curve[r]["ap50_allpoint"] for r in ratios], ratios))
    else:
        row["auc_0_5"] = float("nan")
        row["auc_coco"] = float("nan")
        row["auc_coco50"] = float("nan")

    if weak_curve:
        weak_peak = float(max(weak_curve[r]["ap50"] for r in weak_curve))
        weak_auc = float(trapz([weak_curve[r]["ap50"] for r in sorted(weak_curve)], sorted(weak_curve))) if len(weak_curve) >= 2 else weak_peak
        row["gain_over_weak_peak"] = row["peak_map"] - weak_peak
        row["gain_over_weak_auc_0_5"] = row["auc_0_5"] - weak_auc if np.isfinite(row["auc_0_5"]) else float("nan")
    if oracle_curve:
        oracle_peak = float(max(oracle_curve[r]["ap50"] for r in oracle_curve))
        oracle_auc = float(trapz([oracle_curve[r]["ap50"] for r in sorted(oracle_curve)], sorted(oracle_curve))) if len(oracle_curve) >= 2 else oracle_peak
        row["oracle_regret_peak"] = oracle_peak - row["peak_map"]
        row["oracle_regret_auc_0_5"] = oracle_auc - row["auc_0_5"] if np.isfinite(row["auc_0_5"]) else float("nan")

    for ratio in fixed_ratio_points or ():
        if ratio in curve:
            ref_ratio = float(ratio)
        else:
            ref_ratio = min(ratios, key=lambda x: abs(x - float(ratio)))
        label = f"{float(ratio):.2f}".replace(".", "p")
        row[f"map_at_{label}"] = float(curve[ref_ratio]["ap50"])
        if weak_curve and ref_ratio in weak_curve:
            row[f"gain_over_weak_at_{label}"] = float(curve[ref_ratio]["ap50"] - weak_curve[ref_ratio]["ap50"])
        if oracle_curve and ref_ratio in oracle_curve:
            row[f"oracle_regret_at_{label}"] = float(oracle_curve[ref_ratio]["ap50"] - curve[ref_ratio]["ap50"])

    ratio_errors = list(ratio_errors or [])
    if ratio_errors:
        arr = np.asarray(ratio_errors, dtype=float)
        row["mean_ratio_error"] = float(np.mean(arr))
        row["max_ratio_error"] = float(np.max(arr))
    return row


def aggregate_seeded_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "seed" not in df.columns or df["seed"].nunique(dropna=False) <= 1:
        return df.copy()

    keep_group_cols = [col for col in group_cols if col in df.columns]
    numeric_cols = [
        col for col in df.columns
        if col not in keep_group_cols + ["seed"] and is_numeric_dtype(df[col])
    ]
    passthrough_cols = [
        col for col in df.columns
        if col not in keep_group_cols + ["seed"] + numeric_cols
    ]

    rows: list[dict] = []
    grouped = df.groupby(keep_group_cols, dropna=False, sort=False)
    for keys, grp in grouped:
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(keep_group_cols, key_tuple))
        row["seed_count"] = int(grp["seed"].nunique(dropna=False))
        if "status" in grp.columns:
            statuses = grp["status"].fillna("").astype(str).str.upper()
            if (statuses == "PASS").all():
                row["status"] = "PASS"
            elif (statuses == "FAIL").all():
                row["status"] = "FAIL"
            else:
                row["status"] = "PARTIAL"
        for col in numeric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[col] = float(vals.mean()) if vals.notna().any() else float("nan")
        for col in passthrough_cols:
            if col == "status":
                continue
            series = grp[col]
            non_empty = [v for v in series.tolist() if pd.notna(v) and v != ""]
            row[col] = non_empty[0] if non_empty else series.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_statistical_summary(metrics_runs: pd.DataFrame,
                                offload_summary: pd.DataFrame,
                                bootstrap_samples: int = 1000) -> pd.DataFrame:
    metrics_runs = _ensure_policy_columns(metrics_runs)
    offload_summary = _ensure_policy_columns(offload_summary)
    if metrics_runs.empty:
        return pd.DataFrame()

    summary_rows: list[dict] = []
    metric_cols = [col for col in ("spearman_rho", "r2", "mae", "rmse", "peak_map", "peak_map_coco", "peak_map_coco50", "inference_time_ms") if col in metrics_runs.columns]
    offload_metric_cols = [col for col in ("auc_0_5", "auc_coco", "peak_map", "peak_map_coco", "gain_over_weak_peak", "oracle_regret_auc_0_5", "mean_ratio_error") if col in offload_summary.columns]
    proposed_base_models = _proposed_base_models(metrics_runs, offload_summary)
    proposed_names = set(metrics_runs.loc[metrics_runs["base_model"].isin(proposed_base_models), "estimator"].astype(str)) if "base_model" in metrics_runs.columns else set()
    baseline_df = offload_summary[~offload_summary["base_model"].isin(proposed_base_models)] if "base_model" in offload_summary.columns else pd.DataFrame()
    if "strategy" in baseline_df.columns:
        baseline_df = baseline_df[baseline_df["strategy"].astype(str).isin(("threshold", "oracle", "random", "constant", "fixed"))]
    strongest_baseline = None
    if not baseline_df.empty and "auc_0_5" in baseline_df.columns:
        baseline_means = baseline_df.groupby("estimator", as_index=False)["auc_0_5"].mean()
        if not baseline_means.empty:
            strongest_baseline = str(baseline_means.sort_values("auc_0_5", ascending=False).iloc[0]["estimator"])

    group_cols = [
        col for col in (
            "estimator", "base_model", "stage",
            "offloader_id", "policy_id", "strategy",
            "scenario", "scenario_type",
        ) if col in metrics_runs.columns or col in offload_summary.columns
    ]
    metrics_group_cols = [col for col in group_cols if col in metrics_runs.columns]
    for keys, group in metrics_runs.groupby(metrics_group_cols, dropna=False):
        row = dict(zip(metrics_group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for col in metric_cols:
            vals = group[col].dropna().astype(float).to_numpy()
            if len(vals) == 0:
                continue
            ci_low, ci_high = bootstrap_ci(vals, samples=bootstrap_samples)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals))
            row[f"{col}_ci_low"] = ci_low
            row[f"{col}_ci_high"] = ci_high

        if not offload_summary.empty:
            off_group = offload_summary.copy()
            if "strategy" in off_group.columns and "strategy" not in row:
                off_group = off_group[off_group["strategy"].astype(str).isin(("threshold", "oracle", "random", "constant", "fixed"))]
            for col in ("estimator", "strategy", "scenario", "scenario_type"):
                if col in row and col in off_group.columns:
                    off_group = off_group[off_group[col].fillna("") == str(row.get(col, "") or "")]
            for col in offload_metric_cols:
                if col not in off_group.columns:
                    continue
                vals = off_group[col].dropna().astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                ci_low, ci_high = bootstrap_ci(vals, samples=bootstrap_samples)
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals))
                row[f"{col}_ci_low"] = ci_low
                row[f"{col}_ci_high"] = ci_high

        if strongest_baseline and row.get("estimator") in proposed_names and not offload_summary.empty:
            from scipy.stats import wilcoxon

            target_df = offload_summary[offload_summary["estimator"].astype(str) == str(row["estimator"])]
            base_df = offload_summary[offload_summary["estimator"].astype(str) == strongest_baseline]
            if "strategy" in offload_summary.columns:
                strategy = str(row.get("strategy", "threshold") or "threshold")
                target_df = target_df[target_df["strategy"].astype(str) == strategy]
                base_df = base_df[base_df["strategy"].astype(str) == strategy]
            if "scenario" in offload_summary.columns and row.get("scenario"):
                target_df = target_df[target_df["scenario"].fillna("") == str(row["scenario"])]
                if base_df["scenario"].fillna("").astype(str).ne("").any():
                    base_df = base_df[base_df["scenario"].fillna("") == str(row["scenario"])]
            if "scenario_type" in offload_summary.columns and row.get("scenario_type"):
                target_df = target_df[target_df["scenario_type"].fillna("") == str(row["scenario_type"])]
                if base_df["scenario_type"].fillna("").astype(str).ne("").any():
                    base_df = base_df[base_df["scenario_type"].fillna("") == str(row["scenario_type"])]
            merge_cols = ["seed"]
            for col in ("strategy", "scenario", "scenario_type"):
                if col not in target_df.columns or col not in base_df.columns or col not in row:
                    continue
                target_has_values = target_df[col].fillna("").astype(str).ne("").any()
                base_has_values = base_df[col].fillna("").astype(str).ne("").any()
                if col == "strategy" or (target_has_values and base_has_values):
                    merge_cols.append(col)
            needed = [col for col in ("seed", "auc_0_5", "peak_map", "strategy", "scenario", "scenario_type") if col in target_df.columns]
            merged = target_df[needed].merge(
                base_df[needed], on=merge_cols, suffixes=("", "_baseline")
            )
            if not merged.empty:
                delta_auc = (merged["auc_0_5"] - merged["auc_0_5_baseline"]).to_numpy(dtype=float)
                delta_peak = (merged["peak_map"] - merged["peak_map_baseline"]).to_numpy(dtype=float)
                row["strongest_baseline"] = strongest_baseline
                row["delta_auc_0_5_vs_baseline_mean"] = float(np.mean(delta_auc))
                row["delta_peak_map_vs_baseline_mean"] = float(np.mean(delta_peak))
                row["delta_auc_0_5_vs_baseline_ci_low"], row["delta_auc_0_5_vs_baseline_ci_high"] = bootstrap_ci(delta_auc, samples=bootstrap_samples)
                row["delta_peak_map_vs_baseline_ci_low"], row["delta_peak_map_vs_baseline_ci_high"] = bootstrap_ci(delta_peak, samples=bootstrap_samples)
                if len(delta_auc) > 1 and not np.allclose(delta_auc, 0.0):
                    try:
                        row["delta_auc_0_5_vs_baseline_p"] = float(wilcoxon(delta_auc).pvalue)
                    except ValueError:
                        pass
                if len(delta_peak) > 1 and not np.allclose(delta_peak, 0.0):
                    try:
                        row["delta_peak_map_vs_baseline_p"] = float(wilcoxon(delta_peak).pvalue)
                    except ValueError:
                        pass
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
