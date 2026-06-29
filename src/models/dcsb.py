"""DCSB estimators.

This module keeps the current learned surrogate used in the repo and also
provides a paper-faithful original DCSB baseline.  The learned variant uses a
channel-attention MLP over tabular features, while the original-paper variant
implements the threshold rule over weak-detector object count and minimum
object area.

Reference: https://ieeexplore.ieee.org/abstract/document/10705683

Architecture:
  1. Attention gate: Linear(D→10) → ReLU → Linear(10→D) → Sigmoid
     Element-wise multiply with input (feature re-weighting).
  2. MLP: D→300→150→50→10→1 with BatchNorm + ReLU, final Sigmoid.

Trained with BCE loss on binarised AP-gain labels (gain > 0 → offload).
At inference the sigmoid probability is returned as a continuous score
for ranking-based offloading decisions.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import BaseEstimator


class DCSBDiscriminator:
    """Thin wrapper that builds / returns the PyTorch nn.Module."""

    @staticmethod
    def build(input_dim: int):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, dim):
                super().__init__()
                # Attention gate (feature re-weighting)
                self.attention = nn.Sequential(
                    nn.Linear(dim, 10, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(10, dim, bias=False),
                    nn.Sigmoid(),
                )
                # MLP classifier
                self.hidden1 = nn.Linear(dim, 300)
                self.bn1 = nn.BatchNorm1d(300)
                self.hidden2 = nn.Linear(300, 150)
                self.bn2 = nn.BatchNorm1d(150)
                self.hidden3 = nn.Linear(150, 50)
                self.bn3 = nn.BatchNorm1d(50)
                self.hidden4 = nn.Linear(50, 10)
                self.bn4 = nn.BatchNorm1d(10)
                self.output = nn.Linear(10, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = x.to(torch.float32)
                # Attention re-weighting
                w = self.attention(x).view_as(x)
                x = x * w
                # MLP
                x = self.relu(self.bn1(self.hidden1(x)))
                x = self.relu(self.bn2(self.hidden2(x)))
                x = self.relu(self.bn3(self.hidden3(x)))
                x = self.relu(self.bn4(self.hidden4(x)))
                x = self.sigmoid(self.output(x))
                return x.squeeze(-1)

        return _Net(input_dim)

    @staticmethod
    def init_weights(model):
        """Apply the weight initialisation scheme from the DCSB paper."""
        import torch.nn.init as init

        for name, m in model.named_modules():
            if name.startswith("attention"):
                if hasattr(m, "weight"):
                    init.normal_(m.weight, mean=0, std=0.2)
            elif hasattr(m, "weight") and isinstance(m, type(model.hidden1)):
                init.normal_(m.weight, mean=0, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for score shaping."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _threshold_candidates(values: np.ndarray,
                          max_candidates: int = 128) -> np.ndarray:
    """Build candidate thresholds from a one-dimensional value array.

    For small discrete supports we use all midpoints between unique values.
    For large continuous supports we use quantile-based candidates to keep the
    grid search tractable.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0], dtype=np.float64)

    unique = np.unique(values)
    if unique.size == 1:
        base = float(unique[0])
        pad = max(abs(base) * 0.1, 1.0)
        return np.array([base - pad, base, base + pad], dtype=np.float64)

    if unique.size <= max_candidates:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        return np.unique(np.concatenate((
            [unique[0] - 1.0],
            midpoints,
            [unique[-1] + 1.0],
        ))).astype(np.float64)

    quantiles = np.linspace(0.0, 1.0, max_candidates)
    return np.unique(np.quantile(values, quantiles)).astype(np.float64)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute compact binary-classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    acc = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _rule_search_key(
    metrics: Dict[str, float],
    *,
    offload_rate: float,
    y_pos_rate: float,
    count_threshold: float,
    area_threshold: float,
    confidence_threshold: Optional[float] = None,
    base_threshold: Optional[float] = None,
) -> tuple[float, ...]:
    """Lexicographic objective for DCSB threshold search.

    F1 is the primary signal. Accuracy only breaks ties so the search does not
    collapse to trivial all-negative rules on mildly imbalanced splits.
    """
    key = [
        float(metrics["f1"]),
        float(metrics["accuracy"]),
        -abs(float(offload_rate) - float(y_pos_rate)),
    ]
    if confidence_threshold is not None and base_threshold is not None:
        key.append(-abs(float(confidence_threshold) - float(base_threshold)))
    key.extend([
        -abs(float(count_threshold)),
        -abs(float(area_threshold)),
    ])
    return tuple(key)


def _count_fit_metrics(
    confs: np.ndarray,
    gt_counts: np.ndarray,
    confidence_threshold: float,
) -> Dict[str, float]:
    """Evaluate how well a confidence threshold recovers GT object counts."""
    confs = np.asarray(confs, dtype=np.float64)
    gt_counts = np.asarray(gt_counts, dtype=np.float64).reshape(-1)
    est_counts = np.sum(confs >= float(confidence_threshold), axis=1).astype(np.float64)
    return {
        "count_mae": float(np.mean(np.abs(est_counts - gt_counts))),
        "count_mse": float(np.mean((est_counts - gt_counts) ** 2)),
    }


def _paper_rule_score(counts: np.ndarray, areas: np.ndarray,
                      count_threshold: float, area_threshold: float,
                      count_scale: float, area_scale: float) -> np.ndarray:
    """Continuous score for the paper rule.

    The decision boundary is still the hard paper rule:
    count > count_threshold OR min_area < area_threshold.
    The returned score is only used for ranking / probability output and
    is monotonic with respect to that binary decision.
    """
    counts = np.asarray(counts, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    count_scale = max(float(count_scale), 1e-6)
    area_scale = max(float(area_scale), 1e-6)
    count_margin = (counts - float(count_threshold)) / count_scale
    area_margin = (float(area_threshold) - areas) / area_scale
    margin = np.maximum(count_margin, area_margin)
    return _sigmoid(margin)


def _proposal_summary(confs: np.ndarray, areas: np.ndarray,
                      confidence_threshold: float,
                      base_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize proposal tensors into rule inputs for a confidence threshold."""
    confs = np.asarray(confs, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    base_counts = np.sum(confs >= base_threshold, axis=1)
    est_mask = confs >= confidence_threshold
    est_counts = np.sum(est_mask, axis=1)
    est_min_area = np.where(
        np.any(est_mask, axis=1),
        np.where(est_mask, areas, np.inf).min(axis=1),
        1.0,
    )
    return base_counts.astype(np.float64), est_counts.astype(np.float64), est_min_area.astype(np.float64)


def _extract_dcsb_targets(y) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract the primary difficult/easy labels and optional GT object counts."""
    if isinstance(y, dict):
        primary = y.get("primary")
        if primary is None:
            raise KeyError("DCSB target bundle must include 'primary'.")
        gt_counts = y.get("gt_count")
        primary_arr = np.asarray(primary, dtype=np.float64).reshape(-1)
        gt_arr = None if gt_counts is None else np.asarray(
            gt_counts, dtype=np.float64
        ).reshape(-1)
        return primary_arr, gt_arr
    return np.asarray(y, dtype=np.float64).reshape(-1), None


def _confidence_threshold_candidates(
    confs: np.ndarray,
    *,
    base_threshold: float = 0.5,
    max_candidates: int = 128,
) -> np.ndarray:
    """Candidate confidence thresholds for the DCSB paper rule.

    The paper lowers the weak-detector confidence threshold below 0.5 to
    estimate uncertain objects, so candidates strictly above the base threshold
    are not valid.
    """
    values = np.asarray(confs, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values >= 0.0) & (values < float(base_threshold))]
    if values.size == 0:
        raise ValueError(
            "DCSB original baseline requires cached weak-detector proposals "
            f"below the base confidence threshold {base_threshold:.2f}. "
            "Re-run detection with a low `dataset.detection_conf` (the repo "
            "default is now 0.05) and then re-run prepare."
        )
    unique = np.unique(values)
    if unique.size > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        unique = np.unique(np.quantile(unique, quantiles))
    eps = max(abs(float(base_threshold)) * 1e-6, 1e-6)
    return np.unique(np.clip(unique, 0.0, float(base_threshold) - eps))


def _fit_confidence_threshold(
    confs: np.ndarray,
    gt_counts: np.ndarray,
    *,
    base_threshold: float = 0.5,
) -> Dict[str, float]:
    """Calibrate the DCSB confidence threshold against GT object counts."""
    confs = np.asarray(confs, dtype=np.float64)
    gt_counts = np.asarray(gt_counts, dtype=np.float64).reshape(-1)
    candidates = _confidence_threshold_candidates(
        confs, base_threshold=base_threshold
    )

    best = None
    best_key = (float("inf"), float("inf"), float("inf"))
    for conf_threshold in candidates:
        est_counts = np.sum(confs >= conf_threshold, axis=1).astype(np.float64)
        mae = float(np.mean(np.abs(est_counts - gt_counts)))
        mse = float(np.mean((est_counts - gt_counts) ** 2))
        key = (mae, mse, abs(float(conf_threshold) - 0.25))
        if key < best_key:
            best_key = key
            best = {
                "confidence_threshold": float(conf_threshold),
                "count_mae": mae,
                "count_mse": mse,
            }

    assert best is not None
    return best


def _fit_rule_thresholds(
    confs: np.ndarray,
    areas: np.ndarray,
    labels: np.ndarray,
    *,
    confidence_threshold: float,
    base_threshold: float = 0.5,
) -> Dict[str, float]:
    """Fit the DCSB object-count and min-area rule thresholds."""
    confs = np.asarray(confs, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    base_counts, est_counts, min_areas = _proposal_summary(
        confs, areas, confidence_threshold, base_threshold=base_threshold
    )
    count_candidates = _threshold_candidates(est_counts, max_candidates=64)
    area_candidates = _threshold_candidates(min_areas, max_candidates=64)

    best = None
    best_key = (-1.0, -1.0, -1.0, 0.0, 0.0)
    y_pos_rate = float(labels.mean()) if len(labels) else 0.0

    for area_threshold in area_candidates:
        for count_threshold in count_candidates:
            pred = np.where(
                base_counts == est_counts,
                0,
                np.logical_or(est_counts > count_threshold, min_areas < area_threshold),
            )
            metrics = _binary_metrics(labels, pred.astype(np.int32))
            offload_rate = float(np.mean(pred))
            key = _rule_search_key(
                metrics,
                offload_rate=offload_rate,
                y_pos_rate=y_pos_rate,
                count_threshold=float(count_threshold),
                area_threshold=float(area_threshold),
            )
            if key > best_key:
                best_key = key
                best = {
                    "count_threshold": float(count_threshold),
                    "area_threshold": float(area_threshold),
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "offload_rate": offload_rate,
                }

    assert best is not None
    return best


def _fit_paper_thresholds(confs: np.ndarray,
                          areas: np.ndarray,
                          labels: np.ndarray,
                          base_threshold: float = 0.5) -> Dict[str, float]:
    """Fallback joint search when GT-count calibration metadata is unavailable."""
    confs = np.asarray(confs, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    conf_candidates = _confidence_threshold_candidates(
        confs, base_threshold=base_threshold
    )

    best = None
    best_key = (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0)
    y_pos_rate = float(labels.mean()) if len(labels) else 0.0

    for conf_threshold in conf_candidates:
        base_counts, est_counts, min_areas = _proposal_summary(
            confs, areas, conf_threshold, base_threshold=base_threshold
        )
        count_candidates = _threshold_candidates(est_counts, max_candidates=64)
        area_candidates = _threshold_candidates(min_areas, max_candidates=64)
        for area_threshold in area_candidates:
            for count_threshold in count_candidates:
                pred = np.where(
                    base_counts == est_counts,
                    0,
                    np.logical_or(est_counts > count_threshold, min_areas < area_threshold),
                )
                metrics = _binary_metrics(labels, pred.astype(np.int32))
                offload_rate = float(np.mean(pred))
                key = _rule_search_key(
                    metrics,
                    offload_rate=offload_rate,
                    y_pos_rate=y_pos_rate,
                    confidence_threshold=float(conf_threshold),
                    base_threshold=base_threshold,
                    count_threshold=float(count_threshold),
                    area_threshold=float(area_threshold),
                )
                if key > best_key:
                    best_key = key
                    best = {
                        "confidence_threshold": float(conf_threshold),
                        "count_threshold": float(count_threshold),
                        "area_threshold": float(area_threshold),
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "offload_rate": offload_rate,
                    }

    assert best is not None
    return best


def _fit_gt_calibrated_rule(
    confs: np.ndarray,
    areas: np.ndarray,
    labels: np.ndarray,
    gt_counts: np.ndarray,
    *,
    base_threshold: float = 0.5,
    min_train_f1: float = 0.05,
) -> Dict[str, float]:
    """Fit the paper rule with a fallback when count calibration collapses."""
    conf_fit = _fit_confidence_threshold(
        confs,
        gt_counts,
        base_threshold=base_threshold,
    )
    calibrated = _fit_rule_thresholds(
        confs,
        areas,
        labels,
        confidence_threshold=conf_fit["confidence_threshold"],
        base_threshold=base_threshold,
    )
    calibrated["confidence_threshold"] = conf_fit["confidence_threshold"]
    calibrated.update(
        _count_fit_metrics(confs, gt_counts, calibrated["confidence_threshold"])
    )
    calibrated["search_mode"] = "count_calibrated"

    if calibrated["f1"] > float(min_train_f1):
        return calibrated

    joint = _fit_paper_thresholds(
        confs,
        areas,
        labels,
        base_threshold=base_threshold,
    )
    joint.update(_count_fit_metrics(confs, gt_counts, joint["confidence_threshold"]))

    joint_is_better = (
        joint["f1"] > calibrated["f1"] + 1e-12
        or (
            abs(joint["f1"] - calibrated["f1"]) <= 1e-12
            and joint["accuracy"] > calibrated["accuracy"] + 1e-12
        )
    )
    if not joint_is_better:
        return calibrated

    joint["search_mode"] = "joint_fallback"
    joint["search_note"] = (
        "count-calibrated rule collapsed to near-zero train F1; "
        "using joint threshold search"
    )
    return joint


class DCSBOriginalEstimator(BaseEstimator):
    """Original-paper DCSB baseline.

    This reproduces the paper's threshold-rule style baseline rather than the
    learned surrogate above. It expects the full weak-detector proposal stream
    encoded as a padded proposal tensor so the rule can inspect all candidate
    confidences and area ratios below the normal 0.5 cutoff.
    """

    name = "dcsb_original"
    task_type = "regression"
    stage = "post"
    checkpoint_ext = ".pt"
    input_key = "proposal_full"
    def __init__(self, gain_threshold: float = 0.0,
                 proposal_stride: int = 6,
                 confidence_offset: int = 0,
                 area_offset: int = 5,
                 base_confidence_threshold: float = 0.5,
                 val_fraction: float = 0.1,
                 device: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.gain_threshold = gain_threshold
        self.proposal_stride = proposal_stride
        self.confidence_offset = confidence_offset
        self.area_offset = area_offset
        self.base_confidence_threshold = base_confidence_threshold
        self.val_fraction = val_fraction
        self.device_name = device
        self.fit_metrics: Dict[str, float] = {}

    def _extract_rule_inputs(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {X.shape}")
        if self.proposal_stride <= 0:
            raise ValueError("proposal_stride must be positive.")
        if X.shape[1] % self.proposal_stride != 0:
            raise ValueError(
                f"Expected proposal features with stride {self.proposal_stride}, "
                f"got shape {X.shape}"
            )
        max_offset = max(self.confidence_offset, self.area_offset)
        if X.shape[1] <= max_offset:
            raise ValueError(
                "DCSB original baseline requires proposal confidence/area "
                f"offsets {self.confidence_offset} and {self.area_offset}, "
                f"but input has shape {X.shape} with stride {self.proposal_stride}"
            )
        confs = X[:, self.confidence_offset::self.proposal_stride]
        areas = X[:, self.area_offset::self.proposal_stride]
        return confs, areas

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        X = np.asarray(X, dtype=np.float64)
        y_primary, gt_counts = _extract_dcsb_targets(y)
        confs, areas = self._extract_rule_inputs(X)
        if len(y_primary) != len(X):
            raise ValueError(
                f"DCSB targets must have the same number of rows as X: "
                f"{len(y_primary)} != {len(X)}"
            )
        y_bin = (y_primary >= 1.0).astype(np.int32)
        if gt_counts is not None and len(gt_counts) != len(X):
            raise ValueError(
                f"DCSB gt_count metadata must have the same number of rows as X: "
                f"{len(gt_counts)} != {len(X)}"
            )

        n = len(X)
        val_size = max(1, int(self.val_fraction * n))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:val_size], perm[val_size:]
        if train_idx.size == 0:
            train_idx = val_idx

        train_confs = confs[train_idx]
        train_areas = areas[train_idx]
        train_labels = y_bin[train_idx]
        val_confs = confs[val_idx]
        val_areas = areas[val_idx]
        val_labels = y_bin[val_idx]
        train_gt_counts = None if gt_counts is None else gt_counts[train_idx]
        val_gt_counts = None if gt_counts is None else gt_counts[val_idx]

        if train_gt_counts is not None:
            best = _fit_gt_calibrated_rule(
                train_confs,
                train_areas,
                train_labels,
                train_gt_counts,
                base_threshold=self.base_confidence_threshold,
            )
        else:
            best = _fit_paper_thresholds(
                train_confs,
                train_areas,
                train_labels,
                base_threshold=self.base_confidence_threshold,
            )
            best["search_mode"] = "joint"
        _train_base_counts, train_counts, train_min_areas = _proposal_summary(
            train_confs,
            train_areas,
            best["confidence_threshold"],
            base_threshold=self.base_confidence_threshold,
        )
        _val_base_counts, val_counts, val_min_areas = _proposal_summary(
            val_confs,
            val_areas,
            best["confidence_threshold"],
            base_threshold=self.base_confidence_threshold,
        )

        count_scale = float(max(np.std(train_counts), 1.0))
        area_scale = float(max(np.std(train_min_areas), 1e-6))
        self.model = {
            "variant": "paper_rule",
            "search_mode": best.get("search_mode", "joint"),
            "proposal_stride": int(self.proposal_stride),
            "confidence_offset": int(self.confidence_offset),
            "area_offset": int(self.area_offset),
            "base_confidence_threshold": float(self.base_confidence_threshold),
            "confidence_threshold": best["confidence_threshold"],
            "count_threshold": best["count_threshold"],
            "area_threshold": best["area_threshold"],
            "count_scale": count_scale,
            "area_scale": area_scale,
            "gain_threshold": float(self.gain_threshold),
        }
        self.is_fitted = True

        train_scores = self.predict(X[train_idx])
        val_scores = self.predict(X[val_idx])
        train_pred = (train_scores >= 0.5).astype(np.int32)
        val_pred = (val_scores >= 0.5).astype(np.int32)

        train_metrics = _binary_metrics(train_labels, train_pred)
        val_metrics = _binary_metrics(val_labels, val_pred)
        self.fit_metrics = {
            "search_mode": str(best.get("search_mode", "joint")),
            "confidence_threshold": round(best["confidence_threshold"], 6),
            "count_threshold": round(best["count_threshold"], 6),
            "area_threshold": round(best["area_threshold"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_precision": round(train_metrics["precision"], 6),
            "train_recall": round(train_metrics["recall"], 6),
            "train_f1": round(train_metrics["f1"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "train_offload_rate": round(float(np.mean(train_pred)), 6),
            "val_offload_rate": round(float(np.mean(val_pred)), 6),
        }
        if "search_note" in best:
            self.fit_metrics["search_note"] = str(best["search_note"])
        if train_gt_counts is not None:
            train_est_counts = np.sum(
                train_confs >= best["confidence_threshold"], axis=1
            ).astype(np.float64)
            val_est_counts = np.sum(
                val_confs >= best["confidence_threshold"], axis=1
            ).astype(np.float64)
            self.fit_metrics.update({
                "train_count_mae": round(
                    float(np.mean(np.abs(train_est_counts - train_gt_counts))), 6
                ),
                "val_count_mae": round(
                    float(np.mean(np.abs(val_est_counts - val_gt_counts))), 6
                ),
                "train_count_mse": round(
                    float(np.mean((train_est_counts - train_gt_counts) ** 2)), 6
                ),
                "val_count_mse": round(
                    float(np.mean((val_est_counts - val_gt_counts) ** 2)), 6
                ),
            })

        print(
            "    DCSB-original "
            f"mode={best.get('search_mode', 'joint')}, "
            f"conf_thr={best['confidence_threshold']:.4f}, "
            f"count_thr={best['count_threshold']:.4f}, "
            f"area_thr={best['area_threshold']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise ValueError("DCSBOriginalEstimator is not fitted")
        confs, areas = self._extract_rule_inputs(X)
        base_counts, counts, min_areas = _proposal_summary(
            confs,
            areas,
            self.model["confidence_threshold"],
            base_threshold=self.model["base_confidence_threshold"],
        )
        scores = _paper_rule_score(
            counts,
            min_areas,
            self.model["count_threshold"],
            self.model["area_threshold"],
            self.model["count_scale"],
            self.model["area_scale"],
        )
        scores = np.where(base_counts == counts, 0.0, scores)
        return scores.astype(np.float32)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        scores = self.predict(X)
        return np.stack([1.0 - scores, scores], axis=1)

    def get_info(self) -> Dict[str, Any]:
        desc = "DCSB original-paper rule (count + min area threshold)"
        if self.model is not None:
            desc = (
                "DCSB original-paper rule "
                f"(conf>{self.model['confidence_threshold']:.3f}, "
                f"count>{self.model['count_threshold']:.3f} or "
                f"area<{self.model['area_threshold']:.3f})"
            )
        return {
            "description": desc,
            "gflops": 0.0,
            "params": 0.0,
        }


class DCSBEstimator(BaseEstimator):
    """DCSB discriminator reproduced as a post-inference estimator.

    Binary classifier trained with BCE on binarised AP-gain labels.
    Returns sigmoid probability as a continuous offloading score.
    """

    name = "dcsb"
    task_type = "regression"   # continuous output used for ranking
    stage = "post"
    checkpoint_ext = ".pt"
    input_key = "proposal"

    def __init__(self, epochs: int = 30, lr: float = 0.009,
                 batch_size: int = 128, gain_threshold: float = 0.0,
                 patience: int = 10, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.gain_threshold = gain_threshold
        self.patience = patience
        self.device_name = device
        self._input_dim: Optional[int] = None
        self._scaler: Any = None  # StandardScaler for input normalisation

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        import torch
        import torch.nn as nn
        import torch.utils.data as Data
        from sklearn.preprocessing import StandardScaler
        from tqdm import trange

        from ..losses import bce_loss, extract_loss_params, get_loss

        epochs = kwargs.get("epochs", self.epochs)
        lr = kwargs.get("lr", self.lr)
        batch_size = kwargs.get("batch_size", self.batch_size)
        patience = kwargs.get("patience", self.patience)

        self._input_dim = X.shape[1]
        if getattr(self, 'device_name', None) and self.device_name != "auto":
            device = torch.device(self.device_name)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalise input features
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X).astype(np.float32)

        # Build model
        assert self._input_dim is not None
        net = DCSBDiscriminator.build(self._input_dim).to(device)
        DCSBDiscriminator.init_weights(net)

        # Binarise targets: gain > threshold → should offload (1)
        y_bin = (y > self.gain_threshold).astype(np.float32)

        # Train/val split
        n = len(X_scaled)
        val_size = max(1, int(0.1 * n))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        val_idx, train_idx = perm[:val_size], perm[val_size:]

        X_train_t = torch.FloatTensor(X_scaled[train_idx]).to(device)
        y_train_t = torch.FloatTensor(y_bin[train_idx]).to(device)
        X_val_t = torch.FloatTensor(X_scaled[val_idx]).to(device)
        y_val_t = torch.FloatTensor(y_bin[val_idx]).to(device)

        train_ds = Data.TensorDataset(X_train_t, y_train_t)
        loader = Data.DataLoader(train_ds, batch_size=batch_size,
                                 shuffle=True, num_workers=0)

        # Resolve loss: configurable via kwargs["loss"], default bce
        loss_name = kwargs.get("loss", None)
        loss_params = extract_loss_params(kwargs)
        criterion = get_loss(loss_name, **loss_params) or bce_loss
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        with trange(epochs, desc="  DCSB training", mininterval=2.0) as pbar:
            for epoch in pbar:
                net.train()
                epoch_loss = 0.0
                for batch_x, batch_y in loader:
                    out = net(batch_x)
                    loss = criterion(out, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Validation
                net.eval()
                with torch.no_grad():
                    val_out = net(X_val_t)
                    val_loss = criterion(val_out, y_val_t).item()
                    val_pred = (val_out >= 0.5).float()
                    acc = (val_pred == y_val_t).float().mean().item()

                pbar.set_postfix(loss=epoch_loss / max(len(loader), 1),
                                 val_bce=f"{val_loss:.4f}",
                                 val_acc=f"{acc:.3f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        stopped_epoch = epoch + 1

        # Restore best
        if best_state is not None:
            net.load_state_dict(best_state)

        self.model = net.cpu()
        self.is_fitted = True

        # Compute fit_metrics for pipeline logging
        self.model.eval()
        all_x = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            all_out = self.model(all_x)
        train_bce = criterion(all_out[train_idx], torch.FloatTensor(y_bin[train_idx])).item()
        val_bce = criterion(all_out[val_idx], torch.FloatTensor(y_bin[val_idx])).item()
        train_acc = ((all_out[train_idx] >= 0.5).float()
                     == torch.FloatTensor(y_bin[train_idx])).float().mean().item()
        val_acc = ((all_out[val_idx] >= 0.5).float()
                   == torch.FloatTensor(y_bin[val_idx])).float().mean().item()
        self.fit_metrics = {
            'train_bce': round(train_bce, 6),
            'val_bce': round(val_bce, 6),
            'train_acc': round(train_acc, 6),
            'val_acc': round(val_acc, 6),
            'epochs_run': stopped_epoch,
            'epochs_max': epochs,
        }
        from .. import log
        log.info(f"DCSB  Acc={train_acc:.4f}/{val_acc:.4f}  "
                 f"BCE={train_bce:.4f}/{val_bce:.4f}  "
                 f"[{stopped_epoch}/{epochs} epochs]", indent=8)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        import torch

        X_scaled = self._scaler.transform(np.asarray(X)).astype(np.float32)
        self.model.eval()
        device = next(self.model.parameters()).device
        t = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            out = self.model(t)
        return out.cpu().numpy()

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        scores = self.predict(X)
        return np.stack([1 - scores, scores], axis=1)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> Dict[str, Any]:
        params = 0
        if self.model is not None:
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
        return {
            "description": "DCSB Discriminator (Attention + MLP, BCE)",
            "gflops": 0.0,  # negligible for tabular MLP
            "params": round(params, 4),
        }

    # ------------------------------------------------------------------
    # Persistence (torch-based)
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self._input_dim,
            "is_fitted": self.is_fitted,
            "config": self.config,
            "gain_threshold": self.gain_threshold,
            "scaler": self._scaler,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = None) -> "DCSBEstimator":
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        if device is not None:
            config["device"] = device
        estimator = cls(**config)
        estimator._input_dim = ckpt["input_dim"]
        estimator.gain_threshold = ckpt.get("gain_threshold", 0.0)
        estimator._scaler = ckpt.get("scaler")
        input_dim: int = estimator._input_dim  # type: ignore[assignment]
        estimator.model = DCSBDiscriminator.build(input_dim)
        estimator.model.load_state_dict(ckpt["model_state_dict"])
        if device is not None and device != "auto":
            estimator.model = estimator.model.to(device)
        estimator.model.eval()
        estimator.is_fitted = ckpt.get("is_fitted", True)
        return estimator
