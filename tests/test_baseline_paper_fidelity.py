import numpy as np
import pytest

from config import PipelineConfig
from config.approaches import default_approaches
from src.models.dcsb import DCSBOriginalEstimator
from src.models.edgeml import EdgeMLPaperEstimator, _split_indices
from src.phases.prepare import _required_proxy_families


def _build_dcsb_matrix(samples, proposal_stride: int = 6) -> np.ndarray:
    max_props = max((len(sample) for sample in samples), default=1)
    X = np.zeros((len(samples), max_props * proposal_stride), dtype=np.float32)
    for row_idx, sample in enumerate(samples):
        for prop_idx, (conf, area_ratio) in enumerate(sample):
            X[row_idx, prop_idx * proposal_stride + 0] = conf
            X[row_idx, prop_idx * proposal_stride + 5] = area_ratio
    return X


def test_dcsb_original_requires_low_confidence_proposals():
    X = _build_dcsb_matrix([
        [(0.93, 0.40)],
        [(0.88, 0.22)],
        [(0.91, 0.15)],
    ])
    y = {
        "primary": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "gt_count": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }

    estimator = DCSBOriginalEstimator(val_fraction=0.34)

    with pytest.raises(ValueError, match="below the base confidence threshold"):
        estimator.fit(X, y)


def test_dcsb_original_falls_back_when_count_calibration_collapses():
    samples = []
    labels = []
    gt_counts = []

    for _ in range(9):
        samples.append([(0.60, 0.50), (0.20, 0.01)])
        labels.append(1.0)
        gt_counts.append(1.0)

    for _ in range(10):
        samples.append([(0.60, 0.50)])
        labels.append(0.0)
        gt_counts.append(1.0)

    # This single near-base proposal makes count calibration prefer ~0.5, which
    # would otherwise collapse the classifier to all-negative predictions.
    samples.append([(0.60, 0.50), (0.4999, 0.50)])
    labels.append(0.0)
    gt_counts.append(2.0)

    X = _build_dcsb_matrix(samples)
    y = {
        "primary": np.asarray(labels, dtype=np.float32),
        "gt_count": np.asarray(gt_counts, dtype=np.float32),
    }

    estimator = DCSBOriginalEstimator(val_fraction=0.2)
    estimator.fit(X, y)

    preds = estimator.predict(X)
    pred_labels = (preds >= 0.5).astype(np.int32)

    assert estimator.fit_metrics["search_mode"] == "joint_fallback"
    assert estimator.model["confidence_threshold"] < 0.4999
    assert 0 < int(pred_labels.sum()) < len(pred_labels)
    assert estimator.fit_metrics["train_f1"] > 0.5


def test_edgeml_paper_scaler_uses_training_split_only():
    X = np.zeros((10, 2), dtype=np.float32)
    train_idx, val_idx = _split_indices(len(X), val_fraction=0.1, seed=42)
    X[val_idx] = 1000.0
    y = np.linspace(0.0, 1.0, len(X), dtype=np.float32)

    estimator = EdgeMLPaperEstimator(
        epochs=1,
        batch_size=3,
        hidden=[4, 1],
        grid_search=False,
        cv_folds=2,
        patience=1,
        device="cpu",
    )
    estimator.fit(X, y)

    assert train_idx.size > 0
    assert val_idx.size > 0
    assert np.allclose(X[train_idx], 0.0)
    assert np.allclose(estimator._scaler.mean_, 0.0)


def test_default_edgeml_baseline_uses_moric_target():
    approaches = default_approaches()
    edgeml_cfg = next(cfg for cfg in approaches if cfg.base_model == "edgeml")
    assert edgeml_cfg.proxy_metric == "moric_allpoint"

    cfg = PipelineConfig(approaches=approaches)
    families = _required_proxy_families(cfg)
    assert "dataset_oric" in families
    assert cfg.oric_context_size == 1000
    assert cfg.oric_context_draws == 1
