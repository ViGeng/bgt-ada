"""Evaluation metrics for estimators.

Computes regression and classification metrics for estimator evaluation.
"""

from typing import Dict, List, Optional

import numpy as np


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr, _ = spearmanr(x, y)
    return float(corr) if np.isfinite(corr) else 0.0


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'spearman_rho': _safe_spearman(y_true, y_pred),
    }
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted',
) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average=average)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            if y_proba.ndim == 2:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass

    return metrics


def compute_ranking_metrics(
    actual_gains: np.ndarray,
    predicted_gains: np.ndarray,
    top_k: List[int] = None,
) -> Dict[str, float]:
    """Compute ranking quality metrics."""
    if top_k is None:
        top_k = [10, 50, 100]

    actual_gains = np.asarray(actual_gains)
    predicted_gains = np.asarray(predicted_gains)
    n = len(actual_gains)

    metrics = {
        'spearman_rho': _safe_spearman(actual_gains, predicted_gains),
    }

    for k in top_k:
        if k > n:
            continue
        actual_top_k = set(np.argsort(actual_gains)[-k:])
        pred_top_k = set(np.argsort(predicted_gains)[-k:])
        metrics[f'precision_at_{k}'] = float(len(actual_top_k & pred_top_k) / k)

    relevance = (actual_gains - actual_gains.min()) / (actual_gains.max() - actual_gains.min() + 1e-10)
    pred_order = np.argsort(predicted_gains)[::-1]
    dcg = np.sum(relevance[pred_order] / np.log2(np.arange(2, n + 2)))
    ideal_order = np.argsort(actual_gains)[::-1]
    idcg = np.sum(relevance[ideal_order] / np.log2(np.arange(2, n + 2)))
    metrics['ndcg'] = float(dcg / idcg) if idcg > 0 else 0.0

    return metrics
