"""Virtual estimators — zero-cost baselines for evaluation.

Virtual estimators have no real model, zero GFLOPs, and zero latency.
They serve as reference points in the offloading evaluation:

- **WeakModelEstimator** ("weak_model"): Never offloads (always edge).
  Predicts zero gain for every frame.
- **StrongModelEstimator** ("strong_model"): Always offloads (always cloud).
  Predicts maximum gain for every frame.
- **OracleEstimator** ("oracle"): Perfect foresight — knows the true
  per-frame AP gain and makes optimal offloading decisions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseEstimator


class VirtualEstimator(BaseEstimator):
    """Base class for virtual estimators with zero computational cost.

    Virtual estimators don't load real models.  They report zero GFLOPs,
    zero parameters, and zero latency so that the complexity charts clearly
    separate them from real estimators.
    """

    task_type: str = "regression"
    stage: str = "pre"
    description: str = "Virtual estimator"

    def get_info(self) -> Dict[str, Any]:
        return {"description": self.description, "gflops": 0.0, "params": 0.0}

    def measure_pure_latency(self, *_args, **_kwargs) -> float:
        return 0.0

    def fit(self, X, y, **kwargs) -> None:
        self.is_fitted = True

    def save(self, path: Path) -> None:
        pass  # nothing to persist

    @classmethod
    def load(cls, path: Path) -> "VirtualEstimator":
        p = cls()
        p.is_fitted = True
        return p


class WeakModelEstimator(VirtualEstimator):
    """Always uses edge — predicts zero gain (never offload)."""

    name = "weak_model"
    description = "Always Edge (Never Offload)"

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.zeros(len(X))


class StrongModelEstimator(VirtualEstimator):
    """Always uses cloud — predicts maximum gain (always offload)."""

    name = "strong_model"
    description = "Always Cloud (Always Offload)"

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.ones(len(X))


class OracleEstimator(VirtualEstimator):
    """Perfect estimator — knows the true per-frame AP gain."""

    name = "oracle"
    description = "Oracle (Perfect Knowledge)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ground_truth: Optional[np.ndarray] = None

    def set_ground_truth(self, y: np.ndarray) -> None:
        """Inject ground-truth gain so predict() can return it."""
        self._ground_truth = np.asarray(y)

    def predict(self, X, **kwargs) -> np.ndarray:
        if self._ground_truth is not None:
            return self._ground_truth.copy()
        raise ValueError("Oracle requires ground truth. "
                         "Call set_ground_truth(y) first.")
