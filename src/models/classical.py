"""Classical ML estimators.

Implements lightweight estimators using scikit-learn:
- Ridge Regression (#1)
- Lasso Regression (#2)  
- Logistic Regression (#3)
- Random Forest (#4)
- XGBoost (#5)
- LightGBM (#6)
"""

import logging
from typing import Dict, Optional

import numpy as np

from .base import BaseEstimator

log = logging.getLogger(__name__)


class RidgeEstimator(BaseEstimator):
    """Ridge Regression estimator (L2 regularization).
    
    Estimator #1: Linear model with L2 regularization for AP regression.
    Stage: Pre-inference (temporal features only)
    """
    
    name = "ridge_regression"
    task_type = "regression"
    stage = "pre"
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class LassoEstimator(BaseEstimator):
    """Lasso Regression estimator (L1 regularization).
    
    Estimator #2: Linear model with L1 regularization for AP regression.
    Stage: Post-inference (temporal + current edge features)
    """
    
    name = "lasso_regression"
    task_type = "regression"
    stage = "post"
    
    def __init__(self, alpha: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=self.alpha, max_iter=10000)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X)


class LogisticEstimator(BaseEstimator):
    """Logistic Regression estimator for binary offload decision.
    
    Estimator #3: Binary classification - offload (1) or not (0).
    Stage: Post-inference (temporal + current edge features)
    """
    
    name = "logistic_regression"
    task_type = "classification"
    stage = "post"
    
    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestEstimator(BaseEstimator):
    """Random Forest estimator.
    
    Estimator #4: Ensemble of decision trees for regression or classification.
    Stage: Both (pre with temporal only, post with edge features)
    """
    
    name = "random_forest"
    task_type = "regression"  # Default, can be changed
    stage = "both"
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 task_type: str = "regression", **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.task_type = task_type
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if self.task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                random_state=42, 
                n_jobs=-1
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1
            )
        self.model.fit(X, y)
        self.is_fitted = True
        
    def _set_single_thread(self):
        if not hasattr(self, '_single_thread_set'):
            try:
                self.model.set_params(n_jobs=1)
            except (AttributeError, TypeError):
                pass
            self._single_thread_set = True

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._set_single_thread()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        self._set_single_thread()
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        return None


class XGBoostEstimator(BaseEstimator):
    """XGBoost estimator (gradient boosted trees).
    
    Estimator #5: Gradient boosting for regression or classification.
    Stage: Both
    """
    
    name = "xgboost"
    task_type = "regression"
    stage = "both"
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, task_type: str = "regression", **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.task_type = task_type
        self._backend = "xgboost"  # track which backend is used
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        try:
            import xgboost as xgb
            self._backend = "xgboost"
            if self.task_type == "regression":
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='logloss',
                )
        except ImportError:
            raise ImportError(
                "xgboost package is required for XGBoostEstimator. "
                "Install it with: pip install xgboost\n"
                "The previous silent fallback to sklearn.GradientBoosting "
                "caused ~100x slower inference and was removed."
            )
        self.model.fit(X, y)
        self.is_fitted = True

    def get_info(self) -> Dict:
        return {"description": f"XGBoost ({self._backend})"}

    def _set_single_thread(self):
        """Set single-threaded mode on the XGBoost booster.

        Setting ``n_jobs`` on the sklearn wrapper alone does NOT propagate to
        the underlying C++ booster, which keeps ``nthread=-1`` from training.
        This causes ~35 ms thread-pool overhead per single-sample prediction.
        We must set ``nthread`` on the booster directly.
        """
        if not hasattr(self, '_single_thread_set'):
            try:
                booster = self.model.get_booster()
                booster.set_param({'nthread': 1})
                self.model.n_jobs = 1
            except (AttributeError, TypeError):
                pass
            self._single_thread_set = True

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._set_single_thread()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        self._set_single_thread()
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        return None


class LightGBMEstimator(BaseEstimator):
    """LightGBM estimator (histogram-based boosting).
    
    Estimator #6: Fast histogram-based gradient boosting.
    Stage: Both
    """
    
    name = "lightgbm"
    task_type = "regression"
    stage = "both"
    
    def __init__(self, n_estimators: int = 100, max_depth: int = -1,
                 learning_rate: float = 0.1, num_leaves: int = 31,
                 task_type: str = "regression", **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.task_type = task_type
        self._backend = "lightgbm"  # track which backend is used
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        try:
            import lightgbm as lgb
            self._backend = "lightgbm"
            if self.task_type == "regression":
                self.model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    num_leaves=self.num_leaves,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
        except ImportError:
            log.warning(
                "lightgbm not installed, falling back to sklearn "
                "HistGradientBoosting (comparable speed, slightly "
                "different behaviour). Install: pip install lightgbm"
            )
            self._backend = "sklearn_hist"
            if self.task_type == "regression":
                from sklearn.ensemble import HistGradientBoostingRegressor
                self.model = HistGradientBoostingRegressor(
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth if self.max_depth > 0 else None,
                    learning_rate=self.learning_rate,
                    random_state=42,
                )
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                self.model = HistGradientBoostingClassifier(
                    max_iter=self.n_estimators,
                    max_depth=self.max_depth if self.max_depth > 0 else None,
                    learning_rate=self.learning_rate,
                    random_state=42,
                )
        self.model.fit(X, y)
        self.is_fitted = True

    def get_info(self) -> Dict:
        return {"description": f"LightGBM ({self._backend})"}

    def _set_single_thread(self):
        """Set single-threaded mode for LightGBM prediction.

        Like XGBoost, LightGBM's ``n_jobs=-1`` causes thread-pool overhead
        that dominates single-sample latency (~12 ms → ~0.4 ms).
        """
        if not hasattr(self, '_single_thread_set'):
            try:
                self.model.set_params(n_jobs=1)
            except (AttributeError, TypeError):
                pass
            self._single_thread_set = True

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self._set_single_thread()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        self._set_single_thread()
        if self.task_type == "classification":
            return self.model.predict_proba(X)
        return None
