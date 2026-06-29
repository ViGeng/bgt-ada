"""EdgeML ORIC-based regression estimator.

Paper-faithful reproduction of the EdgeML approach (Qiu et al., 2024).
Post-inference estimator that trains an MLP regressor on weak-detector
proposal features to predict MORIC (CDF-transformed ORIC).

Offloading uses a fixed threshold T = 1 - r for target ratio r, derived
from MORIC's uniform CDF property (paper Section V-B).

Reference: Qiu et al. - Optimizing Edge Offloading Decisions for Object Detection
Architecture: Multi-layer MLP with BatchNorm + ReLU + Dropout,
              input standardised via sklearn StandardScaler.
              Optional grid search with K-fold cross-validation.
"""

import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseEstimator


def _resolve_torch_device(device_name: Optional[str]):
    import torch

    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_float_matrix(X: np.ndarray) -> np.ndarray:
    """Convert arbitrary estimator input into a dense 2D float matrix."""
    arr = np.asarray(X)
    if arr.ndim == 0:
        raise ValueError("EdgeML expects at least one sample.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _split_indices(n: int, val_fraction: float = 0.1, seed: int = 42):
    n = int(n)
    if n <= 1:
        idx = np.arange(n, dtype=int)
        return idx, idx
    val_size = max(1, int(round(n * val_fraction)))
    val_size = min(val_size, n - 1)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx


def _build_mlp(input_dim: int, hidden: List[int], dropout: float = 0.1,
               batch_norm: bool = True):
    import torch
    import torch.nn as nn

    hidden = list(hidden or [64, 32, 16, 1])
    if hidden[-1] != 1:
        hidden = hidden + [1]

    class _MLP(nn.Module):
        def __init__(self, sizes):
            super().__init__()
            stacks = nn.ModuleList()
            for i, (inp, out) in enumerate(zip(sizes[:-1], sizes[1:])):
                is_last = i == len(sizes) - 2
                linear = nn.Linear(inp, out)
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
                nn.init.zeros_(linear.bias)
                modules: list = [linear]
                if not is_last:
                    if batch_norm:
                        modules.append(nn.BatchNorm1d(out))
                    modules.append(nn.ReLU(inplace=True))
                    if dropout and dropout > 0:
                        modules.append(nn.Dropout(dropout))
                stacks.append(nn.Sequential(*modules))
            self.stacks = stacks

        def forward(self, x):
            for stack in self.stacks:
                x = stack(x)
            return x.squeeze(-1)

    return _MLP([input_dim] + hidden)


def _make_loss(loss_name: Optional[str], target_mode: str, loss_params: Dict[str, Any]):
    from ..losses import get_loss, mse_loss, weighted_mse_loss

    default_loss = weighted_mse_loss if target_mode.startswith("moric") else mse_loss
    return get_loss(loss_name, **loss_params) or default_loss


def _train_regressor(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    criterion,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int = 42,
):
    import torch
    import torch.utils.data as Data

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    train_ds = Data.TensorDataset(X_train_t, y_train_t)
    loader = Data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    stopped_epoch = 0

    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())

        stopped_epoch = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_loss": float(best_val_loss),
        "epochs_run": stopped_epoch,
        "epochs_max": int(epochs),
    }


def _cross_validate_candidates(
    X: np.ndarray,
    y: np.ndarray,
    candidates: List[Dict[str, Any]],
    *,
    folds: int,
    target_mode: str,
    loss_name: Optional[str],
    loss_params: Dict[str, Any],
    epochs: int,
    patience: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Select the best hyperparameters using K-fold validation."""
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    folds = max(2, min(int(folds), n))
    if folds <= 1 or len(candidates) <= 1 or n < 2:
        return candidates[0] if candidates else {}

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_indices = np.array_split(indices, folds)

    scored: list[tuple[float, float, Dict[str, Any]]] = []
    for spec in candidates:
        fold_losses = []
        for fold_id in range(folds):
            val_idx = fold_indices[fold_id]
            train_idx = np.concatenate(
                [fold_indices[j] for j in range(folds) if j != fold_id]
            )

            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx]).astype(np.float32)
            X_val = scaler.transform(X[val_idx]).astype(np.float32)
            y_train = y[train_idx].astype(np.float32)
            y_val = y[val_idx].astype(np.float32)

            device = _resolve_torch_device(spec.get("device"))
            model = _build_mlp(
                X.shape[1],
                spec["hidden"],
                dropout=spec["dropout"],
                batch_norm=spec["batch_norm"],
            ).to(device)
            criterion = _make_loss(loss_name, target_mode, loss_params)
            _, stats = _train_regressor(
                model, X_train, y_train, X_val, y_val,
                criterion=criterion, device=device, epochs=epochs,
                batch_size=spec["batch_size"], lr=spec["lr"],
                weight_decay=spec["weight_decay"], patience=patience,
                seed=seed + fold_id,
            )
            fold_losses.append(stats["best_val_loss"])

        scored.append((float(np.mean(fold_losses)), float(np.std(fold_losses)), spec))

    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


class EdgeMLNet:
    """MLP wrapper for proposal-based EdgeML inputs."""

    @staticmethod
    def build(input_dim: int, hidden: Optional[List[int]] = None,
              dropout: float = 0.1, batch_norm: bool = True):
        return _build_mlp(input_dim, hidden or [64, 32, 16, 1],
                          dropout=dropout, batch_norm=batch_norm)


def _expand_search_space(search_space, defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a search space specification into a list of candidate dicts."""
    if search_space is None:
        return [defaults]

    if isinstance(search_space, list):
        return [{**defaults, **spec} for spec in search_space]

    if isinstance(search_space, dict):
        keys = list(search_space.keys())
        value_lists = [list(search_space[k]) for k in keys]
        candidates = []
        for combo in itertools.product(*value_lists):
            spec = dict(zip(keys, combo))
            candidates.append({**defaults, **spec})
        return candidates

    raise TypeError("search_space must be a list of dicts or a dict of iterables")


class EdgeMLEstimator(BaseEstimator):
    """Paper-faithful EdgeML estimator (Qiu et al., 2024).

    Trains an MLP regressor on weak-detector proposal features to predict
    MORIC.  Uses weighted MSE loss (paper Eq. 7) and fixed threshold
    T = 1 - r for sequential offloading at target ratio r.

    Features: top-25 ranked detections flattened into a feature vector.
    Training: MLP with BatchNorm + ReLU + Dropout, StandardScaler,
              optional grid search with K-fold cross-validation.
    """

    name = "edgeml"
    task_type = "regression"
    stage = "post"
    checkpoint_ext = ".pt"
    input_key = "proposal"
    def __init__(self, epochs: int = 100, lr: float = 5e-3,
                 batch_size: int = 64, weight_decay: float = 5e-5,
                 hidden: Optional[List[int]] = None, dropout: float = 0.1,
                 patience: int = 15, cv_folds: int = 5,
                 grid_search: bool = True, batch_norm: bool = True,
                 target_mode: str = "moric", proposal_count: int = 25,
                 search_space=None, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.hidden = hidden
        self.dropout = dropout
        self.patience = patience
        self.cv_folds = cv_folds
        self.grid_search = grid_search
        self.batch_norm = batch_norm
        self.target_mode = target_mode
        self.proposal_count = proposal_count
        self.search_space = search_space
        self.device_name = device
        self._input_dim: int = 0
        self._input_shape: Optional[tuple] = None
        self._scaler: Any = None
        self.best_params_: Dict[str, Any] = {}
        self.cv_summary_: Dict[str, Any] = {}

    def _candidate_specs(self, device_name: str) -> List[Dict[str, Any]]:
        if self.search_space is not None:
            base = {
                "hidden": list(self.hidden) if self.hidden is not None else [64, 32, 16, 1],
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "device": device_name,
            }
            return _expand_search_space(self.search_space, base)

        if self.hidden is not None:
            base = [{
                "hidden": list(self.hidden),
                "dropout": self.dropout,
                "batch_norm": self.batch_norm,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "device": device_name,
            }]
            return _expand_search_space(self.search_space, base[0])

        defaults = {
            "hidden": [64, 32, 16, 1],
            "dropout": self.dropout,
            "batch_norm": self.batch_norm,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "device": device_name,
        }

        if not self.grid_search:
            return [defaults]

        candidates = []
        hidden_choices = [
            [64, 32, 16, 1],
            [128, 64, 32, 1],
            [256, 128, 64, 1],
        ]
        lr_choices = [self.lr, max(self.lr * 0.5, 1e-4)]
        dropout_choices = [self.dropout, 0.0]
        batch_choices = [self.batch_size, max(16, self.batch_size // 2)]
        weight_decay_choices = [self.weight_decay, 0.0]

        for hidden, lr, dropout, batch_size, weight_decay in itertools.product(
            hidden_choices, lr_choices, dropout_choices, batch_choices, weight_decay_choices
        ):
            candidates.append({
                "hidden": list(hidden),
                "dropout": float(dropout),
                "batch_norm": self.batch_norm,
                "batch_size": int(batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "device": device_name,
            })
        return candidates

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        import torch
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.preprocessing import StandardScaler

        from ..losses import extract_loss_params

        raw_shape = tuple(np.asarray(X).shape[1:])
        if not raw_shape:
            raw_shape = (1,)
        X = _as_float_matrix(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows: {len(X)} != {len(y)}")

        self._input_dim = int(X.shape[1])
        self._input_shape = raw_shape
        device = _resolve_torch_device(getattr(self, "device_name", None))
        loss_name = kwargs.get("loss", None)
        loss_params = extract_loss_params(kwargs)
        criterion = _make_loss(loss_name, self.target_mode, loss_params)

        candidate_specs = self._candidate_specs(str(device))
        if self.grid_search and self.cv_folds > 1 and len(candidate_specs) > 1 and len(X) >= self.cv_folds:
            best_spec = _cross_validate_candidates(
                X, y, candidate_specs, folds=self.cv_folds,
                target_mode=self.target_mode, loss_name=loss_name,
                loss_params=loss_params, epochs=self.epochs,
                patience=self.patience, seed=42,
            )
        else:
            best_spec = candidate_specs[0]

        self.best_params_ = dict(best_spec)
        train_idx, val_idx = _split_indices(len(X), val_fraction=0.1, seed=42)
        scaler = StandardScaler().fit(X[train_idx])
        X_scaled = scaler.transform(X).astype(np.float32)
        self._scaler = scaler

        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_val = X_scaled[val_idx]
        y_val = y[val_idx]

        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

        net = EdgeMLNet.build(
            self._input_dim,
            best_spec["hidden"],
            dropout=best_spec["dropout"],
            batch_norm=best_spec["batch_norm"],
        ).to(device)
        net, stats = _train_regressor(
            net, X_train, y_train, X_val, y_val,
            criterion=criterion, device=device, epochs=self.epochs,
            batch_size=best_spec["batch_size"], lr=best_spec["lr"],
            weight_decay=best_spec["weight_decay"], patience=self.patience,
            seed=42,
        )

        self.model = net.cpu()
        self.model.eval()
        self.is_fitted = True

        with torch.no_grad():
            all_pred = self.model(torch.as_tensor(X_scaled, dtype=torch.float32)).numpy()

        train_pred = all_pred[train_idx]
        val_pred = all_pred[val_idx]
        train_mse = float(np.mean((train_pred - y[train_idx]) ** 2))
        val_mse = float(np.mean((val_pred - y[val_idx]) ** 2))
        train_r2 = float(r2_score(y[train_idx], train_pred))
        val_r2 = float(r2_score(y[val_idx], val_pred))
        self.fit_metrics = {
            "loss_fn": loss_name or ("weighted_mse" if self.target_mode.startswith("moric") else "mse"),
            "cv_folds": int(self.cv_folds),
            "cv_best_val_loss": float(stats["best_val_loss"]),
            "train_mse": round(train_mse, 6),
            "val_mse": round(val_mse, 6),
            "train_r2": round(train_r2, 6),
            "val_r2": round(val_r2, 6),
            "train_mae": round(float(mean_absolute_error(y[train_idx], train_pred)), 6),
            "val_mae": round(float(mean_absolute_error(y[val_idx], val_pred)), 6),
            "epochs_run": int(stats["epochs_run"]),
            "epochs_max": int(stats["epochs_max"]),
        }
        self.cv_summary_ = {
            "best_spec": self.best_params_,
            "cv_best_val_loss": float(stats["best_val_loss"]),
        }

        print(f"    EdgeML [{self.fit_metrics['loss_fn']}] "
              f"train_mse={train_mse:.4f}, val_mse={val_mse:.4f}, "
              f"train_r\u00b2={train_r2:.4f}, val_r\u00b2={val_r2:.4f}  "
              f"[{stats['epochs_run']}/{stats['epochs_max']} epochs]")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        import torch

        if self._scaler is None or self.model is None:
            raise RuntimeError("EdgeMLEstimator must be fitted before prediction.")

        X_scaled = self._scaler.transform(_as_float_matrix(X)).astype(np.float32)
        device = next(self.model.parameters()).device
        t = torch.as_tensor(X_scaled, dtype=torch.float32, device=device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(t)
        return out.cpu().numpy()

    def get_info(self) -> Dict[str, Any]:
        params = 0
        if self.model is not None:
            params = sum(p.numel() for p in self.model.parameters()) / 1e6
        return {
            "description": "EdgeML proposal regressor (MLP + scaler + CV, threshold offloader)",
            "gflops": 0.0,
            "params": round(params, 4),
        }

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hyperparams = {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "patience": self.patience,
            "cv_folds": self.cv_folds,
            "grid_search": self.grid_search,
            "batch_norm": self.batch_norm,
            "target_mode": self.target_mode,
            "proposal_count": self.proposal_count,
            "search_space": self.search_space,
            "device": self.device_name,
        }
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self._input_dim,
            "input_shape": self._input_shape,
            "scaler": self._scaler,
            "is_fitted": self.is_fitted,
            "config": self.config,
            "model_config": hyperparams,
            "best_params_": self.best_params_,
            "cv_summary_": self.cv_summary_,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = None) -> "EdgeMLEstimator":
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(ckpt.get("model_config", ckpt.get("config", {})))
        if device is not None:
            config["device"] = device
        estimator = cls(**config)
        estimator._input_dim = int(ckpt["input_dim"])
        estimator._input_shape = tuple(ckpt.get("input_shape") or ())
        estimator._scaler = ckpt["scaler"]
        estimator.best_params_ = ckpt.get("best_params_", {})
        estimator.cv_summary_ = ckpt.get("cv_summary_", {})

        best_hidden = estimator.best_params_.get("hidden", estimator.hidden or [64, 32, 16, 1])
        best_dropout = estimator.best_params_.get("dropout", estimator.dropout)
        best_batch_norm = estimator.best_params_.get("batch_norm", estimator.batch_norm)
        estimator.model = EdgeMLNet.build(
            estimator._input_dim,
            best_hidden,
            dropout=best_dropout,
            batch_norm=best_batch_norm,
        )
        estimator.model.load_state_dict(ckpt["model_state_dict"])
        if device is not None and device != "auto":
            estimator.model = estimator.model.to(device)
        estimator.model.eval()
        estimator.is_fitted = ckpt.get("is_fitted", True)
        return estimator


# Legacy aliases for backwards compatibility with existing checkpoints/imports
EdgeMLPaperEstimator = EdgeMLEstimator
EdgeMLPaperNet = EdgeMLNet


__all__ = [
    "EdgeMLNet",
    "EdgeMLEstimator",
    "EdgeMLPaperEstimator",
    "EdgeMLPaperNet",
]
