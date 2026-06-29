"""Base estimator interface and utilities.

All estimators inherit from BaseEstimator to ensure consistent API.
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class BaseEstimator(ABC):
    """Abstract base class for all estimators.

    Attributes:
        name: Unique identifier for the estimator type
        task_type: Either "regression" or "classification"
        stage: Either "pre" (before edge inference), "post" (after), or "both"
    """

    name: str = "base"
    task_type: str = "regression"
    stage: str = "both"

    checkpoint_ext: str = ".pkl"
    input_key: str = "default"  # "default" | "proposal" | "image"

    @classmethod
    def resolve_target_metadata(cls, target_bundle: dict, data: dict,
                                primary_suffix: str) -> None:
        """Inject family-specific metadata into composite target bundles.

        Override in subclasses that need extra keys (e.g. LCER beta/tau).
        The base implementation adds common metadata bundles when they can be
        inferred from the primary target family so lightweight custom
        estimators do not need boilerplate.
        """
        if not primary_suffix:
            return

        if primary_suffix.startswith("lcer_"):
            metric_family = primary_suffix.removeprefix("lcer_")
            for prefix, key in [("meta_beta_", "meta_beta"), ("meta_tau_", "meta_tau")]:
                full_key = f"{prefix}{metric_family}"
                if full_key in data and key not in target_bundle:
                    target_bundle[key] = data[full_key]

    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.config = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import (accuracy_score, f1_score,
                                     mean_absolute_error, mean_squared_error,
                                     r2_score, roc_auc_score)

        pred = self.predict(X)

        if self.task_type == "regression":
            return {
                'mae': float(mean_absolute_error(y, pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, pred))),
                'r2': float(r2_score(y, pred)),
            }
        else:
            metrics = {
                'accuracy': float(accuracy_score(y, pred)),
                'f1': float(f1_score(y, pred, average='weighted')),
            }
            proba = self.predict_proba(X)
            if proba is not None and len(np.unique(y)) == 2:
                try:
                    metrics['auc_roc'] = float(roc_auc_score(y, proba[:, 1]))
                except ValueError:
                    pass
            return metrics

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'name': self.name,
            'task_type': self.task_type,
            'stage': self.stage,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'config': self.config,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path, device: str = None) -> 'BaseEstimator':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        kwargs = state.get('config', {})
        if device is not None:
            kwargs['device'] = device
        estimator = cls(**kwargs)
        estimator.model = state['model']
        estimator.is_fitted = state['is_fitted']
        return estimator

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, task={self.task_type}, stage={self.stage})"


class ImageDataset:
    """PyTorch Dataset for lazy image loading."""

    def __init__(self, paths: List[str], targets: np.ndarray, transform, image_size: int = 224):
        import torch
        self.torch = torch
        self.paths = paths
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        # Ensure file handles are closed immediately after decode.
        with Image.open(self.paths[idx]) as pil_img:
            img = pil_img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


class ImageEstimator(BaseEstimator):
    """Base class for image-based estimators (CNN, ViT).

    Subclasses must implement ``_setup_model()`` to initialize self.model.

    Class-level defaults can be overridden per-subclass::

        class MyEstimator(ImageEstimator):
            default_epochs = 20
            default_lr = 0.0005
    """

    checkpoint_ext: str = ".pt"
    input_key: str = "image"

    default_epochs: int = 10
    default_batch_size: int = 512
    default_lr: float = 0.001
    default_patience: int = 5
    default_weight_decay: float = 0.01
    pretrained: bool = False

    def __init__(self, image_size: int = 224, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.transform = None
        self.device_name = device
        self.device = None

    def _setup_device(self):
        """Resolve self.device from self.device_name (call once in _setup_model)."""
        import torch
        if self.device_name and self.device_name != "auto":
            self.device = torch.device(self.device_name)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def _setup_model(self):
        """Initialize model architecture. Must set self.model and self.device."""
        pass

    def _setup_transforms(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def _setup_train_transforms(self):
        from torchvision import transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    def _compute_flops(self):
        try:
            import torch
            from thop import profile

            if self.model is None:
                self._setup_model()

            in_channels = 3
            for m in self.model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    in_channels = m.in_channels
                    break

            dummy = torch.randn(1, in_channels, self.image_size, self.image_size)
            dummy = dummy.to(next(self.model.parameters()).device)
            self.model.eval()
            macs, params = profile(self.model, inputs=(dummy,), verbose=False)
            return round(macs * 2 / 1e9, 6), round(params / 1e6, 3)
        except (ImportError, Exception):
            return None, None

    def measure_pure_latency(self, X_paths: List[str],
                             n_warmup: int = 5, n_samples: int = 50) -> float:
        """Measure pure inference latency (no IO) per sample in ms."""
        import time

        import torch
        from PIL import Image

        if not hasattr(self, 'model') or self.model is None:
            if hasattr(self, '_setup_model'):
                self._setup_model()
            else:
                return 0.0

        n = min(len(X_paths), n_samples)
        if n == 0:
            return 0.0

        paths = X_paths[:n]
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception:
                continue

        if not images:
            return 0.0

        use_cuda = self.device.type == 'cuda'
        batch = torch.stack(images).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(batch[:1])
        if use_cuda:
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for i in range(len(images)):
                img = batch[i:i+1]
                if use_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = self.model(img)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        return float(np.mean(times) * 1000)

    def fit(self, X_paths: List[str], y: np.ndarray, **kwargs) -> None:
        import copy
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from tqdm import tqdm as _tqdm

        from ..losses import extract_loss_params, get_loss, mse_loss

        self._setup_model()
        self._setup_train_transforms()

        torch.backends.cudnn.benchmark = True

        epochs = kwargs.get('epochs', self.default_epochs)
        batch_size = kwargs.get('batch_size', self.default_batch_size)
        lr = kwargs.get('lr', self.default_lr)
        patience = kwargs.get('patience', self.default_patience)
        weight_decay = kwargs.get('weight_decay', self.default_weight_decay)
        num_workers = kwargs.get('num_workers', 4)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)

        if self.pretrained and 'lr' not in kwargs:
            lr = lr * 0.1

        # Standardise targets for stable CNN training
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_norm = (y - self._y_mean) / self._y_std

        use_compile = kwargs.get('compile', True)
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

        train_dataset = ImageDataset(X_paths, y_norm, self.train_transform, self.image_size)
        val_dataset = ImageDataset(X_paths, y_norm, self.transform, self.image_size)
        n = len(train_dataset)
        val_size = max(1, int(0.1 * n))
        train_size = n - val_size

        gen = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=gen).tolist()
        train_indices = perm[:train_size]
        val_indices = perm[train_size:]

        train_ds = torch.utils.data.Subset(train_dataset, train_indices)
        val_ds = torch.utils.data.Subset(val_dataset, val_indices)

        use_cuda = self.device.type == 'cuda'
        loader_kwargs = dict(
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0 and kwargs.get('persistent_workers', False)),
            prefetch_factor=4 if num_workers > 0 else None,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                **loader_kwargs)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10,
            steps_per_epoch=steps_per_epoch, epochs=epochs,
            pct_start=0.1,
        )
        # Resolve loss: configurable via kwargs["loss"], default mse
        loss_name = kwargs.get("loss", None)
        loss_params = extract_loss_params(kwargs)
        criterion = get_loss(loss_name, **loss_params) or mse_loss

        # Boundary-aware losses: set the z-normalised decision boundary so
        # that sign-aware weighting still corresponds to the original 0
        # boundary (harmful vs. beneficial offload) after standardisation.
        # Generic: any loss (or its .base_loss) with a 'boundary' attribute
        # gets the z-normalised value automatically.
        z_boundary = (0.0 - self._y_mean) / self._y_std
        if hasattr(criterion, "boundary"):
            criterion.boundary = z_boundary
        if hasattr(criterion, "base_loss") and hasattr(criterion.base_loss, "boundary"):
            criterion.base_loss.boundary = z_boundary

        use_amp = use_cuda
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        best_val = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            pbar = _tqdm(train_loader,
                         desc=f"  Epoch {epoch+1}/{epochs}", leave=False,
                         mininterval=2.0)
            for imgs, targets in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(imgs).squeeze(-1)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # AMP can skip optimizer updates on inf/NaN grads; keep scheduler in sync.
                if scaler.get_scale() >= prev_scale and getattr(optimizer, '_step_count', 0) > 0:
                    scheduler.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            self.model.eval()
            val_loss = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for imgs, targets in val_loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    outputs = self.model(imgs).squeeze(-1)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                _sd = self.model.state_dict()
                best_state = type(_sd)((k, v.cpu().clone()) for k, v in _sd.items())
                if hasattr(_sd, '_metadata'):
                    best_state._metadata = _sd._metadata
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Restore best validation model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.is_fitted = True

        self.fit_metrics = self._evaluate_splits(
            train_loader, val_loader, criterion, use_amp
        )
        stopped_epoch = epoch + 1
        self.fit_metrics['epochs_run'] = stopped_epoch
        self.fit_metrics['epochs_max'] = epochs
        from .. import log
        log.info(f"Train MSE={self.fit_metrics['train_mse']:.4f}  "
                 f"Val MSE={self.fit_metrics['val_mse']:.4f}  "
                 f"Train R\u00b2={self.fit_metrics['train_r2']:.4f}  "
                 f"Val R\u00b2={self.fit_metrics['val_r2']:.4f}  "
                 f"[{stopped_epoch}/{epochs} epochs]", indent=8)

    def predict(self, X_paths: List[str], batch_size: int = 64,
                **kwargs) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm as _tqdm

        num_workers = kwargs.get('num_workers', 2)
        dummy_targets = np.zeros(len(X_paths))
        dataset = ImageDataset(X_paths, dummy_targets, self.transform,
                               self.image_size)

        use_cuda = self.device.type == 'cuda'
        loader_kwargs = dict(
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0 and kwargs.get('persistent_workers', False)),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, **loader_kwargs)

        use_amp = use_cuda
        predictions = []
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            pbar = _tqdm(loader, desc="  Inferencing", leave=False,
                         mininterval=2.0)
            for imgs, _ in pbar:
                imgs = imgs.to(self.device, non_blocking=True)
                outputs = self.model(imgs).squeeze(-1)
                predictions.extend(outputs.cpu().numpy())

        preds = np.array(predictions)
        # Inverse-transform from normalised space to original target scale
        y_mean = getattr(self, '_y_mean', 0.0)
        y_std = getattr(self, '_y_std', 1.0)
        return preds * y_std + y_mean

    def _evaluate_splits(self, train_loader, val_loader, criterion, use_amp):
        import torch
        from sklearn.metrics import mean_absolute_error, r2_score

        metrics = {}
        self.model.eval()
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            all_preds, all_targets = [], []
            total_loss = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for imgs, targets in loader:
                    imgs = imgs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    outputs = self.model(imgs).squeeze(-1)
                    total_loss += criterion(outputs, targets).item()
                    all_preds.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            preds = np.array(all_preds)
            targets = np.array(all_targets)
            metrics[f'{split}_mse'] = round(total_loss / max(len(loader), 1), 6)
            metrics[f'{split}_mae'] = round(float(mean_absolute_error(targets, preds)), 6)
            metrics[f'{split}_r2'] = round(float(r2_score(targets, preds)), 6)

        return metrics

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.model.state_dict()
        clean_state = type(state_dict)(
            (k.replace('_orig_mod.', ''), v) for k, v in state_dict.items()
        )
        if hasattr(state_dict, '_metadata'):
            clean_state._metadata = {
                k.replace('_orig_mod.', ''): v for k, v in state_dict._metadata.items()
            }

        torch.save({
            'model_state_dict': clean_state,
            'is_fitted': self.is_fitted,
            'name': self.name,
            'image_size': self.image_size,
            'y_mean': getattr(self, '_y_mean', 0.0),
            'y_std': getattr(self, '_y_std', 1.0),
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = None) -> 'ImageEstimator':
        import torch
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        estimator = cls(image_size=checkpoint.get('image_size', 224), device=device)
        estimator._setup_model()

        state_dict = checkpoint['model_state_dict']
        clean_keys = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        # Merge into model's own state dict to preserve _metadata (required by e.g. MNASNet)
        model_sd = estimator.model.state_dict()
        for k in model_sd:
            if k in clean_keys:
                model_sd[k] = clean_keys[k]
        estimator.model.load_state_dict(model_sd)

        estimator.is_fitted = checkpoint.get('is_fitted', True)
        estimator._y_mean = checkpoint.get('y_mean', 0.0)
        estimator._y_std = checkpoint.get('y_std', 1.0)
        return estimator
