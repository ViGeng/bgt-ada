"""Conditional reward smoothing for scenario-aware target construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CEORIC_DEFAULT_K = 16
CEORIC_DEFAULT_TAU = 0.5


@dataclass
class KernelNeighborhood:
    """Precomputed visual neighborhoods for train/test smoothing."""

    train_indices: np.ndarray
    train_weights: np.ndarray
    test_indices: Optional[np.ndarray] = None
    test_weights: Optional[np.ndarray] = None


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms


def _kernel_weights(distances: np.ndarray, tau: float) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    scaled = -(distances ** 2) / tau
    scaled -= np.max(scaled, axis=1, keepdims=True)
    weights = np.exp(scaled)
    denom = np.sum(weights, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return weights / denom


def compute_frozen_image_embeddings(
    image_paths: Sequence[str],
    image_size: int = 128,
    batch_size: int = 128,
    device: str = "auto",
    quiet: bool = False,
) -> np.ndarray:
    """Extract frozen MobileNetV3-Small embeddings for image paths."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    class _EmbeddingDataset(Dataset):
        def __init__(self, paths, transform, image_size_):
            self.paths = [_resolve_path(p) for p in paths]
            self.transform = transform
            self.image_size = image_size_

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                with Image.open(path) as pil_img:
                    img = pil_img.convert("RGB")
                return self.transform(img), True
            except Exception:
                return torch.zeros(3, self.image_size, self.image_size), False

    device_name = device
    if device_name in (None, "", "auto"):
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device_name, str) and device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    torch_device = torch.device(device_name)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    try:
        base = models.mobilenet_v3_small(weights="DEFAULT")
    except Exception:
        base = models.mobilenet_v3_small(weights=None)
    base = base.to(torch_device)
    base.eval()
    features = base.features
    avgpool = base.avgpool
    embedding_dim = int(base.classifier[0].in_features)

    dataset = _EmbeddingDataset(image_paths, transform, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    embeddings = np.zeros((len(dataset), embedding_dim), dtype=np.float32)
    cursor = 0
    iterator = loader
    if not quiet and len(dataset) > batch_size:
        from tqdm import tqdm
        iterator = tqdm(loader, desc="  Image embeddings", leave=False)

    with torch.no_grad():
        for images, valid in iterator:
            images = images.to(torch_device, non_blocking=True)
            feats = features(images)
            feats = avgpool(feats).flatten(1).cpu().numpy()
            valid_mask = valid.numpy().astype(bool)
            feats[~valid_mask] = 0.0
            batch_n = feats.shape[0]
            embeddings[cursor:cursor + batch_n] = feats
            cursor += batch_n

    return embeddings


def build_kernel_neighborhood(
    train_embeddings: np.ndarray,
    test_embeddings: Optional[np.ndarray] = None,
    k: int = CEORIC_DEFAULT_K,
    tau: float = CEORIC_DEFAULT_TAU,
) -> KernelNeighborhood:
    """Build kNN neighborhoods once and reuse them across metric families."""
    from sklearn.neighbors import NearestNeighbors

    train_embeddings = _normalize_embeddings(train_embeddings)
    if len(train_embeddings) == 0:
        raise ValueError("train_embeddings must be non-empty")
    test_embeddings = (
        _normalize_embeddings(test_embeddings)
        if test_embeddings is not None and len(test_embeddings) > 0
        else None
    )

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    n_train = len(train_embeddings)
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(train_embeddings)

    if n_train == 1:
        train_indices = np.zeros((1, 1), dtype=np.int64)
        train_weights = np.ones((1, 1), dtype=np.float64)
    else:
        k_train = min(k, n_train - 1)
        dists_full, idx_full = nn.kneighbors(train_embeddings, n_neighbors=k_train + 1)
        train_indices = np.empty((n_train, k_train), dtype=np.int64)
        train_distances = np.empty((n_train, k_train), dtype=np.float64)
        for row_idx in range(n_train):
            keep_mask = idx_full[row_idx] != row_idx
            kept_idx = idx_full[row_idx][keep_mask][:k_train]
            kept_dist = dists_full[row_idx][keep_mask][:k_train]
            if len(kept_idx) < k_train:
                pad_count = k_train - len(kept_idx)
                kept_idx = np.pad(kept_idx, (0, pad_count), mode="edge")
                kept_dist = np.pad(kept_dist, (0, pad_count), mode="edge")
            train_indices[row_idx] = kept_idx
            train_distances[row_idx] = kept_dist
        train_weights = _kernel_weights(train_distances, tau)

    test_indices = None
    test_weights = None
    if test_embeddings is not None:
        k_test = min(k, n_train)
        test_distances, test_indices = nn.kneighbors(test_embeddings, n_neighbors=k_test)
        test_weights = _kernel_weights(test_distances, tau)

    return KernelNeighborhood(
        train_indices=train_indices,
        train_weights=train_weights,
        test_indices=test_indices,
        test_weights=test_weights,
    )


def smooth_rewards(
    train_values: np.ndarray,
    neighborhood: KernelNeighborhood,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply precomputed neighborhoods to a reward vector."""
    (train_smooth, train_var,
     test_smooth, test_var) = smooth_reward_vectors(
        np.asarray(train_values, dtype=np.float64).reshape(-1, 1),
        neighborhood,
    )
    return (
        train_smooth[:, 0],
        train_var[:, 0],
        None if test_smooth is None else test_smooth[:, 0],
        None if test_var is None else test_var[:, 0],
    )


def smooth_reward_vectors(
    train_values: np.ndarray,
    neighborhood: KernelNeighborhood,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply precomputed neighborhoods independently to each vector dimension."""
    train_values = np.asarray(train_values, dtype=np.float64)
    if train_values.ndim != 2:
        raise ValueError(
            f"Expected train_values with shape [N, D], got {train_values.shape}"
        )
    if len(train_values) != neighborhood.train_indices.shape[0]:
        raise ValueError(
            "train_values and neighborhood.train_indices must share the same first dimension"
        )

    train_neighbors = train_values[neighborhood.train_indices]
    train_weights = neighborhood.train_weights[:, :, None]
    train_smooth = np.sum(train_neighbors * train_weights, axis=1)
    train_var = np.sum(
        train_weights * (train_neighbors - train_smooth[:, None, :]) ** 2,
        axis=1,
    )

    test_smooth = None
    test_var = None
    if neighborhood.test_indices is not None and neighborhood.test_weights is not None:
        test_neighbors = train_values[neighborhood.test_indices]
        test_weights = neighborhood.test_weights[:, :, None]
        test_smooth = np.sum(test_neighbors * test_weights, axis=1)
        test_var = np.sum(
            test_weights * (test_neighbors - test_smooth[:, None, :]) ** 2,
            axis=1,
        )

    return train_smooth, train_var, test_smooth, test_var
