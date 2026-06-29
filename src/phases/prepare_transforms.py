"""Prepare-phase transforms and split index helpers."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import PipelineConfig
from config.scenarios import (LCER3_SCENARIO_COMPONENTS,
                              LCER_SCENARIO_COMPONENTS, SCENARIO_COMPONENTS)

from .. import log
from ..conditional_rewards import (CEORIC_DEFAULT_K, CEORIC_DEFAULT_TAU,
                                   build_kernel_neighborhood,
                                   compute_frozen_image_embeddings,
                                   smooth_reward_vectors)
from ..error_decomposition import LCER_ERROR_TYPES

LCER_TAU_GRID = np.array(
    [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, -0.1, -0.05,
     0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    dtype=np.float64,
)


def _extract_lcer_matrix(df, metric_family: str) -> np.ndarray:
    cols = [f"lcer_vec_{metric_family}_{name}" for name in LCER_ERROR_TYPES]
    data = {
        col: df.get(col, pd.Series(np.zeros(len(df), dtype=np.float64)))
        for col in cols
    }
    return pd.DataFrame(data).fillna(0.0).to_numpy(dtype=np.float64)


def _extract_finegrained_matrix(df, metric_family: str) -> np.ndarray:
    cols = [f"finegrained_vec_{metric_family}_{name}" for name in SCENARIO_COMPONENTS]
    data = {
        col: df.get(col, pd.Series(np.zeros(len(df), dtype=np.float64)))
        for col in cols
    }
    return pd.DataFrame(data).fillna(0.0).to_numpy(dtype=np.float64)


def _compute_moric(oric: np.ndarray) -> np.ndarray:
    """Transform ORIC → MORIC via empirical CDF (rank / N).

    MORIC_i = cdf(ORIC_i) ≈ rank(ORIC_i) / N

    Produces a uniform spread of values in (0, 1] representing
    normalized offloading gain ranks (EdgeML paper eq. 6).
    """
    from scipy.stats import rankdata
    return (rankdata(oric) / len(oric)).astype(np.float64)


def _fit_moric_reference(train_oric: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(train_oric, dtype=np.float64))
    if ref.size == 0:
        raise ValueError("Cannot fit MORIC reference on an empty training split.")
    return ref


def _apply_moric(reference: np.ndarray, oric: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    oric = np.asarray(oric, dtype=np.float64)
    return (
        np.searchsorted(reference, oric, side="right") / max(len(reference), 1)
    ).astype(np.float64)


def _compute_moric_plus(oric: np.ndarray, neg_frac: float = 0.27) -> np.ndarray:
    """Transform ORIC → MORIC+ via split-CDF piecewise normalization.

    Maps each ORIC value to [-1, 1] with a smooth distribution and no
    discontinuity gap around zero:

      - ORIC_i == 0  →  MORIC+_i = 0   (exact zeros stay at zero)
      - ORIC_i >  0  →  MORIC+_i ∈ (0, 1]  via CDF of positive subset
      - ORIC_i <  0  →  MORIC+_i ∈ [-1, 0) via CDF of negative subset

    This avoids the "discontinuity gap" of the previous approach where a
    global CDF with tied zeros created a void in the output distribution
    (e.g., no values between 0 and ~0.3).  By computing separate CDFs
    for the positive and negative subsets, we get a smooth spread in each
    half without gaps.

    Args:
        oric: Array of ORIC values.
        neg_frac: (Legacy, unused.)  The split-CDF approach does not
                  require a manually tuned neg_frac parameter.
    """
    from scipy.stats import rankdata

    result = np.zeros_like(oric, dtype=np.float64)

    pos_mask = oric > 0
    neg_mask = oric < 0
    # ORIC == 0 stays as 0 (no action needed, result is pre-filled with zeros)

    # Positive branch: rank among positives → (0, 1]
    n_pos = int(np.sum(pos_mask))
    if n_pos > 0:
        pos_vals = oric[pos_mask]
        pos_cdf = rankdata(pos_vals) / n_pos  # (0, 1]
        result[pos_mask] = pos_cdf

    # Negative branch: rank among negatives → [-1, 0)
    # Most negative gets -1.0, least negative gets just below 0.
    n_neg = int(np.sum(neg_mask))
    if n_neg > 0:
        neg_vals = oric[neg_mask]
        neg_ranks = rankdata(neg_vals)   # 1, 2, ..., n_neg
        # Map rank to [-1, -1/n_neg]:  (rank-1)/n_neg - 1
        # rank=1 → -1.0, rank=n_neg → -1/n_neg (just below 0)
        result[neg_mask] = (neg_ranks - 1) / n_neg - 1.0

    return result


def _fit_moric_plus_reference(train_oric: np.ndarray) -> dict[str, np.ndarray]:
    train_oric = np.asarray(train_oric, dtype=np.float64)
    return {
        "positive": np.sort(train_oric[train_oric > 0]),
        "negative": np.sort(train_oric[train_oric < 0]),
    }


def _apply_moric_plus(reference: dict[str, np.ndarray], oric: np.ndarray) -> np.ndarray:
    oric = np.asarray(oric, dtype=np.float64)
    result = np.zeros_like(oric, dtype=np.float64)

    pos_ref = np.asarray(reference.get("positive", np.array([])), dtype=np.float64)
    neg_ref = np.asarray(reference.get("negative", np.array([])), dtype=np.float64)

    pos_mask = oric > 0
    if pos_mask.any():
        if pos_ref.size == 0:
            result[pos_mask] = 0.0
        else:
            pos_counts = np.searchsorted(pos_ref, oric[pos_mask], side="right")
            pos_counts = np.clip(pos_counts, 1, len(pos_ref))
            result[pos_mask] = pos_counts / len(pos_ref)

    neg_mask = oric < 0
    if neg_mask.any():
        if neg_ref.size == 0:
            result[neg_mask] = 0.0
        else:
            neg_counts = np.searchsorted(neg_ref, oric[neg_mask], side="right")
            neg_counts = np.clip(neg_counts, 1, len(neg_ref))
            result[neg_mask] = (neg_counts - 1) / len(neg_ref) - 1.0

    return result


# ---------------------------------------------------------------------------
#  MORIC★ — sign-anchored uniform quantile (global CDF shifted by q₀)
# ---------------------------------------------------------------------------

def _fit_moric_star_reference(train_oric: np.ndarray) -> dict:
    """Fit MORIC★ reference: sorted ORIC for CDF + non-positive quantile q₀.

    MORIC★ = MORIC - q₀ where q₀ = fraction of non-positive ORIC values.
    This produces a **uniform** distribution with the sign boundary at zero
    and a dataset-adaptive range (-q₀, 1-q₀].
    """
    train_oric = np.asarray(train_oric, dtype=np.float64)
    if train_oric.size == 0:
        raise ValueError("Cannot fit MORIC★ reference on an empty training split.")
    sorted_oric = np.sort(train_oric)
    q0 = float(np.sum(train_oric <= 0)) / len(train_oric)
    return {"sorted_oric": sorted_oric, "q0": q0}


def _apply_moric_star(reference: dict, oric: np.ndarray) -> np.ndarray:
    """Apply MORIC★ = MORIC - q₀ (global CDF shifted to sign boundary)."""
    sorted_ref = reference["sorted_oric"]
    q0 = reference["q0"]
    oric = np.asarray(oric, dtype=np.float64)
    moric = np.searchsorted(sorted_ref, oric, side="right") / max(len(sorted_ref), 1)
    return (moric - q0).astype(np.float64)


# ---------------------------------------------------------------------------
#  Φ-MORIC — probit normal-scores transform (van der Waerden scores)
# ---------------------------------------------------------------------------

def _fit_phi_moric_reference(train_oric: np.ndarray) -> dict:
    """Fit Φ-MORIC reference: sorted ORIC + probit of q₀.

    Φ-MORIC = Φ⁻¹(MORIC) - Φ⁻¹(q₀) where Φ⁻¹ is the probit function.
    This produces an approximately **normal** distribution centered at the
    sign boundary, concentrating gradient near the offload/keep decision edge.
    """
    from scipy.stats import norm as _norm

    train_oric = np.asarray(train_oric, dtype=np.float64)
    if train_oric.size == 0:
        raise ValueError("Cannot fit Φ-MORIC reference on an empty training split.")
    sorted_oric = np.sort(train_oric)
    q0 = float(np.sum(train_oric <= 0)) / len(train_oric)
    probit_q0 = float(_norm.ppf(np.clip(q0, 1e-8, 1 - 1e-8)))
    return {"sorted_oric": sorted_oric, "q0": q0, "probit_q0": probit_q0}


def _apply_phi_moric(reference: dict, oric: np.ndarray) -> np.ndarray:
    """Apply Φ-MORIC = Φ⁻¹(MORIC) - Φ⁻¹(q₀) (van der Waerden normal scores)."""
    from scipy.stats import norm as _norm

    sorted_ref = reference["sorted_oric"]
    probit_q0 = reference["probit_q0"]
    N = max(len(sorted_ref), 1)
    oric = np.asarray(oric, dtype=np.float64)
    moric = np.searchsorted(sorted_ref, oric, side="right") / N
    # Clip to (1/(N+1), N/(N+1)) to avoid ±∞ at boundaries
    moric = np.clip(moric, 1.0 / (N + 1), N / (N + 1))
    return (_norm.ppf(moric) - probit_q0).astype(np.float64)


# ---------------------------------------------------------------------------
#  SigMORIC — sigmoid-scaled sign-anchored quantile (bounded, boundary-focused)
# ---------------------------------------------------------------------------

def _fit_sigmoric_reference(train_oric: np.ndarray, k: float = 4.0) -> dict:
    """Fit SigMORIC reference: sorted ORIC for CDF + sigmoid params.

    SigMORIC = 2σ(k(MORIC - q₀)) - 1 where σ is the logistic sigmoid.
    This produces a **bounded** (-1, 1) distribution with bell-shaped density
    concentrated at the sign boundary, combining the bounded range of MORIC+
    with the boundary focus of Φ-MORIC.
    """
    train_oric = np.asarray(train_oric, dtype=np.float64)
    if train_oric.size == 0:
        raise ValueError("Cannot fit SigMORIC reference on an empty training split.")
    sorted_oric = np.sort(train_oric)
    q0 = float(np.sum(train_oric <= 0)) / len(train_oric)
    return {"sorted_oric": sorted_oric, "q0": q0, "k": k}


def _apply_sigmoric(reference: dict, oric: np.ndarray) -> np.ndarray:
    """Apply SigMORIC = 2σ(k(MORIC - q₀)) - 1 (bounded sigmoid transform)."""
    sorted_ref = reference["sorted_oric"]
    q0 = reference["q0"]
    k = reference["k"]
    oric = np.asarray(oric, dtype=np.float64)
    moric = np.searchsorted(sorted_ref, oric, side="right") / max(len(sorted_ref), 1)
    # Clip to avoid exact 0/1 before sigmoid (though sigmoid is safe either way)
    u = k * (moric - q0)
    sigmoid = 1.0 / (1.0 + np.exp(-u))
    return (2.0 * sigmoid - 1.0).astype(np.float64)


def _random_frame_split_indices(
    n_samples: int,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic row-level holdout split."""
    if n_samples < 2:
        raise ValueError("Need at least two samples to create a train/test split.")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")

    n_test = int(round(n_samples * test_ratio))
    n_test = min(max(n_test, 1), n_samples - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def _resolve_split_indices(
    cfg: PipelineConfig,
    split_labels,
    video_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve train/test indices from native labels or a deterministic fallback."""
    if split_labels is not None:
        split_labels = np.asarray(split_labels).astype(str)
        nonempty_labels = {label for label in split_labels.tolist() if label}
        train_mask = split_labels == "train"
        test_mask = split_labels == "test"
        if train_mask.any() and test_mask.any():
            return np.flatnonzero(train_mask), np.flatnonzero(test_mask), "dataset_split"
        if nonempty_labels:
            present = ", ".join(sorted(nonempty_labels))
            raise ValueError(
                "Derived data contains native split labels but not both train and test "
                f"partitions (present: {present}). This usually means detection or "
                "derive outputs are incomplete for one native split."
            )

    video_names = np.asarray(video_names).astype(str)
    unique_videos = np.unique(video_names)
    if len(unique_videos) >= 2:
        rng = np.random.default_rng(cfg.seed)
        shuffled_videos = rng.permutation(unique_videos)
        n_test_videos = int(round(len(unique_videos) * cfg.dataset.test_ratio))
        n_test_videos = min(max(n_test_videos, 1), len(unique_videos) - 1)
        test_videos = set(shuffled_videos[:n_test_videos].tolist())
        test_mask = np.array([video in test_videos for video in video_names], dtype=bool)
        train_mask = ~test_mask
        if train_mask.any() and test_mask.any():
            return np.flatnonzero(train_mask), np.flatnonzero(test_mask), "video_holdout"

    train_idx, test_idx = _random_frame_split_indices(
        len(video_names), cfg.dataset.test_ratio, cfg.seed,
    )
    return train_idx, test_idx, "frame_holdout"


def _fit_lcer_beta(vectors: np.ndarray, rewards: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
    """Fit the LCER linear projection beta on the training split only."""
    from sklearn.linear_model import Ridge

    vectors = np.asarray(vectors, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != len(LCER_ERROR_TYPES):
        raise ValueError(
            f"Expected LCER vectors of shape [N,{len(LCER_ERROR_TYPES)}], "
            f"got {vectors.shape}"
        )
    if len(vectors) == 0 or not np.any(vectors):
        return np.zeros(vectors.shape[1], dtype=np.float64)

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(vectors, rewards)
    return np.asarray(ridge.coef_, dtype=np.float64)


def _build_csr_targets(train_primary: np.ndarray,
                       test_primary: np.ndarray,
                       tau_grid: np.ndarray = LCER_TAU_GRID) -> tuple[np.ndarray, np.ndarray]:
    """Build ordered CSR labels from the train-normalized primary reward."""
    train_primary = np.asarray(train_primary, dtype=np.float64)
    test_primary = np.asarray(test_primary, dtype=np.float64)
    mean = float(np.mean(train_primary))
    std = float(np.std(train_primary)) or 1.0
    train_norm = (train_primary - mean) / std
    test_norm = (test_primary - mean) / std
    train_targets = (train_norm[:, None] > tau_grid[None, :]).astype(np.float32)
    test_targets = (test_norm[:, None] > tau_grid[None, :]).astype(np.float32)
    return train_targets, test_targets


def _load_or_compute_split_embeddings(
    train_paths: list[str],
    test_paths: list[str],
    image_size: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_embeddings, test_embeddings) using a persistent per-dataset cache.

    The embeddings depend only on image content + image_size + (frozen) backbone
    weights, none of which change between runs.  The cache is keyed by the sorted
    unique path list so it survives seed / split changes.
    """
    unique_sorted = sorted(set(train_paths + test_paths))
    cache_path = cache_dir / f"frozen_embeddings_{image_size}.npz"

    embed_map: dict | None = None
    if cache_path.exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_paths = data["paths"].tolist()
            if cached_paths == unique_sorted:
                embed_map = {p: data["embeddings"][i] for i, p in enumerate(cached_paths)}
                log.cached(f"Frozen image embeddings ({len(cached_paths)} images, {cache_path.name})")
        except Exception as exc:
            log.warn(f"Embedding cache load failed ({exc}), recomputing ...")

    if embed_map is None:
        print("  Computing frozen image embeddings for conditional smoothing ...")
        embeddings = compute_frozen_image_embeddings(
            unique_sorted, image_size=image_size, quiet=False,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Keep .npz suffix on temp file so np.savez_compressed does not alter the path.
        tmp = cache_path.with_suffix(".tmp.npz")
        with tmp.open("wb") as fh:
            np.savez_compressed(
                fh,
                paths=np.array(unique_sorted),
                embeddings=embeddings,
            )
        tmp.replace(cache_path)
        embed_map = dict(zip(unique_sorted, embeddings))
        log.arrow(str(cache_path))

    train_embeddings = np.array([embed_map[p] for p in train_paths])
    test_embeddings = np.array([embed_map[p] for p in test_paths])
    return train_embeddings, test_embeddings


def _compute_conditional_reward_neighborhood(
    train_paths: list[str],
    test_paths: list[str],
    image_size: int = 128,
    cache_dir: Path | None = None,
):
    """Build the shared visual neighborhood used by conditional smoothers."""
    if cache_dir is not None:
        train_embeddings, test_embeddings = _load_or_compute_split_embeddings(
            train_paths, test_paths, image_size, cache_dir,
        )
    else:
        print("  Computing frozen image embeddings for conditional smoothing ...")
        train_embeddings = compute_frozen_image_embeddings(
            train_paths, image_size=image_size, quiet=False,
        )
        test_embeddings = compute_frozen_image_embeddings(
            test_paths, image_size=image_size, quiet=False,
        )
    return build_kernel_neighborhood(
        train_embeddings,
        test_embeddings=test_embeddings,
        k=CEORIC_DEFAULT_K,
        tau=CEORIC_DEFAULT_TAU,
    )


def _scenario_metadata_payload(
    metric_suffix: str,
    component_names: tuple[str, ...],
    scenario_weight_map: dict[str, np.ndarray],
    default_scenario_name: str,
    default_scenario_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Serialize scenario metadata for one target family suffix."""
    return {
        f"meta_scenario_component_names_{metric_suffix}": np.asarray(
            component_names, dtype=object
        ),
        f"meta_scenario_weight_names_{metric_suffix}": np.asarray(
            list(scenario_weight_map.keys()), dtype=object
        ),
        f"meta_scenario_weights_{metric_suffix}": np.stack(
            [scenario_weight_map[name] for name in scenario_weight_map],
            axis=0,
        ).astype(np.float32),
        f"meta_default_scenario_weights_{metric_suffix}": np.asarray(
            default_scenario_weights, dtype=np.float32
        ),
        f"meta_default_scenario_name_{metric_suffix}": np.asarray(
            default_scenario_name, dtype=object
        ),
        f"meta_tau_{metric_suffix}": LCER_TAU_GRID,
    }

