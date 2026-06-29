"""Baseline offloading strategies.

Provides comparison baselines for estimator evaluation:
- Always offload (cloud only)
- Never offload (edge only)
- Random offload
- Oracle (perfect knowledge)
"""

from typing import Dict, List

import numpy as np


def compute_baselines(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
) -> Dict[str, float]:
    """Compute all baseline metrics.
    
    Args:
        edge_maps: Edge model AP per frame
        cloud_maps: Cloud model AP per frame
        
    Returns:
        Dict with baseline mAP values
    """
    edge_maps = np.asarray(edge_maps)
    cloud_maps = np.asarray(cloud_maps)
    
    # Edge only (never offload)
    edge_only = float(np.mean(edge_maps))
    
    # Cloud only (always offload)
    cloud_only = float(np.mean(cloud_maps))
    
    # Oracle: offload whenever cloud is better
    oracle_choices = np.maximum(edge_maps, cloud_maps)
    oracle_unconstrained = float(np.mean(oracle_choices))
    
    # Compute gain stats
    gains = cloud_maps - edge_maps
    positive_gain_ratio = float(np.mean(gains > 0))
    mean_gain = float(np.mean(gains))
    
    return {
        'edge_only': edge_only,
        'cloud_only': cloud_only,
        'oracle_unconstrained': oracle_unconstrained,
        'positive_gain_ratio': positive_gain_ratio,
        'mean_gain': mean_gain,
    }


def compute_oracle_at_ratio(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
    offload_ratio: float,
) -> float:
    """Compute oracle mAP at a fixed offload ratio.
    
    Oracle = perfect knowledge. Offloads the top K% frames sorted by actual gain.
    This is the best possible selection at this budget constraint.
    
    Args:
        edge_maps: Edge model AP per frame
        cloud_maps: Cloud model AP per frame
        offload_ratio: Fraction of frames to offload
        
    Returns:
        Oracle mAP at the given ratio
    """
    edge_maps = np.asarray(edge_maps)
    cloud_maps = np.asarray(cloud_maps)
    
    n_frames = len(edge_maps)
    n_offload = int(n_frames * offload_ratio)
    
    if n_offload == 0:
        return float(np.mean(edge_maps))
    
    if n_offload >= n_frames:
        return float(np.mean(cloud_maps))
    
    # Oracle: offload frames with highest actual gain
    actual_gains = cloud_maps - edge_maps
    offload_indices = np.argsort(actual_gains)[-n_offload:]
    
    final_maps = edge_maps.copy()
    final_maps[offload_indices] = cloud_maps[offload_indices]
    
    return float(np.mean(final_maps))


def compute_random_at_ratio(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
    offload_ratio: float,
    n_trials: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute average mAP for random offloading at given ratio.
    
    Args:
        edge_maps: Edge model AP per frame
        cloud_maps: Cloud model AP per frame
        offload_ratio: Fraction of frames to offload
        n_trials: Number of random trials for averaging
        seed: Random seed for reproducibility
        
    Returns:
        Dict with mean, std, min, max mAP over trials
    """
    edge_maps = np.asarray(edge_maps)
    cloud_maps = np.asarray(cloud_maps)
    
    n_frames = len(edge_maps)
    n_offload = int(n_frames * offload_ratio)
    
    rng = np.random.RandomState(seed)
    trial_maps = []
    
    for _ in range(n_trials):
        offload_mask = np.zeros(n_frames, dtype=bool)
        offload_indices = rng.choice(n_frames, size=n_offload, replace=False)
        offload_mask[offload_indices] = True
        
        final_maps = np.where(offload_mask, cloud_maps, edge_maps)
        trial_maps.append(np.mean(final_maps))
    
    trial_maps = np.array(trial_maps)
    
    return {
        'mean': float(np.mean(trial_maps)),
        'std': float(np.std(trial_maps)),
        'min': float(np.min(trial_maps)),
        'max': float(np.max(trial_maps)),
    }


def compute_all_baselines_at_ratios(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
    ratios: List[float] = None,
    n_random_trials: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Compute all baselines at multiple offload ratios.
    
    Args:
        edge_maps: Edge model AP per frame
        cloud_maps: Cloud model AP per frame
        ratios: Offload ratios to evaluate
        n_random_trials: Trials for random baseline
        
    Returns:
        Nested dict: baseline_type -> ratio -> value
    """
    if ratios is None:
        ratios = [0.2, 0.4, 0.6, 0.8]
    
    results = {
        'oracle': {},
        'random_mean': {},
        'random_std': {},
    }
    
    for ratio in ratios:
        ratio_key = f'{int(ratio * 100)}pct'
        
        results['oracle'][ratio_key] = compute_oracle_at_ratio(
            edge_maps, cloud_maps, ratio
        )
        
        random_results = compute_random_at_ratio(
            edge_maps, cloud_maps, ratio, n_random_trials
        )
        results['random_mean'][ratio_key] = random_results['mean']
        results['random_std'][ratio_key] = random_results['std']
    
    # Add fixed baselines
    base = compute_baselines(edge_maps, cloud_maps)
    results['edge_only'] = base['edge_only']
    results['cloud_only'] = base['cloud_only']
    
    return results
