"""Offloading strategy evaluation.

Computes mAP at various offload ratios using estimator outputs.
Supports both top-K ranking and binary-mask (threshold-based) offloading.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def _percentile_normalize_confidences(dets: list) -> list:
    """Replace raw confidence with percentile rank (0, 1] within this set.

    This makes cross-model confidence scores comparable when detections from
    two models are combined for dataset-wide AP computation.  Within-model
    ordering is preserved, so pure edge-only or cloud-only AP is unchanged.

    Args:
        dets: List of (det_tuple, video_name, frame_id) where
              det_tuple = (left, top, width, height, conf, class).

    Returns:
        New list with the same structure but confidence replaced by rank / N.
    """
    if not dets:
        return dets
    confs = np.array([d[0][4] for d in dets])
    n = len(confs)
    order = np.argsort(confs)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    normalized = ranks / n
    return [
        ((d[0][0], d[0][1], d[0][2], d[0][3], normalized[i], d[0][5]),
         d[1], d[2])
        for i, d in enumerate(dets)
    ]


def _prepare_detection_data(raw_boxes: list) -> Tuple[Any, ...]:
    """Precompute IoU and detection metadata shared across offloading queries.

    Returns a tuple of:
      (valid_indices, precomputed, det_source, det_frame,
       frame_to_valid_idx, gt_by_frame, total_gt)
    or None if there is no valid data.
    """
    from ..metrics import (COCO_IOU_THRESHOLDS, _precompute_detection_ious,
                           _run_greedy_matching)

    if not raw_boxes:
        return None
    missing = [i for i, box in enumerate(raw_boxes) if box is None]
    if missing:
        raise ValueError(
            f"raw_boxes is missing {len(missing)} frame payload(s); offloading "
            "evaluation requires a complete prepared raw-box cache."
        )
    valid_indices = list(range(len(raw_boxes)))
    n_valid = len(valid_indices)

    all_edge_dets = []
    all_cloud_dets = []
    all_gts = []

    for i in valid_indices:
        frame_data = raw_boxes[i]
        v_name = frame_data.get('video_name', 'unknown')
        f_id = frame_data.get('frame_id', 0)
        for d in frame_data['edge_dets']:
            all_edge_dets.append((d, v_name, f_id))
        for d in frame_data['cloud_dets']:
            all_cloud_dets.append((d, v_name, f_id))
        for g in frame_data['gt_objs']:
            all_gts.append((g, v_name, f_id))

    total_gt = len(all_gts)
    if total_gt == 0:
        return None

    gt_by_frame: Dict = {}
    for gt_obj, video, frame in all_gts:
        key = (video, frame)
        if key not in gt_by_frame:
            gt_by_frame[key] = []
        gt_by_frame[key].append(gt_obj)

    all_edge_dets = _percentile_normalize_confidences(all_edge_dets)
    all_cloud_dets = _percentile_normalize_confidences(all_cloud_dets)

    all_dets_combined = all_edge_dets + all_cloud_dets
    sorted_dets = sorted(all_dets_combined, key=lambda x: x[0][4], reverse=True)

    precomputed = _precompute_detection_ious(sorted_dets, gt_by_frame)

    edge_set = set(id(x) for x in all_edge_dets)
    det_source = ['edge' if id(det) in edge_set else 'cloud' for det in sorted_dets]
    det_frame = [(video, frame) for det_tuple, video, frame in sorted_dets]

    frame_to_valid_idx: Dict = {}
    for i in valid_indices:
        fd = raw_boxes[i]
        key = (fd.get('video_name', 'unknown'), fd.get('frame_id', 0))
        frame_to_valid_idx[key] = i

    return (valid_indices, precomputed, det_source, det_frame,
            frame_to_valid_idx, gt_by_frame, total_gt)


def _eval_offload_set(
    offload_orig_indices: set,
    precomputed: list,
    det_source: list,
    det_frame: list,
    frame_to_valid_idx: dict,
    gt_by_frame: dict,
    total_gt: int,
) -> dict:
    """Compute dataset-wide AP for one offloading configuration.

    Args:
        offload_orig_indices: Set of original frame indices to offload.

    Returns:
        Dict with ap50, ap50_allpoint, ap_coco, n_offload.
    """
    from ..metrics import COCO_IOU_THRESHOLDS, _run_greedy_matching

    include_mask = []
    for src, (vid, frm) in zip(det_source, det_frame):
        frame_idx = frame_to_valid_idx.get((vid, frm))
        if frame_idx is None:
            include_mask.append(False)
            continue
        if src == 'edge' and frame_idx not in offload_orig_indices:
            include_mask.append(True)
        elif src == 'cloud' and frame_idx in offload_orig_indices:
            include_mask.append(True)
        else:
            include_mask.append(False)

    filtered = [precomputed[j] for j in range(len(precomputed)) if include_mask[j]]

    results_list = [_run_greedy_matching(filtered, gt_by_frame, total_gt, t)
                    for t in COCO_IOU_THRESHOLDS]
    ap50 = results_list[0]['ap']
    ap50_allpoint = results_list[0]['ap_allpoint']
    ap_coco = float(np.mean([r['ap'] for r in results_list]))

    return {
        'ap50': float(ap50),
        'ap50_allpoint': float(ap50_allpoint),
        'ap_coco': ap_coco,
        'n_offload': len(offload_orig_indices),
    }


def compute_map_at_ratio(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
    predictions: np.ndarray,
    offload_ratio: float,
) -> Tuple[float, int]:
    """Compute mAP when offloading top K% frames by predicted gain.
    
    Args:
        edge_maps: Edge model AP for each frame
        cloud_maps: Cloud model AP for each frame
        predictions: Predicted gain for each frame (higher = more gain from offloading)
        offload_ratio: Fraction of frames to offload (0.0 to 1.0)
        
    Returns:
        Tuple of (mAP, number of frames offloaded)
    """
    n_frames = len(edge_maps)
    n_offload = int(n_frames * offload_ratio)
    
    if n_offload == 0:
        return float(np.mean(edge_maps)), 0
    
    if n_offload >= n_frames:
        return float(np.mean(cloud_maps)), n_frames
    
    # Get indices of top K frames by predicted gain
    offload_indices = np.argsort(predictions)[-n_offload:]
    
    # Use cloud mAP for offloaded frames, edge mAP for others
    final_maps = edge_maps.copy()
    final_maps[offload_indices] = cloud_maps[offload_indices]
    
    return float(np.mean(final_maps)), n_offload


def compute_dataset_map_at_ratio(
    raw_boxes: list,
    predictions: np.ndarray,
    offload_ratio: float,
) -> dict:
    """Compute true dataset-wide AP when offloading top K% frames.
    
    For single-ratio calls. For multiple ratios, use compute_dataset_map_batch.
    """
    results = compute_dataset_map_batch(raw_boxes, predictions, [offload_ratio])
    return results[offload_ratio]


def compute_dataset_map_batch(
    raw_boxes: list,
    predictions: np.ndarray,
    offload_ratios: list,
) -> dict:
    """Compute true dataset-wide AP for multiple offload ratios efficiently.

    Precomputes IoU once for ALL edge+cloud detections, then for each ratio
    just selects which detections to include and runs greedy matching.

    Args:
        raw_boxes: List of dicts with 'edge_dets', 'cloud_dets', 'gt_objs' per frame
        predictions: Predicted gain for each frame
        offload_ratios: List of fractions to evaluate

    Returns:
        Dict mapping ratio -> {'ap50', 'ap50_allpoint', 'ap_coco', 'n_offload'}
    """
    empty = lambda: {'ap50': 0.0, 'ap50_allpoint': 0.0, 'ap_coco': 0.0, 'n_offload': 0}

    prep = _prepare_detection_data(raw_boxes)
    if prep is None:
        return {r: empty() for r in offload_ratios}

    (valid_indices, precomputed, det_source, det_frame,
     frame_to_valid_idx, gt_by_frame, total_gt) = prep

    n_valid = len(valid_indices)
    valid_preds = predictions[valid_indices]

    results = {}
    for ratio in offload_ratios:
        n_offload = int(n_valid * ratio)
        if n_offload > 0:
            top_k = np.argsort(valid_preds)[-n_offload:]
            offload_orig_indices = {valid_indices[k] for k in top_k}
        else:
            offload_orig_indices = set()

        results[ratio] = _eval_offload_set(
            offload_orig_indices, precomputed, det_source, det_frame,
            frame_to_valid_idx, gt_by_frame, total_gt,
        )

    return results


def compute_dataset_map_batch_from_masks(
    raw_boxes: list,
    masks_dict: Dict[Any, np.ndarray],
) -> Dict[Any, dict]:
    """Compute dataset-wide AP for multiple binary offloading masks.

    Shares IoU precomputation across all masks (same optimisation as
    :func:`compute_dataset_map_batch` shares across ratios).

    Args:
        raw_boxes: Per-frame detection data (same format as other functions).
        masks_dict: ``{label: bool_array}`` — one boolean mask per query.
            Each mask has length ``len(raw_boxes)``; ``True`` = offload.

    Returns:
        Dict mapping label -> {'ap50', 'ap50_allpoint', 'ap_coco', 'n_offload'}.
    """
    empty = lambda: {'ap50': 0.0, 'ap50_allpoint': 0.0, 'ap_coco': 0.0, 'n_offload': 0}

    prep = _prepare_detection_data(raw_boxes)
    if prep is None:
        return {label: empty() for label in masks_dict}

    (valid_indices, precomputed, det_source, det_frame,
     frame_to_valid_idx, gt_by_frame, total_gt) = prep

    valid_set = set(valid_indices)

    results = {}
    for label, mask in masks_dict.items():
        mask = np.asarray(mask, dtype=bool)
        offload_orig_indices = {i for i in range(len(mask)) if mask[i] and i in valid_set}

        results[label] = _eval_offload_set(
            offload_orig_indices, precomputed, det_source, det_frame,
            frame_to_valid_idx, gt_by_frame, total_gt,
        )

    return results


def compute_map_with_threshold(
    edge_maps: np.ndarray,
    cloud_maps: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute mAP when offloading frames above a prediction threshold.
    
    Args:
        edge_maps: Edge model AP for each frame
        cloud_maps: Cloud model AP for each frame
        predictions: Predicted gain for each frame
        threshold: Minimum predicted gain to trigger offloading
        
    Returns:
        Tuple of (mAP, actual offload ratio)
    """
    offload_mask = predictions > threshold
    n_offload = np.sum(offload_mask)
    
    final_maps = edge_maps.copy()
    final_maps[offload_mask] = cloud_maps[offload_mask]
    
    return float(np.mean(final_maps)), float(n_offload / len(edge_maps))


class OffloadingEvaluator:
    """Evaluator for offloading strategies.
    
    Computes mAP across different offload ratios and compares to baselines.
    """
    
    def __init__(
        self,
        edge_maps: np.ndarray,
        cloud_maps: np.ndarray,
        offload_ratios: List[float] = None,
    ):
        """Initialize evaluator.
        
        Args:
            edge_maps: Edge model AP per frame
            cloud_maps: Cloud model AP per frame
            offload_ratios: Ratios to evaluate (default: [0.2, 0.4, 0.6, 0.8])
        """
        self.edge_maps = np.asarray(edge_maps)
        self.cloud_maps = np.asarray(cloud_maps)
        self.offload_ratios = offload_ratios or [0.2, 0.4, 0.6, 0.8]
        
        # Precompute baseline metrics
        self._baselines = None
        
    @property
    def baselines(self) -> Dict[str, float]:
        """Get cached baseline metrics."""
        if self._baselines is None:
            from .baselines import compute_baselines
            self._baselines = compute_baselines(self.edge_maps, self.cloud_maps)
        return self._baselines
    
    def evaluate_estimator(
        self,
        predictions: np.ndarray,
        estimator_name: str = "estimator",
    ) -> Dict[str, float]:
        """Evaluate a estimator's offloading performance.
        
        Args:
            predictions: Predicted gain per frame
            estimator_name: Name for result keys
            
        Returns:
            Dict with mAP at each ratio and relative metrics
        """
        results = {}
        
        for ratio in self.offload_ratios:
            mAP, n_offload = compute_map_at_ratio(
                self.edge_maps, self.cloud_maps, predictions, ratio
            )
            results[f'map_at_{int(ratio*100)}pct'] = mAP
            
            # Compare to oracle at same ratio
            from .baselines import compute_oracle_at_ratio
            oracle_map = compute_oracle_at_ratio(
                self.edge_maps, self.cloud_maps, ratio
            )
            results[f'oracle_gap_{int(ratio*100)}pct'] = oracle_map - mAP
        
        # Add overall metrics
        results['edge_only_map'] = self.baselines['edge_only']
        results['cloud_only_map'] = self.baselines['cloud_only']
        results['oracle_map'] = self.baselines['oracle_unconstrained']
        
        return results
    
    def compare_estimators(
        self,
        predictions_dict: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple estimators.
        
        Args:
            predictions_dict: Dict mapping estimator name to predictions
            
        Returns:
            Nested dict of results per estimator
        """
        results = {}
        for name, predictions in predictions_dict.items():
            results[name] = self.evaluate_estimator(predictions, name)
        return results
