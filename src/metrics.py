"""Compute detection metrics (mAP, precision, recall) for video frames."""

from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO-style IoU thresholds: 0.50, 0.55, ..., 0.95 (10 thresholds)
COCO_IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05).tolist()


def compute_iou(bbox1: Tuple[float, float, float, float], 
                bbox2: Tuple[float, float, float, float]) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes in format (left, top, width, height)
        
    Returns:
        IoU score (0 to 1)
    """
    # Convert to (x1, y1, x2, y2)
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Compute union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(detections: List[Tuple], 
                          gt_objects: List[Dict],
                          iou_threshold: float = 0.5) -> Tuple[int, int, int]:
    """
    Match detections to ground truth objects using IoU threshold.
    
    Args:
        detections: List of (left, top, width, height, conf, class)
        gt_objects: List of GT dicts with 'bbox' and 'class'
        iou_threshold: IoU threshold for positive match
        
    Returns:
        (true_positives, false_positives, false_negatives)
    """
    if len(detections) == 0 and len(gt_objects) == 0:
        return 0, 0, 0
    
    if len(detections) == 0:
        return 0, 0, len(gt_objects)
    
    if len(gt_objects) == 0:
        return 0, len(detections), 0
    
    # Sort detections by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    
    # Track which GT objects have been matched
    gt_matched = [False] * len(gt_objects)
    tp = 0
    fp = 0
    
    for det in sorted_dets:
        det_bbox = det[:4]  # (left, top, width, height)
        det_class = det[5]
        
        # Find best matching GT object
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_matched[gt_idx]:
                continue
            
            # Check class match (map DETRAC classes to COCO)
            gt_class = gt_obj['class']
            if not classes_match(det_class, gt_class):
                continue
            
            iou = compute_iou(det_bbox, gt_obj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is valid
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    # Count unmatched GT objects as false negatives
    fn = sum(1 for matched in gt_matched if not matched)
    
    return tp, fp, fn


# Vehicle classes for unified matching
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'van', 'motorcycle', 'others'}

# VOC↔COCO name synonyms (differing names for the same class)
_NAME_SYNONYMS = {
    'aeroplane': 'airplane',
    'airplane': 'aeroplane',
    'motorbike': 'motorcycle',
    'motorcycle': 'motorbike',
    'diningtable': 'dining table',
    'dining table': 'diningtable',
    'pottedplant': 'potted plant',
    'potted plant': 'pottedplant',
    'sofa': 'couch',
    'couch': 'sofa',
    'tvmonitor': 'tv',
    'tv': 'tvmonitor',
}


def classes_match(det_class: str, gt_class: str) -> bool:
    """
    Check if a detection class and a ground-truth class refer to the same
    object category.

    Matching rules (evaluated in order):
    1. Exact (case-insensitive) match.
    2. VOC ↔ COCO synonym match (e.g. 'aeroplane' == 'airplane').
    3. Vehicle-class match — all vehicle types (car, bus, truck, van,
       motorcycle, others) are treated as equivalent, which is necessary
       for UA-DETRAC where GT uses 'van'/'others' but detectors output
       COCO class names.
    """
    d = det_class.lower()
    g = gt_class.lower()

    # 1. Exact match
    if d == g:
        return True

    # 2. Known synonym
    if _NAME_SYNONYMS.get(g) == d or _NAME_SYNONYMS.get(d) == g:
        return True

    # 3. Vehicle-class match (DETRAC ↔ COCO)
    if d in VEHICLE_CLASSES and g in VEHICLE_CLASSES:
        return True

    return False


def compute_11point_ap(detections: List[Tuple],
                       gt_objects: List[Dict],
                       iou_threshold: float = 0.5) -> float:
    """
    Compute Average Precision using 11-point interpolation (VOC-style).
    
    This computes precision at 11 recall thresholds (0, 0.1, ..., 1.0)
    and averages them.
    
    Args:
        detections: List of (left, top, width, height, conf, class)
        gt_objects: List of GT dicts with 'bbox' and 'class'
        iou_threshold: IoU threshold for positive match
        
    Returns:
        11-point interpolated Average Precision
    """
    if len(gt_objects) == 0:
        # No GT objects: if no detections either, AP is trivially 0 (nothing to evaluate).
        # If there are detections, they're all false positives → AP = 0.
        return 0.0
    
    if len(detections) == 0:
        return 0.0
    
    # Sort detections by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    
    # Track which GT objects have been matched
    gt_matched = [False] * len(gt_objects)
    
    # Compute precision and recall at each detection threshold
    precisions = []
    recalls = []
    tp_cumsum = 0
    fp_cumsum = 0
    
    for det in sorted_dets:
        det_bbox = det[:4]
        det_class = det[5]
        
        # Find best matching GT object
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_matched[gt_idx]:
                continue
            
            gt_class = gt_obj['class']
            if not classes_match(det_class, gt_class):
                continue
            
            iou = compute_iou(det_bbox, gt_obj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is valid
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_cumsum += 1
            gt_matched[best_gt_idx] = True
        else:
            fp_cumsum += 1
        
        # Record precision and recall at this point
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(gt_objects)
        precisions.append(precision)
        recalls.append(recall)
    
    # 11-point interpolation
    ap = 0.0
    for r_thresh in np.arange(0, 1.1, 0.1):
        # Find max precision at recall >= r_thresh
        precs_at_recall = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
        if precs_at_recall:
            ap += max(precs_at_recall) / 11
    
    return ap


def compute_allpoint_ap(detections: List[Tuple],
                       gt_objects: List[Dict],
                       iou_threshold: float = 0.5) -> float:
    """
    Compute Average Precision using all-point interpolation (COCO-style).

    Builds the full precision-recall curve, applies monotonic-decreasing
    envelope, and integrates (AUC).  This differs from the VOC 11-point
    method and generally produces slightly different AP values.

    Args:
        detections: List of (left, top, width, height, conf, class)
        gt_objects: List of GT dicts with 'bbox' and 'class'
        iou_threshold: IoU threshold for positive match

    Returns:
        All-point interpolated Average Precision
    """
    if not gt_objects or not detections:
        return 0.0

    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    gt_matched = [False] * len(gt_objects)
    n_gt = len(gt_objects)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    for det in sorted_dets:
        det_bbox = det[:4]
        det_class = det[5]
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_matched[gt_idx]:
                continue
            if not classes_match(det_class, gt_obj['class']):
                continue
            iou = compute_iou(det_bbox, gt_obj['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp_cumsum += 1
            gt_matched[best_gt_idx] = True
        else:
            fp_cumsum += 1

        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / n_gt)

    # Prepend (recall=0, precision=1) sentinel
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    # Make precision monotonically decreasing (right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate AUC (trapezoidal with rectangular steps)
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_coco_frame_ap(detections: List[Tuple],
                         gt_objects: List[Dict]) -> float:
    """
    Compute COCO-style AP for a single frame.
    
    Averages 11-point AP across COCO IoU thresholds (0.5:0.05:0.95).
    
    Returns:
        COCO-style AP (average across 10 IoU thresholds)
    """
    if len(gt_objects) == 0 or len(detections) == 0:
        return 0.0
    
    aps = [compute_11point_ap(detections, gt_objects, t)
           for t in COCO_IOU_THRESHOLDS]
    return float(np.mean(aps))


def compute_precision_recall_ap(detections: List[Tuple],
                                gt_objects: List[Dict],
                                iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and average precision for a frame.
    
    Args:
        detections: List of (left, top, width, height, conf, class)
        gt_objects: List of GT dicts
        iou_threshold: IoU threshold for matching
        
    Returns:
        (precision, recall, average_precision)
    """
    tp, fp, fn = match_detections_to_gt(detections, gt_objects, iou_threshold)
    
    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Use 11-point interpolation for proper AP calculation
    average_precision = compute_11point_ap(detections, gt_objects, iou_threshold)
    
    return precision, recall, average_precision


def compute_f1_score(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_all_metrics(detections: List[Tuple],
                        gt_objects: List[Dict],
                        iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute all detection metrics for a frame.
    
    Returns a dict with: precision, recall, f1_score, ap_11point
    These can all be used as proxy metrics for model selection.
    
    Args:
        detections: List of (left, top, width, height, conf, class)
        gt_objects: List of GT dicts
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dict with all metrics
    """
    tp, fp, fn = match_detections_to_gt(detections, gt_objects, iou_threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = compute_f1_score(precision, recall)
    ap_11point = compute_11point_ap(detections, gt_objects, iou_threshold)
    ap_coco = compute_coco_frame_ap(detections, gt_objects)
    ap_coco50 = compute_allpoint_ap(detections, gt_objects, iou_threshold=0.5)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'ap_11point': ap_11point,
        'ap_coco': ap_coco,
        'ap_coco50': ap_coco50,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_map(detections: List[Tuple],
               gt_objects: List[Dict],
               iou_thresholds: List[float] = None) -> float:
    """
    Compute mean Average Precision across multiple IoU thresholds.
    
    Args:
        detections: List of detections
        gt_objects: List of GT objects
        iou_thresholds: IoU thresholds to average over (default: [0.5])
        
    Returns:
        mAP score
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    
    aps = []
    for iou_thresh in iou_thresholds:
        _, _, ap = compute_precision_recall_ap(detections, gt_objects, iou_thresh)
        aps.append(ap)
    
    return np.mean(aps)


def _compute_iou_vectorized(det_bbox, gt_bboxes: np.ndarray) -> np.ndarray:
    """Vectorized IoU: one detection vs N ground truths.

    Args:
        det_bbox: (left, top, width, height)
        gt_bboxes: np.array shape (N, 4) — (left, top, width, height)

    Returns:
        np.array shape (N,) with IoU values
    """
    x1_1, y1_1, w1, h1 = det_bbox[:4]
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2 = gt_bboxes[:, 0]
    y1_2 = gt_bboxes[:, 1]
    x2_2 = x1_2 + gt_bboxes[:, 2]
    y2_2 = y1_2 + gt_bboxes[:, 3]

    intersection = (np.maximum(0, np.minimum(x2_1, x2_2) - np.maximum(x1_1, x1_2))
                    * np.maximum(0, np.minimum(y2_1, y2_2) - np.maximum(y1_1, y1_2)))
    area1 = w1 * h1
    area2 = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    union = area1 + area2 - intersection
    return np.where(union > 0, intersection / union, 0.0)


import concurrent.futures
import os
import resource
import time

_SHARED_BBOX_CACHE = None
_SHARED_GT_CLASSES = None

def _iou_worker_init(bbox_cache, gt_classes):
    """Initialize global references in the worker processes."""
    global _SHARED_BBOX_CACHE, _SHARED_GT_CLASSES
    _SHARED_BBOX_CACHE = bbox_cache
    _SHARED_GT_CLASSES = gt_classes

def _process_chunk_for_iou(chunk):
    """Worker function to process a chunk of detections."""
    precomputed = []
    for det_tuple, video, frame in chunk:
        key = (video, frame)
        gt_cls_list = _SHARED_GT_CLASSES.get(key)
        if not gt_cls_list:
            precomputed.append((key, None, None))
            continue

        det_bbox = det_tuple[:4]
        det_class = det_tuple[5]
        gt_bboxes = _SHARED_BBOX_CACHE[key]
        ious = _compute_iou_vectorized(det_bbox, gt_bboxes)
        class_ok = np.array([classes_match(det_class, g_cls) for g_cls in gt_cls_list])
        ious[~class_ok] = -1.0
        precomputed.append((key, ious, class_ok))
    return precomputed


import multiprocessing


def _safe_iou_worker_count(desired_workers: int) -> int:
    """Choose a worker count that respects current FD limits.

    ProcessPoolExecutor needs multiple pipes/sockets per worker. When the
    process is already close to RLIMIT_NOFILE, spawning many workers can fail
    with Errno 24.
    """
    if desired_workers <= 1:
        return 1

    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit <= 0 or soft_limit == resource.RLIM_INFINITY:
            return desired_workers

        try:
            in_use = len(os.listdir('/proc/self/fd'))
        except Exception:
            in_use = 0

        reserve_fds = 128
        fd_per_worker = 32
        available = max(0, int(soft_limit) - in_use - reserve_fds)
        fd_cap = max(1, available // fd_per_worker)
        return max(1, min(desired_workers, fd_cap))
    except Exception:
        return desired_workers

def _precompute_detection_ious(sorted_dets, gt_by_frame):
    """Precompute IoU for each detection against GT in its frame."""
    t0 = time.time()
    # Cache numpy bbox arrays per frame
    _bbox_cache = {}
    _gt_classes = {}
    for key, gts in gt_by_frame.items():
        _bbox_cache[key] = np.array([g['bbox'] for g in gts])
        _gt_classes[key] = [g['class'] for g in gts]

    if len(sorted_dets) < 1000 or multiprocessing.current_process().daemon:
        _iou_worker_init(_bbox_cache, _gt_classes)
        res = _process_chunk_for_iou(sorted_dets)
        return res

    num_workers = min(16, (os.cpu_count() or 1) + 4)
    num_workers = _safe_iou_worker_count(num_workers)

    if num_workers <= 1:
        _iou_worker_init(_bbox_cache, _gt_classes)
        return _process_chunk_for_iou(sorted_dets)

    chunk_size = max(1, len(sorted_dets) // (num_workers * 2))
    chunks = [sorted_dets[i:i + chunk_size] for i in range(0, len(sorted_dets), chunk_size)]

    precomputed = []
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_iou_worker_init,
            initargs=(_bbox_cache, _gt_classes)
        ) as executor:
            futures = [executor.submit(_process_chunk_for_iou, chunk) for chunk in chunks]
            for future in futures:
                precomputed.extend(future.result())
    except OSError as e:
        if e.errno == 24:
            _iou_worker_init(_bbox_cache, _gt_classes)
            return _process_chunk_for_iou(sorted_dets)
        raise
            
    return precomputed


def _run_greedy_matching(precomputed, gt_by_frame, total_gt, iou_threshold):
    """Run greedy matching at one IoU threshold using precomputed IoUs."""
    # We won't time every single run to avoid log spam, but we can if needed.
    gt_matched = {key: np.zeros(len(gts), dtype=bool) for key, gts in gt_by_frame.items()}
    n_dets = len(precomputed)
    precisions = np.empty(n_dets)
    recalls = np.empty(n_dets)
    tp_cumsum = 0
    fp_cumsum = 0

    for i, (key, ious, class_ok) in enumerate(precomputed):
        matched_tp = False
        if ious is not None:
            matched = gt_matched.get(key)
            if matched is not None:
                valid = class_ok & ~matched
                if valid.any():
                    masked = np.where(valid, ious, -1.0)
                    best_idx = int(np.argmax(masked))
                    if masked[best_idx] >= iou_threshold:
                        tp_cumsum += 1
                        matched[best_idx] = True
                        matched_tp = True
        if not matched_tp:
            fp_cumsum += 1
        precisions[i] = tp_cumsum / (tp_cumsum + fp_cumsum)
        recalls[i] = tp_cumsum / total_gt

    # 11-point interpolation
    ap_11pt = 0.0
    for r_thresh in np.arange(0, 1.1, 0.1):
        mask = recalls >= r_thresh
        if mask.any():
            ap_11pt += float(precisions[mask].max()) / 11

    # All-point interpolation (COCO-style: monotonic envelope + AUC)
    r_all = np.concatenate(([0.0], recalls))
    p_all = np.concatenate(([1.0], precisions))
    for i in range(len(p_all) - 2, -1, -1):
        p_all[i] = max(p_all[i], p_all[i + 1])
    ap_allpoint = 0.0
    for i in range(1, len(r_all)):
        ap_allpoint += (r_all[i] - r_all[i - 1]) * p_all[i]

    total_tp = tp_cumsum
    total_fp = fp_cumsum
    total_fn = total_gt - total_tp
    final_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    final_rec = total_tp / total_gt if total_gt > 0 else 0.0
    return {
        'precision': final_prec,
        'recall': final_rec,
        'f1_score': compute_f1_score(final_prec, final_rec),
        'ap': ap_11pt,
        'ap_allpoint': float(ap_allpoint),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }


def _empty_result(n_det, n_gt):
    """Return a metrics dict for trivial edge cases."""
    if n_gt == 0:
        v = 1.0 if n_det == 0 else 0.0
        return {'precision': v, 'recall': 1.0, 'f1_score': v, 'ap': v,
                'ap_allpoint': v,
                'total_tp': 0, 'total_fp': n_det, 'total_fn': 0}
    return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ap': 0.0,
            'ap_allpoint': 0.0,
            'total_tp': 0, 'total_fp': 0, 'total_fn': n_gt}


def _compute_dataset_ap_single(all_detections, all_gt_objects, iou_threshold=0.5):
    """Compute dataset-wide AP at a single IoU threshold (optimized)."""
    if not all_gt_objects or not all_detections:
        return _empty_result(len(all_detections), len(all_gt_objects))

    sorted_dets = sorted(all_detections, key=lambda x: x[0][4], reverse=True)
    gt_by_frame = {}
    for gt_obj, video, frame in all_gt_objects:
        key = (video, frame)
        if key not in gt_by_frame:
            gt_by_frame[key] = []
        gt_by_frame[key].append(gt_obj)

    precomputed = _precompute_detection_ious(sorted_dets, gt_by_frame)
    return _run_greedy_matching(precomputed, gt_by_frame, len(all_gt_objects), iou_threshold)


def compute_dataset_ap(all_detections: List[Tuple[Tuple, str, int]],
                       all_gt_objects: List[Tuple[Dict, str, int]],
                       iou_threshold: float = 0.5,
                       iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute dataset-wide Average Precision (global PR curve).

    When *iou_thresholds* is given (COCO-style), IoU is precomputed once and
    matching is run per threshold — ~10× faster than recomputing from scratch.
    """
    if not all_gt_objects or not all_detections:
        return _empty_result(len(all_detections), len(all_gt_objects))

    sorted_dets = sorted(all_detections, key=lambda x: x[0][4], reverse=True)
    gt_by_frame = {}
    for gt_obj, video, frame in all_gt_objects:
        key = (video, frame)
        if key not in gt_by_frame:
            gt_by_frame[key] = []
        gt_by_frame[key].append(gt_obj)
    total_gt = len(all_gt_objects)

    # Precompute IoU once (the expensive part)
    precomputed = _precompute_detection_ious(sorted_dets, gt_by_frame)

    if iou_thresholds is not None:
        results_list = [_run_greedy_matching(precomputed, gt_by_frame, total_gt, t)
                        for t in iou_thresholds]
        avg_ap = float(np.mean([r['ap'] for r in results_list]))
        base = results_list[0]
        base['ap'] = avg_ap
        return base
    else:
        return _run_greedy_matching(precomputed, gt_by_frame, total_gt, iou_threshold)

