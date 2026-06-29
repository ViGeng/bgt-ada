"""Typed detection-error decomposition for LCER targets.

Provides a lightweight, TIDE-inspired categorisation of per-image errors
into six mutually exclusive buckets:
  cls, loc, both, dup, bg, miss

This module intentionally works on per-image detections and ground truth.
The downstream LCER scalar is obtained by fitting a linear projection
against contextual reward labels, so these typed counts act as the
decomposed feature space rather than exact per-type AP deltas.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .metrics import classes_match, compute_iou

LCER_ERROR_TYPES = ("cls", "loc", "both", "dup", "bg", "miss")
LCER_ERROR_INDEX = {
    name: idx for idx, name in enumerate(LCER_ERROR_TYPES)
}
LCER_OVERLAP_FLOOR = 0.1


def _best_iou(det: tuple, gt_objects: Sequence[dict],
              mask: Iterable[bool] | None = None) -> float:
    """Best IoU between a detection and a masked GT subset."""
    best = 0.0
    det_bbox = det[:4]
    if mask is None:
        mask = [True] * len(gt_objects)
    for enabled, gt_obj in zip(mask, gt_objects):
        if not enabled:
            continue
        best = max(best, compute_iou(det_bbox, gt_obj["bbox"]))
    return best


def compute_error_type_counts(
    detections: Sequence[tuple],
    gt_objects: Sequence[dict],
    iou_threshold: float = 0.5,
    overlap_floor: float = LCER_OVERLAP_FLOOR,
) -> np.ndarray:
    """Count typed errors for one image at a single IoU threshold."""
    counts = np.zeros(len(LCER_ERROR_TYPES), dtype=np.float64)
    if not gt_objects:
        if detections:
            counts[LCER_ERROR_INDEX["bg"]] = float(len(detections))
        return counts

    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    gt_matched = [False] * len(gt_objects)

    for det in sorted_dets:
        det_class = det[5]
        same_unmatched = []
        same_any = []
        wrong_unmatched = []
        wrong_any = []

        for idx, gt_obj in enumerate(gt_objects):
            iou = compute_iou(det[:4], gt_obj["bbox"])
            if iou <= 0.0:
                continue
            class_match = classes_match(det_class, gt_obj["class"])
            if class_match:
                same_any.append((idx, iou))
                if not gt_matched[idx]:
                    same_unmatched.append((idx, iou))
            else:
                wrong_any.append((idx, iou))
                if not gt_matched[idx]:
                    wrong_unmatched.append((idx, iou))

        # TP if the best unmatched same-class GT crosses the IoU threshold.
        if same_unmatched:
            best_idx, best_iou = max(same_unmatched, key=lambda x: x[1])
            if best_iou >= iou_threshold:
                gt_matched[best_idx] = True
                continue

        best_same_any = max((iou for _idx, iou in same_any), default=0.0)
        best_wrong_any = max((iou for _idx, iou in wrong_any), default=0.0)

        if best_same_any >= iou_threshold:
            counts[LCER_ERROR_INDEX["dup"]] += 1.0
        elif best_wrong_any >= iou_threshold:
            counts[LCER_ERROR_INDEX["cls"]] += 1.0
        elif best_same_any >= overlap_floor:
            counts[LCER_ERROR_INDEX["loc"]] += 1.0
        elif best_wrong_any >= overlap_floor:
            counts[LCER_ERROR_INDEX["both"]] += 1.0
        else:
            counts[LCER_ERROR_INDEX["bg"]] += 1.0

    counts[LCER_ERROR_INDEX["miss"]] = float(sum(1 for matched in gt_matched
                                                  if not matched))
    return counts


def compute_error_type_counts_multi(
    detections: Sequence[tuple],
    gt_objects: Sequence[dict],
    thresholds: Sequence[float],
    overlap_floor: float = LCER_OVERLAP_FLOOR,
) -> np.ndarray:
    """Average typed error counts over multiple IoU thresholds."""
    if not thresholds:
        return compute_error_type_counts(
            detections, gt_objects, overlap_floor=overlap_floor
        )
    stacked = [
        compute_error_type_counts(
            detections, gt_objects,
            iou_threshold=float(threshold),
            overlap_floor=overlap_floor,
        )
        for threshold in thresholds
    ]
    return np.mean(np.stack(stacked, axis=0), axis=0)


def compute_lcer_delta_vectors(
    edge_dets: Sequence[tuple],
    cloud_dets: Sequence[tuple],
    gt_objects: Sequence[dict],
) -> dict[str, np.ndarray]:
    """Compute weak-minus-strong typed error deltas for all metric families."""
    edge_50 = compute_error_type_counts(edge_dets, gt_objects, iou_threshold=0.5)
    cloud_50 = compute_error_type_counts(cloud_dets, gt_objects, iou_threshold=0.5)
    edge_coco = compute_error_type_counts_multi(
        edge_dets, gt_objects,
        thresholds=np.arange(0.5, 1.0, 0.05).tolist(),
    )
    cloud_coco = compute_error_type_counts_multi(
        cloud_dets, gt_objects,
        thresholds=np.arange(0.5, 1.0, 0.05).tolist(),
    )
    return {
        "11pt": edge_50 - cloud_50,
        "allpoint": edge_50 - cloud_50,
        "coco": edge_coco - cloud_coco,
    }


def compute_lcer_vectors(raw_boxes: Sequence[dict]) -> dict[int, dict[str, np.ndarray]]:
    """Compute LCER delta vectors for every frame in a raw-box sequence."""
    vectors: dict[int, dict[str, np.ndarray]] = {}
    for frame_data in raw_boxes:
        fid = int(frame_data["frame_id"])
        vectors[fid] = compute_lcer_delta_vectors(
            frame_data["edge_dets"],
            frame_data["cloud_dets"],
            frame_data["gt_objs"],
        )
    return vectors
