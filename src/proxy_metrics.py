import concurrent.futures
import heapq
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

from .error_decomposition import compute_lcer_delta_vectors
from .metrics import (COCO_IOU_THRESHOLDS, compute_iou,
                      compute_allpoint_ap,
                      match_detections_to_gt,
                      _precompute_detection_ious,
                      _run_greedy_matching, _safe_iou_worker_count)

_SHARED_BASE_ENTRIES = None
_SHARED_BASE_BY_FRAME = None
_SHARED_CLOUD_ENTRIES = None
_SHARED_GT = None
_SHARED_GT_COUNTS = None
_ORIC_METRIC_NAMES = ("oric_11pt", "oric_allpoint", "oric_coco")


# ---------------------------------------------------------------------------
#  O(D) merge helpers — replace the O(D log D) per-frame re-sort
# ---------------------------------------------------------------------------

def _merge_swapped_precomputed(base_entries, skip_key, cloud_entries):
    """Build the swapped precomputed list via O(D) merge (non-context).

    *base_entries* is already sorted by confidence (descending).
    *cloud_entries* for the target frame is also sorted desc.
    Filtering out *skip_key* from a sorted list preserves order,
    so we merge the filtered base with cloud entries in one pass.

    Returns a list of precomputed entries (key, ious, class_ok)
    sorted by confidence descending — identical to what a
    filter → extend → sort would produce, but in O(D) instead of
    O(D log D).
    """
    result = []
    j = 0
    n_cloud = len(cloud_entries)
    for conf, precomp, fk in base_entries:
        if fk == skip_key:
            continue
        while j < n_cloud and cloud_entries[j][0] >= conf:
            result.append(cloud_entries[j][1])
            j += 1
        result.append(precomp)
    while j < n_cloud:
        result.append(cloud_entries[j][1])
        j += 1
    return result


def _merge_swapped_precomputed_ctx(base_entries, ctx_keys, skip_key,
                                    cloud_entries):
    """Build the swapped precomputed list via O(D) merge (context mode).

    Same as :func:`_merge_swapped_precomputed` but additionally filters
    base entries to only those whose frame_key is in *ctx_keys*.
    """
    result = []
    j = 0
    n_cloud = len(cloud_entries)
    for conf, precomp, fk in base_entries:
        if fk not in ctx_keys or fk == skip_key:
            continue
        while j < n_cloud and cloud_entries[j][0] >= conf:
            result.append(cloud_entries[j][1])
            j += 1
        result.append(precomp)
    while j < n_cloud:
        result.append(cloud_entries[j][1])
        j += 1
    return result


# ---------------------------------------------------------------------------

def _oric_worker_init(base_entries, base_entries_by_frame, cloud_entries_by_frame,
                      gt_by_frame, gt_counts):
    global _SHARED_BASE_ENTRIES, _SHARED_BASE_BY_FRAME
    global _SHARED_CLOUD_ENTRIES, _SHARED_GT, _SHARED_GT_COUNTS
    _SHARED_BASE_ENTRIES = base_entries
    _SHARED_BASE_BY_FRAME = base_entries_by_frame
    _SHARED_CLOUD_ENTRIES = cloud_entries_by_frame
    _SHARED_GT = gt_by_frame
    _SHARED_GT_COUNTS = gt_counts


def _merge_context_base_entries(ctx_keys, skip_key=None):
    iterables = []
    for ctx_key in ctx_keys:
        if ctx_key == skip_key:
            continue
        entries = _SHARED_BASE_BY_FRAME.get(ctx_key, [])
        if entries:
            iterables.append(entries)
    if not iterables:
        return []
    return list(heapq.merge(*iterables, key=lambda item: item[0], reverse=True))

def _process_oric_chunk(chunk):
    results = {}
    for task in chunk:
        idx, key, ctx_keys, n_context, use_context, num_frames, total_gt, base_ap50_11pt, base_ap50_allpoint, base_ap_coco = task

        cloud_entries = _SHARED_CLOUD_ENTRIES.get(key, [])

        if use_context:
            ctx_gt = {
                ctx_key: _SHARED_GT[ctx_key]
                for ctx_key in ctx_keys
                if ctx_key in _SHARED_GT
            }
            ctx_total_gt = sum(_SHARED_GT_COUNTS.get(ctx_key, 0) for ctx_key in ctx_keys)
            if ctx_total_gt == 0:
                results[key] = {"oric_11pt": 0.0, "oric_allpoint": 0.0, "oric_coco": 0.0}
                continue

            base_ctx_entries = _merge_context_base_entries(ctx_keys)
            base_ctx_precomp = [precomp for _conf, precomp in base_ctx_entries]

            mod_precomputed = []
            j = 0
            n_cloud = len(cloud_entries)
            for conf, precomp in _merge_context_base_entries(ctx_keys, skip_key=key):
                while j < n_cloud and cloud_entries[j][0] >= conf:
                    mod_precomputed.append(cloud_entries[j][1])
                    j += 1
                mod_precomputed.append(precomp)
            while j < n_cloud:
                mod_precomputed.append(cloud_entries[j][1])
                j += 1

            ap_coco_base_list = []
            ap_coco_mod_list = []
            ctx_base_11pt = ctx_base_allpoint = 0.0
            ctx_mod_11pt = ctx_mod_allpoint = 0.0

            for t in COCO_IOU_THRESHOLDS:
                b_t = _run_greedy_matching(base_ctx_precomp, ctx_gt, ctx_total_gt, t)
                m_t = _run_greedy_matching(mod_precomputed, ctx_gt, ctx_total_gt, t)
                ap_coco_base_list.append(b_t["ap"])
                ap_coco_mod_list.append(m_t["ap"])
                if abs(t - 0.5) < 1e-9:
                    ctx_base_11pt = b_t["ap"]
                    ctx_base_allpoint = b_t["ap_allpoint"]
                    ctx_mod_11pt = m_t["ap"]
                    ctx_mod_allpoint = m_t["ap_allpoint"]

            ctx_base_coco = float(np.mean(ap_coco_base_list))
            ctx_mod_coco = float(np.mean(ap_coco_mod_list))

            results[key] = {
                "oric_11pt": float(np.float64(ctx_mod_11pt - ctx_base_11pt) * n_context),
                "oric_allpoint": float(np.float64(ctx_mod_allpoint - ctx_base_allpoint) * n_context),
                "oric_coco": float(np.float64(ctx_mod_coco - ctx_base_coco) * n_context),
            }
        else:
            # O(D) merge instead of filter+extend+sort O(D log D)
            mod_precomputed = _merge_swapped_precomputed(
                _SHARED_BASE_ENTRIES, key, cloud_entries)

            coco_aps = []
            mod_ap50_11pt = np.float64(0)
            mod_ap50_allpoint = np.float64(0)
            for t in COCO_IOU_THRESHOLDS:
                res = _run_greedy_matching(mod_precomputed, _SHARED_GT, total_gt, t)
                coco_aps.append(res["ap"])
                if abs(t - 0.5) < 1e-9:
                    mod_ap50_11pt = np.float64(res["ap"])
                    mod_ap50_allpoint = np.float64(res["ap_allpoint"])
            mod_ap_coco = np.float64(np.mean(coco_aps))

            results[key] = {
                "oric_11pt": float(np.float64(mod_ap50_11pt - base_ap50_11pt) * num_frames),
                "oric_allpoint": float(np.float64(mod_ap50_allpoint - base_ap50_allpoint) * num_frames),
                "oric_coco": float(np.float64(mod_ap_coco - base_ap_coco) * num_frames),
            }
    return results


def _run_oric_tasks(
    tasks,
    base_entries,
    base_entries_by_frame,
    cloud_entries_by_frame,
    gt_by_frame,
    video_name: str,
    quiet: bool = False,
):
    """Execute ORIC swap tasks and return metrics keyed by frame tuple."""
    metrics_by_frame: Dict[tuple, Dict[str, float]] = {}

    num_workers = min(16, (os.cpu_count() or 1) + 4)
    num_workers = _safe_iou_worker_count(num_workers)
    chunk_size = max(1, len(tasks) // max(num_workers * 2, 1))
    chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    import multiprocessing
    if len(tasks) < 50 or multiprocessing.current_process().daemon or num_workers <= 1:
        _oric_worker_init(
            base_entries,
            base_entries_by_frame,
            cloud_entries_by_frame,
            gt_by_frame,
            {key: len(gts) for key, gts in gt_by_frame.items()},
        )
        for chunk in chunks:
            res = _process_oric_chunk(chunk)
            metrics_by_frame.update(res)
        return metrics_by_frame

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_oric_worker_init,
            initargs=(
                base_entries,
                base_entries_by_frame,
                cloud_entries_by_frame,
                gt_by_frame,
                {key: len(gts) for key, gts in gt_by_frame.items()},
            ),
        ) as executor:
            futures = [executor.submit(_process_oric_chunk, chunk) for chunk in chunks]

            iterator = concurrent.futures.as_completed(futures)
            if not quiet:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"  ORIC ({video_name})",
                    leave=False,
                )

            for future in iterator:
                metrics_by_frame.update(future.result())
    except OSError as e:
        if e.errno != 24:
            raise
        _oric_worker_init(
            base_entries,
            base_entries_by_frame,
            cloud_entries_by_frame,
            gt_by_frame,
            {key: len(gts) for key, gts in gt_by_frame.items()},
        )
        for chunk in chunks:
            metrics_by_frame.update(_process_oric_chunk(chunk))

    return metrics_by_frame


def _sign_consistency(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    signs = np.sign(values)
    nonzero = signs[signs != 0]
    if nonzero.size == 0:
        return 1.0
    return float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))


def _rank_consistency(rank_values: np.ndarray) -> float:
    rank_values = np.asarray(rank_values, dtype=float).reshape(-1)
    if rank_values.size <= 1:
        return 1.0
    # Rank percentiles live in [0, 1]; 0.5 is the largest possible std.
    return float(np.clip(1.0 - (np.std(rank_values) / 0.5), 0.0, 1.0))


def _summarize_draw_statistics(
    draw_results_seq: List[Dict],
    ordered_keys: List,
) -> Dict:
    """Aggregate ORIC draw values and attach stability diagnostics."""
    if not draw_results_seq:
        return {}

    value_history = {
        key: {metric_name: [] for metric_name in _ORIC_METRIC_NAMES}
        for key in ordered_keys
    }
    rank_history = {
        key: {metric_name: [] for metric_name in _ORIC_METRIC_NAMES}
        for key in ordered_keys
    }

    for draw_results in draw_results_seq:
        for metric_name in _ORIC_METRIC_NAMES:
            metric_values = np.asarray(
                [draw_results.get(key, {}).get(metric_name, 0.0) for key in ordered_keys],
                dtype=float,
            )
            ranks = rankdata(metric_values, method="average") / max(len(metric_values), 1)
            for idx, key in enumerate(ordered_keys):
                value_history[key][metric_name].append(float(metric_values[idx]))
                rank_history[key][metric_name].append(float(ranks[idx]))

    summarized: Dict = {}
    for key in ordered_keys:
        row = {}
        for metric_name in _ORIC_METRIC_NAMES:
            values = np.asarray(value_history[key][metric_name], dtype=float)
            row[metric_name] = float(np.mean(values)) if values.size else 0.0
            row[f"{metric_name}_draw_std"] = float(np.std(values)) if values.size else 0.0
            row[f"{metric_name}_sign_consistency"] = _sign_consistency(values)
            row[f"{metric_name}_rank_consistency"] = _rank_consistency(
                np.asarray(rank_history[key][metric_name], dtype=float)
            )
        summarized[key] = row
    return summarized


def compute_proxy_metrics(
    raw_boxes: List[Dict],
    video_name: str,
    base_ap50_11pt: float,
    base_ap50_allpoint: float,
    base_ap_coco: float,
    quiet: bool = False,
    context_size: int = 1000,
    context_draws: int = 1,
) -> Dict[int, Dict[str, float]]:
    """Compute ORIC proxy metrics for a single video.

    ORIC measures the dataset-wide AP impact of offloading a single frame
    (swapping its edge detections for cloud detections), evaluated over
    an ensemble of context frames.

    When *context_size* > 0 and smaller than the number of frames,
    a random subset of *context_size* other frames is sampled as the
    context for each target frame (matching the EdgeML paper's
    ``num_ensemble`` parameter).  Otherwise all frames in the video
    are used as context.

    **Optimised**: IoU between every detection and its frame's ground truth
    is precomputed once (for both edge and cloud detections).  Per-frame
    swaps then only need list filtering + merge-sort + greedy matching —
    no IoU recomputation.

    Args:
        raw_boxes: List of dicts with 'frame_id', 'edge_dets', 'cloud_dets',
                   'gt_objs'.
        video_name: Name of the video / chunk.
        base_ap50_11pt: Precomputed dataset-wide edge 11-point AP@0.5.
        base_ap50_allpoint: Precomputed dataset-wide edge all-point AP@0.5.
        base_ap_coco: Precomputed dataset-wide edge COCO mAP.
        quiet: If True, suppress progress bars (useful for multiprocessing).
        context_size: Number of context frames to sample per target frame.
                      0 means use all frames in the video (no sampling).
        context_draws: Number of independent context draws to average when
                       context sampling is enabled.

    Returns:
        Dict mapping frame_id → {'oric_11pt', 'oric_allpoint', 'oric_coco'}.
    """
    metrics_by_frame: Dict[int, Dict[str, float]] = {}

    # 1. Build GT structures -----------------------------------------------
    gt_by_frame: Dict[tuple, list] = {}
    total_gt = 0
    for data in raw_boxes:
        fid = data["frame_id"]
        key = (video_name, fid)
        gt_by_frame[key] = data["gt_objs"]
        total_gt += len(data["gt_objs"])

    num_frames = len(raw_boxes)
    if num_frames == 0 or total_gt == 0:
        return {}

    # 2. Build detection lists and precompute IoU ONCE ---------------------
    all_edge_dets = []          # [(det_tuple, video, frame), ...]
    edge_dets_by_frame = {}     # frame_key -> [(det_tuple, video, frame), ...]
    cloud_dets_by_frame = {}

    for data in raw_boxes:
        fid = data["frame_id"]
        key = (video_name, fid)
        e_dets = [(d, video_name, fid) for d in data["edge_dets"]]
        c_dets = [(d, video_name, fid) for d in data["cloud_dets"]]
        edge_dets_by_frame[key] = e_dets
        cloud_dets_by_frame[key] = c_dets
        all_edge_dets.extend(e_dets)

    # Sort all edge dets by confidence (descending) — same order as the
    # base dataset-wide AP computation.
    all_edge_dets.sort(key=lambda x: x[0][4], reverse=True)

    # Precompute IoU for ALL edge detections at once.
    base_precomputed = _precompute_detection_ious(all_edge_dets, gt_by_frame)

    # Tag each entry with (confidence, precomputed_entry, frame_key) so we
    # can filter by frame_key during per-frame swaps.
    base_entries = [
        (det[0][4], base_precomputed[i], (video_name, det[2]))
        for i, det in enumerate(all_edge_dets)
    ]
    base_entries_by_frame: Dict[tuple, list] = {}
    for conf, precomp, frame_key in base_entries:
        base_entries_by_frame.setdefault(frame_key, []).append((conf, precomp))

    # Precompute IoU for cloud detections in a single batched call
    all_cloud_dets = []
    for c_dets in cloud_dets_by_frame.values():
        all_cloud_dets.extend(c_dets)
        
    cloud_precomp_flat = _precompute_detection_ious(all_cloud_dets, gt_by_frame)
    
    cloud_entries_by_frame: Dict[tuple, list] = {}
    for det, precomp in zip(all_cloud_dets, cloud_precomp_flat):
        key = (det[1], det[2])
        if key not in cloud_entries_by_frame:
            cloud_entries_by_frame[key] = []
        cloud_entries_by_frame[key].append((det[0][4], precomp))
        
    for key in cloud_entries_by_frame:
        cloud_entries_by_frame[key].sort(key=lambda x: x[0], reverse=True)

    # 3. Per-frame swap: reuse precomputed IoU -----------------------------
    # Decide whether to use context subsetting (random subset of frames)
    # or the full video (all frames).
    use_context = 0 < context_size < num_frames - 1
    rng = np.random.RandomState(42) if use_context else None
    n_context = context_size + 1 if use_context else num_frames

    n_draws = max(1, int(context_draws))
    if not use_context:
        n_draws = 1

    ordered_fids = [int(frame_data["frame_id"]) for frame_data in raw_boxes]
    draw_results_seq: List[Dict[int, Dict[str, float]]] = []
    candidate_indices = [
        np.concatenate((np.arange(idx, dtype=int),
                        np.arange(idx + 1, num_frames, dtype=int)))
        for idx in range(num_frames)
    ] if use_context else []

    for _draw in range(n_draws):
        tasks = []
        for idx, data in enumerate(raw_boxes):
            fid = data["frame_id"]
            key = (video_name, fid)

            ctx_keys = None
            if use_context:
                ctx_indices = rng.choice(
                    candidate_indices[idx],
                    size=min(context_size, len(candidate_indices[idx])),
                    replace=False,
                )
                ctx_keys = {
                    (video_name, raw_boxes[ci]["frame_id"]) for ci in ctx_indices
                }
                ctx_keys.add(key)

            tasks.append((
                idx, key, ctx_keys, n_context, use_context, num_frames, total_gt,
                float(base_ap50_11pt), float(base_ap50_allpoint), float(base_ap_coco),
            ))

        draw_results = _run_oric_tasks(
            tasks,
            base_entries,
            base_entries_by_frame,
            cloud_entries_by_frame,
            gt_by_frame,
            video_name=video_name,
            quiet=quiet,
        )
        frame_results: Dict[int, Dict[str, float]] = {}
        for key, metrics in draw_results.items():
            fid = int(key[1])
            frame_results[fid] = {
                metric_name: float(metrics.get(metric_name, 0.0))
                for metric_name in _ORIC_METRIC_NAMES
            }
        draw_results_seq.append(frame_results)

    metrics_by_frame = _summarize_draw_statistics(draw_results_seq, ordered_fids)

    return metrics_by_frame


# ---------------------------------------------------------------------------
#  Dataset-wide ORIC (cross-video, split-level)
# ---------------------------------------------------------------------------

def compute_dataset_wide_oric(
    all_raw_boxes: List[Dict],
    quiet: bool = False,
    context_size: int = 0,
    context_draws: int = 1,
) -> Dict[tuple, Dict[str, float]]:
    """Compute dataset-wide ORIC for every frame in a collection.

    Each frame's ORIC quantifies the change in the **set-wide** AP when
    that single frame's edge detections are replaced by cloud detections.
    Because the offloading evaluation merges all detections into one
    global precision–recall curve, a frame's impact depends on every
    other frame in the set — making this the correct ranking signal for
    the oracle offloading strategy.

    ``ORIC_i = (AP_modified_i − AP_base) × N``

    All arithmetic uses float64 to preserve precision for small AP deltas.
    IoU is precomputed once; per-frame swaps only filter, merge, and run
    greedy matching — no IoU recomputation.

    Args:
        all_raw_boxes: Flat list of per-frame dicts (from potentially many
            videos).  Each dict must have ``frame_id``, ``video_name``,
            ``edge_dets``, ``cloud_dets``, ``gt_objs``.
        quiet: Suppress the progress bar.
        context_size: Number of context frames per target.
            0 = use **all** frames in the collection (exact but slower).
        context_draws: Number of independent context draws to average when
            context sampling is enabled.

    Returns:
        Dict mapping ``(video_name, frame_id)`` → dict with float64
        fields ``oric_11pt``, ``oric_allpoint``, ``oric_coco``.
    """
    metrics: Dict[tuple, Dict[str, float]] = {}

    # 1. Build GT structures -----------------------------------------------
    gt_by_frame: Dict[tuple, list] = {}
    total_gt = 0
    for data in all_raw_boxes:
        key = (str(data["video_name"]), int(data["frame_id"]))
        gt_by_frame[key] = data["gt_objs"]
        total_gt += len(data["gt_objs"])

    num_frames = len(all_raw_boxes)
    if num_frames == 0 or total_gt == 0:
        return {}

    # 2. Detection lists + precompute IoU once -----------------------------
    all_edge_dets: list = []
    cloud_dets_by_frame: Dict[tuple, list] = {}

    for data in all_raw_boxes:
        vn = str(data["video_name"])
        fid = int(data["frame_id"])
        key = (vn, fid)
        e_dets = [(d, vn, fid) for d in data["edge_dets"]]
        c_dets = [(d, vn, fid) for d in data["cloud_dets"]]
        cloud_dets_by_frame[key] = c_dets
        all_edge_dets.extend(e_dets)

    all_edge_dets.sort(key=lambda x: x[0][4], reverse=True)

    if not quiet:
        from . import log
        log.info(f"Precomputing IoU for {len(all_edge_dets)} edge + "
                 f"{sum(len(v) for v in cloud_dets_by_frame.values())} cloud dets", indent=6)

    base_precomputed = _precompute_detection_ious(all_edge_dets, gt_by_frame)

    # (confidence, precomputed_entry, frame_key)
    base_entries = [
        (det[0][4], base_precomputed[i], (str(det[1]), int(det[2])))
        for i, det in enumerate(all_edge_dets)
    ]
    base_entries_by_frame: Dict[tuple, list] = {}
    for conf, precomp, frame_key in base_entries:
        base_entries_by_frame.setdefault(frame_key, []).append((conf, precomp))

    # Compute full-set base AP at all COCO thresholds
    base_coco_aps: list = []
    base_ap50_11pt = np.float64(0)
    base_ap50_allpoint = np.float64(0)
    for t in COCO_IOU_THRESHOLDS:
        res = _run_greedy_matching(base_precomputed, gt_by_frame, total_gt, t)
        base_coco_aps.append(res["ap"])
        if abs(t - 0.5) < 1e-9:
            base_ap50_11pt = np.float64(res["ap"])
            base_ap50_allpoint = np.float64(res["ap_allpoint"])
    base_ap_coco = np.float64(np.mean(base_coco_aps))

    # Precompute IoU for cloud detections in a single batched call
    all_cloud_dets = []
    for c_dets in cloud_dets_by_frame.values():
        all_cloud_dets.extend(c_dets)
        
    cloud_precomp_flat = _precompute_detection_ious(all_cloud_dets, gt_by_frame)
    
    cloud_entries_by_frame: Dict[tuple, list] = {}
    for det, precomp in zip(all_cloud_dets, cloud_precomp_flat):
        key = (str(det[1]), int(det[2]))
        if key not in cloud_entries_by_frame:
            cloud_entries_by_frame[key] = []
        cloud_entries_by_frame[key].append((det[0][4], precomp))
        
    for key in cloud_entries_by_frame:
        cloud_entries_by_frame[key].sort(key=lambda x: x[0], reverse=True)

    # 3. Per-frame swap ----------------------------------------------------
    use_context = 0 < context_size < num_frames - 1
    rng = np.random.RandomState(42) if use_context else None
    n_context = context_size + 1 if use_context else num_frames

    n_draws = max(1, int(context_draws))
    if not use_context:
        n_draws = 1

    ordered_keys = [
        (str(data["video_name"]), int(data["frame_id"]))
        for data in all_raw_boxes
    ]
    draw_results_seq: List[Dict[tuple, Dict[str, float]]] = []
    candidate_indices = [
        np.concatenate((np.arange(idx, dtype=int),
                        np.arange(idx + 1, num_frames, dtype=int)))
        for idx in range(num_frames)
    ] if use_context else []

    for _draw in range(n_draws):
        tasks = []
        for idx, data in enumerate(all_raw_boxes):
            vn = str(data["video_name"])
            fid = int(data["frame_id"])
            key = (vn, fid)

            ctx_keys = None
            if use_context:
                ctx_indices = rng.choice(
                    candidate_indices[idx],
                    size=min(context_size, len(candidate_indices[idx])),
                    replace=False,
                )
                ctx_keys = {
                    (str(all_raw_boxes[ci]["video_name"]),
                     int(all_raw_boxes[ci]["frame_id"]))
                    for ci in ctx_indices
                }
                ctx_keys.add(key)

            tasks.append((
                idx, key, ctx_keys, n_context, use_context, num_frames, total_gt,
                float(base_ap50_11pt), float(base_ap50_allpoint), float(base_ap_coco),
            ))

        draw_results = _run_oric_tasks(
            tasks,
            base_entries,
            base_entries_by_frame,
            cloud_entries_by_frame,
            gt_by_frame,
            video_name=f"dataset:{num_frames}",
            quiet=quiet,
        )
        normalized_draw: Dict[tuple, Dict[str, float]] = {}
        for key, values in draw_results.items():
            normalized_draw[key] = {
                metric_name: float(values.get(metric_name, 0.0))
                for metric_name in _ORIC_METRIC_NAMES
            }
        draw_results_seq.append(normalized_draw)

    metrics = _summarize_draw_statistics(draw_results_seq, ordered_keys)

    return metrics


# ---------------------------------------------------------------------------
#  SRRM — Spatially-Resolved Reward Matrix
# ---------------------------------------------------------------------------

def _bbox_center(det: tuple) -> Tuple[float, float]:
    """Return (cx, cy) from a detection tuple (left, top, w, h, conf, cls)."""
    left, top, w, h = det[:4]
    return (left + w / 2.0, top + h / 2.0)


def compute_srrm(
    frame_data: Dict,
    grid_size: int = 8,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Compute the Spatially-Resolved Reward Matrix for one frame.

    Divides the image into an S×S grid.  Each cell captures the localised
    performance discrepancy between the strong (cloud) and weak (edge) models:

        R(x,y) = max_j(c_s,j · IoU(b_s,j, b_gt))
               − max_k(c_w,k · IoU(b_w,k, b_gt))

    for detections whose bbox centres fall within cell (x, y).

    Args:
        frame_data: Dict with 'edge_dets', 'cloud_dets', 'gt_objs'.
            Detections are (left, top, width, height, conf, class).
            GT objects are typically dicts with ``bbox`` and ``class``,
            but tuple-style ``(left, top, width, height, class)`` entries
            are also accepted for backward compatibility.
        grid_size: Number of spatial bins per axis (default 8).
        image_size: (width, height) of the image.  If None, inferred from
            the maximum bbox extent across all detections and GT.

    Returns:
        np.ndarray of shape (grid_size, grid_size).
    """
    S = grid_size
    edge_dets = frame_data.get("edge_dets", [])
    cloud_dets = frame_data.get("cloud_dets", [])
    gt_objs = frame_data.get("gt_objs", [])

    matrix = np.zeros((S, S), dtype=np.float64)

    if not gt_objs:
        return matrix.astype(np.float32)

    def _gt_bbox(gt) -> Tuple[float, float, float, float]:
        if isinstance(gt, dict):
            bbox = gt.get("bbox")
            if bbox is None:
                raise KeyError("GT object dict is missing 'bbox'")
            return tuple(bbox[:4])
        return tuple(gt[:4])

    # Infer image dimensions from bbox extents if not provided
    if image_size is None:
        max_x, max_y = 1.0, 1.0
        for det in edge_dets + cloud_dets:
            max_x = max(max_x, det[0] + det[2])
            max_y = max(max_y, det[1] + det[3])
        for gt in gt_objs:
            gt_bbox = _gt_bbox(gt)
            max_x = max(max_x, gt_bbox[0] + gt_bbox[2])
            max_y = max(max_y, gt_bbox[1] + gt_bbox[3])
        img_w, img_h = max_x, max_y
    else:
        img_w, img_h = image_size

    def _best_conf_iou(dets: list, cell_x: int, cell_y: int) -> float:
        """Best confidence-weighted IoU for detections whose centre is in cell."""
        best = 0.0
        for det in dets:
            cx, cy = _bbox_center(det)
            gx = min(int(cx / img_w * S), S - 1)
            gy = min(int(cy / img_h * S), S - 1)
            if gx != cell_x or gy != cell_y:
                continue
            conf = det[4]
            det_bbox = det[:4]
            for gt in gt_objs:
                iou = compute_iou(det_bbox, _gt_bbox(gt))
                best = max(best, conf * iou)
        return best

    for x in range(S):
        for y in range(S):
            strong_best = _best_conf_iou(cloud_dets, x, y)
            weak_best = _best_conf_iou(edge_dets, x, y)
            matrix[y, x] = strong_best - weak_best

    return matrix.astype(np.float32)


# ---------------------------------------------------------------------------
#  BWD — Bounding-Box Wasserstein Discrepancy
# ---------------------------------------------------------------------------

def compute_bwd(frame_data: Dict,
                image_size: Optional[Tuple[int, int]] = None) -> float:
    """Compute the Bounding-Box Wasserstein Discrepancy for one frame.

    The BWD is the optimal transport cost between the strong and weak
    detection distributions, where each detection is represented by a
    feature vector (cx, cy, w, h, conf) normalised to [0, 1].

    Uses ``scipy.optimize.linear_sum_assignment`` on the squared-Euclidean
    cost matrix to solve the assignment problem.  When detection counts
    differ, the cost matrix is padded (unmatched detections incur a penalty
    equal to the maximum pairwise cost).

    Args:
        frame_data: Dict with 'edge_dets', 'cloud_dets'.
            Detections are (left, top, width, height, conf, class).
        image_size: (width, height) for normalisation.  If None, inferred.

    Returns:
        Scalar BWD value (float).
    """
    edge_dets = frame_data.get("edge_dets", [])
    cloud_dets = frame_data.get("cloud_dets", [])

    # Infer image dimensions for normalisation
    if image_size is None:
        max_x, max_y = 1.0, 1.0
        for det in edge_dets + cloud_dets:
            max_x = max(max_x, det[0] + det[2])
            max_y = max(max_y, det[1] + det[3])
        img_w, img_h = max_x, max_y
    else:
        img_w, img_h = image_size

    def _det_features(dets: list) -> np.ndarray:
        """Build (N, 5) feature matrix: [cx, cy, w, h, conf] normalised."""
        feats = np.empty((len(dets), 5), dtype=np.float64)
        for i, det in enumerate(dets):
            left, top, w, h, conf = det[0], det[1], det[2], det[3], det[4]
            feats[i] = [
                (left + w / 2.0) / img_w,
                (top + h / 2.0) / img_h,
                w / img_w,
                h / img_h,
                conf,
            ]
        return feats

    feat_e = _det_features(edge_dets)
    feat_c = _det_features(cloud_dets)

    if feat_e.size == 0 and feat_c.size == 0:
        return 0.0
    if feat_e.size == 0:
        return float(np.mean((feat_c ** 2).sum(axis=1)))
    if feat_c.size == 0:
        return float(np.mean((feat_e ** 2).sum(axis=1)))

    # Squared-Euclidean cost matrix
    diff = feat_e[:, None, :] - feat_c[None, :, :]  # (Ne, Nc, 5)
    cost = (diff ** 2).sum(axis=2)  # (Ne, Nc)

    # Pad to square if sizes differ
    ne, nc = cost.shape
    if ne != nc:
        max_cost = cost.max() if cost.size > 0 else 1.0
        size = max(ne, nc)
        padded = np.full((size, size), max_cost * 1.5, dtype=np.float64)
        padded[:ne, :nc] = cost
        cost = padded

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # Total transport cost, including dummy assignments when counts differ.
    total_cost = cost[row_ind, col_ind].sum()
    return float(total_cost / max(len(row_ind), 1))


# ---------------------------------------------------------------------------
#  Fine-grained scenario components
# ---------------------------------------------------------------------------

FINEGRAINED_COMPONENTS = (
    "cls",
    "loc",
    "both",
    "dup",
    "bg",
    "miss",
    "ap75",
    "ap90",
    "center_offset",
    "category_precision",
)


def _gt_center(gt_obj) -> Tuple[float, float]:
    bbox = gt_obj["bbox"] if isinstance(gt_obj, dict) else gt_obj[:4]
    left, top, width, height = bbox[:4]
    return left + width / 2.0, top + height / 2.0


def _normalized_center_offset(
    detections: List[tuple],
    gt_objects: List[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Mean normalized center offset across greedily matched detections."""
    if not gt_objects:
        return 0.0
    if not detections:
        return 1.0

    sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
    gt_matched = [False] * len(gt_objects)
    offsets: list[float] = []

    for det in sorted_dets:
        best_iou = 0.0
        best_gt_idx = -1
        det_bbox = det[:4]
        det_class = det[5]

        for gt_idx, gt_obj in enumerate(gt_objects):
            if gt_matched[gt_idx]:
                continue
            gt_class = gt_obj["class"]
            if det_class.lower() != gt_class.lower():
                from .metrics import classes_match
                if not classes_match(det_class, gt_class):
                    continue
            iou = compute_iou(det_bbox, gt_obj["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou < iou_threshold or best_gt_idx < 0:
            continue

        gt_matched[best_gt_idx] = True
        det_cx, det_cy = _bbox_center(det)
        gt_cx, gt_cy = _gt_center(gt_objects[best_gt_idx])
        gt_w = max(float(gt_objects[best_gt_idx]["bbox"][2]), 1e-6)
        gt_h = max(float(gt_objects[best_gt_idx]["bbox"][3]), 1e-6)
        gt_diag = max(float(np.hypot(gt_w, gt_h)), 1e-6)
        offsets.append(float(np.hypot(det_cx - gt_cx, det_cy - gt_cy) / gt_diag))

    if not offsets:
        return 1.0
    return float(np.mean(offsets))


def compute_ap_delta(frame_data: Dict, iou_threshold: float) -> float:
    """Cloud-minus-edge AP delta at a single IoU threshold for one frame."""
    edge_ap = compute_allpoint_ap(
        frame_data.get("edge_dets", []),
        frame_data.get("gt_objs", []),
        iou_threshold=iou_threshold,
    )
    cloud_ap = compute_allpoint_ap(
        frame_data.get("cloud_dets", []),
        frame_data.get("gt_objs", []),
        iou_threshold=iou_threshold,
    )
    return float(cloud_ap - edge_ap)


def compute_category_precision_delta(frame_data: Dict,
                                     iou_threshold: float = 0.5) -> float:
    """Cloud-minus-edge precision delta at a single IoU threshold."""
    gt_objects = frame_data.get("gt_objs", [])
    e_tp, e_fp, _ = match_detections_to_gt(
        frame_data.get("edge_dets", []), gt_objects, iou_threshold=iou_threshold
    )
    c_tp, c_fp, _ = match_detections_to_gt(
        frame_data.get("cloud_dets", []), gt_objects, iou_threshold=iou_threshold
    )
    edge_precision = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
    cloud_precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
    return float(cloud_precision - edge_precision)


def compute_center_offset_delta(frame_data: Dict,
                                iou_threshold: float = 0.5) -> float:
    """Edge-minus-cloud normalized center-offset delta."""
    gt_objects = frame_data.get("gt_objs", [])
    edge_offset = _normalized_center_offset(
        frame_data.get("edge_dets", []), gt_objects, iou_threshold=iou_threshold
    )
    cloud_offset = _normalized_center_offset(
        frame_data.get("cloud_dets", []), gt_objects, iou_threshold=iou_threshold
    )
    return float(edge_offset - cloud_offset)


def compute_finegrained_proxy_vector(frame_data: Dict,
                                     metric_family: str = "coco",
                                     lcer_vector: Optional[np.ndarray] = None) -> np.ndarray:
    """Compose LCER error deltas with direct scenario-specific proxy metrics."""
    if metric_family not in {"11pt", "allpoint", "coco"}:
        raise ValueError(f"Unsupported fine-grained metric family: {metric_family}")

    lcer = (
        np.asarray(lcer_vector, dtype=np.float64)
        if lcer_vector is not None
        else compute_lcer_delta_vectors(
            frame_data.get("edge_dets", []),
            frame_data.get("cloud_dets", []),
            frame_data.get("gt_objs", []),
        )[metric_family]
    )
    extras = np.array([
        compute_ap_delta(frame_data, 0.75),
        compute_ap_delta(frame_data, 0.90),
        compute_center_offset_delta(frame_data, iou_threshold=0.5),
        compute_category_precision_delta(frame_data, iou_threshold=0.5),
    ], dtype=np.float64)
    return np.concatenate([np.asarray(lcer, dtype=np.float64), extras], axis=0)
