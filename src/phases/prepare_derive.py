"""Sub-phase 2a: per-video feature derivation from raw detections + GT."""

import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PipelineConfig

from .. import log
from ..dataset import get_dataset
from ..error_decomposition import LCER_ERROR_TYPES, compute_lcer_vectors
from ..features import (
    encode_float_sequence,
    extract_detection_features,
    extract_full_proposal_rule_inputs,
    extract_topk_proposal_features,
    load_detection_results,
    topk_proposal_feature_columns,
)
from ..metrics import (
    COCO_IOU_THRESHOLDS,
    compute_11point_ap,
    compute_allpoint_ap,
    compute_coco_frame_ap,
    compute_dataset_ap,
    match_detections_to_gt,
)
from ..proxy_metrics import (
    compute_bwd,
    compute_dataset_wide_oric,
    compute_finegrained_proxy_vector,
    compute_proxy_metrics,
    compute_srrm,
)
from config.scenarios import SCENARIO_COMPONENTS
from .prepare_config import _required_prepare_inputs, _required_proxy_families


def _load_raw_boxes_for_frames(
    derived_dir: Path,
    video_names: np.ndarray,
    frame_ids: np.ndarray,
) -> list:
    """Load raw bounding boxes for specific frames from per-video pkl files.

    Returns a flat list of frame dicts (only those found in the pkl files).
    Missing frames are skipped with a warning.
    """
    vid_to_fids: dict = {}
    for i, (v, fid) in enumerate(zip(video_names, frame_ids)):
        v_str = str(v)
        if v_str not in vid_to_fids:
            vid_to_fids[v_str] = {}
        vid_to_fids[v_str][int(fid)] = i

    raw_boxes: list = [None] * len(video_names)
    for v, fid_map in vid_to_fids.items():
        pkl_path = derived_dir / f"{v}_boxes.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as fh:
            v_boxes = pickle.load(fh)
        for frame_data in v_boxes:
            fid = int(frame_data["frame_id"])
            if fid in fid_map:
                raw_boxes[fid_map[fid]] = frame_data

    n_missing = sum(1 for b in raw_boxes if b is None)
    if n_missing:
        log.warn(f"{n_missing}/{len(video_names)} frames missing "
                 "raw boxes for dataset-wide ORIC", indent=6)

    return [b for b in raw_boxes if b is not None]


def _load_srrm_matrices(
    derived_dir: Path,
    video_names: np.ndarray,
    frame_ids: np.ndarray,
    grid_size: int = 8,
) -> np.ndarray:
    """Load SRRM spatial matrices for frames from per-video _srrm.pkl files.

    Returns (N, S, S) array aligned with the video_names/frame_ids order,
    or None if no SRRM files exist.
    """
    vid_to_fids: dict = {}
    for i, (v, fid) in enumerate(zip(video_names, frame_ids)):
        v_str = str(v)
        if v_str not in vid_to_fids:
            vid_to_fids[v_str] = {}
        vid_to_fids[v_str][int(fid)] = i

    n = len(video_names)
    result = np.zeros((n, grid_size, grid_size), dtype=np.float32)
    found_any = False

    for v, fid_map in vid_to_fids.items():
        pkl_path = derived_dir / f"{v}_srrm.pkl"
        if not pkl_path.exists():
            continue
        found_any = True
        with open(pkl_path, "rb") as fh:
            srrm_dict = pickle.load(fh)
        for fid, idx in fid_map.items():
            if fid in srrm_dict:
                result[idx] = srrm_dict[fid]

    return result if found_any else None


def _dataset_oric_cache_path(
    dataset_root: str,
    edge_model: str,
    cloud_model: str,
    seed: int,
    split_name: str,
    context_size: int,
    context_draws: int,
) -> Path:
    """Return the cache file path for dataset-wide ORIC."""
    cache_dir = (
        Path(dataset_root) / "dataset_oric"
        / f"{edge_model}_vs_{cloud_model}"
    )
    return cache_dir / (
        f"{split_name}_seed{seed}_ctx{context_size}_draws{context_draws}.pkl"
    )


def _get_dataset_wide_oric_cached(
    cfg,
    derived_dir: Path,
    video_names: np.ndarray,
    frame_ids: np.ndarray,
    split_name: str,
) -> np.ndarray:
    """Compute (or load cached) dataset-wide ORIC for one split.

    Returns an (N, 3) float64 array of [oric_11pt, oric_allpoint, oric_coco]
    aligned with the supplied *video_names* / *frame_ids* arrays.

    The cache file is written atomically (write tmp then rename) so
    concurrent processes will never see a half-written file.
    """
    cache_file = _dataset_oric_cache_path(
        cfg.dataset.root, cfg.dataset.edge_model,
        cfg.dataset.cloud_model, cfg.seed, split_name,
        cfg.oric_context_size, cfg.oric_context_draws,
    )

    # --- Try loading from cache -------------------------------------------
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as fh:
                cached = pickle.load(fh)
            if (cached.get("num_frames") == len(video_names)
                    and cached.get("edge_model") == cfg.dataset.edge_model
                    and cached.get("cloud_model") == cfg.dataset.cloud_model
                    and cached.get("context_size") == cfg.oric_context_size
                    and cached.get("context_draws") == cfg.oric_context_draws):
                oric_dict = cached["oric"]
                result = np.zeros((len(video_names), 3), dtype=np.float64)
                ok = 0
                for i, (v, fid) in enumerate(zip(video_names, frame_ids)):
                    vals = oric_dict.get((str(v), int(fid)))
                    if vals is not None:
                        result[i, 0] = vals["oric_11pt"]
                        result[i, 1] = vals["oric_allpoint"]
                        result[i, 2] = vals["oric_coco"]
                        ok += 1
                if ok == len(video_names):
                    log.cached(f"Dataset-wide ORIC for {split_name} "
                               f"({log.fmt_count(ok)} frames)")
                    return result
                else:
                    log.info(f"Cache partial ({ok}/{len(video_names)}), "
                             "recomputing dataset-wide ORIC ...")
        except Exception as exc:
            log.warn(f"Cache load failed ({exc}), recomputing ...")

    # --- Compute ----------------------------------------------------------
    log.info(f"Computing dataset-wide ORIC for {split_name} "
             f"({log.fmt_count(len(video_names))} frames) ...")
    raw_boxes = _load_raw_boxes_for_frames(derived_dir, video_names, frame_ids)
    oric_dict = compute_dataset_wide_oric(
        raw_boxes,
        quiet=False,
        context_size=cfg.oric_context_size,
        context_draws=cfg.oric_context_draws,
    )

    # --- Write cache atomically -------------------------------------------
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp." + str(os.getpid()))
    try:
        with open(tmp, "wb") as fh:
            pickle.dump({
                "num_frames": len(video_names),
                "edge_model": cfg.dataset.edge_model,
                "cloud_model": cfg.dataset.cloud_model,
                "seed": cfg.seed,
                "context_size": cfg.oric_context_size,
                "context_draws": cfg.oric_context_draws,
                "oric": oric_dict,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(cache_file)        # atomic on POSIX
    except OSError:
        # Another process may have written concurrently — that's fine.
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    # --- Align to caller's frame order ------------------------------------
    result = np.zeros((len(video_names), 3), dtype=np.float64)
    for i, (v, fid) in enumerate(zip(video_names, frame_ids)):
        vals = oric_dict.get((str(v), int(fid)))
        if vals is not None:
            result[i, 0] = vals["oric_11pt"]
            result[i, 1] = vals["oric_allpoint"]
            result[i, 2] = vals["oric_coco"]
    return result


def _gain_state_label(value: float, atol: float = 1e-12) -> str:
    value = float(value)
    if abs(value) <= atol:
        return "neutral"
    return "beneficial" if value > 0 else "harmful"


def _gain_state_indicators(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    beneficial = (values > 0).astype(np.float32)
    harmful = (values < 0).astype(np.float32)
    neutral = np.isclose(values, 0.0).astype(np.float32)
    return beneficial, harmful, neutral


# ---------------------------------------------------------------------------
#  Image-level proxy metrics (entropy, complexity)
# ---------------------------------------------------------------------------

def _load_image_metadata(
    img_path: str,
    size: int = 128,
    compute_proxy_metrics: bool = False,
) -> dict:
    """Load image size and, optionally, image-level proxy metrics."""
    defaults = {
        "entropy": 0.0,
        "img_complexity": 0.0,
        "img_width": float(size),
        "img_height": float(size),
    }
    try:
        from PIL import Image
        with Image.open(img_path) as pil_img:
            img_width, img_height = pil_img.size
            if not compute_proxy_metrics:
                return {
                    "entropy": 0.0,
                    "img_complexity": 0.0,
                    "img_width": float(img_width),
                    "img_height": float(img_height),
                }
            rgb = pil_img.convert("RGB")
            resized = rgb.resize((size, size))
        arr = np.asarray(resized, dtype=np.float32) / 255.0
    except Exception:
        return defaults

    gray = np.mean(arr, axis=2)

    # Global grayscale entropy (64-bin histogram)
    hist, _ = np.histogram((gray * 255).astype(np.uint8), bins=64, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

    # Laplacian variance (edge density / texture complexity)
    lap_x = np.diff(gray, n=2, axis=1)
    lap_y = np.diff(gray, n=2, axis=0)
    img_complexity = float(lap_x.var() + lap_y.var())

    return {
        "entropy": entropy,
        "img_complexity": img_complexity,
        "img_width": float(img_width),
        "img_height": float(img_height),
    }


def _flatten_lcer_vector(metric_family: str, vector: np.ndarray) -> dict:
    return {
        f"lcer_vec_{metric_family}_{error_name}": float(vector[idx])
        for idx, error_name in enumerate(LCER_ERROR_TYPES)
    }


def _flatten_finegrained_vector(metric_family: str, vector: np.ndarray) -> dict:
    return {
        f"finegrained_vec_{metric_family}_{name}": float(vector[idx])
        for idx, name in enumerate(SCENARIO_COMPONENTS)
    }


# ---------------------------------------------------------------------------
#  Multiprocessing helpers for parallel video derivation (#3)
# ---------------------------------------------------------------------------

_pool_dataset = None


def _init_pool_worker(ds):
    """Pool initializer: make the shared dataset available in each worker."""
    global _pool_dataset
    _pool_dataset = ds
    # Prevent numpy BLAS thread oversubscription: with N forked workers each
    # spawning their own BLAS threads, the total thread count can explode.
    # Per-frame numpy arrays are small, so single-threaded BLAS is optimal.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _derive_video_worker(args):
    """Worker wrapper that passes the process-global dataset to _derive_video."""
    (v, edge_model, cloud_model, data_root, output_dir, conf_threshold,
     oric_context_size, oric_context_draws, families, required_inputs) = args
    ok, edge, cloud, gt = _derive_video(
        v, edge_model, cloud_model, data_root, output_dir,
        conf_threshold, _pool_dataset, quiet=True,
        oric_context_size=oric_context_size,
        oric_context_draws=oric_context_draws,
        families=set(families),
        required_inputs=set(required_inputs),
    )
    return v, ok, edge, cloud, gt

# ========================================================================
#  2a. Derive — raw detections + GT → per-video CSVs
# ========================================================================

def _derive_video(
    video_name: str,
    edge_model: str,
    cloud_model: str,
    data_root: Path,
    output_dir: Path,
    conf_threshold: float = 0.5,
    dataset=None,
    quiet: bool = False,
    oric_context_size: int = 0,
    oric_context_draws: int = 1,
    families: Optional[set[str]] = None,
    required_inputs: Optional[set[str]] = None,
) -> Tuple[bool, List[Tuple], List[Tuple], List[Tuple]]:
    """Process one video: extract features, compute metrics, save CSV.

    Returns (ok, edge_dets, cloud_dets, gt_objects) for dataset-wide mAP.
    """
    split = dataset.get_video_split(video_name)
    families = families or set()
    required_inputs = required_inputs or set()
    need_lcer = "lcer" in families or "adaptive_scenario" in families
    need_finegrained = "adaptive_scenario" in families
    need_bwd = "bwd" in families
    need_srrm = "srrm" in families
    need_image_proxy = "image_proxy" in families
    need_topk = "proposal_topk" in required_inputs
    need_full_proposal = "proposal_full" in required_inputs

    try:
        gt_data = dataset.load_ground_truth(video_name)
    except FileNotFoundError:
        return False, [], [], []

    edge_det_file = dataset.get_detection_file(video_name, edge_model)
    cloud_det_file = dataset.get_detection_file(video_name, cloud_model)

    if not edge_det_file.exists() or not cloud_det_file.exists():
        return False, [], [], []

    edge_det_data = load_detection_results(edge_det_file)
    cloud_det_data = load_detection_results(cloud_det_file)
    ignored_regions = dataset.get_ignored_regions(video_name)

    all_frame_ids = sorted(
        set(gt_data.keys()) | set(edge_det_data.keys()) | set(cloud_det_data.keys())
    )
    if not all_frame_ids:
        return False, [], [], []

    accum_edge, accum_cloud, accum_gt = [], [], []
    rows = []
    raw_boxes = []

    for fid in all_frame_ids:
        e_dets_all = dataset.filter_detections_by_ignored_regions(
            edge_det_data.get(fid, []), ignored_regions)
        c_dets_all = dataset.filter_detections_by_ignored_regions(
            cloud_det_data.get(fid, []), ignored_regions)

        e_dets_feat = [d for d in e_dets_all if d[4] >= conf_threshold]
        c_dets_feat = [d for d in c_dets_all if d[4] >= conf_threshold]

        e_feats = {f"edge_{k}": v for k, v in extract_detection_features(e_dets_feat).items()}
        c_feats = {f"cloud_{k}": v for k, v in extract_detection_features(c_dets_feat).items()}

        gt_raw = gt_data.get(fid, [])
        gt_objs = dataset.filter_gt_by_ignored_regions(gt_raw, ignored_regions)

        # For thresholded detections we only need precision/recall/f1 (no AP),
        # so avoid the expensive full compute_all_metrics call (#4).
        e_tp, e_fp, e_fn = match_detections_to_gt(e_dets_feat, gt_objs)
        e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
        e_f1 = 2 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0

        c_tp, c_fp, c_fn = match_detections_to_gt(c_dets_feat, gt_objs)
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0

        edge_map = compute_11point_ap(e_dets_all, gt_objs, iou_threshold=0.5)
        cloud_map = compute_11point_ap(c_dets_all, gt_objs, iou_threshold=0.5)
        edge_map_coco = compute_coco_frame_ap(e_dets_all, gt_objs)
        cloud_map_coco = compute_coco_frame_ap(c_dets_all, gt_objs)
        edge_map_coco50 = compute_allpoint_ap(
            e_dets_all, gt_objs, iou_threshold=0.5
        )
        cloud_map_coco50 = compute_allpoint_ap(
            c_dets_all, gt_objs, iou_threshold=0.5
        )

        img_path = dataset.get_image_path(video_name, fid)

        # Image metadata is always needed for proposal normalization; the
        # expensive proxy metrics are only computed when explicitly required.
        img_pm = (
            _load_image_metadata(
                str(img_path),
                compute_proxy_metrics=need_image_proxy,
            )
            if img_path.exists()
            else {
                "entropy": 0.0,
                "img_complexity": 0.0,
                "img_width": 128.0,
                "img_height": 128.0,
            }
        )
        edge_prop_feats = (
            extract_topk_proposal_features(
                e_dets_all,
                image_width=img_pm["img_width"],
                image_height=img_pm["img_height"],
                prefix="edge",
            )
            if need_topk
            else {}
        )
        rule_confs, rule_areas = (
            extract_full_proposal_rule_inputs(
                e_dets_all,
                image_width=img_pm["img_width"],
                image_height=img_pm["img_height"],
            )
            if need_full_proposal
            else (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        )
        edge_dets_05 = [d for d in e_dets_all if d[4] >= 0.5]
        cloud_dets_05 = [d for d in c_dets_all if d[4] >= 0.5]

        for d in e_dets_all:
            accum_edge.append((d, video_name, fid))
        for d in c_dets_all:
            accum_cloud.append((d, video_name, fid))
        for g in gt_objs:
            accum_gt.append((g, video_name, fid))

        # Compute AP at IoU=0.75 for high_iou_gain metric
        edge_map_75 = compute_allpoint_ap(e_dets_all, gt_objs, iou_threshold=0.75)
        cloud_map_75 = compute_allpoint_ap(c_dets_all, gt_objs, iou_threshold=0.75)

        # We defer ORIC computation until we have the full video loaded
        # We will add it to the rows down below
        gt_count = len(gt_objs)
        delta_tp = c_tp - e_tp
        rows.append({
            "frame_id": fid,
            "video_name": video_name,
            "split": split,
            "image_path": str(img_path) if img_path.exists() else "",
            **e_feats, **c_feats,
            "gt_count": gt_count,
            "edge_tp": e_tp,
            "edge_fp": e_fp,
            "edge_fn": e_fn,
            "edge_map": edge_map,
            "edge_precision": e_prec,
            "edge_recall": e_rec,
            "edge_f1": e_f1,
            "cloud_tp": c_tp,
            "cloud_fp": c_fp,
            "cloud_fn": c_fn,
            "cloud_map": cloud_map,
            "cloud_precision": c_prec,
            "cloud_recall": c_rec,
            "cloud_f1": c_f1,
            "delta_tp": delta_tp,
            "delta_fp": c_fp - e_fp,
            "delta_fn": c_fn - e_fn,
            "gain_11pt": cloud_map - edge_map,       # Alias for the gain proxy-metric name
            "gain_state_11pt": _gain_state_label(cloud_map - edge_map),
            "map_ratio": cloud_map / max(edge_map, 0.01),
            "f1_gain": c_f1 - e_f1,
            "recall_gain": c_rec - e_rec,
            "edge_map_coco": edge_map_coco,
            "cloud_map_coco": cloud_map_coco,
            "gain_coco": cloud_map_coco - edge_map_coco,
            "gain_state_coco": _gain_state_label(cloud_map_coco - edge_map_coco),
            "edge_map_coco50": edge_map_coco50,
            "cloud_map_coco50": cloud_map_coco50,
            "gain_allpoint": cloud_map_coco50 - edge_map_coco50,
            "gain_state_allpoint": _gain_state_label(cloud_map_coco50 - edge_map_coco50),
            "count_gain_05": float(len(cloud_dets_05) - len(edge_dets_05)),
            # --- New proxy metrics ---
            "rescue_ratio_50": delta_tp / max(1, gt_count),
            "rescue_count_50": delta_tp,
            "precision_gain_50": c_prec - e_prec,
            "fp_reduction_50": e_fp - c_fp,
            "worst_case_gain": min(cloud_map - edge_map, cloud_map_coco - edge_map_coco),
            "high_iou_gain_75": cloud_map_75 - edge_map_75,
            "edge_miss_rate": e_fn / max(1, gt_count),
            "edge_uncertainty": 1.0 - e_feats.get("edge_conf_mean", 0.0),
            "conf_spread": e_feats.get("edge_conf_std", 0.0),
            **edge_prop_feats,
        })
        if need_image_proxy:
            rows[-1]["entropy"] = img_pm["entropy"]
            rows[-1]["img_complexity"] = img_pm["img_complexity"]
        if need_full_proposal:
            rows[-1]["edge_rule_conf_seq"] = encode_float_sequence(rule_confs)
            rows[-1]["edge_rule_area_seq"] = encode_float_sequence(rule_areas)

        raw_boxes.append({
            "frame_id": fid,
            "video_name": video_name,
            "edge_dets": e_dets_all,
            "cloud_dets": c_dets_all,
            "gt_objs": gt_objs,
        })

    # Precompute dataset-wide AP for this video to serve as the ORIC baseline
    base_res_50 = compute_dataset_ap(accum_edge, accum_gt, iou_threshold=0.5)
    base_res_coco = compute_dataset_ap(accum_edge, accum_gt, iou_thresholds=COCO_IOU_THRESHOLDS)

    # Compute proxy metrics (ORIC)
    proxy_metrics = compute_proxy_metrics(
        raw_boxes=raw_boxes,
        video_name=video_name,
        base_ap50_11pt=base_res_50["ap"],
        base_ap50_allpoint=base_res_50.get("ap_allpoint", base_res_50["ap"]),
        base_ap_coco=base_res_coco["ap"],
        quiet=quiet,
        context_size=oric_context_size,
        context_draws=oric_context_draws,
    )
    lcer_vectors = compute_lcer_vectors(raw_boxes) if need_lcer else {}

    # SRRM and BWD proxy metrics (per-frame spatial map + scalar)
    srrm_matrices = {}
    if need_srrm:
        for box_data in raw_boxes:
            fid = box_data["frame_id"]
            srrm_matrices[fid] = compute_srrm(box_data, grid_size=8)

    raw_boxes_by_frame = {
        int(box_data["frame_id"]): box_data for box_data in raw_boxes
    }

    # Attach proxy metrics to rows
    for row in rows:
        fid = row["frame_id"]
        pm = proxy_metrics.get(fid, {"oric_11pt": 0.0, "oric_allpoint": 0.0, "oric_coco": 0.0})
        row.update(pm)
        vecs = lcer_vectors.get(fid)
        if vecs is not None:
            row.update(_flatten_lcer_vector("11pt", vecs["11pt"]))
            row.update(_flatten_lcer_vector("allpoint", vecs["allpoint"]))
            row.update(_flatten_lcer_vector("coco", vecs["coco"]))
        frame_data = raw_boxes_by_frame.get(int(fid), {})
        if need_bwd:
            row["bwd"] = compute_bwd(frame_data)
        if need_finegrained:
            row.update(_flatten_finegrained_vector(
                "coco",
                compute_finegrained_proxy_vector(
                    frame_data,
                    metric_family="coco",
                    lcer_vector=vecs["coco"] if vecs is not None else None,
                ),
            ))

    pd.DataFrame(rows).to_csv(output_dir / f"{video_name}.csv", index=False)

    with open(output_dir / f"{video_name}_boxes.pkl", "wb") as f:
        pickle.dump(raw_boxes, f)

    if need_srrm:
        with open(output_dir / f"{video_name}_srrm.pkl", "wb") as f:
            pickle.dump(srrm_matrices, f)
    else:
        (output_dir / f"{video_name}_srrm.pkl").unlink(missing_ok=True)

    return True, accum_edge, accum_cloud, accum_gt


def derive_features(cfg: PipelineConfig) -> None:
    """Sub-phase 2a: derive per-video CSVs from raw detections + GT."""
    from .prepare_config import _required_derived_columns, _video_cache_complete

    out = cfg.derived_dir
    out.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg.dataset.root)
    data_root.mkdir(parents=True, exist_ok=True)

    ds = get_dataset(cfg.dataset.name, data_root)

    train_vids = ds.get_video_names("train")
    test_vids = ds.get_video_names("test")
    videos = train_vids + test_vids
    families = _required_proxy_families(cfg)
    required_inputs = _required_prepare_inputs(cfg)
    require_srrm = "srrm" in families

    # ------------------------------------------------------------------
    # #1  Skip already-cached videos — only derive the incomplete ones
    # ------------------------------------------------------------------
    videos_to_derive = videos
    required_columns = _required_derived_columns(families, required_inputs)

    if not cfg.force_re_derive:
        videos_to_derive = [
            v for v in videos
            if not _video_cache_complete(
                out, v, required_columns, require_srrm=require_srrm,
            )
        ]
        if not videos_to_derive:
            log.cached(f"Derived data found ({len(videos)} CSVs) "
                       "-- use force_re_derive=True to regenerate")
            return
        n_cached = len(videos) - len(videos_to_derive)
        log.info(f"{log.fmt_count(n_cached)} cached, "
                 f"{log.fmt_count(len(videos_to_derive))} to derive")

    label = ds.sequence_label
    log.kv_group([
        ("Edge model", cfg.dataset.edge_model),
        ("Cloud model", cfg.dataset.cloud_model),
        (label.capitalize(), f"{log.fmt_count(len(videos))} total, "
         f"{log.fmt_count(len(videos_to_derive))} to derive"),
        ("Conf threshold", cfg.dataset.conf_threshold),
    ])

    # ------------------------------------------------------------------
    # Process videos (parallel #3 or sequential)
    # ------------------------------------------------------------------
    ok_count = 0
    all_e, all_c, all_g = [], [], []
    failed_videos: list[str] = []

    # Resolve worker count: 0 = auto, 1 = sequential, N = N workers
    requested = cfg.derive_num_workers
    if requested <= 0:
        n_workers = min(os.cpu_count() or 1, len(videos_to_derive), 8)
    else:
        n_workers = min(requested, len(videos_to_derive))

    if n_workers > 1 and len(videos_to_derive) > 1:
        args_list = [
            (v, cfg.dataset.edge_model, cfg.dataset.cloud_model,
             data_root, out, cfg.dataset.conf_threshold,
             cfg.oric_context_size, cfg.oric_context_draws,
             tuple(sorted(families)), tuple(sorted(required_inputs)))
            for v in videos_to_derive
        ]
        global _pool_dataset
        _pool_dataset = ds
        ctx = mp.get_context('fork')
        log.info(f"Using {n_workers} parallel workers")
        with ctx.Pool(processes=n_workers,
                       initializer=_init_pool_worker,
                       initargs=(ds,)) as pool:
            results = list(tqdm(
                pool.imap_unordered(_derive_video_worker, args_list),
                total=len(args_list),
                desc="  Deriving",
            ))
        for video_name, ok, ed, cd, gd in results:
            if ok:
                ok_count += 1
                all_e.extend(ed)
                all_c.extend(cd)
                all_g.extend(gd)
            else:
                failed_videos.append(video_name)
    else:
        for v in tqdm(videos_to_derive, desc="  Deriving"):
            ok, ed, cd, gd = _derive_video(
                v, cfg.dataset.edge_model, cfg.dataset.cloud_model,
                data_root, out, cfg.dataset.conf_threshold, ds,
                oric_context_size=cfg.oric_context_size,
                oric_context_draws=cfg.oric_context_draws,
                families=families,
                required_inputs=required_inputs,
            )
            if ok:
                ok_count += 1
                all_e.extend(ed)
                all_c.extend(cd)
                all_g.extend(gd)
            else:
                failed_videos.append(v)

    # ------------------------------------------------------------------
    # Load cached videos' detection data for dataset-wide AP (#1)
    # ------------------------------------------------------------------
    cached_videos = [v for v in videos if v not in set(videos_to_derive)]
    if cached_videos:
        log.info(f"Loading {log.fmt_count(len(cached_videos))} cached video "
                 "detection data for dataset-wide AP")
        for v in cached_videos:
            pkl_path = out / f"{v}_boxes.pkl"
            if not pkl_path.exists():
                failed_videos.append(v)
                continue
            with open(pkl_path, "rb") as f:
                raw_boxes = pickle.load(f)
            for data in raw_boxes:
                fid = data["frame_id"]
                for d in data["edge_dets"]:
                    all_e.append((d, v, fid))
                for d in data["cloud_dets"]:
                    all_c.append((d, v, fid))
                for g in data["gt_objs"]:
                    all_g.append((g, v, fid))
            ok_count += 1

    missing_videos = [
        v for v in videos
        if not _video_cache_complete(
            out, v, required_columns, require_srrm=require_srrm,
        )
    ]
    missing_videos = sorted(set(missing_videos + failed_videos))
    if missing_videos:
        example = missing_videos[0]
        raise FileNotFoundError(
            "Prepare could not build a complete derived cache for the native "
            f"dataset split. Missing artifacts for {len(missing_videos)} video(s) "
            f"(e.g. {example}). Ensure detection outputs exist for both train "
            "and test splits before running prepare."
        )

    if all_g:
        log.info(f"Computing dataset-wide AP ({log.fmt_count(len(all_e))} edge, "
                 f"{log.fmt_count(len(all_c))} cloud, "
                 f"{log.fmt_count(len(all_g))} GT)")
        eg50 = compute_dataset_ap(all_e, all_g, iou_threshold=0.5)
        cg50 = compute_dataset_ap(all_c, all_g, iou_threshold=0.5)
        egc = compute_dataset_ap(all_e, all_g, iou_thresholds=COCO_IOU_THRESHOLDS)
        cgc = compute_dataset_ap(all_c, all_g, iou_thresholds=COCO_IOU_THRESHOLDS)

        summary = {
            "edge_model": cfg.dataset.edge_model,
            "cloud_model": cfg.dataset.cloud_model,
            "num_videos": ok_count,
            "num_gt": len(all_g),
            "num_edge_dets": len(all_e),
            "num_cloud_dets": len(all_c),
            "edge_ap50": eg50["ap"], "cloud_ap50": cg50["ap"],
            "edge_ap_coco": egc["ap"], "cloud_ap_coco": cgc["ap"],
        }
        (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

        log.table(
            ["Model", "AP@0.5", "AP@COCO"],
            [
                ["Edge", f"{eg50['ap']:.4f}", f"{egc['ap']:.4f}"],
                ["Cloud", f"{cg50['ap']:.4f}", f"{cgc['ap']:.4f}"],
            ],
            col_widths=[8, 10, 10],
        )

    with open(out / "metadata.txt", "w") as f:
        f.write(f"Edge model: {cfg.dataset.edge_model}\n")
        f.write(f"Cloud model: {cfg.dataset.cloud_model}\n")
        f.write(f"Confidence: {cfg.dataset.conf_threshold}\n")
        f.write(f"Videos: {len(videos)}\n")
        f.write(f"Context draws: {cfg.oric_context_draws}\n")

    n_derived = len(videos_to_derive)
    n_cached = len(cached_videos)
    log.success(f"Derived {n_derived} new + {n_cached} cached = "
                f"{ok_count}/{len(videos)} videos")
    log.arrow(str(out))


