"""Phase 1 – Detect: run detection models on dataset images.

Runs edge and cloud detection models on all images and saves per-video
detection text files. Results are cached — if detection files already
exist for a video/model pair, that pair is skipped.
"""

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from config import PipelineConfig

from ..dataset import get_dataset, model_dir_name, resolve_base_dataset
from ..detector import Detector
from .. import log


def _detection_meta_path(det_file: Path) -> Path:
    """Sidecar metadata file for a cached detection text file."""
    return det_file.with_suffix(det_file.suffix + ".meta.json")


def _detections_exist(ds, video_name: str, model_name: str,
                      conf_threshold: float) -> bool:
    """Check if detection file already exists for this video/model."""
    det_file = ds.get_detection_file(video_name, model_name)
    meta_file = _detection_meta_path(det_file)
    if not (det_file.exists() and det_file.stat().st_size > 0 and meta_file.exists()):
        return False
    try:
        meta = json.loads(meta_file.read_text())
    except json.JSONDecodeError:
        return False
    return float(meta.get("conf_threshold", -1.0)) == float(conf_threshold)


def _process_video(ds, video_name: str, model_name: str,
                   detector: Detector, batch_size: int = 32) -> None:
    """Run detection on all frames of a video/chunk and save results."""
    frames = ds.iter_frames(video_name)
    if not frames:
        return

    det_file = ds.get_detection_file(video_name, model_name)
    det_file.parent.mkdir(parents=True, exist_ok=True)

    detections = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_paths = [str(p) for _, p in batch]
        batch_results = detector.detect_batch(batch_paths)
        for (fid, _), frame_dets in zip(batch, batch_results):
            for obj_id, (left, top, width, height, conf, cls) in enumerate(frame_dets, start=1):
                detections.append(
                    f"{fid},{obj_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.6f},{cls}\n"
                )
    det_file.write_text("".join(detections))
    _detection_meta_path(det_file).write_text(json.dumps({
        "model_name": model_name,
        "conf_threshold": float(detector.conf_threshold),
    }))


def run(cfg: PipelineConfig) -> None:
    """Execute the detection phase for edge and cloud models."""
    with log.phase_timer(1):
        import src.detector as detector_module
        if getattr(cfg, "device", "auto") != "auto":
            detector_module.FORCE_DEVICE = cfg.device

        data_root = Path(cfg.dataset.root)
        data_root.mkdir(parents=True, exist_ok=True)

        ds = get_dataset(cfg.dataset.name, data_root)
        base_name = resolve_base_dataset(cfg.dataset.name)
        vehicle_only = base_name in ("ua-detrac", "detrac")
        models_to_run = [cfg.dataset.edge_model, cfg.dataset.cloud_model]

        # Gather videos
        splits = ["train", "test"] if cfg.dataset.detection_split == "all" else [cfg.dataset.detection_split]
        videos: List[str] = []
        for s in splits:
            videos.extend(ds.get_video_names(s))

        label = ds.sequence_label
        log.kv_group([
            ("Dataset", cfg.dataset.name),
            ("Edge model", cfg.dataset.edge_model),
            ("Cloud model", cfg.dataset.cloud_model),
            (label.capitalize(), log.fmt_count(len(videos))),
            ("Batch size", cfg.dataset.detection_batch_size),
        ])

        for model_name in models_to_run:
            log.subsection(model_name)

            # Check which videos still need detection
            todo = [
                v for v in videos
                if not _detections_exist(
                    ds, v, model_name, cfg.dataset.detection_conf
                )
            ]
            n_cached = len(videos) - len(todo)

            if not todo:
                log.cached(f"All {log.fmt_count(len(videos))} {label} cached")
                continue

            log.info(f"{log.fmt_count(n_cached)} cached, "
                     f"{log.fmt_count(len(todo))} to process")
            detector = Detector(model_name, cfg.dataset.detection_conf,
                                vehicle_only=vehicle_only)

            for vid in tqdm(todo, desc=f"    {model_dir_name(model_name)}"):
                _process_video(ds, vid, model_name, detector,
                               batch_size=cfg.dataset.detection_batch_size)

            log.success(f"Processed {log.fmt_count(len(todo))} {label}")
