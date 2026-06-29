"""Proxy-family/input requirements, cache signature, and cache validity helpers."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from config import PipelineConfig
from config.scenarios import normalize_scenario_weight_map

from .. import log
from ..models import ESTIMATOR_REGISTRY

# Bump when the prepared-data schema changes in a way that should invalidate
# stale cached ``data.npz`` files.
PREPARED_SCHEMA_VERSION = 10


def _required_proxy_families(cfg: PipelineConfig) -> set:
    """Determine which proxy-metric families the enabled estimators need.

    Always includes base families (gain, oric, moric, moric_plus, moric_star,
    phi_moric).  Additional families are only included if at least one enabled
    approach references them in its proxy_metric or target_spec.
    """
    families = {"gain", "oric", "moric", "moric_plus", "moric_star", "phi_moric", "sigmoric"}
    for pcfg in cfg.enabled_approaches():
        all_keys = [pcfg.proxy_metric or ""] + list((pcfg.target_spec or {}).values())
        for k in all_keys:
            if k.startswith("lcer") or k.startswith("csr"):
                families.add("lcer")
            if k == "srrm":
                families.add("srrm")
            if "bwd" in k:
                families.add("bwd")
            if k in {"entropy", "img_complexity"}:
                families.add("image_proxy")
            if (
                k.startswith("scenario_utility")
                or k.startswith("finegrained_vec")
            ):
                families.add("adaptive_scenario")
            if (
                "dataset_" in k
                or k in {"moric_11pt", "moric_allpoint", "moric_coco"}
            ):
                families.add("dataset_oric")
            # Novel proxy-metric families
            if k.startswith("rescue_ratio") or k.startswith("rescue_count"):
                families.add("rescue")
            if k.startswith("precision_gain") or k.startswith("fp_reduction"):
                families.add("precision_gain")
            if k == "f1_gain_50" or k == "f1_gain":
                families.add("f1_gain")
            if k.startswith("worst_case_gain") or k.startswith("high_iou_gain"):
                families.add("conservative")
            if k.startswith("edge_miss_rate"):
                families.add("edge_difficulty")
            if k.startswith("edge_uncertainty") or k.startswith("conf_spread"):
                families.add("confidence")
            if k in {"offload_binary", "gain_top_quartile"}:
                families.add("binary")
    return families


def _required_prepare_inputs(cfg: PipelineConfig) -> set[str]:
    """Determine which non-tabular prepared inputs enabled estimators require."""
    required: set[str] = set()
    for pcfg in cfg.enabled_approaches():
        cls = ESTIMATOR_REGISTRY[pcfg.registry_key]
        input_key = getattr(cls, "input_key", "default")
        if input_key == "proposal":
            required.add("proposal_topk")
        elif input_key == "proposal_full":
            required.add("proposal_full")
        if pcfg.feature_type == "image":
            required.add("image_paths")
    return required


def _required_derived_columns(families: set[str], inputs: set[str]) -> set[str]:
    from ..error_decomposition import LCER_ERROR_TYPES
    from ..features import topk_proposal_feature_columns
    from config.scenarios import SCENARIO_COMPONENTS

    required: set[str] = {
        "frame_id", "video_name", "split", "image_path",
        "edge_map", "cloud_map",
        "edge_map_coco", "cloud_map_coco",
        "edge_map_coco50", "cloud_map_coco50",
        "gain_11pt", "gain_allpoint", "gain_coco",
        "oric_11pt", "oric_allpoint", "oric_coco",
        "gt_count", "count_gain_05",
        "edge_tp", "edge_fp", "edge_fn",
        "cloud_tp", "cloud_fp", "cloud_fn",
        "delta_tp", "delta_fp", "delta_fn",
        "gain_state_11pt", "gain_state_allpoint", "gain_state_coco",
        "oric_11pt_draw_std", "oric_11pt_sign_consistency", "oric_11pt_rank_consistency",
        "oric_allpoint_draw_std", "oric_allpoint_sign_consistency", "oric_allpoint_rank_consistency",
        "oric_coco_draw_std", "oric_coco_sign_consistency", "oric_coco_rank_consistency",
    }
    if "proposal_topk" in inputs:
        required.update(topk_proposal_feature_columns("edge"))
    if "proposal_full" in inputs:
        required.update({"edge_rule_conf_seq", "edge_rule_area_seq"})
    if "image_paths" in inputs:
        required.add("image_path")
    if "lcer" in families:
        for metric_family in ("11pt", "allpoint", "coco"):
            required.update(
                f"lcer_vec_{metric_family}_{error_name}"
                for error_name in LCER_ERROR_TYPES
            )
    if "adaptive_scenario" in families:
        required.update(
            f"finegrained_vec_coco_{name}" for name in SCENARIO_COMPONENTS
        )
    if "bwd" in families:
        required.add("bwd")
    if "image_proxy" in families:
        required.update({"entropy", "img_complexity"})
    # Novel proxy metric columns
    novel_columns = {
        "rescue_ratio_50", "rescue_count_50",
        "precision_gain_50", "fp_reduction_50",
        "worst_case_gain", "high_iou_gain_75",
        "edge_miss_rate", "edge_uncertainty", "conf_spread",
    }
    # Always include novel columns in required set — they are cheap to
    # compute (derived from existing TP/FP/FN and edge features) and
    # needed by the novel metric families.
    novel_families = {
        "rescue", "precision_gain", "f1_gain", "conservative",
        "edge_difficulty", "confidence", "binary",
    }
    if novel_families & families:
        required.update(novel_columns)
    return required


def _video_cache_complete(
    derived_dir: Path,
    video_name: str,
    required_columns: set[str],
    require_srrm: bool,
) -> bool:
    csv_path = derived_dir / f"{video_name}.csv"
    if not csv_path.exists():
        return False

    boxes_path = derived_dir / f"{video_name}_boxes.pkl"
    if not boxes_path.exists():
        log.info(f"Derived cache stale: {boxes_path.name} is missing")
        return False

    if require_srrm:
        srrm_path = derived_dir / f"{video_name}_srrm.pkl"
        if not srrm_path.exists():
            log.info(f"Derived cache stale: {srrm_path.name} is missing")
            return False

    if required_columns:
        cols = set(pd.read_csv(csv_path, nrows=0).columns)
        missing = sorted(required_columns - cols)
        if missing:
            log.info(
                f"Derived cache stale: {csv_path.name} is missing {len(missing)} required columns "
                f"(e.g. {missing[0]})"
            )
            return False

    return True


def _derived_input_fingerprint(
    data_dir: Path,
    families: set[str],
    inputs: set[str],
) -> str:
    """Fingerprint the current derived inputs used to build prepared data."""
    hasher = hashlib.sha256()
    patterns = ["*.csv", "*_boxes.pkl"]
    if "srrm" in families:
        patterns.append("*_srrm.pkl")

    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(data_dir.glob(pattern)):
            if path in seen or not path.exists():
                continue
            seen.add(path)
            stat = path.stat()
            hasher.update(path.name.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    hasher.update(json.dumps(sorted(families)).encode("utf-8"))
    hasher.update(json.dumps(sorted(inputs)).encode("utf-8"))
    hasher.update(str(data_dir.resolve()).encode("utf-8"))
    return hasher.hexdigest()


def _prepare_cache_signature(
    cfg: PipelineConfig,
    families: set[str],
    required_inputs: set[str],
) -> dict:
    adaptive_weights_sig = None
    if "adaptive_scenario" in families:
        weights = normalize_scenario_weight_map(cfg.adaptive_scenario_weights)
        adaptive_weights_sig = json.dumps(
            {name: weights[name].tolist() for name in sorted(weights)},
            sort_keys=True,
        )

    return {
        "prepare_schema_version": PREPARED_SCHEMA_VERSION,
        "seed": int(cfg.seed),
        "test_ratio": float(cfg.dataset.test_ratio),
        "sample_fraction": float(cfg.dataset.sample_fraction),
        "required_families": sorted(families),
        "required_inputs": sorted(required_inputs),
        "derived_input_dir": str(Path(cfg.output.data_dir).resolve()),
        "derived_input_fingerprint": _derived_input_fingerprint(
            Path(cfg.output.data_dir), families, required_inputs,
        ),
        "adaptive_scenario_weights_signature": adaptive_weights_sig,
    }


def _prepared_cache_matches(
    cfg: PipelineConfig,
    meta: dict,
    families: set[str],
    required_inputs: set[str],
) -> bool:
    expected = _prepare_cache_signature(cfg, families, required_inputs)
    return all(meta.get(key) == value for key, value in expected.items())
