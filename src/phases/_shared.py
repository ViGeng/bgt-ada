"""Shared utilities for train and evaluate phases."""

import hashlib
import json
from pathlib import Path
from typing import Any

from config import ApproachConfig, PipelineConfig

from ..models import ESTIMATOR_REGISTRY

# Project root (repo top-level) for resolving relative image paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def approach_config_signature(pcfg: ApproachConfig) -> str:
    """SHA256 hash of approach config fields that affect training output.

    Used to detect stale checkpoints when config changes without renaming.
    """
    data = {
        "estimator": {
            "name": pcfg.estimator.name,
            "base_model": pcfg.estimator.base_model,
            "feature_type": pcfg.estimator.feature_type,
            "stage": pcfg.estimator.stage,
            "proxy_metric": pcfg.estimator.proxy_metric,
            "target_spec": pcfg.estimator.target_spec,
            "loss": pcfg.estimator.loss,
            "params": pcfg.estimator.params,
        },
        "offloader": {
            "name": pcfg.offloader.name,
            "policy_id": pcfg.offloader.policy_id,
            "params": pcfg.offloader.params,
        },
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def config_sidecar_path(ckpt_path: Path) -> Path:
    """Return .config.json sidecar path for a checkpoint."""
    return ckpt_path.with_suffix(".config.json")


def checkpoint_path(cfg: PipelineConfig, pcfg: ApproachConfig, seed: int | None = None) -> Path:
    """Return the checkpoint file path for a given approach config."""
    cls = ESTIMATOR_REGISTRY[pcfg.registry_key]
    stem = pcfg.name if seed is None else f"{pcfg.name}__seed{int(seed)}"
    return cfg.output.checkpoints_dir / f"{stem}{cls.checkpoint_ext}"


def resolve_paths(raw_lines: list[str]) -> list[str]:
    """Convert relative image paths to absolute using project root."""
    return [str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p
            for p in raw_lines]


def select_input(pcfg: ApproachConfig, data: dict, split: str = "train"):
    """Select the correct input (paths or features) for an approach."""
    suffix = "_test" if split == "test" else "_train"
    if pcfg.feature_type == "image":
        return data[f"paths{suffix}"]
    cls = ESTIMATOR_REGISTRY[pcfg.registry_key]
    input_key = getattr(cls, "input_key", "default")
    if input_key == "proposal" and f"X{suffix}_top25" in data:
        return data[f"X{suffix}_top25"]
    if input_key == "proposal_full":
        key = f"X{suffix}_dcsb"
        if key not in data:
            raise KeyError(
                "Prepared data is missing the full-proposal DCSB tensor. "
                "Re-run the prepare phase after updating the derived CSVs."
            )
        return data[key]
    return data[f"X{suffix}"]


def resolve_target(pcfg: ApproachConfig, data: dict,
                   split: str = "train") -> Any:
    """Resolve scalar or composite targets for an approach."""
    default_key = f"y_{split}_{pcfg.proxy_metric}"
    if not pcfg.target_spec:
        return data.get(default_key, data[f"y_{split}"])

    target_bundle = {}
    for name, suffix in pcfg.target_spec.items():
        # SRRM spatial targets stored under srrm_{split} (not y_{split}_...)
        if suffix == "srrm":
            srrm_key = f"srrm_{split}"
            if srrm_key not in data:
                raise KeyError(f"Missing SRRM data '{srrm_key}' for {pcfg.name}")
            target_bundle[name] = data[srrm_key]
            continue
        key = f"y_{split}_{suffix}"
        if key not in data:
            raise KeyError(f"Missing prepared target '{key}' for {pcfg.name}")
        target_bundle[name] = data[key]

    # Let the estimator class inject any family-specific metadata
    primary_suffix = pcfg.target_spec.get("primary", pcfg.proxy_metric or "")
    cls = ESTIMATOR_REGISTRY[pcfg.registry_key]
    cls.resolve_target_metadata(target_bundle, data, primary_suffix)

    return target_bundle


def resolve_primary_target(pcfg: ApproachConfig, data: dict,
                           split: str = "train"):
    """Resolve the scalar target used for reporting metrics."""
    if not pcfg.target_spec:
        default_key = f"y_{split}_{pcfg.proxy_metric}"
        return data.get(default_key, data[f"y_{split}"])

    primary_suffix = pcfg.target_spec.get("primary")
    if primary_suffix is None:
        raise KeyError(f"Composite target for {pcfg.name} has no 'primary'")
    if primary_suffix == "srrm":
        srrm_key = f"srrm_{split}"
        if srrm_key not in data:
            raise KeyError(f"Missing SRRM data '{srrm_key}' for {pcfg.name}")
        return data[srrm_key]
    key = f"y_{split}_{primary_suffix}"
    if key not in data:
        raise KeyError(f"Missing prepared target '{key}' for {pcfg.name}")
    return data[key]
