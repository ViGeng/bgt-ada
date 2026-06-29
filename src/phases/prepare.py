"""Phase 2 – Prepare: derive features from raw detections, split train/test, save.

Sub-phases:
  2a. Derive — process raw detections + GT → per-video CSVs
  2b. Load   — load CSVs, extract feature matrix, split train/test, save npz
"""

import inspect
import json

from config import PipelineConfig

from .. import log

# ---------------------------------------------------------------------------
# Schema version — bump when prepared-data schema changes
# ---------------------------------------------------------------------------

PREPARED_SCHEMA_VERSION = 10

# ---------------------------------------------------------------------------
# Re-exports: all names that external callers import from this module
# ---------------------------------------------------------------------------

from .prepare_config import (  # noqa: F401
    _prepare_cache_signature,
    _prepared_cache_matches,
    _required_derived_columns,
    _required_prepare_inputs,
    _required_proxy_families,
    _video_cache_complete,
)
from .prepare_derive import (  # noqa: F401
    _derive_video,
    derive_features,
)
from .prepare_split import (  # noqa: F401
    extract_features,
    load_csvs,
    split_and_save,
)
from .prepare_transforms import (  # noqa: F401
    LCER_TAU_GRID,
    _apply_moric_star,
    _apply_phi_moric,
    _apply_sigmoric,
    _compute_conditional_reward_neighborhood,
    _fit_moric_star_reference,
    _fit_phi_moric_reference,
    _fit_sigmoric_reference,
    _resolve_split_indices,
)
from ..proxy_metrics import (  # noqa: F401
    compute_bwd,
    compute_finegrained_proxy_vector,
    compute_srrm,
)
from ..error_decomposition import compute_lcer_vectors  # noqa: F401


# ========================================================================
#  Public API
# ========================================================================

def run(cfg: PipelineConfig) -> None:
    """Execute the prepare phase (derive + load + split)."""
    with log.phase_timer(2):
        log.subsection("Sub-phase 2a: Derive features")
        derive_features(cfg)
        families = _required_proxy_families(cfg)
        required_inputs = _required_prepare_inputs(cfg)
        # Check if data.npz already exists with matching config
        npz_path = cfg.output.prepared_dir / "data.npz"
        meta_path = cfg.output.prepared_dir / "metadata.json"
        if npz_path.exists() and meta_path.exists() and not cfg.force_re_derive:
            try:
                meta = json.loads(meta_path.read_text())
                if _prepared_cache_matches(cfg, meta, families, required_inputs):
                    log.cached("data.npz exists and config matches "
                               "(use --force to re-derive)")
                    return
            except (json.JSONDecodeError, KeyError):
                pass  # stale metadata — recompute

        log.subsection(f"Sub-phase 2b: Load & split (test_ratio={cfg.dataset.test_ratio})")
        from pathlib import Path
        df = load_csvs(Path(cfg.output.data_dir), cfg.dataset.sample_fraction, cfg.seed)

        extract_sig = inspect.signature(extract_features)
        if "cfg" in extract_sig.parameters:
            extracted = extract_features(df, cfg=cfg)
        else:
            extracted = extract_features(df)
        if len(extracted) not in (21, 25, 27, 28, 29):
            raise ValueError(
                f"extract_features returned {len(extracted)} values; expected 21 (legacy), 25 (transitional), 27 (current), 28 (adaptive), or 29 (extended diagnostics)."
            )

        split_and_save(cfg, *extracted)
