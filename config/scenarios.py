"""Scenario component metadata and runtime profile normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict

import numpy as np

SCENARIO_COMPONENT_CATALOG: Dict[str, Dict[str, Any]] = {
    "cls": {
        "index": 0,
        "group": "lcer",
        "label": "Classification",
        "description": "Wrong-class detections corrected by the stronger model.",
    },
    "loc": {
        "index": 1,
        "group": "lcer",
        "label": "Localization",
        "description": "Same-class boxes whose IoU improves with the stronger model.",
    },
    "both": {
        "index": 2,
        "group": "lcer",
        "label": "Class+Loc",
        "description": "Detections requiring both class and localization correction.",
    },
    "dup": {
        "index": 3,
        "group": "lcer",
        "label": "Duplicate",
        "description": "Duplicate detections removed by the stronger model.",
    },
    "bg": {
        "index": 4,
        "group": "lcer",
        "label": "Background",
        "description": "Background false positives corrected by the stronger model.",
    },
    "miss": {
        "index": 5,
        "group": "lcer",
        "label": "Miss",
        "description": "Missed objects recovered by the stronger model.",
    },
    "ap75": {
        "index": 6,
        "group": "direct",
        "label": "AP@0.75",
        "description": "Per-frame AP delta at IoU 0.75.",
    },
    "ap90": {
        "index": 7,
        "group": "direct",
        "label": "AP@0.90",
        "description": "Per-frame AP delta at IoU 0.90.",
    },
    "center_offset": {
        "index": 8,
        "group": "direct",
        "label": "Center Offset",
        "description": "Improvement in normalized box-center alignment.",
    },
    "category_precision": {
        "index": 9,
        "group": "direct",
        "label": "Category Precision",
        "description": "Improvement in category precision at IoU 0.5.",
    },
    # --- LCER3 meta-components: PCA-validated grouped axes ---
    # Each meta-component is a weighted projection of 6 LCER components.
    # PCA analysis confirms these align with independent PCA axes (ρ > 0.92).
    "precision": {
        "index": 10,
        "group": "lcer3",
        "label": "Precision",
        "description": "Trust-critical: bg(0.4)+cls(0.3)+dup(0.3). ORIC efficiency=-15.5%.",
    },
    "localization_quality": {
        "index": 11,
        "group": "lcer3",
        "label": "Localization Quality",
        "description": "Quality-critical: loc(0.6)+both(0.4). ORIC efficiency=31.4%.",
    },
    "recall": {
        "index": 12,
        "group": "lcer3",
        "label": "Recall",
        "description": "Safety-critical: miss(1.0). ORIC efficiency=79.5%.",
    },
}

SCENARIO_COMPONENTS = tuple(
    name for name, meta in SCENARIO_COMPONENT_CATALOG.items()
    if meta["group"] != "lcer3"
)
LCER_SCENARIO_COMPONENTS = tuple(
    name
    for name, meta in SCENARIO_COMPONENT_CATALOG.items()
    if meta["group"] == "lcer"
)
DIRECT_SCENARIO_COMPONENTS = tuple(
    name
    for name, meta in SCENARIO_COMPONENT_CATALOG.items()
    if meta["group"] == "direct"
)
LCER3_SCENARIO_COMPONENTS = tuple(
    name
    for name, meta in SCENARIO_COMPONENT_CATALOG.items()
    if meta["group"] == "lcer3"
)
SCENARIO_COMPONENT_INDEX = {
    name: idx for idx, name in enumerate(SCENARIO_COMPONENTS)
}

# Projection matrix from 6-D LCER space to 3-D LCER3 meta-component space.
# Each column is the normalized weight vector of one PCA-validated scenario axis.
# Shape: (6, 3) — rows = [cls, loc, both, dup, bg, miss],
#                  cols = [precision, localization_quality, recall].
LCER6_TO_LCER3_PROJECTION = np.array([
    # precision  localization_quality  recall
    [0.30,       0.00,                 0.00],   # cls
    [0.00,       0.60,                 0.00],   # loc
    [0.00,       0.40,                 0.00],   # both
    [0.30,       0.00,                 0.00],   # dup
    [0.40,       0.00,                 0.00],   # bg
    [0.00,       0.00,                 1.00],   # miss
], dtype=np.float32)


def _coerce_component_names(component_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in component_names)
    unknown = [name for name in names if name not in SCENARIO_COMPONENT_CATALOG]
    if unknown:
        raise ValueError(
            "Unknown scenario components in target space: "
            + ", ".join(sorted(set(unknown)))
        )
    return names


def normalize_scenario_weights(
    weights: Iterable[float],
    component_names: Iterable[str] = SCENARIO_COMPONENTS,
) -> np.ndarray:
    """Normalize a non-negative weight vector to unit sum."""
    names = _coerce_component_names(component_names)
    arr = np.asarray(list(weights), dtype=np.float32).reshape(-1)
    if arr.shape[0] == len(names):
        projected = arr
    elif arr.shape[0] == len(SCENARIO_COMPONENTS):
        full = arr
        projected = np.asarray(
            [full[SCENARIO_COMPONENT_INDEX[name]] for name in names],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            "Expected "
            f"{len(names)} weights for {names} or {len(SCENARIO_COMPONENTS)} "
            f"weights for the full scenario catalog, got {arr.shape[0]}"
        )
    arr = projected
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("Scenario weights must contain at least one positive value.")
    return arr / total


def resolve_scenario_profile(
    profile: Mapping[str, float] | Iterable[float],
    component_names: Iterable[str] = SCENARIO_COMPONENTS,
) -> np.ndarray:
    """Resolve a readable component map or legacy vector into normalized weights."""
    names = _coerce_component_names(component_names)
    if isinstance(profile, Mapping):
        raw = {str(name): float(value) for name, value in profile.items()}
        unknown = sorted(set(raw) - set(SCENARIO_COMPONENT_CATALOG))
        if unknown:
            raise ValueError(
                "Unknown scenario components: " + ", ".join(unknown)
            )
        ordered = [raw.get(name, 0.0) for name in names]
        return normalize_scenario_weights(ordered, component_names=names)
    return normalize_scenario_weights(profile, component_names=names)


DEFAULT_ADAPTIVE_SCENARIO_WEIGHTS: Dict[str, np.ndarray] = {
    # Data-driven scenarios validated by Phase R3 PCA analysis.
    # PCA-derived scenarios converge to the same 3 axes (ρ > 0.92).
    # Using the semantically cleaner bottom-up forms.
    # All 3 are mutually orthogonal (max pairwise ρ = 0.21).
    #
    # precision_first: trust-critical (ORIC efficiency = -15.5%)
    #   bg (0.40) + cls (0.30) + dup (0.30).  ORIC actively *hurts* this axis.
    "precision_first": resolve_scenario_profile(
        {"bg": 0.40, "cls": 0.30, "dup": 0.30}
    ),
    # localization: quality-critical (ORIC efficiency = 31.4%)
    #   loc (0.60) + both (0.40).  ORIC captures less than a third.
    "localization": resolve_scenario_profile(
        {"loc": 0.60, "both": 0.40}
    ),
    # recall_first: safety-critical (ORIC efficiency = 79.5%)
    #   Pure miss (1.00).  Best captured by ORIC but still 20% headroom.
    "recall_first": resolve_scenario_profile(
        {"miss": 1.00}
    ),
}


# Scenario weights for the LCER3 meta-component space.
# Each scenario selects exactly one axis, since the 3 meta-components
# ARE the 3 orthogonal scenario axes (validated by PCA, ρ > 0.92).
DEFAULT_LCER3_SCENARIO_WEIGHTS: Dict[str, np.ndarray] = {
    "precision_first": resolve_scenario_profile(
        {"precision": 1.00}, component_names=LCER3_SCENARIO_COMPONENTS,
    ),
    "localization": resolve_scenario_profile(
        {"localization_quality": 1.00}, component_names=LCER3_SCENARIO_COMPONENTS,
    ),
    "recall_first": resolve_scenario_profile(
        {"recall": 1.00}, component_names=LCER3_SCENARIO_COMPONENTS,
    ),
}


def default_balanced_scenario_weights() -> np.ndarray:
    """Average the named presets into one neutral default utility vector."""
    stacked = np.stack(list(DEFAULT_ADAPTIVE_SCENARIO_WEIGHTS.values()), axis=0)
    return normalize_scenario_weights(stacked.mean(axis=0))


def normalize_scenario_weight_map(
    weight_map: Mapping[str, Mapping[str, float] | Iterable[float]],
    component_names: Iterable[str] = SCENARIO_COMPONENTS,
) -> Dict[str, np.ndarray]:
    """Normalize a scenario-name -> weights mapping."""
    return {
        str(name): resolve_scenario_profile(weights, component_names=component_names)
        for name, weights in weight_map.items()
    }


def serialize_scenario_weight_map(
    weight_map: Mapping[str, Mapping[str, float] | Iterable[float]],
    component_names: Iterable[str] = SCENARIO_COMPONENTS,
    zero_tol: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """Serialize scenario weights as readable component-name maps."""
    normalized = normalize_scenario_weight_map(weight_map, component_names=component_names)
    names = _coerce_component_names(component_names)
    return {
        scenario_name: {
            component: float(weights[idx])
            for idx, component in enumerate(names)
            if float(weights[idx]) > float(zero_tol)
        }
        for scenario_name, weights in normalized.items()
    }
