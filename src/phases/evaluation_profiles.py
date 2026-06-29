"""Evaluation profiles for approach-specific evaluation behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from config import ApproachConfig, PipelineConfig
from config.scenarios import (SCENARIO_COMPONENTS, resolve_scenario_profile,
                              serialize_scenario_weight_map)

from ._shared import resolve_primary_target, resolve_target


@dataclass(frozen=True)
class ScenarioProfile:
    """Single runtime scenario profile used for evaluation expansion."""

    name: str
    weights: tuple[float, ...]
    scenario_type: str = "preset"
    source: str = "config"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self, pcfg: ApproachConfig) -> dict[str, Any]:
        weights = [float(x) for x in self.weights]
        weight_map = {
            str(name): round(float(value), 8)
            for name, value in dict(self.metadata.get("weight_map", {})).items()
        }
        extra_metadata = {
            key: value
            for key, value in dict(self.metadata).items()
            if key not in {"component_names", "weight_map"}
        }
        return {
            "approach": pcfg.name,
            "base_model": pcfg.registry_key,
            "stage": pcfg.stage,
            "scenario": self.name,
            "scenario_type": self.scenario_type,
            "profile_source": self.source,
            "weight_count": len(weights),
            "weight_sum": float(sum(weights)),
            "weights": weights,
            "component_names": list(self.metadata.get("component_names", SCENARIO_COMPONENTS)),
            "weight_map": weight_map,
            **extra_metadata,
        }

    def as_named_weights(self) -> dict[str, float]:
        weight_map = self.metadata.get("weight_map")
        if isinstance(weight_map, dict) and weight_map:
            return {
                str(name): round(float(value), 8)
                for name, value in weight_map.items()
            }
        component_names = self.metadata.get("component_names", SCENARIO_COMPONENTS)
        return {
            str(component_name): round(float(weight), 8)
            for component_name, weight in zip(component_names, self.weights)
            if float(weight) > 0.0
        }


@dataclass(frozen=True)
class EvaluationRun:
    """Single logical evaluation run for an approach."""

    name_suffix: Optional[str] = None
    predict_kwargs: Dict[str, Any] = field(default_factory=dict)
    row_metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_profile: Optional[ScenarioProfile] = None


def _as_scenario_profile(raw: Any) -> ScenarioProfile:
    if isinstance(raw, ScenarioProfile):
        return raw
    if not isinstance(raw, dict):
        raise TypeError("Scenario profiles must be ScenarioProfile instances or dicts.")

    name = str(raw.get("name", raw.get("scenario", "")) or "")
    if not name:
        raise ValueError("Scenario profile entries require a non-empty 'name'.")
    if "weights" not in raw:
        raise ValueError(f"Scenario profile '{name}' is missing 'weights'.")
    raw_weights = raw["weights"]
    if isinstance(raw_weights, dict):
        weight_map = serialize_scenario_weight_map({name: raw_weights})[name]
        weights = tuple(
            float(value)
            for value in np.asarray(
                [weight_map.get(component, 0.0) for component in SCENARIO_COMPONENTS],
                dtype=float,
            ).reshape(-1).tolist()
        )
        metadata = dict(raw.get("metadata", {}))
        metadata.setdefault("component_names", list(SCENARIO_COMPONENTS))
        metadata.setdefault("weight_map", dict(weight_map))
    else:
        weights = tuple(
            float(x) for x in np.asarray(raw_weights, dtype=float).reshape(-1).tolist()
        )
        metadata = dict(raw.get("metadata", {}))
    return ScenarioProfile(
        name=name,
        weights=weights,
        scenario_type=str(raw.get("scenario_type", "preset") or "preset"),
        source=str(raw.get("source", "config") or "config"),
        metadata=metadata,
    )


def _default_scenario_profiles(cfg: PipelineConfig) -> list[ScenarioProfile]:
    profiles: list[ScenarioProfile] = []
    preset_maps = serialize_scenario_weight_map(cfg.adaptive_scenario_weights)
    mix_maps = serialize_scenario_weight_map(getattr(cfg, "adaptive_scenario_mixes", {}))
    for scenario_name, scenario_weights in cfg.adaptive_scenario_weights.items():
        profiles.append(
            ScenarioProfile(
                name=str(scenario_name),
                weights=tuple(
                    float(x)
                    for x in np.asarray(scenario_weights, dtype=float).reshape(-1).tolist()
                ),
                scenario_type="preset",
                source="adaptive_scenario_weights",
                metadata={
                    "component_names": list(SCENARIO_COMPONENTS),
                    "weight_map": dict(preset_maps.get(str(scenario_name), {})),
                },
            )
        )
    for scenario_name, scenario_weights in getattr(cfg, "adaptive_scenario_mixes", {}).items():
        profiles.append(
            ScenarioProfile(
                name=str(scenario_name),
                weights=tuple(
                    float(x)
                    for x in np.asarray(scenario_weights, dtype=float).reshape(-1).tolist()
                ),
                scenario_type="mix",
                source="adaptive_scenario_mixes",
                metadata={
                    "component_names": list(SCENARIO_COMPONENTS),
                    "weight_map": dict(mix_maps.get(str(scenario_name), {})),
                },
            )
        )
    return profiles

def _estimator_component_names(estimator_cls) -> tuple[str, ...]:
    names = getattr(estimator_cls, "default_component_names", SCENARIO_COMPONENTS)
    return tuple(str(name) for name in np.asarray(names, dtype=object).reshape(-1))


def _project_profile_for_estimator(profile: ScenarioProfile,
                                   estimator_cls) -> dict[str, float]:
    component_names = _estimator_component_names(estimator_cls)
    named_weights = profile.as_named_weights()
    projected = resolve_scenario_profile(named_weights, component_names=component_names)
    return {
        str(component_name): round(float(projected[idx]), 8)
        for idx, component_name in enumerate(component_names)
        if float(projected[idx]) > 0.0
    }





class EvaluationProfile:
    """Shared interface for approach-specific evaluation behavior."""

    def iter_runs(self, cfg: PipelineConfig,
                  pcfg: ApproachConfig, estimator_cls) -> list[EvaluationRun]:
        return [EvaluationRun()]

    def resolve_reporting_target(self, pcfg: ApproachConfig, data: dict,
                                 split: str, estimator_cls,
                                 run: EvaluationRun):
        return resolve_primary_target(pcfg, data, split)

    def format_estimator_name(self, pcfg: ApproachConfig,
                              run: EvaluationRun) -> str:
        if run.name_suffix:
            return f"{pcfg.name}|{run.name_suffix}"
        return pcfg.name

    def apply_result_metadata(self, result: dict, run: EvaluationRun) -> None:
        result.update(run.row_metadata)

    def iter_scenario_profiles(self, cfg: PipelineConfig,
                               pcfg: ApproachConfig, estimator_cls) -> list[ScenarioProfile]:
        return []

    def describe_scenario_profiles(self, cfg: PipelineConfig,
                                   pcfg: ApproachConfig, estimator_cls) -> list[dict[str, Any]]:
        return [
            profile.to_record(pcfg)
            for profile in self.iter_scenario_profiles(cfg, pcfg, estimator_cls)
        ]


class DefaultEvaluationProfile(EvaluationProfile):
    """Default single-run profile used by most estimators."""


class AdaptiveScenarioEvaluationProfile(EvaluationProfile):
    """Runtime scenario expansion for adaptive scenario estimators."""

    def iter_scenario_profiles(self, cfg: PipelineConfig,
                               pcfg: ApproachConfig, estimator_cls) -> list[ScenarioProfile]:
        custom_profiles = getattr(estimator_cls, "iter_scenario_profiles", None)
        if callable(custom_profiles):
            return [_as_scenario_profile(profile) for profile in custom_profiles(cfg, pcfg)]
        return _default_scenario_profiles(cfg)

    def iter_runs(self, cfg: PipelineConfig,
                  pcfg: ApproachConfig, estimator_cls) -> list[EvaluationRun]:
        return [
            EvaluationRun(
                name_suffix=profile.name,
                predict_kwargs={
                    "scenario_weights": _project_profile_for_estimator(
                        profile, estimator_cls
                    )
                },
                row_metadata={
                    "scenario": profile.name,
                    "scenario_type": profile.scenario_type,
                },
                scenario_profile=profile,
            )
            for profile in self.iter_scenario_profiles(cfg, pcfg, estimator_cls)
        ]

    def resolve_reporting_target(self, pcfg: ApproachConfig, data: dict,
                                 split: str, estimator_cls,
                                 run: EvaluationRun):
        if run.scenario_profile is None:
            return resolve_primary_target(pcfg, data, split)

        scenario_projector = getattr(estimator_cls, "project_primary_for_scenario", None)
        projected_weights = _project_profile_for_estimator(run.scenario_profile, estimator_cls)
        if callable(scenario_projector):
            try:
                target_bundle = resolve_target(pcfg, data, split)
            except KeyError:
                vector_suffix = (pcfg.target_spec or {}).get("vector")
                if vector_suffix is None:
                    raise
                key = f"y_{split}_{vector_suffix}"
                if key not in data:
                    raise
                target_bundle = {"vector": data[key]}
            return scenario_projector(target_bundle, projected_weights)

        vector_suffix = (pcfg.target_spec or {}).get("vector")
        vector_projector = getattr(estimator_cls, "project_primary_from_vector", None)
        if vector_suffix is not None and callable(vector_projector):
            key = f"y_{split}_{vector_suffix}"
            if key not in data:
                raise KeyError(f"Missing prepared target '{key}' for {pcfg.name}")
            return vector_projector(
                data[key],
                projected_weights,
            )
        return resolve_primary_target(pcfg, data, split)


def get_evaluation_profile(estimator_cls) -> EvaluationProfile:
    """Return the evaluation profile for the estimator class."""
    if getattr(estimator_cls, "scenario_adaptive", False):
        return AdaptiveScenarioEvaluationProfile()
    return DefaultEvaluationProfile()
