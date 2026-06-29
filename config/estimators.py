"""Estimator config building block plus compatibility inventory view."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional


@dataclass
class EstimatorConfig:
    """Configuration for one estimator brick."""

    name: str
    base_model: Optional[str] = None
    enabled: bool = True
    feature_type: str = "tabular"
    stage: str = "post"
    proxy_metric: Optional[str] = None
    target_spec: Optional[Dict[str, str]] = None
    loss: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def registry_key(self) -> str:
        return self.base_model if self.base_model else self.name


def default_estimators() -> list[EstimatorConfig]:
    """Compatibility view derived from the default approach catalog."""
    from .approaches import default_approaches

    estimators: list[EstimatorConfig] = []
    seen: set[str] = set()
    for approach in default_approaches():
        estimator = approach.estimator
        if estimator.name in seen:
            continue
        seen.add(estimator.name)
        estimators.append(
            replace(
                estimator,
                target_spec=(
                    dict(estimator.target_spec)
                    if estimator.target_spec is not None
                    else None
                ),
                params=dict(estimator.params),
            )
        )
    return estimators

