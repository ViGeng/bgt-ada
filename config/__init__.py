"""Pipeline configuration package.

Sub-modules:
  schema      – building-block dataclasses (DatasetConfig, TrainingConfig, etc.)
  estimators  – EstimatorConfig dataclass + compatibility estimator view
  datasets    – known dataset registry (DATASETS)
  approaches  – single-source default approach catalog
  pipeline    – PipelineConfig compositor
"""

from .datasets import DATASETS
from .approaches import default_approaches
from .estimators import EstimatorConfig, default_estimators
from .schema import (ApproachConfig, DatasetConfig, OffloaderConfig,
                     OutputConfig,
                     PipelineConfig, TrainingConfig)
from .offloaders import default_offloaders

__all__ = [
    "ApproachConfig",
    "DatasetConfig",
    "DATASETS",
    "EstimatorConfig",
    "OffloaderConfig",
    "OutputConfig",
    "PipelineConfig",
    "TrainingConfig",
    "default_approaches",
    "default_estimators",
    "default_offloaders",
]
