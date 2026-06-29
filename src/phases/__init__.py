"""Pipeline phases for estimator evaluation.

Five phases:
    1. detect   – Run detection models on images, persist results.
    2. prepare  – Derive features from detections, split train/test, save to disk.
    3. train    – Train specified estimators, persist checkpoints.
    4. evaluate – Load checkpoints, run offloading evaluation, save metrics.
    5. analyse  – Read metrics, generate charts and reports.

Each phase can run independently via the root ``run_pipeline.py``.
"""

from config import (ApproachConfig, DatasetConfig, OutputConfig,
                    PipelineConfig, TrainingConfig)

__all__ = ['ApproachConfig', 'DatasetConfig', 'OutputConfig',
           'PipelineConfig', 'TrainingConfig']
