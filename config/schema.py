"""Config dataclass definitions (building blocks — no default values)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .estimators import EstimatorConfig
from .scenarios import (DEFAULT_ADAPTIVE_SCENARIO_WEIGHTS,
                        normalize_scenario_weight_map,
                        serialize_scenario_weight_map)


@dataclass
class OffloaderConfig:
    """Configuration for the second-stage offloader bound to an approach."""

    name: str = "default"
    policy_id: str = "native_threshold"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApproachConfig:
    """An approach = named estimator + offloader.

    The estimator brick (base model, proxy-metric, loss, hyperparameters)
    lives in ``estimator``.  The offloader is configured separately at
    evaluation time.

    ``name``      — unique display / checkpoint identifier.
    ``estimator`` — the estimator brick configuration.
    """

    name: str
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    offloader: OffloaderConfig = field(default_factory=OffloaderConfig)
    enabled: bool = True  # approach-level switch (estimator.enabled is independent)

    # -- Delegate properties ------------------------------------------------
    # Permanent API surface: pipeline code reads pcfg.registry_key,
    # pcfg.feature_type, etc. without reaching into pcfg.estimator.

    @property
    def base_model(self) -> Optional[str]:
        return self.estimator.base_model

    @property
    def registry_key(self) -> str:
        """Key for ESTIMATOR_REGISTRY lookup."""
        return self.estimator.registry_key

    @property
    def is_active(self) -> bool:
        """True when both the approach and its estimator are enabled."""
        return self.enabled and self.estimator.enabled

    @property
    def feature_type(self) -> str:
        return self.estimator.feature_type

    @property
    def stage(self) -> str:
        return self.estimator.stage

    @property
    def proxy_metric(self) -> Optional[str]:
        return self.estimator.proxy_metric

    @property
    def target_spec(self) -> Optional[Dict[str, str]]:
        return self.estimator.target_spec

    @property
    def loss(self) -> Optional[str]:
        return self.estimator.loss

    @property
    def params(self) -> Dict[str, Any]:
        return self.estimator.params

    @property
    def offloader_name(self) -> str:
        return self.offloader.name

    @property
    def policy_id(self) -> str:
        return self.offloader.policy_id

    @property
    def offloader_params(self) -> Dict[str, Any]:
        return self.offloader.params

    @property
    def display_name(self) -> str:
        """Compact display label derived from the structured name.

        Extracts ``model (proxy_metric)`` from the pipe-delimited name.
        Falls back to the full name if the format is unexpected.
        """
        parts = self.name.split("|")
        if len(parts) >= 3:
            return f"{parts[1]} ({parts[2]})"
        return self.name


@dataclass
class DatasetConfig:
    """Dataset identity, models, thresholds, and data-handling settings."""

    name: str = ""
    root: str = ""
    edge_model: str = ""
    cloud_model: str = ""
    conf_threshold: Optional[float] = None

    # Data-handling (dataset-specific)
    sample_fraction: float = 0.3
    test_ratio: float = 0.2

    # MORIC+ zero-anchor: fraction of the CDF allocated to the negative
    # (keep-local) region.  Set empirically per dataset to match the
    # observed proportion of negative ORIC values.
    #   VOC      : 0.27
    #   UA-DETRAC: (TBD, measure from data)
    #   COCO     : (TBD, measure from data)
    moric_plus_neg_frac: float = 0.27

    # Detection phase (dataset-specific)
    detection_batch_size: int = 32
    detection_conf: Optional[float] = 0.05   # low cache threshold for EdgeML/DCSB proposal baselines
    # Which dataset split to run detection on.  Available options depend on
    # the dataset:
    #   UA-DETRAC : "train", "test"         (separate video sets)
    #   COCO      : "train" (train2017), "test" (val2017)
    #   VOC       : "train" (trainval),  "test" (val)
    # Use "all" to detect on both train + test splits.
    detection_split: str = "test"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 512
    num_workers: int = 4
    epochs: int = 10
    lr: float = 0.001
    patience: int = 10
    compile_models: bool = False


@dataclass
class OutputConfig:
    """Output directory layout. Convenience dirs derive from *base_dir*."""

    base_dir: str = ""
    data_dir: str = ""  # auto-set to derived_dir in PipelineConfig.__post_init__

    @property
    def prepared_dir(self) -> Path:
        return Path(self.base_dir) / "prepared"

    @property
    def checkpoints_dir(self) -> Path:
        return Path(self.base_dir) / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return Path(self.base_dir) / "metrics"

    @property
    def charts_dir(self) -> Path:
        return Path(self.base_dir) / "charts"

    @property
    def scenario_eval_dir(self) -> Path:
        return Path(self.base_dir) / "scenario_eval"


# ---- Composite config -------------------------------------------------

@dataclass
class PipelineConfig:
    """Full pipeline configuration composing all sub-configs."""

    # Sub-configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    approaches: List[ApproachConfig] = field(default_factory=list)

    # General pipeline settings (defaults from config/pipeline.py)
    device: str = ""
    seed: Optional[int] = None
    force_retrain: Optional[bool] = None
    force_re_derive: Optional[bool] = None
    offload_ratios: List[float] = field(default_factory=list)
    derive_num_workers: Optional[int] = None   # 0 = auto, 1 = sequential, N = N workers
    evaluation_num_workers: Optional[int] = None  # 0 = auto (CPU-only), 1 = sequential, N = N workers
    oric_context_size: Optional[int] = None    # 0 = use all frames, N = sample N context frames
    oric_context_draws: Optional[int] = None   # 0 = use default K context draws
    latency_warmup: Optional[int] = None       # 0 = use default
    latency_samples: Optional[int] = None      # 0 = use default
    evaluation_seeds: List[int] = field(default_factory=list)
    bootstrap_samples: Optional[int] = None
    calibration_bins: Optional[int] = None
    fixed_ratio_points: List[float] = field(default_factory=list)
    adaptive_scenario_weights: Dict[str, Any] = field(default_factory=dict)
    adaptive_scenario_mixes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from . import pipeline as _defaults
        from .datasets import DATASETS
        from .approaches import default_approaches

        # Apply pipeline-level defaults
        if not self.device:
            self.device = _defaults.DEVICE
        if self.seed is None:
            self.seed = _defaults.SEED
        if self.force_retrain is None:
            self.force_retrain = _defaults.FORCE_RETRAIN
        if self.force_re_derive is None:
            self.force_re_derive = _defaults.FORCE_RE_DERIVE
        if not self.offload_ratios:
            self.offload_ratios = list(_defaults.OFFLOAD_RATIOS)
        if self.derive_num_workers is None:
            self.derive_num_workers = _defaults.DERIVE_NUM_WORKERS
        if self.evaluation_num_workers is None:
            self.evaluation_num_workers = _defaults.EVALUATION_NUM_WORKERS
        if self.oric_context_size is None:
            self.oric_context_size = _defaults.ORIC_CONTEXT_SIZE
        if self.oric_context_draws is None:
            self.oric_context_draws = _defaults.ORIC_CONTEXT_DRAWS
        if self.latency_warmup is None:
            self.latency_warmup = _defaults.LATENCY_WARMUP
        if self.latency_samples is None:
            self.latency_samples = _defaults.LATENCY_SAMPLES
        if not self.evaluation_seeds:
            self.evaluation_seeds = [int(self.seed)]
        else:
            self.evaluation_seeds = [int(seed) for seed in self.evaluation_seeds]
        if self.bootstrap_samples is None:
            self.bootstrap_samples = _defaults.BOOTSTRAP_SAMPLES
        if self.calibration_bins is None:
            self.calibration_bins = _defaults.CALIBRATION_BINS
        if not self.fixed_ratio_points:
            self.fixed_ratio_points = list(_defaults.FIXED_RATIO_POINTS)
        else:
            self.fixed_ratio_points = [float(x) for x in self.fixed_ratio_points]
        if not self.adaptive_scenario_weights:
            self.adaptive_scenario_weights = {
                name: weights.tolist()
                for name, weights in DEFAULT_ADAPTIVE_SCENARIO_WEIGHTS.items()
            }
        else:
            self.adaptive_scenario_weights = {
                name: weights.tolist()
                for name, weights in normalize_scenario_weight_map(
                    self.adaptive_scenario_weights
                ).items()
            }
        if self.adaptive_scenario_mixes:
            self.adaptive_scenario_mixes = {
                name: weights.tolist()
                for name, weights in normalize_scenario_weight_map(
                    self.adaptive_scenario_mixes
                ).items()
            }
        else:
            self.adaptive_scenario_mixes = {}

        project_root = Path(__file__).resolve().parent.parent
        ds = self.dataset

        # Apply dataset-specific defaults from registry
        base_name = ""
        if ds.name in DATASETS:
            base_name = ds.name
        else:
            # Longest-prefix match to resolve variants (e.g. "voc-v2" -> "voc")
            # to their base configuration defaults.
            for k in DATASETS:
                if ds.name.startswith(k) and len(k) > len(base_name):
                    base_name = k

        if base_name in DATASETS:
            defaults = DATASETS[base_name]
            # Use the canonical base name from the registry entry (e.g. "voc")
            # for path derivation.
            resolved_ds_name = defaults.name if defaults.name else base_name
            if not ds.root:
                ds.root = defaults.root
            if not ds.edge_model:
                ds.edge_model = defaults.edge_model
            if not ds.cloud_model:
                ds.cloud_model = defaults.cloud_model
            if ds.conf_threshold is None:
                ds.conf_threshold = defaults.conf_threshold
            # Copy dataset-specific data-handling defaults
            _ds_defaults = DatasetConfig()
            if ds.sample_fraction == _ds_defaults.sample_fraction:
                ds.sample_fraction = defaults.sample_fraction
            if ds.test_ratio == _ds_defaults.test_ratio:
                ds.test_ratio = defaults.test_ratio
            if ds.detection_batch_size == _ds_defaults.detection_batch_size:
                ds.detection_batch_size = defaults.detection_batch_size
            if ds.detection_conf is None:
                ds.detection_conf = defaults.detection_conf
            if ds.detection_split == _ds_defaults.detection_split:
                ds.detection_split = defaults.detection_split
        else:
            resolved_ds_name = ds.name
            # Only apply generic fallbacks for explicitly named but
            # unrecognised datasets — skip when name is empty (first
            # __post_init__ call before the dataset name is assigned).
            if ds.name:
                if not ds.edge_model:
                    ds.edge_model = "fasterrcnn_mobilenet_v3_large_fpn"
                if not ds.cloud_model:
                    ds.cloud_model = "fasterrcnn_resnet50_fpn_v2"
                if ds.conf_threshold is None:
                    ds.conf_threshold = 0.3

        # Resolve relative dataset root
        if ds.root:
            dr = Path(ds.root)
            if not dr.is_absolute():
                ds.root = str((project_root / dr).resolve())

        # Auto-set output dirs
        if not self.output.base_dir and ds.name:
            self.output.base_dir = str(
                project_root / "results" / "full_eval" / ds.name
            )
        if not self.output.data_dir:
            self.output.data_dir = str(self.derived_dir)
        if not self.approaches:
            self.approaches = default_approaches()

    # ----- convenience properties -----

    @property
    def derived_dir(self) -> Path:
        """Global cache directory for derived features.

        This directory is shared across all variants that use the same
        dataset, model pair, confidence threshold, and ORIC context size.
        """
        from .datasets import DATASETS
        ds_name = self.dataset.name
        
        # Consistent base-name resolution with __post_init__
        base_name = ""
        if ds_name in DATASETS:
            base_name = ds_name
        else:
            for k in DATASETS:
                if ds_name.startswith(k) and len(k) > len(base_name):
                    base_name = k
        
        resolved = DATASETS[base_name].name if base_name in DATASETS else ds_name
        resolved = resolved if resolved else "unknown"

        conf_label = f"conf_{str(self.dataset.conf_threshold).replace('.', '_')}"
        ctx_label = f"ctx{self.oric_context_size}_k{self.oric_context_draws}"

        project_root = Path(__file__).resolve().parent.parent
        return (
            project_root / "results" / "derived"
            / resolved
            / f"{self.dataset.edge_model}_vs_{self.dataset.cloud_model}"
            / conf_label
            / ctx_label
        )

    # ----- helpers -----

    def enabled_approaches(self) -> List[ApproachConfig]:
        return [p for p in self.approaches if p.is_active]

    # ----- persistence -----

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            raw = json.load(f)
        approaches = []
        for p in raw.pop("approaches", []):
            est_raw = p.pop("estimator", {})
            est = EstimatorConfig(**est_raw)
            offloader_raw = p.pop("offloader", {})
            offloader = OffloaderConfig(**offloader_raw)
            approaches.append(ApproachConfig(estimator=est, offloader=offloader, **p))
        dataset = DatasetConfig(**raw.pop("dataset", {}))
        training = TrainingConfig(**raw.pop("training", {}))
        output = OutputConfig(**raw.pop("output", {}))
        return cls(
            dataset=dataset, training=training,
            output=output, approaches=approaches, **raw,
        )

    def _to_dict(self) -> dict:
        return {
            "dataset": {
                "name": self.dataset.name,
                "root": self.dataset.root,
                "edge_model": self.dataset.edge_model,
                "cloud_model": self.dataset.cloud_model,
                "conf_threshold": self.dataset.conf_threshold,
                "sample_fraction": self.dataset.sample_fraction,
                "test_ratio": self.dataset.test_ratio,
                "moric_plus_neg_frac": self.dataset.moric_plus_neg_frac,
                "detection_batch_size": self.dataset.detection_batch_size,
                "detection_conf": self.dataset.detection_conf,
                "detection_split": self.dataset.detection_split,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "num_workers": self.training.num_workers,
                "epochs": self.training.epochs,
                "lr": self.training.lr,
                "patience": self.training.patience,
                "compile_models": self.training.compile_models,
            },
            "output": {
                "base_dir": self.output.base_dir,
                "data_dir": self.output.data_dir,
            },
            "device": self.device,
            "seed": self.seed,
            "force_retrain": self.force_retrain,
            "force_re_derive": self.force_re_derive,
            "offload_ratios": self.offload_ratios,
            "derive_num_workers": self.derive_num_workers,
            "evaluation_num_workers": self.evaluation_num_workers,
            "oric_context_size": self.oric_context_size,
            "oric_context_draws": self.oric_context_draws,
            "latency_warmup": self.latency_warmup,
            "latency_samples": self.latency_samples,
            "evaluation_seeds": self.evaluation_seeds,
            "bootstrap_samples": self.bootstrap_samples,
            "calibration_bins": self.calibration_bins,
            "fixed_ratio_points": self.fixed_ratio_points,
            "adaptive_scenario_weights": serialize_scenario_weight_map(
                self.adaptive_scenario_weights
            ),
            "adaptive_scenario_mixes": serialize_scenario_weight_map(
                self.adaptive_scenario_mixes
            ),
            "approaches": [
                {
                    "name": p.name,
                    "enabled": p.enabled,
                    "offloader": {
                        "name": p.offloader.name,
                        "policy_id": p.offloader.policy_id,
                        "params": p.offloader.params,
                    },
                    "estimator": {
                        "name": p.estimator.name,
                        "base_model": p.estimator.base_model,
                        "enabled": p.estimator.enabled,
                        "feature_type": p.estimator.feature_type,
                        "stage": p.estimator.stage,
                        "proxy_metric": p.estimator.proxy_metric,
                        "target_spec": p.estimator.target_spec,
                        "loss": p.estimator.loss,
                        "params": p.estimator.params,
                    },
                }
                for p in self.approaches
            ],
        }
