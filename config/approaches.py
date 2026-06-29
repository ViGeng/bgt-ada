"""Default approach catalog (paper subset).

This is the single source of truth for the repo's default experiment
inventory. Each entry defines the estimator brick inline together with the
offloader policy that consumes it.

This catalog is the curated subset behind the paper *"Budget-Adaptive Routing:
Skipping the Weak When the Strong Answers Anyway"*. It holds exactly the
approaches evaluated in the paper:

* the five headline methods (Tab. 2):
  - skipping (pre-stage) MobileNetV2-Lite on OffloadBin and on MORIC+,
  - our conditioned (post-stage) XGBoost on MORIC,
  - the EdgeML and DCSB conditioned baselines;
* the eleven MobileNetV2-Lite learning-target ablations (Tab. 5);
* the EfficientNet-B0-Lite backbone ablation (Tab. 4).

The broader research repo additionally registered scenario-adaptive / LCER /
hybrid routes and an architecture-diversity sweep; those are outside the paper
and are not part of this release.
"""

from __future__ import annotations

from .schema import ApproachConfig, EstimatorConfig, OffloaderConfig


_OFFLOADER_PRESETS: dict[str, tuple[str, dict]] = {
    "native_threshold": ("native_threshold", {}),
    "online_ecdf_calibrated": ("online_ecdf_calibrated", {}),
    "fixed_classifier": ("fixed_classifier", {}),
}


def _offloader(name: str, **params) -> OffloaderConfig:
    policy_id, base_params = _OFFLOADER_PRESETS[name]
    return OffloaderConfig(
        name=name,
        policy_id=policy_id,
        params={**base_params, **params},
    )


def _approach(
    estimator_name: str,
    *,
    base_model: str,
    feature_type: str,
    stage: str,
    proxy_metric: str,
    offloader: str,
    target_spec: dict[str, str] | None = None,
    loss: str | None = None,
    params: dict | None = None,
    offloader_params: dict | None = None,
) -> ApproachConfig:
    estimator = EstimatorConfig(
        name=estimator_name,
        base_model=base_model,
        feature_type=feature_type,
        stage=stage,
        proxy_metric=proxy_metric,
        target_spec=target_spec,
        loss=loss,
        params=dict(params or {}),
    )
    offloader_cfg = _offloader(offloader, **dict(offloader_params or {}))
    return ApproachConfig(
        name=f"{estimator_name}|{offloader_cfg.name}",
        estimator=estimator,
        offloader=offloader_cfg,
    )


def default_approaches() -> list[ApproachConfig]:
    """Return the default approach inventory (paper subset)."""
    return [
        # ------------------------------------------------------------------
        # Headline methods (Tab. 2)
        # ------------------------------------------------------------------
        # Skipping (pre-stage): MobileNetV2-Lite, the two reported targets.
        _approach(
            "pre|mobilenet_v2|OffloadBin|focal",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="offload_binary",
            loss="focal",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|MORIC+-AP",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="moric_plus_allpoint",
            params={"epochs": 50, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        # Conditioned (post-stage), ours: XGBoost on MORIC.
        _approach(
            "post|xgboost|MORIC-AP",
            base_model="xgboost",
            feature_type="tabular",
            stage="post",
            proxy_metric="moric_allpoint",
            params={"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
            offloader="online_ecdf_calibrated",
        ),
        # Conditioned baselines from the literature.
        _approach(
            "post|edgeml|MORIC-AP|wmse",
            base_model="edgeml",
            feature_type="tabular",
            stage="post",
            proxy_metric="moric_allpoint",
            loss="weighted_mse",
            params={
                "epochs": 100,
                "lr": 5e-3,
                "batch_size": 64,
                "patience": 15,
                "cv_folds": 5,
                "grid_search": True,
                "target_mode": "moric",
            },
            offloader="native_threshold",
        ),
        _approach(
            "post|dcsb|CountGain-05",
            base_model="dcsb",
            feature_type="tabular",
            stage="post",
            proxy_metric="count_gain_05",
            target_spec={"primary": "count_gain_05", "gt_count": "gt_count"},
            params={"base_confidence_threshold": 0.5},
            offloader="fixed_classifier",
            offloader_params={"threshold": 0.5},
        ),
        # ------------------------------------------------------------------
        # Learning-target ablation on the fixed MobileNetV2-Lite trunk (Tab. 5)
        # ------------------------------------------------------------------
        _approach(
            "pre|mobilenet_v2|TopQuartile|focal",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="gain_top_quartile",
            loss="focal",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|HighIoUGain|huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="high_iou_gain_75",
            loss="huber",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|MORIC+-AP|quantile_75",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="moric_plus_allpoint",
            loss="quantile_75",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|F1Gain|huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="f1_gain_50",
            loss="huber",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|MORIC+-AP|wing",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="moric_plus_allpoint",
            loss="wing",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|RescueRatio|huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="rescue_ratio_50",
            loss="huber",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|RescueRatio|wing",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="rescue_ratio_50",
            loss="wing",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|WorstCaseGain|huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="worst_case_gain",
            loss="huber",
            params={"epochs": 100, "lr": 5e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|SigMORIC-AP|sign_rank_huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="sigmoric_allpoint",
            loss="sign_rank_huber",
            params={
                "epochs": 200,
                "lr": 1e-4,
                "batch_size": 1024,
                "loss_lam": 0.5,
                "loss_tau_cross": 0.2,
                "loss_tau_same": 0.05,
                "loss_delta": 1.0,
            },
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|MORICSTAR-AP|sign_rank_huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="moric_star_allpoint",
            loss="sign_rank_huber",
            params={
                "epochs": 200,
                "lr": 1e-4,
                "batch_size": 1024,
                "loss_lam": 0.5,
                "loss_tau_cross": 0.2,
                "loss_tau_same": 0.05,
                "loss_delta": 1.0,
            },
            offloader="online_ecdf_calibrated",
        ),
        _approach(
            "pre|mobilenet_v2|PhiMORIC-AP|sign_rank_huber",
            base_model="mobilenet_v2",
            feature_type="image",
            stage="pre",
            proxy_metric="phi_moric_allpoint",
            loss="sign_rank_huber",
            params={
                "epochs": 200,
                "lr": 1e-4,
                "batch_size": 1024,
                "loss_lam": 0.5,
                "loss_tau_cross": 0.2,
                "loss_tau_same": 0.05,
                "loss_delta": 1.0,
            },
            offloader="online_ecdf_calibrated",
        ),
        # ------------------------------------------------------------------
        # Backbone ablation: EfficientNet-B0-Lite vs MobileNetV2-Lite (Tab. 4),
        # both on the identical OffloadBin / focal target.
        # ------------------------------------------------------------------
        _approach(
            "pre|efficientnet_b0_lite|OffloadBin|focal",
            base_model="efficientnet_b0_lite",
            feature_type="image",
            stage="pre",
            proxy_metric="offload_binary",
            loss="focal",
            params={"epochs": 100, "lr": 1e-4, "batch_size": 512},
            offloader="online_ecdf_calibrated",
        ),
    ]
