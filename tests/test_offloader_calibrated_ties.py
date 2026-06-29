import numpy as np
import pytest

from src.offloader import ConfiguredOffloader, OffloadContext, Offloader


def test_decide_calibrated_preserves_budget_under_ties():
    offloader = Offloader("moric_allpoint")
    decision = offloader.decide_calibrated(
        np.ones(10, dtype=float),
        target_ratio=0.5,
        calibration_predictions=np.ones(10, dtype=float),
    )

    assert decision.n_offload == 5
    assert decision.actual_ratio == 0.5


def test_decide_calibrated_requires_train_calibration_predictions():
    offloader = Offloader("moric_allpoint")

    with pytest.raises(ValueError, match="calibration_predictions"):
        offloader.decide_calibrated(np.ones(8, dtype=float), target_ratio=0.25)


def test_online_ecdf_calibrated_policy_uses_context_train_predictions():
    offloader = ConfiguredOffloader(
        name="online_ecdf_calibrated",
        policy_id="online_ecdf_calibrated",
    )
    context = OffloadContext(
        predictions=np.array([0.1, 0.3, 0.5, 0.7], dtype=float),
        proxy_metric="gain_11pt",
        train_predictions=np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float),
    )

    decision = offloader.decide(context, target_ratio=0.5)
    assert decision.n_total == 4
    assert 0.0 <= decision.actual_ratio <= 1.0


def test_online_ecdf_calibrated_policy_errors_without_train_predictions():
    offloader = ConfiguredOffloader(
        name="online_ecdf_calibrated",
        policy_id="online_ecdf_calibrated",
    )
    context = OffloadContext(
        predictions=np.array([0.1, 0.3, 0.5, 0.7], dtype=float),
        proxy_metric="gain_11pt",
        train_predictions=None,
    )

    with pytest.raises(ValueError, match="calibration_predictions"):
        offloader.decide(context, target_ratio=0.5)
