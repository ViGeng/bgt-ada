import numpy as np

from src.offloader import ConfiguredOffloader, OffloadContext


def test_forced_exact_ratio_selects_exact_top_k_count():
    offloader = ConfiguredOffloader(
        name="forced_exact_ratio",
        policy_id="forced_exact_ratio",
    )
    predictions = np.array([0.1, 0.6, 0.2, 0.9, 0.8], dtype=float)
    context = OffloadContext(predictions=predictions, proxy_metric="gain_11pt")

    decision = offloader.decide(context, target_ratio=0.4)

    assert decision.n_offload == 2
    assert decision.actual_ratio == 0.4
    assert decision.mask.tolist() == [False, False, False, True, True]


def test_forced_exact_ratio_handles_ratio_edges():
    offloader = ConfiguredOffloader(
        name="forced_exact_ratio",
        policy_id="forced_exact_ratio",
    )
    predictions = np.array([0.2, 0.4, 0.6], dtype=float)
    context = OffloadContext(predictions=predictions, proxy_metric="gain_11pt")

    decision_zero = offloader.decide(context, target_ratio=0.0)
    decision_all = offloader.decide(context, target_ratio=1.0)

    assert decision_zero.n_offload == 0
    assert decision_zero.actual_ratio == 0.0
    assert decision_all.n_offload == 3
    assert decision_all.actual_ratio == 1.0
