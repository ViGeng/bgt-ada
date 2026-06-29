"""Tests for SigMORIC proxy-metric transform and SignRankHuberLoss."""

import numpy as np
import pytest

from src.phases.prepare import (
    _apply_moric_star,
    _apply_sigmoric,
    _fit_moric_star_reference,
    _fit_sigmoric_reference,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_oric():
    """Synthetic ORIC with ~25% negative, ~5% zero, ~70% positive."""
    rng = np.random.default_rng(42)
    neg = rng.uniform(-1.0, -0.01, size=250)
    zero = np.zeros(50)
    pos = rng.uniform(0.01, 2.0, size=700)
    return np.concatenate([neg, zero, pos])


@pytest.fixture
def all_positive_oric():
    rng = np.random.default_rng(7)
    return rng.uniform(0.01, 3.0, size=500)


@pytest.fixture
def all_negative_oric():
    rng = np.random.default_rng(7)
    return rng.uniform(-3.0, -0.01, size=500)


# ===================================================================
#  SigMORIC proxy-metric tests
# ===================================================================

class TestSigMORIC:

    def test_monotonicity(self, mixed_oric):
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, mixed_oric)
        order = np.argsort(mixed_oric)
        sorted_result = result[order]
        assert np.all(np.diff(sorted_result) >= -1e-12)

    def test_sign_preservation(self, mixed_oric):
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, mixed_oric)
        pos_mask = mixed_oric > 0
        assert np.all(result[pos_mask] > 0)
        neg_mask = mixed_oric < 0
        assert np.all(result[neg_mask] < 0)

    def test_bounded_range(self, mixed_oric):
        """SigMORIC must stay within (-1, 1)."""
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, mixed_oric)
        assert np.all(result > -1.0)
        assert np.all(result < 1.0)

    def test_zero_boundary(self, mixed_oric):
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, mixed_oric)
        zero_mask = mixed_oric == 0
        zero_vals = result[zero_mask]
        # ORIC=0 should map to values near 0
        assert np.all(np.abs(zero_vals) < 0.1)

    def test_boundary_concentration(self, mixed_oric):
        """More values should cluster near zero than at extremes."""
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, mixed_oric)
        near_boundary = np.sum(np.abs(result) < 0.3)
        near_extremes = np.sum(np.abs(result) > 0.7)
        assert near_boundary > near_extremes, (
            f"Expected more values near boundary ({near_boundary}) "
            f"than at extremes ({near_extremes})"
        )

    def test_q0_correctness(self, mixed_oric):
        ref = _fit_sigmoric_reference(mixed_oric)
        expected_q0 = np.sum(mixed_oric <= 0) / len(mixed_oric)
        assert abs(ref["q0"] - expected_q0) < 1e-12

    def test_steepness_parameter(self, mixed_oric):
        """Higher k → more concentration at boundary."""
        ref_low = _fit_sigmoric_reference(mixed_oric, k=2.0)
        ref_high = _fit_sigmoric_reference(mixed_oric, k=8.0)
        result_low = _apply_sigmoric(ref_low, mixed_oric)
        result_high = _apply_sigmoric(ref_high, mixed_oric)
        # Higher k should produce more values near ±1 (less near 0)
        high_extreme = np.sum(np.abs(result_high) > 0.8)
        low_extreme = np.sum(np.abs(result_low) > 0.8)
        assert high_extreme > low_extreme

    def test_dataset_adaptivity(self, all_positive_oric, all_negative_oric):
        ref_pos = _fit_sigmoric_reference(all_positive_oric)
        ref_neg = _fit_sigmoric_reference(all_negative_oric)
        assert ref_pos["q0"] < 0.01
        assert ref_neg["q0"] > 0.99

    def test_test_split_application(self, mixed_oric):
        rng = np.random.default_rng(99)
        test_oric = rng.uniform(-0.5, 1.5, size=100)
        ref = _fit_sigmoric_reference(mixed_oric)
        result = _apply_sigmoric(ref, test_oric)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))
        assert np.all(result > -1.0)
        assert np.all(result < 1.0)

    def test_preserves_ranking_of_moric_star(self, mixed_oric):
        """SigMORIC and MORIC★ should produce the same ranking (both are
        monotone transforms of MORIC)."""
        ref_star = _fit_moric_star_reference(mixed_oric)
        ref_sig = _fit_sigmoric_reference(mixed_oric)
        star = _apply_moric_star(ref_star, mixed_oric)
        sig = _apply_sigmoric(ref_sig, mixed_oric)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(star, sig)
        assert rho > 0.999, f"SigMORIC and MORIC★ rankings should match, got rho={rho}"

    def test_edge_case_single_element(self):
        ref = _fit_sigmoric_reference(np.array([0.5]))
        result = _apply_sigmoric(ref, np.array([0.5]))
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert -1.0 < result[0] < 1.0

    def test_edge_case_all_zeros(self):
        zeros = np.zeros(100)
        ref = _fit_sigmoric_reference(zeros)
        assert ref["q0"] == 1.0
        result = _apply_sigmoric(ref, zeros)
        assert np.all(np.isfinite(result))
        assert np.all(result > -1.0)
        assert np.all(result < 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _fit_sigmoric_reference(np.array([]))


# ===================================================================
#  SignRankHuberLoss tests
# ===================================================================

class TestSignRankHuberLoss:

    def test_default_construction(self):
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss()
        assert loss.delta == 1.0
        assert loss.lam == 0.5
        assert loss.tau_cross == 0.2
        assert loss.tau_same == 0.05
        assert loss.boundary == 0.0

    def test_forward_shape(self):
        import torch
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss()
        pred = torch.randn(32)
        target = torch.randn(32)
        result = loss(pred, target)
        assert result.shape == ()
        assert result.item() >= 0

    def test_perfect_prediction_low_loss(self):
        import torch
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss()
        target = torch.randn(64)
        # Perfect prediction: both Huber and ranking loss should be minimal
        l_perfect = loss(target.clone(), target).item()
        # Random prediction: should have higher loss
        l_random = loss(torch.randn(64), target).item()
        assert l_perfect < l_random

    def test_ranking_component_matters(self):
        """With lam > 0, swapping order should increase loss."""
        import torch
        from src.losses import SignRankHuberLoss
        torch.manual_seed(42)
        target = torch.linspace(-1, 1, 100)
        # Correct order
        pred_correct = target + 0.01 * torch.randn(100)
        # Reversed order
        pred_reversed = -target + 0.01 * torch.randn(100)
        loss = SignRankHuberLoss(lam=1.0)
        l_correct = loss(pred_correct, target).item()
        l_reversed = loss(pred_reversed, target).item()
        assert l_reversed > l_correct

    def test_cross_boundary_larger_margin(self):
        """Cross-boundary pairs should get penalised more for inversions."""
        import torch
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss(lam=1.0, tau_cross=0.5, tau_same=0.01,
                                boundary=0.0)
        # Two targets: one positive, one negative (cross-boundary)
        target_cross = torch.tensor([-0.5, 0.5])
        # Inverted prediction (wrong order)
        pred_cross = torch.tensor([0.4, -0.4])

        # Two targets: both positive (same-sign)
        target_same = torch.tensor([0.1, 0.5])
        pred_same = torch.tensor([0.4, 0.0])

        l_cross = loss(pred_cross, target_cross).item()
        l_same = loss(pred_same, target_same).item()
        # Cross-boundary inversion should be penalised more
        assert l_cross > l_same

    def test_boundary_parameter_shifts_detection(self):
        import torch
        from src.losses import SignRankHuberLoss
        target = torch.tensor([-0.3, -0.1, 0.1, 0.3])
        pred = torch.zeros(4)
        # boundary=0: two targets each side
        loss_b0 = SignRankHuberLoss(boundary=0.0, lam=1.0)
        # boundary=-0.5: all targets above boundary (same-sign)
        loss_bm5 = SignRankHuberLoss(boundary=-0.5, lam=1.0)
        l0 = loss_b0(pred, target).item()
        lm5 = loss_bm5(pred, target).item()
        # With boundary at 0, there are cross-boundary pairs (larger margins)
        assert l0 > lm5

    def test_lam_zero_reduces_to_huber(self):
        import torch
        from src.losses import SignRankHuberLoss
        torch.manual_seed(42)
        pred = torch.randn(64)
        target = torch.randn(64)
        loss_with_rank = SignRankHuberLoss(lam=0.0)
        loss_huber = torch.nn.functional.huber_loss(pred, target, delta=1.0)
        assert abs(loss_with_rank(pred, target).item() - loss_huber.item()) < 1e-6

    def test_small_batch(self):
        """Should work with batch size 1 (no ranking possible)."""
        import torch
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss()
        pred = torch.tensor([0.5])
        target = torch.tensor([0.3])
        result = loss(pred, target)
        assert result.item() >= 0

    def test_gradient_flows(self):
        """Verify gradients flow through all components."""
        import torch
        from src.losses import SignRankHuberLoss
        loss = SignRankHuberLoss(lam=0.5)
        pred = torch.randn(32, requires_grad=True)
        target = torch.randn(32)
        result = loss(pred, target)
        result.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_registry_lookup(self):
        from src.losses import get_loss
        loss = get_loss("sign_rank_huber", lam=0.3, tau_cross=0.1)
        assert loss is not None
        assert loss.lam == 0.3
        assert loss.tau_cross == 0.1


# ===================================================================
#  Integration: SigMORIC + SignRankHuber interaction
# ===================================================================

class TestSigMORICSignRankIntegration:

    def test_sigmoric_values_work_with_loss(self, mixed_oric):
        """SigMORIC targets should be usable with SignRankHuberLoss."""
        import torch
        from src.losses import SignRankHuberLoss
        ref = _fit_sigmoric_reference(mixed_oric)
        targets = _apply_sigmoric(ref, mixed_oric)
        # Simulate z-normalisation (as ImageEstimator does)
        y_mean = targets.mean()
        y_std = targets.std()
        z_targets = (targets - y_mean) / y_std
        z_boundary = (0.0 - y_mean) / y_std
        loss = SignRankHuberLoss(boundary=z_boundary)
        t = torch.tensor(z_targets, dtype=torch.float32)
        p = torch.randn_like(t)
        result = loss(p, t)
        assert result.item() >= 0
        assert np.isfinite(result.item())

    def test_config_roundtrip(self):
        """Estimator config should correctly wire proxy_metric and loss."""
        from config.estimators import default_estimators
        ests = {e.name: e for e in default_estimators()}
        est = ests.get("pre|mobilenet_v2|SigMORIC-AP|sign_rank_huber")
        assert est is not None
        assert est.proxy_metric == "sigmoric_allpoint"
        assert est.loss == "sign_rank_huber"
        assert est.base_model == "mobilenet_v2"
        assert est.stage == "pre"
        assert est.feature_type == "image"
        assert est.params["loss_lam"] == 0.5

    def test_approach_exists(self):
        """Approach should be wired in default_approaches."""
        from config.approaches import default_approaches
        approaches = {a.name: a for a in default_approaches()}
        name = "pre|mobilenet_v2|SigMORIC-AP|sign_rank_huber|online_ecdf_calibrated"
        assert name in approaches
