"""Tests for MORIC★ and Φ-MORIC proxy-metric transforms."""

import numpy as np
import pytest
from scipy.stats import kstest, shapiro

from src.phases.prepare import (
    _apply_moric_star,
    _apply_phi_moric,
    _fit_moric_star_reference,
    _fit_phi_moric_reference,
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
#  MORIC★ tests
# ===================================================================

class TestMoricStar:

    def test_monotonicity(self, mixed_oric):
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, mixed_oric)
        order = np.argsort(mixed_oric)
        sorted_result = result[order]
        assert np.all(np.diff(sorted_result) >= -1e-12)

    def test_sign_preservation(self, mixed_oric):
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, mixed_oric)
        # Positive ORIC → positive MORIC★
        pos_mask = mixed_oric > 0
        assert np.all(result[pos_mask] > 0)
        # Negative ORIC → negative MORIC★
        neg_mask = mixed_oric < 0
        assert np.all(result[neg_mask] < 0)

    def test_zero_boundary(self, mixed_oric):
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, mixed_oric)
        zero_mask = mixed_oric == 0
        zero_vals = result[zero_mask]
        # ORIC=0 should map to values near 0
        assert np.all(np.abs(zero_vals) < 0.05)

    def test_uniformity(self, mixed_oric):
        """Training-set MORIC★ should be approximately uniform."""
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, mixed_oric)
        # Rescale to [0, 1] for KS test
        lo, hi = result.min(), result.max()
        rescaled = (result - lo) / (hi - lo)
        stat, pval = kstest(rescaled, "uniform")
        assert pval > 0.01, f"KS test failed: stat={stat:.4f}, p={pval:.4f}"

    def test_q0_correctness(self, mixed_oric):
        ref = _fit_moric_star_reference(mixed_oric)
        expected_q0 = np.sum(mixed_oric <= 0) / len(mixed_oric)
        assert abs(ref["q0"] - expected_q0) < 1e-12

    def test_dataset_adaptivity(self, all_positive_oric, all_negative_oric):
        ref_pos = _fit_moric_star_reference(all_positive_oric)
        ref_neg = _fit_moric_star_reference(all_negative_oric)
        assert ref_pos["q0"] < 0.01  # No non-positive samples
        assert ref_neg["q0"] > 0.99  # All non-positive

    def test_train_test_consistency(self, mixed_oric):
        """fit on train, apply to train should match direct computation."""
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, mixed_oric)
        # Verify range: (-q0, 1-q0]
        q0 = ref["q0"]
        assert result.min() >= -q0 - 1e-10
        assert result.max() <= 1 - q0 + 1e-10

    def test_test_split_application(self, mixed_oric):
        """Apply to unseen test data using training reference."""
        rng = np.random.default_rng(99)
        test_oric = rng.uniform(-0.5, 1.5, size=100)
        ref = _fit_moric_star_reference(mixed_oric)
        result = _apply_moric_star(ref, test_oric)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    def test_edge_case_single_element(self):
        ref = _fit_moric_star_reference(np.array([0.5]))
        result = _apply_moric_star(ref, np.array([0.5]))
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_edge_case_all_zeros(self):
        zeros = np.zeros(100)
        ref = _fit_moric_star_reference(zeros)
        assert ref["q0"] == 1.0
        result = _apply_moric_star(ref, zeros)
        assert np.all(np.isfinite(result))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _fit_moric_star_reference(np.array([]))


# ===================================================================
#  Φ-MORIC tests
# ===================================================================

class TestPhiMoric:

    def test_monotonicity(self, mixed_oric):
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, mixed_oric)
        order = np.argsort(mixed_oric)
        sorted_result = result[order]
        assert np.all(np.diff(sorted_result) >= -1e-12)

    def test_sign_preservation(self, mixed_oric):
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, mixed_oric)
        pos_mask = mixed_oric > 0
        assert np.all(result[pos_mask] > 0)
        neg_mask = mixed_oric < 0
        assert np.all(result[neg_mask] < 0)

    def test_boundary_anchoring(self, mixed_oric):
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, mixed_oric)
        zero_mask = mixed_oric == 0
        zero_vals = result[zero_mask]
        assert np.all(np.abs(zero_vals) < 0.3)

    def test_normality(self, mixed_oric):
        """Training-set Φ-MORIC should be approximately normal."""
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, mixed_oric)
        # Standardize for Shapiro-Wilk (limited to 5000 samples)
        n = min(len(result), 5000)
        sample = result[:n]
        sample_std = (sample - sample.mean()) / (sample.std() or 1.0)
        _, pval = shapiro(sample_std)
        assert pval > 0.001, f"Shapiro-Wilk test failed: p={pval:.6f}"

    def test_tail_stretching(self, mixed_oric):
        """Extreme ORIC values get more separation in Φ-MORIC than MORIC★."""
        ref_star = _fit_moric_star_reference(mixed_oric)
        ref_phi = _fit_phi_moric_reference(mixed_oric)
        star = _apply_moric_star(ref_star, mixed_oric)
        phi = _apply_phi_moric(ref_phi, mixed_oric)
        # Compare spread of top 5% vs middle 50%
        star_sorted = np.sort(star)
        phi_sorted = np.sort(phi)
        n = len(star_sorted)
        star_tail_spread = star_sorted[-1] - star_sorted[int(0.95 * n)]
        phi_tail_spread = phi_sorted[-1] - phi_sorted[int(0.95 * n)]
        star_mid_spread = star_sorted[int(0.75 * n)] - star_sorted[int(0.25 * n)]
        phi_mid_spread = phi_sorted[int(0.75 * n)] - phi_sorted[int(0.25 * n)]
        # Phi should have relatively more tail spread vs middle spread
        phi_ratio = phi_tail_spread / (phi_mid_spread + 1e-8)
        star_ratio = star_tail_spread / (star_mid_spread + 1e-8)
        assert phi_ratio > star_ratio

    def test_no_infinities(self, mixed_oric):
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, mixed_oric)
        assert np.all(np.isfinite(result))

    def test_q0_and_probit_q0(self, mixed_oric):
        from scipy.stats import norm
        ref = _fit_phi_moric_reference(mixed_oric)
        expected_q0 = np.sum(mixed_oric <= 0) / len(mixed_oric)
        assert abs(ref["q0"] - expected_q0) < 1e-12
        expected_probit = float(norm.ppf(np.clip(expected_q0, 1e-8, 1 - 1e-8)))
        assert abs(ref["probit_q0"] - expected_probit) < 1e-10

    def test_test_split_application(self, mixed_oric):
        rng = np.random.default_rng(99)
        test_oric = rng.uniform(-0.5, 1.5, size=100)
        ref = _fit_phi_moric_reference(mixed_oric)
        result = _apply_phi_moric(ref, test_oric)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result))

    def test_edge_case_single_element(self):
        ref = _fit_phi_moric_reference(np.array([0.5]))
        result = _apply_phi_moric(ref, np.array([0.5]))
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_edge_case_all_positive(self, all_positive_oric):
        ref = _fit_phi_moric_reference(all_positive_oric)
        result = _apply_phi_moric(ref, all_positive_oric)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)  # All positive ORIC → all positive Φ-MORIC

    def test_edge_case_all_negative(self, all_negative_oric):
        ref = _fit_phi_moric_reference(all_negative_oric)
        result = _apply_phi_moric(ref, all_negative_oric)
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)  # All negative ORIC → all negative Φ-MORIC

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _fit_phi_moric_reference(np.array([]))


# ---------------------------------------------------------------------------
#  AsymmetricUMSELoss boundary-awareness tests
# ---------------------------------------------------------------------------

class TestAsymmetricUMSELossBoundary:
    """Test that AsymmetricUMSELoss correctly uses the boundary parameter."""

    def test_default_boundary_is_zero(self):
        from src.losses import AsymmetricUMSELoss
        loss = AsymmetricUMSELoss()
        assert loss.boundary == 0.0

    def test_boundary_shifts_sign_detection(self):
        """Targets below boundary should get beta upweighting."""
        import torch
        from src.losses import AsymmetricUMSELoss

        pred = torch.zeros(3)
        target = torch.tensor([-0.3, 0.0, 0.3])

        # boundary=0: -0.3 is harmful (gets beta=4 weight)
        loss_b0 = AsymmetricUMSELoss(boundary=0.0)
        l0 = loss_b0(pred, target).item()

        # boundary=-0.5: all targets >= -0.5, no asymmetric penalty
        loss_bm5 = AsymmetricUMSELoss(boundary=-0.5)
        lm5 = loss_bm5(pred, target).item()

        assert l0 > lm5, "Loss with b=0 should be higher (more penalty on -0.3)"

    def test_u_shape_centered_at_boundary(self):
        """U-shape weight should be minimal at the boundary."""
        import torch
        from src.losses import AsymmetricUMSELoss

        boundary = 0.5
        loss = AsymmetricUMSELoss(alpha=1.0, epsilon=0.1, boundary=boundary)
        pred = torch.tensor([0.0])  # some prediction
        target_at_b = torch.tensor([boundary])  # at boundary
        target_far = torch.tensor([boundary + 1.0])  # far from boundary

        l_at = loss(pred, target_at_b).item()
        l_far = loss(pred, target_far).item()
        # Far from boundary should have higher weight (|target-b| is larger)
        assert l_far > l_at

    def test_z_normalized_boundary_preserves_sign(self):
        """Simulates what ImageEstimator.fit() does: z-normalize then set boundary."""
        import torch
        from src.losses import AsymmetricUMSELoss

        # Simulate MORIC★ distribution
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.uniform(-0.3, 0, 250), rng.uniform(0, 0.7, 750)])
        y_mean, y_std = float(np.mean(y)), float(np.std(y))
        z_boundary = (0.0 - y_mean) / y_std

        # A sample at original 0 should map to z_boundary
        z_at_zero = (0.0 - y_mean) / y_std
        assert abs(z_at_zero - z_boundary) < 1e-12

        # A sample originally negative should be below z_boundary
        z_neg = (-0.1 - y_mean) / y_std
        assert z_neg < z_boundary

        # A sample originally positive should be above z_boundary
        z_pos = (0.1 - y_mean) / y_std
        assert z_pos > z_boundary

        # The loss should treat z_neg as harmful (below boundary)
        loss = AsymmetricUMSELoss(alpha=1.0, beta=4.0, epsilon=0.1, boundary=z_boundary)
        pred = torch.zeros(1)
        target_harmful = torch.tensor([z_neg])
        target_beneficial = torch.tensor([z_pos])

        l_harmful = loss(pred, target_harmful).item()
        l_beneficial = loss(pred, target_beneficial).item()
        # Harmful should get higher loss due to beta upweighting
        assert l_harmful > l_beneficial
