"""Configurable loss functions for estimator training.

Provides a registry of loss functions that can be selected per-estimator
via the ``loss`` field in ApproachConfig.  Each loss is a callable
``(pred, target) -> scalar`` compatible with PyTorch autograd.

In this repo, the loss is the objective that pushes the estimator to
approximate a chosen proxy-metric.

Available losses
----------------
Regression:
  mse              -- Standard Mean Squared Error (default for most estimators)
  weighted_mse     -- Target-weighted MSE from EdgeML paper (eq. 7):
                      L = mean(target * (pred - target)^2).
                      Designed for MORIC proxy-metrics in [0,1];
                      emphasises high-gain samples.
  asymmetric_u_mse -- Asymmetric U-shaped weighted MSE for MORIC+ proxy-metrics
                      in [-1, 1].  Penalises extremes and overweights sparse
                      negative gains.  Configurable alpha/beta/epsilon via
                      loss_alpha / loss_beta / loss_epsilon in estimator params.
                      Epsilon (default 0.1) prevents zero-gradient for neutral
                      proxy-metric values.
  uniform_regularized -- AsymmetricUMSE + soft-rank CDF uniformity penalty.
                      Pushes prediction distribution toward uniform via
                      differentiable soft-rank ECDF matching.  Configurable
                      loss_lam / loss_tau plus base alpha/beta/epsilon.
  huber            -- Huber loss (delta=1.0), robust to outliers
  mae / l1         -- Mean Absolute Error
  smooth_l1        -- Smooth L1 loss (beta=1.0)
  log_cosh         -- log(cosh(pred - target)), smooth L1 approximation

Spatial:
  spatial_weighted_mse -- Spatially weighted MSE for SRRM proxy-metrics (S×S grids).
                      Upweights cells where the strong model dominates.
                      Configurable loss_alpha for positive-cell amplification.

Ranking:
  sign_rank_huber    -- Huber regression + sign-aware pairwise ranking for
                      SigMORIC.  Combines robust regression with direct
                      ranking optimisation, using larger margins for
                      cross-boundary pairs.  Configurable delta / lam /
                      tau_cross / tau_same via estimator params.
  contrastive_ranking -- Regression + pairwise margin ranking loss for BWD.
                      Preserves correct image ordering for threshold offloading.
                      Configurable loss_beta / loss_lam / loss_tau.

Classification:
  bce              -- Binary Cross-Entropy (expects sigmoid outputs)
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Type alias for loss callables: (pred, target) -> scalar tensor
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
#  Individual loss implementations
# ---------------------------------------------------------------------------

def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard Mean Squared Error."""
    return torch.mean((pred - target) ** 2)


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Target-weighted MSE (EdgeML paper eq. 7).

    L = mean(target_i * (pred_i - target_i)^2)

    Weights errors by the target value so that high-gain samples
    (closer to 1 in MORIC space) contribute more to the loss.
    Best paired with MORIC proxy metrics (values in [0, 1]).
    """
    return torch.mean(target * (pred - target) ** 2)


class AsymmetricUMSELoss(nn.Module):
    """Asymmetric U-Shaped Weighted MSE for sign-aware proxy-metrics.

    L = mean(lambda_i * (|target_i - boundary|^alpha + epsilon) * (pred_i - target_i)^2)

    where:
      - |target - boundary|^alpha + epsilon  creates a U-shaped weight profile
        centered at *boundary*: errors far from the decision boundary are
        penalised more than near-boundary errors.  The epsilon baseline ensures
        that boundary-adjacent (neutral) samples still receive a gradient
        signal, preventing the "zero-gradient trap".
      - lambda_i = 1.0 if target >= boundary, else beta.  The asymmetry
        multiplier forces the network to pay attention to the sparse
        negative (keep-local / hallucination-avoidance) samples.

    The *boundary* parameter allows the loss to remain correct even when
    targets are z-normalised.  ``ImageEstimator.fit()`` automatically
    computes ``boundary = (0 - y_mean) / y_std`` so that the original
    sign boundary at 0 is preserved in normalised space.

    Args:
        alpha:   Exponent for the U-shape magnitude weight (default 1.0).
        beta:    Multiplier for negative proxy-metric samples (default 4.0).
                 Higher values force more attention on negative gains.
        epsilon: Minimum weight floor (default 0.1).  Ensures boundary-adjacent
                 proxy-metric values still contribute to the loss.  Without this,
                 ~30-50% of the dataset (neutral frames) receives zero gradient.
        boundary: Decision boundary in target space (default 0.0).  Targets
                 below *boundary* are treated as harmful offloads.  When using
                 z-normalised targets, set to ``(0 - y_mean) / y_std``.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 4.0,
                 epsilon: float = 0.1, boundary: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.boundary = boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = self.boundary
        # lambda_i: asymmetry weight (beta for targets below the boundary)
        lam = torch.where(target >= b,
                          torch.ones_like(target),
                          torch.full_like(target, self.beta))
        # U-shape weight centered at boundary
        u_weight = torch.abs(target - b).pow(self.alpha) + self.epsilon
        # Weighted MSE
        return torch.mean(lam * u_weight * (pred - target) ** 2)

    def __repr__(self) -> str:
        return (f"AsymmetricUMSELoss(alpha={self.alpha}, beta={self.beta}, "
                f"epsilon={self.epsilon}, boundary={self.boundary})")


def huber_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss (delta=1.0), robust to outliers."""
    return nn.functional.huber_loss(pred, target, delta=1.0)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error (L1)."""
    return torch.mean(torch.abs(pred - target))


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 loss (beta=1.0)."""
    return nn.functional.smooth_l1_loss(pred, target, beta=1.0)


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-cosh loss: smooth approximation of L1."""
    diff = pred - target
    return torch.mean(diff + nn.functional.softplus(-2.0 * diff) - 0.6931471805599453)


class UniformRegularizedLoss(nn.Module):
    """L = L_regression(AsymmetricUMSE) + lam * L_uniform(soft-rank ECDF vs diagonal).

    Combines the base regression loss with a penalty that pushes the batch
    prediction distribution toward uniform.  Uses differentiable soft-rank
    ECDF compared to the ideal diagonal.

    The soft-rank is affine-invariant, so it works correctly regardless of
    proxy-metric normalization (ImageEstimator z-normalizes proxy-metric values).

    Args:
        lam: Regularization strength (default 0.1).
        tau: Soft-rank temperature (default 0.1). Lower = sharper ranks.
        alpha, beta, epsilon: Forwarded to AsymmetricUMSELoss base.
    """

    def __init__(self, lam: float = 0.1, tau: float = 0.1,
                 alpha: float = 1.0, beta: float = 4.0,
                 epsilon: float = 0.1):
        super().__init__()
        self.lam = lam
        self.tau = tau
        self.base_loss = AsymmetricUMSELoss(alpha=alpha, beta=beta,
                                            epsilon=epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L_reg = self.base_loss(pred, target)

        N = pred.shape[0]
        if N < 8:
            return L_reg

        # Subsample for memory: O(N^2) pairwise diffs
        if N > 2048:
            idx = torch.randperm(N, device=pred.device)[:2048]
            p = pred[idx]
            N = 2048
        else:
            p = pred

        # Soft-rank: soft_rank_i = sum_j sigmoid((p_i - p_j) / tau)
        diff = p.unsqueeze(1) - p.unsqueeze(0)  # [N, N]
        soft_ranks = torch.sigmoid(diff / self.tau).sum(dim=1)
        soft_ecdf_sorted, _ = torch.sort(soft_ranks / N)
        ideal_ecdf = torch.linspace(0.5 / N, 1 - 0.5 / N, N,
                                    device=pred.device)

        L_uniform = torch.mean((soft_ecdf_sorted - ideal_ecdf) ** 2)
        return L_reg + self.lam * L_uniform

    def __repr__(self) -> str:
        return (f"UniformRegularizedLoss(lam={self.lam}, tau={self.tau}, "
                f"base={self.base_loss})")


class SpatiallyWeightedMSELoss(nn.Module):
    """Spatially weighted MSE for SRRM proxy-metrics on (B, S, S) grids.

    L = mean((1 + alpha * max(0, target)) * (pred - target)^2)

    Cells where the strong model dominates (positive target) receive
    amplified penalty so the estimator learns to identify spatially
    localised detection failures of the weak model.

    Args:
        alpha: Amplification for positive-target cells (default 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.alpha * torch.clamp(target, min=0.0)
        return torch.mean(weight * (pred - target) ** 2)

    def __repr__(self) -> str:
        return f"SpatiallyWeightedMSELoss(alpha={self.alpha})"


class ContrastiveRankingLoss(nn.Module):
    """Regression + pairwise margin ranking loss for BWD proxy-metrics.

    L = beta * MSE(pred, target) + lam * RankLoss(pred, target)

    The ranking term samples random pairs and penalises order inversions
    via a hinge margin, explicitly optimising the estimator to preserve
    relative image complexity ordering for threshold-based offloading.

    Args:
        beta: Regression weight (default 1.0).
        lam:  Ranking weight (default 0.5).
        tau:  Ranking margin (default 0.1).
    """

    def __init__(self, beta: float = 1.0, lam: float = 0.5,
                 tau: float = 0.1):
        super().__init__()
        self.beta = beta
        self.lam = lam
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_term = self.beta * torch.mean((pred - target) ** 2)

        N = pred.shape[0]
        if N < 2:
            return mse_term

        # Subsample pairs for O(N) instead of O(N^2)
        n_pairs = min(N, 256)
        idx_i = torch.randint(0, N, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, N, (n_pairs,), device=pred.device)
        sign = torch.sign(target[idx_i] - target[idx_j])
        diff = pred[idx_i] - pred[idx_j]
        rank_term = torch.mean(torch.clamp(-sign * diff + self.tau, min=0.0))

        return mse_term + self.lam * rank_term

    def __repr__(self) -> str:
        return (f"ContrastiveRankingLoss(beta={self.beta}, lam={self.lam}, "
                f"tau={self.tau})")


class SignRankHuberLoss(nn.Module):
    """Huber regression + sign-aware pairwise ranking for SigMORIC targets.

    L = L_huber(pred, target) + lam * L_signrank(pred, target)

    The Huber base provides robust regression (no gradient explosion from
    outliers), while the ranking term directly optimises image ordering —
    the quantity that Spearman rho and NDCG actually evaluate.

    Cross-sign pairs (one target above the boundary, one below) receive a
    larger margin than same-sign pairs, prioritising the critical
    offload/keep boundary decision.

    Args:
        delta:     Huber threshold (default 1.0).
        lam:       Ranking loss weight (default 0.5).
        tau_cross: Margin for cross-boundary pairs (default 0.2).
        tau_same:  Margin for same-sign pairs (default 0.05).
        boundary:  Decision boundary in target space (default 0.0).
                   Auto-set to z-normalised boundary by ImageEstimator.
    """

    def __init__(self, delta: float = 1.0, lam: float = 0.5,
                 tau_cross: float = 0.2, tau_same: float = 0.05,
                 boundary: float = 0.0):
        super().__init__()
        self.delta = delta
        self.lam = lam
        self.tau_cross = tau_cross
        self.tau_same = tau_same
        self.boundary = boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = nn.functional.huber_loss(pred, target, delta=self.delta)

        N = pred.shape[0]
        if N < 2 or self.lam == 0:
            return huber

        # Sample random pairs (O(N) not O(N^2))
        n_pairs = min(N, 256)
        idx_i = torch.randint(0, N, (n_pairs,), device=pred.device)
        idx_j = torch.randint(0, N, (n_pairs,), device=pred.device)

        # Target ordering direction
        sign = torch.sign(target[idx_i] - target[idx_j])

        # Sign-aware margin: larger for cross-boundary pairs
        b = self.boundary
        cross_mask = (target[idx_i] - b) * (target[idx_j] - b) < 0
        margin = torch.where(cross_mask,
                             torch.full_like(sign, self.tau_cross),
                             torch.full_like(sign, self.tau_same))

        # Hinge ranking loss: penalise order inversions
        diff = pred[idx_i] - pred[idx_j]
        rank_loss = torch.mean(torch.clamp(-sign * diff + margin, min=0.0))

        return huber + self.lam * rank_loss

    def __repr__(self) -> str:
        return (f"SignRankHuberLoss(delta={self.delta}, lam={self.lam}, "
                f"tau_cross={self.tau_cross}, tau_same={self.tau_same}, "
                f"boundary={self.boundary})")


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary Cross-Entropy with logits (autocast-safe, expects raw logits)."""
    return nn.functional.binary_cross_entropy_with_logits(pred.float(), target.float())


class FocalLoss(nn.Module):
    """Focal Loss for binary classification targets.

    L = mean(alpha * (1-p_t)^gamma * BCE)

    Down-weights easy examples and focuses on hard ones.  Handles class
    imbalance via the alpha parameter.  Expects raw logits (no sigmoid).

    Args:
        alpha: Balancing factor for positive class (default 0.25).
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

    def __repr__(self) -> str:
        return f"FocalLoss(alpha={self.alpha}, gamma={self.gamma})"


class QuantileLoss(nn.Module):
    """Quantile (pinball) loss for conservative regression.

    L = mean(max(tau*(t-p), (tau-1)*(t-p)))

    With tau > 0.5, the loss penalises under-prediction more than
    over-prediction, producing conservative estimates that tend to
    over-predict offloading gain.

    Args:
        tau: Quantile level (default 0.75).
    """

    def __init__(self, tau: float = 0.75):
        super().__init__()
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = target - pred
        return torch.mean(torch.max(self.tau * diff, (self.tau - 1) * diff))

    def __repr__(self) -> str:
        return f"QuantileLoss(tau={self.tau})"


class WingLoss(nn.Module):
    """Wing loss for boundary-sensitive regression.

    L = w * ln(1 + |d|/eps)  if |d| < w
        |d| - C              otherwise

    where C = w - w*ln(1 + w/eps) ensures continuity.

    Amplifies small errors near the decision boundary (|d| < w) while
    behaving like L1 for large deviations.  Designed for face alignment,
    effective for any boundary-focused regression.

    Args:
        w:       Wing width (default 10.0). Errors below w get log amplification.
        epsilon: Curvature parameter (default 2.0).
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self._C = w - w * np.log(1 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        import math
        d = torch.abs(pred - target)
        small = self.w * torch.log(1 + d / self.epsilon)
        large = d - self._C
        return torch.mean(torch.where(d < self.w, small, large))

    def __repr__(self) -> str:
        return f"WingLoss(w={self.w}, epsilon={self.epsilon})"


class OrdinalCrossEntropyLoss(nn.Module):
    """Ordinal cross-entropy via K-1 cumulative binary subtasks.

    Discretizes regression targets into K ordinal bins and trains K-1
    binary classifiers P(y > k) with BCE, preserving ordinal structure
    without imposing magnitude assumptions.

    The bin edges are stored as a buffer at construction (from training
    data) to ensure consistency between train and eval.

    Args:
        n_bins: Number of ordinal bins (default 5).
        bin_edges: Pre-computed bin edges (K-1 thresholds). If None,
                   must be set before first forward via set_bin_edges().
    """

    def __init__(self, n_bins: int = 5, bin_edges: torch.Tensor = None):
        super().__init__()
        self.n_bins = n_bins
        if bin_edges is not None:
            self.register_buffer("bin_edges", bin_edges)
        else:
            self.register_buffer("bin_edges", torch.zeros(n_bins - 1))
        self._edges_fitted = bin_edges is not None

    def set_bin_edges(self, train_targets: torch.Tensor) -> None:
        """Fit bin edges from training targets (K-1 quantile thresholds)."""
        quantiles = torch.linspace(0, 1, self.n_bins + 1)[1:-1].to(train_targets.device)
        edges = torch.quantile(train_targets.float(), quantiles)
        self.bin_edges = edges.to(train_targets.device)
        self._edges_fitted = True

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self._edges_fitted:
            self.set_bin_edges(target)

        # pred shape: (B,) or (B, K-1). If scalar, expand to K-1 outputs.
        if pred.dim() == 1:
            pred = pred.unsqueeze(1).expand(-1, self.n_bins - 1)

        # Build cumulative binary labels: y_k = 1 if target > edge_k
        edges = self.bin_edges.to(target.device)
        binary_targets = (target.unsqueeze(1) > edges.unsqueeze(0)).float()

        return nn.functional.binary_cross_entropy_with_logits(
            pred, binary_targets)

    def __repr__(self) -> str:
        return f"OrdinalCrossEntropyLoss(n_bins={self.n_bins})"


# ---------------------------------------------------------------------------
#  Registry
#
#  Entries are either:
#    - A plain callable (pred, target) -> scalar  (for parameter-free losses)
#    - A *class* (nn.Module subclass) that is instantiated with optional
#      kwargs by get_loss().  This allows parametric losses like
#      AsymmetricUMSELoss(alpha=..., beta=...).
# ---------------------------------------------------------------------------

LOSS_REGISTRY: Dict[str, Union[LossFn, type]] = {
    "mse": mse_loss,
    "weighted_mse": weighted_mse_loss,
    "asymmetric_u_mse": AsymmetricUMSELoss,
    "uniform_regularized": UniformRegularizedLoss,
    "spatial_weighted_mse": SpatiallyWeightedMSELoss,
    "contrastive_ranking": ContrastiveRankingLoss,
    "sign_rank_huber": SignRankHuberLoss,
    "huber": huber_loss,
    "mae": mae_loss,
    "l1": mae_loss,
    "smooth_l1": smooth_l1_loss,
    "log_cosh": log_cosh_loss,
    "bce": bce_loss,
    "focal": FocalLoss,
    "quantile_75": QuantileLoss,
    "wing": WingLoss,
    "ordinal_ce": OrdinalCrossEntropyLoss,
}


def get_loss(name: Optional[str], **loss_params) -> Optional[LossFn]:
    """Look up (and optionally instantiate) a loss function by name.

    For parametric losses (registered as classes), ``loss_params`` are
    forwarded to the constructor.  Estimator configs can supply these
    via ``params`` keys prefixed with ``loss_`` (e.g. ``loss_alpha=2.0``).

    Returns ``None`` when *name* is ``None`` (estimator uses its own default).
    Raises ``ValueError`` for unknown names.
    """
    if name is None:
        return None
    key = name.lower().strip()
    if key not in LOSS_REGISTRY:
        available = sorted(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss: {name!r}. Available: {available}")

    entry = LOSS_REGISTRY[key]

    # If the entry is a class, instantiate it (possibly with parameters)
    if isinstance(entry, type):
        return entry(**loss_params)

    return entry


def extract_loss_params(kwargs: dict) -> dict:
    """Extract loss-specific params from a kwargs dict.

    Looks for keys prefixed with ``loss_`` (excluding ``loss`` itself)
    and strips the prefix.  E.g. ``{"loss_alpha": 2.0, "loss_beta": 6.0}``
    becomes ``{"alpha": 2.0, "beta": 6.0}``.
    """
    return {
        k[len("loss_"):]: v
        for k, v in kwargs.items()
        if k.startswith("loss_") and k != "loss"
    }


def list_losses() -> Dict[str, str]:
    """Return {name: one-line description} for every registered loss."""
    descriptions = {
        "mse": "Standard Mean Squared Error",
        "weighted_mse": "Target-weighted MSE (EdgeML paper eq. 7, best with MORIC proxy-metrics)",
        "asymmetric_u_mse": "Asymmetric U-shaped weighted MSE (for MORIC+ proxy-metrics in [-1,1])",
        "uniform_regularized": "AsymmetricUMSE + soft-rank CDF uniformity penalty",
        "spatial_weighted_mse": "Spatially weighted MSE for SRRM proxy-metrics (S×S grids)",
        "contrastive_ranking": "Regression + pairwise margin ranking loss for BWD proxy-metrics",
        "sign_rank_huber": "Huber regression + sign-aware pairwise ranking (for SigMORIC proxy-metrics)",
        "huber": "Huber loss (delta=1.0, outlier-robust)",
        "mae": "Mean Absolute Error (L1)",
        "l1": "Mean Absolute Error (alias for mae)",
        "smooth_l1": "Smooth L1 loss (beta=1.0)",
        "log_cosh": "Log-cosh loss (smooth L1 approximation)",
        "bce": "Binary Cross-Entropy (for classification targets)",
        "focal": "Focal loss (for binary targets, handles class imbalance)",
        "quantile_75": "Quantile loss (tau=0.75, conservative regression)",
        "wing": "Wing loss (boundary-sensitive, amplifies small errors)",
        "ordinal_ce": "Ordinal cross-entropy (K=5 bins, preserves order)",
    }
    return {k: descriptions.get(k, "") for k in LOSS_REGISTRY}
