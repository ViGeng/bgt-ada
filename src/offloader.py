"""Offloading decision-maker.

Converts continuous estimator predictions into binary offload/keep-local
decisions given a target offloading ratio constraint.

All strategies are sequential (per-image decisions):
  - **threshold**: Uses the metric's native threshold formula to convert a
    target ratio into a decision boundary.  For CDF-based metrics (MORIC,
    MORIC+) the threshold is analytic; for absolute metrics (gain, ORIC) it
    derives the threshold from *training-set* predictions (empirical
    percentile) and applies it to test predictions.  The resulting actual
    ratio reveals how well the estimator's prediction distribution
    generalises from train to test.
    - **calibrated**: Train-ECDF-calibrated threshold.  Fits an empirical CDF
        on calibration predictions (typically train split), maps each incoming
        frame score to that CDF space, then applies threshold (1 - target_ratio).
    - **forced_exact_ratio**: Offline evaluation-only top-k sweep used only
        when an exact-ratio oracle-style curve is explicitly requested.

Metric types:
  - MORIC  [0,1]:   CDF-based.  Threshold = 1 - ratio.
  - MORIC+ [-1,1]:  Zero-anchored piecewise CDF.  Threshold = 1 - 2*ratio.
  - Gain/ORIC:      Unbounded.  Threshold from training-set empirical
                     percentile (or test-set fallback if no training data).
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


class MetricType(enum.Enum):
    """Classification of proxy metrics by their threshold strategy."""

    MORIC = "moric"            # [0, 1], CDF-based
    MORIC_PLUS = "moric_plus"  # [-1, 1], zero-anchored piecewise CDF
    ABSOLUTE = "absolute"      # gain_*, oric_*, dataset_oric_* — unbounded


def classify_metric(proxy_metric: Optional[str]) -> MetricType:
    """Determine the threshold strategy for a proxy metric.

    Handles both regular and ``dataset_``-prefixed variants.
    """
    if proxy_metric is None:
        return MetricType.ABSOLUTE
    base = proxy_metric.removeprefix("dataset_")
    # Order matters: moric_plus before moric
    if base.startswith("moric_plus"):
        return MetricType.MORIC_PLUS
    if base.startswith("moric"):
        return MetricType.MORIC
    return MetricType.ABSOLUTE


@dataclass
class OffloadDecision:
    """Result of an offloading decision."""

    mask: np.ndarray        # bool[N], True = offload to cloud
    threshold: float        # threshold used for the decision
    target_ratio: float     # requested offloading ratio
    actual_ratio: float     # achieved offloading ratio
    n_offload: int          # number of frames offloaded
    n_total: int            # total number of frames
    metric_type: MetricType
    lambda_mean: float = float("nan")
    lambda_final: float = float("nan")
    trace: Optional[dict[str, Any]] = None

    @property
    def ratio_error(self) -> float:
        """Absolute difference between target and actual ratio."""
        return abs(self.actual_ratio - self.target_ratio)


class Offloader:
    """Makes binary offloading decisions from continuous predictions.

    For CDF-based metrics (MORIC, MORIC+), ``decide`` uses the analytic
    threshold derived from the metric's theoretical distribution.  The gap
    between target and actual ratio directly measures how well the estimator
    preserves the CDF property.

    For absolute metrics, ``decide`` derives the threshold from training-set
    predictions (when provided via ``threshold_predictions``) and applies it
    to test predictions.  The resulting ratio error measures how well the
    estimator's prediction distribution generalises from train to test.

    ``decide_calibrated`` applies train-ECDF thresholding for any metric type.
    """

    def __init__(self, proxy_metric: Optional[str]):
        self.proxy_metric = proxy_metric
        self.metric_type = classify_metric(proxy_metric)

    @staticmethod
    def _build_trace(
        mask: np.ndarray,
        target_ratio: float,
        order: np.ndarray,
        control_trace: Optional[np.ndarray] = None,
        control_name: str = "threshold",
    ) -> dict[str, Any]:
        order = np.asarray(order, dtype=int).reshape(-1)
        ordered_mask = np.asarray(mask, dtype=bool)[order]
        steps = np.arange(1, len(order) + 1, dtype=int)
        cumulative_offload = np.cumsum(ordered_mask.astype(np.int32))
        cumulative_ratio = (
            cumulative_offload / steps if len(steps) > 0 else np.asarray([], dtype=float)
        )
        budget_debt = cumulative_offload.astype(float) - (float(target_ratio) * steps)
        trace = {
            "step": steps,
            "order": order,
            "offload": ordered_mask.astype(np.int8),
            "cumulative_ratio": cumulative_ratio.astype(float),
            "budget_debt": budget_debt.astype(float),
        }
        if control_trace is not None:
            trace[control_name] = np.asarray(control_trace, dtype=float).reshape(-1)
        return trace

    @property
    def is_ratio_compatible(self) -> bool:
        """True if the metric has an analytic threshold formula."""
        return self.metric_type in (MetricType.MORIC, MetricType.MORIC_PLUS)

    def threshold_for_ratio(
        self,
        target_ratio: float,
        predictions: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the decision threshold for a target offloading ratio.

        For CDF metrics the threshold is analytic (tests distribution fidelity).
        For absolute metrics it falls back to empirical percentile.

        Args:
            target_ratio: Desired fraction of frames to offload (0.0 to 1.0).
            predictions: Required for ABSOLUTE metrics (percentile fallback).

        Returns:
            Threshold value.  Frames with ``prediction > threshold`` are
            offloaded.
        """
        if target_ratio <= 0.0:
            return float("inf")
        if target_ratio >= 1.0:
            return float("-inf")

        if self.metric_type == MetricType.MORIC:
            # MORIC in [0,1]: P(MORIC > t) ~ 1 - t  (CDF property)
            return 1.0 - target_ratio

        if self.metric_type == MetricType.MORIC_PLUS:
            # MORIC+ in [-1,1]: P(MORIC+ > t) ~ (1 - t) / 2
            return 1.0 - 2.0 * target_ratio

        # ABSOLUTE: empirical percentile fallback
        if predictions is None:
            raise ValueError(
                f"predictions required for absolute metric '{self.proxy_metric}'"
            )
        return float(np.percentile(predictions, (1.0 - target_ratio) * 100.0))

    def decide(
        self,
        predictions: np.ndarray,
        target_ratio: float,
        threshold_predictions: np.ndarray = None,
    ) -> OffloadDecision:
        """Make binary offloading decisions for all frames.

        Uses the metric-native threshold.  For CDF metrics (MORIC/MORIC+),
        the ratio error reflects how well the estimator's output distribution
        matches the theoretical CDF.  For absolute metrics, the threshold is
        derived from ``threshold_predictions`` (typically training-set
        predictions) so that ratio error measures distribution generalisation.

        Args:
            predictions: Continuous estimator outputs, shape (N,).
            target_ratio: Desired fraction of frames to offload.
            threshold_predictions: Optional predictions used to derive the
                threshold for ABSOLUTE metrics.  Ignored for CDF metrics
                (MORIC/MORIC+) which use analytic thresholds.  When *None*,
                falls back to ``predictions`` (test-set percentile).

        Returns:
            OffloadDecision with binary mask and ratio metadata.
        """
        predictions = np.asarray(predictions, dtype=float)
        n = len(predictions)

        # For absolute metrics, derive threshold from training predictions
        # when available; CDF metrics ignore this (analytic thresholds).
        thresh_source = (
            np.asarray(threshold_predictions, dtype=float)
            if threshold_predictions is not None
            else predictions
        )
        threshold = self.threshold_for_ratio(target_ratio, thresh_source)
        mask = predictions > threshold
        actual_n = int(mask.sum())
        order = np.arange(n, dtype=int)

        return OffloadDecision(
            mask=mask,
            threshold=threshold,
            target_ratio=target_ratio,
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=self.metric_type,
            trace=self._build_trace(
                mask,
                target_ratio,
                order,
                control_trace=np.full(n, threshold, dtype=float),
            ),
        )

    def decide_calibrated(
        self,
        predictions: np.ndarray,
        target_ratio: float,
        calibration_predictions: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        """Apply train-ECDF thresholding for a target offload ratio.

        This method is online-compatible: each test prediction is calibrated
        only against a fixed calibration distribution (typically training
        predictions), never by ranking against future test samples.
        """

        predictions = np.asarray(predictions, dtype=float)
        n = len(predictions)
        order = np.arange(n, dtype=int)

        if calibration_predictions is None:
            raise ValueError(
                "decide_calibrated requires calibration_predictions "
                "(typically training predictions)"
            )
        calib = np.sort(np.asarray(calibration_predictions, dtype=float).reshape(-1))
        if calib.size == 0:
            raise ValueError("calibration_predictions must be non-empty")

        # Calibrate each score against a fixed reference CDF (train/calibration
        # split), avoiding any lookahead over the test stream.
        left = np.searchsorted(calib, predictions, side="left")
        right = np.searchsorted(calib, predictions, side="right")
        calibrated = (left + right) / (2.0 * float(calib.size))
        threshold = 1.0 - float(target_ratio)

        # Ties at the threshold are resolved online by budget debt,
        # guaranteeing no future-frame lookahead.
        mask = np.zeros(n, dtype=bool)
        n_offload = 0
        for step, idx in enumerate(order, start=1):
            value = calibrated[idx]
            if value > threshold:
                take = True
            elif value < threshold:
                take = False
            else:
                take = n_offload < (float(target_ratio) * step)
            mask[idx] = take
            if take:
                n_offload += 1
        actual_n = int(n_offload)

        return OffloadDecision(
            mask=mask,
            threshold=threshold,
            target_ratio=float(target_ratio),
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=self.metric_type,
            trace=self._build_trace(
                mask,
                float(target_ratio),
                order,
                control_trace=np.full(n, threshold, dtype=float),
            ),
        )


@dataclass
class OffloadContext:
    """Runtime inputs provided by an approach-owned offloader."""

    predictions: np.ndarray
    proxy_metric: Optional[str]
    train_predictions: Optional[np.ndarray] = None
    predict_outputs: Optional[dict[str, Any]] = None
    train_predict_outputs: Optional[dict[str, Any]] = None
    stream_order: Optional[np.ndarray] = None


class OffloadPolicy(ABC):
    """Reusable offloading policy module selected by an offloader config."""

    policy_id: str = ""
    mode: str = "curve"

    def __init__(self, **params):
        self.params = dict(params)

    @property
    def label(self) -> str:
        return self.policy_id.replace("_", " ").title()

    def is_available(self, context: OffloadContext) -> bool:
        return True

    @abstractmethod
    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        raise NotImplementedError


class NativeThresholdPolicy(OffloadPolicy):
    policy_id = "native_threshold"
    mode = "curve"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("native_threshold requires a target_ratio")
        offloader = Offloader(context.proxy_metric)
        return offloader.decide(
            context.predictions,
            float(target_ratio),
            threshold_predictions=context.train_predictions,
        )


class ECDFCalibratedPolicy(OffloadPolicy):
    policy_id = "online_ecdf_calibrated"
    mode = "curve"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("online_ecdf_calibrated requires a target_ratio")
        offloader = Offloader(context.proxy_metric)
        return offloader.decide_calibrated(
            context.predictions,
            float(target_ratio),
            calibration_predictions=context.train_predictions,
        )


class ForcedExactRatioPolicy(OffloadPolicy):
    """Offline evaluation-only exact ratio sweep by global top-k selection."""

    policy_id = "forced_exact_ratio"
    mode = "curve"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("forced_exact_ratio requires a target_ratio")

        predictions = np.asarray(context.predictions, dtype=float)
        n = len(predictions)
        order = np.arange(n, dtype=int)

        k = int(np.clip(np.round(float(target_ratio) * n), 0, n))
        if k <= 0:
            mask = np.zeros(n, dtype=bool)
            threshold = float("inf")
        elif k >= n:
            mask = np.ones(n, dtype=bool)
            threshold = float("-inf")
        else:
            ranked = np.argsort(-predictions, kind="mergesort")
            mask = np.zeros(n, dtype=bool)
            chosen = ranked[:k]
            mask[chosen] = True
            threshold = float(predictions[ranked[k - 1]])

        actual_n = int(mask.sum())
        return OffloadDecision(
            mask=mask,
            threshold=threshold,
            target_ratio=float(target_ratio),
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=classify_metric(context.proxy_metric),
            trace=Offloader._build_trace(
                mask,
                float(target_ratio),
                order,
                control_trace=np.full(n, threshold, dtype=float),
            ),
        )


class FixedClassifierPolicy(OffloadPolicy):
    policy_id = "fixed_classifier"
    mode = "fixed"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        threshold = self.params.get("threshold")
        if threshold is None:
            raise ValueError("fixed_classifier requires a configured threshold")
        predictions = np.asarray(context.predictions, dtype=float)
        mask = predictions > float(threshold)
        actual_n = int(mask.sum())
        return OffloadDecision(
            mask=mask,
            threshold=float(threshold),
            target_ratio=float("nan"),
            actual_ratio=actual_n / len(mask) if len(mask) > 0 else 0.0,
            n_offload=actual_n,
            n_total=len(mask),
            metric_type=classify_metric(context.proxy_metric),
        )


class SequentialCSRPolicy(OffloadPolicy):
    policy_id = "sequential_csr"
    mode = "curve"

    def is_available(self, context: OffloadContext) -> bool:
        outputs = context.predict_outputs or {}
        return {"tau", "survival"}.issubset(outputs)

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("sequential_csr requires a target_ratio")
        outputs = context.predict_outputs or {}
        offloader = SequentialCSROffloader(
            outputs["tau"],
            decision_threshold=float(self.params.get("decision_threshold", 0.5)),
        )
        return offloader.decide(
            outputs["survival"],
            float(target_ratio),
            order=context.stream_order,
        )


class SequentialCSRUtilityPolicy(OffloadPolicy):
    policy_id = "sequential_csr_utility"
    mode = "curve"

    def is_available(self, context: OffloadContext) -> bool:
        outputs = context.predict_outputs or {}
        return {"tau", "survival", "primary_norm"}.issubset(outputs)

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("sequential_csr_utility requires a target_ratio")
        outputs = context.predict_outputs or {}
        offloader = SequentialCSRUtilityOffloader(
            outputs["tau"],
            decision_threshold=float(self.params.get("decision_threshold", 0.5)),
        )
        return offloader.decide(
            outputs["primary_norm"],
            outputs["survival"],
            float(target_ratio),
            order=context.stream_order,
        )


class OnlineSQTPolicy(OffloadPolicy):
    policy_id = "online_sqt"
    mode = "curve"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("online_sqt requires a target_ratio")
        offloader = StochasticQuantileTracker(
            eta=float(self.params.get("eta", 0.01)),
            initial_threshold=float(self.params.get("initial_threshold", 0.0)),
        )
        return offloader.decide(
            context.predictions,
            float(target_ratio),
            order=context.stream_order,
        )


class OnlineLVQPolicy(OffloadPolicy):
    policy_id = "online_lvq"
    mode = "curve"

    def decide(self, context: OffloadContext,
               target_ratio: Optional[float] = None) -> OffloadDecision:
        if target_ratio is None:
            raise ValueError("online_lvq requires a target_ratio")
        offloader = LyapunovVirtualQueue(
            V=float(self.params.get("V", 1.0)),
        )
        return offloader.decide(
            context.predictions,
            float(target_ratio),
            order=context.stream_order,
        )


OFFLOAD_POLICY_REGISTRY: dict[str, type[OffloadPolicy]] = {
    NativeThresholdPolicy.policy_id: NativeThresholdPolicy,
    ECDFCalibratedPolicy.policy_id: ECDFCalibratedPolicy,
    ForcedExactRatioPolicy.policy_id: ForcedExactRatioPolicy,
    FixedClassifierPolicy.policy_id: FixedClassifierPolicy,
    SequentialCSRPolicy.policy_id: SequentialCSRPolicy,
    SequentialCSRUtilityPolicy.policy_id: SequentialCSRUtilityPolicy,
    OnlineSQTPolicy.policy_id: OnlineSQTPolicy,
    OnlineLVQPolicy.policy_id: OnlineLVQPolicy,
}


class ConfiguredOffloader:
    """Approach-owned runtime offloader selecting exactly one reusable policy."""

    def __init__(self, name: str, policy_id: str, *, params: Optional[dict[str, Any]] = None):
        self.name = str(name)
        self.policy_id = str(policy_id)
        self.params = dict(params or {})
        policy_cls = OFFLOAD_POLICY_REGISTRY.get(self.policy_id)
        if policy_cls is None:
            raise ValueError(f"Unknown offload policy '{self.policy_id}'")
        self.policy = policy_cls(**self.params)

    @property
    def mode(self) -> str:
        return self.policy.mode

    @property
    def metric_type(self) -> MetricType:
        """Compatibility shim for legacy stubs that inspect offloader metric type."""
        proxy_metric = self.params.get("proxy_metric")
        return classify_metric(proxy_metric)

    def validate(self, context: OffloadContext) -> None:
        if not self.policy.is_available(context):
            raise ValueError(
                f"Offloader '{self.name}' requires unavailable estimator outputs "
                f"for policy '{self.policy_id}'"
            )

    def decide(self, context: OffloadContext | np.ndarray,
               target_ratio: Optional[float] = None,
               threshold_predictions: Optional[np.ndarray] = None) -> OffloadDecision:
        if isinstance(context, OffloadContext):
            self.validate(context)
            return self.policy.decide(context, target_ratio)

        compat_context = OffloadContext(
            predictions=np.asarray(context, dtype=float),
            proxy_metric=self.params.get("proxy_metric"),
            train_predictions=(
                np.asarray(threshold_predictions, dtype=float)
                if threshold_predictions is not None else None
            ),
        )
        self.validate(compat_context)
        return self.policy.decide(compat_context, target_ratio)

    def decide_calibrated(
        self,
        predictions: np.ndarray,
        target_ratio: float,
        calibration_predictions: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        proxy_metric = self.params.get("proxy_metric")
        return Offloader(proxy_metric).decide_calibrated(
            predictions,
            target_ratio,
            calibration_predictions=calibration_predictions,
        )

class SequentialCSROffloader:
    """Sequential offloader operating on CSR survival probabilities."""

    def __init__(self, tau_grid: np.ndarray, decision_threshold: float = 0.5):
        tau_grid = np.asarray(tau_grid, dtype=float).reshape(-1)
        if tau_grid.ndim != 1 or len(tau_grid) == 0:
            raise ValueError("tau_grid must be a non-empty 1D array")
        self.tau_grid = np.sort(tau_grid)
        self.decision_threshold = float(decision_threshold)

    @staticmethod
    def _monotone_survival(survival_probs: np.ndarray) -> np.ndarray:
        survival_probs = np.clip(np.asarray(survival_probs, dtype=float), 0.0, 1.0)
        return np.minimum.accumulate(survival_probs, axis=1)

    def _interp_survival(self, survival_probs: np.ndarray, lam: float) -> float:
        return float(np.interp(
            np.clip(lam, self.tau_grid[0], self.tau_grid[-1]),
            self.tau_grid,
            survival_probs,
        ))

    def decide(
        self,
        survival_probs: np.ndarray,
        target_ratio: float,
        order: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        survival_probs = np.asarray(survival_probs, dtype=float)
        if survival_probs.ndim != 2 or survival_probs.shape[1] != len(self.tau_grid):
            raise ValueError(
                "survival_probs must have shape [N, len(tau_grid)]"
            )

        n = survival_probs.shape[0]
        order = np.arange(n, dtype=int) if order is None else np.asarray(order, dtype=int)
        monotone = self._monotone_survival(survival_probs)
        tau_min = float(self.tau_grid.min())
        tau_max = float(self.tau_grid.max())
        lambda_t = 0.0
        step = 1.0 / np.sqrt(max(n, 1))
        mask = np.zeros(n, dtype=bool)
        lambda_trace = []

        for idx in order:
            lambda_trace.append(lambda_t)
            p_t = self._interp_survival(monotone[idx], lambda_t)
            offload = p_t >= self.decision_threshold
            mask[idx] = offload
            lambda_t = float(np.clip(
                lambda_t + ((1.0 if offload else 0.0) - target_ratio) * step,
                tau_min,
                tau_max,
            ))

        actual_n = int(mask.sum())
        lambda_mean = float(np.mean(lambda_trace)) if lambda_trace else float("nan")
        return OffloadDecision(
            mask=mask,
            threshold=lambda_t,
            target_ratio=target_ratio,
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=MetricType.ABSOLUTE,
            lambda_final=lambda_t,
            lambda_mean=lambda_mean,
            trace=Offloader._build_trace(
                mask,
                target_ratio,
                order,
                control_trace=np.asarray(lambda_trace, dtype=float),
                control_name="lambda_trace",
            ),
        )


class SequentialCSRUtilityOffloader(SequentialCSROffloader):
    """Sequential CSR offloader gated by both survival and utility heads."""

    def decide(
        self,
        primary_norm: np.ndarray,
        survival_probs: np.ndarray,
        target_ratio: float,
        order: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        primary_norm = np.asarray(primary_norm, dtype=float).reshape(-1)
        survival_probs = np.asarray(survival_probs, dtype=float)
        if survival_probs.ndim != 2 or survival_probs.shape[1] != len(self.tau_grid):
            raise ValueError("survival_probs must have shape [N, len(tau_grid)]")
        if len(primary_norm) != survival_probs.shape[0]:
            raise ValueError("primary_norm and survival_probs must share the same length")

        n = survival_probs.shape[0]
        order = np.arange(n, dtype=int) if order is None else np.asarray(order, dtype=int)
        monotone = self._monotone_survival(survival_probs)
        tau_min = float(self.tau_grid.min())
        tau_max = float(self.tau_grid.max())
        lambda_t = 0.0
        step = 1.0 / np.sqrt(max(n, 1))
        mask = np.zeros(n, dtype=bool)
        lambda_trace = []

        for idx in order:
            lambda_trace.append(lambda_t)
            p_t = self._interp_survival(monotone[idx], lambda_t)
            offload = (p_t >= self.decision_threshold) and (primary_norm[idx] > lambda_t)
            mask[idx] = offload
            lambda_t = float(np.clip(
                lambda_t + ((1.0 if offload else 0.0) - target_ratio) * step,
                tau_min,
                tau_max,
            ))

        actual_n = int(mask.sum())
        lambda_mean = float(np.mean(lambda_trace)) if lambda_trace else float("nan")
        return OffloadDecision(
            mask=mask,
            threshold=lambda_t,
            target_ratio=target_ratio,
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=MetricType.ABSOLUTE,
            lambda_final=lambda_t,
            lambda_mean=lambda_mean,
            trace=Offloader._build_trace(
                mask,
                target_ratio,
                order,
                control_trace=np.asarray(lambda_trace, dtype=float),
                control_name="lambda_trace",
            ),
        )


class StochasticQuantileTracker:
    """Robbins-Monro stochastic quantile tracking for online offloading.

    Processes frames sequentially, adaptively tracking the (1−r)-th quantile
    of the estimator's output distribution to maintain a target offloading
    ratio r.

    Decision for frame t:
        a_t = I[m̂_t > T_t]
    Threshold update:
        T_{t+1} = T_t + η · (I[m̂_t ≤ T_t] − (1 − r))

    If the stream becomes globally "easier" (low predictions), the threshold
    decreases to ensure the budget is fully utilised.  Converges to the true
    quantile under mild regularity conditions on the prediction CDF.

    Args:
        tau_grid: Not used (for API compat with SequentialCSROffloader).
        eta: Step size for the Robbins-Monro update (default 0.01).
        initial_threshold: Starting threshold (default 0.0).
    """

    def __init__(self, tau_grid: np.ndarray = None, *,
                 eta: float = 0.01, initial_threshold: float = 0.0):
        self.eta = float(eta)
        self.initial_threshold = float(initial_threshold)

    def decide(
        self,
        predictions: np.ndarray,
        target_ratio: float,
        order: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        """Process frames sequentially, updating threshold after each."""
        predictions = np.asarray(predictions, dtype=float)
        n = len(predictions)
        order = np.arange(n, dtype=int) if order is None else np.asarray(order, dtype=int)

        mask = np.zeros(n, dtype=bool)
        threshold = self.initial_threshold
        threshold_trace = []

        for idx in order:
            threshold_trace.append(threshold)
            offload = predictions[idx] > threshold
            mask[idx] = offload
            indicator = 0.0 if offload else 1.0
            threshold += self.eta * (indicator - (1.0 - target_ratio))

        actual_n = int(mask.sum())
        thresh_mean = float(np.mean(threshold_trace)) if threshold_trace else float("nan")
        return OffloadDecision(
            mask=mask,
            threshold=threshold,
            target_ratio=target_ratio,
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=MetricType.ABSOLUTE,
            lambda_final=threshold,
            lambda_mean=thresh_mean,
            trace=Offloader._build_trace(
                mask,
                target_ratio,
                order,
                control_trace=np.asarray(threshold_trace, dtype=float),
                control_name="threshold_trace",
            ),
        )


class LyapunovVirtualQueue:
    """Lyapunov drift-plus-penalty offloader with virtual queue budget control.

    Frames the offloading decision as an online stochastic optimisation
    problem.  A virtual queue Z_t tracks accumulated budget debt.  The
    decision balances the immediate reward (estimated detection improvement)
    against the queue penalty.

    Decision for frame t:
        a_t = 1  if  V · pred_t > Z_t
    Queue evolution:
        Z_{t+1} = max(0, Z_t + a_t − r)

    Higher V prioritises accuracy at the cost of transient budget overshoots.
    The queue is self-stabilising: over-offloading grows Z, raising the bar
    for subsequent frames.

    Args:
        tau_grid: Not used (for API compat with SequentialCSROffloader).
        V: Control parameter balancing reward vs budget compliance (default 1.0).
    """

    def __init__(self, tau_grid: np.ndarray = None, *,
                 V: float = 1.0):
        self.V = float(V)

    def decide(
        self,
        predictions: np.ndarray,
        target_ratio: float,
        order: Optional[np.ndarray] = None,
    ) -> OffloadDecision:
        """Process frames sequentially with virtual queue tracking."""
        predictions = np.asarray(predictions, dtype=float)
        n = len(predictions)
        order = np.arange(n, dtype=int) if order is None else np.asarray(order, dtype=int)

        mask = np.zeros(n, dtype=bool)
        Z = 0.0
        z_trace = []

        for idx in order:
            z_trace.append(Z)
            offload = self.V * predictions[idx] > Z
            mask[idx] = offload
            Z = max(0.0, Z + float(offload) - target_ratio)

        actual_n = int(mask.sum())
        z_mean = float(np.mean(z_trace)) if z_trace else float("nan")
        return OffloadDecision(
            mask=mask,
            threshold=Z,
            target_ratio=target_ratio,
            actual_ratio=actual_n / n if n > 0 else 0.0,
            n_offload=actual_n,
            n_total=n,
            metric_type=MetricType.ABSOLUTE,
            lambda_final=Z,
            lambda_mean=z_mean,
            trace=Offloader._build_trace(
                mask,
                target_ratio,
                order,
                control_trace=np.asarray(z_trace, dtype=float),
                control_name="queue_trace",
            ),
        )
