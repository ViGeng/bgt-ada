"""Pipeline defaults — edit here to change pipeline behavior."""

# Compute
DEVICE = "cuda:0"
SEED = 42

# Control flags
FORCE_RETRAIN = False
FORCE_RE_DERIVE = False

# Offloading evaluation granularity (0%, 10%, 20%, ..., 100%)
OFFLOAD_RATIOS = [round(i * 0.1, 2) for i in range(0, 11)]

# Prepare-phase parallelism
# 0 = auto (min(cpu_count, 8)), 1 = sequential, N = use N workers
DERIVE_NUM_WORKERS = 32

# Evaluate-phase parallelism
# 0 = auto (CPU-only), 1 = sequential, N = use N workers
# CUDA evaluation stays sequential to avoid GPU contention and OOM.
EVALUATION_NUM_WORKERS = 0

# ORIC context size: number of randomly sampled context frames per target.
# 0 = use all frames in the video (no sampling).
ORIC_CONTEXT_SIZE = 1000

# Number of independent context draws to average when deriving contextual
# rewards and LCER targets. The paper-faithful EdgeML MORIC baseline uses
# a single representative context draw with |E| = 1000 by default.
ORIC_CONTEXT_DRAWS = 1

# Latency measurement settings
# Number of warm-up iterations before timing (avoids cold-start artifacts)
LATENCY_WARMUP = 5
# Number of single-image samples to measure and average
LATENCY_SAMPLES = 50

# Paper-oriented evaluation defaults
EVALUATION_SEEDS = []  # empty -> use [SEED] for backward-compatible single-run behaviour
BOOTSTRAP_SAMPLES = 1000
CALIBRATION_BINS = 10
FIXED_RATIO_POINTS = [0.2, 0.4, 0.6, 0.8]
