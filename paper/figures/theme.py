"""theme.py -- self-contained semantic color/style map for the paper figures.

The plotting scripts (compute_savings.py, map_vs_rho.py, slice_benefit.py,
target_distribution.py) import this module and address colors by CONCEPT
(``T.ADAPT``, ``T.SKIP``, ...), never by raw hue. A one-line remap here reflows
every figure without touching the plot scripts.

This is a vendored, dependency-free version: the typeset paper used an external
house palette, but for the public release we inline equivalent hex values and a
small matplotlib rcParams setup so the figures regenerate with only
numpy / pandas / matplotlib installed. Colors are visually close to but not
byte-identical with the typeset figures (the canonical PDFs are shipped
alongside these scripts).
"""

import matplotlib as mpl

# --- base palette (semantic primaries) ---------------------------------
TEAL = "#2A9D8F"
AZURE = "#3D7EAA"
VIVID = "#E76F51"
PINK = "#D67BA0"
RED = "#D1495B"
GREEN = "#4C956C"
SLATE = "#5C6B73"
BLUE = "#2E5EAA"


def use_style():
    """Apply a clean, paper-friendly matplotlib rcParams baseline."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
    })


use_style()  # apply on import, matching the original module's behavior


def _tint(hex_color, frac):
    """Blend a palette hue toward white by `frac` (0=hue, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * frac) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


# --- offloading strategies (map_vs_rho) --------------------------------
ADAPT = TEAL          # adaptive envelope = our hero
SKIP = AZURE          # skipping (MV2-Lite + OffloadBin), our routing
COND = VIVID          # conditioned (XGBoost + MORIC), our routing
SKIM = _tint(AZURE, 0.45)  # softer skip variant (MV2 + MORIC^+): light azure
EDGEML = PINK         # EdgeML baseline (conditioned)
DCSB = RED            # DCSB fixed-point baseline
REF = "0.45"          # offline oracle reference (grey)
REF2 = "0.70"         # random reference (light grey)
GUIDE = "0.55"        # weak-/strong-only guide lines (grey)

# --- per-frame cost components (compute_savings) -----------------------
EST = AZURE           # skipping estimator C_e (the cost skipping adds)
WEAK = VIVID          # weak / edge detector pass
CLOUD = _tint(SLATE, 0.50)  # cloud / strong band: light slate (less dominant)
SAVE = GREEN          # per-frame saving (cond - skip)
PENALTY = PINK        # sub-breakeven latency penalty

# --- learning-target distributions (target_distribution) ---------------
RAWTGT = SLATE        # raw Delta-AP histogram
MORIC = BLUE          # MORIC^+ histogram
OFFBIN = GREEN        # OffloadBin binary target (our best routing target)
CDF = RED             # CDF overlay line

# --- offload-benefit partition by stratum (slice_benefit) --------------
BENEFIT = _tint(GREEN, 0.30)  # offloading raises per-frame AP (the win)
HARM = _tint(RED, 0.32)       # offloading lowers it (the risk)
NEUTRAL = _tint(SLATE, 0.62)  # unchanged: the dominant, recessive mass
INK = "#333333"               # dark label ink for in-bar numbers
HEADROOM = _tint(BLUE, 0.28)  # weak->strong mAP gain bar (secondary metric)

# --- this venue's column widths (inches), SIGCOMM/ACM ------------------
COL = 3.335           # ACM single-column width
DBL = 7.0             # ACM full text width
