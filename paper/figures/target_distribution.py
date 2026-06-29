"""target_distribution.pdf: the per-frame learning targets on VOC test, as a
ridgeline comparison of three rows. Top ridge: raw Delta-AP is degenerate (a
62.6% point mass at 0, drawn as a stem, plus short signed tails). Middle ridge:
the MORIC^+ transform spreads that mass into a smooth, symmetric target on
[-1,1]. Bottom ridge: OffloadBin reduces the target to a binary class label
(1 iff Delta-AP > 0), a ~1:3 split shown as two stems (the classifier target).
A red CDF climbs each ridge (the raw CDF jumps at 0; the binary CDF is a step).

Source: ../data/target_distributions.csv.
Styling: shared figstyle kit via theme.py (colors by meaning, not hue).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme as T  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA / "target_distributions.csv")
draw = df["delta_ap"].to_numpy()
mor = df["moric_plus"].to_numpy()
obin = df["offload_bin"].to_numpy()
zero_frac = float(np.mean(draw == 0.0))      # 0.626
draw_nz = draw[draw != 0.0]                  # signed tails
bin_pos = float(np.mean(obin == 1))          # 0.251 (offload helps)
bin_neg = 1.0 - bin_pos                      # 0.749 (keep local)


def kde_density(x, lo=-1.0, hi=1.0, n=241):
    """Smooth Gaussian-KDE density on a grid (numpy only; Silverman bandwidth).

    A real KDE instead of a binned-then-smoothed histogram, so the curve has
    no bin-to-bin staircase/wiggle and reads as a natural distribution.
    """
    x = np.asarray(x, float)
    grid = np.linspace(lo, hi, n)
    # Medium smoothing: ~0.4x the Silverman bandwidth, so a roughly uniform
    # target (MORIC+) keeps its real undulation instead of collapsing to a
    # flat plateau, while still avoiding per-bin staircase noise.
    bw = 0.40 * 1.06 * np.std(x) * len(x) ** (-0.2)
    bw = max(bw, 0.045)
    u = (grid[None, :] - x[:, None]) / bw
    dens = np.exp(-0.5 * u * u).sum(axis=0) / (len(x) * bw * np.sqrt(2 * np.pi))
    return grid, dens


def ecdf(v):
    s = np.sort(v)
    return s, np.arange(1, len(s) + 1) / len(s)


xc_raw, d_raw = kde_density(draw_nz)
xc_mor, d_mor = kde_density(mor)
RIDGE_H = 1.0
d_raw_n = d_raw / d_raw.max() * RIDGE_H
d_mor_n = d_mor / d_mor.max() * RIDGE_H

fig, ax = plt.subplots(figsize=(T.COL, T.COL * 0.98))
# three stacked baselines (bottom -> top): OffloadBin, MORIC+, raw Delta-AP.
GAP = 0.55                       # clear gap above each unit-height ridge
base_bin = 0.0
base_mor = base_bin + RIDGE_H + GAP   # 1.55
base_raw = base_mor + RIDGE_H + GAP   # 3.10
STEM_H = 1.55                    # raw point-mass stem, tall to dwarf the tails

# bottom ridge: OffloadBin (binary target) -- two class stems + red step CDF
ax.plot([-1, 1], [base_bin, base_bin], color="0.6", lw=0.6, zorder=1)
h0 = RIDGE_H                      # majority class 0 (keep local) sets the height
h1 = RIDGE_H * bin_pos / bin_neg  # class 1 (offload) scaled to the same ridge
for x, h, frac, dx in ((0.0, h0, bin_neg, 0.10), (1.0, h1, bin_pos, 0.0)):
    ax.plot([x, x], [base_bin, base_bin + h], color=T.OFFBIN, lw=2.4,
            solid_capstyle="round", zorder=8)
    ax.plot([x], [base_bin + h], marker="o", ms=3.6, color=T.OFFBIN, zorder=9)
ax.annotate(f"{bin_neg*100:.1f}%", xy=(0.0, base_bin + h0),
            xytext=(0.12, base_bin + h0), color=T.OFFBIN, fontsize=7.0,
            va="center", ha="left", zorder=10)
ax.annotate(f"{bin_pos*100:.1f}%", xy=(1.0, base_bin + h1),
            xytext=(0.98, base_bin + h1 + 0.18), color=T.OFFBIN, fontsize=7.0,
            va="bottom", ha="center", zorder=10)
# binary ECDF is a step: 0 below 0, climbs to bin_neg at x=0, to 1 at x=1.
step_x = np.array([-1.0, 0.0, 0.0, 1.0, 1.0])
step_F = np.array([0.0, 0.0, bin_neg, bin_neg, 1.0])
ax.plot(step_x, base_bin + step_F * RIDGE_H, color=T.CDF, lw=1.1, ls="-",
        zorder=6)

# middle ridge: MORIC+ (good target) -- density fill + curve + red CDF
ax.fill_between(xc_mor, base_mor, base_mor + d_mor_n, color=T.MORIC,
                alpha=0.22, lw=0, zorder=2)
ax.plot(xc_mor, base_mor + d_mor_n, color=T.MORIC, lw=1.3, zorder=3)
ax.plot([-1, 1], [base_mor, base_mor], color="0.6", lw=0.6, zorder=1)
xm, ym = ecdf(mor)
ax.plot(xm, base_mor + ym * RIDGE_H, color=T.CDF, lw=1.1, ls="-", zorder=4)

# top ridge: raw Delta-AP (degenerate target) -- fill + curve + red CDF
ax.fill_between(xc_raw, base_raw, base_raw + d_raw_n, color=T.RAWTGT,
                alpha=0.20, lw=0, zorder=4)
ax.plot(xc_raw, base_raw + d_raw_n, color=T.RAWTGT, lw=1.3, ls=(0, (5, 1.6)),
        zorder=5)
ax.plot([-1, 1], [base_raw, base_raw], color="0.6", lw=0.6, zorder=1)
# CDF over the NONZERO frames only: the full ECDF of dAP would jump ~0.63 at
# x=0 (the point mass); conditioning on dAP != 0 gives a smooth curve. The
# 62.6% mass itself is shown by the stem below, not by the CDF.
xr, yr = ecdf(draw_nz)
ax.plot(xr, base_raw + yr * RIDGE_H, color=T.CDF, lw=1.2, ls="-", zorder=6)

# Point mass at 0: a slate stem is the density-view marker for the 62.6% mass.
# It is drawn over the smooth CDF, which it merely crosses at a single point
# (no jump to occlude, so nothing looks broken).
ax.plot([0, 0], [base_raw, base_raw + STEM_H], color=T.RAWTGT, lw=1.8,
        solid_capstyle="round", zorder=9)
ax.plot([0], [base_raw + STEM_H], marker="o", ms=3.4, color=T.RAWTGT, zorder=10)
ax.annotate(f"{zero_frac*100:.1f}% @ 0",
            xy=(0.04, base_raw + STEM_H),
            xytext=(0.12, base_raw + STEM_H - 0.02),
            color=T.RAWTGT, fontsize=7.0, va="center", ha="left", zorder=10)

# ridge names in the LEFT margin (rotated, outside the data: no collisions)
ax.text(-1.16, base_raw + 0.5 * RIDGE_H, r"raw $\Delta\!\mathrm{AP}$",
        color=T.RAWTGT, fontsize=8.0, rotation=90, ha="center", va="center")
ax.text(-1.16, base_mor + 0.5 * RIDGE_H, r"$\mathrm{MORIC}^{+}$",
        color=T.MORIC, fontsize=8.0, rotation=90, ha="center", va="center")
ax.text(-1.16, base_bin + 0.5 * RIDGE_H, "OffloadBin",
        color=T.OFFBIN, fontsize=8.0, rotation=90, ha="center", va="center")
# CDF tags in the RIGHT margin, at each curve's endpoint
ax.text(1.05, base_bin + RIDGE_H, "CDF", color=T.CDF, fontsize=6.8,
        ha="left", va="center")
ax.text(1.05, base_mor + RIDGE_H, "CDF", color=T.CDF, fontsize=6.8,
        ha="left", va="center")
ax.text(1.05, base_raw + RIDGE_H, "CDF", color=T.CDF, fontsize=6.8,
        ha="left", va="center")

ax.axvline(0.0, color="0.8", lw=0.5, zorder=0)
ax.set_xlim(-1.24, 1.16)
ax.set_ylim(base_bin - 0.10, base_raw + STEM_H + 0.32)
ax.set_xlabel("learning target value")
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([])
ax.spines["left"].set_visible(False)
fig.tight_layout(pad=0.4)

out = Path(__file__).with_suffix(".pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(Path(__file__).resolve().parent / "png" / "target_distribution_preview.png",
            dpi=200, bbox_inches="tight")
print("wrote", out)
