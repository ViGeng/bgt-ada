"""slice_benefit.pdf: a visual replacement for tab:slice. Where does offloading
help? Two weak-detector strata (confidence, crowding), each split into quartiles
Q1-Q4. Left of each row: a 100% horizontal stacked bar per quartile, anchored at
0 so the Benefit (green) segment reads against a common baseline -- its length and
its gradient down the panel are the story; Harm (red) abuts it and the dominant
Neutral (slate) mass fills the rest, making the "sparse partition" visible. Right
of each row: a weak->strong mAP headroom dumbbell (weak orange circle -> strong
slate circle) on the same rows, showing the headroom widens in the same strata
where benefit concentrates.

The right strip shows the weak->strong per-frame mAP gain directly as a labeled
bar in points (pp) -- the quantity the dumbbell made the reader subtract by eye --
so the numbers are small and self-explaining; the absolute mAP_w/mAP_s stay in the
retained tab:slice.

House grammar mirrors compute_savings/map_vs_rho: (a)/(b) titles at 8.5pt, white
segment edges, light value-axis grid, frameless Patch legends, restrained grey
annotations, colors addressed by meaning via theme.py.

Source: ../data/slice_benefit.csv (transcribed from tab:slice in appendix.tex).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme as T  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA / "slice_benefit.csv", comment="#")

# (key, panel title, vertical-axis direction label). The quartile rows run
# Q1 (top) -> Q4 (bottom), i.e. least -> most of the stratifying variable, so
# the rotated axis label reads top-to-bottom in that same direction.
PANELS = [
    ("conf", "(a) Weak confidence", r"low $\rightarrow$ high confidence"),
    ("count", "(b) Scene crowding", r"sparse $\rightarrow$ crowded"),
]
BAR_H = 0.64
HEAD_MAX = (df.map_s - df.map_w).max() * 100 * 1.42   # shared pp x-range + label room

fig = plt.figure(figsize=(T.COL, T.COL * 0.88))
gs = GridSpec(2, 2, figure=fig, width_ratios=[3.0, 1.12],
              height_ratios=[1, 1], hspace=0.46, wspace=0.14,
              left=0.2, right=0.985, top=0.93, bottom=0.12)


def bar_label(ax, x_right, y, text, seg_w):
    """Benefit % in dark ink: inside its segment when it fits, else past it."""
    if seg_w >= 0.16:
        ax.text(x_right - seg_w / 2, y, text, ha="center", va="center",
                color=T.INK, fontsize=6.6, fontweight="bold", zorder=6)
    else:
        ax.text(x_right + 0.02, y, text, ha="left", va="center",
                color=T.INK, fontsize=6.6, fontweight="bold", zorder=6)


for r, (key, title, ydir) in enumerate(PANELS):
    d = df[df.stratum == key].reset_index(drop=True)
    y = list(range(len(d)))[::-1]          # Q1 on top, Q4 at the bottom

    # ---- left: 100% stacked benefit / harm / neutral -------------------
    axb = fig.add_subplot(gs[r, 0])
    axb.barh(y, d.benefit, height=BAR_H, color=T.BENEFIT,
             edgecolor="white", lw=0.5, zorder=3)
    axb.barh(y, d.harm, left=d.benefit, height=BAR_H, color=T.HARM,
             edgecolor="white", lw=0.5, zorder=3)
    axb.barh(y, d.neutral, left=d.benefit + d.harm, height=BAR_H,
             color=T.NEUTRAL, edgecolor="white", lw=0.5, zorder=3)
    for yi, b in zip(y, d.benefit):
        bar_label(axb, b, yi, f"{b*100:.0f}%", b)

    axb.set_xlim(0, 1)
    axb.set_ylim(-0.52, len(d) - 0.48)
    axb.set_yticks(y)
    axb.set_yticklabels(d.quartile)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xticklabels(["0", "25", "50", "75", "100%"])
    axb.tick_params(length=2)
    axb.grid(axis="x", alpha=0.18, lw=0.5, zorder=0)
    axb.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.set_title(title, fontsize=8.5, loc="left", pad=4)
    # vertical axis naming the stratifying variable and its Q1->Q4 direction
    axb.text(-0.26, 0.5, ydir, transform=axb.transAxes, rotation=270,
             rotation_mode="anchor", ha="center", va="center",
             fontsize=6.6, color="0.4")

    # ---- right: weak -> strong mAP gain, as a labeled bar in points -----
    axm = fig.add_subplot(gs[r, 1], sharey=axb)
    gain = (d.map_s - d.map_w) * 100          # percentage points (pp)
    axm.barh(y, gain, height=BAR_H, color=T.HEADROOM, edgecolor="white",
             lw=0.5, zorder=3)
    for yi, g in zip(y, gain):
        axm.text(g + HEAD_MAX * 0.04, yi, f"{g:.1f}", ha="left", va="center",
                 color=T.INK, fontsize=6.3, zorder=6)
    axm.set_xlim(0, HEAD_MAX)
    axm.set_xticks([])
    axm.tick_params(length=2, labelleft=False)
    for s in ("top", "right", "left", "bottom"):
        axm.spines[s].set_visible(False)
    axm.set_title("mAP gain (pp)", fontsize=6.8, color="0.4", pad=4)

# shared frameless legend along the bottom ------------------------------
bar_handles = [
    Patch(facecolor=T.BENEFIT, label="Benefit"),
    Patch(facecolor=T.HARM, label="Harm"),
    Patch(facecolor=T.NEUTRAL, label="Neutral"),
]
fig.legend(handles=bar_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
           fontsize=6.8, handlelength=1.1, columnspacing=1.4,
           handletextpad=0.5)

out = Path(__file__).with_suffix(".pdf")
fig.savefig(out, bbox_inches="tight")
png = Path(__file__).resolve().parent / "png" / "slice_benefit_preview.png"
png.parent.mkdir(exist_ok=True)
fig.savefig(png, dpi=220, bbox_inches="tight")
print("wrote", out)
