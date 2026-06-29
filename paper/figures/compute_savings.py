"""compute_savings.pdf: per-frame system cost decomposition (Cond vs Skip).

Both panels: paired stacked bars at $\\rho{\\in}\\{0.1,...,0.9\\}$, with C
(\\conditioned, left) and S (\\skipping, right) for each $\\rho$. Stack
components: Estimator $C_e$ + Weak detector + Cloud $\\rho C_s$. Saving
annotations on top show (cond - skip) per frame; penalty color for the
sub-breakeven case.

Panel (a) Compute uses a broken y-axis since the cloud band ($\\rho C_s$,
$C_s{=}280.37$\\,GFLOPs) compresses the device-side band into invisibility.

Source: ../data/estimator_metrics.csv (measured GFLOPs and timings for the
weak/strong detectors and the MobileNetV2-Lite skip estimator).
Styling: shared figstyle kit via theme.py (colors by meaning, not hue).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theme as T  # noqa: E402

# Per-frame costs --- read straight from the measured results so the figure
# stays grounded in data/estimator_metrics.csv (no hand-entered numbers):
#   weak / strong detector cost = its forward pass (gflops, detector_time_ms)
#   skipping estimator cost     = MobileNetV2-Lite trained on OffloadBin
#                                 (gflops, inference_time_ms)
M = pd.read_csv(Path(__file__).resolve().parent.parent / "data" /
                "estimator_metrics.csv")
SKIP_EST = "pre|mobilenet_v2|OffloadBin|focal|online_ecdf_calibrated"
weak = M[M["base_model"] == "weak_model"].iloc[0]
strong = M[M["base_model"] == "strong_model"].iloc[0]
est = M[M["estimator"] == SKIP_EST].iloc[0]
C_w_g, C_w_t = float(weak["gflops"]), float(weak["detector_time_ms"])
C_s_g, C_s_t = float(strong["gflops"]), float(strong["detector_time_ms"])
C_e_g, C_e_t = float(est["gflops"]), float(est["inference_time_ms"])

rho = np.arange(0.1, 0.95, 0.1)


def parts(C_w, C_s, C_e):
    cond_e = np.zeros_like(rho)
    cond_w = np.full_like(rho, C_w)
    cond_c = rho * C_s
    skip_e = np.full_like(rho, C_e)
    skip_w = (1 - rho) * C_w
    skip_c = rho * C_s
    return cond_e, cond_w, cond_c, skip_e, skip_w, skip_c


cE_g, cW_g, cC_g, sE_g, sW_g, sC_g = parts(C_w_g, C_s_g, C_e_g)
cE_t, cW_t, cC_t, sE_t, sW_t, sC_t = parts(C_w_t, C_s_t, C_e_t)
T_cond_g = cE_g + cW_g + cC_g
T_skip_g = sE_g + sW_g + sC_g
T_cond_t = cE_t + cW_t + cC_t
T_skip_t = sE_t + sW_t + sC_t
save_g = T_cond_g - T_skip_g
save_t = T_cond_t - T_skip_t

W = 0.038
GAP = 0.006
xs_C = rho - (W + GAP) / 2
xs_S = rho + (W + GAP) / 2


def stacked_pair(ax, e_C, w_C, c_C, e_S, w_S, c_S):
    ax.bar(xs_C, e_C, width=W, color=T.EST, edgecolor="white", lw=0.4)
    ax.bar(xs_C, w_C, width=W, bottom=e_C, color=T.WEAK, edgecolor="white", lw=0.4)
    ax.bar(xs_C, c_C, width=W, bottom=e_C + w_C,
           color=T.CLOUD, edgecolor="white", lw=0.4)
    ax.bar(xs_S, e_S, width=W, color=T.EST, edgecolor="white", lw=0.4)
    ax.bar(xs_S, w_S, width=W, bottom=e_S, color=T.WEAK, edgecolor="white", lw=0.4)
    ax.bar(xs_S, c_S, width=W, bottom=e_S + w_S,
           color=T.CLOUD, edgecolor="white", lw=0.4)


def cs_labels(ax, ymax):
    pad = ymax * 0.018
    for r in rho:
        ax.text(r - (W + GAP) / 2, -pad, "C",
                ha="center", va="top", fontsize=6.0, color="0.4", clip_on=False)
        ax.text(r + (W + GAP) / 2, -pad, "S",
                ha="center", va="top", fontsize=6.0, color="0.4", clip_on=False)
    ax.tick_params(axis="x", pad=12, length=0)


def annotate_savings(ax, T_C, T_S, save, ymax):
    pad = ymax * 0.012
    for r, tc, ts, sv in zip(rho, T_C, T_S, save):
        txt = rf"$-${abs(sv):.1f}" if sv >= 0 else rf"$+${abs(sv):.1f}"
        ax.annotate(txt, xy=(r, max(tc, ts) + pad),
                    ha="center", va="bottom",
                    fontsize=6.6, color=T.SAVE if sv >= 0 else T.PENALTY,
                    fontweight="bold")


fig = plt.figure(figsize=(T.DBL, 2.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22,
                      left=0.07, right=0.985, top=0.91, bottom=0.20)

# Panel (a) Compute: broken y-axis -------------------------------------------
gs_a = gs[0].subgridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.08)
ax_top = fig.add_subplot(gs_a[0])
ax_bot = fig.add_subplot(gs_a[1])

stacked_pair(ax_top, cE_g, cW_g, cC_g, sE_g, sW_g, sC_g)
stacked_pair(ax_bot, cE_g, cW_g, cC_g, sE_g, sW_g, sC_g)

TOP_LO, TOP_HI = 30, max(T_cond_g.max(), T_skip_g.max()) * 1.18
BOT_LO, BOT_HI = 0, 12
ax_top.set_ylim(TOP_LO, TOP_HI)
ax_bot.set_ylim(BOT_LO, BOT_HI)
ax_top.set_xlim(0.04, 0.96)
ax_bot.set_xlim(0.04, 0.96)

ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)
ax_top.set_xticks(rho)
ax_bot.set_xticks(rho)
ax_bot.set_xticklabels([f"{r:.1f}" for r in rho])

d_y = 0.018
d_x = 0.010
kw = dict(color="k", clip_on=False, lw=0.7)
ax_top.plot([0 - d_x, 0 + d_x], [-d_y, +d_y], transform=ax_top.transAxes, **kw)
ax_top.plot([1 - d_x, 1 + d_x], [-d_y, +d_y], transform=ax_top.transAxes, **kw)
ax_bot.plot([0 - d_x, 0 + d_x], [1 - d_y, 1 + d_y], transform=ax_bot.transAxes, **kw)
ax_bot.plot([1 - d_x, 1 + d_x], [1 - d_y, 1 + d_y], transform=ax_bot.transAxes, **kw)

annotate_savings(ax_top, T_cond_g, T_skip_g, save_g, TOP_HI)
cs_labels(ax_bot, BOT_HI)

ax_top.set_title("(a) Compute (GFLOPs / frame)", fontsize=8.5)
fig.text(0.015,
         (gs_a[0].get_position(fig).y0 + gs_a[1].get_position(fig).y1) / 2,
         "GFLOPs / frame", rotation=90, ha="center", va="center", fontsize=8)
ax_bot.set_xlabel(r"Offload budget $\rho$  (C=Cond, S=Skip)", labelpad=10)
for ax in (ax_top, ax_bot):
    ax.grid(axis="y", alpha=0.18, lw=0.5)

handles = [
    Patch(facecolor=T.CLOUD, label=r"Cloud  $\rho C_s$"),
    Patch(facecolor=T.WEAK, label=r"Weak detector"),
    Patch(facecolor=T.EST, label=r"Estimator $C_e$"),
]
ax_top.legend(handles=handles, loc="upper left", handlelength=1.2,
              handleheight=1.0, labelspacing=0.25, borderpad=0.2, fontsize=6.4)

# Panel (b) Latency ----------------------------------------------------------
ax_t = fig.add_subplot(gs[1])
stacked_pair(ax_t, cE_t, cW_t, cC_t, sE_t, sW_t, sC_t)

YMAX_T = max(T_cond_t.max(), T_skip_t.max()) * 1.20
ax_t.set_xlim(0.04, 0.96)
ax_t.set_ylim(0, YMAX_T)
ax_t.set_xticks(rho)
ax_t.set_xticklabels([f"{r:.1f}" for r in rho])
annotate_savings(ax_t, T_cond_t, T_skip_t, save_t, YMAX_T)
cs_labels(ax_t, YMAX_T)
ax_t.set_xlabel(r"Offload budget $\rho$  (C=Cond, S=Skip)", labelpad=10)
ax_t.set_ylabel("ms / frame", labelpad=2)
ax_t.set_title("(b) Latency (ms / frame)", fontsize=8.5)
ax_t.grid(axis="y", alpha=0.18, lw=0.5)
ax_t.legend(handles=handles, loc="upper left", handlelength=1.2,
            handleheight=1.0, labelspacing=0.25, borderpad=0.2, fontsize=6.4)

out = Path(__file__).resolve().parent / "compute_savings.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(Path(__file__).resolve().parent / "png" / "compute_savings_preview.png",
            dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print(f"rho*_compute = {C_e_g/C_w_g:.3f}   rho*_latency = {C_e_t/C_w_t:.3f}")
for r, sg, st in zip(rho, save_g, save_t):
    print(f"  rho={r:.1f}: GFLOPs +{sg:.2f}, ms {st:+.2f}")
