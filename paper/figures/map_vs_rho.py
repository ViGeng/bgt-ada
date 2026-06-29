"""map_vs_rho.pdf: end-to-end mAP@0.5 vs offload budget rho on PASCAL VOC.

Source: ../data/offloading_results.csv (sweep) and offloading_summary.csv (DCSB).
Adaptive envelope is the per-rho max of skipping (MobileNetV2 + OffloadBin)
and conditioned (XGBoost on MORIC).
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
df = pd.read_csv(DATA / "offloading_results.csv")
summary = pd.read_csv(DATA / "offloading_summary.csv")

SKIP = "pre|mobilenet_v2|OffloadBin|focal|online_ecdf_calibrated"
SKIP_M = "pre|mobilenet_v2|MORIC+-AP|online_ecdf_calibrated"
COND_X = "post|xgboost|MORIC-AP|online_ecdf_calibrated"
COND_E = "post|edgeml|MORIC-AP|wmse|native_threshold"
ORACLE = "oracle"
RANDOM = "random"


def curve(name):
    sub = df[df["estimator"] == name].sort_values("ratio")
    return sub["ratio"].to_numpy(), sub["mAP"].to_numpy()


rho_skip, map_skip = curve(SKIP)
rho_skim, map_skim = curve(SKIP_M)
rho_xg, map_xg = curve(COND_X)
rho_em, map_em = curve(COND_E)
rho_or, map_or = curve(ORACLE)
rho_rd, map_rd = curve(RANDOM)

# Adaptive envelope: per-rho max(skip, cond-best)
common = np.intersect1d(rho_skip, rho_xg)
env_skip = pd.Series(map_skip, index=rho_skip).reindex(common).to_numpy()
env_xg = pd.Series(map_xg, index=rho_xg).reindex(common).to_numpy()
map_adapt = np.maximum(env_skip, env_xg)
mask = common > 0.05
flips = np.where(np.diff(np.sign(env_skip[mask] - env_xg[mask])) > 0)[0]
rho_frontier = float(common[mask][flips[0] + 1]) if len(flips) else float("nan")
print("per-rho diff (skip - cond):",
      list(zip(common.tolist(), (env_skip - env_xg).round(4).tolist())))

dcsb_peak = float(summary[summary["estimator"].str.startswith("post|dcsb")]["peak_map"].iloc[0])
# DCSB is a fixed binary rule; its native operating point is recorded in the
# upstream threshold-results CSV (actual_ratio, mAP).
DCSB_RHO = 0.7948
DCSB_MAP = dcsb_peak  # 0.7890

WEAK = 0.7599458346862418
STRONG = 0.7907622758810510

fig, ax = plt.subplots(1, 1, figsize=(T.COL, T.COL * 0.8))

# references ------------------------------------------------------------------
ax.plot(rho_or, map_or, color=T.REF, lw=1.0, ls=":", label="Oracle")
ax.plot(rho_rd, map_rd, color=T.REF2, lw=1.0, ls="--", label="Random")
ax.axhline(WEAK, color=T.GUIDE, lw=0.6, ls="-.", alpha=0.6)
ax.text(0.01, WEAK + 0.0015, "Weak only", fontsize=6, color="0.4")
ax.axhline(STRONG, color=T.GUIDE, lw=0.6, ls="-.", alpha=0.6)
ax.text(0.01, STRONG + 0.0015, "Strong only", fontsize=6, color="0.4")

# baselines + our routings ----------------------------------------------------
ax.plot(rho_em, map_em, color=T.EDGEML, lw=1.3, ls="--",
        label="EdgeML (cond)", marker="^", ms=2.5)
ax.scatter([DCSB_RHO], [DCSB_MAP], color=T.DCSB, marker="*", s=40,
           zorder=6, label="DCSB (cond, fixed)")
ax.plot(rho_xg, map_xg, color=T.COND, lw=1.5,
        label="XGBoost+MORIC (cond, ours)", marker="s", ms=2.5)
ax.plot(rho_skim, map_skim, color=T.SKIM, lw=1.3, ls="--",
        label=r"MV2+MORIC$^{+}$ (skip, ours)", marker="v", ms=2.5)
ax.plot(rho_skip, map_skip, color=T.SKIP, lw=1.5,
        label="MV2+OffloadBin (skip, ours)", marker="o", ms=2.5)
ax.plot(common, map_adapt, color=T.ADAPT, lw=1.8,
        label="Adaptive (envelope, ours)", marker="D", ms=2.5, zorder=5)

if not np.isnan(rho_frontier):
    ax.axvline(rho_frontier, color="0.3", lw=0.6, ls=":", alpha=0.7)
    ax.text(rho_frontier + 0.01, WEAK - 0.005,
            rf"$\rho_{{\mathrm{{frontier}}}}{{\approx}}{rho_frontier:.2f}$",
            fontsize=6.5, color="0.3")

ax.set_xlim(0, 1)
ax.set_xlabel(r"Offload budget $\rho$")
ax.set_ylabel("End-to-end mAP@0.5")
ax.grid(axis="y", alpha=0.2, lw=0.5)
ax.legend(loc="lower right", handlelength=1.8, labelspacing=0.2,
          borderpad=0.2, ncol=1, fontsize=6)

fig.tight_layout(pad=0.3)
out = Path(__file__).resolve().parent / "map_vs_rho.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(Path(__file__).resolve().parent / "png" / "map_vs_rho_preview.png",
            dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print(f"rho_frontier (skip overtakes cond) = {rho_frontier:.3f}")
print(f"DCSB peak = {dcsb_peak:.4f}")


def auc_full(rhos, vals):
    return float(np.trapezoid(np.asarray(vals), np.asarray(rhos)))


print("\nAUC of mAP@0.5 over rho in [0, 1]:")
for name, r, v in [
    ("Skipping (MV2 + OffloadBin)", rho_skip, map_skip),
    ("Skipping (MV2 + MORIC+)", rho_skim, map_skim),
    ("Conditioned (XGBoost MORIC)", rho_xg, map_xg),
    ("Conditioned (EdgeML)", rho_em, map_em),
    ("Adaptive envelope", common, map_adapt),
    ("Random", rho_rd, map_rd),
    ("Oracle", rho_or, map_or),
]:
    print(f"  {name:35s} AUC={auc_full(r, v):.4f}  Peak={max(v):.4f}")

out_csv = Path(__file__).resolve().parent.parent / "data" / "adaptive_envelope.csv"
pd.DataFrame({"ratio": common, "mAP": map_adapt}).to_csv(out_csv, index=False)
print(f"wrote {out_csv}")
