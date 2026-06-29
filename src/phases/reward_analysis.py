"""Proxy-metric distribution analysis — gain/proxy-metric diagnostics.

Analyses the absolute and relative value distributions of all gain and
proxy-metric families (gain, ORIC, MORIC, MORIC+, MORIC★, Φ-MORIC) across train/test
splits. Helps diagnose proxy-metric design and loss choice.

Results are cached: if the prepared data (data.npz) hasn't changed,
previously computed statistics and figures are reused.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import PipelineConfig

from .. import log

# Proxy metric families and their members
METRIC_FAMILIES = {
    "gain": ["gain_11pt", "gain_allpoint", "gain_coco"],
    "oric": ["oric_11pt", "oric_allpoint", "oric_coco"],
    "moric": ["moric_11pt", "moric_allpoint", "moric_coco"],
    "moric_plus": ["moric_plus_11pt", "moric_plus_allpoint", "moric_plus_coco"],
    "moric_star": ["moric_star_11pt", "moric_star_allpoint", "moric_star_coco"],
    "phi_moric": ["phi_moric_11pt", "phi_moric_allpoint", "phi_moric_coco"],
    "sigmoric": ["sigmoric_11pt", "sigmoric_allpoint", "sigmoric_coco"],
    "lcer": ["lcer_11pt", "lcer_allpoint", "lcer_coco"],
}

ALL_METRICS = [m for fam in METRIC_FAMILIES.values() for m in fam]

# Display labels for figures
_METRIC_LABELS = {
    "gain_11pt": "Gain (11pt)",
    "gain_allpoint": "Gain (AllPt)",
    "gain_coco": "Gain (COCO)",
    "oric_11pt": "ORIC (11pt)",
    "oric_allpoint": "ORIC (AllPt)",
    "oric_coco": "ORIC (COCO)",
    "moric_11pt": "MORIC (11pt)",
    "moric_allpoint": "MORIC (AllPt)",
    "moric_coco": "MORIC (COCO)",
    "moric_plus_11pt": "MORIC+ (11pt)",
    "moric_plus_allpoint": "MORIC+ (AllPt)",
    "moric_plus_coco": "MORIC+ (COCO)",
    "moric_star_11pt": "MORIC★ (11pt)",
    "moric_star_allpoint": "MORIC★ (AllPt)",
    "moric_star_coco": "MORIC★ (COCO)",
    "phi_moric_11pt": "Φ-MORIC (11pt)",
    "phi_moric_allpoint": "Φ-MORIC (AllPt)",
    "phi_moric_coco": "Φ-MORIC (COCO)",
    "sigmoric_11pt": "SigMORIC (11pt)",
    "sigmoric_allpoint": "SigMORIC (AllPt)",
    "sigmoric_coco": "SigMORIC (COCO)",
    "lcer_11pt": "LCER (11pt)",
    "lcer_allpoint": "LCER (AllPt)",
    "lcer_coco": "LCER (COCO)",
}

_FAMILY_COLORS = {
    "gain": "#4C9BE8",
    "oric": "#E8734C",
    "moric": "#6BBF59",
    "moric_plus": "#B07CD8",
    "moric_star": "#D84B9E",
    "phi_moric": "#7B4FD1",
    "sigmoric": "#E05D9C",
    "lcer": "#2F8F9D",
}
_CACHE_VERSION = "proxy-metric-refresh-v15"

# ---- Fingerprinting / caching -------------------------------------------


def _compute_fingerprint(npz_path: Path, seed: int) -> str:
    """Fingerprint based on data.npz mtime + size + config seed."""
    stat = npz_path.stat()
    raw = f"{stat.st_mtime_ns}:{stat.st_size}:{seed}:{_CACHE_VERSION}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _is_cached(stats_path: Path, fingerprint: str,
               expected_figures: List[Path]) -> bool:
    """Check whether cached results are still valid."""
    if not stats_path.exists():
        return False
    try:
        with open(stats_path) as f:
            cached = json.load(f)
        if cached.get("_fingerprint") != fingerprint:
            return False
    except (json.JSONDecodeError, KeyError):
        return False
    return all(p.exists() for p in expected_figures)


# ---- Statistics computation ----------------------------------------------


def _compute_metric_stats(values: np.ndarray) -> Dict[str, Any]:
    """Compute descriptive statistics for a single metric array."""
    from scipy.stats import kurtosis, skew

    n = len(values)
    if n == 0:
        return {}

    q1, median, q3 = np.percentile(values, [25, 50, 75])
    n_pos = int(np.sum(values > 0))
    n_neg = int(np.sum(values < 0))
    n_zero = int(np.sum(values == 0))

    return {
        "count": n,
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
        "median": round(float(median), 6),
        "q1": round(float(q1), 6),
        "q3": round(float(q3), 6),
        "skewness": round(float(skew(values)), 6),
        "kurtosis": round(float(kurtosis(values)), 6),
        "frac_positive": round(n_pos / n, 4),
        "frac_negative": round(n_neg / n, 4),
        "frac_zero": round(n_zero / n, 4),
        "iqr": round(float(q3 - q1), 6),
    }


def compute_all_stats(data: dict) -> Dict[str, Dict[str, Any]]:
    """Compute statistics for all available proxy metrics (train + test)."""
    stats = {}
    for metric in ALL_METRICS:
        for split in ("train", "test"):
            key = f"y_{split}_{metric}"
            arr = data.get(key)
            if arr is not None and len(arr) > 0:
                stats[f"{split}/{metric}"] = _compute_metric_stats(arr)
    return stats


# ---- Figure generation ---------------------------------------------------


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
    })
    return plt


def _metric_family(metric: str) -> str:
    return next((fam for fam, members in METRIC_FAMILIES.items() if metric in members),
                "other")


def _metric_display_bounds(metric: str):
    if metric.startswith("gain_"):
        return (-1.0, 1.0)
    if metric.startswith("moric_plus_"):
        return (-1.0, 1.0)
    if metric.startswith("lcer_"):
        return None
    if metric.startswith("moric_"):
        return (0.0, 1.0)
    return None


def _robust_plot_bounds(*arrays, metric: str = "",
                        lower_q: float = 0.5,
                        upper_q: float = 99.5,
                        pad_frac: float = 0.08):
    valid = [np.asarray(arr, dtype=float).reshape(-1) for arr in arrays
             if arr is not None and len(arr) > 0]
    if not valid:
        return (0.0, 1.0)

    combined = np.concatenate(valid)
    lo = float(np.nanpercentile(combined, lower_q))
    hi = float(np.nanpercentile(combined, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(combined))
        hi = float(np.nanmax(combined))

    hard_bounds = _metric_display_bounds(metric)
    if hard_bounds is not None:
        lo = min(lo, hard_bounds[0])
        hi = max(hi, hard_bounds[1])
    if hi <= lo:
        hi = lo + 1.0

    pad = max((hi - lo) * pad_frac, 0.02)
    return lo - pad, hi + pad


def plot_histograms(data: dict, out_dir: Path,
                    dataset_label: str = "") -> Path:
    """Wide grid of train/test overlaid histograms with robust display ranges."""
    plt = _setup_matplotlib()

    available = [m for m in ALL_METRICS
                 if f"y_train_{m}" in data and f"y_test_{m}" in data]
    n = len(available)
    if n == 0:
        return out_dir / "proxy_metric_histograms.png"

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.45 * ncols, 2.85 * nrows),
                             squeeze=False)

    for idx, metric in enumerate(available):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        train_vals = data[f"y_train_{metric}"]
        test_vals = data[f"y_test_{metric}"]

        lo, hi = _robust_plot_bounds(train_vals, test_vals, metric=metric)
        bins = np.linspace(lo, hi, 44)
        clip_train = float(np.mean((train_vals < lo) | (train_vals > hi)) * 100.0)
        clip_test = float(np.mean((test_vals < lo) | (test_vals > hi)) * 100.0)
        ax.hist(train_vals, bins=bins, alpha=0.48, label="train",
                color="#4C9BE8", density=True)
        ax.hist(test_vals, bins=bins, alpha=0.42, label="test",
                color="#E8734C", density=True)
        ax.set_xlim(lo, hi)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=9.5)
        if r == nrows - 1:
            ax.set_xlabel("Value")
        if c == 0:
            ax.set_ylabel("Density")
        ax.text(0.02, 0.96,
                f"clip train/test={clip_train:.1f}%/{clip_test:.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D7D7D7", alpha=0.9))
        ax.grid(True, alpha=0.18, ls="--")
        if idx == 0:
            ax.legend(frameon=False, loc="upper right")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    suptitle = "Proxy Metric Distributions (Train vs Test)"
    if dataset_label:
        suptitle += f" [{dataset_label}]"
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / "proxy_metric_histograms.png"
    plt.savefig(path)
    plt.close()
    return path


def _families_with_proxy_metric_data(data: dict) -> dict:
    families = {
        fam: [m for m in members if f"y_train_{m}" in data and f"y_test_{m}" in data]
        for fam, members in METRIC_FAMILIES.items()
    }
    return {k: v for k, v in families.items() if v}


def _plot_proxy_metric_family_axes(axes, data: dict, families_with_data: dict) -> None:
    metric_colors = ["#4C9BE8", "#E8734C", "#6BBF59"]
    linestyles = ["-", "--", ":"]

    n = len(families_with_data)
    for idx, (fam, members) in enumerate(families_with_data.items()):
        ax = axes[idx]
        ax_cdf = ax.twinx()

        family_arrays = []
        for metric in members:
            family_arrays.append(data[f"y_train_{metric}"])
            family_arrays.append(data[f"y_test_{metric}"])
        lo, hi = _robust_plot_bounds(*family_arrays, metric="")
        bins = np.linspace(lo, hi, 42)

        for j, metric in enumerate(members):
            color = metric_colors[j % len(metric_colors)]
            label = _METRIC_LABELS.get(metric, metric)
            train_vals = data[f"y_train_{metric}"]
            test_vals = data[f"y_test_{metric}"]

            ax.hist(train_vals, bins=bins, density=True, color=color,
                    alpha=0.18, histtype="stepfilled")
            ax.hist(test_vals, bins=bins, density=True, color=color,
                    histtype="step", linewidth=1.25, label=label)

            vals = np.sort(train_vals)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax_cdf.plot(vals, cdf, color=color,
                        ls=linestyles[j % len(linestyles)], lw=1.6, alpha=0.95)

        ax.set_title(f"{fam.upper()} Family", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Value")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.grid(True, alpha=0.18, ls="--")
        ax.text(0.03, 0.95, "fill=train density | outline=test density | line=train CDF",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.4,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D7D7D7", alpha=0.9))
        ax.legend(frameon=False, loc="upper right")

        ax_cdf.set_xlim(lo, hi)
        ax_cdf.set_ylim(0.0, 1.02)
        if idx == n - 1:
            ax_cdf.set_ylabel("CDF")
        else:
            ax_cdf.set_yticklabels([])
        ax_cdf.grid(False)


def _plot_proxy_metric_boxplot_axis(ax, data: dict) -> bool:
    available = [m for m in ALL_METRICS if f"y_train_{m}" in data]
    if not available:
        return False

    values = [data[f"y_train_{m}"] for m in available]
    labels = [_METRIC_LABELS.get(m, m) for m in available]
    vp = ax.violinplot(values, positions=np.arange(1, len(available) + 1),
                       showmeans=False, showmedians=False, showextrema=False,
                       widths=0.88)

    all_values = np.concatenate(values)
    display_lo, display_hi = _robust_plot_bounds(
        all_values, metric="", lower_q=0.5, upper_q=99.5
    )

    for i, (metric, body) in enumerate(zip(available, vp["bodies"]), start=1):
        fam = _metric_family(metric)
        color = _FAMILY_COLORS.get(fam, "#AAAAAA")
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)

        vals = np.asarray(data[f"y_train_{metric}"], dtype=float)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        q05, q95 = np.percentile(vals, [5, 95])
        mean = float(np.mean(vals))

        ax.vlines(i, q05, q95, color="#3A3A3A", lw=1.1, alpha=0.55, zorder=2)
        ax.vlines(i, q1, q3, color="#1E1E1E", lw=3.6, alpha=0.85, zorder=3)
        ax.scatter(i, med, color="#111111", s=18, zorder=4, label=None)
        ax.scatter(i, mean, color="white", edgecolor="#111111",
                   marker="D", s=28, zorder=4, label=None)

    ax.set_ylabel("Value")
    ax.set_ylim(display_lo, display_hi)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, alpha=0.18, ls="--", axis="y")
    ax.axhline(y=0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.margins(x=0.02)

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color=_FAMILY_COLORS["gain"], alpha=0.45, label="Gain"),
        mpatches.Patch(color=_FAMILY_COLORS["oric"], alpha=0.45, label="ORIC"),
        mpatches.Patch(color=_FAMILY_COLORS["moric"], alpha=0.45, label="MORIC"),
        mpatches.Patch(color=_FAMILY_COLORS["moric_plus"], alpha=0.45, label="MORIC+"),
        mpatches.Patch(color=_FAMILY_COLORS["lcer"], alpha=0.45, label="LCER"),
        mlines.Line2D([], [], color="#111111", marker="o", linestyle="None",
                      markersize=4, label="Median"),
        mlines.Line2D([], [], color="#111111", marker="D", markerfacecolor="white",
                      linestyle="None", markersize=5, label="Mean"),
    ]
    ax.legend(handles=handles, ncol=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.03), frameon=False)
    return True


def plot_cdfs(data: dict, out_dir: Path,
              dataset_label: str = "") -> Path:
    """Standalone family-level density and CDF figure."""
    plt = _setup_matplotlib()

    families_with_data = _families_with_proxy_metric_data(data)
    n = len(families_with_data)
    if n == 0:
        return out_dir / "proxy_metric_cdfs.png"

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 5.5 * nrows), squeeze=False)
    flat_axes = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    _plot_proxy_metric_family_axes(flat_axes[:n], data, families_with_data)
    for extra in flat_axes[n:]:
        extra.set_visible(False)

    suptitle = "Proxy Metric Family Densities + CDFs"
    if dataset_label:
        suptitle += f" [{dataset_label}]"
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / "proxy_metric_cdfs.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_boxplots(data: dict, out_dir: Path,
                  dataset_label: str = "") -> Path:
    """Standalone violin-style distribution summary with median/mean overlays."""
    plt = _setup_matplotlib()

    families_with_data = _families_with_proxy_metric_data(data)
    available = [m for m in ALL_METRICS if f"y_train_{m}" in data]
    if not available or not families_with_data:
        return out_dir / "proxy_metric_violin_summary.png"

    family_count = len(families_with_data)

    fig, ax = plt.subplots(
        figsize=(max(14.0, len(available) * 1.1), 5.5)
    )
    _plot_proxy_metric_boxplot_axis(ax, data)
    title = "Proxy Metric Violin Summary (Train)"
    if dataset_label:
        title += f" [{dataset_label}]"
    ax.set_title(title, fontsize=11)
    # Rotate x-tick labels when many violins to avoid overlap
    if len(available) > 12:
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = out_dir / "proxy_metric_violin_summary.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_correlation(data: dict, out_dir: Path,
                     dataset_label: str = "") -> Path:
    """Actionable quality view: proxy usefulness vs gain target (train).

    Left panel: Spearman rho to gain_11pt (or available gain fallback).
    Right panel: directional agreement with gain sign (>0 means helpful offload).
    """
    plt = _setup_matplotlib()
    from scipy.stats import spearmanr

    available = [m for m in ALL_METRICS if f"y_train_{m}" in data]
    if len(available) < 2:
        return out_dir / "proxy_metric_correlation.png"

    target_metric = next((m for m in ["gain_11pt", "gain_allpoint", "gain_coco"]
                          if f"y_train_{m}" in data), None)
    if target_metric is None:
        return out_dir / "proxy_metric_correlation.png"

    target = np.asarray(data[f"y_train_{target_metric}"]).reshape(-1)
    n = len(target)
    rows = []
    for metric in available:
        values = np.asarray(data[f"y_train_{metric}"]).reshape(-1)
        if len(values) != n:
            continue

        rho = float(spearmanr(values, target).correlation)
        if np.isnan(rho):
            rho = 0.0

        sign_match = float(np.mean((values > 0) == (target > 0)))
        family = next((fam for fam, members in METRIC_FAMILIES.items()
                       if metric in members), "other")
        rows.append({
            "metric": metric,
            "label": _METRIC_LABELS.get(metric, metric),
            "family": family,
            "rho": rho,
            "sign_match": sign_match,
        })

    if not rows:
        return out_dir / "proxy_metric_correlation.png"

    order = sorted(rows, key=lambda x: x["rho"], reverse=True)
    labels = [r["label"] for r in order]
    rho_vals = [r["rho"] for r in order]
    sign_vals = [r["sign_match"] for r in order]

    family_colors = {
        "gain": "#4C9BE8",
        "oric": "#E8734C",
        "moric": "#6BBF59",
        "moric_plus": "#B07CD8",
        "lcer": "#2F8F9D",
        "other": "#AAAAAA",
    }
    colors = [family_colors.get(r["family"], "#AAAAAA") for r in order]

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(10, len(order) * 0.8),
                                     max(5, len(order) * 0.45)),
                             squeeze=False)
    ax_rho, ax_sign = axes[0]

    bars_rho = ax_rho.barh(labels, rho_vals, color=colors)
    ax_rho.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax_rho.set_xlim(-1.0, 1.0)
    ax_rho.set_xlabel(f"Spearman rho vs {_METRIC_LABELS.get(target_metric, target_metric)}")
    ax_rho.set_title("Monotonic Correlation")
    ax_rho.invert_yaxis()
    for bar, v in zip(bars_rho, rho_vals):
        ax_rho.text(v + (0.02 if v >= 0 else -0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)

    bars_sign = ax_sign.barh(labels, sign_vals, color=colors)
    ax_sign.set_xlim(0.0, 1.0)
    ax_sign.set_xlabel("Directional Agreement with Gain Sign")
    ax_sign.set_title("Decision Consistency")
    ax_sign.invert_yaxis()
    for bar, v in zip(bars_sign, sign_vals):
        ax_sign.text(min(v + 0.015, 0.98),
                     bar.get_y() + bar.get_height() / 2,
                     f"{v:.2f}", va="center", ha="left", fontsize=8)

    title = "Proxy Metric Usefulness (Train)"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=12, y=1.02)
    ax_rho.grid(True, alpha=0.2, ls="--", axis="x")
    ax_sign.grid(True, alpha=0.2, ls="--", axis="x")
    plt.tight_layout()
    path = out_dir / "proxy_metric_correlation.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_pos_neg_breakdown(data: dict, out_dir: Path,
                           dataset_label: str = "") -> Path:
    """Stacked bar chart showing positive / negative / zero fractions."""
    plt = _setup_matplotlib()

    available = [m for m in ALL_METRICS if f"y_train_{m}" in data]
    if not available:
        return out_dir / "proxy_metric_pos_neg.png"

    labels = [_METRIC_LABELS.get(m, m) for m in available]
    frac_pos, frac_neg, frac_zero = [], [], []
    for m in available:
        v = data[f"y_train_{m}"]
        n = len(v)
        frac_pos.append(np.sum(v > 0) / n)
        frac_neg.append(np.sum(v < 0) / n)
        frac_zero.append(np.sum(v == 0) / n)

    frac_pos = np.array(frac_pos)
    frac_neg = np.array(frac_neg)
    frac_zero = np.array(frac_zero)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x, frac_pos, label="Positive (> 0)", color="#6BBF59")
    ax.bar(x, frac_zero, bottom=frac_pos, label="Zero (= 0)", color="#CCCCCC")
    ax.bar(x, frac_neg, bottom=frac_pos + frac_zero,
           label="Negative (< 0)", color="#E85555")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of Samples")
    title = "Positive / Zero / Negative Breakdown (Train)"
    if dataset_label:
        title += f" [{dataset_label}]"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, ls="--", axis="y")
    plt.tight_layout()
    path = out_dir / "proxy_metric_pos_neg.png"
    plt.savefig(path)
    plt.close()
    return path


# ---- Public API ----------------------------------------------------------


def _load_proxy_metric_data(cfg: PipelineConfig) -> dict:
    """Load proxy metric arrays from prepared data.npz."""
    npz_path = cfg.output.prepared_dir / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found at {npz_path}. Run 'prepare' first.")

    d = np.load(npz_path, allow_pickle=True)
    result = {}
    for key in d.keys():
        if key.startswith("y_train_") or key.startswith("y_test_"):
            result[key] = d[key]
    return result, npz_path


def analyse_proxy_metric_distributions(cfg: PipelineConfig) -> None:
    """Run gain/proxy-metric distribution analysis with caching.

    Computes statistics and generates figures for all proxy metrics.
    Skips computation if cached results match the current data fingerprint.
    """
    log.subsection("Proxy-Metric Distribution Analysis")

    data, npz_path = _load_proxy_metric_data(cfg)
    if not data:
        log.skip("No proxy metric data found")
        return

    fingerprint = _compute_fingerprint(npz_path, cfg.seed)
    charts_dir = cfg.output.charts_dir
    charts_dir.mkdir(parents=True, exist_ok=True)
    stats_path = cfg.output.metrics_dir / "proxy_metric_stats.json"

    # Remove deprecated charts to avoid stale outputs from previous runs.
    for deprecated in (
        "reward_pos_neg.png",
        "reward_correlation.png",
        "reward_histograms.png",
        "reward_boxplots.png",
        "reward_cdfs.png",
        "reward_violin_summary.png",
        "proxy_metric_histograms.png",
        "proxy_metric_pos_neg.png",
        "proxy_metric_correlation.png",
    ):
        p = charts_dir / deprecated
        if p.exists():
            p.unlink()

    figure_paths = [
        charts_dir / "proxy_metric_histograms.png",
        charts_dir / "proxy_metric_cdfs.png",
        charts_dir / "proxy_metric_violin_summary.png",
        charts_dir / "proxy_metric_pos_neg.png",
        charts_dir / "proxy_metric_correlation.png",
    ]

    if _is_cached(stats_path, fingerprint, figure_paths):
        log.cached("Data unchanged, skipping recomputation")
        return

    ds_label = cfg.dataset.name.upper()

    log.info("Computing distribution statistics...")
    stats = compute_all_stats(data)

    log.info("Generating figures...")
    p1 = plot_cdfs(data, charts_dir, dataset_label=ds_label)
    log.arrow(f"CDFs: {p1}")

    p2 = plot_boxplots(data, charts_dir, dataset_label=ds_label)
    log.arrow(f"Violin summary: {p2}")

    # Save stats with fingerprint
    stats["_fingerprint"] = fingerprint
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.arrow(str(stats_path))

    # Print summary table
    headers = ["Metric", "Mean", "Std", "Min", "Max", "%Pos", "%Neg", "Skew"]
    rows = []
    for metric in ALL_METRICS:
        key = f"train/{metric}"
        if key not in stats:
            continue
        s = stats[key]
        label = _METRIC_LABELS.get(metric, metric)
        rows.append([
            label,
            f"{s['mean']:.4f}",
            f"{s['std']:.4f}",
            f"{s['min']:.4f}",
            f"{s['max']:.4f}",
            f"{s['frac_positive']:.1%}",
            f"{s['frac_negative']:.1%}",
            f"{s['skewness']:.2f}",
        ])
    log.table(
        headers, rows,
        col_widths=[22, 8, 8, 8, 8, 6, 6, 7],
        alignments=["<", ">", ">", ">", ">", ">", ">", ">"],
    )
    log.success("Proxy-metric distribution analysis complete")


def analyse_reward_distributions(cfg: PipelineConfig) -> None:
    """Backward-compatible alias for older callers."""
    analyse_proxy_metric_distributions(cfg)
