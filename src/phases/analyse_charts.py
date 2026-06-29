"""Core metric quality, timing, complexity, offloading, and prediction distribution charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .analyse_helpers import (
    _FOCUS_TARGET_RATIOS,
    _ORACLE_NAME,
    _REFERENCE_COLORS,
    _STAGE_COLORS,
    _VIRTUAL_NAMES,
    _add_stage_legend,
    _build_estimator_styles,
    _canonical_strategy_name,
    _collapse_scenario_variants,
    _compact_approach_label,
    _compact_estimator_label,
    _detector_display_name,
    _detector_reference_value,
    _ensure_policy_columns,
    _estimator_model_key,
    _figure_title,
    _filter_headline_estimators,
    _format_multiplier,
    _format_value,
    _headline_estimators,
    _prediction_diagnostics_figure,
    _pretty_slice_group_label,
    _pretty_slice_value,
    _real_estimator_rows,
    _reference_callout_text,
    _relative_annotation,
    _setup_matplotlib,
    _slice_heatmap_estimator_labels,
    _slice_sort_key,
    _stage_from_name,
    _with_chart_strategy,
)


def _is_oracle_estimator(name: str) -> bool:
    """Return True if the estimator name is any oracle variant."""
    return name == _ORACLE_NAME or name.startswith("oracle")



def _complexity_panel_specs(df: pd.DataFrame) -> List[dict]:
    panels: List[dict] = []
    if "gflops" in df.columns and df["gflops"].notna().any():
        panels.append({
            "value_col": "gflops",
            "reference_col": "gflops",
            "title": "GFLOPs",
            "xlabel": "GFLOPs",
            "unit": "GFLOPs",
        })
    if "params" in df.columns and df["params"].notna().any():
        panels.append({
            "value_col": "params",
            "reference_col": "params",
            "title": "Parameters",
            "xlabel": "Parameters (M)",
            "unit": "M",
        })
    if "inference_time_ms" in df.columns and df["inference_time_ms"].notna().any():
        panels.append({
            "value_col": "inference_time_ms",
            "reference_col": "detector_time_ms",
            "title": "Latency",
            "xlabel": "Inference time (ms / sample)",
            "unit": "ms / sample",
        })
    return panels


def _complexity_plot_rows(df: pd.DataFrame, panels: List[dict]) -> pd.DataFrame:
    metric_cols = [panel["value_col"] for panel in panels]
    df_plot = df.copy()
    if "status" in df_plot.columns:
        status = df_plot["status"].fillna("").astype(str).str.upper()
        df_plot = df_plot[status != "FAIL"]
    df_plot = df_plot[~df_plot["estimator"].isin(_VIRTUAL_NAMES)].copy()

    keep_mask = np.zeros(len(df_plot), dtype=bool)
    for col in metric_cols:
        keep_mask |= df_plot[col].notna().to_numpy()
    df_plot = df_plot[keep_mask].copy()
    if df_plot.empty:
        return df_plot

    primary_col = next(
        (panel["value_col"] for panel in panels if df_plot[panel["value_col"]].notna().any()),
        metric_cols[0],
    )
    stage_rank = {"pre": 0, "post": 1, "other": 2}
    df_plot["_stage_rank"] = df_plot["estimator"].map(
        lambda name: stage_rank.get(_stage_from_name(name), 2)
    )
    df_plot["_metric_rank"] = df_plot[primary_col].astype(float).fillna(np.inf)
    df_plot = df_plot.sort_values(
        ["_stage_rank", "_metric_rank", "estimator"],
        kind="mergesort",
    ).drop(columns=["_stage_rank", "_metric_rank"])
    return df_plot


def _complexity_legend_handles(dataset_summary: dict = None) -> List[object]:
    import matplotlib.lines as mlines

    weak_name = _detector_display_name(
        "weak_model", (dataset_summary or {}).get("edge_model", "")
    )
    strong_name = _detector_display_name(
        "strong_model", (dataset_summary or {}).get("cloud_model", "")
    )
    return [
        mlines.Line2D([], [], color=_STAGE_COLORS["pre"], marker="o",
                      markeredgecolor="white", linestyle="None",
                      markersize=7, label="Pre-stage estimator"),
        mlines.Line2D([], [], color=_STAGE_COLORS["post"], marker="o",
                      markeredgecolor="white", linestyle="None",
                      markersize=7, label="Post-stage baseline"),
        mlines.Line2D([], [], color="#556270", lw=1.7, ls="-",
                      label=weak_name),
        mlines.Line2D([], [], color="#8C5C46", lw=1.7, ls="--",
                      label=strong_name),
    ]


def _plot_complexity_metric_axis(ax, df_plot: pd.DataFrame, value_col: str,
                                 xlabel: str, title: str,
                                 weak_reference: Optional[float],
                                 strong_reference: Optional[float],
                                 unit: str,
                                 show_ylabels: bool = False) -> None:
    import matplotlib.ticker as mticker

    if value_col not in df_plot.columns or df_plot[value_col].dropna().empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    y = np.arange(len(df_plot))
    values = df_plot[value_col].astype(float).to_numpy()
    valid = np.isfinite(values)
    colors = [
        _STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
        for name in df_plot["estimator"]
    ]

    positive_values = [float(v) for v in values[valid] if v > 0]
    positive_refs = [
        float(v) for v in (weak_reference, strong_reference)
        if v is not None and np.isfinite(v) and v > 0
    ]
    positives = positive_values + positive_refs
    if not positives:
        ax.set_title(f"{title} (no positive data)")
        ax.set_axis_off()
        return

    min_positive = min(positives)
    max_positive = max(positives)
    linthresh = max(min_positive / 2.0, 1e-4)

    ax.set_xscale("symlog", linthresh=linthresh, linscale=1.0, base=10)
    ax.set_xlim(0, max_positive * 1.22)
    ax.hlines(y[valid], 0, values[valid], color="#D7DEE5", lw=2.0, zorder=1)
    ax.scatter(values[valid], y[valid], s=58, c=np.array(colors, dtype=object)[valid],
               edgecolor="white", linewidth=0.9, zorder=2)

    if weak_reference is not None and np.isfinite(weak_reference) and weak_reference > 0:
        ax.axvline(weak_reference, color="#556270", lw=1.7, ls="-", alpha=0.9, zorder=0)
    if strong_reference is not None and np.isfinite(strong_reference) and strong_reference > 0:
        ax.axvline(strong_reference, color="#8C5C46", lw=1.7, ls="--", alpha=0.9, zorder=0)

    def _tick_formatter(value: float, _pos: int) -> str:
        if abs(value) < linthresh * 0.5:
            return "0"
        if value < 0:
            return ""
        return _format_value(value)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_tick_formatter))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([
            _compact_estimator_label(name, multiline=True)
            for name in df_plot["estimator"]
        ])
        ax.set_ylabel("Estimators")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.margins(y=0.10)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.24, ls="--")
    ax.set_axisbelow(True)

    label_floor = linthresh * 0.55
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        xpos = value * 1.14 if value > 0 else label_floor
        ax.text(xpos, y[idx], _format_value(value),
                va="center", ha="left", fontsize=8.1)

    reference_lines = []
    if weak_reference is not None and np.isfinite(weak_reference):
        reference_lines.append(f"Weak: {_format_value(weak_reference)} {unit}")
    if strong_reference is not None and np.isfinite(strong_reference):
        reference_lines.append(f"Strong: {_format_value(strong_reference)} {unit}")
    if reference_lines:
        ax.text(0.98, 0.03, "\n".join(reference_lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.24", fc="white",
                          ec="#D0D0D0", alpha=0.94))


def _dataset_gap_rows(summary: dict, metrics_df: pd.DataFrame = None) -> List[dict]:
    weak_pf_ap50 = None
    strong_pf_ap50 = None
    weak_pf_coco = None
    strong_pf_coco = None
    if metrics_df is not None and "estimator" in metrics_df.columns:
        wm = metrics_df[metrics_df["estimator"] == "weak_model"]
        sm = metrics_df[metrics_df["estimator"] == "strong_model"]
        if not wm.empty:
            weak_pf_ap50 = wm.iloc[0].get("peak_map_coco50", np.nan)
            weak_pf_coco = wm.iloc[0].get("peak_map_coco", np.nan)
        if not sm.empty:
            strong_pf_ap50 = sm.iloc[0].get("peak_map_coco50", np.nan)
            strong_pf_coco = sm.iloc[0].get("peak_map_coco", np.nan)

    rows = [
        {
            "label": "Global AP@0.5\n(IoU\u226550%, pooled PR curve)",
            "weak": float(summary.get("edge_ap50", np.nan)),
            "strong": float(summary.get("cloud_ap50", np.nan)),
        },
        {
            "label": "Per-frame AP@0.5\n(IoU\u226550%, avg over frames)",
            "weak": float(weak_pf_ap50) if pd.notna(weak_pf_ap50) else np.nan,
            "strong": float(strong_pf_ap50) if pd.notna(strong_pf_ap50) else np.nan,
        },
        {
            "label": "Global COCO mAP@[.5:.95]\n(avg over 10 IoU thresholds)",
            "weak": float(summary.get("edge_ap_coco", np.nan)),
            "strong": float(summary.get("cloud_ap_coco", np.nan)),
        },
        {
            "label": "Per-frame COCO mAP@[.5:.95]\n(10 IoU thresholds, avg over frames)",
            "weak": float(weak_pf_coco) if pd.notna(weak_pf_coco) else np.nan,
            "strong": float(strong_pf_coco) if pd.notna(strong_pf_coco) else np.nan,
        },
    ]
    return [row for row in rows if pd.notna(row["weak"]) and pd.notna(row["strong"])]


def _plot_dataset_gap_ax(ax, rows: List[dict], summary: dict, title: str,
                         show_legend: bool = True) -> None:
    if not rows:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    y = np.arange(len(rows))
    weak_vals = np.array([row["weak"] for row in rows], dtype=float)
    strong_vals = np.array([row["strong"] for row in rows], dtype=float)
    x_min = float(min(weak_vals.min(), strong_vals.min()))
    x_max = float(max(weak_vals.max(), strong_vals.max()))
    span = max(x_max - x_min, 0.04)
    ax.set_xlim(max(0.0, x_min - 0.16 * span), min(1.0, x_max + 0.28 * span))

    for idx, row in enumerate(rows):
        weak = row["weak"]
        strong = row["strong"]
        delta = strong - weak
        ax.plot([weak, strong], [idx, idx], color="#B0B7C3", lw=3.0,
                solid_capstyle="round", zorder=1)
        ax.scatter(weak, idx, s=82, color=_REFERENCE_COLORS["weak_model"],
                   edgecolor="white", linewidth=0.8, zorder=2)
        ax.scatter(strong, idx, s=82, color=_REFERENCE_COLORS["strong_model"],
                   edgecolor="white", linewidth=0.8, zorder=2)
        ax.text(weak - 0.010, idx - 0.16, f"{weak:.3f}",
                ha="right", va="center", fontsize=8.3,
                color=_REFERENCE_COLORS["weak_model"])
        ax.text(strong + 0.010, idx - 0.16, f"{strong:.3f}",
                ha="left", va="center", fontsize=8.3,
                color=_REFERENCE_COLORS["strong_model"])
        ax.text(strong + 0.010, idx + 0.16, f"Δ={delta:+.3f}",
                ha="left", va="center", fontsize=8.3, color="#555555")

    import matplotlib.lines as mlines

    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows])
    ax.invert_yaxis()
    ax.margins(y=0.14)
    ax.set_ylabel("Average Precision")
    ax.set_xlabel("Average Precision")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, ls="--")

    if show_legend:
        handles = [
            mlines.Line2D([], [], color=_REFERENCE_COLORS["weak_model"], marker="o",
                          linestyle="None", markersize=7,
                          label=_detector_display_name("weak_model", summary.get("edge_model", ""))),
            mlines.Line2D([], [], color=_REFERENCE_COLORS["strong_model"], marker="o",
                          linestyle="None", markersize=7,
                          label=_detector_display_name("strong_model", summary.get("cloud_model", ""))),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False)


def _plot_estimator_reference_axis(ax, df: pd.DataFrame, value_col: str,
                                   xlabel: str, title: str,
                                   weak_reference: Optional[float],
                                   strong_reference: Optional[float],
                                   reference_unit: str,
                                   weak_model_name: str = "",
                                   strong_model_name: str = "",
                                   relation_descriptor: str = "smaller") -> None:
    df_plot = _real_estimator_rows(df, value_col).sort_values(value_col, ascending=True)
    if df_plot.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    y = np.arange(len(df_plot))
    values = df_plot[value_col].astype(float).to_numpy()
    colors = [
        _STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
        for name in df_plot["estimator"]
    ]

    ax.hlines(y, 0, values, color="#D7DEE5", lw=2.5, zorder=1)
    ax.scatter(values, y, s=64, c=colors, edgecolor="white",
               linewidth=0.9, zorder=2)

    xmax = max(float(values.max()), 1e-6)
    xpad = max(xmax * 0.42, 0.12)
    ax.set_xlim(0, xmax + xpad)
    ax.set_yticks(y)
    ax.set_yticklabels([_compact_estimator_label(name) for name in df_plot["estimator"]])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.set_axisbelow(True)

    label_pad = max(xmax * 0.02, 0.02)
    for idx, value in enumerate(values):
        label = _format_value(value)
        extra = _relative_annotation(value, weak_reference, relation_descriptor)
        ax.text(value + label_pad, y[idx], f"{label}{extra}",
                va="center", ha="left", fontsize=8.5)

    callout = _reference_callout_text(
        weak_reference, strong_reference, float(values.max()), reference_unit,
        weak_name=_detector_display_name("weak_model", weak_model_name),
        strong_name=_detector_display_name("strong_model", strong_model_name),
    )
    ax.text(1.02, 0.98, callout, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D0D0", alpha=0.95))
    _add_stage_legend(ax)


def _plot_compact_complexity_axis(ax, df: pd.DataFrame, value_col: str,
                                  xlabel: str, title: str,
                                  weak_reference: Optional[float],
                                  strong_reference: Optional[float],
                                  unit: str) -> None:
    df_plot = _real_estimator_rows(df, value_col).sort_values(value_col, ascending=True)
    if df_plot.empty:
        ax.set_title(f"{title} (no data)")
        ax.set_axis_off()
        return

    y = np.arange(len(df_plot))
    values = df_plot[value_col].astype(float).to_numpy()
    colors = [
        _STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
        for name in df_plot["estimator"]
    ]
    labels = [_compact_estimator_label(name) for name in df_plot["estimator"]]

    ax.hlines(y, 0, values, color="#D7DEE5", lw=2.2, zorder=1)
    ax.scatter(values, y, s=58, c=colors, edgecolor="white",
               linewidth=0.9, zorder=2)

    xmax = max(float(values.max()), 1e-6)
    xpad = max(xmax * 0.18, 0.05)
    ax.set_xlim(0, xmax + xpad)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.24, ls="--")
    ax.set_axisbelow(True)

    label_pad = max(xmax * 0.015, 0.012)
    for idx, value in enumerate(values):
        ax.text(value + label_pad, y[idx], _format_value(value),
                va="center", ha="left", fontsize=8.3)

    max_estimator = float(values.max())
    summary_lines = []
    if weak_reference is not None:
        summary_lines.append(f"Weak detector: {_format_value(weak_reference)} {unit}")
    if strong_reference is not None:
        summary_lines.append(f"Strong detector: {_format_value(strong_reference)} {unit}")
    if weak_reference is not None and max_estimator > 0:
        summary_lines.append(
            f"Largest estimator = {100.0 * max_estimator / weak_reference:.1f}% of weak"
        )
    if summary_lines:
        ax.text(0.98, 0.06, "\n".join(summary_lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.7,
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec="#D0D0D0", alpha=0.95))


def _export_metric_ranking_csv(df_plot: pd.DataFrame, metric: str,
                               csv_path: Path) -> None:
    export = pd.DataFrame({
        "rank": np.arange(1, len(df_plot) + 1),
        "estimator": df_plot["estimator"].values,
        "stage": df_plot.get("stage", pd.Series([
            _stage_from_name(name) for name in df_plot["estimator"]
        ])).values,
        "status": df_plot.get("status", pd.Series(["PASS"] * len(df_plot))).values,
        metric: df_plot[metric].astype(float).values,
    })
    export.to_csv(csv_path, index=False)


def plot_metric_comparison(df: pd.DataFrame, metric: str, out_dir: Path,
                           title: str = None, csv_path: Path = None) -> Path:
    """Compact lollipop ranking for estimator quality metrics."""
    plt = _setup_matplotlib()

    col = metric
    path = out_dir / f"comparison_{metric}.png"
    if col not in df.columns or df[col].dropna().empty:
        return path

    df_plot = _real_estimator_rows(df, col).sort_values(col, ascending=False)
    if df_plot.empty:
        return path

    if csv_path is not None:
        _export_metric_ranking_csv(df_plot, metric, csv_path)

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, len(df_plot) * 0.55)))
    y = np.arange(len(df_plot))
    values = df_plot[col].astype(float).to_numpy()
    colors = [
        _STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
        for name in df_plot["estimator"]
    ]

    ax.hlines(y, 0, values, color="#D8DEE8", lw=2.2, zorder=1)
    ax.scatter(values, y, c=colors, s=70, edgecolor="white",
               linewidth=0.9, zorder=2)
    ax.axvline(0, color="#666666", lw=0.8, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([_compact_estimator_label(name) for name in df_plot["estimator"]])
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Estimator Comparison: {metric.upper()}")
    ax.grid(axis="x", alpha=0.25, ls="--")

    lo = min(0.0, float(np.nanmin(values)))
    hi = max(0.0, float(np.nanmax(values)))
    span = max(hi - lo, 0.1)
    ax.set_xlim(lo - 0.05 * span, hi + 0.18 * span)

    for idx, value in enumerate(values):
        offset = 0.015 * span
        ax.text(value + (offset if value >= 0 else -offset), y[idx],
                f"{value:.4f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8.5)

    _add_stage_legend(ax)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_timing(df: pd.DataFrame, out_dir: Path,
                dataset_label: str = "",
                edge_model: str = "", cloud_model: str = "") -> Path:
    """Lollipop chart for estimator latency with detector references."""
    plt = _setup_matplotlib()

    col = "inference_time_ms"
    path = out_dir / "timing.png"
    if col not in df.columns or df[col].dropna().empty:
        return path

    weak_ref = _detector_reference_value(df, "weak_model", "detector_time_ms")
    strong_ref = _detector_reference_value(df, "strong_model", "detector_time_ms")

    fig, ax = plt.subplots(figsize=(11.5, max(3.5, len(_real_estimator_rows(df, col)) * 0.75)))
    title = "Estimator Latency vs Detector References"
    if dataset_label:
        title += f" [{dataset_label}]"

    _plot_estimator_reference_axis(
        ax, df, col,
        xlabel="Inference time (ms / sample)",
        title=title,
        weak_reference=weak_ref,
        strong_reference=strong_ref,
        reference_unit="ms / sample",
        weak_model_name=edge_model,
        strong_model_name=cloud_model,
        relation_descriptor="faster",
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_complexity(df: pd.DataFrame, out_dir: Path,
                    dataset_label: str = "",
                    dataset_summary: dict = None) -> Path:
    """Shared-row estimator efficiency figure with timing merged in."""
    plt = _setup_matplotlib()

    path = out_dir / "complexity.png"
    panels = _complexity_panel_specs(df)
    if not panels:
        return path

    df_plot = _complexity_plot_rows(df, panels)
    if df_plot.empty:
        return path

    fig, axes_grid = plt.subplots(
        1, len(panels),
        figsize=(5.2 * len(panels) + 5.0, max(5.0, len(df_plot) * 0.95)),
        squeeze=False,
        sharey=True,
    )
    axes = list(axes_grid[0])

    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        weak_ref = _detector_reference_value(df, "weak_model", panel["reference_col"])
        strong_ref = _detector_reference_value(df, "strong_model", panel["reference_col"])
        _plot_complexity_metric_axis(
            ax, df_plot, panel["value_col"],
            xlabel=panel["xlabel"],
            title=panel["title"],
            weak_reference=weak_ref,
            strong_reference=strong_ref,
            unit=panel["unit"],
            show_ylabels=(idx == 0),
        )

    fig.legend(handles=_complexity_legend_handles(dataset_summary),
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=4, frameon=False)
    suptitle = "Estimator Complexity and Latency"
    if dataset_label:
        suptitle += f" [{dataset_label}]"
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.text(0.5, 0.93,
             "Estimator rows are shared across metrics; detector references are the vertical lines.",
             ha="center", va="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.14,
                        top=0.84, wspace=0.18)
    plt.savefig(path)
    plt.close()
    return path


def _plot_one_offloading_ax(ax, offload_df: pd.DataFrame,
                            metrics_df: pd.DataFrame,
                            map_col: str, title: str, ylabel: str,
                            estimator_styles: dict = None,
                            headline_names: list = None,
                            show_legend: bool = True,
                            x_col: str = "ratio",
                            x_label: str = "Offload Ratio"):
    """Plot offloading curves on a single axis for a given mAP column.

    When *headline_names* is provided, headline estimators are drawn with
    full weight (lw=2, markers, legend entry) while non-headline estimators
    are drawn faded (lw=0.7, alpha=0.2, no markers, no legend entry).
    """
    import matplotlib.pyplot as plt

    if map_col not in offload_df.columns or x_col not in offload_df.columns:
        ax.set_title(f"{title} (no data)")
        return

    if estimator_styles is None:
        estimator_styles = _build_estimator_styles(offload_df)

    headline_set = set(headline_names) if headline_names else None

    groups = dict(list(offload_df.groupby("estimator")))

    oracle_groups = {n: g for n, g in groups.items()
                     if _is_oracle_estimator(n)}
    non_oracle_groups = {n: g for n, g in groups.items()
                         if not _is_oracle_estimator(n)}

    oracle_data = None
    if oracle_groups:
        merged = pd.concat(oracle_groups.values())
        if map_col in merged.columns and x_col in merged.columns:
            oracle_data = (merged.groupby(x_col, as_index=False)
                          .agg({map_col: "max"})
                          .sort_values(x_col))

    edge_ref = None
    cloud_ref = None
    any_pred = oracle_data if oracle_data is not None else next(iter(non_oracle_groups.values()), None)
    if any_pred is not None and map_col in any_pred.columns and x_col in any_pred.columns:
        x_values = any_pred[x_col].astype(float)
        r0 = any_pred.loc[np.isclose(x_values, 0.0), map_col]
        r1 = any_pred.loc[np.isclose(x_values, 1.0), map_col]
        if not r0.empty:
            edge_ref = float(r0.iloc[0])
        if not r1.empty:
            cloud_ref = float(r1.iloc[0])

    if edge_ref is None:
        peak_col = (f"peak_{map_col.replace('mAP', 'map')}"
                    if map_col != "mAP" else "peak_map")
        if peak_col not in metrics_df.columns:
            peak_col = "peak_map"
        weak = metrics_df.loc[metrics_df["estimator"] == "weak_model", peak_col]
        edge_ref = float(weak.iloc[0]) if not weak.empty else 0
    if cloud_ref is None:
        peak_col = (f"peak_{map_col.replace('mAP', 'map')}"
                    if map_col != "mAP" else "peak_map")
        if peak_col not in metrics_df.columns:
            peak_col = "peak_map"
        strong = metrics_df.loc[metrics_df["estimator"] == "strong_model", peak_col]
        if not strong.empty:
            cloud_ref = float(strong.iloc[0])

    ax.axhline(y=edge_ref, color="black", linestyle="--", lw=1.5,
               label=f"Edge only ({edge_ref:.4f})")
    if cloud_ref is not None:
        ax.axhline(y=cloud_ref, color="blue",
                   linestyle="-.", lw=1.5, label=f"Cloud only ({cloud_ref:.4f})")

    items = []
    for name, g in non_oracle_groups.items():
        if map_col not in g.columns or x_col not in g.columns:
            continue
        g = g[np.isfinite(pd.to_numeric(g[x_col], errors="coerce"))].copy()
        if g.empty:
            continue
        g[x_col] = g[x_col].astype(float)
        g = (g.sort_values(x_col)
              .groupby(x_col, as_index=False)
              .agg({map_col: "max"}))
        auc = np.trapezoid(g[map_col], g[x_col]) if len(g) >= 2 else 0.0
        items.append((name, g[x_col].values, g[map_col].values, auc))
    items.sort(key=lambda x: x[3], reverse=True)

    # Assign ranks among headline estimators (sorted desc by AUC)
    headline_items = [it for it in items
                      if headline_set is None or it[0] in headline_set]
    rank_map = {it[0]: rank for rank, it in enumerate(headline_items, 1)}

    for name, ratios, maps, auc in items:
        s = estimator_styles.get(name, {"color": "gray", "marker": "o",
                                        "linestyle": "-"})
        is_headline = headline_set is None or name in headline_set
        if is_headline:
            rank = rank_map.get(name, 0)
            lbl = _compact_estimator_label(name)
            rank_tag = f"#{rank} " if rank else ""
            ax.plot(ratios, maps, marker=s["marker"], linestyle=s["linestyle"],
                    lw=2, label=f"{rank_tag}{lbl} (AUC={auc:.4f})",
                    color=s["color"], markersize=5, alpha=0.85)
        else:
            ax.plot(ratios, maps, linestyle=s["linestyle"],
                    lw=0.7, color=s["color"], alpha=0.2)

    if oracle_data is not None and map_col in oracle_data.columns:
        og = oracle_data.sort_values(x_col)
        auc = np.trapezoid(og[map_col].values, og[x_col].values) if len(og) >= 2 else 0.0
        ax.plot(og[x_col].values, og[map_col].values, marker="*", lw=2.5,
                ls="-", label=f"oracle (AUC={auc:.4f})",
                color="gold", ms=10, zorder=10)

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    if show_legend:
        ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, ls="--")


def plot_offloading_3panel(offload_df: pd.DataFrame,
                           metrics_df: pd.DataFrame,
                           out_dir: Path,
                           dataset_label: str = "",
                           offload_summary_df: pd.DataFrame = None,
                           actual_df: pd.DataFrame = None) -> Path:
    """Dual-view offloading chart: exact forced ratios and realized ratios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("mAP", "11-point mAP@0.5", "mAP@0.5 (11-pt)"),
        ("mAP_coco50", "COCO AP@0.5 (all-point)", "AP@0.5 (COCO)"),
        ("mAP_coco", "COCO mAP@[.5:.95]", "mAP@[.5:.95]"),
    ]
    panels = [(c, t, y) for c, t, y in panels if c in offload_df.columns]
    n = len(panels)
    if n == 0:
        return out_dir / "estimator_offload_ratio_sweeping_map.png"

    # Collapse scenario variants for cleaner chart
    plot_df = _collapse_scenario_variants(offload_df)
    actual_plot_df = _collapse_scenario_variants(actual_df) if actual_df is not None and not actual_df.empty else pd.DataFrame()
    plot_metrics = _collapse_scenario_variants(metrics_df)

    # Determine headline estimators for emphasis
    headline_names = None
    if offload_summary_df is not None and not offload_summary_df.empty:
        collapsed_summary = _collapse_scenario_variants(offload_summary_df)
        headline_names = _headline_estimators(collapsed_summary, limit=8)
    style_df = plot_df if actual_plot_df.empty else pd.concat([plot_df, actual_plot_df], ignore_index=True, sort=False)
    styles = _build_estimator_styles(style_df)

    n_rows = 2 if not actual_plot_df.empty else 1
    fig, axes = plt.subplots(n_rows, n, figsize=(7.5 * n, 5.6 * n_rows), squeeze=False)
    for idx, (col, title, ylabel) in enumerate(panels):
        _plot_one_offloading_ax(axes[0, idx], plot_df, plot_metrics,
                                col, title, ylabel, styles,
                                headline_names=headline_names,
                                show_legend=False,
                                x_col="ratio",
                                x_label="Forced Offload Ratio")
        if n_rows > 1:
            _plot_one_offloading_ax(
                axes[1, idx], actual_plot_df, plot_metrics,
                col, title, ylabel, styles,
                headline_names=headline_names,
                show_legend=False,
                x_col="actual_ratio",
                x_label="Actual Offload Ratio",
            )

    # Single shared legend below all panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.05), ncol=min(4, max(1, len(handles))),
                   fontsize=7.5, frameon=False)

    suptitle = "Offloading Performance by Estimator"
    if dataset_label:
        suptitle += f" [{dataset_label}]"
    fig.suptitle(suptitle, fontsize=14, y=1.01)
    fig.text(0.018, 0.73, "Forced exact-ratio sweep\n(batch top-k ranking)", ha="center", va="center",
             fontsize=9, color="#444444", rotation=90)
    if n_rows > 1:
        fig.text(0.018, 0.29, "Realized-ratio deployment\n(actual achieved ratios)", ha="center",
                 va="center", fontsize=9, color="#444444", rotation=90)
    fig.subplots_adjust(bottom=0.18, hspace=0.32)
    plt.tight_layout(rect=[0.04, 0.10, 1, 0.95])
    path = out_dir / "estimator_offload_ratio_sweeping_map.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_dataset_wide_map(summary_path: Path, out_dir: Path,
                          metrics_df: pd.DataFrame = None,
                          dataset_label: str = "") -> Path:
    """Gap-focused dumbbell chart for weak vs strong detector AP."""
    plt = _setup_matplotlib()
    path = out_dir / "dataset_map.png"
    if not summary_path.exists():
        return path

    with open(summary_path) as f:
        s = json.load(f)

    rows = _dataset_gap_rows(s, metrics_df)
    if not rows:
        return path

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, 1.35 * len(rows) + 1.6)))
    title = "Weak vs Strong Detector Gap"
    if dataset_label:
        title += f" [{dataset_label}]"
    _plot_dataset_gap_ax(ax, rows, s, title, show_legend=True)
    fig.text(
        0.5, -0.02,
        "AP@0.5 = single IoU threshold (lenient). COCO mAP@[.5:.95] = averaged over 10 IoU thresholds from 0.50 to 0.95 (stricter).\n"
        "Global = one PR curve from all detections. Per-frame = AP computed per image then averaged.\n"
        "\u0394 = gap between weak (edge) and strong (cloud) detector; larger gaps indicate more room for selective offloading.",
        ha="center", va="top", fontsize=8, color="#555555",
        wrap=True,
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def generate_report(metrics_df: pd.DataFrame,
                    dataset_summary: dict = None,
                    dataset_label: str = "") -> str:
    header = f"ESTIMATOR EVALUATION REPORT"
    if dataset_label:
        header += f" — {dataset_label}"
    lines = [
        "=" * 60,
        header,
        "=" * 60, "",
        f"Total estimators evaluated: {len(metrics_df)}", "",
    ]
    if dataset_summary:
        lines.append("--- Dataset-Wide AP (Global PR Curve) ---")
        lines.append(f"  Edge  AP@0.5: {dataset_summary.get('edge_ap50', 'N/A')}")
        lines.append(f"  Edge  AP@COCO: {dataset_summary.get('edge_ap_coco', 'N/A')}")
        lines.append(f"  Cloud AP@0.5: {dataset_summary.get('cloud_ap50', 'N/A')}")
        lines.append(f"  Cloud AP@COCO: {dataset_summary.get('cloud_ap_coco', 'N/A')}")
        lines.append("")

    # Best metrics (exclude virtual estimators for meaningful comparison)
    real = metrics_df[~metrics_df["estimator"].isin(_VIRTUAL_NAMES)]
    if "spearman_rho" in real.columns:
        valid = real.dropna(subset=["spearman_rho"])
        if not valid.empty:
            best = valid.loc[valid["spearman_rho"].idxmax()]
            lines.append(
                f"Best Spearman rho: {best['estimator']} "
                f"({best['spearman_rho']:.4f})"
            )
    elif "r2" in real.columns:
        # Backward-compatible fallback for legacy metrics files.
        valid = real.dropna(subset=["r2"])
        if not valid.empty:
            best = valid.loc[valid["r2"].idxmax()]
            lines.append(f"Best R²: {best['estimator']} ({best['r2']:.4f})")
    if "mae" in real.columns:
        valid = real.dropna(subset=["mae"])
        if not valid.empty:
            best = valid.loc[valid["mae"].idxmin()]
            lines.append(f"Best MAE: {best['estimator']} ({best['mae']:.4f})")
    if "peak_map" in real.columns:
        valid = real.dropna(subset=["peak_map"])
        if not valid.empty:
            best = valid.loc[valid["peak_map"].idxmax()]
            lines.append(f"Best Peak mAP@0.5: {best['estimator']} "
                         f"({best['peak_map']:.4f})")
    if "peak_map_coco" in real.columns:
        valid = real.dropna(subset=["peak_map_coco"])
        if not valid.empty:
            best = valid.loc[valid["peak_map_coco"].idxmax()]
            lines.append(f"Best Peak mAP@COCO: {best['estimator']} "
                         f"({best['peak_map_coco']:.4f})")

    # Complexity overview
    lines += ["", "--- Computational Complexity ---"]
    cols = ["estimator", "inference_time_ms", "gflops", "params"]
    avail = [c for c in cols if c in metrics_df.columns]
    if len(avail) > 1:
        lines.append(metrics_df[avail].to_string(index=False))
    lines.append("")

    # Virtual estimator reference values
    lines.append("--- Virtual Estimators ---")
    for _, row in metrics_df[metrics_df["estimator"].isin(_VIRTUAL_NAMES)].iterrows():
        desc = row.get("description", "")
        spearman = row.get("spearman_rho", np.nan)
        if not np.isnan(spearman):
            corr_s = f"Spearman={spearman:.4f}"
        else:
            # Backward-compatible fallback for legacy metrics files.
            r2 = row.get("r2", np.nan)
            corr_s = f"R²={r2:.4f}" if not np.isnan(r2) else "Spearman=N/A"
        peak = row.get("peak_map", np.nan)
        peak_s = f"Peak mAP={peak:.4f}" if not np.isnan(peak) else ""
        lines.append(f"  {row['estimator']:<15} {desc:<30} {corr_s}  {peak_s}")

    lines += ["", "-" * 60, "", metrics_df.to_string(index=False)]
    return "\n".join(lines)


def plot_ratio_accuracy(thresh_df: pd.DataFrame, out_dir: Path,
                        dataset_label: str = "",
                        offload_summary_df: pd.DataFrame = None) -> Path:
    """Target vs actual offloading ratio with explicit native/calibrated semantics."""
    thresh_df = _with_chart_strategy(thresh_df)
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    path = out_dir / "ratio_accuracy.png"
    df = thresh_df[thresh_df["chart_strategy"].isin(
        ("threshold", "calibrated", "fixed", "sequential_csr",
         "sequential_csr_utility",
         "online_sqt", "online_lvq")
    )].copy()
    if df.empty:
        return path

    # Collapse scenario variants for cleaner chart
    df = _collapse_scenario_variants(df)
    if "chart_strategy" not in df.columns:
        df["chart_strategy"] = df["strategy"].astype(str).map(_canonical_strategy_name)

    # Compute native-threshold MAE per approach to rank ratio control quality.
    approach_mae = {}
    for name in df["estimator"].unique():
        edf = df[df["estimator"] == name]
        if "target_ratio" in edf.columns and "actual_ratio" in edf.columns:
            valid = edf.dropna(subset=["target_ratio", "actual_ratio"])
            if not valid.empty:
                mae = float(np.mean(np.abs(
                    valid["target_ratio"].astype(float)
                    - valid["actual_ratio"].astype(float)
                )))
                approach_mae[name] = mae

    # Sort by MAE (best ratio controllers first)
    names = sorted(
        df["estimator"].unique(),
        key=lambda name: approach_mae.get(name, 1.0),
    )
    mae_rank = {name: rank for rank, name in enumerate(names, 1)}
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    color_map = {name: colors[idx % len(colors)] for idx, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(12.0, 7.4))
    ax.plot([0, 1], [0, 1], color="#444444", ls=":", lw=1.2, alpha=0.8)

    rendered_names = []
    for name in names:
        color = color_map[name]
        strategy_name = _canonical_strategy_name(
            df.loc[df["estimator"] == name, "chart_strategy"].astype(str).iloc[0]
        )
        rendered = False

        gt = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "threshold")].sort_values("target_ratio")
        gc = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "calibrated")].sort_values("target_ratio")
        gs = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "sequential_csr")].sort_values("target_ratio")
        gu = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "sequential_csr_utility")].sort_values("target_ratio")
        gq = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "online_sqt")].sort_values("target_ratio")
        gl = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "online_lvq")].sort_values("target_ratio")
        gf = df[(df["estimator"] == name) &
                (df["chart_strategy"] == "fixed")]

        if strategy_name == "fixed" and not gf.empty:
            actual_ratio = float(gf["actual_ratio"].iloc[0])
            ax.scatter([actual_ratio], [actual_ratio], color=color, marker="D",
                       s=70, edgecolor="white", linewidth=1.0, zorder=5)
            ax.annotate("DCSB @ 0.5", xy=(actual_ratio, actual_ratio),
                        xytext=(8, -10), textcoords="offset points",
                        fontsize=7.2, color=color,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec="#D7D7D7", alpha=0.88))
            rendered = True
        else:
            if not gt.empty:
                ax.plot(gt["target_ratio"], gt["actual_ratio"],
                        marker="o", color=color, ls="-", lw=1.8,
                        markersize=4.8, alpha=0.95)
                rendered = True
            if not gc.empty:
                ax.plot(gc["target_ratio"], gc["actual_ratio"],
                        marker="s", color=color, ls="--", lw=1.5,
                        markersize=4.0, alpha=0.8)
                rendered = True
            if not gs.empty:
                ax.plot(gs["target_ratio"], gs["actual_ratio"],
                        marker="^", color=color, ls="-.", lw=1.4,
                        markersize=4.2, alpha=0.85)
                rendered = True
            if not gu.empty:
                ax.plot(gu["target_ratio"], gu["actual_ratio"],
                        marker="v", color=color, ls=(0, (5, 2)), lw=1.4,
                        markersize=4.4, alpha=0.88)
                rendered = True
            if not gq.empty:
                ax.plot(gq["target_ratio"], gq["actual_ratio"],
                        marker="<", color=color, ls=(0, (1, 1)), lw=1.35,
                        markersize=4.2, alpha=0.88)
                rendered = True
            if not gl.empty:
                ax.plot(gl["target_ratio"], gl["actual_ratio"],
                        marker=">", color=color, ls=(0, (6, 1)), lw=1.35,
                        markersize=4.2, alpha=0.88)
                rendered = True

        if rendered:
            rendered_names.append(name)

    ax.set_xlabel("Target Offload Ratio")
    ax.set_ylabel("Actual Offload Ratio")
    ax.set_title(
        _figure_title(
            "Ratio Control Accuracy",
            "Closer to the diagonal means the policy hits the requested budget more reliably.",
            dataset_label=dataset_label,
        )
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, ls="--")

    estimator_handles = []
    for name in rendered_names:
        rank = mae_rank.get(name, 0)
        mae_val = approach_mae.get(name)
        strategy_name = _canonical_strategy_name(
            df.loc[df["estimator"] == name, "chart_strategy"].astype(str).iloc[0]
        )
        lbl = _compact_approach_label(name, strategy_name)
        mae_str = f"MAE={mae_val:.3f}" if mae_val is not None else ""
        rank_tag = f"#{rank} " if rank else ""
        legend_label = f"{rank_tag}{lbl} ({mae_str})" if mae_str else f"{rank_tag}{lbl}"
        estimator_handles.append(
            mlines.Line2D([], [], color=color_map[name], lw=2.0, label=legend_label)
        )
    mode_handles = [
        mlines.Line2D([], [], color="#444444", marker="o", ls="-", lw=1.8,
                      markersize=5, label="Native threshold"),
        mlines.Line2D([], [], color="#444444", marker="s", ls="--", lw=1.5,
                      markersize=4.2, label="ECDF-calibrated"),
        mlines.Line2D([], [], color="#444444", marker="^", ls="-.", lw=1.4,
                      markersize=4.2, label="Sequential CSR"),
        mlines.Line2D([], [], color="#444444", marker="v", ls=(0, (5, 2)), lw=1.4,
                      markersize=4.2, label="CSR + utility"),
        mlines.Line2D([], [], color="#444444", marker="<", ls=(0, (1, 1)), lw=1.4,
                      markersize=4.2, label="Online SQT"),
        mlines.Line2D([], [], color="#444444", marker=">", ls=(0, (6, 1)), lw=1.4,
                      markersize=4.2, label="Online LVQ"),
        mlines.Line2D([], [], color="#444444", marker="D", ls="None",
                      markersize=6.5, label="Fixed classifier"),
        mlines.Line2D([], [], color="#444444", ls=":", lw=1.2,
                      label="Perfect control"),
    ]

    estimator_legend = fig.legend(
        handles=estimator_handles, title="Approaches",
        loc="upper left", bbox_to_anchor=(0.72, 0.97),
        bbox_transform=fig.transFigure,
        frameon=False, fontsize=7.4, title_fontsize=8.2,
    )
    fig.add_artist(estimator_legend)
    fig.legend(
        handles=mode_handles, title="Decision Mode",
        loc="upper left", bbox_to_anchor=(0.72, 0.52),
        bbox_transform=fig.transFigure,
        frameon=False, fontsize=7.4, title_fontsize=8.2,
    )

    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.11, top=0.90)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _metric_range(metric_type: str):
    """Return (lo, hi) for the expected output range of a metric type."""
    if metric_type == "moric":
        return 0.0, 1.0
    if metric_type == "moric_plus":
        return -1.0, 1.0
    return None, None  # absolute — no fixed range


def _metric_type_order(metric_type: str) -> int:
    return {"absolute": 0, "moric": 1, "moric_plus": 2}.get(metric_type, 9)


def _robust_axis_bounds(*arrays: Optional[np.ndarray],
                        lower_q: float = 1.0,
                        upper_q: float = 99.0,
                        pad_frac: float = 0.08,
                        hard_bounds: Tuple[float, float] = None) -> Tuple[float, float]:
    valid = [np.asarray(arr, dtype=float).reshape(-1) for arr in arrays
             if arr is not None and len(arr) > 0]
    if not valid:
        if hard_bounds is not None:
            lo, hi = hard_bounds
            return float(lo), float(hi)
        return 0.0, 1.0

    combined = np.concatenate(valid)
    lo = float(np.nanpercentile(combined, lower_q))
    hi = float(np.nanpercentile(combined, upper_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(combined))
        hi = float(np.nanmax(combined))
    if hard_bounds is not None:
        lo = min(lo, hard_bounds[0])
        hi = max(hi, hard_bounds[1])
    if hi <= lo:
        hi = lo + 1.0

    spread = hi - lo
    pad = max(spread * pad_frac, 0.02)
    return lo - pad, hi + pad


def _normalize_to_unit(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _compute_ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(values)
    ecdf = np.arange(1, len(sorted_vals) + 1, dtype=float) / len(sorted_vals)
    return sorted_vals, ecdf


def _load_prediction_entries(npz_path: Path,
                             estimator_allowed: Optional[Callable[[str], bool]] = None) -> List[dict]:
    from scipy.stats import ks_2samp

    from ..offloader import classify_metric

    entries: List[dict] = []
    with np.load(npz_path, allow_pickle=True) as d:
        for key in d.files:
            if not key.startswith("pred_"):
                continue
            if key.startswith("pred_train_"):
                continue

            name = key[5:]
            if estimator_allowed is not None and not estimator_allowed(name):
                continue
            proxy_key = f"proxy_{name}"
            proxy = str(d[proxy_key].item()) if proxy_key in d.files else "gain_11pt"
            preds = np.asarray(d[key], dtype=float).reshape(-1)
            train_preds_key = f"pred_train_{name}"
            train_preds = (
                np.asarray(d[train_preds_key], dtype=float).reshape(-1)
                if train_preds_key in d.files else None
            )
            gt_key = f"y_test_{proxy}"
            gt = (np.asarray(d[gt_key], dtype=float).reshape(-1)
                  if gt_key in d.files else None)
            gt_train_key = f"y_train_{proxy}"
            gt_train = (
                np.asarray(d[gt_train_key], dtype=float).reshape(-1)
                if gt_train_key in d.files else None
            )

            if proxy.startswith("lcer_"):
                metric_type_name = "absolute"
            else:
                metric_type_name = classify_metric(proxy).value
            fixed_lo, fixed_hi = _metric_range(metric_type_name)
            hard_bounds = ((fixed_lo, fixed_hi)
                           if fixed_lo is not None and fixed_hi is not None
                           else None)
            hist_lo, hist_hi = _robust_axis_bounds(preds, gt, hard_bounds=hard_bounds)
            if hard_bounds is not None:
                norm_lo, norm_hi = hard_bounds
            else:
                norm_lo, norm_hi = _robust_axis_bounds(
                    preds, gt, pad_frac=0.0, lower_q=1.0, upper_q=99.0
                )

            pred_x, pred_ecdf = _compute_ecdf(preds)
            gt_x, gt_ecdf = _compute_ecdf(gt) if gt is not None else (np.array([]), np.array([]))
            train_pred_x, train_pred_ecdf = _compute_ecdf(train_preds) if train_preds is not None else (np.array([]), np.array([]))
            pred_norm = _normalize_to_unit(preds, norm_lo, norm_hi)
            gt_norm = (_normalize_to_unit(gt, norm_lo, norm_hi)
                       if gt is not None else None)
            train_pred_norm = (_normalize_to_unit(train_preds, norm_lo, norm_hi)
                               if train_preds is not None else None)
            pred_x_norm, pred_ecdf_norm = _compute_ecdf(pred_norm)
            gt_x_norm, gt_ecdf_norm = (_compute_ecdf(gt_norm)
                                       if gt_norm is not None else (np.array([]), np.array([])))
            train_pred_x_norm, train_pred_ecdf_norm = (
                _compute_ecdf(train_pred_norm)
                if train_pred_norm is not None else (np.array([]), np.array([]))
            )

            ks_uniform = None
            if hard_bounds is not None and len(pred_x_norm) > 0:
                ks_uniform = float(np.max(np.abs(pred_ecdf_norm - pred_x_norm)))
            ks_gt = (float(ks_2samp(pred_norm, gt_norm).statistic)
                     if gt_norm is not None and len(gt_norm) > 0 else None)

            entries.append({
                "name": name,
                "label": _compact_estimator_label(name),
                "title": _compact_estimator_label(name, multiline=True),
                "proxy": proxy,
                "metric_type": metric_type_name,
                "preds": preds,
                "train_preds": train_preds,
                "gt": gt,
                "gt_train": gt_train,
                "hist_bounds": (hist_lo, hist_hi),
                "norm_bounds": (norm_lo, norm_hi),
                "pred_ecdf": (pred_x, pred_ecdf),
                "train_pred_ecdf": (train_pred_x, train_pred_ecdf),
                "gt_ecdf": (gt_x, gt_ecdf),
                "pred_ecdf_norm": (pred_x_norm, pred_ecdf_norm),
                "train_pred_ecdf_norm": (train_pred_x_norm, train_pred_ecdf_norm),
                "gt_ecdf_norm": (gt_x_norm, gt_ecdf_norm),
                "fixed_range": hard_bounds,
                "ks_uniform": ks_uniform,
                "ks_gt": ks_gt,
                "train_test_pred_ks": (
                    float(ks_2samp(train_preds, preds).statistic)
                    if train_preds is not None and len(train_preds) > 0 else None
                ),
            })

    entries.sort(key=lambda item: (_metric_type_order(item["metric_type"]), item["name"]))
    return entries


def plot_proxy_metric_distributions(npz_path: Path, out_dir: Path,
                                  dataset_label: str = "",
                                  estimator_allowed: Optional[Callable[[str], bool]] = None) -> Path:
    """Merged figure: histograms plus ECDF diagnostics per estimator."""
    plt = _setup_matplotlib()
    path = out_dir / "proxy-metric_distributions.png"
    entries = _load_prediction_entries(npz_path, estimator_allowed=estimator_allowed)
    if not entries:
        return path

    # Deduplicate scenario variants: keep only the first per base_model
    seen_models: set = set()
    deduped: list = []
    for entry in entries:
        model_key = _estimator_model_key(entry["name"])
        # Check if this is a scenario variant (name contains scenario suffix)
        parts = entry["name"].split("|")
        is_scenario_var = len(parts) >= 5  # stage|model|metric|offloader|scenario
        if is_scenario_var:
            if model_key in seen_models:
                continue
            seen_models.add(model_key)
        deduped.append(entry)
    entries = deduped

    n = len(entries)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(4.8 * ncols, 10), max(4.2 * nrows, 5)),
        squeeze=False,
    )

    pred_hist_color = "#4C9BE8"
    train_hist_color = "#9FD3FF"
    gt_hist_color = "#E8734C"
    pred_cdf_color = "#24557A"
    train_cdf_color = "#4C9BE8"
    gt_cdf_color = "#A14F2A"
    ideal_cdf_color = "#5FAF5F"

    for idx, entry in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        lo, hi = entry["hist_bounds"]
        bins = np.linspace(lo, hi, 34)
        preds = entry["preds"]
        train_preds = entry["train_preds"]
        gt = entry["gt"]

        if entry["fixed_range"] is not None:
            exp_lo, exp_hi = entry["fixed_range"]
            ax.axvspan(exp_lo, exp_hi, color="#B9E3B9", alpha=0.18, zorder=0)
            ax.axvline(exp_lo, color=ideal_cdf_color, ls="--", lw=0.8, alpha=0.6)
            ax.axvline(exp_hi, color=ideal_cdf_color, ls="--", lw=0.8, alpha=0.6)

        ax.hist(preds, bins=bins, density=True, alpha=0.45, color=pred_hist_color,
                edgecolor="white", linewidth=0.35, zorder=1)
        if train_preds is not None:
            ax.hist(train_preds, bins=bins, density=True, alpha=0.25, color=train_hist_color,
                    edgecolor="white", linewidth=0.35, zorder=1)
        if gt is not None:
            ax.hist(gt, bins=bins, density=True, alpha=0.35, color=gt_hist_color,
                    edgecolor="white", linewidth=0.35, zorder=1)

        ax2 = ax.twinx()
        pred_x, pred_ecdf = entry["pred_ecdf"]
        ax2.plot(pred_x, pred_ecdf, color=pred_cdf_color, lw=1.4, zorder=3)
        if train_preds is not None:
            train_pred_x, train_pred_ecdf = entry["train_pred_ecdf"]
            ax2.plot(train_pred_x, train_pred_ecdf, color=train_cdf_color, lw=1.0, ls="-.", zorder=3)
        if gt is not None:
            gt_x, gt_ecdf = entry["gt_ecdf"]
            ax2.plot(gt_x, gt_ecdf, color=gt_cdf_color, lw=1.1, ls="--", zorder=3)
        if entry["fixed_range"] is not None:
            exp_lo, exp_hi = entry["fixed_range"]
            ref_x = np.linspace(exp_lo, exp_hi, 128)
            ref_y = _normalize_to_unit(ref_x, exp_lo, exp_hi)
            ax2.plot(ref_x, ref_y, color=ideal_cdf_color, lw=1.0, ls=":", zorder=2)

        info_lines = [f"{entry['metric_type']} | μ={preds.mean():.3f} σ={preds.std():.3f}"]
        if entry["ks_gt"] is not None:
            info_lines.append(f"KS(pred,gt)={entry['ks_gt']:.3f}")
        elif entry["ks_uniform"] is not None:
            info_lines.append(f"KS(uniform)={entry['ks_uniform']:.3f}")

        ax.set_xlim(lo, hi)
        ax2.set_ylim(0.0, 1.02)
        ax2.grid(False)
        ax.set_title(entry["title"], fontsize=9.5)
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D7D7D7", alpha=0.9))
        if col == 0:
            ax.set_ylabel("Density")
        if col == ncols - 1:
            ax2.set_ylabel("ECDF")
        else:
            ax2.set_yticklabels([])
        if row == nrows - 1:
            ax.set_xlabel("Prediction value")
        ax.grid(axis="y", alpha=0.18, ls="--")

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        _figure_title(
            "Prediction Distributions and ECDF Alignment",
            "Read left-to-right within a panel: train/test shift, target alignment, and whether scores collapse near one region.",
            dataset_label=dataset_label,
        ),
        fontsize=12.5,
        y=1.05,
    )

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    legend_items = [
        mpatches.Patch(color=train_hist_color, alpha=0.25, label="Estimator output (train set)"),
        mpatches.Patch(color=pred_hist_color, alpha=0.45, label="Estimator output (test set)"),
        mpatches.Patch(color=gt_hist_color, alpha=0.35, label="Proxy-metric target (test set)"),
        mlines.Line2D([], [], color=train_cdf_color, lw=1.0, ls="-.", label="ECDF — estimator (train)"),
        mlines.Line2D([], [], color=pred_cdf_color, lw=1.4, label="ECDF — estimator (test)"),
        mlines.Line2D([], [], color=gt_cdf_color, lw=1.1, ls="--", label="ECDF — proxy-metric target"),
        mlines.Line2D([], [], color=ideal_cdf_color, lw=1.0, ls=":", label="Uniform calibration ref."),
    ]
    fig.legend(handles=legend_items, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(path)
    plt.close()
    return path


def plot_empirical_cdf(npz_path: Path, out_dir: Path,
                       dataset_label: str = "",
                       estimator_allowed: Optional[Callable[[str], bool]] = None) -> Path:
    """Compact normalized ECDF companion summary for all estimators."""
    plt = _setup_matplotlib()
    path = out_dir / "empirical_cdf.png"
    entries = _load_prediction_entries(npz_path, estimator_allowed=estimator_allowed)
    if not entries:
        return path

    n = len(entries)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.0 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )

    pred_color = "#24557A"
    gt_color = "#A14F2A"
    ideal_color = "#5FAF5F"

    for idx, entry in enumerate(entries):
        ax = axes[idx // ncols, idx % ncols]
        pred_x, pred_ecdf = entry["pred_ecdf_norm"]
        gt_x, gt_ecdf = entry["gt_ecdf_norm"]

        ax.plot(pred_x, pred_ecdf, color=pred_color, lw=1.35)
        if len(gt_x) > 0:
            ax.plot(gt_x, gt_ecdf, color=gt_color, lw=1.1, ls="--")
        if entry["fixed_range"] is not None:
            ax.plot([0, 1], [0, 1], color=ideal_color, lw=1.0, ls=":")

        note = [entry["metric_type"]]
        if entry["ks_gt"] is not None:
            note.append(f"KS(gt)={entry['ks_gt']:.3f}")
        if entry["ks_uniform"] is not None:
            note.append(f"KS(u)={entry['ks_uniform']:.3f}")

        ax.set_title(entry["label"], fontsize=9.5)
        ax.text(0.03, 0.04, " | ".join(note), transform=ax.transAxes,
                fontsize=6.5, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D7D7D7", alpha=0.9))
        ax.grid(alpha=0.2, ls="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        if idx % ncols == 0:
            ax.set_ylabel("ECDF")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Normalized prediction")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    title = "Normalized Empirical CDF Summary"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=12.5, y=1.03)

    import matplotlib.lines as mlines

    fig.legend(
        handles=[
            mlines.Line2D([], [], color=pred_color, lw=1.35, label="ECDF — estimator (test)"),
            mlines.Line2D([], [], color=gt_color, lw=1.1, ls="--", label="ECDF — proxy-metric target"),
            mlines.Line2D([], [], color=ideal_color, lw=1.0, ls=":", label="Uniform calibration ref."),
        ],
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(path)
    plt.close()
    return path


def plot_estimator_regression(df: pd.DataFrame, out_dir: Path,
                              dataset_label: str = "") -> Path:
    """Grouped lollipop chart: R² and Spearman ρ per base_model."""
    plt = _setup_matplotlib()
    path = out_dir / "estimator_regression.png"
    if "base_model" not in df.columns:
        return path

    df_ok = df[(df.get("status", "PASS") == "PASS")
               & (~df["estimator"].isin(_VIRTUAL_NAMES))].copy()
    if df_ok.empty:
        return path

    # Pick available metrics
    metrics = []
    for col, label in (("spearman_rho", "Spearman ρ"), ("r2", "R²")):
        if col in df_ok.columns and df_ok[col].notna().any():
            metrics.append((col, label))
    if not metrics:
        return path

    # Best value per base_model
    rows = []
    for bm, grp in df_ok.groupby("base_model"):
        stage = grp["stage"].iloc[0] if "stage" in grp.columns else "other"
        row = {"base_model": bm, "stage": stage}
        for col, _ in metrics:
            vals = grp[col].dropna()
            if not vals.empty:
                row[col] = float(vals.max())
        rows.append(row)
    plot_df = pd.DataFrame(rows).dropna(subset=[m[0] for m in metrics], how="all")
    if plot_df.empty:
        return path
    plot_df = plot_df.sort_values(metrics[0][0], ascending=False)

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics,
                             max(3.0, len(plot_df) * 0.55)), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    y = np.arange(len(plot_df))
    colors = [_STAGE_COLORS.get(s, _STAGE_COLORS["other"])
              for s in plot_df["stage"]]

    for ax, (col, label) in zip(axes, metrics):
        vals = plot_df[col].astype(float).fillna(0).to_numpy()
        ax.hlines(y, 0, vals, color="#D8DEE8", lw=2.2, zorder=1)
        ax.scatter(vals, y, c=colors, s=70, edgecolor="white",
                   linewidth=0.9, zorder=2)
        ax.axvline(0, color="#666666", lw=0.8, alpha=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["base_model"])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        title = f"Estimator {label}"
        if dataset_label:
            title += f" [{dataset_label}]"
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25, ls="--")

        lo = min(0.0, float(np.nanmin(vals)))
        hi = max(0.0, float(np.nanmax(vals)))
        span = max(hi - lo, 0.1)
        ax.set_xlim(lo - 0.05 * span, hi + 0.18 * span)
        for idx, v in enumerate(vals):
            offset = 0.015 * span
            ax.text(v + (offset if v >= 0 else -offset), y[idx],
                    f"{v:.4f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5)

    _add_stage_legend(axes[-1])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_calibration_diagnostics(pred_diag_df: pd.DataFrame, out_dir: Path,
                                 dataset_label: str = "") -> Path:
    path = out_dir / "calibration_diagnostics.png"
    return _prediction_diagnostics_figure(pred_diag_df, path, dataset_label=dataset_label)


def plot_slice_heatmap(slice_df: pd.DataFrame, out_dir: Path,
                       dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "slice_heatmap.png"
    if slice_df.empty or "regret_at_25" not in slice_df.columns:
        return path

    df = slice_df.copy()
    df["slice_name"] = df["slice_name"].astype(str)
    df["slice_value"] = df["slice_value"].astype(str)
    heatmap_df = df.groupby(
        ["estimator", "slice_name", "slice_value"], as_index=False
    )["regret_at_25"].mean()
    ordered_slices = sorted(
        heatmap_df[["slice_name", "slice_value"]].drop_duplicates().itertuples(index=False, name=None),
        key=lambda item: _slice_sort_key(*item),
    )
    pivot = heatmap_df.pivot(
        index="estimator",
        columns=["slice_name", "slice_value"],
        values="regret_at_25",
    )
    if pivot.empty:
        return path

    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(ordered_slices))
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    row_labels = _slice_heatmap_estimator_labels(pivot.index.tolist())

    fig, ax = plt.subplots(
        figsize=(max(12.5, pivot.shape[1] * 0.58), max(6.8, pivot.shape[0] * 0.52))
    )
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(
        [_pretty_slice_value(slice_name, slice_value) for slice_name, slice_value in pivot.columns.tolist()],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Dataset slice")
    ax.set_ylabel("Estimator")

    dataset_suffix = f" on {dataset_label}" if dataset_label else ""
    fig.suptitle(
        f"Slice-wise failure map{dataset_suffix}: mean Regret@25 for each estimator",
        y=0.96,
    )
    fig.text(
        0.5,
        0.915,
        "Rows = estimators. Columns = image subsets grouped by feature quartile or video chunk. Lower/lighter is better.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#3b3b3b",
    )

    boundary_positions: List[float] = []
    group_start = 0
    columns = pivot.columns.tolist()
    while group_start < len(columns):
        slice_name = columns[group_start][0]
        group_end = group_start
        while group_end + 1 < len(columns) and columns[group_end + 1][0] == slice_name:
            group_end += 1
        group_center = (group_start + group_end) / 2
        ax.text(
            group_center,
            1.01,
            _pretty_slice_group_label(slice_name),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#3b3b3b",
        )
        if group_end < len(columns) - 1:
            boundary_positions.append(group_end + 0.5)
        group_start = group_end + 1

    for xpos in boundary_positions:
        ax.axvline(x=xpos, color="#4b5563", linewidth=0.8, alpha=0.25)

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Mean Regret@25 per slice (lower is better)",
    )
    fig.text(
        0.5,
        0.02,
        "Key idea: dark vertical bands mark hard slices; dark horizontal bands mark estimators that are brittle across many slices.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#3b3b3b",
    )

    plt.tight_layout(rect=(0.0, 0.06, 1.0, 0.89))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
