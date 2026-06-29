"""Shared constants, label helpers, and data-prep utilities used across analyse submodules."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import PipelineConfig

# Virtual estimator names (zero-cost baselines)
_VIRTUAL_NAMES = {"weak_model", "strong_model"}
_ORACLE_NAME = "oracle"
_DCSB_KEYS = {"dcsb", "dcsb_paper", "dcsb_original"}
_CHART_BASELINE_KEYS = {
    "oracle",
    "random",
    "dcsb",
    "dcsb_paper",
    "dcsb_original",
    "edgeml",
    "edgeml_paper",
    "edgeml_original",
    "edgeml_oric",
}

# ---- chart helpers ----------------------------------------------------

_STAGE_COLORS = {
    "pre": "#4C9BE8",
    "post": "#E8734C",
    "other": "#7A7A7A",
}
_REFERENCE_COLORS = {
    "weak_model": "#4C9BE8",
    "strong_model": "#E8734C",
}
_FOCUS_TARGET_RATIOS = [0.2, 0.4, 0.6, 0.8]
_SLICE_GROUP_LABELS = {
    "edge_conf_mean": "Edge conf. mean",
    "edge_det_count": "Edge det. count",
    "entropy": "Entropy",
    "frame_quartile": "Frame quartile",
    "img_complexity": "Image complexity",
    "video_name": "Video chunk",
}
_SLICE_GROUP_ORDER = {name: idx for idx, name in enumerate(_SLICE_GROUP_LABELS)}
_SLICE_VALUE_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
_LEGACY_POLICY_LABELS = {
    "native_threshold": "threshold",
    "ecdf_calibrated": "calibrated",
    "fixed_classifier": "fixed",
}
_CANONICAL_POLICY_LABELS = {
    "native_threshold": "threshold",
    "ecdf_calibrated": "calibrated",
    "fixed_classifier": "fixed",
}


def _ensure_policy_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    if "policy_id" not in out.columns and "strategy" in out.columns:
        out["policy_id"] = out["strategy"].astype(str)
    if "offloader_id" not in out.columns:
        if "policy_id" in out.columns:
            out["offloader_id"] = out["policy_id"].astype(str)
        elif "strategy" in out.columns:
            out["offloader_id"] = out["strategy"].astype(str)
    if "strategy" not in out.columns and "policy_id" in out.columns:
        policy = out["policy_id"].astype(str)
        out["strategy"] = policy.map(_LEGACY_POLICY_LABELS).fillna(policy)
    return out


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })
    return plt


def _figure_title(title: str, subtitle: str = "", dataset_label: str = "") -> str:
    parts = [title]
    if dataset_label:
        parts[0] = f"{parts[0]} [{dataset_label}]"
    if subtitle:
        parts.append(subtitle)
    return "\n".join(parts)


def _ranking_auc_column(df: pd.DataFrame) -> Optional[str]:
    """Return preferred AUC column for ranking (COCO first, then AP@0.5)."""
    for col in ("auc_coco", "auc_0_5"):
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _headline_estimators(offload_summary_df: pd.DataFrame, limit: int = 5) -> list[str]:
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    if offload_summary_df.empty or "estimator" not in offload_summary_df.columns:
        return []
    df = offload_summary_df.copy()
    if "strategy" in df.columns:
        threshold_df = df[df["strategy"].astype(str) == "threshold"].copy()
        if not threshold_df.empty:
            df = threshold_df
    baseline_names = {"weak_model", "strong_model", "oracle", "random"}
    baseline_names.update(df.loc[df["base_model"].astype(str).isin(_CHART_BASELINE_KEYS), "estimator"].astype(str).tolist() if "base_model" in df.columns else [])
    learned = df[
        ~df["estimator"].astype(str).isin(baseline_names)
    ].copy()
    ranking_col = _ranking_auc_column(learned)
    if ranking_col is not None:
        learned = learned.sort_values(ranking_col, ascending=False)
    top_learned = learned["estimator"].astype(str).drop_duplicates().head(limit).tolist()
    return list(dict.fromkeys([*sorted(baseline_names), *top_learned]))


def _filter_headline_estimators(df: pd.DataFrame, headline_names: list[str]) -> pd.DataFrame:
    if df.empty or "estimator" not in df.columns or not headline_names:
        return df.copy()
    keep = set(headline_names)
    return df[df["estimator"].astype(str).isin(keep)].copy()


def _stage_from_name(name: str) -> str:
    if "|" not in name:
        return "other"
    stage = name.split("|", 1)[0]
    return stage if stage in ("pre", "post") else "other"


def _estimator_model_key(name: str) -> str:
    if name in _VIRTUAL_NAMES or name == "random":
        return name
    if name == _ORACLE_NAME or name.startswith("oracle"):
        return _ORACLE_NAME

    parts = name.split("|")
    if len(parts) >= 3:
        return parts[1]
    return name


def _chart_estimator_allowlist(cfg: PipelineConfig) -> Tuple[Set[str], Set[str]]:
    configured = list(cfg.enabled_approaches())
    allowed_names = {pcfg.name for pcfg in configured}
    allowed_names.update(_VIRTUAL_NAMES)
    allowed_names.update({"oracle", "random"})

    allowed_keys = {pcfg.registry_key for pcfg in configured}
    allowed_keys.update(_CHART_BASELINE_KEYS)
    return allowed_names, allowed_keys


def _configured_approach_allowlist(cfg: PipelineConfig,
                                   include_virtual: bool = False) -> Tuple[Set[str], Set[str]]:
    configured = list(cfg.enabled_approaches())
    allowed_names = {pcfg.name for pcfg in configured}
    allowed_keys = {pcfg.registry_key for pcfg in configured}
    if include_virtual:
        allowed_names.update(_VIRTUAL_NAMES)
        allowed_keys.update(_VIRTUAL_NAMES)
    return allowed_names, allowed_keys


def _is_chart_estimator_allowed(name: str,
                                allowed_names: Set[str],
                                allowed_keys: Set[str]) -> bool:
    if name in allowed_names:
        return True
    return _estimator_model_key(name) in allowed_keys


def _filter_chart_estimators(df: pd.DataFrame,
                             allowed_names: Set[str],
                             allowed_keys: Set[str]) -> pd.DataFrame:
    if df.empty or "estimator" not in df.columns:
        return df.copy()

    mask = df["estimator"].astype(str).map(
        lambda name: _is_chart_estimator_allowed(name, allowed_names, allowed_keys)
    )
    return df[mask].copy()


def _compact_estimator_label(name: str, multiline: bool = False) -> str:
    if name == "weak_model":
        return "Weak detector"
    if name == "strong_model":
        return "Strong detector"

    parts = name.split("|")
    if len(parts) >= 3:
        model, metric = parts[1], parts[2]
        return f"{model}\n{metric}" if multiline else f"{model} ({metric})"
    return name


def _extract_proxy_metric(name: str) -> str:
    """Extract the proxy-metric label from a pipe-delimited estimator/approach name.

    Convention: ``stage|model|proxy_metric|...``  →  ``parts[2]``.
    Returns ``"n/a"`` for baselines (oracle, weak_model, etc.) or names that
    do not follow the convention.
    """
    parts = name.split("|")
    if len(parts) >= 3:
        return parts[2]
    return "n/a"


def _estimator_name_without_offloader(name: str,
                                      offloader_id: str | None = None,
                                      policy_id: str | None = None) -> str:
    label = str(name)
    if label in _VIRTUAL_NAMES or label == "random" or label == _ORACLE_NAME:
        return label

    parts = label.split("|")
    candidates = [candidate for candidate in (offloader_id, policy_id) if candidate]
    for candidate in candidates:
        candidate = str(candidate)
        if candidate in parts[1:]:
            idx = max(i for i, part in enumerate(parts) if part == candidate)
            return "|".join(parts[:idx])
    return label


def _approach_policy_label(row: pd.Series) -> str:
    for key in ("offloader_id", "policy_id", "strategy"):
        value = row.get(key, "")
        if pd.notna(value) and str(value):
            return _trace_strategy_label(str(value))
    return "n/a"


_TRACE_STRATEGY_LABELS = {
    "calibrated": "Calibrated",
    "ecdf_calibrated": "Calibrated",
    "fixed": "Fixed",
    "fixed_classifier": "Fixed classifier",
    "native_threshold": "Threshold",
    "online_lvq": "Online LVQ",
    "online_sqt": "Online SQT",
    "sequential_csr": "Seq. CSR",
    "sequential_csr_utility": "Seq. CSR util.",
    "threshold": "Threshold",
}


def _trace_strategy_label(name: str) -> str:
    return _TRACE_STRATEGY_LABELS.get(str(name), str(name).replace("_", " ").title())


def _canonical_strategy_name(name: str) -> str:
    return _CANONICAL_POLICY_LABELS.get(str(name), str(name))


def _with_chart_strategy(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_policy_columns(df)
    if out.empty:
        return out
    out = out.copy()
    out["chart_strategy"] = out["strategy"].astype(str).map(_canonical_strategy_name)
    return out


def _nearest_available_ratio(target: float, available_ratios: list[float]) -> float:
    if not available_ratios:
        return float(target)
    target = float(target)
    return min(available_ratios, key=lambda ratio: abs(float(ratio) - target))


def _resolve_trace_plot_ratios(available_ratios: list[float],
                               focus_ratios: list[float] | None = None) -> list[tuple[float, float]]:
    available = sorted(float(ratio) for ratio in available_ratios if float(ratio) > 0.0)
    if not available:
        return []
    if not focus_ratios:
        return [(ratio, ratio) for ratio in available]

    resolved: list[tuple[float, float]] = []
    used_actual: list[float] = []
    for requested_ratio in focus_ratios:
        display_ratio = float(requested_ratio)
        actual_ratio = _nearest_available_ratio(display_ratio, available)
        if any(np.isclose(actual_ratio, seen_ratio) for seen_ratio in used_actual):
            continue
        used_actual.append(actual_ratio)
        resolved.append((display_ratio, actual_ratio))
    return resolved


def _pretty_proxy_metric_label(name: str, multiline: bool = False) -> str:
    label = str(name)
    replacements = (
        ("compressed_weak_moric_plus", "Compressed weak MORIC+"),
        ("scenario_utility", "Scenario utility"),
        ("moric_plus", "MORIC+"),
        ("moric", "MORIC"),
        ("oric", "ORIC"),
        ("lcer", "LCER"),
        ("bwd", "BWD"),
        ("allpoint", "all-point"),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.replace("_", " ").replace("::", " / ")
    if multiline:
        label = label.replace(" / ", "\n")
    return label


def _proxy_metric_family(name: str) -> str:
    metric = str(name)
    if metric.startswith("dataset_"):
        metric = metric[len("dataset_"):]
    if "moric_plus" in metric:
        return "MORIC+"
    if metric.startswith("oric"):
        return "ORIC"
    if metric.startswith("moric"):
        return "MORIC"
    if metric.startswith("lcer"):
        return "LCER"
    if metric.startswith("scenario_utility"):
        return "Scenario utility"
    if metric.startswith("bwd"):
        return "BWD"
    if metric.startswith("gain_"):
        return "Gain"
    return "Other"


def _proxy_family_color(family: str) -> str:
    return {
        "MORIC+": "#2E8B57",
        "MORIC": "#4C9BE8",
        "ORIC": "#7A7A7A",
        "CEORIC": "#9B59B6",
        "BWMORIC": "#F4A261",
        "LCER": "#E8734C",
        "Scenario utility": "#C8553D",
        "BWD": "#3B3B98",
        "Gain": "#2A9D8F",
        "Other": "#7A7A7A",
    }.get(family, "#7A7A7A")


def _normalized_highlight_color(score: Optional[float]) -> tuple[float, float, float, float]:
    import matplotlib

    if score is None or not np.isfinite(score):
        return (0.94, 0.95, 0.97, 1.0)
    rgba = matplotlib.colormaps["RdYlGn"](float(np.clip(score, 0.0, 1.0)))
    blend = 0.78
    return tuple(1.0 - blend * (1.0 - channel) for channel in rgba[:3]) + (1.0,)


def _metric_highlight_score(values: pd.Series, value: float,
                            higher_is_better: bool) -> Optional[float]:
    finite = values.astype(float)
    finite = finite[np.isfinite(finite)]
    if finite.empty or not np.isfinite(value):
        return None
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo < 1e-12:
        return 0.5
    score = (float(value) - lo) / (hi - lo)
    return score if higher_is_better else 1.0 - score


def _render_overview_table(table_df: pd.DataFrame,
                           display_cols: list[tuple[str, str, Optional[bool]]],
                           path: Path,
                           title: str,
                           subtitle: str,
                           footer: str,
                           label_col: str = "estimator") -> Path:
    plt = _setup_matplotlib()
    if table_df.empty or len(display_cols) <= 1:
        return path

    def _format_value(col: str, value: object) -> str:
        if col == label_col:
            return _compact_estimator_label(str(value), multiline=True)
        if col == "stage":
            return str(value).upper()
        if col in ("policy", "proxy_metric"):
            return str(value)
        if not np.isfinite(float(value)) if pd.notna(value) else True:
            return "n/a"
        if col.startswith("latency_"):
            return f"{float(value):.2f}x"
        return f"{float(value):.3f}"

    width_map = {
        label_col: 0.27,
        "stage": 0.08,
        "proxy_metric": 0.14,
        "policy": 0.14,
    }
    col_widths = [width_map.get(col, 0.10) for col, _, _ in display_cols]
    fig_height = max(4.0, 1.15 + 0.5 * len(table_df))
    fig, ax = plt.subplots(figsize=(sum(col_widths) * 14, fig_height))
    ax.axis("off")

    cell_text = []
    for _, row in table_df.iterrows():
        cell_text.append([_format_value(col, row.get(col, np.nan)) for col, _, _ in display_cols])

    table = ax.table(
        cellText=cell_text,
        colLabels=[label for _, label, _ in display_cols],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#E5E7EB")
        cell.set_linewidth(0.7)
        if row_idx == 0:
            cell.set_facecolor("#22313F")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue

        column_key, _, direction = display_cols[col_idx]
        series = table_df[column_key].astype(str) if direction is None else table_df[column_key].astype(float)
        raw_value = table_df.iloc[row_idx - 1][column_key]

        if column_key == "stage":
            stage = str(raw_value)
            cell.set_facecolor(_STAGE_COLORS.get(stage.lower(), "#D1D5DB"))
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        if column_key == label_col:
            stage = _stage_from_name(str(table_df.iloc[row_idx - 1][label_col]))
            cell.set_facecolor(_normalized_highlight_color(0.55 if stage == "pre" else 0.35 if stage == "post" else 0.45))
            cell.get_text().set_ha("left")
            cell.get_text().set_fontweight("bold")
            continue
        if column_key in ("policy", "proxy_metric"):
            cell.set_facecolor("#EEF2F7")
            cell.get_text().set_fontweight("bold")
            continue

        score = None
        if direction is not None and pd.notna(raw_value):
            score = _metric_highlight_score(series, float(raw_value), higher_is_better=direction)
        cell.set_facecolor(_normalized_highlight_color(score))
        if score is not None and (score >= 0.97 or score <= 0.03):
            cell.get_text().set_fontweight("bold")

    fig.suptitle(
        _figure_title(title, subtitle),
        y=0.98,
    )
    fig.text(
        0.5, 0.035, footer,
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.92))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _prediction_diagnostics_figure(pred_diag_df: pd.DataFrame, out_path: Path,
                                   dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    if pred_diag_df.empty or "calibration_gap" not in pred_diag_df.columns:
        return out_path

    base_df = _collapse_scenario_variants(
        pred_diag_df, metric="calibration_gap", higher_is_better=False
    )
    keep = [
        col for col in (
            "estimator", "calibration_gap", "ks_distance", "spread_ratio",
            "near_boundary_rate_pred", "train_test_pred_ks"
        ) if col in base_df.columns
    ]
    plot_df = base_df[keep].groupby("estimator", as_index=False).mean(numeric_only=True)
    if plot_df.empty:
        return out_path

    if "spread_ratio" in plot_df.columns:
        plot_df["spread_mismatch"] = np.abs(
            plot_df["spread_ratio"].astype(float) - 1.0
        )
    metric_cols = [
        col for col in (
            "calibration_gap", "ks_distance", "spread_mismatch",
            "near_boundary_rate_pred", "train_test_pred_ks"
        ) if col in plot_df.columns
    ]
    if not metric_cols:
        return out_path

    rank_frame = plot_df[metric_cols].rank(method="average", pct=True, ascending=True)
    plot_df["diagnostic_score"] = rank_frame.mean(axis=1)
    plot_df = plot_df.sort_values("diagnostic_score", ascending=True).reset_index(drop=True)

    panels = [
        ("calibration_gap", "Calibration gap", "Lower means score levels match the target."),
        ("ks_distance", "KS distance", "Lower means the full score CDF is closer."),
        ("spread_mismatch", "Spread mismatch", "Lower means less collapse or over-stretch."),
        ("near_boundary_rate_pred", "Near-threshold density", "Lower means fewer brittle threshold decisions."),
        ("train_test_pred_ks", "Train/test drift", "Lower means the score distribution shifts less."),
    ]
    panels = [panel for panel in panels if panel[0] in plot_df.columns]
    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.2 * n_cols, max(5.4, 0.34 * len(plot_df) * n_rows + 1.6)),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    y = np.arange(len(plot_df))
    labels = [_compact_estimator_label(name) for name in plot_df["estimator"]]
    colors = [
        _STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
        for name in plot_df["estimator"]
    ]

    for ax, (metric_col, title, subtitle) in zip(axes_flat, panels):
        values = plot_df[metric_col].astype(float).to_numpy()
        ax.barh(y, values, color=colors, edgecolor="white")
        ax.set_title(f"{title}\n{subtitle}")
        ax.grid(axis="x", alpha=0.25, ls="--")
        ax.set_yticks(y)
        if ax is axes_flat[0]:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticklabels([])

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        _figure_title(
            "Prediction Diagnostics",
            "Merged view: calibration, distribution mismatch, spread collapse, boundary crowding, and train/test drift.\n"
            "All panels are error diagnostics, so lower is better across the board.",
            dataset_label=dataset_label,
        ),
        y=1.03,
    )
    fig.text(
        0.5, 0.01,
        "This replaces the old split between calibration_diagnostics and prediction_quality so the same estimator ordering can be read across all score-space failure modes.",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _format_value(value: float, decimals: int = 3) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.{decimals}f}"
    return f"{value:.4g}"


def _format_multiplier(multiplier: float) -> str:
    if multiplier >= 1000:
        return f"{multiplier / 1000:.1f}k"
    if multiplier >= 100:
        return f"{multiplier:.0f}"
    if multiplier >= 10:
        return f"{multiplier:.1f}"
    return f"{multiplier:.2f}"


def _relative_annotation(value: float, weak_reference: Optional[float],
                         descriptor: str) -> str:
    if weak_reference is None or weak_reference <= 0 or value <= 0:
        return ""
    ratio = weak_reference / value
    return f" | {_format_multiplier(ratio)}x {descriptor} vs weak"


def _add_stage_legend(ax):
    import matplotlib.lines as mlines

    handles = []
    for stage in ("pre", "post"):
        handles.append(
            mlines.Line2D([], [], color=_STAGE_COLORS[stage], marker="o",
                          linestyle="None",
                          label="Pre-stage" if stage == "pre" else "Post-stage")
        )
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.02, 0.02),
              borderaxespad=0.0, frameon=False)


def _stage_handles():
    import matplotlib.lines as mlines

    return [
        mlines.Line2D([], [], color=_STAGE_COLORS["pre"], marker="o",
                      linestyle="None", label="Pre-stage"),
        mlines.Line2D([], [], color=_STAGE_COLORS["post"], marker="o",
                      linestyle="None", label="Post-stage"),
    ]


def _collapse_scenario_variants(df: pd.DataFrame,
                                metric: str = "auc_0_5",
                                higher_is_better: bool = True) -> pd.DataFrame:
    """Keep only the best scenario variant per adaptive base_model.

    For non-scenario charts, the 5 scenario expansions of each adaptive
    estimator clutter the visualisation.  This helper picks the best
    variant (by *metric*) and relabels it to drop the scenario suffix so
    it shows as a single entry.
    """
    if df.empty or "scenario" not in df.columns:
        return df.copy()

    scenario_col = df["scenario"].fillna("").astype(str)
    is_scenario = scenario_col != ""
    if not is_scenario.any():
        return df.copy()

    non_scenario = df[~is_scenario].copy()
    scenario_rows = df[is_scenario].copy()

    bm_col = "base_model" if "base_model" in scenario_rows.columns else None
    group_key = bm_col or "estimator"

    kept: list[pd.DataFrame] = []
    for _key, grp in scenario_rows.groupby(group_key):
        if metric in grp.columns and grp[metric].notna().any():
            best_idx = (grp[metric].astype(float).idxmax() if higher_is_better
                        else grp[metric].astype(float).idxmin())
        else:
            best_idx = grp.index[0]
        row = grp.loc[[best_idx]].copy()
        # Strip scenario suffix from estimator name for cleaner labels
        est = str(row["estimator"].iloc[0])
        parts = est.rsplit("|", 1)
        if len(parts) == 2 and parts[1] in str(row["scenario"].iloc[0]).replace(" ", "_"):
            row["estimator"] = parts[0]
        row["scenario"] = ""
        if "scenario_type" in row.columns:
            row["scenario_type"] = ""
        kept.append(row)

    if kept:
        return pd.concat([non_scenario, *kept], ignore_index=True)
    return non_scenario


def _real_estimator_rows(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    df_plot = df.copy()
    if "status" in df_plot.columns:
        status = df_plot["status"].fillna("").astype(str).str.upper()
        df_plot = df_plot[status != "FAIL"]
    df_plot = df_plot[~df_plot["estimator"].isin(_VIRTUAL_NAMES)]
    return df_plot.dropna(subset=[value_col]).copy()


def _detector_reference_value(df: pd.DataFrame, estimator: str,
                              value_col: str) -> Optional[float]:
    rows = df[df["estimator"] == estimator]
    if rows.empty:
        return None
    value = rows.iloc[0].get(value_col, np.nan)
    if pd.isna(value):
        return None
    return float(value)


def _reference_callout_text(weak_ref: Optional[float],
                            strong_ref: Optional[float],
                            max_estimator: Optional[float],
                            unit: str,
                            weak_name: str = "Weak detector",
                            strong_name: str = "Strong detector") -> str:
    lines = ["Detector refs"]
    for label, value, descriptor in (
        (weak_name, weak_ref, "weak"),
        (strong_name, strong_ref, "strong"),
    ):
        if value is None:
            continue
        lines.append(f"{label}: {_format_value(value)} {unit}")
        if max_estimator is not None and max_estimator > 0 and value > 0:
            ratio = value / max_estimator
            lines.append(f"  {_format_multiplier(ratio)}x above max estimator")
    return "\n".join(lines)


def _detector_display_name(key: str, model_name: str = "") -> str:
    base = "Weak detector" if key == "weak_model" else "Strong detector"
    return f"{base} ({model_name})" if model_name else base


def _pretty_slice_group_label(name: str) -> str:
    return _SLICE_GROUP_LABELS.get(str(name), str(name).replace("_", " "))


def _slice_sort_key(slice_name: str, slice_value: str) -> tuple[int, int, str]:
    value_str = str(slice_value)
    numeric_suffix = "".join(ch for ch in value_str if ch.isdigit())
    numeric_rank = int(numeric_suffix) if numeric_suffix else 10_000
    return (
        _SLICE_GROUP_ORDER.get(str(slice_name), len(_SLICE_GROUP_ORDER)),
        _SLICE_VALUE_ORDER.get(value_str, numeric_rank),
        value_str,
    )


def _pretty_slice_value(slice_name: str, slice_value: str) -> str:
    value_str = str(slice_value)
    if slice_name == "video_name" and value_str.startswith("val_chunk_"):
        return value_str.rsplit("_", 1)[-1]
    return value_str


def _slice_heatmap_estimator_labels(names: List[str]) -> List[str]:
    base_labels = [_compact_estimator_label(name) for name in names]
    duplicate_labels = {label for label, count in Counter(base_labels).items() if count > 1}

    labels: List[str] = []
    for raw_name, base_label in zip(names, base_labels):
        if base_label not in duplicate_labels:
            labels.append(base_label)
            continue

        parts = raw_name.split("|")
        scenario_suffix = ""
        if len(parts) >= 5 and parts[4] and parts[4] != "default":
            scenario_suffix = _pretty_scenario_label(parts[4])
        elif len(parts) >= 4 and parts[3] and parts[3] != "default":
            scenario_suffix = parts[3].replace("_", " ")

        short_base = parts[1] if len(parts) >= 2 else base_label
        labels.append(f"{short_base} [{scenario_suffix or raw_name}]")

    final_labels: List[str] = []
    seen: Counter[str] = Counter()
    for label in labels:
        seen[label] += 1
        final_labels.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return final_labels


def _compact_approach_label(estimator: str, strategy: str) -> str:
    """Compact label for approach = estimator + strategy."""
    est_label = _compact_estimator_label(estimator)
    strategy_short = {
        "native_threshold": "thresh",
        "threshold": "thresh",
        "ecdf_calibrated": "calib",
        "calibrated": "calib",
        "sequential_csr": "seq-csr",
        "sequential_csr_utility": "seq-csr-util",
        "online_sqt": "online-sqt",
        "online_lvq": "online-lvq",
        "fixed_classifier": "fixed",
        "fixed": "fixed",
    }.get(strategy, strategy)
    return f"{est_label} [{strategy_short}]"


def _pretty_scenario_label(name: str) -> str:
    return str(name).replace("__mix__", " + ").replace("_", " ")


def _scenario_adaptive_base_models(df: pd.DataFrame) -> set[str]:
    if df.empty or "scenario" not in df.columns or "base_model" not in df.columns:
        return set()
    mask = df["scenario"].fillna("").astype(str) != ""
    return set(df.loc[mask, "base_model"].astype(str).tolist())


def _resource_reference_value(resource_df: pd.DataFrame, value_col: str,
                              estimator_name: str) -> Optional[float]:
    if resource_df.empty or value_col not in resource_df.columns:
        return None

    estimator_mask = resource_df["estimator"].astype(str) == estimator_name
    if estimator_mask.any():
        values = resource_df.loc[estimator_mask, value_col].astype(float)
        if not values.empty and np.isfinite(values).any():
            return float(values[np.isfinite(values)].median())

    ratio_col = "actual_ratio" if "actual_ratio" in resource_df.columns else "target_ratio"
    if ratio_col not in resource_df.columns:
        values = resource_df[value_col].astype(float)
        return float(values[np.isfinite(values)].min()) if np.isfinite(values).any() else None

    ratios = resource_df[ratio_col].astype(float)
    values = resource_df[value_col].astype(float)
    finite_mask = np.isfinite(ratios) & np.isfinite(values)
    if not finite_mask.any():
        return None

    if estimator_name == "weak_model":
        nearest = finite_mask & (np.abs(ratios) == np.abs(ratios).min())
        subset = values[nearest]
        return float(subset.min()) if not subset.empty else float(values[finite_mask].min())

    nearest = finite_mask & (np.abs(ratios - 1.0) == np.abs(ratios - 1.0).min())
    subset = values[nearest]
    return float(subset.max()) if not subset.empty else float(values[finite_mask].max())


def _build_estimator_styles(offload_df: pd.DataFrame) -> dict:
    """Build a fixed name→style mapping so all subplots use consistent styling."""
    import matplotlib.pyplot as plt

    _MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]
    _LINESTYLES = ["-", "--", "-.", ":"]
    _COLORS = list(plt.cm.tab20.colors)

    names = sorted(n for n in offload_df["estimator"].unique()
                   if n != _ORACLE_NAME and not str(n).startswith("oracle"))
    styles = {}
    for i, name in enumerate(names):
        styles[name] = {
            "color": _COLORS[i % len(_COLORS)],
            "marker": _MARKERS[i % len(_MARKERS)],
            "linestyle": _LINESTYLES[i % len(_LINESTYLES)],
        }
    return styles


def _prepare_scenario_method_comparison(offload_summary_df: pd.DataFrame) -> Optional[dict]:
    if offload_summary_df.empty or "scenario" not in offload_summary_df.columns:
        return None

    df_all = offload_summary_df.copy()
    if "base_model" not in df_all.columns:
        df_all["base_model"] = df_all["estimator"].astype(str).map(_estimator_model_key)

    strategy_order = (
        "sequential_csr_utility",
        "sequential_csr",
        "calibrated",
        "threshold",
    )
    available = set(df_all.get("strategy", pd.Series(dtype=object)).astype(str))
    chosen_strategy = next((name for name in strategy_order if name in available), None)
    if chosen_strategy is None:
        return None

    df_chosen = df_all[df_all["strategy"].astype(str) == chosen_strategy].copy()
    if df_chosen.empty:
        return None

    adaptive_df = df_chosen[
        df_chosen["scenario"].fillna("").astype(str) != ""
    ].copy()
    if adaptive_df.empty:
        return None
    adaptive_base_models = sorted(adaptive_df["base_model"].astype(str).unique().tolist())
    if not adaptive_base_models:
        return None

    scenario_levels = (
        adaptive_df[["scenario", "scenario_type"]]
        .drop_duplicates()
        .assign(
            _scenario_group=lambda x: x["scenario_type"].astype(str).map(
                {"preset": 0, "mix": 1}
            ).fillna(2),
            _scenario_sort=lambda x: x["scenario"].astype(str),
        )
        .sort_values(["_scenario_group", "_scenario_sort"])
    )
    scenario_order = scenario_levels["scenario"].astype(str).tolist()
    if not scenario_order:
        return None

    baseline_df = df_chosen[df_chosen["scenario"].fillna("") == ""].copy()
    baseline_df = baseline_df[
        ~baseline_df["base_model"].astype(str).isin(
            {*adaptive_base_models, "weak_model", "strong_model", "random", "oracle"}
        )
    ].copy()

    aside_candidates = baseline_df[
        (baseline_df["stage"].astype(str) == "pre") &
        (~baseline_df["base_model"].astype(str).isin({"edgeml", *list(_DCSB_KEYS)})) &
        baseline_df["auc_0_5"].notna()
    ].copy()
    if aside_candidates.empty:
        return None
    best_aside_row = aside_candidates.sort_values("auc_0_5", ascending=False).iloc[0]
    best_aside_key = str(best_aside_row["base_model"])

    edge_rows = baseline_df[
        (baseline_df["base_model"].astype(str) == "edgeml") &
        baseline_df["auc_0_5"].notna()
    ].copy()
    if edge_rows.empty:
        return None

    dcsb_rows = df_all[
        (df_all["scenario"].fillna("") == "") &
        (df_all["base_model"].astype(str).isin(_DCSB_KEYS)) &
        (df_all["strategy"].astype(str) == "fixed") &
        df_all["peak_map"].notna()
    ].copy()

    adaptive_auc = (
        adaptive_df.pivot_table(
            index="scenario",
            columns="base_model",
            values="auc_0_5",
            aggfunc="mean",
        ).reindex(scenario_order)
    )
    adaptive_peak = (
        adaptive_df.pivot_table(
            index="scenario",
            columns="base_model",
            values="peak_map",
            aggfunc="mean",
        ).reindex(scenario_order)
    )
    adaptive_labels = (
        adaptive_df[["base_model", "estimator"]]
        .drop_duplicates(subset=["base_model"])
        .assign(label=lambda x: x["estimator"].astype(str).map(_compact_estimator_label))
        .set_index("base_model")["label"]
        .to_dict()
    )

    return {
        "chosen_strategy": chosen_strategy,
        "scenario_order": scenario_order,
        "adaptive_base_models": adaptive_base_models,
        "adaptive_auc": adaptive_auc,
        "adaptive_peak": adaptive_peak,
        "adaptive_labels": adaptive_labels,
        "best_aside_key": best_aside_key,
        "best_aside_label": _compact_estimator_label(str(best_aside_row["estimator"])),
        "best_aside_auc": float(
            baseline_df.loc[baseline_df["base_model"].astype(str) == best_aside_key, "auc_0_5"].mean()
        ),
        "best_aside_peak": float(
            baseline_df.loc[baseline_df["base_model"].astype(str) == best_aside_key, "peak_map"].mean()
        ),
        "edge_auc": float(edge_rows["auc_0_5"].mean()),
        "edge_peak": float(edge_rows["peak_map"].mean()),
        "dcsb_peak": (
            float(dcsb_rows["peak_map"].mean()) if not dcsb_rows.empty else None
        ),
    }
