"""Resource tradeoff, frontier charts, and paper tables."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .analyse_helpers import (
    _DCSB_KEYS,
    _FOCUS_TARGET_RATIOS,
    _ORACLE_NAME,
    _STAGE_COLORS,
    _canonical_strategy_name,
    _collapse_scenario_variants,
    _compact_approach_label,
    _ensure_policy_columns,
    _figure_title,
    _ranking_auc_column,
    _resource_reference_value,
    _setup_matplotlib,
    _stage_from_name,
    _with_chart_strategy,
)


def _resource_tradeoff_focus_estimators(resource_df: pd.DataFrame,
                                        offload_summary_df: pd.DataFrame,
                                        limit: int = 5) -> list[str]:
    """Pick a compact set of approaches for the cross-budget trajectory view."""
    if resource_df.empty or "estimator" not in resource_df.columns:
        return []

    available = resource_df["estimator"].astype(str).drop_duplicates().tolist()
    available_set = set(available)
    excluded = {"weak_model", "strong_model", "oracle", "random"}

    summary_df = _collapse_scenario_variants(_ensure_policy_columns(offload_summary_df))
    ranked_names: list[str] = []
    special_names: list[str] = []
    if not summary_df.empty and "estimator" in summary_df.columns:
        summary_df = summary_df[
            summary_df["estimator"].astype(str).isin(available_set)
        ].copy()
        if not summary_df.empty:
            baseline_models = {"oracle", "random", "weak_model", "strong_model"}
            if "base_model" in summary_df.columns:
                ranked_df = summary_df[
                    ~summary_df["base_model"].astype(str).isin(
                        baseline_models.union(_DCSB_KEYS).union({"edgeml"})
                    )
                ].copy()
                special_mask = summary_df["base_model"].astype(str).isin(
                    _DCSB_KEYS.union({"edgeml"})
                )
                special_names = (
                    summary_df.loc[special_mask, "estimator"]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )
            else:
                ranked_df = summary_df[
                    ~summary_df["estimator"].astype(str).isin(excluded)
                ].copy()

            ranking_col = _ranking_auc_column(ranked_df)
            if ranking_col is not None:
                ranked_df = ranked_df.assign(
                    _auc_sort=ranked_df[ranking_col].astype(float).fillna(-np.inf)
                ).sort_values(["_auc_sort", "estimator"], ascending=[False, True])
            ranked_names = ranked_df["estimator"].astype(str).drop_duplicates().tolist()

    if not ranked_names:
        fallback = resource_df[
            ~resource_df["estimator"].astype(str).isin(excluded)
        ].copy()
        if "mAP" in fallback.columns:
            fallback = fallback.assign(
                _map_sort=fallback["mAP"].astype(float).fillna(-np.inf)
            ).sort_values(["_map_sort", "estimator"], ascending=[False, True])
        ranked_names = fallback["estimator"].astype(str).drop_duplicates().tolist()

    chosen: list[str] = []
    for name in ranked_names:
        if name in available_set and name not in excluded and name not in chosen:
            chosen.append(name)
        if len(chosen) >= limit:
            break
    for name in special_names:
        if name in available_set and name not in excluded and name not in chosen:
            chosen.append(name)
    return chosen


def _pareto_frontier_order(x_vals: np.ndarray, y_vals: np.ndarray,
                           tol: float = 1e-9) -> np.ndarray:
    """Return non-dominated points for minimise-x / maximise-y trade-offs."""
    if x_vals.size == 0 or y_vals.size == 0:
        return np.array([], dtype=int)

    finite = np.isfinite(x_vals) & np.isfinite(y_vals)
    if not finite.any():
        return np.array([], dtype=int)

    finite_indices = np.flatnonzero(finite)
    order = finite_indices[np.lexsort((-y_vals[finite_indices], x_vals[finite_indices]))]
    frontier: list[int] = []
    best_y = -np.inf
    for idx in order:
        value = float(y_vals[idx])
        if value > best_y + tol:
            frontier.append(int(idx))
            best_y = value
    return np.array(frontier, dtype=int)


def _resource_efficiency_matrix(resource_df: pd.DataFrame,
                                estimator_names: list[str],
                                target_ratios: list[float],
                                x_col: str,
                                weak_ref_value: Optional[float],
                                weak_map: Optional[float]) -> np.ndarray:
    """Build gain-density heatmap: (mAP - weak) / normalized resource."""
    matrix = np.full((len(estimator_names), len(target_ratios)), np.nan, dtype=float)
    if weak_ref_value is None or weak_map is None or not np.isfinite(weak_ref_value) or weak_ref_value <= 0:
        return matrix

    for row_idx, estimator_name in enumerate(estimator_names):
        est_df = resource_df[resource_df["estimator"].astype(str) == estimator_name].copy()
        if est_df.empty or x_col not in est_df.columns:
            continue
        est_df["target_ratio"] = est_df["target_ratio"].astype(float)
        for col_idx, ratio in enumerate(target_ratios):
            match = est_df[np.isclose(est_df["target_ratio"], float(ratio), atol=1e-6)].copy()
            if match.empty:
                continue
            row = match.sort_values("mAP", ascending=False).iloc[0]
            cost = float(row.get(x_col, np.nan))
            map_value = float(row.get("mAP", np.nan))
            if not np.isfinite(cost) or not np.isfinite(map_value):
                continue
            normalized_cost = cost / float(weak_ref_value)
            if not np.isfinite(normalized_cost) or normalized_cost <= 0:
                continue
            matrix[row_idx, col_idx] = (map_value - float(weak_map)) / normalized_cost
    return matrix


def _normalize_heatmap_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to [0, 1], centering zero at 0.5 when signs mix."""
    normalized = np.full_like(matrix, np.nan, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return normalized

    min_value = float(finite.min())
    max_value = float(finite.max())
    if min_value < 0.0 < max_value:
        span = max(abs(min_value), abs(max_value))
        if span <= 1e-12:
            normalized[np.isfinite(matrix)] = 0.5
        else:
            normalized[np.isfinite(matrix)] = np.clip(
                0.5 + 0.5 * (matrix[np.isfinite(matrix)] / span),
                0.0,
                1.0,
            )
        return normalized

    if max_value - min_value <= 1e-12:
        normalized[np.isfinite(matrix)] = 0.5
        return normalized

    normalized[np.isfinite(matrix)] = (
        (matrix[np.isfinite(matrix)] - min_value) / (max_value - min_value)
    )
    return normalized


def _prepare_resource_tradeoff_context(resource_df: pd.DataFrame,
                                       offload_summary_df: pd.DataFrame,
                                       focus_limit: int = 5) -> Optional[dict]:
    resource_df = _with_chart_strategy(resource_df)
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    if resource_df.empty:
        return None

    base_df = resource_df.copy()
    base_df = base_df[base_df["chart_strategy"].astype(str).isin(
        ("threshold", "calibrated", "sequential_csr",
         "sequential_csr_utility", "online_sqt", "online_lvq",
         "oracle", "random", "constant", "fixed", "topk")
    )].copy()
    if base_df.empty or "target_ratio" not in base_df.columns:
        return None

    approach_rows = base_df[~base_df["estimator"].isin({"weak_model", "strong_model"})]
    weak_ref = None
    strong_ref = None
    if not approach_rows.empty:
        r0 = approach_rows.loc[approach_rows["target_ratio"].astype(float).abs() < 1e-6, "mAP"]
        r1 = approach_rows.loc[(approach_rows["target_ratio"].astype(float) - 1.0).abs() < 1e-6, "mAP"]
        if not r0.empty:
            weak_ref = float(r0.astype(float).mean())
        if not r1.empty:
            strong_ref = float(r1.astype(float).mean())
    if weak_ref is None:
        wm = base_df.loc[base_df["estimator"] == "weak_model", "mAP"]
        weak_ref = float(wm.astype(float).iloc[0]) if not wm.empty else None
    if strong_ref is None:
        sm = base_df.loc[base_df["estimator"] == "strong_model", "mAP"]
        strong_ref = float(sm.astype(float).iloc[0]) if not sm.empty else None

    x_specs = [
        ("estimated_end_to_end_ms_per_frame", "Latency", "ms/frame"),
        ("estimated_end_to_end_gflops_per_frame", "GFLOPs", "GFLOPs/frame"),
    ]
    x_specs = [(c, name, unit) for c, name, unit in x_specs if c in base_df.columns]
    if not x_specs:
        return None

    weak_refs = {
        x_col: _resource_reference_value(base_df, x_col, "weak_model")
        for x_col, _, _ in x_specs
    }
    strong_refs = {
        x_col: _resource_reference_value(base_df, x_col, "strong_model")
        for x_col, _, _ in x_specs
    }

    focus_estimators = _resource_tradeoff_focus_estimators(base_df, offload_summary_df, limit=focus_limit)
    focus_names = focus_estimators or (
        base_df.loc[
            ~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"}),
            "estimator",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()[:6]
    )
    strategy_lookup = (
        base_df.groupby("estimator")["chart_strategy"]
        .first()
        .astype(str)
        .to_dict()
    )
    label_lookup = {
        name: _compact_approach_label(name, _canonical_strategy_name(strategy_lookup.get(name, "")))
        for name in base_df["estimator"].astype(str).drop_duplicates()
    }
    full_curve_names = [
        name for name in focus_names
        if base_df.loc[base_df["estimator"].astype(str) == name, "target_ratio"].nunique() >= 4
    ]
    ratio_source_names = full_curve_names or focus_names
    ratio_source_df = base_df[base_df["estimator"].astype(str).isin(ratio_source_names)].copy()
    if ratio_source_df.empty:
        ratio_source_df = base_df[~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"})].copy()
    all_ratio_values = sorted(
        ratio for ratio in ratio_source_df["target_ratio"].astype(float).unique().tolist()
        if np.isfinite(ratio) and 0.0 <= float(ratio) <= 1.0
    )
    return {
        "base_df": base_df,
        "weak_ref": weak_ref,
        "strong_ref": strong_ref,
        "x_specs": x_specs,
        "weak_refs": weak_refs,
        "strong_refs": strong_refs,
        "focus_names": focus_names,
        "full_curve_names": full_curve_names,
        "all_ratio_values": all_ratio_values,
        "label_lookup": label_lookup,
    }


def plot_resource_tradeoff(resource_df: pd.DataFrame, offload_summary_df: pd.DataFrame,
                           out_dir: Path, dataset_label: str = "") -> Path:
    resource_df = _with_chart_strategy(resource_df)
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    plt = _setup_matplotlib()
    path = out_dir / "resource_tradeoff.png"
    if resource_df.empty:
        return path

    base_df = resource_df.copy()
    base_df = base_df[base_df["chart_strategy"].astype(str).isin(
        ("threshold", "calibrated", "sequential_csr",
         "sequential_csr_utility", "online_sqt", "online_lvq",
         "oracle", "random", "constant", "fixed", "topk")
    )].copy()
    if base_df.empty or "target_ratio" not in base_df.columns:
        return path

    # Derive weak/strong reference mAP from the data at ratio extremes.
    # Use the actual offloading-simulation AP (approach rows at ratio=0/1)
    # so all points share the same metric.
    approach_rows = base_df[~base_df["estimator"].isin({"weak_model", "strong_model"})]
    weak_ref = None
    strong_ref = None
    if not approach_rows.empty:
        r0 = approach_rows.loc[approach_rows["target_ratio"].astype(float).abs() < 1e-6, "mAP"]
        r1 = approach_rows.loc[(approach_rows["target_ratio"].astype(float) - 1.0).abs() < 1e-6, "mAP"]
        if not r0.empty:
            weak_ref = float(r0.astype(float).mean())
        if not r1.empty:
            strong_ref = float(r1.astype(float).mean())
    # Fall back to weak_model/strong_model rows if no approach extremes exist
    if weak_ref is None:
        wm = base_df.loc[base_df["estimator"] == "weak_model", "mAP"]
        weak_ref = float(wm.astype(float).iloc[0]) if not wm.empty else None
    if strong_ref is None:
        sm = base_df.loc[base_df["estimator"] == "strong_model", "mAP"]
        strong_ref = float(sm.astype(float).iloc[0]) if not sm.empty else None

    target_ratios = list(_FOCUS_TARGET_RATIOS)
    x_specs = [
        ("estimated_end_to_end_ms_per_frame", "Latency", "ms/frame"),
        ("estimated_end_to_end_gflops_per_frame", "GFLOPs", "GFLOPs/frame"),
    ]
    x_specs = [(c, name, unit) for c, name, unit in x_specs if c in base_df.columns]
    if not x_specs:
        return path

    weak_refs = {
        x_col: _resource_reference_value(base_df, x_col, "weak_model")
        for x_col, _, _ in x_specs
    }
    strong_refs = {
        x_col: _resource_reference_value(base_df, x_col, "strong_model")
        for x_col, _, _ in x_specs
    }

    focus_estimators = _resource_tradeoff_focus_estimators(base_df, offload_summary_df, limit=5)
    strategy_lookup = (
        base_df.groupby("estimator")["chart_strategy"]
        .first()
        .astype(str)
        .to_dict()
    )
    label_lookup = {
        name: _compact_approach_label(name, _canonical_strategy_name(strategy_lookup.get(name, "")))
        for name in base_df["estimator"].astype(str).drop_duplicates()
    }
    full_curve_names = [
        name for name in focus_estimators
        if base_df.loc[base_df["estimator"].astype(str) == name, "target_ratio"].nunique() >= 4
    ]
    ratio_source_names = full_curve_names or focus_estimators
    ratio_source_df = base_df[base_df["estimator"].astype(str).isin(ratio_source_names)].copy()
    if ratio_source_df.empty:
        ratio_source_df = base_df[~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"})].copy()
    all_ratio_values = sorted(
        ratio for ratio in ratio_source_df["target_ratio"].astype(float).unique().tolist()
        if np.isfinite(ratio) and 0.0 <= float(ratio) <= 1.0
    )
    heatmap_estimators = full_curve_names or focus_estimators

    n_rows = len(target_ratios) + 2
    n_cols = len(x_specs)
    height_ratios = [1.0] * len(target_ratios) + [1.15, 1.0]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 4.3 * len(target_ratios) + 9.2),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    for row_idx, ratio in enumerate(target_ratios):
        # Select nearest available ratio per estimator (exclude weak/strong constants)
        df = base_df[~base_df["estimator"].isin({"weak_model", "strong_model"})].copy()
        df["ratio_distance"] = np.abs(df["target_ratio"].astype(float) - ratio)
        df = (
            df.sort_values(["estimator", "ratio_distance"])
            .groupby("estimator", as_index=False)
            .first()
        )
        if df.empty:
            for col_idx in range(n_cols):
                axes[row_idx, col_idx].set_visible(False)
            continue

        colors = [_STAGE_COLORS.get(_stage_from_name(name), _STAGE_COLORS["other"])
                  for name in df["estimator"]]

        for col_idx, (x_col, metric_name, raw_unit_label) in enumerate(x_specs):
            ax = axes[row_idx, col_idx]
            weak_ref_value = weak_refs.get(x_col)
            strong_ref_value = strong_refs.get(x_col)
            x_raw = df[x_col].astype(float).to_numpy()
            if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
                x_vals = x_raw / float(weak_ref_value)
                x_label = f"Estimated {metric_name.lower()} (x weak-only)"
            else:
                x_vals = x_raw
                x_label = f"Estimated {raw_unit_label}"

            ax.scatter(x_vals, df["mAP"].astype(float),
                       c=colors, s=80, edgecolor="white", linewidth=0.8, zorder=5)

            frontier_idx = _pareto_frontier_order(x_vals, df["mAP"].astype(float).to_numpy())
            if frontier_idx.size >= 2:
                ax.plot(
                    x_vals[frontier_idx],
                    df["mAP"].astype(float).to_numpy()[frontier_idx],
                    color="#111827",
                    ls=":",
                    lw=1.2,
                    alpha=0.55,
                    zorder=4,
                    label="Pareto frontier" if row_idx == 0 and col_idx == 0 else None,
                )

            # Reference lines for weak/strong model
            if weak_ref is not None:
                ax.axhline(y=weak_ref, color="black", ls="--", lw=1.2, alpha=0.6,
                           label=f"Weak detector ({weak_ref:.3f})")
            if strong_ref is not None:
                ax.axhline(y=strong_ref, color="blue", ls="-.", lw=1.2, alpha=0.6,
                           label=f"Strong detector ({strong_ref:.3f})")

            # Labels
            sorted_rows = sorted(
                zip(x_vals.tolist(), df.iterrows()),
                key=lambda r: float(r[1][1]["mAP"])
            )
            for i, (x_plot, (_, row)) in enumerate(sorted_rows):
                offset_y = 6 + (i % 4) * 10
                offset_x = 6 if (i % 2 == 0) else -60
                ax.annotate(
                    _compact_approach_label(
                        row["estimator"],
                        _canonical_strategy_name(row.get("chart_strategy", row.get("strategy", ""))),
                    ),
                    (x_plot, row["mAP"]),
                    textcoords="offset points", xytext=(offset_x, offset_y), fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="grey", alpha=0.4, lw=0.5),
                )

            if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
                ax.axvline(x=1.0, color="#111827", ls="--", lw=1.0, alpha=0.6)
                if strong_ref_value is not None and np.isfinite(strong_ref_value):
                    strong_ratio = float(strong_ref_value) / float(weak_ref_value)
                    ax.axvline(x=strong_ratio, color="#2563EB", ls="-.", lw=1.0, alpha=0.6)
                baseline_lines = [f"weak={weak_ref_value:.1f} {raw_unit_label}"]
                if strong_ref_value is not None and np.isfinite(strong_ref_value):
                    baseline_lines.append(f"strong={strong_ref_value:.1f} {raw_unit_label}")
                ax.text(
                    0.98, 0.04,
                    "\n".join(baseline_lines),
                    transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#374151",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#D1D5DB"),
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel("mAP@0.5")
            pct = int(ratio * 100)
            ax.set_title(f"{pct}% offload budget — {metric_name}")
            ax.grid(alpha=0.25, ls="--")
            if row_idx == 0 and col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7, loc="lower right")

    import matplotlib
    import matplotlib.lines as mlines

    ratio_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    ratio_cmap = matplotlib.colormaps["viridis"]
    palette = matplotlib.colormaps["tab10"]
    line_styles = ["-", "--", "-.", ":"]
    focus_names = focus_estimators or (
        base_df.loc[
            ~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"}),
            "estimator",
        ]
        .astype(str)
        .drop_duplicates()
        .tolist()[:6]
    )
    estimator_colors = {
        name: palette(idx % getattr(palette, "N", 10))
        for idx, name in enumerate(focus_names)
    }
    trajectory_handles: list[mlines.Line2D] = []
    trajectory_labels: list[str] = []
    trajectory_row_idx = len(target_ratios)
    heatmap_row_idx = trajectory_row_idx + 1

    for col_idx, (x_col, metric_name, raw_unit_label) in enumerate(x_specs):
        ax = axes[trajectory_row_idx, col_idx]
        weak_ref_value = weak_refs.get(x_col)
        strong_ref_value = strong_refs.get(x_col)

        if weak_ref is not None:
            ax.axhline(y=weak_ref, color="black", ls="--", lw=1.2, alpha=0.6)
        if strong_ref is not None:
            ax.axhline(y=strong_ref, color="blue", ls="-.", lw=1.2, alpha=0.6)

        if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
            ax.axvline(x=1.0, color="#111827", ls="--", lw=1.0, alpha=0.6)
            if strong_ref_value is not None and np.isfinite(strong_ref_value):
                strong_ratio = float(strong_ref_value) / float(weak_ref_value)
                ax.axvline(x=strong_ratio, color="#2563EB", ls="-.", lw=1.0, alpha=0.6)
            x_label = f"Estimated {metric_name.lower()} (x weak-only)"
        else:
            x_label = f"Estimated {raw_unit_label}"

        plotted_any = False
        all_frontier_x: list[float] = []
        all_frontier_y: list[float] = []
        for est_idx, estimator_name in enumerate(focus_names):
            est_df = base_df[base_df["estimator"].astype(str) == estimator_name].copy()
            if est_df.empty or x_col not in est_df.columns:
                continue

            est_df = est_df[
                est_df["target_ratio"].astype(float).between(0.0, 1.0, inclusive="both")
            ].copy()
            if est_df.empty:
                continue

            est_df = est_df.sort_values("target_ratio")
            x_raw = est_df[x_col].astype(float).to_numpy()
            if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
                x_vals = x_raw / float(weak_ref_value)
            else:
                x_vals = x_raw
            y_vals = est_df["mAP"].astype(float).to_numpy()
            ratios = est_df["target_ratio"].astype(float).to_numpy()
            finite = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(ratios)
            if not finite.any():
                continue

            x_vals = x_vals[finite]
            y_vals = y_vals[finite]
            ratios = ratios[finite]
            if x_vals.size == 0:
                continue

            line_color = estimator_colors.get(
                estimator_name, palette(est_idx % getattr(palette, "N", 10))
            )
            stage = _stage_from_name(estimator_name)
            line_style = line_styles[est_idx % len(line_styles)]
            if stage == "post" and line_style == "-":
                line_style = "--"

            if x_vals.size > 1:
                ax.plot(
                    x_vals,
                    y_vals,
                    color=line_color,
                    ls=line_style,
                    lw=2.0,
                    alpha=0.85,
                    zorder=2,
                )

            ax.scatter(
                x_vals,
                y_vals,
                c=ratio_cmap(ratio_norm(ratios)),
                s=58,
                edgecolor=line_color,
                linewidth=1.1,
                zorder=4,
            )
            all_frontier_x.extend(x_vals.tolist())
            all_frontier_y.extend(y_vals.tolist())

            if estimator_name not in trajectory_labels:
                trajectory_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=line_color,
                        ls=line_style,
                        lw=2.0,
                        marker="o",
                        markerfacecolor="white",
                        markeredgecolor=line_color,
                        markersize=5,
                    )
                )
                trajectory_labels.append(estimator_name)
            plotted_any = True

        if plotted_any:
            frontier_idx = _pareto_frontier_order(
                np.asarray(all_frontier_x, dtype=float),
                np.asarray(all_frontier_y, dtype=float),
            )
            if frontier_idx.size >= 2:
                ax.plot(
                    np.asarray(all_frontier_x, dtype=float)[frontier_idx],
                    np.asarray(all_frontier_y, dtype=float)[frontier_idx],
                    color="#111827",
                    ls=":",
                    lw=1.4,
                    alpha=0.6,
                    zorder=3,
                )

        if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
            baseline_lines = [f"weak={weak_ref_value:.1f} {raw_unit_label}"]
            if strong_ref_value is not None and np.isfinite(strong_ref_value):
                baseline_lines.append(f"strong={strong_ref_value:.1f} {raw_unit_label}")
            ax.text(
                0.98,
                0.04,
                "\n".join(baseline_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#374151",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#D1D5DB",
                ),
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"Budget trajectories — {metric_name}")
        ax.grid(alpha=0.25, ls="--")
        if not plotted_any:
            ax.set_visible(False)

    if heatmap_estimators and all_ratio_values:
        ratio_labels = [f"{int(round(float(ratio) * 100.0))}%" for ratio in all_ratio_values]
        for col_idx, (x_col, metric_name, _raw_unit_label) in enumerate(x_specs):
            ax = axes[heatmap_row_idx, col_idx]
            raw_matrix = _resource_efficiency_matrix(
                base_df,
                heatmap_estimators,
                all_ratio_values,
                x_col,
                weak_refs.get(x_col),
                weak_ref,
            )
            finite_values = raw_matrix[np.isfinite(raw_matrix)]
            if finite_values.size == 0:
                ax.set_visible(False)
                continue

            matrix = _normalize_heatmap_matrix(raw_matrix)
            sign_mixed = float(finite_values.min()) < 0.0 < float(finite_values.max())
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
            cmap = matplotlib.colormaps["RdYlGn"].copy()
            cmap.set_bad("#F3F4F6")

            image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
            ax.set_xticks(np.arange(len(all_ratio_values)))
            ax.set_xticklabels(ratio_labels)
            ax.set_yticks(np.arange(len(heatmap_estimators)))
            ax.set_yticklabels([label_lookup.get(name, name) for name in heatmap_estimators], fontsize=7)
            ax.set_xlabel("Target offload ratio")
            ax.set_ylabel("Approach")
            ax.set_title(f"Normalized gain-density heatmap — {metric_name}")
            ax.set_xticks(np.arange(-0.5, len(all_ratio_values), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(heatmap_estimators), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)

            for row_idx, row_values in enumerate(matrix):
                finite_row = np.isfinite(row_values)
                if not finite_row.any():
                    continue
                best_col = int(np.nanargmax(row_values))
                ax.scatter(
                    best_col,
                    row_idx,
                    s=28,
                    marker="s",
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=0.8,
                    zorder=5,
                )

            cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_label(
                "Normalized gain density"
                + (" (0.5 = zero)" if sign_mixed else "")
            )
    else:
        for col_idx in range(n_cols):
            axes[heatmap_row_idx, col_idx].set_visible(False)

    if trajectory_handles:
        fig.legend(
            trajectory_handles,
            [label_lookup.get(name, name) for name in trajectory_labels],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.018),
            ncol=min(3, max(1, len(trajectory_handles))),
            frameon=False,
            fontsize=7,
        )

    if focus_names:
        sm = matplotlib.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[trajectory_row_idx, :], fraction=0.03, pad=0.02)
        cbar.set_label("Target offload ratio")

    fig.suptitle(
        _figure_title(
            "Resource Trade-off",
            "Costs are normalized to the weak-only baseline so the modeled detector latency does not swamp the visual.\n"
            "Top rows show fixed-budget snapshots with Pareto frontiers; the middle row connects full trajectories across budgets.\n"
            "Bottom heatmaps summarize normalized gain density, so efficient offloading ratios stand out quickly even when latency and GFLOPs live on different scales.",
            dataset_label=dataset_label,
        ),
        y=1.02,
    )
    fig.subplots_adjust(left=0.08, right=0.93, top=0.965, bottom=0.10, hspace=0.46, wspace=0.24)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_resource_frontier_comparison(resource_df: pd.DataFrame,
                                      offload_summary_df: pd.DataFrame,
                                      out_dir: Path,
                                      dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "resource_frontier.png"
    ctx = _prepare_resource_tradeoff_context(resource_df, offload_summary_df, focus_limit=6)
    if ctx is None:
        return path

    base_df = ctx["base_df"]
    weak_ref = ctx["weak_ref"]
    strong_ref = ctx["strong_ref"]
    x_specs = ctx["x_specs"]
    weak_refs = ctx["weak_refs"]
    strong_refs = ctx["strong_refs"]
    focus_names = ctx["focus_names"]
    label_lookup = ctx["label_lookup"]

    import matplotlib
    import matplotlib.lines as mlines

    ratio_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    ratio_cmap = matplotlib.colormaps["viridis"]
    palette = matplotlib.colormaps["tab10"]
    line_styles = ["-", "--", "-.", ":"]
    estimator_colors = {
        name: palette(idx % getattr(palette, "N", 10))
        for idx, name in enumerate(focus_names)
    }

    fig, axes = plt.subplots(1, len(x_specs), figsize=(7.4 * len(x_specs), 6.6), squeeze=False)
    legend_handles = [
        mlines.Line2D([], [], color="#111827", ls=":", lw=1.8, label="Global Pareto frontier")
    ]
    legend_labels = ["Global Pareto frontier"]

    for col_idx, (x_col, metric_name, raw_unit_label) in enumerate(x_specs):
        ax = axes[0, col_idx]
        weak_ref_value = weak_refs.get(x_col)
        strong_ref_value = strong_refs.get(x_col)

        all_df = base_df[~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"})].copy()
        all_df = all_df.sort_values(["target_ratio", "estimator"]).reset_index(drop=True)
        x_all_raw = all_df[x_col].astype(float).to_numpy()
        if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
            x_all = x_all_raw / float(weak_ref_value)
            x_label = f"Estimated {metric_name.lower()} (x weak-only)"
        else:
            x_all = x_all_raw
            x_label = f"Estimated {raw_unit_label}"
        y_all = all_df["mAP"].astype(float).to_numpy()
        ratio_all = all_df["target_ratio"].astype(float).to_numpy()
        finite_all = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(ratio_all)
        x_plot_all = x_all[finite_all]
        y_plot_all = y_all[finite_all]
        ratio_plot_all = ratio_all[finite_all]
        frontier_df = all_df.loc[np.flatnonzero(finite_all)].reset_index(drop=True)

        ax.scatter(
            x_plot_all,
            y_plot_all,
            c=ratio_cmap(ratio_norm(ratio_plot_all)),
            s=26,
            alpha=0.18,
            edgecolor="none",
            zorder=1,
        )

        frontier_idx = _pareto_frontier_order(x_plot_all, y_plot_all)
        if frontier_idx.size >= 2:
            ax.plot(
                x_plot_all[frontier_idx],
                y_plot_all[frontier_idx],
                color="#111827",
                ls=":",
                lw=1.8,
                alpha=0.85,
                zorder=5,
            )
            for order_idx, frontier_pos in enumerate(frontier_idx.tolist()):
                row = frontier_df.iloc[frontier_pos]
                if str(row["estimator"]) not in set(focus_names):
                    continue
                ax.annotate(
                    f"{label_lookup.get(str(row['estimator']), str(row['estimator']))}\n{int(round(float(row['target_ratio']) * 100.0))}%",
                    (x_plot_all[frontier_pos], y_plot_all[frontier_pos]),
                    textcoords="offset points",
                    xytext=(8, 8 + 10 * (order_idx % 2)),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="#6B7280", alpha=0.35, lw=0.5),
                    zorder=6,
                )

        for est_idx, estimator_name in enumerate(focus_names):
            est_df = base_df[base_df["estimator"].astype(str) == estimator_name].copy()
            if est_df.empty or x_col not in est_df.columns:
                continue
            est_df = est_df.sort_values("target_ratio")
            x_raw = est_df[x_col].astype(float).to_numpy()
            if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
                x_vals = x_raw / float(weak_ref_value)
            else:
                x_vals = x_raw
            y_vals = est_df["mAP"].astype(float).to_numpy()
            ratios = est_df["target_ratio"].astype(float).to_numpy()
            finite = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(ratios)
            if not finite.any():
                continue

            x_vals = x_vals[finite]
            y_vals = y_vals[finite]
            ratios = ratios[finite]
            line_color = estimator_colors[estimator_name]
            line_style = line_styles[est_idx % len(line_styles)]
            ax.plot(
                x_vals,
                y_vals,
                color=line_color,
                ls=line_style,
                lw=2.1,
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                x_vals,
                y_vals,
                c=ratio_cmap(ratio_norm(ratios)),
                s=62,
                edgecolor=line_color,
                linewidth=1.1,
                zorder=4,
            )

            if estimator_name not in legend_labels:
                legend_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=line_color,
                        ls=line_style,
                        lw=2.1,
                        marker="o",
                        markerfacecolor="white",
                        markeredgecolor=line_color,
                        markersize=5,
                    )
                )
                legend_labels.append(estimator_name)

        if weak_ref is not None:
            ax.axhline(y=weak_ref, color="black", ls="--", lw=1.2, alpha=0.6)
        if strong_ref is not None:
            ax.axhline(y=strong_ref, color="blue", ls="-.", lw=1.2, alpha=0.6)
        if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
            ax.axvline(x=1.0, color="#111827", ls="--", lw=1.0, alpha=0.6)
            if strong_ref_value is not None and np.isfinite(strong_ref_value):
                ax.axvline(
                    x=float(strong_ref_value) / float(weak_ref_value),
                    color="#2563EB",
                    ls="-.",
                    lw=1.0,
                    alpha=0.6,
                )

        ax.set_xlabel(x_label)
        ax.set_ylabel("mAP@0.5")
        ax.set_title(f"Global Pareto Frontier — {metric_name}")
        ax.grid(alpha=0.25, ls="--")
        ax.text(
            0.02,
            0.03,
            "How to read:\nPareto frontier = points where you cannot\nlower cost without also lowering mAP,\nand cannot raise mAP without also raising cost.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#374151",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                alpha=0.88,
                edgecolor="#D1D5DB",
            ),
        )

    fig.legend(
        legend_handles,
        [
            "Global Pareto frontier" if label == "Global Pareto frontier" else label_lookup.get(label, label)
            for label in legend_labels
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(3, max(1, len(legend_handles))),
        frameon=False,
        fontsize=7,
    )
    sm = matplotlib.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Target offload ratio")

    fig.suptitle(
        _figure_title(
            "Resource Frontier Comparison",
            "Each point is one approach at one offload ratio. The black dotted curve is the global Pareto frontier.\n"
            "A point is on the frontier when no other point is better on both axes at once: lower cost and higher mAP.",
            dataset_label=dataset_label,
        ),
        y=1.01,
    )
    fig.subplots_adjust(left=0.08, right=0.93, top=0.90, bottom=0.16, wspace=0.22)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_resource_frontier_summary(resource_df: pd.DataFrame,
                                   offload_summary_df: pd.DataFrame,
                                   out_dir: Path,
                                   dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "resource_frontier_summary.png"
    ctx = _prepare_resource_tradeoff_context(resource_df, offload_summary_df, focus_limit=6)
    if ctx is None:
        return path

    base_df = ctx["base_df"]
    x_specs = ctx["x_specs"]
    weak_refs = ctx["weak_refs"]
    focus_names = ctx["full_curve_names"] or ctx["focus_names"]
    all_ratio_values = ctx["all_ratio_values"]
    label_lookup = ctx["label_lookup"]
    if not focus_names or not all_ratio_values:
        return path

    import matplotlib

    membership = {
        x_col: np.full((len(focus_names), len(all_ratio_values)), np.nan, dtype=float)
        for x_col, _, _ in x_specs
    }
    counts = {x_col: np.zeros(len(focus_names), dtype=int) for x_col, _, _ in x_specs}

    base_points_df = base_df[~base_df["estimator"].astype(str).isin({"weak_model", "strong_model"})].copy()
    for col_idx, (x_col, _metric_name, _raw_unit_label) in enumerate(x_specs):
        weak_ref_value = weak_refs.get(x_col)
        for ratio_idx, ratio in enumerate(all_ratio_values):
            ratio_df = base_points_df[np.isclose(base_points_df["target_ratio"].astype(float), float(ratio), atol=1e-6)].copy()
            if ratio_df.empty:
                continue
            ratio_df = ratio_df.reset_index(drop=True)
            x_raw = ratio_df[x_col].astype(float).to_numpy()
            if weak_ref_value is not None and np.isfinite(weak_ref_value) and weak_ref_value > 0:
                x_vals = x_raw / float(weak_ref_value)
            else:
                x_vals = x_raw
            y_vals = ratio_df["mAP"].astype(float).to_numpy()
            finite = np.isfinite(x_vals) & np.isfinite(y_vals)
            if not finite.any():
                continue
            ratio_df = ratio_df.iloc[np.flatnonzero(finite)].reset_index(drop=True)
            frontier_idx = _pareto_frontier_order(x_vals[finite], y_vals[finite])
            frontier_names = set(ratio_df.iloc[frontier_idx]["estimator"].astype(str).tolist())
            available_names = set(ratio_df["estimator"].astype(str).tolist())
            for row_idx, estimator_name in enumerate(focus_names):
                if estimator_name in frontier_names:
                    membership[x_col][row_idx, ratio_idx] = 1.0
                    counts[x_col][row_idx] += 1
                elif estimator_name in available_names:
                    membership[x_col][row_idx, ratio_idx] = 0.0

    fig, axes = plt.subplots(1, len(x_specs), figsize=(7.4 * len(x_specs), 5.2), squeeze=False)
    cmap = matplotlib.colormaps["Greens"].copy()
    cmap.set_bad("#F3F4F6")
    ratio_labels = [f"{int(round(float(ratio) * 100.0))}%" for ratio in all_ratio_values]

    for col_idx, (x_col, metric_name, _raw_unit_label) in enumerate(x_specs):
        ax = axes[0, col_idx]
        matrix = membership[x_col]
        if not np.isfinite(matrix).any():
            ax.set_visible(False)
            continue
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
        display_labels = [
            f"{label_lookup.get(name, name)} ({counts[x_col][row_idx]})"
            for row_idx, name in enumerate(focus_names)
        ]
        ax.set_xticks(np.arange(len(all_ratio_values)))
        ax.set_xticklabels(ratio_labels)
        ax.set_yticks(np.arange(len(focus_names)))
        ax.set_yticklabels(display_labels, fontsize=7)
        ax.set_xlabel("Target offload ratio")
        ax.set_ylabel("Approach (frontier count)")
        ax.set_title(f"Pareto-Optimal Ratios — {metric_name}")
        ax.set_xticks(np.arange(-0.5, len(all_ratio_values), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(focus_names), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.text(
            0.02,
            0.03,
            "Green = this approach-ratio point is on the Pareto frontier.\nLight = some other approach gives higher mAP\nat the same or lower cost.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            color="#374151",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                alpha=0.88,
                edgecolor="#D1D5DB",
            ),
        )

        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_ticks([0.0, 1.0])
        cbar.set_ticklabels(["Dominated", "Frontier"])
        cbar.set_label("Pareto status\n(best trade-off vs beaten by another point)")

    fig.suptitle(
        _figure_title(
            "Resource Frontier Summary",
            "Green cells mark ratios where an approach lies on the Pareto frontier for the chosen resource metric.\n"
            "Pareto frontier means no other approach-ratio point gives both lower cost and higher mAP at the same time.",
            dataset_label=dataset_label,
        ),
        y=1.02,
    )
    fig.subplots_adjust(left=0.17, right=0.94, top=0.88, bottom=0.14, wspace=0.30)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def export_paper_tables(metrics_df: pd.DataFrame, ranking_df: pd.DataFrame,
                        offload_summary_df: pd.DataFrame,
                        statistical_df: pd.DataFrame, metrics_dir: Path) -> list[Path]:
    outputs: list[Path] = []

    if not metrics_df.empty and not offload_summary_df.empty:
        off_df = offload_summary_df.copy()
        if "strategy" in off_df.columns:
            # Use threshold as primary strategy; include oracle/random/constant for baselines
            off_df = off_df[off_df["strategy"].astype(str).isin(("threshold", "oracle", "random", "constant", "fixed"))]
            off_df = off_df.sort_values("auc_0_5", ascending=False).drop_duplicates(subset=["estimator"], keep="first")
        rank_df = ranking_df.copy()
        keep_rank = [col for col in ("estimator", "scenario", "scenario_type", "spearman_rho", "ndcg") if col in rank_df.columns]
        rank_df = rank_df[keep_rank].groupby([col for col in ("estimator", "scenario", "scenario_type") if col in rank_df.columns], as_index=False).mean(numeric_only=True) if keep_rank else pd.DataFrame()
        merge_cols = [col for col in ("estimator", "scenario", "scenario_type") if col in off_df.columns and (rank_df.empty or col in rank_df.columns) and col in metrics_df.columns]
        table_df = metrics_df.merge(off_df, on=[col for col in merge_cols if col in metrics_df.columns and col in off_df.columns], how="left", suffixes=("", "_offload"))
        if not rank_df.empty:
            table_df = table_df.merge(rank_df, on=merge_cols, how="left", suffixes=("", "_rank"))
        wanted = [col for col in ("estimator", "base_model", "scenario", "scenario_type", "strategy", "spearman_rho", "ndcg", "peak_map", "peak_map_coco", "auc_0_5", "auc_coco", "oracle_regret_auc_0_5", "mean_ratio_error", "inference_time_ms") if col in table_df.columns]
        main_path = metrics_dir / "main_benchmark_table.csv"
        table_df[wanted].to_csv(main_path, index=False)
        outputs.append(main_path)

        scenario_df = table_df[table_df["scenario"].fillna("") != ""] if "scenario" in table_df.columns else pd.DataFrame()
        if not scenario_df.empty:
            scenario_path = metrics_dir / "scenario_benchmark_table.csv"
            scenario_df[wanted].to_csv(scenario_path, index=False)
            outputs.append(scenario_path)

    if not statistical_df.empty:
        stats_path = metrics_dir / "statistical_summary_table.csv"
        statistical_df.to_csv(stats_path, index=False)
        outputs.append(stats_path)

    return outputs
