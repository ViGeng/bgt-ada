"""Scenario-adaptive and estimator/approach overview charts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .analyse_helpers import (
    _DCSB_KEYS,
    _ORACLE_NAME,
    _STAGE_COLORS,
    _VIRTUAL_NAMES,
    _approach_policy_label,
    _build_estimator_styles,
    _canonical_strategy_name,
    _collapse_scenario_variants,
    _compact_approach_label,
    _compact_estimator_label,
    _ensure_policy_columns,
    _estimator_model_key,
    _estimator_name_without_offloader,
    _extract_proxy_metric,
    _figure_title,
    _filter_headline_estimators,
    _headline_estimators,
    _prediction_diagnostics_figure,
    _prepare_scenario_method_comparison,
    _pretty_proxy_metric_label,
    _pretty_scenario_label,
    _proxy_family_color,
    _proxy_metric_family,
    _ranking_auc_column,
    _render_overview_table,
    _resolve_trace_plot_ratios,
    _resource_reference_value,
    _setup_matplotlib,
    _stage_from_name,
    _trace_strategy_label,
)

# _compact_approach_label is re-exported for backward compatibility
# (it was defined in this module in early drafts)


def plot_scenario_comparison(offload_summary_df: pd.DataFrame, out_dir: Path,
                             dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "scenario_comparison.png"
    prepared = _prepare_scenario_method_comparison(offload_summary_df)
    if prepared is None:
        return path

    scenario_order = prepared["scenario_order"]
    scenario_labels = [_pretty_scenario_label(name) for name in scenario_order]
    x = np.arange(len(scenario_order))

    fig, (ax_auc, ax_peak) = plt.subplots(
        2, 1, figsize=(12.5, 8.8), sharex=True, constrained_layout=True
    )

    adaptive_colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    best_aside_label = f"Best non-adaptive pre-stage ({prepared['best_aside_label']})"

    for idx, base_model in enumerate(prepared["adaptive_base_models"]):
        adaptive_auc = prepared["adaptive_auc"].get(base_model)
        if adaptive_auc is None:
            continue
        ax_auc.plot(
            x,
            adaptive_auc.to_numpy(dtype=float),
            marker="o",
            linewidth=2.2,
            color=adaptive_colors[idx % len(adaptive_colors)],
            label=prepared["adaptive_labels"].get(base_model, base_model),
        )
    ax_auc.plot(x, np.full(len(x), prepared["best_aside_auc"]), linestyle="--",
                linewidth=1.8, color="#2E8B57", label=best_aside_label)
    ax_auc.plot(x, np.full(len(x), prepared["edge_auc"]), linestyle="-.",
                linewidth=1.8, color="#E8734C", label="EdgeML")
    ax_auc.set_ylabel("AUC@0.5")
    ax_auc.set_title(f"AUC@0.5 Comparison ({prepared['chosen_strategy']})")
    ax_auc.grid(axis="y", alpha=0.25, ls="--")
    ax_auc.legend(loc="upper left", frameon=False)
    # Zoom Y-axis to data range for visible differentiation
    auc_vals = [v for bm in prepared["adaptive_base_models"]
                if (v_arr := prepared["adaptive_auc"].get(bm)) is not None
                for v in v_arr.to_numpy(dtype=float)]
    auc_vals.extend([prepared["best_aside_auc"], prepared["edge_auc"]])
    if auc_vals:
        auc_lo = min(auc_vals) - 0.003
        auc_hi = max(auc_vals) + 0.003
        ax_auc.set_ylim(auc_lo, auc_hi)

    for idx, base_model in enumerate(prepared["adaptive_base_models"]):
        adaptive_peak = prepared["adaptive_peak"].get(base_model)
        if adaptive_peak is None:
            continue
        ax_peak.plot(
            x,
            adaptive_peak.to_numpy(dtype=float),
            marker="o",
            linewidth=2.2,
            color=adaptive_colors[idx % len(adaptive_colors)],
            label=prepared["adaptive_labels"].get(base_model, base_model),
        )
    ax_peak.plot(x, np.full(len(x), prepared["best_aside_peak"]), linestyle="--",
                 linewidth=1.8, color="#2E8B57", label=best_aside_label)
    ax_peak.plot(x, np.full(len(x), prepared["edge_peak"]), linestyle="-.",
                 linewidth=1.8, color="#E8734C", label="EdgeML")
    if prepared["dcsb_peak"] is not None:
        ax_peak.plot(x, np.full(len(x), prepared["dcsb_peak"]), linestyle=":",
                     linewidth=2.0, color="#7A7A7A", label="DCSB")
    ax_peak.set_ylabel("Peak mAP")
    ax_peak.set_title("Peak mAP Comparison")
    ax_peak.grid(axis="y", alpha=0.25, ls="--")
    ax_peak.legend(loc="upper left", frameon=False)
    ax_peak.set_xticks(x)
    ax_peak.set_xticklabels(scenario_labels, rotation=16, ha="right")
    ax_peak.set_xlabel("Runtime scenario")

    if dataset_label:
        fig.suptitle(f"Scenario Comparison [{dataset_label}]", fontsize=14)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scenario_radar(offload_summary_df: pd.DataFrame, out_dir: Path,
                        dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "scenario_radar.png"
    prepared = _prepare_scenario_method_comparison(offload_summary_df)
    if prepared is None:
        return path

    scenario_labels = [_pretty_scenario_label(name) for name in prepared["scenario_order"]]
    values_map = {
        prepared["adaptive_labels"].get(base_model, base_model): prepared["adaptive_peak"][base_model].to_numpy(dtype=float)
        for base_model in prepared["adaptive_base_models"]
        if base_model in prepared["adaptive_peak"]
    }
    values_map[f"Best non-adaptive pre-stage ({prepared['best_aside_label']})"] = np.full(
            len(scenario_labels), prepared["best_aside_peak"], dtype=float
        )
    values_map["EdgeML"] = np.full(len(scenario_labels), prepared["edge_peak"], dtype=float)
    if prepared["dcsb_peak"] is not None:
        values_map["DCSB"] = np.full(len(scenario_labels), prepared["dcsb_peak"], dtype=float)

    angles = np.linspace(0, 2 * np.pi, len(scenario_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.8, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, polar=True)
    colors = {
        "EdgeML": "#E8734C",
        "DCSB": "#7A7A7A",
    }
    for idx, base_model in enumerate(prepared["adaptive_base_models"]):
        label = prepared["adaptive_labels"].get(base_model, base_model)
        colors[label] = (list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors))[idx % 30]
    colors[f"Best non-adaptive pre-stage ({prepared['best_aside_label']})"] = "#2E8B57"

    all_values = np.concatenate([vals for vals in values_map.values()])
    radial_min = float(all_values.min()) - 0.005
    radial_max = float(all_values.max()) + 0.005
    ax.set_ylim(radial_min, radial_max)

    for label, vals in values_map.items():
        closed = np.concatenate([vals, vals[:1]])
        ax.plot(angles, closed, linewidth=2.0, label=label, color=colors.get(label, "#444444"))
        if label in {
            prepared["adaptive_labels"].get(base_model, base_model)
            for base_model in prepared["adaptive_base_models"]
        }:
            ax.fill(angles, closed, color=colors[label], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(scenario_labels)
    ax.set_title("Scenario Radar (Peak mAP)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=False)
    if dataset_label:
        fig.suptitle(f"Scenario Radar [{dataset_label}]", fontsize=14)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scenario_perf_heatmap(offload_summary_df: pd.DataFrame,
                               out_dir: Path,
                               dataset_label: str = "") -> Path:
    """Heatmap of AUC@0.5 for each (adaptive approach × scenario) cell."""
    plt = _setup_matplotlib()
    path = out_dir / "scenario_perf_heatmap.png"

    if offload_summary_df.empty or "scenario" not in offload_summary_df.columns:
        return path

    df = _ensure_policy_columns(offload_summary_df.copy())
    scenario_mask = df["scenario"].fillna("").astype(str) != ""
    if not scenario_mask.any():
        return path
    df = df[scenario_mask].copy()
    if "base_model" not in df.columns:
        df["base_model"] = df["estimator"].astype(str).map(_estimator_model_key)

    strategy_order = ("sequential_csr_utility", "sequential_csr",
                      "calibrated", "threshold")
    available = set(df.get("strategy", pd.Series(dtype=object)).fillna("").astype(str))
    chosen = next((s for s in strategy_order if s in available), None)
    if chosen is None:
        return path
    df = df[df["strategy"].astype(str) == chosen].copy()

    if "auc_0_5" not in df.columns or df["auc_0_5"].isna().all():
        return path

    pivot = df.pivot_table(index="scenario", columns="base_model",
                           values="auc_0_5", aggfunc="mean")
    if pivot.empty:
        return path

    # Order scenarios: presets first, then mixes
    order_meta = (
        df[["scenario", "scenario_type"]].drop_duplicates()
        .assign(_g=lambda x: x["scenario_type"].astype(str).map(
            {"preset": 0, "mix": 1}).fillna(2))
        .sort_values(["_g", "scenario"])
    )
    pivot = pivot.reindex(order_meta["scenario"].tolist())

    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(
        figsize=(max(4.5, 2.4 * n_cols), max(3.5, 0.65 * n_rows)),
        constrained_layout=True,
    )
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn",
                   vmin=float(np.nanmin(data)) - 0.003,
                   vmax=float(np.nanmax(data)) + 0.003)
    plt.colorbar(im, ax=ax, label="AUC@0.5", fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([_compact_estimator_label(c) for c in pivot.columns],
                       rotation=25, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_pretty_scenario_label(s) for s in pivot.index])
    ax.set_xlabel("Adaptive approach")
    ax.set_ylabel("Runtime scenario")

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    title = f"Scenario Performance Heatmap (AUC@0.5, {chosen})"
    if dataset_label:
        title += f" [{dataset_label}]"
    ax.set_title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scenario_weight_profiles(scenario_profiles_df: pd.DataFrame,
                                   out_dir: Path,
                                   dataset_label: str = "") -> Path:
    """Heatmap of component weights across scenario profiles."""
    import ast
    from config.scenarios import SCENARIO_COMPONENT_CATALOG

    plt = _setup_matplotlib()
    path = out_dir / "scenario_weight_profiles.png"

    if scenario_profiles_df is None or scenario_profiles_df.empty:
        return path
    if "scenario" not in scenario_profiles_df.columns or "weight_map" not in scenario_profiles_df.columns:
        return path

    df = scenario_profiles_df.drop_duplicates(subset=["scenario"]).copy()

    weight_records = []
    for _, row in df.iterrows():
        wmap = row.get("weight_map", "{}")
        if isinstance(wmap, str):
            try:
                wmap = ast.literal_eval(wmap)
            except Exception:
                continue
        if not isinstance(wmap, dict) or not wmap:
            continue
        record = {"scenario": str(row["scenario"])}
        record.update({str(k): float(v) for k, v in wmap.items()})
        weight_records.append(record)

    if not weight_records:
        return path

    weight_df = pd.DataFrame(weight_records).set_index("scenario")
    weight_df = weight_df.loc[:, (weight_df > 0).any()]
    if weight_df.empty:
        return path

    if "scenario_type" in df.columns:
        order_meta = (
            df[["scenario", "scenario_type"]].drop_duplicates()
            .assign(_g=lambda x: x["scenario_type"].astype(str).map(
                {"preset": 0, "mix": 1}).fillna(2))
            .sort_values(["_g", "scenario"])
        )
        ordered = [s for s in order_meta["scenario"].tolist() if s in weight_df.index]
        weight_df = weight_df.reindex(ordered)

    col_labels = [
        SCENARIO_COMPONENT_CATALOG.get(c, {}).get("label", c)
        for c in weight_df.columns
    ]

    data = weight_df.values.astype(float)
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.3 * n_cols), max(3.0, 0.65 * n_rows)),
        constrained_layout=True,
    )
    im = ax.imshow(data, aspect="auto", cmap="Blues",
                   vmin=0.0, vmax=float(data.max()) + 0.05)
    plt.colorbar(im, ax=ax, label="Weight", fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_pretty_scenario_label(s) for s in weight_df.index])

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if val > 0.55 else "black")

    title = "Scenario Component Weights"
    if dataset_label:
        title += f" [{dataset_label}]"
    ax.set_title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scenario_sensitivity(offload_summary_df: pd.DataFrame,
                               out_dir: Path,
                               dataset_label: str = "") -> Path:
    """Bar chart showing per-approach AUC range across scenarios."""
    plt = _setup_matplotlib()
    path = out_dir / "scenario_sensitivity.png"

    if offload_summary_df.empty or "scenario" not in offload_summary_df.columns:
        return path

    df = _ensure_policy_columns(offload_summary_df.copy())
    scenario_mask = df["scenario"].fillna("").astype(str) != ""
    if not scenario_mask.any():
        return path

    scen_df = df[scenario_mask].copy()
    if "base_model" not in scen_df.columns:
        scen_df["base_model"] = scen_df["estimator"].astype(str).map(_estimator_model_key)

    strategy_order = ("sequential_csr_utility", "sequential_csr",
                      "calibrated", "threshold")
    available = set(scen_df.get("strategy", pd.Series(dtype=object)).fillna("").astype(str))
    chosen = next((s for s in strategy_order if s in available), None)
    if chosen is None:
        return path

    scen_df = scen_df[scen_df["strategy"].astype(str) == chosen].copy()
    if "auc_0_5" not in scen_df.columns or scen_df["auc_0_5"].isna().all():
        return path

    stats = (
        scen_df.groupby("base_model")["auc_0_5"]
        .agg(mean="mean", lo="min", hi="max")
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    if stats.empty:
        return path

    # Best static (non-adaptive) pre-stage baseline for reference
    non_scen = df[~scenario_mask].copy()
    if "base_model" not in non_scen.columns:
        non_scen["base_model"] = non_scen["estimator"].astype(str).map(_estimator_model_key)
    skip_keys = _DCSB_KEYS | {"edgeml", "random", "oracle", "weak_model", "strong_model"}
    skip_keys.update(set(stats["base_model"].astype(str).tolist()))
    static = non_scen[
        (~non_scen["base_model"].astype(str).isin(skip_keys)) &
        (non_scen["stage"].astype(str) == "pre") &
        non_scen["auc_0_5"].notna()
    ]
    best_static = float(static["auc_0_5"].max()) if not static.empty else None

    x = np.arange(len(stats))
    fig, ax = plt.subplots(
        figsize=(max(5, 2.0 * len(stats)), 4.5),
        constrained_layout=True,
    )
    ax.bar(x, stats["mean"].values, color="#4C9BE8", alpha=0.8, label="Mean AUC@0.5")
    for i, row in enumerate(stats.itertuples()):
        ax.plot([i, i], [row.lo, row.hi], color="#222222", linewidth=2.5, zorder=3)
        ax.plot(i, row.lo, "v", color="#222222", markersize=7, zorder=4)
        ax.plot(i, row.hi, "^", color="#222222", markersize=7, zorder=4)
    if best_static is not None:
        ax.axhline(best_static, linestyle="--", color="#2E8B57", linewidth=1.8,
                   label=f"Best static pre-stage ({best_static:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels([_compact_estimator_label(bm) for bm in stats["base_model"]],
                       rotation=18, ha="right")
    ax.set_ylabel("AUC@0.5")
    ax.grid(axis="y", alpha=0.25, ls="--")
    ax.legend(loc="upper right", frameon=False)
    title = f"Scenario Sensitivity — AUC range across scenarios ({chosen})"
    if dataset_label:
        title += f" [{dataset_label}]"
    ax.set_title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scenario_offloading_curves(offload_df: pd.DataFrame,
                                     out_dir: Path,
                                     dataset_label: str = "") -> Path:
    """Faceted offloading curves — one subplot per scenario."""
    plt = _setup_matplotlib()
    path = out_dir / "scenario_offloading_curves.png"

    if offload_df.empty or "scenario" not in offload_df.columns:
        return path

    df = _ensure_policy_columns(offload_df.copy())
    if "base_model" not in df.columns:
        df["base_model"] = df["estimator"].astype(str).map(_estimator_model_key)

    scenario_mask = df["scenario"].fillna("").astype(str) != ""
    scen_df = df[scenario_mask].copy()
    if scen_df.empty:
        return path

    strategy_order = ("sequential_csr_utility", "sequential_csr",
                      "calibrated", "threshold")
    available = set(scen_df.get("strategy", pd.Series(dtype=object)).fillna("").astype(str))
    chosen = next((s for s in strategy_order if s in available), None)
    if chosen is None:
        return path
    scen_df = scen_df[scen_df["strategy"].astype(str) == chosen].copy()

    scenarios = sorted(scen_df["scenario"].astype(str).unique())
    if not scenarios:
        return path

    baselines_df = df[
        ~scenario_mask &
        df["base_model"].astype(str).isin({"oracle", "random", "weak_model"})
    ].copy()

    adaptive_models = sorted(scen_df["base_model"].astype(str).unique())
    adaptive_colors = list(plt.cm.tab10.colors)

    ncols = min(3, len(scenarios))
    nrows = (len(scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.0 * nrows),
                              constrained_layout=True)
    axes_flat = np.array(axes).reshape(-1) if len(scenarios) > 1 else [axes]

    for idx, (ax, scenario) in enumerate(zip(axes_flat, scenarios)):
        sdf = scen_df[scen_df["scenario"].astype(str) == scenario]
        for midx, bm in enumerate(adaptive_models):
            bdf = sdf[sdf["base_model"].astype(str) == bm].sort_values("ratio")
            if bdf.empty or "mAP" not in bdf.columns:
                continue
            ax.plot(bdf["ratio"].astype(float), bdf["mAP"].astype(float),
                    marker="o", markersize=4, linewidth=1.8,
                    color=adaptive_colors[midx % len(adaptive_colors)],
                    label=_compact_estimator_label(bm))
        for bname, bcolor, bstyle in [
            ("oracle",     "#333333", "--"),
            ("random",     "#AAAAAA", ":"),
            ("weak_model", "#4C9BE8", "-."),
        ]:
            bdf = baselines_df[baselines_df["base_model"].astype(str) == bname].sort_values("ratio")
            if not bdf.empty and "mAP" in bdf.columns:
                ax.plot(bdf["ratio"].astype(float), bdf["mAP"].astype(float),
                        linewidth=1.4, color=bcolor, linestyle=bstyle, label=bname)
        ax.set_title(_pretty_scenario_label(scenario))
        ax.set_xlabel("Offload ratio")
        ax.set_ylabel("mAP@0.5")
        ax.grid(alpha=0.2, ls="--")
        if idx == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7)

    for ax in axes_flat[len(scenarios):]:
        ax.set_visible(False)

    title = f"Per-Scenario Offloading Curves ({chosen})"
    if dataset_label:
        title += f" [{dataset_label}]"
    fig.suptitle(title, fontsize=13)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_estimator_overview(offload_summary_df: pd.DataFrame,
                            out_dir: Path,
                            dataset_label: str = "") -> Path:
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    path = out_dir / "estimator_overview.png"
    if offload_summary_df.empty:
        return path

    summary_df = _collapse_scenario_variants(offload_summary_df)
    if summary_df.empty:
        return path

    summary_df = summary_df.copy()
    summary_df["estimator_name"] = summary_df.apply(
        lambda row: _estimator_name_without_offloader(
            row.get("estimator", ""),
            offloader_id=row.get("offloader_id"),
            policy_id=row.get("policy_id"),
        ),
        axis=1,
    )
    summary_df["stage"] = summary_df["estimator_name"].astype(str).map(_stage_from_name)
    summary_df["policy"] = summary_df.apply(_approach_policy_label, axis=1)

    strategy_order = {
        "threshold": 0,
        "calibrated": 1,
        "fixed": 2,
        "sequential_csr_utility": 3,
        "sequential_csr": 4,
        "online_sqt": 5,
        "online_lvq": 6,
        "oracle": 7,
        "random": 8,
        "constant": 9,
    }
    summary_df["_strategy_rank"] = summary_df["strategy"].astype(str).map(
        lambda name: strategy_order.get(name, 99)
    )
    sort_cols = ["estimator_name", "_strategy_rank"]
    ranking_col = _ranking_auc_column(summary_df)
    if ranking_col is not None:
        sort_cols.append(ranking_col)
        summary_df = summary_df.sort_values(sort_cols, ascending=[True, True, False])
    else:
        summary_df = summary_df.sort_values(sort_cols, ascending=[True, True])

    table_df = summary_df.drop_duplicates(subset=["estimator_name"]).copy()
    table_df["proxy_metric"] = table_df["estimator_name"].astype(str).map(_extract_proxy_metric)
    ranking_col = _ranking_auc_column(table_df)
    if ranking_col is not None:
        table_df = table_df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
    else:
        table_df = table_df.sort_values(["stage", "estimator_name"]).reset_index(drop=True)

    display_cols = [
        ("estimator_name", "Estimator", None),
        ("stage", "Stage", None),
        ("proxy_metric", "Proxy Metric", None),
        ("policy", "Rep. policy", None),
        ("auc_0_5", "auc_0_5", True),
        ("auc_coco", "auc_coco", True),
        ("auc_coco50", "auc_coco50", True),
        ("peak_map", "Peak", True),
    ]
    display_cols = [item for item in display_cols if item[0] in table_df.columns]
    if len(display_cols) <= 2:
        return path

    return _render_overview_table(
        table_df,
        display_cols,
        path,
        title=_figure_title(
            "Estimator Overview",
            "Estimator-centric summary using one representative routing policy per estimator. Green cells are better column-wise.",
            dataset_label=dataset_label,
        ),
        subtitle="",
        footer="This table is estimator-level: ratio control and runtime are moved to the approach overview.",
        label_col="estimator_name",
    )


def plot_approach_overview(offload_summary_df: pd.DataFrame,
                           resource_df: pd.DataFrame,
                           out_dir: Path,
                           dataset_label: str = "") -> Path:
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    resource_df = _ensure_policy_columns(resource_df)
    path = out_dir / "approach_overview.png"
    if offload_summary_df.empty:
        return path

    from .analyse_helpers import _FOCUS_TARGET_RATIOS

    summary_df = _collapse_scenario_variants(offload_summary_df)
    if summary_df.empty:
        return path

    table_df = summary_df.copy()
    table_df["stage"] = table_df["estimator"].astype(str).map(_stage_from_name)
    table_df["proxy_metric"] = table_df["estimator"].astype(str).map(_extract_proxy_metric)
    table_df["policy"] = table_df.apply(_approach_policy_label, axis=1)

    target_ratios = list(_FOCUS_TARGET_RATIOS)
    latency_lookup: dict[float, dict[str, float]] = {tr: {} for tr in target_ratios}
    weak_latency = _resource_reference_value(
        resource_df, "estimated_end_to_end_ms_per_frame", "weak_model"
    )
    if not resource_df.empty and weak_latency and np.isfinite(weak_latency) and weak_latency > 0:
        rf = _collapse_scenario_variants(resource_df.copy(), metric="estimated_end_to_end_ms_per_frame",
                                         higher_is_better=False)
        if "target_ratio" in rf.columns:
            for tr in target_ratios:
                rf_copy = rf.copy()
                rf_copy["ratio_distance"] = np.abs(rf_copy["target_ratio"].astype(float) - tr)
                rf_focus = (
                    rf_copy.sort_values(["estimator", "ratio_distance"])
                    .groupby("estimator", as_index=False)
                    .first()
                )
                for _, row in rf_focus.iterrows():
                    raw = float(row.get("estimated_end_to_end_ms_per_frame", np.nan))
                    if np.isfinite(raw):
                        latency_lookup[tr][str(row["estimator"])] = raw / weak_latency

    for tr in target_ratios:
        col = f"latency_{int(tr * 100)}"
        table_df[col] = table_df["estimator"].astype(str).map(latency_lookup.get(tr, {}))

    ranking_col = _ranking_auc_column(table_df)
    if ranking_col is not None:
        table_df = table_df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
    else:
        table_df = table_df.sort_values(["stage", "estimator"]).reset_index(drop=True)

    display_cols = [
        ("estimator", "Approach", None),
        ("stage", "Stage", None),
        ("proxy_metric", "Proxy Metric", None),
        ("policy", "Policy", None),
        ("auc_0_5", "auc_0_5", True),
        ("auc_coco", "auc_coco", True),
        ("auc_coco50", "auc_coco50", True),
        ("peak_map", "Peak", True),
        ("mean_ratio_error", "Ratio err", False),
        ("latency_20", "20% lat", False),
        ("latency_40", "40% lat", False),
        ("latency_60", "60% lat", False),
        ("latency_80", "80% lat", False),
    ]
    display_cols = [item for item in display_cols if item[0] in table_df.columns]
    if len(display_cols) <= 3:
        return path

    latency_note = (
        f"Latency columns are normalized to weak-only cost ({weak_latency:.1f} ms/frame)."
        if weak_latency else
        "Latency columns are omitted when resource trade-off data is unavailable."
    )
    return _render_overview_table(
        table_df,
        display_cols,
        path,
        title=_figure_title(
            "Approach Overview",
            "Approach-level summary of routing quality, ratio control, and runtime. Green cells are better column-wise.\n"
            + latency_note,
            dataset_label=dataset_label,
        ),
        subtitle="",
        footer="Read across each row: AUC/Peak should go up, while Ratio err and normalized latency should go down.",
    )


def plot_proxy_metric_stability(proxy_df: pd.DataFrame, out_dir: Path,
                                dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "proxy_metric_stability.png"
    if proxy_df.empty or "proxy_metric" not in proxy_df.columns:
        return path

    draw_std = proxy_df[proxy_df["proxy_metric"].astype(str).str.endswith("_draw_std")].copy()
    sign = proxy_df[proxy_df["proxy_metric"].astype(str).str.endswith("_sign_consistency")].copy()
    rank = proxy_df[proxy_df["proxy_metric"].astype(str).str.endswith("_rank_consistency")].copy()
    cols = set(proxy_df.columns)
    alignment_col = "oracle_spearman" if "oracle_spearman" in cols else (
        "utility_spearman" if "utility_spearman" in cols else None
    )

    core_df = proxy_df.copy()
    metric_names = core_df["proxy_metric"].astype(str)
    core_df = core_df[
        ~metric_names.str.endswith(("_draw_std", "_sign_consistency", "_rank_consistency"))
        & ~metric_names.str.startswith(("gain_beneficial_", "gain_harmful_", "gain_neutral_", "edge_", "cloud_", "delta_", "dataset_"))
        & ~metric_names.str.contains("::", regex=False)
        & ~metric_names.isin({"gt_count"})
    ].copy()
    if alignment_col and "std" in core_df.columns and not core_df.empty:
        core_df["alignment"] = core_df[alignment_col].astype(float).abs()
        core_df["stability_score"] = core_df["alignment"] / (1.0 + core_df["std"].astype(float).clip(lower=0.0))
        core_df = core_df.sort_values("stability_score", ascending=False)

    if draw_std.empty and sign.empty and rank.empty and core_df.empty:
        return path

    panel_count = (2 if not core_df.empty and alignment_col and "std" in core_df.columns else 0) + (
        1 if not (draw_std.empty and sign.empty and rank.empty) else 0
    )
    fig, axes = plt.subplots(1, panel_count, figsize=(6.0 * panel_count, 5.2), squeeze=False)
    ax_iter = iter(axes[0])

    if not core_df.empty and alignment_col and "std" in core_df.columns:
        scatter_ax = next(ax_iter)
        scatter_df = core_df.head(12).copy().sort_values("stability_score", ascending=True)
        colors = [
            _proxy_family_color(_proxy_metric_family(name))
            for name in scatter_df["proxy_metric"]
        ]
        sizes = 90 + 180 * (1.0 - scatter_df.get("zero_rate", pd.Series(np.zeros(len(scatter_df)))).fillna(0.0).clip(0.0, 1.0))
        scatter_ax.scatter(
            scatter_df["std"].astype(float),
            scatter_df["alignment"].astype(float),
            s=sizes,
            c=colors,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
        )
        for _, row in scatter_df.iterrows():
            scatter_ax.annotate(
                _pretty_proxy_metric_label(row["proxy_metric"]),
                (float(row["std"]), float(row["alignment"])),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )
        scatter_ax.set_xlabel("Std of proxy values (lower = more stable)")
        scatter_ax.set_ylabel(f"|{alignment_col.replace('_', ' ')}| (higher = more informative)")
        scatter_ax.set_title("Stability vs informativeness\nTop proxy candidates by stability-adjusted alignment")
        scatter_ax.grid(alpha=0.25, ls="--")

        bar_ax = next(ax_iter)
        leaderboard = core_df.head(10).copy().sort_values("stability_score", ascending=True)
        leaderboard_colors = [
            _proxy_family_color(_proxy_metric_family(name))
            for name in leaderboard["proxy_metric"]
        ]
        y = np.arange(len(leaderboard))
        bar_ax.barh(y, leaderboard["stability_score"].astype(float), color=leaderboard_colors, edgecolor="white")
        bar_ax.set_yticks(y)
        bar_ax.set_yticklabels([
            _pretty_proxy_metric_label(name) for name in leaderboard["proxy_metric"]
        ])
        bar_ax.set_title("Best stability-adjusted proxies\nscore = alignment / (1 + std)")
        bar_ax.grid(axis="x", alpha=0.25, ls="--")

    if not (draw_std.empty and sign.empty and rank.empty):
        heat_ax = next(ax_iter)
        matrices: list[pd.DataFrame] = []
        if not draw_std.empty and "mean" in draw_std.columns:
            draw_panel = draw_std.copy()
            draw_panel["label"] = draw_panel["proxy_metric"].astype(str).str.removesuffix("_draw_std")
            draw_panel["display"] = 1.0 / (1.0 + draw_panel["mean"].astype(float).clip(lower=0.0))
            matrices.append(draw_panel[["label", "display", "mean"]].rename(
                columns={"display": "Draw stability", "mean": "Draw std raw"}
            ))
        for df, suffix, title in (
            (sign, "_sign_consistency", "Sign consistency"),
            (rank, "_rank_consistency", "Rank consistency"),
        ):
            if not df.empty and "mean" in df.columns:
                panel = df.copy()
                panel["label"] = panel["proxy_metric"].astype(str).str.removesuffix(suffix)
                matrices.append(panel[["label", "mean"]].rename(columns={"mean": title}))

        merged = None
        for panel in matrices:
            merged = panel if merged is None else merged.merge(panel, on="label", how="outer")
        merged = merged.fillna(np.nan).sort_values("label") if merged is not None else pd.DataFrame()
        metric_cols = [col for col in ("Draw stability", "Sign consistency", "Rank consistency") if col in merged.columns]
        if not merged.empty and metric_cols:
            values = merged[metric_cols].to_numpy(dtype=float)
            heat_ax.imshow(values, aspect="auto", cmap="YlGn", vmin=0.0, vmax=1.0)
            heat_ax.set_yticks(np.arange(len(merged)))
            heat_ax.set_yticklabels([
                _pretty_proxy_metric_label(label) for label in merged["label"]
            ])
            heat_ax.set_xticks(np.arange(len(metric_cols)))
            heat_ax.set_xticklabels(metric_cols, rotation=12, ha="right")
            raw_lookup = dict(zip(merged["label"], merged.get("Draw std raw", pd.Series(np.nan, index=merged.index))))
            for row_idx, label in enumerate(merged["label"]):
                for col_idx, metric_col in enumerate(metric_cols):
                    shown = values[row_idx, col_idx]
                    if not np.isfinite(shown):
                        continue
                    text = f"{shown:.2f}"
                    if metric_col == "Draw stability":
                        raw = raw_lookup.get(label, np.nan)
                        if np.isfinite(raw):
                            text = f"{shown:.2f}\n({raw:.3f})"
                    heat_ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=7, color="#1F2937")
            heat_ax.set_title("Explicit resampling diagnostics\nDraw stability is shown as 1 / (1 + draw std)")
        else:
            heat_ax.set_visible(False)

    fig.suptitle(
        _figure_title(
            "Proxy Metric Stability",
            "The old chart only showed degenerate resampling metrics. This version pairs stability with informativeness so the proxy trade-off is visible.",
            dataset_label=dataset_label,
        ),
        y=1.05,
    )
    fig.text(
        0.5, 0.02,
        "Key reading: a useful proxy should be low-variance under resampling and still align with oracle utility. Perfect resampling consistency alone is not enough.",
        ha="center", va="bottom", fontsize=8.5, color="#555555",
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_training_diagnostics(history_df: pd.DataFrame, out_dir: Path,
                              dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "training_diagnostics.png"
    if history_df.empty or "epoch" not in history_df.columns:
        return path

    numeric_cols = [
        col for col in history_df.columns
        if col not in {"epoch", "seed"} and pd.api.types.is_numeric_dtype(history_df[col])
    ]
    preferred = [col for col in numeric_cols if any(token in col.lower() for token in ("loss", "train", "val"))]
    series_cols = (preferred or numeric_cols)[:4]
    if not series_cols:
        return path

    grouped = history_df.groupby("epoch", as_index=False)[series_cols].mean(numeric_only=True)
    fig, axes = plt.subplots(1, len(series_cols), figsize=(4.6 * len(series_cols), 4.0), squeeze=False)
    for ax, col in zip(axes[0], series_cols):
        ax.plot(grouped["epoch"], grouped[col], marker="o", color="#4C9BE8", lw=1.8)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25, ls="--")
    axes[0][0].set_ylabel("Mean value across runs")
    fig.suptitle(
        _figure_title(
            "Training Diagnostics",
            "Look for steady improvement and small train/val gaps; flat or diverging curves point to underfitting or instability.",
            dataset_label=dataset_label,
        ),
        y=1.05,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_prediction_quality(pred_diag_df: pd.DataFrame, out_dir: Path,
                            dataset_label: str = "") -> Path:
    path = out_dir / "prediction_quality.png"
    return _prediction_diagnostics_figure(pred_diag_df, path, dataset_label=dataset_label)


def plot_selection_quality(selection_df: pd.DataFrame, offload_summary_df: pd.DataFrame,
                           out_dir: Path, dataset_label: str = "") -> Path:
    selection_df = _ensure_policy_columns(selection_df)
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    plt = _setup_matplotlib()
    path = out_dir / "selection_quality.png"
    if selection_df.empty:
        return path

    df = selection_df[selection_df["strategy"].astype(str) == "threshold"].copy()
    if df.empty:
        df = selection_df.copy()
    headline_names = _headline_estimators(offload_summary_df)
    df = _filter_headline_estimators(df, headline_names)
    if df.empty:
        return path

    styles = _build_estimator_styles(df)
    panels = [
        ("gain_capture_ratio_vs_oracle", "Gain capture vs oracle", "Closer to 1 means the selector captures most of the oracle's available gain."),
        ("harmful_offload_rate", "Harmful offload rate", "Lower means the selector avoids sending images that hurt the final metric."),
        ("oracle_overlap_recall", "Oracle overlap recall", "Higher means the selector picks many of the same frames the oracle would pick."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    for ax, (metric_col, title, subtitle) in zip(axes[0], panels):
        for name, group in df.groupby("estimator"):
            if metric_col not in group.columns:
                continue
            group = group.sort_values("target_ratio")
            style = styles.get(name, {"color": "#7A7A7A", "marker": "o", "linestyle": "-"})
            ax.plot(
                group["target_ratio"].astype(float),
                group[metric_col].astype(float),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                lw=1.8,
                markersize=4.5,
                alpha=0.85,
                label=_compact_estimator_label(name),
            )
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xlabel("Target ratio")
        ax.grid(alpha=0.25, ls="--")
    axes[0, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=min(4, max(1, len(labels))), frameon=False)
    fig.suptitle(
        _figure_title(
            "Selection Quality",
            "These curves explain why some estimators win: more oracle gain captured and fewer harmful offloads.",
            dataset_label=dataset_label,
        ),
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_budget_stability_traces(trace_df: pd.DataFrame, offload_summary_df: pd.DataFrame,
                                 out_dir: Path, dataset_label: str = "",
                                 focus_ratios: list[float] | None = None) -> Path:
    trace_df = _ensure_policy_columns(trace_df)
    offload_summary_df = _ensure_policy_columns(offload_summary_df)
    plt = _setup_matplotlib()
    path = out_dir / "budget_stability_traces.png"
    if trace_df.empty or "strategy" not in trace_df.columns:
        return path

    summary = offload_summary_df.copy()
    if summary.empty:
        return path
    fixed_mask = trace_df["strategy"].astype(str) == "fixed"
    if fixed_mask.any():
        non_fixed_ratios = set(
            trace_df.loc[~fixed_mask, "target_ratio"].dropna().astype(float).unique().tolist()
        )
        if non_fixed_ratios:
            fixed_overlap_mask = fixed_mask & trace_df["target_ratio"].astype(float).isin(non_fixed_ratios)
            trace_df = trace_df[~fixed_overlap_mask].copy()
    if "auc_0_5" in summary.columns:
        summary = summary.sort_values("auc_0_5", ascending=False)
    chosen = summary.groupby("strategy", as_index=False).first()
    chosen_pairs = set(
        chosen[["estimator", "strategy"]].itertuples(index=False, name=None)
    )
    trace_df = trace_df[
        trace_df[["estimator", "strategy"]].apply(tuple, axis=1).isin(chosen_pairs)
    ].copy()
    if trace_df.empty:
        return path

    available_ratios = sorted(trace_df["target_ratio"].dropna().astype(float).unique().tolist())
    # Exclude degenerate ratio 0.0 (never-offload); keep all real budget targets
    plot_ratios = _resolve_trace_plot_ratios(available_ratios, focus_ratios)
    if not plot_ratios:
        return path

    n = len(plot_ratios)
    # The canonical trace view uses four focus budgets, which should render as
    # a balanced 2x2 grid instead of a sparse 3-column layout.
    ncols = 2 if n == 4 else min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.1 * ncols, 4.3 * nrows), squeeze=False)
    strategy_names = sorted(trace_df["strategy"].astype(str).unique().tolist())
    colors = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    styles = {
        name: {
            "color": colors[idx % len(colors)],
            "linestyle": ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))][idx % 6],
        }
        for idx, name in enumerate(strategy_names)
    }
    # Hide any surplus axes (when grid cells > number of ratios)
    flat_axes = axes.flatten()
    for ax in flat_axes[len(plot_ratios):]:
        ax.set_visible(False)
    for idx, (ax, (display_ratio, actual_ratio)) in enumerate(zip(flat_axes, plot_ratios)):
        subset = trace_df[np.isclose(trace_df["target_ratio"].astype(float), actual_ratio)].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()  # separate axis for budget debt
        for (estimator, strategy), group in subset.groupby(["estimator", "strategy"]):
            group = group.sort_values("step")
            x = group["step"].astype(float).to_numpy()
            x = x / max(x.max(), 1.0)
            label = f"{_trace_strategy_label(strategy)} | {_compact_estimator_label(estimator)}"
            style = styles.get(strategy, {"color": "#444444", "linestyle": "-"})
            ax.plot(x, group["cumulative_ratio"].astype(float), color=style["color"], lw=1.8, ls=style["linestyle"], label=label)
            debt_vals = group["budget_debt"].astype(float)
            ax2.plot(x, debt_vals, color=style["color"], lw=0.9, ls="--", alpha=0.5)
        ax.axhline(display_ratio, color="#444444", lw=1.0, ls=":")
        # Clip cumulative ratio axis to reasonable range
        ax.set_ylim(0, min(max(display_ratio, actual_ratio) * 2.1, 1.05))
        if idx % ncols == 0:
            ax.set_ylabel("Cumulative ratio", fontsize=8)
        else:
            ax.set_ylabel("")
        if (idx % ncols == ncols - 1) or (idx == len(plot_ratios) - 1):
            ax2.set_ylabel("Budget debt", fontsize=8, color="#888888")
        else:
            ax2.set_ylabel("")
        # Clip debt axis to 99th percentile
        all_debt = subset["budget_debt"].astype(float).dropna()
        if not all_debt.empty:
            debt_hi = max(float(np.percentile(all_debt, 99)), 1.0)
            ax2.set_ylim(-debt_hi * 0.1, debt_hi * 1.1)
        ax2.tick_params(axis="y", labelsize=7, colors="#888888")
        title = f"Target budget {display_ratio * 100:.0f}%"
        if not np.isclose(display_ratio, actual_ratio):
            title += f"\n(trace from {actual_ratio * 100:.1f}%)"
        elif "source_target_ratio" in subset.columns:
            sources = subset["source_target_ratio"].dropna().astype(float).unique().tolist()
            if len(sources) == 1 and not np.isclose(sources[0], display_ratio):
                title += f"\n(trace from {sources[0] * 100:.1f}%)"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Stream progress")
        ax.grid(alpha=0.25, ls="--")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        legend_ncol = 3 if len(labels) >= 6 else min(len(labels), 2)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=max(1, legend_ncol),
            frameon=False,
            columnspacing=1.2,
            handlelength=2.8,
        )
    fig.suptitle(
        _figure_title(
            "Budget Stability Traces",
            "Solid lines show how quickly a policy settles to the requested budget; dashed lines show accumulated budget debt.\n"
            "Faster convergence and smaller debt magnitude are better.",
            dataset_label=dataset_label,
        ),
        y=0.98,
    )
    fig.subplots_adjust(
        top=0.72,
        bottom=0.24 if handles else 0.12,
        wspace=0.30,
        hspace=0.35,
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


_COMPONENT_DISPLAY_NAMES = {
    "cls": "Classification",
    "loc": "Localisation",
    "both": "Cls + Loc",
    "dup": "Duplicate",
    "bg": "Background",
    "miss": "Missed",
    "ap75": "AP@75",
    "ap90": "AP@90",
    "center_offset": "Center offset",
    "category_precision": "Category prec.",
    "overall": "Overall",
}


def _pretty_component_name(name: str) -> str:
    if name in _COMPONENT_DISPLAY_NAMES:
        return _COMPONENT_DISPLAY_NAMES[name]
    if name.startswith("component_"):
        return f"Comp {name.split('_', 1)[1]}"
    if name.startswith("tau_"):
        return f"\u03c4={name[4:]}"
    return name


def plot_component_diagnostics(component_df: pd.DataFrame, offload_summary_df: pd.DataFrame,
                               out_dir: Path, dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "component_diagnostics.png"
    if component_df.empty or "diagnostic_family" not in component_df.columns:
        return path

    # Use the full structured-diagnostic table here. Headline filtering works
    # for summary charts, but it can collapse this figure to a single row when
    # only one structured estimator survives the offloading-based selection.
    df = component_df.copy()

    family_specs = {
        "vector": ("spearman_rho", "Vector component correlation",
                   "Spearman rank correlation per error component.\nHigher = estimator ranks frames correctly for that error type."),
        "survival": ("accuracy", "Survival head accuracy",
                     "Accuracy of the survival (will-it-help?) classifier head.\nHigher = better at predicting whether offloading helps."),
        "ordinal": ("exact_bucket_accuracy", "Ordinal bucket accuracy",
                    "Fraction of frames placed in the correct gain bucket.\nHigher = finer-grained gain magnitude prediction."),
    }
    families = [fam for fam in family_specs if fam in df["diagnostic_family"].astype(str).unique()]
    if not families:
        return path

    n_fam = len(families)
    fig, axes = plt.subplots(1, n_fam, figsize=(6.0 * n_fam, max(5.0, df["estimator"].nunique() * 0.5 + 2.5)),
                             squeeze=False)
    for ax, family in zip(axes[0], families):
        value_col, title, explanation = family_specs[family]
        if value_col not in df.columns:
            ax.set_visible(False)
            continue
        family_df = df[df["diagnostic_family"].astype(str) == family].copy()
        pivot = family_df.pivot_table(index="estimator", columns="component", values=value_col, aggfunc="mean")
        if pivot.empty:
            ax.set_visible(False)
            continue
        # Rename columns and rows for readability
        pivot.columns = [_pretty_component_name(c) for c in pivot.columns]
        pivot.index = [_compact_estimator_label(name) for name in pivot.index]
        # Annotate cells with values
        data = pivot.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu", interpolation="nearest")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    text_color = "white" if val > (np.nanmax(data) + np.nanmin(data)) / 2 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
        ax.set_title(f"{title}", fontsize=10, pad=8)
        ax.set_xlabel(explanation, fontsize=7.5, labelpad=8, color="#555555")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label=value_col.replace("_", " "))

    fig.suptitle(
        _figure_title(
            "Component Diagnostics",
            "Per-component accuracy of structured estimators. Each column is one error component;\n"
            "a dark (low) cell pinpoints a specific weakness rather than a global failure.\n"
            "Rows = estimators, columns = detection error components (TIDE-style decomposition).",
            dataset_label=dataset_label,
        ),
        y=1.06,
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_slice_opportunity(slice_opp_df: pd.DataFrame, out_dir: Path,
                           dataset_label: str = "") -> Path:
    plt = _setup_matplotlib()
    path = out_dir / "slice_opportunity.png"
    if slice_opp_df.empty:
        return path

    df = slice_opp_df.copy()
    df["label"] = df["slice_name"].astype(str) + " / " + df["slice_value"].astype(str)
    upside = df.sort_values("headroom_mean", ascending=True).tail(12)
    hard = df.sort_values("harmful_rate", ascending=True).tail(12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(np.arange(len(upside)), upside["headroom_mean"].to_numpy(dtype=float), color="#4C9BE8", edgecolor="white")
    axes[0].set_yticks(np.arange(len(upside)))
    axes[0].set_yticklabels(upside["label"].tolist())
    axes[0].set_title("High-upside slices\nHigher headroom means more value if the selector gets these right.")
    axes[0].grid(axis="x", alpha=0.25, ls="--")

    axes[1].barh(np.arange(len(hard)), hard["harmful_rate"].to_numpy(dtype=float), color="#E8734C", edgecolor="white")
    axes[1].set_yticks(np.arange(len(hard)))
    axes[1].set_yticklabels(hard["label"].tolist())
    axes[1].set_title("Risky slices\nHigher harmful rate means offloading mistakes are easier to make here.")
    axes[1].grid(axis="x", alpha=0.25, ls="--")

    fig.suptitle(
        _figure_title(
            "Slice Opportunity",
            "Read this as opportunity vs risk: left shows where selective routing can pay off, right shows where it can go wrong.",
            dataset_label=dataset_label,
        ),
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
