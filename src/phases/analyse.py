"""Phase 5 – Analyse: load saved metrics, generate charts and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple

import pandas as pd

from config import PipelineConfig

from .. import log

# Re-export submodule contents for backward compatibility
from .analyse_helpers import (  # noqa: F401
    _CHART_BASELINE_KEYS,
    _VIRTUAL_NAMES,
    _ensure_policy_columns,
    _collapse_scenario_variants,
    _estimator_model_key,
    _setup_matplotlib,
    _FOCUS_TARGET_RATIOS,
    _resolve_trace_plot_ratios,
    _prepare_scenario_method_comparison,
)
from .analyse_charts import (  # noqa: F401
    plot_metric_comparison,
    plot_timing,
    plot_complexity,
    plot_offloading_3panel,
    plot_dataset_wide_map,
    generate_report,
    plot_ratio_accuracy,
    plot_proxy_metric_distributions,
    plot_empirical_cdf,
    plot_estimator_regression,
    plot_calibration_diagnostics,
    plot_slice_heatmap,
)
from .analyse_scenarios import (  # noqa: F401
    plot_scenario_comparison,
    plot_scenario_radar,
    plot_scenario_perf_heatmap,
    plot_scenario_weight_profiles,
    plot_scenario_sensitivity,
    plot_scenario_offloading_curves,
    plot_estimator_overview,
    plot_approach_overview,
    plot_proxy_metric_stability,
    plot_training_diagnostics,
    plot_prediction_quality,
    plot_selection_quality,
    plot_budget_stability_traces,
    plot_component_diagnostics,
    plot_slice_opportunity,
)
from .analyse_resource import (  # noqa: F401
    plot_resource_tradeoff,
    plot_resource_frontier_comparison,
    plot_resource_frontier_summary,
    export_paper_tables,
)


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


def _run_scenario_eval(cfg: PipelineConfig, ds_label: str = "") -> None:
    """Generate fine-grained scenario charts into scenario_eval_dir."""
    eval_dir = cfg.output.scenario_eval_dir
    summary_path = eval_dir / "scenario_offloading_summary.csv"
    if not summary_path.exists():
        return  # No scenario data — nothing to do

    def _load(path: Path) -> pd.DataFrame:
        try:
            return _ensure_policy_columns(pd.read_csv(path))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame()

    scen_summary = _load(summary_path)
    if scen_summary.empty:
        return

    scen_offload  = _load(eval_dir / "scenario_offloading.csv")
    profiles_df   = _load(eval_dir / "scenario_profiles.csv")

    # Merge in non-scenario baselines from main metrics dir for context
    main_summary  = _load(cfg.output.metrics_dir / "offloading_summary.csv")
    main_offload  = _load(cfg.output.metrics_dir / "offloading_results.csv")

    def _non_scenario(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "scenario" not in df.columns:
            return df
        return df[df["scenario"].fillna("").astype(str) == ""].copy()

    combined_summary = (
        pd.concat([scen_summary, _non_scenario(main_summary)], ignore_index=True)
        if not main_summary.empty else scen_summary
    )
    combined_offload = (
        pd.concat([scen_offload, _non_scenario(main_offload)], ignore_index=True)
        if not scen_offload.empty and not main_offload.empty
        else scen_offload
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    log.section("Scenario Evaluation Charts")
    outputs = []

    for fn, label, kwargs in [
        (plot_scenario_perf_heatmap,    "Scenario perf heatmap",   {"offload_summary_df": combined_summary}),
        (plot_scenario_weight_profiles, "Scenario weight profiles", {"scenario_profiles_df": profiles_df}),
        (plot_scenario_sensitivity,     "Scenario sensitivity",     {"offload_summary_df": combined_summary}),
        (plot_scenario_offloading_curves, "Scenario offloading curves", {"offload_df": combined_offload}),
    ]:
        try:
            p = fn(**kwargs, out_dir=eval_dir, dataset_label=ds_label)
            outputs.append((label, str(p)))
        except Exception as e:
            log.warn(f"{label} failed: {e}")

    for label, path in outputs:
        log.arrow(f"{label}: {path}")


# ---- public API -------------------------------------------------------

def run(cfg: PipelineConfig) -> None:
    """Execute the analyse phase."""
    with log.phase_timer(5):
        ds_label = cfg.dataset.name.upper()

        metrics_path = cfg.output.metrics_dir / "estimator_metrics.csv"
        offload_path = cfg.output.metrics_dir / "offloading_results.csv"

        if not metrics_path.exists():
            raise FileNotFoundError(
                "Metrics not found. Run 'evaluate' phase first.")

        metrics_df = pd.read_csv(metrics_path)
        charts_dir = cfg.output.charts_dir
        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_allowed_names, chart_allowed_keys = _chart_estimator_allowlist(cfg)
        chart_estimator_allowed = lambda name: _is_chart_estimator_allowed(
            name, chart_allowed_names, chart_allowed_keys
        )
        chart_metrics_df = _filter_chart_estimators(
            metrics_df, chart_allowed_names, chart_allowed_keys
        )
        # Collapse scenario variants for non-scenario charts
        chart_metrics_df = _collapse_scenario_variants(chart_metrics_df)

        # Remove deprecated charts to avoid stale outputs from previous runs.
        for deprecated in ("comparison_r2.png",
                           "comparison_spearman_rho.png",
                           "empirical_cdf.png", "timing.png",
                           "overview_summary.png"):
            p = charts_dir / deprecated
            if p.exists():
                p.unlink()
        deprecated_csv = cfg.output.metrics_dir / "comparison_spearman_rho.csv"
        if deprecated_csv.exists():
            deprecated_csv.unlink()

        ds_summary_path = cfg.output.metrics_dir / "dataset_summary.json"
        ds_summary = None
        if ds_summary_path.exists():
            with open(ds_summary_path) as f:
                ds_summary = json.load(f)

        # Pre-load offload summary for headline selection in charts
        offload_summary_path = cfg.output.metrics_dir / "offloading_summary.csv"
        offload_summary_df = (_ensure_policy_columns(pd.read_csv(offload_summary_path))
                              if offload_summary_path.exists()
                              else pd.DataFrame())
        thresh_path = cfg.output.metrics_dir / "offloading_threshold_results.csv"

        log.section("Charts")
        chart_outputs = []
        try:
            corr_metric = "spearman_rho" if "spearman_rho" in metrics_df.columns else "r2"
            if corr_metric != "spearman_rho" or "base_model" not in chart_metrics_df.columns:
                corr_title = (
                    f"Spearman Correlation Comparison [{ds_label}]"
                    if corr_metric == "spearman_rho"
                    else f"R\u00b2 Comparison [{ds_label}]"
                )
                spearman_csv = (cfg.output.metrics_dir / "comparison_spearman_rho.csv"
                                if corr_metric == "spearman_rho" else None)
                c1 = plot_metric_comparison(chart_metrics_df, corr_metric, charts_dir,
                                            title=corr_title, csv_path=spearman_csv)
                chart_outputs.append(("Correlation", str(c1)))
                if spearman_csv is not None:
                    chart_outputs.append(("Correlation CSV", str(spearman_csv)))

            c6 = plot_complexity(chart_metrics_df, charts_dir, dataset_label=ds_label,
                                 dataset_summary=ds_summary)
            chart_outputs.append(("Complexity", str(c6)))

            if offload_path.exists():
                try:
                    offload_df = _ensure_policy_columns(pd.read_csv(offload_path))
                    chart_offload_df = _filter_chart_estimators(
                        offload_df, chart_allowed_names, chart_allowed_keys
                    )
                    forced_path = cfg.output.metrics_dir / "offloading_forced_results.csv"
                    if forced_path.exists():
                        forced_df = _ensure_policy_columns(pd.read_csv(forced_path))
                        chart_forced_df = _filter_chart_estimators(
                            forced_df, chart_allowed_names, chart_allowed_keys
                        )
                    else:
                        chart_forced_df = chart_offload_df

                    actual_plot_df = pd.DataFrame()
                    if thresh_path.exists():
                        try:
                            thresh_df_for_plot = _ensure_policy_columns(pd.read_csv(thresh_path))
                            actual_plot_df = _filter_chart_estimators(
                                thresh_df_for_plot, chart_allowed_names, chart_allowed_keys
                            )
                        except pd.errors.EmptyDataError:
                            actual_plot_df = pd.DataFrame()
                    extra_actual = chart_offload_df[
                        chart_offload_df["estimator"].astype(str).isin({"oracle", "random"})
                    ].copy()
                    if not extra_actual.empty:
                        extra_actual["actual_ratio"] = extra_actual["ratio"].astype(float)
                        actual_plot_df = (
                            pd.concat([actual_plot_df, extra_actual], ignore_index=True, sort=False)
                            if not actual_plot_df.empty else extra_actual
                        )

                    c3 = plot_offloading_3panel(chart_forced_df, chart_metrics_df, charts_dir,
                                                dataset_label=ds_label,
                                                offload_summary_df=offload_summary_df,
                                                actual_df=actual_plot_df)
                    chart_outputs.append(("Offloading (dual-view)", str(c3)))
                except pd.errors.EmptyDataError:
                    log.warn("offloading_results.csv is empty, skipping offload chart")
            else:
                offload_df = pd.DataFrame()

            if ds_summary:
                c5 = plot_dataset_wide_map(ds_summary_path, charts_dir,
                                           chart_metrics_df, dataset_label=ds_label)
                chart_outputs.append(("Dataset-wide AP", str(c5)))

            # Offloader threshold charts
            if thresh_path.exists():
                try:
                    thresh_df = _ensure_policy_columns(pd.read_csv(thresh_path))
                    ratio_allowed_names, ratio_allowed_keys = _configured_approach_allowlist(cfg)
                    chart_thresh_df = _filter_chart_estimators(
                        thresh_df, ratio_allowed_names, ratio_allowed_keys
                    )
                    c7 = plot_ratio_accuracy(chart_thresh_df, charts_dir,
                                             dataset_label=ds_label,
                                             offload_summary_df=offload_summary_df)
                    chart_outputs.append(("Ratio accuracy", str(c7)))
                except pd.errors.EmptyDataError:
                    log.warn("Threshold results CSV is empty, skipping offloader charts")

            # Prediction distribution charts
            dist_path = cfg.output.metrics_dir / "proxy-metric_distributions.npz"
            if dist_path.exists():
                c9 = plot_proxy_metric_distributions(dist_path, charts_dir,
                                                   dataset_label=ds_label,
                                                   estimator_allowed=chart_estimator_allowed)
                chart_outputs.append(("Prediction distributions", str(c9)))

            # Estimator-level charts (grouped by base_model)
            if "base_model" in chart_metrics_df.columns:
                c_er = plot_estimator_regression(chart_metrics_df, charts_dir,
                                                 dataset_label=ds_label)
                chart_outputs.append(("Estimator regression", str(c_er)))

            ranking_path = cfg.output.metrics_dir / "ranking_metrics.csv"
            ranking_df = pd.read_csv(ranking_path) if ranking_path.exists() else pd.DataFrame()

            pred_diag_path = cfg.output.metrics_dir / "prediction_diagnostics.csv"
            if pred_diag_path.exists():
                pred_diag_df = pd.read_csv(pred_diag_path)
                c_cal = plot_calibration_diagnostics(pred_diag_df, charts_dir,
                                                     dataset_label=ds_label)
                chart_outputs.append(("Calibration diagnostics", str(c_cal)))
                c_pred_quality = plot_prediction_quality(
                    _collapse_scenario_variants(pred_diag_df, metric="calibration_gap",
                                                higher_is_better=False),
                    charts_dir, dataset_label=ds_label)
                chart_outputs.append(("Prediction quality", str(c_pred_quality)))
            else:
                pred_diag_df = pd.DataFrame()

            slice_path = cfg.output.metrics_dir / "slice_metrics.csv"
            if slice_path.exists():
                slice_df = pd.read_csv(slice_path)
                c_slice = plot_slice_heatmap(slice_df, charts_dir,
                                             dataset_label=ds_label)
                chart_outputs.append(("Slice heatmap", str(c_slice)))
            else:
                slice_df = pd.DataFrame()

            slice_opp_path = cfg.output.metrics_dir / "slice_opportunity.csv"
            if slice_opp_path.exists():
                slice_opp_df = pd.read_csv(slice_opp_path)
                c_slice_opp = plot_slice_opportunity(slice_opp_df, charts_dir,
                                                     dataset_label=ds_label)
                chart_outputs.append(("Slice opportunity", str(c_slice_opp)))
            else:
                slice_opp_df = pd.DataFrame()

            if not offload_summary_df.empty:
                c_scenario = plot_scenario_comparison(offload_summary_df, charts_dir,
                                                      dataset_label=ds_label)
                chart_outputs.append(("Scenario comparison", str(c_scenario)))
                c_radar = plot_scenario_radar(offload_summary_df, charts_dir,
                                              dataset_label=ds_label)
                chart_outputs.append(("Scenario radar", str(c_radar)))

            proxy_stats_path = cfg.output.metrics_dir / "proxy_metric_stats.csv"
            if proxy_stats_path.exists():
                proxy_stats_df = pd.read_csv(proxy_stats_path)
                c_proxy_stability = plot_proxy_metric_stability(proxy_stats_df, charts_dir,
                                                                dataset_label=ds_label)
                chart_outputs.append(("Proxy metric stability", str(c_proxy_stability)))
            else:
                proxy_stats_df = pd.DataFrame()

            history_path = cfg.output.metrics_dir / "loss_component_history.csv"
            if history_path.exists():
                history_df = pd.read_csv(history_path)
                c_train = plot_training_diagnostics(history_df, charts_dir,
                                                    dataset_label=ds_label)
                chart_outputs.append(("Training diagnostics", str(c_train)))
            else:
                history_df = pd.DataFrame()

            selection_path = cfg.output.metrics_dir / "selection_diagnostics.csv"
            if selection_path.exists():
                selection_df = _ensure_policy_columns(pd.read_csv(selection_path))
                c_sel = plot_selection_quality(selection_df, offload_summary_df, charts_dir,
                                               dataset_label=ds_label)
                chart_outputs.append(("Selection quality", str(c_sel)))
            else:
                selection_df = pd.DataFrame()

            if not offload_summary_df.empty:
                c_estimator_overview = plot_estimator_overview(
                    offload_summary_df, charts_dir, dataset_label=ds_label
                )
                chart_outputs.append(("Estimator overview", str(c_estimator_overview)))

            resource_path = cfg.output.metrics_dir / "resource_tradeoff.csv"
            if resource_path.exists():
                resource_df = _ensure_policy_columns(pd.read_csv(resource_path))
                c_resource = plot_resource_tradeoff(resource_df, offload_summary_df, charts_dir,
                                                    dataset_label=ds_label)
                chart_outputs.append(("Resource trade-off", str(c_resource)))
                c_resource_frontier = plot_resource_frontier_comparison(
                    resource_df, offload_summary_df, charts_dir, dataset_label=ds_label
                )
                chart_outputs.append(("Resource frontier", str(c_resource_frontier)))
                c_resource_frontier_summary = plot_resource_frontier_summary(
                    resource_df, offload_summary_df, charts_dir, dataset_label=ds_label
                )
                chart_outputs.append(("Resource frontier summary", str(c_resource_frontier_summary)))
                c_approach_overview = plot_approach_overview(
                    offload_summary_df, resource_df, charts_dir, dataset_label=ds_label
                )
                chart_outputs.append(("Approach overview", str(c_approach_overview)))
            else:
                resource_df = pd.DataFrame()

            trace_path = cfg.output.metrics_dir / "trace_diagnostics.csv"
            if trace_path.exists():
                trace_df = _ensure_policy_columns(pd.read_csv(trace_path))
                c_trace = plot_budget_stability_traces(trace_df, offload_summary_df, charts_dir,
                                                       dataset_label=ds_label,
                                                       focus_ratios=list(_FOCUS_TARGET_RATIOS))
                chart_outputs.append(("Budget stability traces", str(c_trace)))
            else:
                trace_df = pd.DataFrame()

            component_path = cfg.output.metrics_dir / "component_diagnostics.csv"
            if component_path.exists():
                component_df = pd.read_csv(component_path)
                c_component = plot_component_diagnostics(component_df, offload_summary_df,
                                                         charts_dir, dataset_label=ds_label)
                chart_outputs.append(("Component diagnostics", str(c_component)))
            else:
                component_df = pd.DataFrame()

            stats_path = cfg.output.metrics_dir / "statistical_summary.csv"
            stats_df = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
            for exported in export_paper_tables(metrics_df, ranking_df, offload_summary_df,
                                                stats_df, cfg.output.metrics_dir):
                chart_outputs.append(("Paper table", str(exported)))
        except Exception as e:
            log.fail(f"Chart generation failed: {e}")
            import traceback
            traceback.print_exc()

        for label, path in chart_outputs:
            log.arrow(f"{label}: {path}")

        log.section("Report")
        report = generate_report(metrics_df, ds_summary, dataset_label=ds_label)
        report_path = Path(cfg.output.base_dir) / "report.txt"
        report_path.write_text(report)
        log.arrow(str(report_path))

        # Proxy-metric distribution analysis (cached)
        try:
            from .reward_analysis import analyse_proxy_metric_distributions
            analyse_proxy_metric_distributions(cfg)
        except Exception as e:
            log.fail(f"Proxy-metric distribution analysis failed: {e}")
            import traceback
            traceback.print_exc()

        # Fine-grained scenario evaluation
        try:
            _run_scenario_eval(cfg, ds_label)
        except Exception as e:
            log.fail(f"Scenario evaluation charts failed: {e}")
            import traceback
            traceback.print_exc()
