"""Sub-phase 2b: load CSVs, extract features, split train/test, save npz."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import PipelineConfig
from config.scenarios import (
    DEFAULT_LCER3_SCENARIO_WEIGHTS,
    LCER3_SCENARIO_COMPONENTS,
    LCER6_TO_LCER3_PROJECTION,
    LCER_SCENARIO_COMPONENTS,
    SCENARIO_COMPONENTS,
    default_balanced_scenario_weights,
    normalize_scenario_weight_map,
)

from .. import log
from ..conditional_rewards import (
    CEORIC_DEFAULT_K,
    CEORIC_DEFAULT_TAU,
    smooth_reward_vectors,
)
from ..features import (
    EDGE_FEATURES,
    build_rule_feature_matrix,
    decode_float_sequence,
    topk_proposal_feature_columns,
)

from .prepare_config import (
    _required_prepare_inputs,
    _required_proxy_families,
    _prepare_cache_signature,
)
from .prepare_derive import (
    _gain_state_label,
    _gain_state_indicators,
    _load_srrm_matrices,
    _get_dataset_wide_oric_cached,
)
from .prepare_transforms import (
    LCER_TAU_GRID,
    _apply_moric,
    _apply_moric_plus,
    _apply_moric_star,
    _apply_phi_moric,
    _apply_sigmoric,
    _build_csr_targets,
    _compute_conditional_reward_neighborhood,
    _extract_finegrained_matrix,
    _extract_lcer_matrix,
    _fit_lcer_beta,
    _fit_moric_plus_reference,
    _fit_moric_reference,
    _fit_moric_star_reference,
    _fit_phi_moric_reference,
    _fit_sigmoric_reference,
    _resolve_split_indices,
    _scenario_metadata_payload,
)


def load_csvs(data_dir: Path, sample_fraction: float = 1.0,
              seed: int = 42) -> pd.DataFrame:
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {data_dir}")

    if sample_fraction < 1.0:
        rng = np.random.RandomState(seed)
        split_groups: dict[str, list[Path]] = {}
        unlabeled: list[Path] = []

        for csv_path in csvs:
            split_label = ""
            try:
                split_df = pd.read_csv(csv_path, usecols=["split"], nrows=1)
                if not split_df.empty:
                    split_label = str(split_df.iloc[0]["split"]).strip().lower()
            except (ValueError, pd.errors.EmptyDataError, KeyError):
                split_label = ""

            if split_label in {"train", "test"}:
                split_groups.setdefault(split_label, []).append(csv_path)
            else:
                unlabeled.append(csv_path)

        sampled: list[Path] = []
        if split_groups and not unlabeled:
            for split_label, group in sorted(split_groups.items()):
                n_group = max(1, int(round(len(group) * sample_fraction)))
                n_group = min(n_group, len(group))
                sampled.extend(rng.choice(group, size=n_group, replace=False).tolist())
            log.info(
                f"Sampled {len(sampled)} videos ({sample_fraction:.0%}) with split stratification"
            )
        else:
            n = max(1, int(len(csvs) * sample_fraction))
            sampled = rng.choice(csvs, size=n, replace=False).tolist()
            log.info(f"Sampled {len(sampled)} videos ({sample_fraction:.0%})")

        csvs = sorted(sampled)

    frames = []
    it = tqdm(csvs, desc="  Loading CSVs") if len(csvs) > 10 else csvs
    for p in it:
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {log.fmt_count(len(df))} frames from "
             f"{log.fmt_count(len(csvs))} videos")
    return df


def extract_features(df: pd.DataFrame, cfg: Optional[PipelineConfig] = None):
    required_inputs = _required_prepare_inputs(cfg) if cfg is not None else {
        "proposal_topk", "proposal_full", "image_paths",
    }
    need_topk = "proposal_topk" in required_inputs
    need_dcsb = "proposal_full" in required_inputs

    available = [c for c in EDGE_FEATURES if c in df.columns]
    X = df[available].fillna(0).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    proposal_cols = [
        c for c in topk_proposal_feature_columns("edge")
        if c in df.columns
    ] if need_topk else []
    if need_topk and proposal_cols:
        X_top25 = df[proposal_cols].fillna(0.0).to_numpy(dtype=np.float32)
    elif need_topk:
        X_top25 = np.zeros(
            (len(df), len(topk_proposal_feature_columns("edge"))),
            dtype=np.float32,
        )
    else:
        X_top25 = np.zeros((len(df), 0), dtype=np.float32)
    if need_dcsb and {"edge_rule_conf_seq", "edge_rule_area_seq"}.issubset(df.columns):
        conf_sequences = [
            decode_float_sequence(v) for v in df["edge_rule_conf_seq"].tolist()
        ]
        area_sequences = [
            decode_float_sequence(v) for v in df["edge_rule_area_seq"].tolist()
        ]
        X_dcsb = build_rule_feature_matrix(conf_sequences, area_sequences)
    elif need_dcsb:
        import warnings
        warnings.warn(
            "Derived CSVs do not contain the full DCSB proposal stream. "
            "Re-run the derive/prepare phases to build a paper-faithful DCSB baseline.",
            RuntimeWarning,
            stacklevel=2,
        )
        X_dcsb = np.zeros((len(df), 0), dtype=np.float32)
    else:
        X_dcsb = np.zeros((len(df), 0), dtype=np.float32)

    if not X.any():
        import warnings
        warnings.warn(
            "All tabular features are zero! This likely means the "
            "conf_threshold filtered out all detections. Check your "
            "dataset config (conf_threshold, detection_split).",
            RuntimeWarning,
            stacklevel=2,
        )

    image_paths = df["image_path"].fillna("").tolist()
    # Handle backward compatibility for column names
    y_gain_11pt = df.get("gain_11pt", df.get("map_gain", pd.Series(np.zeros(len(df))))).values
    y_gain_allpoint = df.get("gain_allpoint", df.get("map_gain_coco50", pd.Series(y_gain_11pt))).values
    y_gain_coco = df.get("gain_coco", df.get("map_gain_coco", pd.Series(y_gain_11pt))).values
    y_count_gain_05 = df.get("count_gain_05", pd.Series(np.zeros(len(df)))).values
    y_gt_count = df.get("gt_count", pd.Series(np.zeros(len(df)))).values

    y_oric_11pt = df.get("oric_11pt", pd.Series(y_gain_11pt)).values
    y_oric_allpoint = df.get("oric_allpoint", pd.Series(y_gain_allpoint)).values
    y_oric_coco = df.get("oric_coco", pd.Series(y_gain_coco)).values
    y_lcer_vec_11pt = _extract_lcer_matrix(df, "11pt")
    y_lcer_vec_allpoint = _extract_lcer_matrix(df, "allpoint")
    y_lcer_vec_coco = _extract_lcer_matrix(df, "coco")
    y_finegrained_vec_coco = _extract_finegrained_matrix(df, "coco")

    y_bwd = df.get("bwd", pd.Series(np.zeros(len(df)))).values
    y_entropy = df.get("entropy", pd.Series(np.zeros(len(df)))).values
    y_img_complexity = df.get("img_complexity", pd.Series(np.zeros(len(df)))).values

    extra_targets = {
        "edge_tp": df.get("edge_tp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "edge_fp": df.get("edge_fp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "edge_fn": df.get("edge_fn", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "cloud_tp": df.get("cloud_tp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "cloud_fp": df.get("cloud_fp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "cloud_fn": df.get("cloud_fn", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "delta_tp": df.get("delta_tp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "delta_fp": df.get("delta_fp", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "delta_fn": df.get("delta_fn", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        # Novel proxy metrics
        "rescue_ratio_50": df.get("rescue_ratio_50", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "rescue_count_50": df.get("rescue_count_50", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "precision_gain_50": df.get("precision_gain_50", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "fp_reduction_50": df.get("fp_reduction_50", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "f1_gain_50": df.get("f1_gain", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "worst_case_gain": df.get("worst_case_gain", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "high_iou_gain_75": df.get("high_iou_gain_75", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "edge_miss_rate": df.get("edge_miss_rate", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "edge_uncertainty": df.get("edge_uncertainty", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
        "conf_spread": df.get("conf_spread", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float32),
    }
    for suffix, values in (
        ("11pt", y_gain_11pt),
        ("allpoint", y_gain_allpoint),
        ("coco", y_gain_coco),
    ):
        labels = df.get(
            f"gain_state_{suffix}",
            pd.Series([_gain_state_label(v) for v in values], dtype=object),
        ).astype(str).to_numpy(dtype=object)
        beneficial, harmful, neutral = _gain_state_indicators(values)
        extra_targets[f"gain_state_{suffix}"] = labels
        extra_targets[f"gain_beneficial_{suffix}"] = beneficial
        extra_targets[f"gain_harmful_{suffix}"] = harmful
        extra_targets[f"gain_neutral_{suffix}"] = neutral
    for metric_name in ("oric_11pt", "oric_allpoint", "oric_coco"):
        for suffix in ("draw_std", "sign_consistency", "rank_consistency"):
            col = f"{metric_name}_{suffix}"
            if col in df.columns:
                extra_targets[col] = df[col].to_numpy(dtype=np.float32)

    edge_map = df["edge_map"].values
    cloud_map = df["cloud_map"].values
    edge_map_coco = df["edge_map_coco"].values if "edge_map_coco" in df.columns else edge_map
    cloud_map_coco = df["cloud_map_coco"].values if "cloud_map_coco" in df.columns else cloud_map
    edge_map_coco50 = df["edge_map_coco50"].values if "edge_map_coco50" in df.columns else edge_map
    cloud_map_coco50 = df["cloud_map_coco50"].values if "cloud_map_coco50" in df.columns else cloud_map
    video_names = df["video_name"].values
    frame_ids = df["frame_id"].values
    split_labels = (
        df["split"].fillna("").astype(str).values
        if "split" in df.columns
        else np.full(len(df), "", dtype=object)
    )

    return (X, X_top25, X_dcsb, image_paths,
            y_gain_11pt, y_gain_allpoint, y_gain_coco, y_count_gain_05, y_gt_count,
            y_oric_11pt, y_oric_allpoint, y_oric_coco,
            y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
            y_finegrained_vec_coco,
            y_bwd,
            y_entropy, y_img_complexity,
            edge_map, cloud_map,
            edge_map_coco, cloud_map_coco, edge_map_coco50, cloud_map_coco50,
            video_names, frame_ids, split_labels, extra_targets)


def split_and_save(cfg, X, *args) -> None:
    """Split prepared arrays and save them, accepting legacy and current payloads."""
    extra_targets = {}
    if len(args) == 28:
        (X_top25, X_dcsb, image_paths,
         y_gain_11pt, y_gain_allpoint, y_gain_coco, y_count_gain_05, y_gt_count,
         y_oric_11pt, y_oric_allpoint, y_oric_coco,
         y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
         y_finegrained_vec_coco,
         y_bwd,
         y_entropy, y_img_complexity,
         edge_map, cloud_map,
         edge_map_coco, cloud_map_coco, edge_map_coco50,
         cloud_map_coco50, video_names, frame_ids,
         split_labels, extra_targets) = args
    elif len(args) == 26:
        (X_top25, X_dcsb, image_paths,
         y_gain_11pt, y_gain_allpoint, y_gain_coco, y_count_gain_05, y_gt_count,
         y_oric_11pt, y_oric_allpoint, y_oric_coco,
         y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
         y_bwd,
         y_entropy, y_img_complexity,
         edge_map, cloud_map,
         edge_map_coco, cloud_map_coco, edge_map_coco50,
         cloud_map_coco50, video_names, frame_ids,
         split_labels) = args
    elif len(args) == 26 + 1:
        (X_top25, X_dcsb, image_paths,
         y_gain_11pt, y_gain_allpoint, y_gain_coco, y_count_gain_05, y_gt_count,
         y_oric_11pt, y_oric_allpoint, y_oric_coco,
         y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
         y_finegrained_vec_coco,
         y_bwd,
         y_entropy, y_img_complexity,
         edge_map, cloud_map,
         edge_map_coco, cloud_map_coco, edge_map_coco50,
         cloud_map_coco50, video_names, frame_ids,
         split_labels) = args
    elif len(args) == 25:
        (X_top25, image_paths,
         y_gain_11pt, y_gain_allpoint, y_gain_coco, y_count_gain_05, y_gt_count,
         y_oric_11pt, y_oric_allpoint, y_oric_coco,
         y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
         y_bwd,
         y_entropy, y_img_complexity,
         edge_map, cloud_map,
         edge_map_coco, cloud_map_coco, edge_map_coco50,
         cloud_map_coco50, video_names, frame_ids,
         split_labels) = args
        X_dcsb = X_top25.copy()
        y_finegrained_vec_coco = np.zeros(
            (len(X), len(SCENARIO_COMPONENTS)), dtype=np.float32
        )
    elif len(args) == 20:
        (image_paths,
         y_gain_11pt, y_gain_allpoint, y_gain_coco,
         y_oric_11pt, y_oric_allpoint, y_oric_coco,
         y_lcer_vec_11pt, y_lcer_vec_allpoint, y_lcer_vec_coco,
         y_entropy, y_img_complexity,
         edge_map, cloud_map,
         edge_map_coco, cloud_map_coco, edge_map_coco50,
         cloud_map_coco50, video_names, frame_ids) = args
        X_top25 = np.zeros(
            (len(X), len(topk_proposal_feature_columns("edge"))),
            dtype=np.float32,
        )
        X_dcsb = X_top25.copy()
        y_count_gain_05 = np.zeros(len(X), dtype=np.float32)
        y_gt_count = np.zeros(len(X), dtype=np.float32)
        y_finegrained_vec_coco = np.zeros(
            (len(X), len(SCENARIO_COMPONENTS)), dtype=np.float32
        )
        y_bwd = np.zeros(len(X), dtype=np.float32)
        split_labels = None
    else:
        raise TypeError(
            "split_and_save expected the extended 29-value extract_features payload, "
            "the adaptive 28-value extract_features payload, the current 27-value payload, "
            "the transitional 26-value payload, "
            "or the legacy 21-value payload."
        )

    n_images = sum(1 for p in image_paths if Path(p).exists())
    if n_images < len(image_paths):
        log.info(f"{n_images}/{len(image_paths)} images on disk "
                 "(tabular estimators unaffected)")

    n = len(X)
    train_idx, test_idx, split_method = _resolve_split_indices(
        cfg, split_labels, video_names,
    )
    n_train = len(train_idx)
    n_test = len(test_idx)

    # Determine which proxy-metric families are needed by enabled estimators
    families = _required_proxy_families(cfg)

    # MORIC now refers to the paper-faithful representative-context target
    # and is therefore derived later from dataset-wide ORIC when requested.
    moric_train_11pt = moric_test_11pt = None
    moric_train_allpoint = moric_test_allpoint = None
    moric_train_coco = moric_test_coco = None

    # MORIC+ remains a repo-side extension over the local ORIC family.
    neg_frac = cfg.dataset.moric_plus_neg_frac
    _ = neg_frac  # retained for config compatibility / metadata
    moric_plus_ref_11pt = _fit_moric_plus_reference(y_oric_11pt[train_idx])
    moric_plus_ref_allpoint = _fit_moric_plus_reference(
        y_oric_allpoint[train_idx]
    )
    moric_plus_ref_coco = _fit_moric_plus_reference(y_oric_coco[train_idx])
    y_moric_plus_train_11pt = _apply_moric_plus(
        moric_plus_ref_11pt, y_oric_11pt[train_idx]
    )
    y_moric_plus_test_11pt = _apply_moric_plus(
        moric_plus_ref_11pt, y_oric_11pt[test_idx]
    )
    y_moric_plus_train_allpoint = _apply_moric_plus(
        moric_plus_ref_allpoint, y_oric_allpoint[train_idx]
    )
    y_moric_plus_test_allpoint = _apply_moric_plus(
        moric_plus_ref_allpoint, y_oric_allpoint[test_idx]
    )
    y_moric_plus_train_coco = _apply_moric_plus(
        moric_plus_ref_coco, y_oric_coco[train_idx]
    )
    y_moric_plus_test_coco = _apply_moric_plus(
        moric_plus_ref_coco, y_oric_coco[test_idx]
    )

    # MORIC★ — sign-anchored uniform quantile (global CDF - q₀)
    moric_star_ref_11pt = _fit_moric_star_reference(y_oric_11pt[train_idx])
    moric_star_ref_allpoint = _fit_moric_star_reference(y_oric_allpoint[train_idx])
    moric_star_ref_coco = _fit_moric_star_reference(y_oric_coco[train_idx])
    y_moric_star_train_11pt = _apply_moric_star(moric_star_ref_11pt, y_oric_11pt[train_idx])
    y_moric_star_test_11pt = _apply_moric_star(moric_star_ref_11pt, y_oric_11pt[test_idx])
    y_moric_star_train_allpoint = _apply_moric_star(moric_star_ref_allpoint, y_oric_allpoint[train_idx])
    y_moric_star_test_allpoint = _apply_moric_star(moric_star_ref_allpoint, y_oric_allpoint[test_idx])
    y_moric_star_train_coco = _apply_moric_star(moric_star_ref_coco, y_oric_coco[train_idx])
    y_moric_star_test_coco = _apply_moric_star(moric_star_ref_coco, y_oric_coco[test_idx])

    # Φ-MORIC — probit normal-scores transform (Φ⁻¹(MORIC) - Φ⁻¹(q₀))
    phi_moric_ref_11pt = _fit_phi_moric_reference(y_oric_11pt[train_idx])
    phi_moric_ref_allpoint = _fit_phi_moric_reference(y_oric_allpoint[train_idx])
    phi_moric_ref_coco = _fit_phi_moric_reference(y_oric_coco[train_idx])
    y_phi_moric_train_11pt = _apply_phi_moric(phi_moric_ref_11pt, y_oric_11pt[train_idx])
    y_phi_moric_test_11pt = _apply_phi_moric(phi_moric_ref_11pt, y_oric_11pt[test_idx])
    y_phi_moric_train_allpoint = _apply_phi_moric(phi_moric_ref_allpoint, y_oric_allpoint[train_idx])
    y_phi_moric_test_allpoint = _apply_phi_moric(phi_moric_ref_allpoint, y_oric_allpoint[test_idx])
    y_phi_moric_train_coco = _apply_phi_moric(phi_moric_ref_coco, y_oric_coco[train_idx])
    y_phi_moric_test_coco = _apply_phi_moric(phi_moric_ref_coco, y_oric_coco[test_idx])

    # SigMORIC — sigmoid-scaled sign-anchored quantile (2σ(k(MORIC-q₀))-1)
    sigmoric_ref_11pt = _fit_sigmoric_reference(y_oric_11pt[train_idx])
    sigmoric_ref_allpoint = _fit_sigmoric_reference(y_oric_allpoint[train_idx])
    sigmoric_ref_coco = _fit_sigmoric_reference(y_oric_coco[train_idx])
    y_sigmoric_train_11pt = _apply_sigmoric(sigmoric_ref_11pt, y_oric_11pt[train_idx])
    y_sigmoric_test_11pt = _apply_sigmoric(sigmoric_ref_11pt, y_oric_11pt[test_idx])
    y_sigmoric_train_allpoint = _apply_sigmoric(sigmoric_ref_allpoint, y_oric_allpoint[train_idx])
    y_sigmoric_test_allpoint = _apply_sigmoric(sigmoric_ref_allpoint, y_oric_allpoint[test_idx])
    y_sigmoric_train_coco = _apply_sigmoric(sigmoric_ref_coco, y_oric_coco[train_idx])
    y_sigmoric_test_coco = _apply_sigmoric(sigmoric_ref_coco, y_oric_coco[test_idx])

    # ------------------------------------------------------------------
    # Novel proxy-metric families: binary targets and CDF transforms
    # ------------------------------------------------------------------
    if "binary" in families:
        # offload_binary: 1 if oric_allpoint > 0 else 0
        offload_binary_all = (y_oric_allpoint > 0).astype(np.float32)
        extra_targets["offload_binary"] = offload_binary_all
        # gain_top_quartile: 1 if oric > P75(oric_train) else 0
        # Threshold fitted on train split only to avoid data leakage
        p75 = float(np.percentile(y_oric_allpoint[train_idx], 75))
        gain_top_q_all = (y_oric_allpoint > p75).astype(np.float32)
        extra_targets["gain_top_quartile"] = gain_top_q_all

    # CDF-transformed variants of novel metrics (moric_ prefix)
    novel_continuous_metrics = [
        "rescue_ratio_50", "precision_gain_50",
    ]
    for metric_name in novel_continuous_metrics:
        if metric_name in extra_targets:
            arr = extra_targets[metric_name]
            train_vals = arr[train_idx]
            ref = np.sort(train_vals.astype(np.float64))
            if ref.size > 0:
                moric_train = (np.searchsorted(ref, train_vals.astype(np.float64), side="right") / max(len(ref), 1)).astype(np.float32)
                moric_test = (np.searchsorted(ref, arr[test_idx].astype(np.float64), side="right") / max(len(ref), 1)).astype(np.float32)
                extra_targets[f"moric_{metric_name}"] = np.zeros(len(arr), dtype=np.float32)
                extra_targets[f"moric_{metric_name}"][train_idx] = moric_train
                extra_targets[f"moric_{metric_name}"][test_idx] = moric_test

    # ------------------------------------------------------------------
    # Conditionally compute expensive proxy-metric families.
    # Only families required by enabled estimators are computed.
    # ------------------------------------------------------------------
    train_paths = [image_paths[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    conditional_neighborhood = None
    if "adaptive_scenario" in families:
        conditional_neighborhood = _compute_conditional_reward_neighborhood(
            train_paths, test_paths, cache_dir=cfg.derived_dir,
        )

    # LCER scalar targets and CSR labels (needed by Hybrid LCER-CSR)
    lcer_vec_train_coco = y_lcer_vec_coco[train_idx].astype(np.float32)
    lcer_vec_test_coco = y_lcer_vec_coco[test_idx].astype(np.float32)
    if "lcer" in families:
        lcer_vec_train_11pt = y_lcer_vec_11pt[train_idx]
        lcer_vec_test_11pt = y_lcer_vec_11pt[test_idx]
        lcer_vec_train_allpoint = y_lcer_vec_allpoint[train_idx]
        lcer_vec_test_allpoint = y_lcer_vec_allpoint[test_idx]
        lcer_vec_train_coco = lcer_vec_train_coco.astype(np.float64, copy=False)
        lcer_vec_test_coco = lcer_vec_test_coco.astype(np.float64, copy=False)

        beta_11pt = _fit_lcer_beta(lcer_vec_train_11pt, y_oric_11pt[train_idx])
        beta_allpoint = _fit_lcer_beta(
            lcer_vec_train_allpoint, y_oric_allpoint[train_idx]
        )
        beta_coco = _fit_lcer_beta(lcer_vec_train_coco, y_oric_coco[train_idx])

        lcer_train_11pt = lcer_vec_train_11pt @ beta_11pt
        lcer_test_11pt = lcer_vec_test_11pt @ beta_11pt
        lcer_train_allpoint = lcer_vec_train_allpoint @ beta_allpoint
        lcer_test_allpoint = lcer_vec_test_allpoint @ beta_allpoint
        lcer_train_coco = lcer_vec_train_coco @ beta_coco
        lcer_test_coco = lcer_vec_test_coco @ beta_coco

        csr_train_11pt, csr_test_11pt = _build_csr_targets(
            lcer_train_11pt, lcer_test_11pt
        )
        csr_train_allpoint, csr_test_allpoint = _build_csr_targets(
            lcer_train_allpoint, lcer_test_allpoint
        )
        csr_train_coco, csr_test_coco = _build_csr_targets(
            lcer_train_coco, lcer_test_coco
        )

    # Dataset-wide ORIC (needed by paper-faithful MORIC and dataset_* aliases)
    if "dataset_oric" in families:
        log.info("Computing dataset-wide ORIC (cached in dataset dir) ...")
        ds_oric_train = _get_dataset_wide_oric_cached(
            cfg, cfg.derived_dir,
            video_names[train_idx], frame_ids[train_idx], "train",
        )
        ds_oric_test = _get_dataset_wide_oric_cached(
            cfg, cfg.derived_dir,
            video_names[test_idx], frame_ids[test_idx], "test",
        )

        ds_oric_train_11pt = ds_oric_train[:, 0]
        ds_oric_train_allpt = ds_oric_train[:, 1]
        ds_oric_train_coco = ds_oric_train[:, 2]
        ds_oric_test_11pt = ds_oric_test[:, 0]
        ds_oric_test_allpt = ds_oric_test[:, 1]
        ds_oric_test_coco = ds_oric_test[:, 2]

        ds_moric_ref_11pt = _fit_moric_reference(ds_oric_train_11pt)
        ds_moric_ref_allpt = _fit_moric_reference(ds_oric_train_allpt)
        ds_moric_ref_coco = _fit_moric_reference(ds_oric_train_coco)
        ds_moric_train_11pt = _apply_moric(ds_moric_ref_11pt, ds_oric_train_11pt)
        ds_moric_test_11pt = _apply_moric(ds_moric_ref_11pt, ds_oric_test_11pt)
        ds_moric_train_allpt = _apply_moric(ds_moric_ref_allpt, ds_oric_train_allpt)
        ds_moric_test_allpt = _apply_moric(ds_moric_ref_allpt, ds_oric_test_allpt)
        ds_moric_train_coco = _apply_moric(ds_moric_ref_coco, ds_oric_train_coco)
        ds_moric_test_coco = _apply_moric(ds_moric_ref_coco, ds_oric_test_coco)

        # Public MORIC target family: split-wide/context-sampled ORIC using
        # the paper-style representative context rather than chunk-local ORIC.
        moric_train_11pt = ds_moric_train_11pt
        moric_test_11pt = ds_moric_test_11pt
        moric_train_allpoint = ds_moric_train_allpt
        moric_test_allpoint = ds_moric_test_allpt
        moric_train_coco = ds_moric_train_coco
        moric_test_coco = ds_moric_test_coco

        ds_moric_plus_ref_11pt = _fit_moric_plus_reference(ds_oric_train_11pt)
        ds_moric_plus_ref_allpt = _fit_moric_plus_reference(ds_oric_train_allpt)
        ds_moric_plus_ref_coco = _fit_moric_plus_reference(ds_oric_train_coco)
        ds_moric_plus_train_11pt = _apply_moric_plus(
            ds_moric_plus_ref_11pt, ds_oric_train_11pt)
        ds_moric_plus_test_11pt = _apply_moric_plus(
            ds_moric_plus_ref_11pt, ds_oric_test_11pt)
        ds_moric_plus_train_allpt = _apply_moric_plus(
            ds_moric_plus_ref_allpt, ds_oric_train_allpt)
        ds_moric_plus_test_allpt = _apply_moric_plus(
            ds_moric_plus_ref_allpt, ds_oric_test_allpt)
        ds_moric_plus_train_coco = _apply_moric_plus(
            ds_moric_plus_ref_coco, ds_oric_train_coco)
        ds_moric_plus_test_coco = _apply_moric_plus(
            ds_moric_plus_ref_coco, ds_oric_test_coco)

        # Dataset-wide MORIC★
        ds_moric_star_ref_11pt = _fit_moric_star_reference(ds_oric_train_11pt)
        ds_moric_star_ref_allpt = _fit_moric_star_reference(ds_oric_train_allpt)
        ds_moric_star_ref_coco = _fit_moric_star_reference(ds_oric_train_coco)
        ds_moric_star_train_11pt = _apply_moric_star(ds_moric_star_ref_11pt, ds_oric_train_11pt)
        ds_moric_star_test_11pt = _apply_moric_star(ds_moric_star_ref_11pt, ds_oric_test_11pt)
        ds_moric_star_train_allpt = _apply_moric_star(ds_moric_star_ref_allpt, ds_oric_train_allpt)
        ds_moric_star_test_allpt = _apply_moric_star(ds_moric_star_ref_allpt, ds_oric_test_allpt)
        ds_moric_star_train_coco = _apply_moric_star(ds_moric_star_ref_coco, ds_oric_train_coco)
        ds_moric_star_test_coco = _apply_moric_star(ds_moric_star_ref_coco, ds_oric_test_coco)

        # Dataset-wide Φ-MORIC
        ds_phi_moric_ref_11pt = _fit_phi_moric_reference(ds_oric_train_11pt)
        ds_phi_moric_ref_allpt = _fit_phi_moric_reference(ds_oric_train_allpt)
        ds_phi_moric_ref_coco = _fit_phi_moric_reference(ds_oric_train_coco)
        ds_phi_moric_train_11pt = _apply_phi_moric(ds_phi_moric_ref_11pt, ds_oric_train_11pt)
        ds_phi_moric_test_11pt = _apply_phi_moric(ds_phi_moric_ref_11pt, ds_oric_test_11pt)
        ds_phi_moric_train_allpt = _apply_phi_moric(ds_phi_moric_ref_allpt, ds_oric_train_allpt)
        ds_phi_moric_test_allpt = _apply_phi_moric(ds_phi_moric_ref_allpt, ds_oric_test_allpt)
        ds_phi_moric_train_coco = _apply_phi_moric(ds_phi_moric_ref_coco, ds_oric_train_coco)
        ds_phi_moric_test_coco = _apply_phi_moric(ds_phi_moric_ref_coco, ds_oric_test_coco)

        # Dataset-wide SigMORIC
        ds_sigmoric_ref_11pt = _fit_sigmoric_reference(ds_oric_train_11pt)
        ds_sigmoric_ref_allpt = _fit_sigmoric_reference(ds_oric_train_allpt)
        ds_sigmoric_ref_coco = _fit_sigmoric_reference(ds_oric_train_coco)
        ds_sigmoric_train_11pt = _apply_sigmoric(ds_sigmoric_ref_11pt, ds_oric_train_11pt)
        ds_sigmoric_test_11pt = _apply_sigmoric(ds_sigmoric_ref_11pt, ds_oric_test_11pt)
        ds_sigmoric_train_allpt = _apply_sigmoric(ds_sigmoric_ref_allpt, ds_oric_train_allpt)
        ds_sigmoric_test_allpt = _apply_sigmoric(ds_sigmoric_ref_allpt, ds_oric_test_allpt)
        ds_sigmoric_train_coco = _apply_sigmoric(ds_sigmoric_ref_coco, ds_oric_train_coco)
        ds_sigmoric_test_coco = _apply_sigmoric(ds_sigmoric_ref_coco, ds_oric_test_coco)

    scenario_weight_map = normalize_scenario_weight_map(
        cfg.adaptive_scenario_weights
    )
    default_scenario_name = "balanced"
    default_scenario_weights = default_balanced_scenario_weights()
    if "adaptive_scenario" in families:
        finegrained_train_coco = y_finegrained_vec_coco[train_idx].astype(np.float32)
        finegrained_test_coco = y_finegrained_vec_coco[test_idx].astype(np.float32)
        (finegrained_smooth_train_coco, _finegrained_var_train_coco,
         finegrained_smooth_test_coco, _finegrained_var_test_coco) = smooth_reward_vectors(
            finegrained_train_coco,
            conditional_neighborhood,
        )
        finegrained_smooth_train_coco = finegrained_smooth_train_coco.astype(np.float32)
        finegrained_smooth_test_coco = (
            np.asarray(finegrained_smooth_test_coco, dtype=np.float32)
            if finegrained_smooth_test_coco is not None
            else np.zeros_like(finegrained_test_coco)
        )
        scenario_weight_map_lcer = normalize_scenario_weight_map(
            cfg.adaptive_scenario_weights,
            component_names=LCER_SCENARIO_COMPONENTS,
        )
        default_scenario_weights_lcer = normalize_scenario_weight_map(
            {default_scenario_name: default_scenario_weights},
            component_names=LCER_SCENARIO_COMPONENTS,
        )[default_scenario_name]
        scenario_utility_train_coco = (
            finegrained_train_coco @ default_scenario_weights.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_test_coco = (
            finegrained_test_coco @ default_scenario_weights.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_train_lcer_coco = (
            lcer_vec_train_coco @ default_scenario_weights_lcer.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_test_lcer_coco = (
            lcer_vec_test_coco @ default_scenario_weights_lcer.astype(np.float32)
        ).astype(np.float32)
        # LCER3: project 6-D LCER to 3-D meta-component space
        # (precision, localization_quality, recall) via the PCA-validated
        # grouping matrix.  The 3 axes are nearly independent (max ρ=0.21).
        lcer3_vec_train_coco = (
            lcer_vec_train_coco.astype(np.float32) @ LCER6_TO_LCER3_PROJECTION
        )
        lcer3_vec_test_coco = (
            lcer_vec_test_coco.astype(np.float32) @ LCER6_TO_LCER3_PROJECTION
        )
        default_scenario_weights_lcer3 = normalize_scenario_weight_map(
            {default_scenario_name: {comp: 1.0 for comp in LCER3_SCENARIO_COMPONENTS}},
            component_names=LCER3_SCENARIO_COMPONENTS,
        )[default_scenario_name]
        scenario_utility_train_lcer3_coco = (
            lcer3_vec_train_coco @ default_scenario_weights_lcer3.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_test_lcer3_coco = (
            lcer3_vec_test_coco @ default_scenario_weights_lcer3.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_train_smooth_coco = (
            finegrained_smooth_train_coco @ default_scenario_weights.astype(np.float32)
        ).astype(np.float32)
        scenario_utility_test_smooth_coco = (
            finegrained_smooth_test_coco @ default_scenario_weights.astype(np.float32)
        ).astype(np.float32)
    # ------------------------------------------------------------------
    # Build data.npz dynamically based on required families
    # ------------------------------------------------------------------
    out = cfg.output.prepared_dir
    out.mkdir(parents=True, exist_ok=True)

    npz = dict(
        X_train=X[train_idx], X_test=X[test_idx],
        X_train_top25=X_top25[train_idx], X_test_top25=X_top25[test_idx],
        X_train_dcsb=X_dcsb[train_idx], X_test_dcsb=X_dcsb[test_idx],
        # Legacy back-compat
        y_train=y_gain_11pt[train_idx], y_test=y_gain_11pt[test_idx],
        # Base proxy-metric arrays (always computed)
        y_train_gain_11pt=y_gain_11pt[train_idx], y_test_gain_11pt=y_gain_11pt[test_idx],
        y_train_gain_allpoint=y_gain_allpoint[train_idx], y_test_gain_allpoint=y_gain_allpoint[test_idx],
        y_train_gain_coco=y_gain_coco[train_idx], y_test_gain_coco=y_gain_coco[test_idx],
        y_train_count_gain_05=y_count_gain_05[train_idx], y_test_count_gain_05=y_count_gain_05[test_idx],
        y_train_gt_count=y_gt_count[train_idx], y_test_gt_count=y_gt_count[test_idx],
        y_train_oric_11pt=y_oric_11pt[train_idx], y_test_oric_11pt=y_oric_11pt[test_idx],
        y_train_oric_allpoint=y_oric_allpoint[train_idx], y_test_oric_allpoint=y_oric_allpoint[test_idx],
        y_train_oric_coco=y_oric_coco[train_idx], y_test_oric_coco=y_oric_coco[test_idx],
        y_train_moric_plus_11pt=y_moric_plus_train_11pt, y_test_moric_plus_11pt=y_moric_plus_test_11pt,
        y_train_moric_plus_allpoint=y_moric_plus_train_allpoint, y_test_moric_plus_allpoint=y_moric_plus_test_allpoint,
        y_train_moric_plus_coco=y_moric_plus_train_coco, y_test_moric_plus_coco=y_moric_plus_test_coco,
        # MORIC★ and Φ-MORIC (always computed alongside MORIC+)
        y_train_moric_star_11pt=y_moric_star_train_11pt, y_test_moric_star_11pt=y_moric_star_test_11pt,
        y_train_moric_star_allpoint=y_moric_star_train_allpoint, y_test_moric_star_allpoint=y_moric_star_test_allpoint,
        y_train_moric_star_coco=y_moric_star_train_coco, y_test_moric_star_coco=y_moric_star_test_coco,
        y_train_phi_moric_11pt=y_phi_moric_train_11pt, y_test_phi_moric_11pt=y_phi_moric_test_11pt,
        y_train_phi_moric_allpoint=y_phi_moric_train_allpoint, y_test_phi_moric_allpoint=y_phi_moric_test_allpoint,
        y_train_phi_moric_coco=y_phi_moric_train_coco, y_test_phi_moric_coco=y_phi_moric_test_coco,
        # q₀ metadata for MORIC★ and Φ-MORIC reproducibility
        meta_moric_star_q0_11pt=np.float64(moric_star_ref_11pt["q0"]),
        meta_moric_star_q0_allpoint=np.float64(moric_star_ref_allpoint["q0"]),
        meta_moric_star_q0_coco=np.float64(moric_star_ref_coco["q0"]),
        meta_phi_moric_probit_q0_11pt=np.float64(phi_moric_ref_11pt["probit_q0"]),
        meta_phi_moric_probit_q0_allpoint=np.float64(phi_moric_ref_allpoint["probit_q0"]),
        meta_phi_moric_probit_q0_coco=np.float64(phi_moric_ref_coco["probit_q0"]),
        # SigMORIC (always computed alongside MORIC★ and Φ-MORIC)
        y_train_sigmoric_11pt=y_sigmoric_train_11pt, y_test_sigmoric_11pt=y_sigmoric_test_11pt,
        y_train_sigmoric_allpoint=y_sigmoric_train_allpoint, y_test_sigmoric_allpoint=y_sigmoric_test_allpoint,
        y_train_sigmoric_coco=y_sigmoric_train_coco, y_test_sigmoric_coco=y_sigmoric_test_coco,
        meta_sigmoric_q0_11pt=np.float64(sigmoric_ref_11pt["q0"]),
        meta_sigmoric_q0_allpoint=np.float64(sigmoric_ref_allpoint["q0"]),
        meta_sigmoric_q0_coco=np.float64(sigmoric_ref_coco["q0"]),
        meta_sigmoric_k_11pt=np.float64(sigmoric_ref_11pt["k"]),
        meta_sigmoric_k_allpoint=np.float64(sigmoric_ref_allpoint["k"]),
        meta_sigmoric_k_coco=np.float64(sigmoric_ref_coco["k"]),
        y_train_bwd=y_bwd[train_idx], y_test_bwd=y_bwd[test_idx],
        y_train_entropy=y_entropy[train_idx], y_test_entropy=y_entropy[test_idx],
        y_train_img_complexity=y_img_complexity[train_idx], y_test_img_complexity=y_img_complexity[test_idx],
        edge_test=edge_map[test_idx], cloud_test=cloud_map[test_idx],
        edge_test_coco=edge_map_coco[test_idx],
        cloud_test_coco=cloud_map_coco[test_idx],
        edge_test_coco50=edge_map_coco50[test_idx],
        cloud_test_coco50=cloud_map_coco50[test_idx],
        video_name_test=video_names[test_idx],
        frame_id_test=frame_ids[test_idx],
    )

    for name, values in extra_targets.items():
        arr = np.asarray(values)
        npz[f"y_train_{name}"] = arr[train_idx]
        npz[f"y_test_{name}"] = arr[test_idx]

    if "lcer" in families:
        npz.update(
            y_train_lcer_vec_11pt=lcer_vec_train_11pt,
            y_test_lcer_vec_11pt=lcer_vec_test_11pt,
            y_train_lcer_vec_allpoint=lcer_vec_train_allpoint,
            y_test_lcer_vec_allpoint=lcer_vec_test_allpoint,
            y_train_lcer_vec_coco=lcer_vec_train_coco,
            y_test_lcer_vec_coco=lcer_vec_test_coco,
            y_train_lcer_11pt=lcer_train_11pt,
            y_test_lcer_11pt=lcer_test_11pt,
            y_train_lcer_allpoint=lcer_train_allpoint,
            y_test_lcer_allpoint=lcer_test_allpoint,
            y_train_lcer_coco=lcer_train_coco,
            y_test_lcer_coco=lcer_test_coco,
            y_train_csr_11pt=csr_train_11pt,
            y_test_csr_11pt=csr_test_11pt,
            y_train_csr_allpoint=csr_train_allpoint,
            y_test_csr_allpoint=csr_test_allpoint,
            y_train_csr_coco=csr_train_coco,
            y_test_csr_coco=csr_test_coco,
            meta_beta_11pt=beta_11pt,
            meta_beta_allpoint=beta_allpoint,
            meta_beta_coco=beta_coco,
            meta_tau_11pt=LCER_TAU_GRID,
            meta_tau_allpoint=LCER_TAU_GRID,
            meta_tau_coco=LCER_TAU_GRID,
        )

    if "adaptive_scenario" in families:
        npz.update(
            y_train_finegrained_vec_coco=finegrained_train_coco,
            y_test_finegrained_vec_coco=finegrained_test_coco,
            y_train_finegrained_vec_smooth_coco=finegrained_smooth_train_coco,
            y_test_finegrained_vec_smooth_coco=finegrained_smooth_test_coco,
            y_train_scenario_utility_coco=scenario_utility_train_coco,
            y_test_scenario_utility_coco=scenario_utility_test_coco,
            y_train_scenario_utility_lcer_coco=scenario_utility_train_lcer_coco,
            y_test_scenario_utility_lcer_coco=scenario_utility_test_lcer_coco,
            y_train_scenario_utility_smooth_coco=scenario_utility_train_smooth_coco,
            y_test_scenario_utility_smooth_coco=scenario_utility_test_smooth_coco,
            **_scenario_metadata_payload(
                "coco",
                SCENARIO_COMPONENTS,
                scenario_weight_map,
                default_scenario_name,
                default_scenario_weights,
            ),
            **_scenario_metadata_payload(
                "lcer_coco",
                LCER_SCENARIO_COMPONENTS,
                scenario_weight_map_lcer,
                default_scenario_name,
                default_scenario_weights_lcer,
            ),
            **_scenario_metadata_payload(
                "smooth_coco",
                SCENARIO_COMPONENTS,
                scenario_weight_map,
                default_scenario_name,
                default_scenario_weights,
            ),
            # LCER3 meta-component targets
            y_train_lcer3_vec_coco=lcer3_vec_train_coco,
            y_test_lcer3_vec_coco=lcer3_vec_test_coco,
            y_train_scenario_utility_lcer3_coco=scenario_utility_train_lcer3_coco,
            y_test_scenario_utility_lcer3_coco=scenario_utility_test_lcer3_coco,
            **_scenario_metadata_payload(
                "lcer3_coco",
                LCER3_SCENARIO_COMPONENTS,
                DEFAULT_LCER3_SCENARIO_WEIGHTS,
                default_scenario_name,
                default_scenario_weights_lcer3,
            ),
        )

    if "dataset_oric" in families:
        npz.update(
            y_train_dataset_oric_11pt=ds_oric_train_11pt,
            y_test_dataset_oric_11pt=ds_oric_test_11pt,
            y_train_dataset_oric_allpoint=ds_oric_train_allpt,
            y_test_dataset_oric_allpoint=ds_oric_test_allpt,
            y_train_dataset_oric_coco=ds_oric_train_coco,
            y_test_dataset_oric_coco=ds_oric_test_coco,
            y_train_moric_11pt=moric_train_11pt,
            y_test_moric_11pt=moric_test_11pt,
            y_train_moric_allpoint=moric_train_allpoint,
            y_test_moric_allpoint=moric_test_allpoint,
            y_train_moric_coco=moric_train_coco,
            y_test_moric_coco=moric_test_coco,
            # Retained as backward-compatible aliases for older configs.
            y_train_dataset_moric_11pt=moric_train_11pt,
            y_test_dataset_moric_11pt=moric_test_11pt,
            y_train_dataset_moric_allpoint=moric_train_allpoint,
            y_test_dataset_moric_allpoint=moric_test_allpoint,
            y_train_dataset_moric_coco=moric_train_coco,
            y_test_dataset_moric_coco=moric_test_coco,
            y_train_dataset_moric_plus_11pt=ds_moric_plus_train_11pt,
            y_test_dataset_moric_plus_11pt=ds_moric_plus_test_11pt,
            y_train_dataset_moric_plus_allpoint=ds_moric_plus_train_allpt,
            y_test_dataset_moric_plus_allpoint=ds_moric_plus_test_allpt,
            y_train_dataset_moric_plus_coco=ds_moric_plus_train_coco,
            y_test_dataset_moric_plus_coco=ds_moric_plus_test_coco,
            # Dataset-wide MORIC★ and Φ-MORIC
            y_train_dataset_moric_star_11pt=ds_moric_star_train_11pt,
            y_test_dataset_moric_star_11pt=ds_moric_star_test_11pt,
            y_train_dataset_moric_star_allpoint=ds_moric_star_train_allpt,
            y_test_dataset_moric_star_allpoint=ds_moric_star_test_allpt,
            y_train_dataset_moric_star_coco=ds_moric_star_train_coco,
            y_test_dataset_moric_star_coco=ds_moric_star_test_coco,
            y_train_dataset_phi_moric_11pt=ds_phi_moric_train_11pt,
            y_test_dataset_phi_moric_11pt=ds_phi_moric_test_11pt,
            y_train_dataset_phi_moric_allpoint=ds_phi_moric_train_allpt,
            y_test_dataset_phi_moric_allpoint=ds_phi_moric_test_allpt,
            y_train_dataset_phi_moric_coco=ds_phi_moric_train_coco,
            y_test_dataset_phi_moric_coco=ds_phi_moric_test_coco,
            # Dataset-wide SigMORIC
            y_train_dataset_sigmoric_11pt=ds_sigmoric_train_11pt,
            y_test_dataset_sigmoric_11pt=ds_sigmoric_test_11pt,
            y_train_dataset_sigmoric_allpoint=ds_sigmoric_train_allpt,
            y_test_dataset_sigmoric_allpoint=ds_sigmoric_test_allpt,
            y_train_dataset_sigmoric_coco=ds_sigmoric_train_coco,
            y_test_dataset_sigmoric_coco=ds_sigmoric_test_coco,
        )

    np.savez_compressed(out / "data.npz", **npz)

    (out / "paths_train.txt").write_text("\n".join(train_paths))
    (out / "paths_test.txt").write_text("\n".join(test_paths))

    # SRRM spatial matrices: stored separately to avoid bloating data.npz
    srrm_npz_path = out / "srrm.npz"
    if "srrm" in families:
        srrm_all = _load_srrm_matrices(
            Path(cfg.output.data_dir), video_names, frame_ids,
        )
        if srrm_all is None:
            raise FileNotFoundError(
                "Prepared data requires SRRM targets, but the derived SRRM "
                "sidecars are missing. Re-run prepare after deriving SRRM."
            )
        np.savez_compressed(
            srrm_npz_path,
            srrm_train=srrm_all[train_idx],
            srrm_test=srrm_all[test_idx],
        )
        print(f"  SRRM spatial matrices: {srrm_all.shape[1]}×{srrm_all.shape[2]} grid, "
              f"train={len(train_idx)}, test={len(test_idx)}")
    else:
        srrm_npz_path.unlink(missing_ok=True)

    meta = {
        "n_total": n, "n_train": n_train, "n_test": n_test,
        "n_features": X.shape[1],
        "n_features_top25": X_top25.shape[1],
        "n_features_dcsb": X_dcsb.shape[1],
        "split_method": split_method,
        "train_videos": int(len(np.unique(video_names[train_idx]))),
        "test_videos": int(len(np.unique(video_names[test_idx]))),
        "oric_context_draws": cfg.oric_context_draws,
        "conditional_reward_knn_k": CEORIC_DEFAULT_K,
        "conditional_reward_kernel_tau": CEORIC_DEFAULT_TAU,
    }
    meta.update(_prepare_cache_signature(cfg, families, _required_prepare_inputs(cfg)))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    log.kv_group([
        ("Train", log.fmt_count(n_train)),
        ("Test", log.fmt_count(n - n_train)),
        ("Features", X.shape[1]),
    ])
    log.arrow(str(out))
