"""Feature constants and utilities for estimators.

Extracted from temporal estimator for shared use across all estimator types.
Also provides functions for loading detection results and extracting features.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Base features per model (21 each)
BASE_FEATURES = [
    'det_count',
    'conf_mean', 'conf_std', 'conf_min', 'conf_max', 'conf_median', 'conf_q25', 'conf_q75',
    'area_mean', 'area_std', 'area_min', 'area_max', 'area_total',
    'cx_mean', 'cx_std', 'cy_mean', 'cy_std',
    'aspect_mean', 'aspect_std',
    'area_cv', 'conf_area_corr',
]

# Features for both edge and cloud models
EDGE_FEATURES = [f'edge_{f}' for f in BASE_FEATURES]
CLOUD_FEATURES = [f'cloud_{f}' for f in BASE_FEATURES]
FEATURE_COLUMNS = EDGE_FEATURES + CLOUD_FEATURES

# Paper-faithful weak-detector proposal representation used by the original
# post-stage baselines (EdgeML / DCSB).
TOPK_PROPOSALS = 25
PROPOSAL_FEATURES = [
    "conf",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_ratio",
]


def topk_proposal_feature_columns(
    prefix: str = "edge",
    topk: int = TOPK_PROPOSALS,
) -> List[str]:
    """Column order for flattened top-K weak-detector proposals."""
    cols: List[str] = []
    for idx in range(topk):
        for feat in PROPOSAL_FEATURES:
            cols.append(f"{prefix}_proposal_{idx:02d}_{feat}")
    return cols


def encode_float_sequence(values) -> str:
    """Encode a float sequence compactly for CSV storage."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ""
    return " ".join(f"{float(v):.8g}" for v in arr)


def decode_float_sequence(text) -> np.ndarray:
    """Decode a float sequence encoded by ``encode_float_sequence``."""
    if text is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(text, float) and np.isnan(text):
        return np.zeros(0, dtype=np.float32)
    s = str(text).strip()
    if not s:
        return np.zeros(0, dtype=np.float32)
    return np.fromstring(s, sep=" ", dtype=np.float32)


def extract_full_proposal_rule_inputs(
    detections: List[Tuple],
    image_width: float,
    image_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full weak-detector proposal stream needed by DCSB.

    Proposals are sorted by confidence and represented only by the fields the
    DCSB rule actually uses: confidence and normalized area ratio.
    """
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    image_area = max(width * height, 1.0)

    ordered = sorted(detections, key=lambda det: det[4], reverse=True)
    if not ordered:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    confs = np.array([det[4] for det in ordered], dtype=np.float32)
    areas = np.array(
        [(det[2] * det[3]) / image_area for det in ordered],
        dtype=np.float32,
    )
    return confs, areas


def build_rule_feature_matrix(
    confidence_sequences,
    area_sequences,
    *,
    proposal_stride: int = len(PROPOSAL_FEATURES),
    confidence_offset: int = 0,
    area_offset: int = 5,
) -> np.ndarray:
    """Pad full proposal streams into a dense matrix for DCSB.

    Only confidence and area-ratio slots are populated because the original
    DCSB rule uses exactly those two signals.
    """
    conf_list = [np.asarray(seq, dtype=np.float32).reshape(-1)
                 for seq in confidence_sequences]
    area_list = [np.asarray(seq, dtype=np.float32).reshape(-1)
                 for seq in area_sequences]
    n = len(conf_list)
    max_props = max((len(seq) for seq in conf_list), default=0)
    if max_props == 0:
        X = np.zeros((n, proposal_stride), dtype=np.float32)
        X[:, confidence_offset::proposal_stride] = -1.0
        return X

    X = np.zeros((n, max_props * proposal_stride), dtype=np.float32)
    X[:, confidence_offset::proposal_stride] = -1.0
    for row_idx, (confs, areas) in enumerate(zip(conf_list, area_list)):
        limit = min(len(confs), len(areas))
        if limit == 0:
            continue
        X[row_idx, confidence_offset::proposal_stride][:limit] = confs[:limit]
        X[row_idx, area_offset::proposal_stride][:limit] = areas[:limit]
    return X


def get_feature_columns(df: pd.DataFrame, include_cloud: bool = True) -> List[str]:
    """Get available feature columns from a DataFrame.
    
    Args:
        df: DataFrame with feature columns
        include_cloud: Whether to include cloud features (not available at inference)
    
    Returns:
        List of feature column names present in the DataFrame
    """
    if include_cloud:
        return [c for c in FEATURE_COLUMNS if c in df.columns]
    return [c for c in EDGE_FEATURES if c in df.columns]


def load_detection_results(det_file: Path) -> Dict[int, List[Tuple]]:
    """Load detection results from a text file.

    Expected format per line:
        frame_id,object_id,left,top,width,height,confidence,class

    Args:
        det_file: Path to the detection results file.

    Returns:
        Dict mapping frame_id -> list of (left, top, width, height, conf, class).
    """
    det_data: Dict[int, List[Tuple]] = {}
    with open(det_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            conf = float(parts[6])
            cls = parts[7].strip()
            det_data.setdefault(frame_id, []).append((left, top, width, height, conf, cls))
    return det_data


def extract_detection_features(detections: List[Tuple]) -> Dict[str, float]:
    """Extract statistical features from a list of detections for one frame.

    Computes features matching BASE_FEATURES: detection count, confidence
    statistics, bounding-box area statistics, centroid statistics, aspect
    ratio statistics, area CV, and confidence-area correlation.

    Args:
        detections: List of (left, top, width, height, conf, class) tuples.

    Returns:
        Dict mapping feature name -> value.
    """
    n = len(detections)
    if n == 0:
        return {f: 0.0 for f in BASE_FEATURES}

    confs = np.array([d[4] for d in detections])
    widths = np.array([d[2] for d in detections])
    heights = np.array([d[3] for d in detections])
    areas = widths * heights
    cxs = np.array([d[0] + d[2] / 2 for d in detections])
    cys = np.array([d[1] + d[3] / 2 for d in detections])
    aspects = widths / np.maximum(heights, 1e-6)

    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas)) if n > 1 else 0.0
    area_cv = area_std / max(area_mean, 1e-6)

    if n > 1:
        # np.corrcoef issues a RuntimeWarning if either array has zero variance.
        # Check standard deviations first.
        conf_std = np.std(confs)
        area_std_val = np.std(areas)
        if conf_std > 1e-9 and area_std_val > 1e-9:
            corr_mat = np.corrcoef(confs, areas)
            conf_area_corr = float(corr_mat[0, 1]) if not np.isnan(corr_mat[0, 1]) else 0.0
        else:
            conf_area_corr = 0.0
    else:
        conf_area_corr = 0.0

    return {
        'det_count': float(n),
        'conf_mean': float(np.mean(confs)),
        'conf_std': float(np.std(confs)) if n > 1 else 0.0,
        'conf_min': float(np.min(confs)),
        'conf_max': float(np.max(confs)),
        'conf_median': float(np.median(confs)),
        'conf_q25': float(np.percentile(confs, 25)),
        'conf_q75': float(np.percentile(confs, 75)),
        'area_mean': area_mean,
        'area_std': area_std,
        'area_min': float(np.min(areas)),
        'area_max': float(np.max(areas)),
        'area_total': float(np.sum(areas)),
        'cx_mean': float(np.mean(cxs)),
        'cx_std': float(np.std(cxs)) if n > 1 else 0.0,
        'cy_mean': float(np.mean(cys)),
        'cy_std': float(np.std(cys)) if n > 1 else 0.0,
        'aspect_mean': float(np.mean(aspects)),
        'aspect_std': float(np.std(aspects)) if n > 1 else 0.0,
        'area_cv': area_cv,
        'conf_area_corr': conf_area_corr,
    }


def extract_topk_proposal_features(
    detections: List[Tuple],
    image_width: float,
    image_height: float,
    prefix: str = "edge",
    topk: int = TOPK_PROPOSALS,
) -> Dict[str, float]:
    """Flatten the top-K weak-detector proposals into a fixed-length vector.

    Each proposal contributes confidence plus normalized geometry. Missing
    proposals are zero-padded.
    """
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    image_area = max(width * height, 1.0)

    ordered = sorted(detections, key=lambda det: det[4], reverse=True)[:topk]
    features = {col: 0.0 for col in topk_proposal_feature_columns(prefix, topk)}

    for idx, det in enumerate(ordered):
        left, top, box_w, box_h, conf, _cls = det
        cx = left + box_w / 2.0
        cy = top + box_h / 2.0
        area_ratio = (box_w * box_h) / image_area
        values = {
            "conf": float(conf),
            "cx_norm": float(cx / width),
            "cy_norm": float(cy / height),
            "w_norm": float(box_w / width),
            "h_norm": float(box_h / height),
            "area_ratio": float(area_ratio),
        }
        for feat_name, value in values.items():
            features[f"{prefix}_proposal_{idx:02d}_{feat_name}"] = value

    return features
