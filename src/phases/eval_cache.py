"""Evaluation result cache: load/save/validate + decision serialization."""

import json
from pathlib import Path

import numpy as np

from config import ApproachConfig, PipelineConfig

from ._shared import approach_config_signature, checkpoint_path as _checkpoint_path
from ..offloader import OffloadDecision

# ---- Evaluation cache ---------------------------------------------------
_EVAL_CACHE_DIR = "_eval_cache"


def _sanitize_name(name: str) -> str:
    """Make approach name safe for directory names."""
    return name.replace("|", "__").replace("/", "_").replace(" ", "_")


def _eval_cache_signature(pcfg: ApproachConfig, cfg: PipelineConfig) -> str:
    """Hash of everything that affects evaluation output for an approach."""
    import hashlib
    seeds = list(getattr(cfg, "evaluation_seeds", None) or [cfg.seed])
    ckpt = _checkpoint_path(cfg, pcfg, seeds[0] if len(seeds) > 1 else None)
    ckpt_mtime = ckpt.stat().st_mtime_ns if ckpt.exists() else 0
    data = {
        "approach_sig": approach_config_signature(pcfg),
        "ckpt_mtime": ckpt_mtime,
        "offload_ratios": sorted(float(r) for r in cfg.offload_ratios),
        "seeds": sorted(int(s) for s in seeds),
        "calibration_bins": cfg.calibration_bins,
        "fixed_ratio_points": sorted(float(r) for r in (cfg.fixed_ratio_points or [])),
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def _eval_cache_dir_for(cfg: PipelineConfig, pcfg: ApproachConfig) -> Path:
    return cfg.output.metrics_dir / _EVAL_CACHE_DIR / _sanitize_name(pcfg.name)


def _eval_cache_valid(adir: Path, expected_sig: str) -> bool:
    """Check if cached evaluation results are still valid."""
    sig_path = adir / "eval_signature.json"
    if not sig_path.exists():
        return False
    try:
        stored = json.loads(sig_path.read_text())
        return stored.get("signature") == expected_sig
    except (json.JSONDecodeError, OSError):
        return False


def _save_eval_cache(adir: Path, payloads: list[dict], sig: str) -> None:
    """Save task payloads for one approach to disk."""
    adir.mkdir(parents=True, exist_ok=True)
    # Separate numpy arrays from JSON-serializable data
    json_payloads = []
    arrays = {}
    for i, p in enumerate(payloads):
        jp = {}
        def _arr_or_json(arr_key: str, arr: np.ndarray):
            if arr.dtype.kind == "O":
                return None, _json_safe(arr)
            arrays[arr_key] = arr
            return {"__ndarray__": arr_key}, None

        for k, v in p.items():
            if isinstance(v, np.ndarray):
                ref, fallback = _arr_or_json(f"payload_{i}_{k}", v)
                jp[k] = ref if ref is not None else fallback
            elif k == "decisions":
                jp[k] = _serialize_decisions(v)
            elif k == "fixed_decision":
                jp[k] = _serialize_decision(v) if v is not None else None
            elif k == "predict_outputs" and isinstance(v, dict):
                po = {}
                for pk, pv in v.items():
                    if isinstance(pv, np.ndarray):
                        ref, fallback = _arr_or_json(f"payload_{i}_po_{pk}", pv)
                        po[pk] = ref if ref is not None else fallback
                    else:
                        po[pk] = _json_safe(pv)
                jp[k] = po
            elif k == "train_predict_outputs" and isinstance(v, dict):
                tpo = {}
                for pk, pv in v.items():
                    if isinstance(pv, np.ndarray):
                        ref, fallback = _arr_or_json(f"payload_{i}_tpo_{pk}", pv)
                        tpo[pk] = ref if ref is not None else fallback
                    else:
                        tpo[pk] = _json_safe(pv)
                jp[k] = tpo
            else:
                jp[k] = _json_safe(v)
        json_payloads.append(jp)

    (adir / "eval_signature.json").write_text(json.dumps({"signature": sig}))
    (adir / "payloads.json").write_text(json.dumps(json_payloads, default=str))
    if arrays:
        np.savez_compressed(adir / "arrays.npz", **arrays)


def _load_eval_cache(adir: Path) -> list[dict] | None:
    """Load cached payloads for one approach."""
    try:
        payloads_raw = json.loads((adir / "payloads.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None

    arrays = {}
    arr_path = adir / "arrays.npz"
    if arr_path.exists():
        with np.load(arr_path, allow_pickle=True) as npz:
            for k in npz.files:
                arrays[k] = npz[k]

    payloads = []
    for jp in payloads_raw:
        p = {}
        for k, v in jp.items():
            if isinstance(v, dict) and "__ndarray__" in v:
                p[k] = arrays.get(v["__ndarray__"])
            elif k == "decisions":
                p[k] = _deserialize_decisions(v)
            elif k == "fixed_decision":
                p[k] = _deserialize_decision(v) if v is not None else None
            elif k in ("predict_outputs", "train_predict_outputs") and isinstance(v, dict):
                restored = {}
                for pk, pv in v.items():
                    if isinstance(pv, dict) and "__ndarray__" in pv:
                        restored[pk] = arrays.get(pv["__ndarray__"])
                    else:
                        restored[pk] = pv
                p[k] = restored
            else:
                p[k] = v
        payloads.append(p)
    return payloads


# ---- Baseline cache (random + oracle) -----------------------------------
_BASELINE_CACHE_NAME = "_baselines"


def _baseline_cache_sig(data_npz: Path, offload_ratios: list) -> str:
    import hashlib
    mtime = data_npz.stat().st_mtime_ns if data_npz.exists() else 0
    data = {"data_mtime": mtime, "offload_ratios": sorted(float(r) for r in offload_ratios)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_baseline_cache(cfg) -> "tuple[dict, dict] | None":
    """Return (random_offload, oracle_offload) if valid cache exists, else None."""
    cache_dir = cfg.output.metrics_dir / _EVAL_CACHE_DIR / _BASELINE_CACHE_NAME
    sig_path = cache_dir / "eval_signature.json"
    if not sig_path.exists():
        return None
    data_npz = cfg.output.prepared_dir / "data.npz"
    sig = _baseline_cache_sig(data_npz, cfg.offload_ratios)
    try:
        stored = json.loads(sig_path.read_text())
        if stored.get("signature") != sig:
            return None
        payload = json.loads((cache_dir / "baselines.json").read_text())
        random_offload = {float(k): v for k, v in payload["random"].items()}
        oracle_offload = {float(k): v for k, v in payload["oracle"].items()}
        return random_offload, oracle_offload
    except Exception:
        return None


def save_baseline_cache(cfg, random_offload: dict, oracle_offload: dict) -> None:
    """Persist random and oracle baseline curves to disk."""
    cache_dir = cfg.output.metrics_dir / _EVAL_CACHE_DIR / _BASELINE_CACHE_NAME
    data_npz = cfg.output.prepared_dir / "data.npz"
    sig = _baseline_cache_sig(data_npz, cfg.offload_ratios)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "random": {str(k): v for k, v in random_offload.items()},
        "oracle": {str(k): v for k, v in oracle_offload.items()},
    }
    (cache_dir / "baselines.json").write_text(json.dumps(payload))
    (cache_dir / "eval_signature.json").write_text(json.dumps({"signature": sig}))


def _json_safe(v):
    """Convert a value to JSON-serializable form."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _serialize_decision(d) -> dict | None:
    if d is None:
        return None
    return {
        "mask": d.mask.tolist(),
        "threshold": float(d.threshold),
        "target_ratio": float(d.target_ratio),
        "actual_ratio": float(d.actual_ratio),
        "n_offload": int(d.n_offload),
        "n_total": int(d.n_total),
        "metric_type": d.metric_type.value,
        "lambda_mean": None if np.isnan(d.lambda_mean) else float(d.lambda_mean),
        "lambda_final": None if np.isnan(d.lambda_final) else float(d.lambda_final),
    }


def _serialize_decisions(decisions: dict) -> dict:
    return {str(k): _serialize_decision(v) for k, v in decisions.items()}


def _deserialize_decision(d: dict | None):
    if d is None:
        return None
    from ..offloader import MetricType
    return OffloadDecision(
        mask=np.array(d["mask"], dtype=bool),
        threshold=d["threshold"],
        target_ratio=d["target_ratio"],
        actual_ratio=d["actual_ratio"],
        n_offload=d["n_offload"],
        n_total=d["n_total"],
        metric_type=MetricType(d["metric_type"]),
        lambda_mean=float("nan") if d.get("lambda_mean") is None else d["lambda_mean"],
        lambda_final=float("nan") if d.get("lambda_final") is None else d["lambda_final"],
    )


def _deserialize_decisions(data: dict) -> dict:
    return {float(k): _deserialize_decision(v) for k, v in data.items()}
