"""Phase 3 – Train: train each enabled estimator, save checkpoints."""

import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config import ApproachConfig, PipelineConfig

from ..models import get_estimator
from .. import log
from ._shared import (approach_config_signature, checkpoint_path as _checkpoint_path,
                      config_sidecar_path, resolve_paths,
                      resolve_primary_target, resolve_target, select_input)
from .eval_metrics import _safe_spearman
from .paper_eval import resolve_evaluation_seeds


def _load_prepared(cfg: PipelineConfig):
    d = np.load(cfg.output.prepared_dir / "data.npz", allow_pickle=True)
    paths_train = resolve_paths(
        (cfg.output.prepared_dir / "paths_train.txt").read_text().splitlines())
    paths_test = resolve_paths(
        (cfg.output.prepared_dir / "paths_test.txt").read_text().splitlines())
    result = {
        "X_train": d["X_train"], "X_test": d["X_test"],
        "y_train": d["y_train"], "y_test": d["y_test"],
        "edge_test": d["edge_test"], "cloud_test": d["cloud_test"],
        "paths_train": paths_train, "paths_test": paths_test,
    }
    for key in d.keys():
        if (key.startswith("y_train_") or key.startswith("y_test_")
                or key.startswith("X_train_") or key.startswith("X_test_")
                or key.startswith("meta_")):
            result[key] = d[key]

    srrm_path = cfg.output.prepared_dir / "srrm.npz"
    if srrm_path.exists():
        srrm = np.load(srrm_path)
        result["srrm_train"] = srrm["srrm_train"]
        result["srrm_test"] = srrm["srrm_test"]

    return result


def _train_input(pcfg: ApproachConfig, data: dict, split: str = 'train'):
    return select_input(pcfg, data, split)


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _serializable(value):
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _history_rows(result: dict, pcfg: ApproachConfig) -> list[dict]:
    history = result.get("fit_history")
    if not history:
        return []

    common = {
        "estimator": result.get("name", pcfg.name),
        "base_model": pcfg.registry_key,
        "stage": pcfg.stage,
        "seed": result.get("seed"),
    }

    def _row_for_epoch(epoch_idx: int, payload: dict) -> dict:
        row = {**common, "epoch": int(epoch_idx) + 1}
        for key, value in payload.items():
            if isinstance(value, (np.ndarray, list, tuple)):
                arr = np.asarray(value)
                if arr.ndim == 0:
                    row[key] = float(arr.item())
            elif isinstance(value, (np.floating, np.integer, float, int)):
                row[key] = float(value)
            else:
                row[key] = value
        return row

    rows: list[dict] = []
    if isinstance(history, list):
        for epoch_idx, item in enumerate(history):
            if isinstance(item, dict):
                rows.append(_row_for_epoch(epoch_idx, item))
            else:
                rows.append(_row_for_epoch(epoch_idx, {"value": item}))
        return rows

    if isinstance(history, dict):
        list_like = {
            key: value for key, value in history.items()
            if isinstance(value, (list, tuple, np.ndarray))
            and np.asarray(value).ndim <= 1
        }
        if list_like:
            max_len = max(len(np.asarray(v).reshape(-1)) for v in list_like.values())
            for epoch_idx in range(max_len):
                payload = {}
                for key, value in history.items():
                    if key in list_like:
                        seq = np.asarray(list_like[key]).reshape(-1)
                        if epoch_idx < len(seq):
                            payload[key] = seq[epoch_idx]
                    elif not isinstance(value, (dict, list, tuple, np.ndarray)):
                        payload[key] = value
                rows.append(_row_for_epoch(epoch_idx, payload))
            return rows

        rows.append(_row_for_epoch(0, history))
        return rows

    return []


def _sweep_row(result: dict, pcfg: ApproachConfig) -> dict:
    fit_metrics = result.get("fit_metrics", {}) or {}
    hyperparams = result.get("hyperparams", {}) or {}
    row = {
        "estimator": result.get("name", pcfg.name),
        "base_model": pcfg.registry_key,
        "stage": pcfg.stage,
        "seed": result.get("seed"),
        "status": result.get("status", ""),
        "train_time": result.get("train_time", float("nan")),
        "hyperparams_json": json.dumps(_serializable(hyperparams), sort_keys=True),
        "validation_metric": fit_metrics.get("val_r2", fit_metrics.get("best_val_loss", float("nan"))),
        "test_metric": fit_metrics.get("test_spearman", fit_metrics.get("test_r2", float("nan"))),
        "test_r2": fit_metrics.get("test_r2", float("nan")),
        "test_spearman": fit_metrics.get("test_spearman", float("nan")),
        "test_mae": fit_metrics.get("test_mae", float("nan")),
        "latency": float("nan"),
        "oracle_regret": float("nan"),
    }
    for key in ("lr", "batch_size", "epochs", "patience", "seed"):
        if key in hyperparams:
            row[key] = hyperparams[key]
    return row


def train_one(pcfg: ApproachConfig, data: dict,
              checkpoint_path: Path, cfg: PipelineConfig = None,
              seed: int | None = None) -> Dict:
    result = {"name": pcfg.name, "status": "FAIL", "seed": seed}

    try:
        if seed is not None:
            _set_global_seed(seed)
        estimator = get_estimator(pcfg.registry_key, device=getattr(cfg, "device", "auto"), **pcfg.params)
        X_train = _train_input(pcfg, data)
        y_train = resolve_target(pcfg, data, "train")

        train_kwargs = {}
        if cfg is not None:
            train_kwargs = {
                "batch_size": cfg.training.batch_size,
                "num_workers": cfg.training.num_workers,
                "epochs": cfg.training.epochs,
                "lr": cfg.training.lr,
                "patience": cfg.training.patience,
                "compile": getattr(cfg.training, "compile_models", False),
            }
        train_kwargs.update(pcfg.params)
        if seed is not None:
            train_kwargs.setdefault("seed", int(seed))

        if pcfg.loss is not None:
            train_kwargs["loss"] = pcfg.loss
        result["hyperparams"] = _serializable(train_kwargs)

        if pcfg.registry_key in ("weak_model", "strong_model"):
            estimator.fit(X_train, y_train)
            result["train_time"] = 0.0
        else:
            start = time.perf_counter()
            estimator.fit(X_train, y_train, **train_kwargs)
            result["train_time"] = time.perf_counter() - start

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic save: write to a temp file then rename so that a mid-save
        # interrupt never corrupts the previous good checkpoint.
        tmp_ckpt = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        try:
            estimator.save(tmp_ckpt)
            tmp_ckpt.replace(checkpoint_path)
        except BaseException:
            tmp_ckpt.unlink(missing_ok=True)
            raise

        # Write sidecar here, immediately after the checkpoint is committed,
        # so that a kill between save and sidecar-write does not discard a
        # valid checkpoint on the next restart.
        _sidecar = config_sidecar_path(checkpoint_path)
        _tmp_sidecar = _sidecar.with_suffix(".tmp")
        try:
            _tmp_sidecar.write_text(json.dumps({
                "signature": approach_config_signature(pcfg),
                "approach": pcfg.name,
            }))
            _tmp_sidecar.replace(_sidecar)
        except BaseException:
            _tmp_sidecar.unlink(missing_ok=True)
            raise

        if hasattr(estimator, "get_info"):
            result.update(estimator.get_info())

        fit_history = getattr(estimator, "fit_history", None)
        if fit_history:
            result["fit_history"] = _serializable(fit_history)

        if hasattr(estimator, 'fit_metrics') and estimator.fit_metrics:
            result['fit_metrics'] = _serializable(estimator.fit_metrics)

            try:
                from sklearn.metrics import mean_absolute_error, r2_score
                X_test = _train_input(pcfg, data, split='test')
                y_test = resolve_primary_target(pcfg, data, "test")
                preds = estimator.predict(X_test)
                test_mse = float(np.mean((preds - y_test) ** 2))
                test_mae = float(mean_absolute_error(y_test, preds))
                test_r2 = float(r2_score(y_test, preds))
                test_rho = float(_safe_spearman(y_test, preds))
                result['fit_metrics']['test_mse'] = round(test_mse, 6)
                result['fit_metrics']['test_mae'] = round(test_mae, 6)
                result['fit_metrics']['test_r2'] = round(test_r2, 6)
                result['fit_metrics']['test_spearman'] = round(test_rho, 6)
                log.info(f"Test  MSE={test_mse:.4f}  R\u00b2={test_r2:.4f}  "
                         f"Spearman \u03c1={test_rho:.4f}", indent=8)
            except Exception as e:
                log.warn(f"Test eval skipped: {e}", indent=8)

        result["status"] = "PASS"

    except Exception as e:
        result["error"] = str(e)
        import traceback
        traceback.print_exc()

    return result


def run(cfg: PipelineConfig) -> None:
    """Execute the training phase."""
    with log.phase_timer(3):
        if not (cfg.output.prepared_dir / "data.npz").exists():
            raise FileNotFoundError("Prepared data not found. Run 'prepare' first.")

        data = _load_prepared(cfg)
        n_train = len(data["X_train"])
        n_test = len(data["X_test"])
        log.kv_group([
            ("Train samples", log.fmt_count(n_train)),
            ("Test samples", log.fmt_count(n_test)),
            ("Total", log.fmt_count(n_train + n_test)),
        ])

        if not data["X_train"].any():
            log.warn("All tabular features (X_train) are ZERO!")
            log.info("This usually means the prepare phase used a conf_threshold", indent=6)
            log.info("that filtered out all detections. Re-run prepare with", indent=6)
            log.info("the correct conf_threshold (e.g. 0.3).", indent=6)

        approaches = cfg.enabled_approaches()
        seeds = resolve_evaluation_seeds(cfg)
        use_seeded_checkpoints = len(seeds) > 1
        if use_seeded_checkpoints:
            log.kv("Seeds", ", ".join(str(seed) for seed in seeds))

        # Pre-scan: classify every (approach, seed) pair as cached or needs training.
        # This is cheap (file stat + small JSON read) and lets us print a full
        # overview before the first training job starts.
        cached_items: list = []  # (pcfg, seed, ckpt, title)
        train_items: list = []   # (pcfg, seed, ckpt, stale:bool, title)
        for pcfg in approaches:
            current_sig = approach_config_signature(pcfg)
            for seed in seeds:
                ckpt = _checkpoint_path(cfg, pcfg, seed if use_seeded_checkpoints else None)
                sidecar = config_sidecar_path(ckpt)
                title = pcfg.name if not use_seeded_checkpoints else f"{pcfg.name} [seed={seed}]"

                is_cached = False
                stale = False
                if ckpt.exists() and not cfg.force_retrain:
                    if sidecar.exists():
                        try:
                            is_cached = (
                                json.loads(sidecar.read_text()).get("signature") == current_sig
                            )
                        except (json.JSONDecodeError, OSError):
                            pass
                    if not is_cached:
                        stale = True  # checkpoint exists but sig changed
                if is_cached:
                    cached_items.append((pcfg, seed, ckpt, title))
                else:
                    train_items.append((pcfg, seed, ckpt, stale, title))

        n_total = len(cached_items) + len(train_items)
        log.section(f"Training Overview — {n_total} Approaches")
        log.kv_group([
            ("Total",          n_total),
            ("Cache hits",     len(cached_items)),
            ("Needs training", len(train_items)),
        ])

        train_log: List[Dict] = []
        history_rows: List[dict] = []
        sweep_rows: List[dict] = []
        pass_count = 0
        cache_count = 0
        fail_count = 0

        if cached_items:
            log.section(f"Cached ({len(cached_items)})")
            for pcfg, seed, ckpt, title in cached_items:
                log.cached(title)
                cached_row = {"name": pcfg.name, "seed": seed, "status": "CACHED"}
                train_log.append(cached_row)
                sweep_rows.append(_sweep_row(cached_row, pcfg))
                cache_count += 1

        if train_items:
            n_train_items = len(train_items)
            log.section(f"Training ({n_train_items})")
            for idx, (pcfg, seed, ckpt, stale, title) in enumerate(train_items, 1):
                progress = f"[{idx}/{n_train_items}]"
                if stale:
                    log.warn(f"{progress} {title}: config changed, retraining (stale checkpoint)")
                log.subsection(f"{progress} {title}")
                result = train_one(pcfg, data, ckpt, cfg, seed=seed)
                train_log.append(result)
                history_rows.extend(_history_rows(result, pcfg))
                sweep_rows.append(_sweep_row(result, pcfg))

                if result["status"] == "PASS":
                    t = result.get("train_time", 0)
                    log.success(f"Saved {ckpt.name} ({t:.1f}s)", indent=8)
                    pass_count += 1
                else:
                    log.fail(f"{result.get('error')}", indent=8)
                    fail_count += 1

        log.section("Training Summary")
        log.kv_group([
            ("Trained", pass_count),
            ("Cached", cache_count),
            ("Failed", fail_count),
        ])

        log_path = cfg.output.metrics_dir
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "train_log.json").write_text(
            json.dumps([_serializable(row) for row in train_log], indent=2, default=str))
        log.arrow(f"{log_path / 'train_log.json'}")

        if history_rows:
            history_df = pd.DataFrame(history_rows)
            history_df.to_csv(log_path / "loss_component_history.csv", index=False)
            log.arrow(f"{log_path / 'loss_component_history.csv'}")
        elif (log_path / "loss_component_history.csv").exists():
            (log_path / "loss_component_history.csv").unlink()

        if sweep_rows:
            sweep_df = pd.DataFrame(sweep_rows)
            sweep_df.to_csv(log_path / "sweep_results.csv", index=False)
            log.arrow(f"{log_path / 'sweep_results.csv'}")
        elif (log_path / "sweep_results.csv").exists():
            (log_path / "sweep_results.csv").unlink()
