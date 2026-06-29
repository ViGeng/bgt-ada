#!/usr/bin/env python3
"""Unified entry point for the 5-phase pipeline.

Usage:
    python run_pipeline.py --phase all                 # run all 5 phases
    python run_pipeline.py --phase detect              # phase 1 only
    python run_pipeline.py --phase prepare             # phase 2 only
    python run_pipeline.py --phase train               # phase 3 only
    python run_pipeline.py --phase evaluate            # phase 4 only
    python run_pipeline.py --phase analyse             # phase 5 only
    python run_pipeline.py --phase train evaluate      # phases 3 + 4

Options:
    --phase PHASE [PHASE]  Phase(s) to run: detect, prepare, train, evaluate, analyse, or 'all'
    --config CONFIG.json   Load config from file
    --dataset NAME [NAME]  Dataset name(s): voc (or 'all')
    --data-root DIR        Dataset root directory (only with single dataset)
    --sample FRACTION      Override sample fraction (0-1)
    --seed SEED            Random seed
    --force                Force retrain even if checkpoints exist
    --approaches a b c     Only run these approaches
"""

import argparse
import sys
import time
import urllib.request

from config import DATASETS, DatasetConfig, PipelineConfig
from src.log import (config_block, dataset_separator, pipeline_banner,
                     pipeline_done)

NTFY_TOPIC = "pipeline-video-det"


def _ntfy(title: str, message: str, priority: str = "default") -> None:
    """Send a notification via ntfy.sh."""
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode(),
            headers={"Title": title, "Priority": priority},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[ntfy] notification failed: {e}")
from config.datasets import _BASE_DATASETS

PHASES = {"detect", "prepare", "train", "evaluate", "analyse", "all"}
ORDERED = ["detect", "prepare", "train", "evaluate", "analyse"]

# --dataset all runs the base datasets only (not model-pair variants)
ALL_DATASETS = list(_BASE_DATASETS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the 5-phase pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--phase", nargs="+", default=["all"], type=str,
                   help=f"Phase(s) to run: {', '.join(sorted(PHASES))}")
    p.add_argument("--config", type=str, help="Path to config JSON")
    p.add_argument("--dataset", nargs="+", type=str,
                   help="Dataset name(s): voc (or 'all')")
    p.add_argument("--data-root", type=str, help="Dataset root directory")
    p.add_argument("--sample", type=float, help="Sample fraction (0-1)")
    p.add_argument("--seed", type=int, help="Random seed")
    p.add_argument("--force", action="store_true", help="Force retrain")
    p.add_argument("--approaches", nargs="+", help="Subset of approaches")
    p.add_argument("--device", type=str, help="Device to use (e.g., 'cuda:0' or 'cpu')")
    return p.parse_args()


def _build_configs(args) -> list:
    """Build one PipelineConfig per dataset requested."""
    # Resolve dataset list
    datasets = None
    if args.dataset:
        datasets = []
        for d in args.dataset:
            if d.lower() == "all":
                datasets.extend(ALL_DATASETS)
            else:
                datasets.append(d.lower())
        # deduplicate while preserving order
        seen = set()
        datasets = [d for d in datasets if not (d in seen or seen.add(d))]

    if not datasets:
        print("Error: No dataset specified. Please specify using --dataset (e.g., --dataset voc or 'all')")
        sys.exit(1)

    configs = []
    for ds_name in datasets:
        cfg = (PipelineConfig.load(args.config) if args.config
               else PipelineConfig())

        if ds_name:
            cfg.dataset.name = ds_name
            if args.data_root:
                cfg.dataset.root = args.data_root
            # Rebuild output-dependent dirs for this dataset
            cfg.output.base_dir = ""
            cfg.output.data_dir = ""
            cfg.__post_init__()

        if args.sample is not None:
            cfg.dataset.sample_fraction = args.sample
        if args.seed is not None:
            cfg.seed = args.seed
        if args.force:
            cfg.force_retrain = True
        if args.approaches:
            for p in cfg.approaches:
                p.enabled = p.name in args.approaches
        if args.device:
            cfg.device = args.device

        configs.append(cfg)
    return configs


def _run_phases(cfg: PipelineConfig, phases_to_run: list):
    """Run the requested phases for a single config."""
    for phase in phases_to_run:
        if phase == "detect":
            from src.phases.detect import run as run_detect
            run_detect(cfg)
        elif phase == "prepare":
            from src.phases.prepare import run as run_prepare
            run_prepare(cfg)
        elif phase == "train":
            from src.phases.train import run as run_train
            run_train(cfg)
        elif phase == "evaluate":
            from src.phases.evaluate import run as run_evaluate
            run_evaluate(cfg)
        elif phase == "analyse":
            from src.phases.analyse import run as run_analyse
            run_analyse(cfg)


class TeeLogger:
    def __init__(self, filename):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log_file = open(filename, "w")
        sys.stdout = self
        sys.stderr = self

    def write(self, message):
        self.terminal_stdout.write(message)
        # tqdm refreshes progress bars with carriage returns; keep those on the
        # live terminal but skip them in the persisted log file.
        if "\r" in message:
            return
        self.log_file.write(message)

    def flush(self):
        self.terminal_stdout.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal_stdout.isatty()

    def close(self):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        self.log_file.close()


def main():
    args = parse_args()

    for ph in args.phase:
        if ph not in PHASES:
            print(f"Unknown phase: '{ph}'. Choose from {sorted(PHASES)}")
            sys.exit(1)

    run_all = "all" in args.phase
    phases_to_run = ORDERED if run_all else [p for p in ORDERED
                                              if p in args.phase]

    configs = _build_configs(args)

    dataset_names = ", ".join(cfg.dataset.name for cfg in configs)
    pipeline_t0 = time.perf_counter()
    try:
        for i, cfg in enumerate(configs):
            # Create output dir if it doesn't exist to store the log
            import os
            os.makedirs(cfg.output.base_dir, exist_ok=True)
            log_path = os.path.join(cfg.output.base_dir, "run.log")
            logger = TeeLogger(log_path)

            try:
                if len(configs) > 1:
                    dataset_separator(i + 1, len(configs), cfg.dataset.name)

                pipeline_banner(phases_to_run, cfg.dataset.name,
                                cfg.output.base_dir)
                config_block(cfg)

                _run_phases(cfg, phases_to_run)

            finally:
                logger.close()

        elapsed = time.perf_counter() - pipeline_t0
        pipeline_done()
        _ntfy("Pipeline finished", f"phases={phases_to_run} datasets={dataset_names}")
    except Exception as exc:
        _ntfy("Pipeline FAILED", f"phases={phases_to_run} datasets={dataset_names}\n{exc}", priority="high")
        raise

if __name__ == "__main__":
    main()
