"""Centralized pipeline logging with structured, readable output.

Provides consistent formatting primitives for phase headers, sections,
key-value displays, tables, status indicators, and timing.
All output goes through print() so TeeLogger captures it.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ── Box-drawing characters ────────────────────────────────────────────
_H = "─"  # horizontal
_V = "│"  # vertical
_TL = "┌"  # top-left
_TR = "┐"  # top-right
_BL = "└"  # bottom-left
_BR = "┘"  # bottom-right
_VR = "├"  # vertical-right (left tee)
_VL = "┤"  # vertical-left  (right tee)
_HD = "┬"  # horizontal-down
_HU = "┴"  # horizontal-up
_CR = "┼"  # cross

# Status markers
OK = "✓"
FAIL = "✗"
SKIP = "○"
WARN = "!"
ARROW = "→"
BULLET = "•"
CACHED = "◆"

# Standard widths
W = 76  # default box width (inner content)
W_WIDE = 120  # wide table width


def _box_top(width: int = W) -> str:
    return f"  {_TL}{_H * (width + 2)}{_TR}"


def _box_bottom(width: int = W) -> str:
    return f"  {_BL}{_H * (width + 2)}{_BR}"


def _box_mid(width: int = W) -> str:
    return f"  {_VR}{_H * (width + 2)}{_VL}"


def _box_line(text: str, width: int = W) -> str:
    stripped = text[:width]
    return f"  {_V} {stripped:<{width}} {_V}"


# ── Phase headers ─────────────────────────────────────────────────────

PHASE_NAMES = {
    1: "DETECT",
    2: "PREPARE",
    3: "TRAIN",
    4: "EVALUATE",
    5: "ANALYSE",
}


def phase_header(phase_num: int, subtitle: str = "") -> None:
    """Print a prominent phase banner."""
    name = PHASE_NAMES.get(phase_num, f"PHASE {phase_num}")
    title = f"PHASE {phase_num} {_H * 3} {name}"
    if subtitle:
        title += f"  {_V}  {subtitle}"
    print()
    print(f"  {_TL}{_H * (W + 2)}{_TR}")
    print(_box_line(title))
    print(f"  {_BL}{_H * (W + 2)}{_BR}")


def phase_complete(phase_num: int, elapsed: Optional[float] = None) -> None:
    """Print phase completion with optional elapsed time."""
    name = PHASE_NAMES.get(phase_num, f"Phase {phase_num}")
    time_str = f" ({_fmt_duration(elapsed)})" if elapsed is not None else ""
    print(f"\n  {OK} {name} complete{time_str}")
    print()


# ── Section headers ───────────────────────────────────────────────────

def section(title: str) -> None:
    """Print a section divider within a phase."""
    line = f"{_H * 2} {title} "
    line = line + _H * max(0, 60 - len(line))
    print(f"\n  {line}")


def subsection(title: str) -> None:
    """Print a lighter sub-section header."""
    print(f"\n  {_VR}{_H} {title}")


# ── Key-value display ─────────────────────────────────────────────────

def kv(key: str, value: Any, indent: int = 4) -> None:
    """Print a single key-value pair."""
    pad = " " * indent
    print(f"{pad}{key:<24} {value}")


def kv_group(pairs: Sequence[Tuple[str, Any]], indent: int = 4) -> None:
    """Print multiple key-value pairs."""
    for k, v in pairs:
        kv(k, v, indent)


# ── Status lines ──────────────────────────────────────────────────────

def info(msg: str, indent: int = 4) -> None:
    """Informational line."""
    print(f"{' ' * indent}{msg}")


def status(label: str, msg: str, marker: str = BULLET, indent: int = 4) -> None:
    """Status line with a marker."""
    print(f"{' ' * indent}{marker} {label}: {msg}")


def success(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{OK} {msg}")


def fail(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{FAIL} {msg}")


def warn(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{WARN} {msg}")


def skip(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{SKIP} {msg}")


def cached(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{CACHED} {msg}")


def arrow(msg: str, indent: int = 4) -> None:
    print(f"{' ' * indent}{ARROW} {msg}")


# ── Tables ────────────────────────────────────────────────────────────

def table(headers: List[str], rows: List[List[str]],
          col_widths: Optional[List[int]] = None,
          alignments: Optional[List[str]] = None,
          indent: int = 4) -> None:
    """Print a box-drawn table.

    alignments: list of '<' (left) or '>' (right) per column.
    """
    n_cols = len(headers)
    if col_widths is None:
        col_widths = []
        for i in range(n_cols):
            col_w = len(headers[i])
            for row in rows:
                if i < len(row):
                    col_w = max(col_w, len(str(row[i])))
            col_widths.append(col_w + 1)  # +1 for padding
    if alignments is None:
        alignments = ["<"] * n_cols

    pad = " " * indent
    total_w = sum(col_widths) + n_cols + 1  # +1 for each separator + end

    # Top border
    segs = [_H * (w) for w in col_widths]
    print(f"{pad}{_TL}{_HD.join(segs)}{_TR}")

    # Header row
    cells = []
    for i, h in enumerate(headers):
        w = col_widths[i]
        a = alignments[i]
        cells.append(f"{h:{a}{w}}")
    print(f"{pad}{_V}{'│'.join(cells)}{_V}")

    # Header separator
    segs = [_H * (w) for w in col_widths]
    print(f"{pad}{_VR}{_CR.join(segs)}{_VL}")

    # Data rows
    for row in rows:
        cells = []
        for i in range(n_cols):
            w = col_widths[i]
            a = alignments[i]
            val = str(row[i]) if i < len(row) else ""
            cells.append(f"{val:{a}{w}}")
        print(f"{pad}{_V}{'│'.join(cells)}{_V}")

    # Bottom border
    segs = [_H * (w) for w in col_widths]
    print(f"{pad}{_BL}{_HU.join(segs)}{_BR}")


# ── Metric formatting ─────────────────────────────────────────────────

def fmt_metric(value: float, precision: int = 4, width: int = 0) -> str:
    """Format a metric value, returning '-' for NaN."""
    import math
    if math.isnan(value):
        return f"{'-':>{width}}" if width else "-"
    s = f"{value:.{precision}f}"
    return f"{s:>{width}}" if width else s


def fmt_pct(value: float, precision: int = 1) -> str:
    """Format as percentage."""
    return f"{value * 100:.{precision}f}%"


def fmt_count(n: int) -> str:
    """Format large numbers with comma separators."""
    return f"{n:,}"


# ── Timing ────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


@contextmanager
def timed(label: str = "", indent: int = 4, show_start: bool = False):
    """Context manager that measures and prints elapsed time.

    Yields a dict with 'elapsed' key (populated after exit).
    """
    result = {"elapsed": 0.0}
    if show_start and label:
        info(f"{label} ...", indent)
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - t0
        if label:
            info(f"{label} ({_fmt_duration(result['elapsed'])})", indent)


@contextmanager
def phase_timer(phase_num: int):
    """Context manager that prints phase header on entry and completion on exit."""
    phase_header(phase_num)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        phase_complete(phase_num, elapsed)


# ── Pipeline banner ───────────────────────────────────────────────────

def pipeline_banner(phases: List[str], dataset: str,
                    output_dir: str) -> None:
    """Print the main pipeline startup banner."""
    phase_str = " → ".join(p.capitalize() for p in phases)
    print()
    print(f"  {_TL}{_H * (W + 2)}{_TR}")
    print(_box_line(f"SELECTIVE OFFLOADING PIPELINE"))
    print(f"  {_VR}{_H * (W + 2)}{_VL}")
    print(_box_line(f"Dataset    {dataset}"))
    print(_box_line(f"Phases     {phase_str}"))
    print(_box_line(f"Output     {output_dir}"))
    print(f"  {_BL}{_H * (W + 2)}{_BR}")


def dataset_separator(index: int, total: int, name: str) -> None:
    """Print a dataset separator for multi-dataset runs."""
    print()
    print(f"  {'━' * (W + 4)}")
    print(f"  ┃  DATASET {index}/{total}: {name.upper()}")
    print(f"  {'━' * (W + 4)}")


def config_block(cfg) -> None:
    """Print the configuration summary block."""
    section("Configuration")

    subsection("Hyperparameters")
    kv_group([
        ("sample_fraction", cfg.dataset.sample_fraction),
        ("test_ratio", cfg.dataset.test_ratio),
        ("seed", cfg.seed),
        ("batch_size", cfg.training.batch_size),
        ("num_workers", cfg.training.num_workers),
        ("epochs", cfg.training.epochs),
        ("learning_rate", cfg.training.lr),
        ("patience", cfg.training.patience),
        ("compile_models", cfg.training.compile_models),
    ])

    approaches = cfg.enabled_approaches()
    subsection(f"Enabled Approaches ({len(approaches)})")

    headers = ["Name", "Type", "Stage", "Params"]
    rows = []
    for p in approaches:
        params_str = str(p.params) if p.params else "defaults"
        rows.append([p.name, p.feature_type, p.stage, params_str])
    table(headers, rows,
          col_widths=[28, 10, 8, 30],
          alignments=["<", "<", "<", "<"])


def pipeline_done() -> None:
    """Print the final pipeline completion message."""
    print(f"\n  {OK} Pipeline complete.")
    print()
