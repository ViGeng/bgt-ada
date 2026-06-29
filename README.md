# Budget-Adaptive Routing — Skipping the Weak When the Strong Answers Anyway

Artifacts for the paper **"Budget-Adaptive Routing: Skipping the Weak When the
Strong Answers Anyway"** (Wei Geng, Nitinder Mohan, Jörg Ott), to appear at the
**SIGCOMM '26 Workshop on Networks for AI Computing (NAIC '26)**.

> **TL;DR.** In edge–cloud object-detection offloading, prior systems run the
> weak detector on *every* frame and only *then* decide whether to escalate to
> the cloud. We show that a tiny **skipping** estimator (0.15 GFLOPs, ~29× lighter
> than the weak detector) can read the offload decision *directly from the raw
> image* — so the weak pass can be skipped on frames that will be offloaded
> anyway. Neither placement wins everywhere, so a **budget-adaptive** router
> picks between *skipping* and *conditioned* by the offload budget ρ, tracing the
> upper accuracy envelope of both while cutting up to **19.1 ms (~30%)** of
> per-frame latency. At its peak it is **+1.7 pp mAP over the strong model
> itself**, with far less compute.

`ASIDE` is the system name used in the code and the paper.

---

## What's in this repository

This is a **curated, paper-focused subset** of a larger research pipeline. It
contains exactly the methods, code, and result artifacts behind the paper — not
the full research codebase (which additionally explores scenario-adaptive
routing, error-decomposition targets, more datasets and backbones; none of that
is part of the paper).

```
bgt-ada/
├── run_pipeline.py            # CLI entry point: detect → prepare → train → evaluate → analyse
├── config/                    # pipeline + dataset + approach catalog (trimmed to the paper)
│   ├── approaches.py          # the 17 paper approaches (5 headline + 11 target ablations + 1 backbone)
│   ├── datasets.py            # PASCAL VOC (weak/strong = MobileNetV3-FPN / ResNet50-FPN-v2)
│   ├── pipeline.py, schema.py # global defaults, dataclasses, output layout
│   └── ...
├── src/
│   ├── detector.py            # torchvision Faster R-CNN weak/strong detectors
│   ├── proxy_metrics.py       # ORIC / MORIC family (dataset-wide swap reward)
│   ├── losses.py              # focal, sign-rank-huber, wing, quantile, weighted-MSE, …
│   ├── offloader.py           # budget-tracking thresholders (ECDF-calibrated, native, fixed)
│   ├── models/                # estimators: MobileNetV2-Lite/EfficientNet-B0-Lite, XGBoost, EdgeML, DCSB
│   └── phases/                # the 5 pipeline phases
├── tests/                     # focused unit tests for the released code
├── results/voc/               # ← VOC evaluation artifacts (paper methods only, seed 42)
│   ├── metrics/               # the CSV/JSON numbers behind the paper tables
│   └── prepared/metadata.json # split provenance (N_test = 3105, seed 42, detectors)
├── paper/
│   ├── data/                  # tidy per-figure CSVs (paper methods only)
│   └── figures/               # scripts + PDFs that regenerate every data figure
└── checkpoints/               # trained estimator weights (see checkpoints/README.md)
```

---

## The idea in one figure's worth of words

For each frame the router emits a binary decision *keep-local* (weak detector
`M_w`) vs *offload* (strong detector `M_s`), subject to an **offload budget** ρ
(the fraction of frames allowed to the cloud).

* **Conditioned** routing (all prior work): estimator fires *after* `M_w`, on its
  proposal features. The weak pass runs on every frame — an **implicit compute
  tax** of up to `ρ·C_w` paid on frames that get offloaded anyway.
* **Skipping** routing (ours): estimator fires *before* `M_w`, from raw pixels
  (a 0.15 GFLOPs MobileNetV2-Lite). The weak pass is skipped on offloaded frames.
* **Budget-adaptive** routing (ours): two offline thresholds
  `ρ_frontier = 0.3`, `ρ_ceiling = 0.8` select conditioned for `ρ ≤ 0.2` and
  `ρ ≥ 0.8`, skipping for `0.3 ≤ ρ ≤ 0.7` — the per-budget winner.

The reason skipping is even possible: predicting **whether** a frame benefits
from the cloud is far cheaper than predicting **what** it contains. We train on
a binary `OffloadBin` target (does offloading raise dataset-wide AP for this
frame?) with focal loss, which decouples hardness from content.

---

## Headline results (PASCAL VOC, mAP@0.5, seed 42)

Routing quality across all estimators (paper Tab. 2). `ρs` is Spearman vs. the
per-frame oracle; `AUCρ` is the area under the mAP–ρ curve.

| Estimator | ρs | Peak mAP | AUCρ | Est. GFLOPs |
| --- | :-: | :-: | :-: | :-: |
| *Always weak* | 0.000 | 0.760 | 0.760 | 0 |
| *Always strong* | 0.000 | 0.791 | 0.791 | 0 |
| *Uniform random* | 0.000 | — | 0.777 | 0 |
| *Oracle (ΔAP)* | — | 0.827 | 0.816 | — |
| EdgeML *(conditioned)* | 0.138 | 0.793 | 0.784 | ≈0 |
| DCSB *(conditioned)* | 0.359 | 0.789 | — | ≈0 |
| **XGBoost / MORIC (ours, conditioned)** | 0.472 | 0.804 | 0.794 | ≈0 |
| MobileNetV2-Lite + MORIC⁺ *(skipping, ours)* | 0.350 | 0.801 | 0.790 | 0.15 |
| **MobileNetV2-Lite + OffloadBin *(skipping, ours)*** | **0.557** | **0.808** | **0.795** | 0.15 |
| **Budget-adaptive (OffloadBin ↔ MORIC, ours)** | — | **0.808** | **0.796** | 0.15† |

† conditioned branch additionally pays the weak pass `C_w = 4.49` GFLOPs.

* **Compute:** skipping saves `(1−ρ)→` up to **3.9 GFLOPs/frame** at ρ=0.9
  (≈ one full weak pass); break-even at ρ\* ≈ 0.03.
* **Latency:** up to **19.1 ms (~30%)** lower at ρ=0.9 (weak pass is a serial
  24.70 ms dependency for conditioned routing; skipping runs a 3.08 ms estimator
  instead). Crossover at ρ\* ≈ 0.12.
* **Accuracy:** the adaptive envelope peaks at **0.808 mAP@0.5 = +1.7 pp over the
  strong model** (0.791).

Detector setup (profiled on an NVIDIA A40):

| | Model | GFLOPs | Latency | mAP@0.5 |
| --- | --- | :-: | :-: | :-: |
| Weak `M_w` | `fasterrcnn_mobilenet_v3_large_fpn` | 4.49 | 24.70 ms | 0.760 |
| Strong `M_s` | `fasterrcnn_resnet50_fpn_v2` | 280.37 | 44.12 ms | 0.791 |

---

## Method components (where to look in the code)

**Learning targets / proxy metrics** — `src/proxy_metrics.py`,
`src/phases/prepare_derive.py`, `src/phases/prepare_transforms.py`,
`src/phases/prepare_split.py`. All CDF references are fit on the **train split
only** (no test leakage).

| Target | As implemented | Paper |
| --- | --- | --- |
| ΔAP / `gain_*` | Per-frame cloud−weak AP delta (raw signal). | Fig. target dist. |
| ORIC | Dataset-wide AP change from swapping one frame's detections weak→cloud over a single global PR curve. | basis of MORIC |
| MORIC | Train-ECDF of ORIC into (0,1]. | EdgeML / XGBoost target |
| MORIC⁺ | Sign-anchored split-CDF of ORIC onto [−1,1]. | Eq. (3) |
| **OffloadBin** | `1[ORIC > 0]` — does the cloud strictly help this frame? Focal loss. | Eq. (4), **best** |
| TopQuartile | `1[ORIC > train-P75]`. | Tab. 5 |
| MORIC\*, Φ-MORIC, SigMORIC | Sign-aware reshapings of MORIC (shift / probit / sigmoid). | Tab. 5 |
| HighIoUGain, F1Gain, RescueRatio, WorstCaseGain | Alternative per-frame gain proxies. | Tab. 5 |
| count_gain (DCSB) | Δ(#detections ≥ 0.5 conf), weak→cloud. | DCSB baseline |

**Estimators** — `src/models/`:
`mobilenet_v2` → `MobileNetV2LiteEstimator` (torchvision MobileNetV2 truncated to
`features[:14]`, 128×128 input, **0.15 GFLOPs / 0.54 M params**);
`efficientnet_b0_lite` → `EfficientNetB0LiteEstimator` (backbone ablation,
Tab. 4); `xgboost` → `XGBoostEstimator` (conditioned, ours); `edgeml` →
`EdgeMLEstimator` (MLP on top-25 proposals); `dcsb` → `DCSBOriginalEstimator`
(fixed count/area rule); weak/strong/oracle reference points → `src/models/virtual.py`.

**Offloaders (budget control)** — `src/offloader.py`. `online_ecdf_calibrated`
calibrates scores against the train ECDF and thresholds online at `1−ρ` with
budget-debt tie-breaking (no test-set lookahead); `native_threshold` uses the
metric's analytic threshold (EdgeML); `fixed_classifier` is DCSB's single fixed
operating point.

**Budget-adaptive arbiter.** The pipeline evaluates each placement
*independently* across the ρ sweep. The two-threshold arbiter — `α(ρ)` in the
paper — is realised **offline** as the per-ρ max envelope of the skipping and
conditioned curves; see `paper/figures/map_vs_rho.py`, which computes the
envelope and reads off `ρ_frontier`/`ρ_ceiling`. At runtime the arbiter is a
constant-time lookup on ρ.

---

## Reproducing the results

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PASCAL VOC is pulled from the Hugging Face hub on first run (`src/dataset.py`);
no manual download is needed. A CUDA GPU is recommended for `detect`/`train`.

### 2. Run the pipeline

The default approach catalog **is** the paper subset (17 approaches), so a plain
run reproduces every paper number:

```bash
python run_pipeline.py --phase all --dataset voc --seed 42
```

The five phases and their outputs:

| Phase | What it does | Writes |
| --- | --- | --- |
| `detect` | cache weak & strong detections | derived detection cache |
| `prepare` | derive proxy-metric targets, features, train/test split | prepared payloads |
| `train` | fit the configured estimators | `checkpoints/` |
| `evaluate` | routing simulation over the ρ sweep | `results/.../metrics/*.csv` |
| `analyse` | paper tables, charts, `report.txt` | `results/.../charts`, `report.txt` |

Run phases individually with e.g. `--phase train evaluate --dataset voc`, restrict
to specific approaches with `--approaches '<name>' …`, and force retraining with
`--force`. The released `checkpoints/` let you skip `train` and run
`--phase evaluate analyse` — point the pipeline's working dir at them first
(`ln -s ../../../checkpoints results/full_eval/voc/checkpoints`); see
`checkpoints/README.md`.

### 3. Regenerate the figures

After evaluation (or directly from the shipped `paper/data/` CSVs):

```bash
cd paper/figures
python map_vs_rho.py          # adaptive envelope + ρ_frontier/ρ_ceiling
python compute_savings.py     # GFLOPs & latency vs ρ
python target_distribution.py # ΔAP / MORIC⁺ / OffloadBin target distributions
python slice_benefit.py       # where offloading helps, by weak-detector stratum
```

Each writes a `*.pdf` next to the script (the shipped PDFs are exactly these
outputs) and a preview PNG under `png/`.

---

## Released artifacts → paper tables/figures

**`results/voc/metrics/`** (filtered to the paper methods + reference points):

| File | Backs |
| --- | --- |
| `main_benchmark_table.csv` | Tab. 2 (routing quality), Tab. 4/5 peaks |
| `statistical_summary_table.csv` | Tab. 2 (means / CIs) |
| `offloading_results.csv` | per-ρ mAP sweep (oracle / random reference curves) |
| `offloading_summary.csv` | AUCρ for the reference points |
| `estimator_metrics.csv` | weak/strong detector specs (Tab. 1) |
| `slice_opportunity.csv` | Tab. 6 / slice-benefit figure (per-stratum benefit) |
| `dataset_summary.json` | dataset-wide AP, detection counts (Tab. 1) |

**`paper/data/`** — tidy per-figure CSVs (the exact numbers the figure scripts
consume), incl. the per-ρ curves for the five paper estimators and the
`adaptive_envelope.csv`.

---

## Notes & caveats

* **Curated subset.** A few modules (`config/scenarios.py`,
  `src/error_decomposition.py`, `src/conditional_rewards.py`,
  `src/phases/analyse_scenarios.py`) are retained because the paper pipeline
  imports them, but their out-of-paper code paths stay dormant for the VOC run
  (gated by the active approach set). Mentions of "scenario"/"LCER" in code
  comments refer to those dormant branches, not to anything evaluated in the paper.
* **Checkpoints.** Trained weights live in `checkpoints/` (see its README). If
  the directory is empty, regenerate them with the `train` phase.
* **Figure colors.** The figure scripts use a vendored, dependency-free
  `theme.py`; regenerated figures are visually equivalent to but not
  byte-identical with the typeset versions (the canonical PDFs are shipped).
* **Result snapshot.** `results/voc/metrics/` is the full-evaluation export;
  `paper/data/` is the per-figure snapshot the paper figures were typeset from.
  Small numeric differences between the two reflect different evaluation
  snapshots; the paper tables are the canonical values.

---

## Citation

```bibtex
@inproceedings{geng2026budget,
  title     = {Budget-Adaptive Routing: Skipping the Weak When the Strong Answers Anyway},
  author    = {Geng, Wei and Mohan, Nitinder and Ott, J{\"o}rg},
  booktitle = {Proceedings of the Workshop on Networks for AI Computing (NAIC '26)},
  year      = {2026},
  doi       = {10.1145/3789240.3828740},
}
```

## License

Code and artifacts are released under the [MIT License](LICENSE). © 2026 Wei Geng.

## Acknowledgements

Supported by the Dutch National Growth Fund "Future Network Services".
