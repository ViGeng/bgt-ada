# Trained checkpoints

Trained estimator checkpoints for the PASCAL VOC paper run (seed 42), so the
evaluation can be reproduced without retraining. There is **one checkpoint per
learned approach in the paper subset** (17 total), matching `config/approaches.py`.

Each weight file ships with a `.config.json` sidecar holding the training
signature (used by the `train` phase to detect stale checkpoints). Filenames are
the pipe-delimited approach IDs the pipeline assigns (`<stage>|<model>|<target>|<loss>|<offloader>`);
keep them verbatim — the loader reconstructs exactly these names.

## Contents

**Headline methods — Tab. 2**

| File | Estimator / target | Paper role |
| --- | --- | --- |
| `pre\|mobilenet_v2\|OffloadBin\|focal\|online_ecdf_calibrated.pt` | MobileNetV2-Lite, OffloadBin / focal | **skipping, headline (best)** |
| `pre\|mobilenet_v2\|MORIC+-AP\|online_ecdf_calibrated.pt` | MobileNetV2-Lite, MORIC⁺ | skipping |
| `post\|xgboost\|MORIC-AP\|online_ecdf_calibrated.pkl` | XGBoost on MORIC | conditioned, ours |
| `post\|edgeml\|MORIC-AP\|wmse\|native_threshold.pt` | EdgeML MLP on MORIC | conditioned baseline |
| `post\|dcsb\|CountGain-05\|fixed_classifier.pt` | DCSB count/area rule | conditioned baseline † |

† DCSB is a fixed rule; its `.pt` only stores the operating point, not trained weights.

**Learning-target ablations on the fixed MobileNetV2-Lite trunk — Tab. 5** (11)

| File |
| --- |
| `pre\|mobilenet_v2\|TopQuartile\|focal\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|HighIoUGain\|huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|MORIC+-AP\|quantile_75\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|F1Gain\|huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|MORIC+-AP\|wing\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|RescueRatio\|huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|RescueRatio\|wing\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|WorstCaseGain\|huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|SigMORIC-AP\|sign_rank_huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|MORICSTAR-AP\|sign_rank_huber\|online_ecdf_calibrated.pt` |
| `pre\|mobilenet_v2\|PhiMORIC-AP\|sign_rank_huber\|online_ecdf_calibrated.pt` |

**Backbone ablation — Tab. 4** (1)

| File | Estimator / target |
| --- | --- |
| `pre\|efficientnet_b0_lite\|OffloadBin\|focal\|online_ecdf_calibrated.pt` | EfficientNet-B0-Lite, OffloadBin / focal |

## Using the released checkpoints (skip `train`)

The `evaluate` phase loads checkpoints from the pipeline's working output dir —
`results/full_eval/voc/checkpoints/` for `--dataset voc` (see
`OutputConfig.checkpoints_dir`), **not** from this directory. That working dir is
gitignored scratch, so point it at these released weights once:

```bash
# from the repo root
mkdir -p results/full_eval/voc
ln -s ../../../checkpoints results/full_eval/voc/checkpoints   # or: cp -r checkpoints results/full_eval/voc/

python run_pipeline.py --phase evaluate analyse --dataset voc --seed 42
```

If this directory is empty, the checkpoints have not been attached to the
release. You can always regenerate them from scratch with the `train` phase —
see the top-level `README.md` (Reproducing the results).
