# Trained checkpoints

This directory holds the trained estimator checkpoints for the PASCAL VOC paper
run (seed 42), so the evaluation can be reproduced without retraining.

Expected contents (one per learned estimator in the paper subset):

| File | Estimator | Paper role |
| --- | --- | --- |
| `pre_mobilenet_v2_OffloadBin_focal.pt` | MobileNetV2-Lite, OffloadBin / focal | skipping, headline (Tab. 2) |
| `pre_mobilenet_v2_MORIC+-AP.pt` | MobileNetV2-Lite, MORIC⁺ | skipping (Tab. 2) |
| `post_xgboost_MORIC-AP.json` | XGBoost on MORIC | conditioned, ours (Tab. 2) |
| `post_edgeml_MORIC-AP_wmse.pt` | EdgeML MLP on MORIC | conditioned baseline (Tab. 2) |
| `pre_efficientnet_b0_lite_OffloadBin_focal.pt` | EfficientNet-B0-Lite, OffloadBin / focal | backbone ablation (Tab. 4) |
| *(+ the MobileNetV2-Lite learning-target ablation checkpoints, Tab. 5)* | | |

DCSB (`post|dcsb|CountGain-05`) is a fixed rule and has no trained weights.

If this directory is empty, the checkpoints have not been attached to the
release yet. You can always regenerate them from scratch with the `train`
phase — see the top-level `README.md` (Reproducing the results).
