"""Dataset registry — known datasets with their default configurations.

This release targets the PASCAL VOC setting used throughout the paper. The
broader research repo additionally registered COCO, UA-DETRAC and several
weak/strong model-pair variants; those are outside the paper and are not
included here.
"""

from .schema import DatasetConfig

# Base datasets — the primary dataset configs (used for --dataset all)
_BASE_DATASETS = ["voc"]

DATASETS = {
    # PASCAL VOC: classic object detection benchmark.
    # detection_split options: "train" (→ trainval), "test" (→ val), "all"
    "voc": DatasetConfig(
        name="voc",
        root="data/VOC",
        edge_model="fasterrcnn_mobilenet_v3_large_fpn",
        cloud_model="fasterrcnn_resnet50_fpn_v2",
        conf_threshold=0.3,
        sample_fraction=0.6,
        test_ratio=0.2,
        detection_split="all",
        moric_plus_neg_frac=0.27,  # ~27% of VOC ORIC values are negative
    ),
}
