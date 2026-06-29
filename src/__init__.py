"""Detection offloading pipeline.

Provides dataset handling, feature extraction, estimator models,
and a 5-phase pipeline: detect → prepare → train → evaluate → analyse.
"""

from .features import (BASE_FEATURES, CLOUD_FEATURES, EDGE_FEATURES,
                       FEATURE_COLUMNS, extract_detection_features,
                       get_feature_columns, load_detection_results)

__all__ = [
    'BASE_FEATURES',
    'EDGE_FEATURES',
    'CLOUD_FEATURES',
    'FEATURE_COLUMNS',
    'get_feature_columns',
    'load_detection_results',
    'extract_detection_features',
]

__version__ = '3.0.0'
