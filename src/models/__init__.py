"""Estimator models package.

Registry of all available estimators with factory function.
"""

from typing import Dict, Type

from .base import BaseEstimator, ImageDataset, ImageEstimator
from .classical import (LassoEstimator, LightGBMEstimator, LogisticEstimator,
                        RandomForestEstimator, RidgeEstimator,
                        XGBoostEstimator)
from .cnn import (CNNEstimator, CNNRegressorLiteEstimator,
                  CompressedWeakModelEstimator,
                  CompressedWeakModelMoricPlusEstimator,
                  ConvNeXtTinyLiteEstimator,
                  EfficientNetB0LiteEstimator,
                  ExtremelyLightweightCNNEstimator, LeNetEstimator,
                  LeNetLargeEstimator, LeNetLiteEstimator,
                  LightweightCNNEstimator, LightweightResNetEstimator,
                  MNASNet050Estimator,
                  MobileNetV2Estimator, MobileNetV2LiteEstimator,
                  MoricPlusCNNEstimator, MoricPlusCNNLiteEstimator,
                  ReducedMobileNetV3LiteEstimator,
                  RegNetY200MFEstimator,
                  ShuffleNetEstimator,
                  SqueezeNetEstimator, TinyYOLOEstimator, ViTEstimator)
from .dcsb import DCSBEstimator, DCSBOriginalEstimator
from .edgeml import EdgeMLEstimator, EdgeMLPaperEstimator
from .virtual import (OracleEstimator, StrongModelEstimator, VirtualEstimator,
                      WeakModelEstimator)

ESTIMATOR_REGISTRY: Dict[str, Type[BaseEstimator]] = {
    # Classical ML
    'ridge': RidgeEstimator,
    'ridge_regression': RidgeEstimator,
    'lasso': LassoEstimator,
    'lasso_regression': LassoEstimator,
    'logistic': LogisticEstimator,
    'logistic_regression': LogisticEstimator,
    'rf': RandomForestEstimator,
    'random_forest': RandomForestEstimator,
    'xgboost': XGBoostEstimator,
    'xgb': XGBoostEstimator,
    'xgboost_lite': XGBoostEstimator,
    'lightgbm': LightGBMEstimator,
    'lgbm': LightGBMEstimator,

    # CNN
    'cnn': CNNEstimator,
    'cnn_regressor': CNNEstimator,
    'cnn_lenet': LeNetEstimator,
    'cnn_lenet_lite': LeNetLiteEstimator,
    'cnn_squeezenet': SqueezeNetEstimator,
    'cnn_shufflenet_v2': ShuffleNetEstimator,
    'cnn_mobilenet_v2': MobileNetV2Estimator,
    'weak_model': WeakModelEstimator,
    'strong_model': StrongModelEstimator,
    'oracle': OracleEstimator,
    'vit': ViTEstimator,
    'vit_encoder': ViTEstimator,
    'extremely_lightweight_cnn': ExtremelyLightweightCNNEstimator,
    'lightweight_cnn': LightweightCNNEstimator,
    'cnn_lenet_large': LeNetLargeEstimator,
    'lightweight_resnet': LightweightResNetEstimator,
    'tiny_yolo': TinyYOLOEstimator,
    'reduced_mobilenetv3': ReducedMobileNetV3LiteEstimator,
    'cnn_moric_plus': MoricPlusCNNEstimator,

    # Compressed weak model (detection backbone extraction + truncation)
    'compressed_weak': CompressedWeakModelEstimator,
    'compressed_weak_moric_plus': CompressedWeakModelMoricPlusEstimator,

    # CNN Lite (reduced resolution / truncated for speed)
    # Compact aliases (preferred) + legacy long names
    'cnn_reg': CNNRegressorLiteEstimator,
    'cnn_regressor_lite': CNNRegressorLiteEstimator,
    'mobilenet_v2': MobileNetV2LiteEstimator,
    'cnn_mobilenet_v2_lite': MobileNetV2LiteEstimator,
    'moric_plus_cnn': MoricPlusCNNLiteEstimator,
    'cnn_moric_plus_lite': MoricPlusCNNLiteEstimator,
    'reduced_mobilenetv3_lite': ReducedMobileNetV3LiteEstimator,
    'efficientnet_b0_lite': EfficientNetB0LiteEstimator,
    'cnn_efficientnet_b0_lite': EfficientNetB0LiteEstimator,
    'regnety_200mf': RegNetY200MFEstimator,
    'cnn_regnety_200mf': RegNetY200MFEstimator,
    'mnasnet050': MNASNet050Estimator,
    'cnn_mnasnet050': MNASNet050Estimator,
    'convnext_tiny_lite': ConvNeXtTinyLiteEstimator,
    'cnn_convnext_tiny_lite': ConvNeXtTinyLiteEstimator,

    # DCSB
    'dcsb': DCSBOriginalEstimator,
    'dcsb_paper': DCSBOriginalEstimator,
    'dcsb_original': DCSBOriginalEstimator,

    # EdgeML ORIC regressor (paper-faithful, threshold offloader)
    'edgeml': EdgeMLEstimator,
    'edgeml_paper': EdgeMLEstimator,
    'edgeml_original': EdgeMLEstimator,
    'edgeml_oric': EdgeMLEstimator,
}


def get_estimator(name: str, **kwargs) -> BaseEstimator:
    """Get a estimator instance by name."""
    name_lower = name.lower()
    if name_lower not in ESTIMATOR_REGISTRY:
        available = sorted(set(ESTIMATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown estimator: {name}. Available: {available}")
    return ESTIMATOR_REGISTRY[name_lower](**kwargs)


def list_estimators() -> Dict[str, str]:
    """List all available estimators with descriptions."""
    seen = set()
    result = {}
    for name, cls in ESTIMATOR_REGISTRY.items():
        if cls not in seen:
            seen.add(cls)
            doc = cls.__doc__.split('\n')[0] if cls.__doc__ else ''
            result[cls.name] = f"{doc} (stage={cls.stage})"
    return result


__all__ = [
    'BaseEstimator', 'ImageEstimator', 'ImageDataset',
    'VirtualEstimator', 'WeakModelEstimator', 'StrongModelEstimator',
    'OracleEstimator',
    'RidgeEstimator', 'LassoEstimator', 'LogisticEstimator',
    'RandomForestEstimator', 'XGBoostEstimator', 'LightGBMEstimator',
    'CompressedWeakModelEstimator', 'CompressedWeakModelMoricPlusEstimator',
    'CNNEstimator', 'LeNetEstimator', 'LeNetLiteEstimator',
    'LeNetLargeEstimator', 'LightweightResNetEstimator',
    'SqueezeNetEstimator', 'ShuffleNetEstimator', 'MobileNetV2Estimator',
    'ViTEstimator', 'MoricPlusCNNEstimator',
    'CNNRegressorLiteEstimator', 'MobileNetV2LiteEstimator',
    'MoricPlusCNNLiteEstimator', 'ReducedMobileNetV3LiteEstimator',
    'EfficientNetB0LiteEstimator', 'RegNetY200MFEstimator',
    'MNASNet050Estimator', 'ConvNeXtTinyLiteEstimator',
    'ExtremelyLightweightCNNEstimator', 'LightweightCNNEstimator',
    'TinyYOLOEstimator',
    'DCSBEstimator', 'DCSBOriginalEstimator',
    'EdgeMLEstimator', 'EdgeMLPaperEstimator',
    'ESTIMATOR_REGISTRY', 'get_estimator', 'list_estimators',
]
