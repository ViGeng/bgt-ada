"""CNN-based estimators for image input.

Implements image-based estimators:
- CNN Regressor (MobileNetV3)
- LeNet / LeNetLite
- SqueezeNet, ShuffleNet, MobileNetV2
- ViT Encoder
- Extremely Lightweight / Lightweight CNNs
- Compressed Weak Model
"""

from typing import Any, Dict

import numpy as np

from .base import ImageDataset, ImageEstimator


class CNNEstimator(ImageEstimator):
    """CNN estimator using MobileNetV3-Small backbone.

    Stage: Pre-inference (raw current frame)
    """

    name = "cnn_regressor"
    task_type = "regression"
    stage = "pre"
    pretrained = True

    def __init__(self, backbone: str = "mobilenet", image_size: int = 224, **kwargs):
        super().__init__(image_size=image_size, **kwargs)
        self.backbone = backbone

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.mobilenet_v3_small(weights='DEFAULT')
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, 1)
        self.model = base.to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV3-Small", "gflops": gf or 0.06, "params": pm or 2.5}


class LeNetEstimator(ImageEstimator):
    """Enhanced LeNet CNN: 5 Conv+BN blocks → GAP → FC head, ~0.8M params.

    Deeper feature extractor with BatchNorm for stable training and
    AdaptiveAvgPool for resolution flexibility. Trains from scratch
    (no pretrained weights).
    """

    name = "cnn_lenet"
    task_type = "regression"
    stage = "pre"

    def _setup_model(self):
        import torch.nn as nn

        class LeNetReg(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256, 128), nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                )

            def forward(self, x):
                return self.head(self.features(x))

        self._setup_device()
        self.model = LeNetReg().to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "LeNet CNN (5 Conv+BN, GAP, FC head)",
                "gflops": gf or 0.3, "params": pm or 0.8}


class LeNetLiteEstimator(ImageEstimator):
    """Lightweight LeNet — 64x64 input, ~60K params, ~0.005 GFLOPs."""

    name = "cnn_lenet_lite"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 64, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn
        import torch.nn.functional as F

        class LeNetLiteReg(nn.Module):
            def __init__(self, image_size=64):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                conv_out = (image_size - 4) // 2
                conv_out = (conv_out - 4) // 2
                self.fc1 = nn.Linear(16 * conv_out * conv_out, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        self._setup_device()
        self.model = LeNetLiteReg(image_size=self.image_size).to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "LeNet-Lite (64x64, smaller FC)", "gflops": gf or 0.005, "params": pm or 0.06}


class SqueezeNetEstimator(ImageEstimator):
    """SqueezeNet 1.1 estimator."""

    name = "cnn_squeezenet"
    stage = "pre"
    pretrained = True
    default_epochs = 8
    default_lr = 0.0005

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self.model = models.squeezenet1_1(weights='DEFAULT')
        self.model.classifier[1] = nn.Conv2d(512, 1, kernel_size=1)
        self.model.num_classes = 1
        self.model.to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "SqueezeNet 1.1", "gflops": gf or 0.35, "params": pm or 1.2}


class ShuffleNetEstimator(ImageEstimator):
    """ShuffleNet V2 x0.5 estimator."""

    name = "cnn_shufflenet_v2"
    stage = "pre"
    pretrained = True

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self.model = models.shufflenet_v2_x0_5(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.model.to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "ShuffleNet V2 x0.5", "gflops": gf or 0.04, "params": pm or 1.4}


class MobileNetV2Estimator(ImageEstimator):
    """MobileNetV2 estimator."""

    name = "cnn_mobilenet_v2"
    stage = "pre"
    pretrained = True

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self.model = models.mobilenet_v2(weights='DEFAULT')
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        self.model.to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV2", "gflops": gf or 0.3, "params": pm or 3.5}


class ViTEstimator(ImageEstimator):
    """Micro Vision Transformer: Patch embed → Transformer → FC."""

    name = "vit_encoder"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2, **kwargs):
        super().__init__(image_size=image_size, **kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def _setup_model(self):
        import torch
        import torch.nn as nn

        self._setup_device()
        self._setup_transforms()

        class ViT(nn.Module):
            def __init__(self, image_size, patch_size, d_model, n_heads, n_layers):
                super().__init__()
                num_patches = (image_size // patch_size) ** 2
                patch_dim = 3 * patch_size * patch_size
                self.patch_size = patch_size
                self.patch_embedding = nn.Linear(patch_dim, d_model)
                self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                    dropout=0.1, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
                self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

            def forward(self, x):
                batch_size = x.shape[0]
                patches = x.unfold(2, self.patch_size, self.patch_size)
                patches = patches.unfold(3, self.patch_size, self.patch_size)
                patches = patches.contiguous().view(
                    batch_size, -1, 3 * self.patch_size * self.patch_size)
                x = self.patch_embedding(patches)
                cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                x = x + self.position_embedding
                x = self.transformer(x)
                return self.head(x[:, 0]).squeeze(-1)

        self.model = ViT(
            self.image_size, self.patch_size, self.d_model,
            self.n_heads, self.n_layers
        ).to(self.device)


class ExtremelyLightweightCNNEstimator(ImageEstimator):
    """Grayscale 64x64 with depthwise-separable convolutions + GAP."""

    name = "extremely_lightweight_cnn"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 64, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_transforms(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def _setup_train_transforms(self):
        from torchvision import transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def _setup_model(self):
        import torch.nn as nn

        self._setup_device()
        self._setup_transforms()

        def depthwise_separable_conv(nin, nout, stride=1):
            return nn.Sequential(
                nn.Conv2d(nin, nin, 3, stride=stride, padding=1, groups=nin, bias=False),
                nn.BatchNorm2d(nin), nn.ReLU6(inplace=True),
                nn.Conv2d(nin, nout, 1, bias=False),
                nn.BatchNorm2d(nout), nn.ReLU6(inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU6(inplace=True),
            depthwise_separable_conv(8, 16, stride=2),
            depthwise_separable_conv(16, 32, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1),
        ).to(self.device)

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "Extremely lightweight CNN (Grayscale, Depthwise, GAP)",
                "gflops": gf or 0.0001, "params": pm or 0.001}

    def fit(self, X, y, **kwargs):
        self._setup_model()
        self.is_fitted = True


class LightweightCNNEstimator(ImageEstimator):
    """112x112 RGB, DS-Conv + Squeeze-and-Excite attention, ~15K params."""

    name = "lightweight_cnn"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 112, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn

        self._setup_device()
        self._setup_transforms()

        class SEBlock(nn.Module):
            def __init__(self, channels, reduction=4):
                super().__init__()
                mid = max(channels // reduction, 4)
                self.fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(channels, mid), nn.ReLU(inplace=True),
                    nn.Linear(mid, channels), nn.Sigmoid(),
                )
            def forward(self, x):
                return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

        class DSConvBlock(nn.Module):
            def __init__(self, nin, nout, stride=1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(nin, nin, 3, stride=stride, padding=1, groups=nin, bias=False),
                    nn.BatchNorm2d(nin), nn.ReLU6(inplace=True),
                    nn.Conv2d(nin, nout, 1, bias=False),
                    nn.BatchNorm2d(nout), nn.ReLU6(inplace=True),
                )
                self.se = SEBlock(nout)
            def forward(self, x):
                return self.se(self.block(x))

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            DSConvBlock(16, 32, stride=2),
            DSConvBlock(32, 64, stride=2),
            DSConvBlock(64, 64, stride=1),
            DSConvBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(128, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        ).to(self.device)

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "Lightweight CNN (DS-Conv, SE, MLP head)",
                "gflops": gf or 0.005, "params": pm or 0.035}


class LeNetLargeEstimator(ImageEstimator):
    """Larger LeNet: 4 conv blocks with BN + GAP + FC head, ~600K params.

    Larger than standard LeNet but still much smaller than the weak detection
    model (~19M params).  Uses BatchNorm for stable training and
    AdaptiveAvgPool for resolution flexibility.
    """

    name = "cnn_lenet_large"
    task_type = "regression"
    stage = "pre"

    def _setup_model(self):
        import torch.nn as nn

        class LeNetLargeReg(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(4),
                )
                self.head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True), nn.Dropout(0.3),
                    nn.Linear(256, 1),
                )

            def forward(self, x):
                return self.head(self.features(x))

        self._setup_device()
        self.model = LeNetLargeReg().to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "LeNet-Large (4 Conv+BN, GAP, FC head)",
                "gflops": gf or 0.3, "params": pm or 0.6}


class TinyYOLOEstimator(ImageEstimator):
    """Tiny YOLO-style estimator with Darknet backbone, ~1.3M params.

    Architecture: Stack of Conv+BN+LeakyReLU blocks with MaxPool,
    inspired by the Tiny-YOLO / Darknet-Reference backbone.
    Uses Global Average Pooling → FC(1) for regression.
    """

    name = "tiny_yolo"
    task_type = "regression"
    stage = "pre"

    def _setup_model(self):
        import torch.nn as nn

        class DarknetBlock(nn.Module):
            """Conv2d → BatchNorm → LeakyReLU."""
            def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel, stride=stride,
                              padding=padding, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            def forward(self, x):
                return self.block(x)

        class TinyDarknet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    DarknetBlock(3, 16),
                    nn.MaxPool2d(2, 2),
                    DarknetBlock(16, 32),
                    nn.MaxPool2d(2, 2),
                    DarknetBlock(32, 64),
                    nn.MaxPool2d(2, 2),
                    DarknetBlock(64, 128),
                    nn.MaxPool2d(2, 2),
                    DarknetBlock(128, 256),
                    nn.MaxPool2d(2, 2),
                    DarknetBlock(256, 512),
                )
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 1),
                )

            def forward(self, x):
                return self.head(self.features(x))

        self._setup_device()
        self.model = TinyDarknet().to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "Tiny YOLO Darknet backbone (6 Conv+BN+LeakyReLU, GAP)",
                "gflops": gf or 0.9, "params": pm or 1.3}


class LightweightResNetEstimator(ImageEstimator):
    """Small ResNet with residual connections, ~200K params.

    Architecture: Conv stem → 3 stages of residual blocks → GAP → FC.
    Residual connections improve gradient flow for this deeper network.
    """

    name = "lightweight_resnet"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 112, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, channels, stride=1, expand=None):
                super().__init__()
                out_c = expand or channels
                self.conv1 = nn.Conv2d(channels, out_c, 3, stride=stride,
                                       padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_c)
                self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_c)
                self.relu = nn.ReLU(inplace=True)
                self.shortcut = nn.Sequential()
                if stride != 1 or channels != out_c:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(channels, out_c, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_c),
                    )

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                return self.relu(out)

        class SmallResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
                self.stage1 = nn.Sequential(ResBlock(16), ResBlock(16))
                self.stage2 = nn.Sequential(
                    ResBlock(16, stride=2, expand=32), ResBlock(32))
                self.stage3 = nn.Sequential(
                    ResBlock(32, stride=2, expand=64), ResBlock(64))
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                return self.head(x)

        self._setup_device()
        self.model = SmallResNet().to(self.device)
        self._setup_transforms()

    def get_info(self):
        gf, pm = self._compute_flops()
        return {"description": "Lightweight ResNet (6 ResBlocks, 3 stages)",
                "gflops": gf or 0.05, "params": pm or 0.2}


class MoricPlusCNNEstimator(CNNEstimator):
    """MobileNetV3-Small estimator for MORIC+ proxy-metrics.

    Same architecture as CNNEstimator but with a unique name for checkpoint
    separation.  Designed to be paired with:
      - proxy_metric="moric_plus_allpoint" (signed offloading gain proxy-metric in [-1, 1])
      - loss="asymmetric_u_mse" (penalises extremes, overweights negatives)
    """

    name = "cnn_moric_plus"

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV3-Small (MORIC+)", "gflops": gf or 0.06, "params": pm or 2.5}


# ---------------------------------------------------------------------------
#  Lightweight (Lite) variants — reduced resolution / truncated for speed
# ---------------------------------------------------------------------------

class CNNRegressorLiteEstimator(ImageEstimator):
    """MobileNetV3-Small at 128x128 — ~3x faster than the 224x224 variant.

    Uses the same pretrained backbone but with reduced input resolution,
    cutting FLOPs quadratically while retaining most prediction quality.
    """

    name = "cnn_regressor_lite"
    task_type = "regression"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.mobilenet_v3_small(weights='DEFAULT')
        base.classifier[-1] = nn.Linear(base.classifier[-1].in_features, 1)
        self.model = base.to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV3-Small-Lite (128x128)",
                "gflops": gf or 0.02, "params": pm or 2.5}


class MobileNetV2LiteEstimator(ImageEstimator):
    """MobileNetV2 at 128x128 with truncated features — ~5x faster.

    Keeps the first 14 of 19 inverted-residual blocks and uses 128x128
    input, reducing both parameter count and compute significantly.
    """

    name = "cnn_mobilenet_v2_lite"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.mobilenet_v2(weights='DEFAULT')
        base.features = base.features[:14]
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out_channels = base.features(dummy).shape[1]
        self.model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, 1),
        ).to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV2-Lite (128x128, truncated)",
                "gflops": gf or 0.06, "params": pm or 1.8}


class EfficientNetB0LiteEstimator(ImageEstimator):
    """EfficientNet-B0 at 128x128, truncated after stage 5 of 8.

    Compound-scaled NAS design — different inductive bias from MobileNet.
    Uses pretrained ImageNet weights, truncated for speed.
    """

    name = "cnn_efficientnet_b0_lite"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.efficientnet_b0(weights='DEFAULT')
        # Truncate: keep first 6 of 8 feature blocks
        base.features = base.features[:6]
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out_channels = base.features(dummy).shape[1]
        self.model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, 1),
        ).to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "EfficientNet-B0-Lite (128x128, truncated)",
                "gflops": gf or 0.04, "params": pm or 1.5}


class RegNetY200MFEstimator(ImageEstimator):
    """RegNetY-200MF at 128x128 (full model).

    Quantized linear design space with SE attention, already tiny (~3.2M).
    Different design philosophy from MobileNet (fixed width/depth ratios).
    """

    name = "cnn_regnety_200mf"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.regnet_y_400mf(weights='DEFAULT')
        base.fc = nn.Linear(base.fc.in_features, 1)
        self.model = base.to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "RegNetY-400MF (128x128)",
                "gflops": gf or 0.4, "params": pm or 4.3}


class MNASNet050Estimator(ImageEstimator):
    """MNASNet-0.50 at 128x128 (full model).

    NAS-found architecture with different block choices from MobileNet.
    """

    name = "cnn_mnasnet050"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.mnasnet0_5(weights='DEFAULT')
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 1)
        self.model = base.to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MNASNet-0.50 (128x128)",
                "gflops": gf or 0.1, "params": pm or 2.2}


class ConvNeXtTinyLiteEstimator(ImageEstimator):
    """ConvNeXt-Tiny at 128x128, truncated to stages 0-1 of 4.

    Large-kernel pure ConvNet inspired by ViT.  Truncated for speed,
    retaining early stages with different representational bias.
    """

    name = "cnn_convnext_tiny_lite"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.convnext_tiny(weights='DEFAULT')
        # Truncate: keep first 4 of 8 feature blocks (stages 0-1 of 4,
        # each stage has a downsample + blocks pair)
        base.features = base.features[:4]
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out_channels = base.features(dummy).shape[1]
        self.model = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, 1),
        ).to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "ConvNeXt-Tiny-Lite (128x128, stages 0-1)",
                "gflops": gf or 0.08, "params": pm or 2.0}


class MoricPlusCNNLiteEstimator(CNNRegressorLiteEstimator):
    """Lightweight MobileNetV3-Small (128x128) for MORIC+ proxy-metrics.

    Same architecture as CNNRegressorLiteEstimator but with a unique name
    for checkpoint separation.  Designed for hallucination-aware training
    with asymmetric_u_mse loss.
    """

    name = "cnn_moric_plus_lite"

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "MobileNetV3-Small-Lite (128x128, MORIC+)",
                "gflops": gf or 0.02, "params": pm or 2.5}


class ReducedMobileNetV3LiteEstimator(ImageEstimator):
    """Reduced MobileNetV3-Small at 128x128 — truncated + low-res.

    Combines layer truncation (first 5 blocks) with reduced resolution
    for maximum speed. Suitable as a fast baseline.
    """

    name = "reduced_mobilenetv3_lite"
    task_type = "regression"
    stage = "pre"

    def __init__(self, image_size: int = 128, **kwargs):
        super().__init__(image_size=image_size, **kwargs)

    def _setup_model(self):
        import torch
        import torch.nn as nn
        from torchvision import models

        self._setup_device()
        self._setup_transforms()

        base = models.mobilenet_v3_small(weights='DEFAULT')
        base.features = base.features[:5]
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out_channels = base.features(dummy).shape[1]
        base.classifier = nn.Sequential(
            nn.Linear(out_channels, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1),
        )
        self.model = base.to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": "Reduced MobileNetV3-Lite (128x128, 5 blocks)",
                "gflops": gf or 0.008, "params": pm or 1.0}


# ---------------------------------------------------------------------------
#  Compressed Weak Model — detection backbone extraction + truncation
# ---------------------------------------------------------------------------

def _convert_frozen_bn(module):
    """Replace FrozenBatchNorm2d with trainable BatchNorm2d in-place."""
    import torch.nn as nn

    for name, child in module.named_children():
        if type(child).__name__ == 'FrozenBatchNorm2d':
            num_features = child.weight.shape[0]
            bn = nn.BatchNorm2d(num_features)
            bn.weight.data.copy_(child.weight)
            bn.bias.data.copy_(child.bias)
            bn.running_mean.data.copy_(child.running_mean)
            bn.running_var.data.copy_(child.running_var)
            bn.num_batches_tracked.zero_()
            setattr(module, name, bn)
        else:
            _convert_frozen_bn(child)


class CompressedWeakModelEstimator(ImageEstimator):
    """Compressed weak detection model backbone for proxy-metric prediction.

    Extracts the MobileNetV3-Large backbone from the pretrained
    FasterRCNN_MobileNet_V3_Large_FPN detection model, truncates it to
    ``truncate_after`` layers, and adds a lightweight regression head.

    Unlike generic ImageNet-pretrained estimators, the backbone starts with
    detection-specific weights that encode how the weak model perceives images.
    """

    name = "compressed_weak"
    task_type = "regression"
    stage = "pre"
    pretrained = True

    def __init__(self, image_size: int = 128, truncate_after: int = 10,
                 freeze_backbone: bool = False, **kwargs):
        super().__init__(image_size=image_size, **kwargs)
        self.truncate_after = truncate_after
        self.freeze_backbone = freeze_backbone

    def _setup_model(self):
        import torch
        import torch.nn as nn
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_fpn,
        )

        self._setup_device()
        self._setup_transforms()

        det_model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        body_layers = list(det_model.backbone.body.children())
        trunk = nn.Sequential(*body_layers[:self.truncate_after + 1])
        _convert_frozen_bn(trunk)
        del det_model

        if self.freeze_backbone:
            for p in trunk.parameters():
                p.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            out_channels = trunk(dummy).shape[1]

        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.model = nn.Sequential(trunk, head).to(self.device)

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": f"Compressed weak model (truncate={self.truncate_after})",
                "gflops": gf or 0.02, "params": pm or 0.2}

    def save(self, path) -> None:
        import torch
        from pathlib import Path as _Path

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.model.state_dict()
        clean_state = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
        torch.save({
            'model_state_dict': clean_state,
            'is_fitted': self.is_fitted,
            'name': self.name,
            'image_size': self.image_size,
            'truncate_after': self.truncate_after,
            'y_mean': getattr(self, '_y_mean', 0.0),
            'y_std': getattr(self, '_y_std', 1.0),
        }, path)

    @classmethod
    def load(cls, path, device: str = None) -> 'CompressedWeakModelEstimator':
        import torch

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        estimator = cls(
            image_size=checkpoint.get('image_size', 128),
            truncate_after=checkpoint.get('truncate_after', 10),
            device=device,
        )
        estimator._setup_model()
        state_dict = checkpoint['model_state_dict']
        clean_state = {k.replace('_orig_mod.', ''): v
                       for k, v in state_dict.items()}
        estimator.model.load_state_dict(clean_state)
        estimator.is_fitted = checkpoint.get('is_fitted', True)
        estimator._y_mean = checkpoint.get('y_mean', 0.0)
        estimator._y_std = checkpoint.get('y_std', 1.0)
        return estimator


class CompressedWeakModelMoricPlusEstimator(CompressedWeakModelEstimator):
    """Compressed weak model for MORIC+ proxy-metrics.

    Same architecture as CompressedWeakModelEstimator but with a unique name
    for checkpoint separation.  Designed for asymmetric_u_mse loss with
    moric_plus_allpoint proxy-metric values.
    """

    name = "compressed_weak_moric_plus"

    def get_info(self) -> Dict[str, Any]:
        gf, pm = self._compute_flops()
        return {"description": f"Compressed weak model MORIC+ (truncate={self.truncate_after})",
                "gflops": gf or 0.02, "params": pm or 0.2}
