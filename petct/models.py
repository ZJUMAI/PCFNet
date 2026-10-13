"""Model registry for main, baseline, and two-stage experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    densenet121,
    efficientnet_b0,
    resnet18,
)
from torchvision.models.video import R3D_18_Weights, r3d_18

from petct.config import ConfigError


def _expand_conv2d(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        mean_weight = conv.weight.mean(dim=1, keepdim=True)
        replacement.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1) * (3 / in_channels))
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


def _expand_conv3d(conv: nn.Conv3d, in_channels: int) -> nn.Conv3d:
    replacement = nn.Conv3d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        mean_weight = conv.weight.mean(dim=1, keepdim=True)
        replacement.weight.copy_(
            mean_weight.repeat(1, in_channels, 1, 1, 1) * (conv.in_channels / in_channels)
        )
        if conv.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


class VolumeBackbone(nn.Module):
    """R3D-18 feature extractor for one volumetric modality."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        network = r3d_18(weights=weights)
        network.stem[0] = _expand_conv3d(network.stem[0], 1)
        self.feature_dim = network.fc.in_features
        network.fc = nn.Identity()
        self.network = network

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 4:
            volume = volume.unsqueeze(1)
        return self.network(volume)

    def forward_features(self, volume: torch.Tensor) -> torch.Tensor:
        """Return the final spatial feature map without global pooling."""

        if volume.ndim == 4:
            volume = volume.unsqueeze(1)
        features = self.network.stem(volume)
        for stage in (
            self.network.layer1,
            self.network.layer2,
            self.network.layer3,
            self.network.layer4,
        ):
            features = stage(features)
        return features


class BidirectionalCrossAttention(nn.Module):
    """Symmetric CT-to-PET and PET-to-CT feature attention."""

    def __init__(self, feature_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ct_to_pet = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pet_to_ct = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim * 2)
        self.projection = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        ct_token = ct.unsqueeze(1)
        pet_token = pet.unsqueeze(1)
        ct_attended, _ = self.ct_to_pet(ct_token, pet_token, pet_token, need_weights=False)
        pet_attended, _ = self.pet_to_ct(pet_token, ct_token, ct_token, need_weights=False)
        fused = torch.cat([ct_attended.squeeze(1), pet_attended.squeeze(1)], dim=1)
        return self.projection(self.norm(fused))


class PETCTFusionClassifier(nn.Module):
    """Dual R3D-18 classifier with bidirectional cross-attention."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_heads: int = 8,
        pretrained: bool = True,
    ):
        super().__init__()
        self.ct_backbone = VolumeBackbone(pretrained)
        self.pet_backbone = VolumeBackbone(pretrained)
        self.fusion = BidirectionalCrossAttention(512, num_heads, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(self.ct_backbone(ct), self.pet_backbone(pet)))


def _validate_attention_dimensions(embedding_dim: int, num_heads: int) -> None:
    if not isinstance(embedding_dim, int) or not isinstance(num_heads, int):
        raise ConfigError("embedding_dim and num_heads must be integers")
    if embedding_dim <= 0 or num_heads <= 0 or embedding_dim % num_heads:
        raise ConfigError("embedding_dim must be positive and divisible by positive num_heads")


class SpatialTokenEncoder(nn.Module):
    """Pool a 3D feature map to 4x4x4 and project its 64 spatial tokens."""

    num_tokens = 64

    def __init__(self, feature_dim: int = 512, embedding_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.pool(features).flatten(2).transpose(1, 2)
        return self.projection(tokens)


class SpatialBidirectionalCrossAttention(nn.Module):
    """One pre-normalized, bidirectional cross-attention block with residuals.

    Both directions use the original normalized tokens, not sequential updates.
    No gates, feed-forward blocks, or auxiliary prediction heads are added.
    """

    def __init__(self, embedding_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        _validate_attention_dimensions(embedding_dim, num_heads)
        self.ct_norm = nn.LayerNorm(embedding_dim)
        self.pet_norm = nn.LayerNorm(embedding_dim)
        self.ct_to_pet = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pet_to_ct = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ct_normalized = self.ct_norm(ct)
        pet_normalized = self.pet_norm(pet)
        ct_attended, _ = self.ct_to_pet(
            ct_normalized, pet_normalized, pet_normalized, need_weights=False
        )
        pet_attended, _ = self.pet_to_ct(
            pet_normalized, ct_normalized, ct_normalized, need_weights=False
        )
        return (
            ct + self.residual_dropout(ct_attended),
            pet + self.residual_dropout(pet_attended),
        )


class PETCTSpatialFusionClassifier(nn.Module):
    """Dual R3D-18 streams with 64 low-dimensional spatial tokens per modality."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.5,
        embedding_dim: int = 128,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        # Validate attention dimensions before constructing/downloading backbones.
        _validate_attention_dimensions(embedding_dim, num_heads)
        self.ct_backbone = VolumeBackbone(pretrained)
        self.pet_backbone = VolumeBackbone(pretrained)
        self.ct_tokens = SpatialTokenEncoder(self.ct_backbone.feature_dim, embedding_dim)
        self.pet_tokens = SpatialTokenEncoder(self.pet_backbone.feature_dim, embedding_dim)
        self.fusion = SpatialBidirectionalCrossAttention(
            embedding_dim, num_heads, dropout=attention_dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        ct_tokens = self.ct_tokens(self.ct_backbone.forward_features(ct))
        pet_tokens = self.pet_tokens(self.pet_backbone.forward_features(pet))
        ct_updated, pet_updated = self.fusion(ct_tokens, pet_tokens)
        fused = torch.cat([ct_updated.mean(dim=1), pet_updated.mean(dim=1)], dim=1)
        return self.classifier(fused)


class DualStreamDenseNet(nn.Module):
    """DenseNet-121 baseline treating axial slices as input channels."""

    def __init__(self, depth: int = 64, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.ct_stream = densenet121(weights=weights)
        self.pet_stream = densenet121(weights=weights)
        self.ct_stream.features.conv0 = _expand_conv2d(self.ct_stream.features.conv0, depth)
        self.pet_stream.features.conv0 = _expand_conv2d(self.pet_stream.features.conv0, depth)
        feature_dim = self.ct_stream.classifier.in_features + self.pet_stream.classifier.in_features
        self.ct_stream.classifier = nn.Identity()
        self.pet_stream.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.ct_stream(ct), self.pet_stream(pet)], dim=1))


def _two_stage_stream(name: str, depth: int, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet":
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.conv1 = _expand_conv2d(model.conv1, depth)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if name == "densenet":
        model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        model.features.conv0 = _expand_conv2d(model.features.conv0, depth)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if name == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.features[0][0] = _expand_conv2d(model.features[0][0], depth)
        feature_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    raise ConfigError(f"Unsupported two-stage backbone: {name}")


class DualStreamFeatureExtractor(nn.Module):
    """Two-dimensional dual-stream CNN with reusable fused features."""

    def __init__(
        self,
        backbone: str = "resnet",
        depth: int = 64,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.ct_stream, ct_dim = _two_stage_stream(backbone, depth, pretrained)
        self.pet_stream, pet_dim = _two_stage_stream(backbone, depth, pretrained)
        self.feature_dim = ct_dim + pet_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def extract_features(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.ct_stream(ct), self.pet_stream(pet)], dim=1)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(ct, pet))


MODEL_REGISTRY = {
    "fusion3d": PETCTFusionClassifier,
    "fusion3d_spatial": PETCTSpatialFusionClassifier,
    "densenet": DualStreamDenseNet,
    "feature_extractor": DualStreamFeatureExtractor,
}


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = config.get("model", config)
    name = model_config.get("name", "fusion3d")
    if name not in MODEL_REGISTRY:
        raise ConfigError(f"Unknown model {name!r}; available models: {', '.join(MODEL_REGISTRY)}")
    parameters = dict(model_config.get("params", {}))
    return MODEL_REGISTRY[name](**parameters)
