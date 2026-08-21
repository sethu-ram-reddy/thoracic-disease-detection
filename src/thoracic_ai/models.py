from __future__ import annotations

import torch.nn as nn
from torchvision import models


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, num_classes)
        )
        return model

    if name == "vit_b16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        input_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_features, num_classes)
        )
        return model

    raise ValueError(f"Unsupported model: {name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in _classification_head(model, model_name).parameters():
        parameter.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name != "densenet121":
        raise ValueError("Grad-CAM is currently supported for DenseNet-121 only.")
    return model.features.norm5


def _classification_head(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "densenet121":
        return model.classifier
    if model_name == "vit_b16":
        return model.heads
    raise ValueError(f"Unsupported model: {model_name}")
