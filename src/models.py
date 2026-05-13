"""
src/models.py
=============
Model factory for ResNet50, EfficientNet-B0, and EfficientNet-B2.
All models are loaded with ImageNet pretrained weights and their
classifier heads replaced for 16-class output.

Usage:
    from src.models import build_model
    model = build_model("resnet50", num_classes=16, dropout=0.4)
"""
from typing import Tuple 
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Internal builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_resnet50(num_classes: int, dropout: float, freeze_base: bool) -> nn.Module:
    model = tv_models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_efficientnet_b0(num_classes: int, dropout: float, freeze_base: bool) -> nn.Module:
    model = tv_models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_efficientnet_b2(num_classes: int, dropout: float, freeze_base: bool) -> nn.Module:
    model = tv_models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = ("resnet50", "efficientnet_b0", "efficientnet_b2")


def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    dropout: float   = 0.4,
    freeze_base: bool = False,
) -> nn.Module:
    """
    Build and return a pretrained CNN with a custom classification head.

    Parameters
    ----------
    model_name   : str   — one of: 'resnet50', 'efficientnet_b0', 'efficientnet_b2'
    num_classes  : int   — number of output classes (default 16)
    dropout      : float — dropout rate before final linear layer
    freeze_base  : bool  — if True, backbone weights are frozen (feature extraction only)

    Returns
    -------
    torch.nn.Module  (ready for .to(device) and training)
    """
    model_name = model_name.lower().strip()
    assert model_name in SUPPORTED_MODELS, \
        f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}"

    builders = {
        "resnet50":         _build_resnet50,
        "efficientnet_b0":  _build_efficientnet_b0,
        "efficientnet_b2":  _build_efficientnet_b2,
    }

    model = builders[model_name](num_classes, dropout, freeze_base)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [{model_name}]  trainable params: {trainable:,} / {total:,}")
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


# Quick sanity check
if __name__ == "__main__":
    from typing import Tuple
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in SUPPORTED_MODELS:
        m = build_model(name).to(device)
        x = torch.randn(2, 3, 224, 224).to(device)
        out = m(x)
        print(f"  {name}  →  output shape: {out.shape}")   # (2, 16)
