"""
Pretrained CNN backbones with replaced classification heads.

Supported (per project_rules.md and Project_Brief.docx):
    resnet50, resnet101, efficientnet_b0, efficientnet_b2, efficientnet_b3, vgg16

The factory returns:
    model, info_dict
where info_dict carries metadata used in the model card.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from config.config import NUM_CLASSES


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def build_model(name: str, num_classes: int = NUM_CLASSES, dropout: float = 0.2) -> Tuple[nn.Module, Dict]:
    """
    Build a pretrained CNN with its classifier replaced for `num_classes`.

    Returns (model, info) where `info` contains the metadata used in
    model cards (total/trainable params, input_size, backbone).
    """
    name = name.lower().strip()

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        info = _make_info(name, m, input_size=224)
    elif name == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        info = _make_info(name, m, input_size=224)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        info = _make_info(name, m, input_size=224)
    elif name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        info = _make_info(name, m, input_size=224)
    elif name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        info = _make_info(name, m, input_size=224)
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        info = _make_info(name, m, input_size=224)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Supported: resnet50, resnet101, "
            f"efficientnet_b0, efficientnet_b2, efficientnet_b3, vgg16"
        )

    return m, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _make_info(name: str, model: nn.Module, input_size: int) -> Dict:
    total, trainable = count_parameters(model)
    return {
        "model_name": name,
        "input_size": input_size,
        "total_params": total,
        "trainable_params": trainable,
        "total_params_M": round(total / 1e6, 2),
    }


def get_param_groups(model: nn.Module, lr_head: float, lr_backbone: float, weight_decay: float):
    """
    Two-LR optimisation: a higher LR for the new classifier head and a smaller
    LR for the pretrained backbone. Identifying the head differs per model.
    """
    # Decide which parameter names are "head" vs "backbone"
    head_keys = []
    if hasattr(model, "fc"):  # ResNet family
        head_keys.append("fc.")
    if hasattr(model, "classifier"):  # EfficientNet, VGG, DenseNet
        head_keys.append("classifier.")

    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if any(name.startswith(k) for k in head_keys):
            head_params.append(p)
        else:
            backbone_params.append(p)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
