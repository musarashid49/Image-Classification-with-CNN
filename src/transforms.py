"""
Image transforms for train / val / test.

CRITICAL: augmentation is applied to TRAINING DATA ONLY.
The val and test transforms must be deterministic (resize -> centre-crop ->
normalise). Any randomness in val/test = data leakage on metrics.
"""
from __future__ import annotations

from typing import Tuple

from torchvision import transforms

from config.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def _resize_target() -> int:
    """Slightly larger than IMG_SIZE so centre-crop has something to bite."""
    return int(IMG_SIZE * 1.15)  # 224 -> 257


def get_train_transform() -> transforms.Compose:
    """
    Training augmentation: rotation, flipping, brightness, zoom (RandomResizedCrop),
    cropping. All five techniques required by project_rules.md are present.
    """
    return transforms.Compose([
        # Zoom + crop (scale=zoom range, ratio=aspect)
        transforms.RandomResizedCrop(
            IMG_SIZE,
            scale=(0.75, 1.0),     # zoom
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),                    # flipping
        transforms.RandomRotation(degrees=15),                     # rotation
        transforms.ColorJitter(
            brightness=0.25,                                       # brightness
            contrast=0.15,
            saturation=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Mild erasing for occlusion robustness (faces in news photos vary a lot)
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10)),
    ])


def get_eval_transform() -> transforms.Compose:
    """Deterministic transform for validation and test."""
    return transforms.Compose([
        transforms.Resize(_resize_target()),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_all_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Convenience: returns (train_tf, val_tf, test_tf). Val and test are the same."""
    train_tf = get_train_transform()
    eval_tf = get_eval_transform()
    return train_tf, eval_tf, eval_tf
