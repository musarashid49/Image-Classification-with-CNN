"""
src/dataset.py
==============
Dataset and DataLoader utilities.
Assumes the dataset on Google Drive is already split into:
    <root>/train/<class>/...
    <root>/val/<class>/...
    <root>/test/<class>/...

Usage in Colab:
    from src.dataset import get_dataloaders
    loaders = get_dataloaders(train_dir, val_dir, test_dir)
"""

import os
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.transforms import get_transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import BATCH_SIZE, NUM_WORKERS, CLASS_NAMES


# ─────────────────────────────────────────────
# Dataset validation helper
# ─────────────────────────────────────────────

def validate_dataset_structure(root_dir: str, split: str) -> None:
    """
    Validates that the split directory has the expected class folders
    and that each folder is non-empty.  Prints a per-class summary.
    """
    print(f"\n── Validating '{split}' split at: {root_dir} ──")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    found_classes = sorted(os.listdir(root_dir))
    total_images  = 0

    for cls in found_classes:
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        n = len(images)
        total_images += n
        flag = "✓" if n >= 1 else "✗ EMPTY"
        print(f"  {flag}  {cls:<35} {n:>4} images")

    print(f"\n  Total images in '{split}': {total_images}")
    print(f"  Classes found:            {len(found_classes)}")

    # Warn if a configured class is missing
    for expected in CLASS_NAMES:
        if expected not in found_classes:
            print(f"  ⚠  WARNING: expected class '{expected}' not found in {split}")


# ─────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────

def get_dataloaders(
    train_dir: str,
    val_dir:   str,
    test_dir:  str,
    batch_size: int  = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    image_size: int  = 224,
    verbose: bool    = True,
) -> Dict[str, DataLoader]:
    """
    Builds ImageFolder datasets and wraps them in DataLoaders.

    Returns
    -------
    dict with keys "train", "val", "test".
    Also prints dataset sizes when verbose=True.
    """
    datasets = {
        "train": ImageFolder(train_dir, transform=get_transforms("train", image_size)),
        "val":   ImageFolder(val_dir,   transform=get_transforms("val",   image_size)),
        "test":  ImageFolder(test_dir,  transform=get_transforms("test",  image_size)),
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if verbose:
        print("\n── DataLoader Summary ──────────────────────────────────────")
        for split, ds in datasets.items():
            print(f"  {split:<8} {len(ds):>5} images  |  "
                  f"{len(loaders[split]):>4} batches  |  "
                  f"classes: {len(ds.classes)}")
        print(f"\n  Class-to-index mapping:")
        for cls, idx in datasets["train"].class_to_idx.items():
            print(f"    [{idx:>2}]  {cls}")

    return loaders


def get_class_to_idx(train_dir: str) -> Dict[str, int]:
    """Returns the class→index mapping from the training ImageFolder."""
    ds = ImageFolder(train_dir)
    return ds.class_to_idx


def get_idx_to_class(train_dir: str) -> Dict[int, str]:
    """Returns the index→class mapping (inverse of class_to_idx)."""
    return {v: k for k, v in get_class_to_idx(train_dir).items()}
