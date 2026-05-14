"""
ImageFolder-based DataLoaders for the Pakistani Politicians dataset.

The dataset layout on Drive (after splitting) is:

    dataset_resplit/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

We validate that:
  1. All three split folders exist.
  2. Every class folder from CLASS_NAMES is present in each split.
  3. ImageFolder's class_to_idx matches the alphabetical CLASS_NAMES order
     exactly — otherwise downstream metrics would map to wrong labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from config.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATASET_DIR,
    NUM_WORKERS,
    PIN_MEMORY,
)
from src.transforms import get_all_transforms


def _verify_class_alignment(image_folder: datasets.ImageFolder, split_name: str) -> None:
    """Hard-fail if ImageFolder discovered different classes than CLASS_NAMES."""
    discovered = sorted(image_folder.classes)
    expected = sorted(CLASS_NAMES)
    if discovered != expected:
        missing = set(expected) - set(discovered)
        extra = set(discovered) - set(expected)
        msg = [
            f"Class mismatch in {split_name}:",
            f"  expected ({len(expected)}): {expected}",
            f"  found    ({len(discovered)}): {discovered}",
        ]
        if missing:
            msg.append(f"  missing folders: {sorted(missing)}")
        if extra:
            msg.append(f"  unexpected folders: {sorted(extra)}")
        raise RuntimeError("\n".join(msg))


def build_datasets(
    dataset_root: Path | str | None = None,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Build the three ImageFolder datasets. Returns (train, val, test)."""
    root = Path(dataset_root) if dataset_root is not None else DATASET_DIR

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not d.exists():
            raise FileNotFoundError(f"{name} split directory not found: {d}")

    train_tf, val_tf, test_tf = get_all_transforms()

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)

    _verify_class_alignment(train_ds, "train")
    _verify_class_alignment(val_ds, "val")
    _verify_class_alignment(test_ds, "test")

    # Sanity check: all three must agree on class_to_idx
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError("class_to_idx mismatch between train and val")
    if train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("class_to_idx mismatch between train and test")

    return train_ds, val_ds, test_ds


def build_dataloaders(
    dataset_root: Path | str | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Build dataloaders for train/val/test.

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """
    train_ds, val_ds, test_ds = build_datasets(dataset_root)

    # Generator pinned to a seed for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


def count_images_per_class(dataset_root: Path | str | None = None) -> Dict[str, Dict[str, int]]:
    """
    Return {split: {class_name: n_images}} for diagnostics.
    Used by the data audit notebook.
    """
    root = Path(dataset_root) if dataset_root is not None else DATASET_DIR
    out: Dict[str, Dict[str, int]] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            out[split] = {}
            continue
        counts: Dict[str, int] = {}
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            n = sum(
                1 for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in image_exts
            )
            counts[cls_dir.name] = n
        out[split] = counts
    return out
