"""
src/transforms.py
=================
Torchvision transform pipelines for train / val / test splits.
Augmentation is ONLY applied to training data — never val or test.
Import and use get_transforms() in your DataLoader setup.
"""

import torchvision.transforms as T
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    AUG_ROTATION, AUG_FLIP, AUG_BRIGHTNESS,
    AUG_CONTRAST, AUG_CROP_SCALE_LOW,
)


def get_transforms(split: str, image_size: int = IMAGE_SIZE) -> T.Compose:
    """
    Returns the appropriate transform pipeline for a given split.

    Parameters
    ----------
    split : str
        One of "train", "val", "test".
    image_size : int
        Target spatial resolution (default from config).

    Returns
    -------
    torchvision.transforms.Compose
    """
    assert split in ("train", "val", "test"), \
        f"split must be 'train', 'val', or 'test', got '{split}'"

    if split == "train":
        return T.Compose([
            # ── Geometric augmentations ──────────────────────────────────
            T.RandomResizedCrop(
                image_size,
                scale=(AUG_CROP_SCALE_LOW, 1.0),   # slight zoom / crop
                ratio=(0.85, 1.15),
            ),
            T.RandomHorizontalFlip(p=0.5 if AUG_FLIP else 0.0),
            T.RandomRotation(degrees=AUG_ROTATION),
            # ── Photometric augmentations ─────────────────────────────────
            T.ColorJitter(
                brightness=AUG_BRIGHTNESS,
                contrast=AUG_CONTRAST,
                saturation=0.1,
                hue=0.05,
            ),
            # ── Normalise ─────────────────────────────────────────────────
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    else:   # val or test — deterministic, no augmentation
        return T.Compose([
            T.Resize(int(image_size * 1.14)),   # slight over-size then centre-crop
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


if __name__ == "__main__":
    # Quick sanity check
    from PIL import Image
    import numpy as np

    dummy = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
    for s in ("train", "val", "test"):
        tfm = get_transforms(s)
        out = tfm(dummy)
        print(f"[{s}] output tensor shape: {out.shape}")   # expect torch.Size([3, 224, 224])
