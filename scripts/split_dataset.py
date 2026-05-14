"""
Idempotent dataset splitter.

Usage (from a Colab cell or terminal):
    python scripts/split_dataset.py \
        --src /content/drive/MyDrive/dataset \
        --dst /content/drive/MyDrive/dataset_resplit \
        --train 0.75 --val 0.15 --test 0.10 --seed 42

If the destination already contains a valid split (train/val/test each
non-empty for every class, and >= MIN_TEST_IMAGES_PER_CLASS test images
per class), the script exits cleanly with a "nothing to do" message.

Splits images per-class to guarantee class balance across train/val/test.
Pass --force to overwrite an existing destination.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Allow running this file as `python scripts/split_dataset.py` from repo root
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.config import (  # noqa: E402
    CLASS_NAMES,
    MIN_TEST_IMAGES_PER_CLASS,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
def _list_class_images(class_dir: Path) -> List[Path]:
    return sorted(
        [p for p in class_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def _gather_all_images_from_split(src_root: Path, classes: List[str]) -> dict:
    """
    If src_root has train/val/test subdirs, gather images from all three;
    otherwise read directly from class folders.
    """
    out = {c: [] for c in classes}
    has_splits = all((src_root / s).exists() for s in ("train", "val", "test"))
    if has_splits:
        for split in ("train", "val", "test"):
            for c in classes:
                d = src_root / split / c
                if d.exists():
                    out[c].extend(_list_class_images(d))
    else:
        for c in classes:
            d = src_root / c
            if d.exists():
                out[c].extend(_list_class_images(d))
            else:
                print(f"[warn] class folder missing in src: {c}")
    return out


def _is_destination_valid(dst: Path, classes: List[str],
                          min_test_per_class: int) -> Tuple[bool, str]:
    """Return (valid, message). Valid = all splits exist + every class has
    enough test images."""
    for s in ("train", "val", "test"):
        if not (dst / s).exists():
            return False, f"missing split dir: {dst / s}"
    for c in classes:
        n_test = sum(
            1 for p in (dst / "test" / c).glob("*") if p.is_file()
        ) if (dst / "test" / c).exists() else 0
        if n_test < min_test_per_class:
            return False, f"class '{c}' has only {n_test} test images "\
                          f"(< {min_test_per_class})"
    return True, "destination is already a valid split"


def _split_for_class(
    images: List[Path], ratios: Tuple[float, float, float], rng: random.Random,
) -> Tuple[List[Path], List[Path], List[Path]]:
    images = list(images)
    rng.shuffle(images)
    n = len(images)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_test = n - n_train - n_val
    if n_test < 1:
        # Trade one from val if val>1, else from train
        if n_val > 1:
            n_val -= 1
            n_test += 1
        elif n_train > 1:
            n_train -= 1
            n_test += 1
    return (
        images[:n_train],
        images[n_train:n_train + n_val],
        images[n_train + n_val:n_train + n_val + n_test],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotent train/val/test splitter.")
    parser.add_argument("--src", required=True, help="source dataset directory")
    parser.add_argument("--dst", required=True, help="destination split directory")
    parser.add_argument("--train", type=float, default=0.75)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="overwrite destination even if a valid split exists")
    parser.add_argument("--min-test", type=int, default=MIN_TEST_IMAGES_PER_CLASS)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"[error] source does not exist: {src}", file=sys.stderr)
        return 2

    ratios = (args.train, args.val, args.test)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print(f"[error] ratios must sum to 1: {ratios}", file=sys.stderr)
        return 2

    classes = list(CLASS_NAMES)

    # Idempotency check
    if dst.exists() and not args.force:
        valid, msg = _is_destination_valid(dst, classes, args.min_test)
        if valid:
            print(f"[skip] {msg} — pass --force to re-split.")
            return 0
        else:
            print(f"[info] destination exists but is invalid ({msg}); re-splitting.")
            shutil.rmtree(dst)
    elif dst.exists() and args.force:
        print(f"[force] removing existing destination: {dst}")
        shutil.rmtree(dst)

    rng = random.Random(args.seed)

    print(f"[split] source: {src}")
    print(f"[split] dest:   {dst}")
    print(f"[split] ratios: train={ratios[0]}, val={ratios[1]}, test={ratios[2]}")
    print(f"[split] seed:   {args.seed}")

    per_class_images = _gather_all_images_from_split(src, classes)

    total_n = sum(len(v) for v in per_class_images.values())
    print(f"[split] gathered {total_n} images across {len(classes)} classes")

    summary = {"train": 0, "val": 0, "test": 0}
    for c in classes:
        imgs = per_class_images[c]
        if not imgs:
            print(f"[error] class '{c}' has zero images in source!", file=sys.stderr)
            return 3
        train_imgs, val_imgs, test_imgs = _split_for_class(imgs, ratios, rng)
        for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            out_dir = dst / split_name / c
            out_dir.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy2(p, out_dir / p.name)
            summary[split_name] += len(split_imgs)
        print(f"  {c:<28s} train={len(train_imgs):3d} val={len(val_imgs):3d} test={len(test_imgs):3d}")

    print(f"\n[done] totals: train={summary['train']} val={summary['val']} test={summary['test']}")

    # Final validation pass
    valid, msg = _is_destination_valid(dst, classes, args.min_test)
    if not valid:
        print(f"[error] post-split validation failed: {msg}", file=sys.stderr)
        return 4
    print("[ok] destination is a valid split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
