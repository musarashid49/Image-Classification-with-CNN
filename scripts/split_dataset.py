"""
scripts/split_dataset.py
========================
Splits a flat raw dataset into train / val / test directories.

YOUR DRIVE FOLDER STRUCTURE (before running this):
    DRIVE_DATASET_PATH/
        imran_khan/
            img_001.jpg
            img_002.jpg
            ...
        nawaz_sharif/
            ...

AFTER RUNNING:
    output_dir/
        train/imran_khan/...
        val/imran_khan/...
        test/imran_khan/...

Run in Colab:
    !python scripts/split_dataset.py \
        --src "/content/drive/MyDrive/pk_politicians_dataset" \
        --dst "/content/data"
"""

import os, shutil, random, argparse
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def split_dataset(
    src_dir: str,
    dst_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    seed:        int   = RANDOM_SEED,
    dry_run:     bool  = False,
) -> None:
    """
    Splits raw per-class image folders into train/val/test.

    Parameters
    ----------
    src_dir     : path to raw dataset (one subfolder per class)
    dst_dir     : destination root (will create train/val/test subfolders)
    dry_run     : if True, print plan but do not copy any files
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    src = Path(src_dir)
    dst = Path(dst_dir)

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {src_dir}")

    print(f"\n── Dataset Split ───────────────────────────────────────────")
    print(f"  Source      : {src_dir}")
    print(f"  Destination : {dst_dir}")
    print(f"  Ratios      : train={train_ratio}  val={val_ratio}  test={test_ratio}")
    print(f"  Seed        : {seed}")
    print(f"  Dry run     : {dry_run}\n")

    total_copied = {"train": 0, "val": 0, "test": 0}

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in VALID_EXTS
        ])
        n = len(images)
        random.shuffle(images)

        n_train = round(n * train_ratio)
        n_val   = round(n * val_ratio)
        # remainder goes to test to avoid rounding drift
        n_test  = n - n_train - n_val

        splits_files = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        print(f"  {cls_name:<35}  total={n:>4}  "
              f"train={n_train:>4}  val={n_val:>3}  test={n_test:>3}")

        if n < 80:
            print(f"    ⚠  WARNING: only {n} images — minimum required is 80!")

        if not dry_run:
            for split, files in splits_files.items():
                out_dir = dst / split / cls_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for img_path in files:
                    shutil.copy2(img_path, out_dir / img_path.name)
                total_copied[split] += len(files)

    if not dry_run:
        print(f"\n  Files copied:")
        for split, count in total_copied.items():
            print(f"    {split:<8} {count:>5} images")
        print("\n  ✓ Split complete — no leakage between sets.")
    else:
        print("\n  (dry run — no files were copied)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--src",        required=True,  help="Raw dataset root folder")
    parser.add_argument("--dst",        required=True,  help="Output root folder")
    parser.add_argument("--train",      type=float,     default=TRAIN_RATIO)
    parser.add_argument("--val",        type=float,     default=VAL_RATIO)
    parser.add_argument("--test",       type=float,     default=TEST_RATIO)
    parser.add_argument("--seed",       type=int,       default=RANDOM_SEED)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    split_dataset(
        src_dir    = args.src,
        dst_dir    = args.dst,
        train_ratio= args.train,
        val_ratio  = args.val,
        test_ratio = args.test,
        seed       = args.seed,
        dry_run    = args.dry_run,
    )
