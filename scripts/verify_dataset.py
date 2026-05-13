"""
scripts/verify_dataset.py
=========================
Quality-control checks on the split dataset:
  - per-class image counts
  - missing / below-minimum classes
  - corrupt / unreadable image detection
  - class distribution bar chart

Run in Colab:
    !python scripts/verify_dataset.py --data "/content/data"
"""

import os, argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # headless-safe

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import CLASS_NAMES, PLOTS_DIR

MIN_IMAGES = 80


def check_split(split_dir: str, split_name: str) -> dict:
    """
    Checks a single split directory. Returns {class_name: count}.
    """
    split_path = Path(split_dir)
    if not split_path.is_dir():
        print(f"  ✗ {split_name} directory not found: {split_dir}")
        return {}

    counts = {}
    corrupt = []

    for cls_dir in sorted(split_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        images = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        counts[cls_name] = len(images)

        if PIL_AVAILABLE:
            for img_path in images:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt.append(str(img_path))

    return counts, corrupt


def print_report(split_name: str, counts: dict) -> None:
    print(f"\n── {split_name.upper()} split ──────────────────────────────────────")
    total = 0
    for cls in CLASS_NAMES:
        n    = counts.get(cls, 0)
        flag = "✓" if n >= 1 else "✗ MISSING"
        low  = "  ⚠ LOW" if n < MIN_IMAGES and split_name.lower() == "train" else ""
        print(f"  {flag}  {cls:<35} {n:>4}{low}")
        total += n
    # Report unexpected classes
    for cls in counts:
        if cls not in CLASS_NAMES:
            print(f"  ⚠  Unexpected class: {cls}  ({counts[cls]} images)")
    print(f"\n  Total images: {total}  |  Classes: {len(counts)}")


def plot_distribution(all_counts: dict, save_path: str = None) -> None:
    """Bar chart of image counts per class per split."""
    splits = list(all_counts.keys())
    classes = CLASS_NAMES
    n_classes = len(classes)
    n_splits  = len(splits)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    x = range(n_classes)
    width = 0.8 / n_splits

    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (split, color) in enumerate(zip(splits, colors)):
        counts = [all_counts[split].get(cls, 0) for cls in classes]
        offset = (i - n_splits / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], counts, width, label=split, color=color, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Image Count")
    ax.set_title("Dataset Distribution per Class per Split")
    ax.axhline(y=MIN_IMAGES, color="red", linestyle="--", linewidth=0.8, label=f"Min ({MIN_IMAGES})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"\n  Distribution chart saved → {save_path}")
    plt.close(fig)


def verify(data_dir: str) -> None:
    all_counts = {}
    all_corrupt = []

    for split in ("train", "val", "test"):
        split_dir = os.path.join(data_dir, split)
        counts, corrupt = check_split(split_dir, split)
        all_counts[split] = counts
        all_corrupt.extend(corrupt)
        print_report(split, counts)

    if all_corrupt:
        print(f"\n  ✗ {len(all_corrupt)} corrupt image(s) found:")
        for p in all_corrupt[:20]:
            print(f"    {p}")
        if len(all_corrupt) > 20:
            print(f"    ... and {len(all_corrupt) - 20} more.")
    else:
        print("\n  ✓ No corrupt images detected.")

    chart_path = os.path.join(PLOTS_DIR, "dataset_distribution.png")
    plot_distribution(all_counts, save_path=chart_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="/content/data",
        help="Root directory containing train/, val/, test/ subfolders"
    )
    args = parser.parse_args()
    verify(args.data)
