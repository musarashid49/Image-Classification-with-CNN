"""
scripts/extract_rar.py
======================
Extracts the dataset RAR from Google Drive and auto-detects the internal
folder structure (three common layouts are handled automatically).

Supported RAR layouts
─────────────────────
Layout A — class folders at root level (most common):
    dataset.rar/
        imran_khan/001.jpg ...
        nawaz_sharif/001.jpg ...

Layout B — one extra wrapper folder:
    dataset.rar/
        politicians_dataset/
            imran_khan/001.jpg ...
            nawaz_sharif/001.jpg ...

Layout C — flat (all images in root, filename contains class name):
    dataset.rar/
        imran_khan_001.jpg
        nawaz_sharif_001.jpg
    NOT supported automatically — contact instructor to re-organize.

Run in Colab cell:
    !pip install -q rarfile
    !apt-get install -q unrar
    !python scripts/extract_rar.py \
        --rar  "/content/drive/MyDrive/politicians.rar" \
        --dst  "/content/raw_dataset"
"""

import os, shutil, argparse
from pathlib import Path

try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import CLASS_NAMES

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_layout(rar_path: str) -> str:
    """
    Peeks inside the RAR to detect whether class folders are at the root
    or wrapped in one extra parent directory.

    Returns 'A' (root-level class folders) or 'B' (wrapped).
    """
    import rarfile as rf
    with rf.RarFile(rar_path) as r:
        names = r.namelist()

    # Collect unique top-level names
    top_level = set()
    for n in names:
        parts = Path(n).parts
        if parts:
            top_level.add(parts[0])

    # If top-level dirs match (some of) our class names → Layout A
    overlap = top_level & set(CLASS_NAMES)
    if len(overlap) >= max(1, len(CLASS_NAMES) // 2):
        return "A"

    # Otherwise assume one wrapper folder → Layout B
    return "B"


def extract_rar(rar_path: str, dst_dir: str, dry_run: bool = False) -> str:
    """
    Extracts the RAR file to dst_dir and returns the path to the
    flat class-organised root (one subfolder per class).
    """
    assert os.path.isfile(rar_path), f"RAR not found: {rar_path}"

    if not RAR_AVAILABLE:
        raise ImportError(
            "rarfile not installed.\n"
            "Run:  !pip install rarfile && !apt-get install -q unrar"
        )

    import rarfile as rf

    extract_root = os.path.join(dst_dir, "_rar_extracted")

    if os.path.isdir(extract_root):
        print(f"  Extraction folder already exists: {extract_root}")
        print("  Skipping extraction (delete folder to re-extract).")
    else:
        print(f"  Extracting {rar_path}  →  {extract_root}")
        if not dry_run:
            os.makedirs(extract_root, exist_ok=True)
            with rf.RarFile(rar_path) as r:
                r.extractall(extract_root)
            print("  Extraction complete ✓")

    # Auto-detect layout
    layout = detect_layout(rar_path)
    print(f"  Detected layout: {layout}")

    if layout == "A":
        class_root = extract_root
    else:
        # Layout B: descend one level
        subdirs = [
            d for d in Path(extract_root).iterdir() if d.is_dir()
        ]
        if len(subdirs) == 1:
            class_root = str(subdirs[0])
            print(f"  Wrapper folder unwrapped: {class_root}")
        else:
            # Multiple subdirs — pick the one with the most class matches
            best, best_count = str(subdirs[0]), 0
            for sd in subdirs:
                children = {d.name for d in sd.iterdir() if d.is_dir()}
                count = len(children & set(CLASS_NAMES))
                if count > best_count:
                    best, best_count = str(sd), count
            class_root = best
            print(f"  Best matching subfolder: {class_root}")

    # Report what was found
    print(f"\n  Class folders found in: {class_root}")
    found = sorted([d.name for d in Path(class_root).iterdir() if d.is_dir()])
    for cls in found:
        imgs = list((Path(class_root) / cls).rglob("*"))
        imgs = [f for f in imgs if f.suffix.lower() in VALID_EXTS]
        flag = "✓" if len(imgs) >= 80 else "⚠ LOW"
        print(f"    {flag}  {cls:<35} {len(imgs):>4} images")

    missing = set(CLASS_NAMES) - set(found)
    if missing:
        print(f"\n  ⚠  Classes in config but NOT in RAR:")
        for m in sorted(missing):
            print(f"      {m}")
        print("\n  → Update CLASS_NAMES in config/config.py to match actual folder names.")

    return class_root


def rename_folders_to_config(class_root: str) -> None:
    """
    Interactive helper: prints a mapping suggestion if folder names
    don't exactly match CLASS_NAMES.  Does NOT rename automatically.
    """
    found = sorted([d.name for d in Path(class_root).iterdir() if d.is_dir()])
    expected = set(CLASS_NAMES)

    extra = set(found) - expected
    if extra:
        print("\n  ⚠  These folder names are NOT in CLASS_NAMES in config/config.py:")
        for e in sorted(extra):
            print(f"      '{e}'")
        print("\n  Fix options:")
        print("   1. Rename the folders in your Drive RAR to match config CLASS_NAMES")
        print("   2. OR update CLASS_NAMES in config/config.py to match your folder names")
        print("      (Option 2 is easier — just edit config.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rar", required=True, help="Path to .rar file on Drive")
    parser.add_argument("--dst", required=True, help="Local destination directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    class_root = extract_rar(args.rar, args.dst, args.dry_run)
    rename_folders_to_config(class_root)
    print(f"\n  ✓ Dataset root ready at: {class_root}")
    print("  Pass this path as --src to scripts/split_dataset.py")
