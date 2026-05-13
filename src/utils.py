"""
src/utils.py
============
Shared utility functions:
  - reproducibility seed setter
  - Drive mount / copy helpers (Colab)
  - checkpoint loader
  - pretty experiment logger
"""

import os, random, shutil, json, time
import numpy as np
import torch


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Sets seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  Global seed set to {seed}")


# ─────────────────────────────────────────────
# Google Drive helpers (Colab)
# ─────────────────────────────────────────────

def mount_drive() -> None:
    """Mounts Google Drive inside Colab. No-op if not in Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("  Google Drive mounted at /content/drive")
    except ImportError:
        print("  Not running in Colab — Drive mount skipped.")


def copy_dataset_from_drive(
    drive_dataset_path: str,
    local_data_dir: str = "/content/data",
) -> None:
    """
    Copies the already-split dataset from Drive to local Colab SSD.
    Expected Drive layout:
        drive_dataset_path/
            train/<class>/...
            val/<class>/...
            test/<class>/...
    """
    if os.path.isdir(local_data_dir) and len(os.listdir(local_data_dir)) > 0:
        print(f"  Dataset already exists at {local_data_dir} — skipping copy.")
        return

    print(f"  Copying dataset from Drive …")
    t0 = time.time()
    shutil.copytree(drive_dataset_path, local_data_dir)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  →  {local_data_dir}")


def save_results_to_drive(
    local_results_dir: str,
    drive_results_path: str,
) -> None:
    """Copies local results folder back to Drive after training."""
    print(f"  Saving results to Drive …")
    if os.path.isdir(drive_results_path):
        shutil.rmtree(drive_results_path)
    shutil.copytree(local_results_dir, drive_results_path)
    print(f"  Results saved → {drive_results_path}")


# ─────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────

def load_checkpoint(model, checkpoint_path: str, device: torch.device = None):
    """
    Loads model weights from a checkpoint file.
    Returns the checkpoint dict (which may contain epoch, val_acc, etc.).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Checkpoint loaded: {checkpoint_path}")
    print(f"    epoch={ckpt.get('epoch', '?')}  val_acc={ckpt.get('val_acc', '?'):.4f}")
    return ckpt


# ─────────────────────────────────────────────
# Experiment logger
# ─────────────────────────────────────────────

class ExperimentLogger:
    """
    Lightweight JSON-based experiment tracker.
    Creates / appends to experiments/experiment_log.json.

    Usage:
        logger = ExperimentLogger()
        logger.log(model_name="resnet50", val_acc=0.93, ...)
    """
    def __init__(self, log_path: str = "experiments/experiment_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.isfile(log_path):
            with open(log_path, "w") as f:
                json.dump([], f)

    def log(self, **kwargs) -> None:
        kwargs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "r") as f:
            records = json.load(f)
        records.append(kwargs)
        with open(self.log_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Experiment logged → {self.log_path}")
        for k, v in kwargs.items():
            print(f"    {k}: {v}")

    def get_all(self):
        with open(self.log_path, "r") as f:
            return json.load(f)


# ─────────────────────────────────────────────
# Dataset split verifier
# ─────────────────────────────────────────────

def verify_no_leakage(train_dir: str, val_dir: str, test_dir: str) -> None:
    """
    Checks for filename overlap across splits.
    Raises AssertionError if leakage is detected.
    """
    def get_filenames(root):
        names = set()
        for cls in os.listdir(root):
            cls_path = os.path.join(root, cls)
            if os.path.isdir(cls_path):
                for f in os.listdir(cls_path):
                    names.add(f)
        return names

    train_files = get_filenames(train_dir)
    val_files   = get_filenames(val_dir)
    test_files  = get_filenames(test_dir)

    tv = train_files & val_files
    tt = train_files & test_files
    vt = val_files   & test_files

    assert len(tv) == 0, f"LEAKAGE: {len(tv)} files shared between train and val!"
    assert len(tt) == 0, f"LEAKAGE: {len(tt)} files shared between train and test!"
    assert len(vt) == 0, f"LEAKAGE: {len(vt)} files shared between val and test!"
    print("  ✓ No leakage detected between train / val / test splits.")
