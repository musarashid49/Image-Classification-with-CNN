"""
Central configuration for the Pakistani Politicians Image Classification project.

All paths assume Google Colab + Google Drive. Override in notebooks if running
locally on Mac (no GPU) — only the dataset path and DEVICE matter then.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Class names (folder names, must match exactly what's on Drive)
# ---------------------------------------------------------------------------
# Keep these alphabetised so ImageFolder gives the same class_to_idx every run.
CLASS_NAMES: List[str] = sorted([
    "asif_ali_zardari",
    "benazir_bhutto",
    "bilawal_bhutto_zardari",
    "fazal_ur_rehman",
    "imran_khan",
    "ishaq_dar",
    "khawaja_asif",
    "mahmood_khan_achakzai",
    "maryam_nawaz",
    "nawaz_sharif",
    "pervez_musharraf",
    "rana_sanaullah",
    "shahbaz_sharif",
    "sheikh_rasheed",
    "yousaf_raza_gillani",
    "asim_munir",  # military spokesperson / COAS — adjust if folder differs
])

NUM_CLASSES: int = len(CLASS_NAMES)
assert NUM_CLASSES == 16, f"Expected 16 classes, got {NUM_CLASSES}"

# ---------------------------------------------------------------------------
# Paths (Colab + Drive layout)
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive")
DATASET_DIR = DRIVE_ROOT / "dataset_resplit"          # the GOOD split
DATASET_DIR_LEGACY = DRIVE_ROOT / "dataset"           # original (bad) split

# Where Drive stores all run outputs (checkpoints + plots + logs)
RESULTS_DRIVE = DRIVE_ROOT / "pk_politicians_results"
CHECKPOINTS_DRIVE = RESULTS_DRIVE / "checkpoints"
PLOTS_DRIVE = RESULTS_DRIVE / "plots"
LOGS_DRIVE = RESULTS_DRIVE / "logs"

# Local (Colab session) workspace — wiped on disconnect
LOCAL_ROOT = Path("/content/Image-Classification-with-CNN")
LOCAL_RESULTS = LOCAL_ROOT / "results"
LOCAL_PLOTS = LOCAL_RESULTS / "plots"
LOCAL_CHECKPOINTS = LOCAL_RESULTS / "checkpoints"
LOCAL_METRICS = LOCAL_RESULTS / "metrics"
REPORT_FIGURES = LOCAL_ROOT / "report" / "figures"

# ---------------------------------------------------------------------------
# Split ratios (must match Project_Brief.docx exactly)
# ---------------------------------------------------------------------------
SPLIT_TRAIN = 0.75
SPLIT_VAL = 0.15
SPLIT_TEST = 0.10
assert abs(SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST - 1.0) < 1e-9

MIN_IMAGES_PER_CLASS = 80          # strict minimum from project rules
MIN_TEST_IMAGES_PER_CLASS = 5      # hard validation threshold (Notebook 01)

# ---------------------------------------------------------------------------
# Image / training defaults
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
NUM_WORKERS = 2          # Colab has limited CPU; 2 is safe
PIN_MEMORY = True

# Training defaults — override per model in notebooks
DEFAULT_EPOCHS = 30
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_BACKBONE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 7

SEED = 42

# ---------------------------------------------------------------------------
# Repo
# ---------------------------------------------------------------------------
GITHUB_REPO_URL = "https://github.com/musarashid49/Image-Classification-with-CNN.git"
DEFAULT_BRANCH = "main"
