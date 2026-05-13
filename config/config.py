"""
config/config.py
================
Central configuration for the Pakistani Politicians Classification project.
Edit DRIVE_DATASET_PATH to point to your Google Drive dataset folder.
All other modules import from here — change once, applies everywhere.
"""

import os

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────

# Root of your dataset folder on Google Drive (Colab path after mounting)
# Example: "/content/drive/MyDrive/pk_politicians_dataset"
DRIVE_DATASET_PATH = "/content/drive/MyDrive/pk_politicians_dataset"

# Local Colab working directory (fast SSD — copy dataset here before training)
COLAB_DATA_DIR    = "/content/data"
TRAIN_DIR         = os.path.join(COLAB_DATA_DIR, "train")
VAL_DIR           = os.path.join(COLAB_DATA_DIR, "val")
TEST_DIR          = os.path.join(COLAB_DATA_DIR, "test")

# Results output (also save back to Drive after training)
RESULTS_DIR       = "results"
PLOTS_DIR         = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR       = os.path.join(RESULTS_DIR, "metrics")
CHECKPOINTS_DIR   = os.path.join(RESULTS_DIR, "checkpoints")

# Drive mirror for saving results
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/pk_politicians_results"

# ─────────────────────────────────────────────
# 2. DATASET / CLASSES
# ─────────────────────────────────────────────

CLASS_NAMES = [
    "ahmed_sharif_chaudhry",
    "altaf_hussain",
    "asfandyar_wali",
    "asif_ali_zardari",
    "bilawal_bhutto",
    "chaudhry_nisar",
    "fazlur_rehman",
    "imran_khan",
    "maryam_nawaz",
    "nawaz_sharif",
    "pervez_khattak",
    "pervez_musharraf",
    "rana_sanaullah",
    "shah_mehmood_qureshi",
    "shehbaz_sharif",
    "sirajul_haq",
]

NUM_CLASSES = len(CLASS_NAMES)   # 16

# ─────────────────────────────────────────────
# 3. SPLIT RATIOS
# ─────────────────────────────────────────────

TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10
RANDOM_SEED = 42          # keep fixed for reproducibility

# ─────────────────────────────────────────────
# 4. IMAGE SETTINGS
# ─────────────────────────────────────────────

IMAGE_SIZE   = 224        # used for ResNet50 and EfficientNet-B0/B2
CHANNELS     = 3

# ImageNet normalisation (standard for pretrained CNNs)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# 5. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────

BATCH_SIZE       = 32
NUM_EPOCHS       = 30
LEARNING_RATE    = 1e-4          # initial LR for fine-tuning
WEIGHT_DECAY     = 1e-4
MOMENTUM         = 0.9           # for SGD if used
LR_PATIENCE      = 4             # ReduceLROnPlateau patience
EARLY_STOP_PAT   = 8             # early stopping patience
NUM_WORKERS      = 4             # DataLoader workers

# ─────────────────────────────────────────────
# 6. MODEL CONFIGS
# ─────────────────────────────────────────────

MODELS = {
    "resnet50": {
        "name":        "resnet50",
        "pretrained":  True,
        "freeze_base": False,     # set True for feature-extraction only phase
        "dropout":     0.4,
        "image_size":  224,
    },
    "efficientnet_b0": {
        "name":        "efficientnet_b0",
        "pretrained":  True,
        "freeze_base": False,
        "dropout":     0.4,
        "image_size":  224,
    },
    "efficientnet_b2": {
        "name":        "efficientnet_b2",
        "pretrained":  True,
        "freeze_base": False,
        "dropout":     0.4,
        "image_size":  260,       # B2 native resolution
    },
}

# ─────────────────────────────────────────────
# 7. AUGMENTATION FLAGS
# ─────────────────────────────────────────────

AUG_ROTATION       = 20      # degrees
AUG_FLIP           = True    # horizontal flip
AUG_BRIGHTNESS     = 0.3     # ColorJitter brightness delta
AUG_CONTRAST       = 0.2
AUG_ZOOM           = 0.15    # RandomResizedCrop scale lower bound = 1 - AUG_ZOOM
AUG_CROP_SCALE_LOW = 0.85    # RandomResizedCrop scale = (0.85, 1.0)
