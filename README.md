# Pakistani Politicians Image Classification (CNN)

16-class facial image classification of Pakistani political and military figures, using transfer learning with two pretrained CNN backbones.

| Backbone | Test Accuracy | Macro F1 | Weighted F1 | Parameters |
|---|---|---|---|---|
| **ResNet-50** (ImageNet) | **88.68%** | **0.8890** | **0.8854** | ~23.6 M |
| EfficientNet-B2 (ImageNet) | 85.53% | 0.8540 | 0.8524 | ~7.7 M |

Target was ≥90% test accuracy. ResNet-50 falls just short at 88.68%; both models confirm that ImageNet transfer learning is effective even with under 1,500 training images. Full per-class metrics, confusion matrices, and discussion are in the IEEE-format report under `report/`.

Final deliverables: trained model checkpoints, evaluation metrics, IEEE-styled figures (PNG + PDF), and a model-comparison report.

---

## Classes (16)

`Ahmed_Sharif_Chaudhry, Altaf_Hussain, Asfandyar_Wali, Asif_Ali_Zardari, Bilawal_Bhutto, Chaudhry_Nisar, Fazlur_Rehman, Imran_Khan, Maryam_Nawaz, Nawaz_Sharif, Pervez_Khattak, Pervez_Musharraf, Rana_Sanaullah, Shah_Mehmood_Qureshi, Shehbaz_Sharif, Sirajul_Haq`

Dataset: 1,428 images manually curated from Google Images, Wikipedia, official news sites, and government pages. Per-class counts range from 70 to 125 images. Split 75 / 15 / 10 (train / val / test) with a minimum of 5 test images per class enforced. Filename-level leakage check (using `(class, filename)` pairs) confirms no image appears in more than one split.

---

## Repository layout

```
.
├── config/
│   └── config.py                  # All paths, hyperparameters, class names
├── src/
│   ├── transforms.py              # Train + eval image transforms
│   ├── dataset.py                 # ImageFolder loaders + class alignment check
│   ├── models.py                  # Model builders (ResNet, EfficientNet, VGG)
│   ├── train.py                   # Trainer with crash-resilient per-epoch saves
│   ├── evaluate.py                # Metrics + IEEE-styled plots (PNG + PDF)
│   └── utils.py                   # Drive mount, JSON/CSV writers, model cards
├── scripts/
│   └── split_dataset.py           # Idempotent 75/15/10 splitter
├── notebooks/
│   ├── 01_setup_and_data_audit.ipynb
│   ├── 02_split_dataset.ipynb
│   ├── 03_train_model_A.ipynb           (ResNet-50)
│   ├── 04_train_model_B.ipynb           (EfficientNet-B2)
│   ├── 05_evaluation_and_figures.ipynb
│   └── 06_misclassification_analysis.ipynb
├── results/
│   ├── metrics/                   # JSON metrics, model cards, CSV logs
│   └── plots/                     # PNG + PDF figures
├── report/
│   ├── main.tex                   # IEEE-format report source
│   └── figures/                   # Figures copied here for Overleaf upload
├── requirements.txt
└── README.md
```

---

## How to run (Colab)

The dataset lives on Google Drive at `/content/drive/MyDrive/dataset_resplit`. Checkpoints and per-epoch logs are saved to `/content/drive/MyDrive/pk_politicians_results/` **immediately during training** so a session crash does not lose progress.

Run the notebooks in order:

1. **01_setup_and_data_audit** — verifies dataset is present, counts images per class per split, hard-fails if any class has fewer than 5 test images. Writes `results/metrics/dataset_audit.json`.
2. **02_split_dataset** — idempotent. Skips if the split already looks correct. Use `--force` flag inside the notebook to re-split.
3. **03_train_model_A** — trains ResNet-50. Set `RESUME = True` to continue from the latest checkpoint after a crash; set `FORCE_RETRAIN = True` to start fresh.
4. **04_train_model_B** — same template, EfficientNet-B2.
5. **05_evaluation_and_figures** — loads both checkpoints from Drive, computes test metrics, generates the comparison figure and `results/metrics/model_comparison.csv`.
6. **06_misclassification_analysis** — confusion-pair analysis, most-confidently-wrong gallery, barely-wrong margin analysis, per-class accuracy ranking. Set `MODEL_NAME` at the top of the notebook to switch between models.

Every notebook clones this GitHub repo at the start so you always run the latest code.

---

## Training configuration

Both models share these hyperparameters (single source of truth: `config/config.py` and the per-notebook config cell):

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Weight decay | 1 × 10⁻⁴ |
| LR (classification head) | 1 × 10⁻³ |
| LR (backbone, ResNet-50) | 1 × 10⁻⁴ |
| LR (backbone, EfficientNet-B2) | 5 × 10⁻⁵ |
| Label smoothing | 0.05 |
| Dropout (head) | 0.2 |
| Gradient clip norm (ℓ₂) | 1.0 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Batch size | 32 |
| Max epochs | 35 |
| Early stopping patience | 10 epochs |
| Image size | 224×224 |
| Seed | 42 |

The **two-learning-rate** strategy (lower LR for the pretrained backbone, higher LR for the new classification head) prevents destroying ImageNet features while letting the new head converge quickly.

---

## Data augmentation

Applied **only to the training split**, after the train/val/test split is fixed. Validation and test transforms are deterministic (resize to 256×256 → centre-crop to 224×224 → normalise).

Train transforms (`src/transforms.py`):

- `RandomResizedCrop(224, scale=(0.75, 1.0))`
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(±15°)`
- `ColorJitter(brightness=±0.25)`
- `RandomErasing(p=0.15, scale=(0.02, 0.33))`

All splits are normalised with ImageNet channel statistics (`μ = [0.485, 0.456, 0.406]`, `σ = [0.229, 0.224, 0.225]`).

---

## Crash recovery (the key change from the prior build)

Per-epoch, the trainer writes to **local AND Drive**:
- `epoch_log.csv` — appended after each epoch
- `history.json` — overwritten after each epoch
- `<model>_best.pth` — saved on every val-accuracy improvement, includes optimizer + scheduler + epoch number + best_val_acc + full history (a true full-state checkpoint)

If the Colab runtime dies mid-training:

```python
RESUME = True
FORCE_RETRAIN = False
```

…then re-run the training notebook. It will pick up from the last saved epoch with the optimizer and LR scheduler restored.

---

## Reproducibility

- Seed: `42` (set in `config/config.py`, applied via `src.utils.set_seed`)
- Image size: `224×224`
- Batch size: `32`
- All hyperparameters and run metadata are written to a model card (`.json` + `.md`) under `results/metrics/` after each training run.
- An append-only `results/metrics/experiment_log.jsonl` records every completed run (model name, test accuracy, macro F1, epochs, duration) for cross-run comparison.

---

## Hardware

Training is done on Google Colab with an A100 (40 GB) GPU. ResNet-50 takes ~10 minutes for 35 epochs at batch size 32; EfficientNet-B2 takes ~15 minutes.

---

## Report

The IEEE-format report under `report/main.tex` documents dataset methodology, architectures, training strategy, and per-class results. It expects six figures in `figures/` (uploaded to Overleaf separately):

- `resnet50_training_curves.png`
- `efficientnet_b2_training_curves.png`
- `resnet50_confusion_matrix_normalized.png`
- `efficientnet_b2_confusion_matrix_normalized.png`
- `resnet50_per_class_metrics.png`
- `efficientnet_b2_per_class_metrics.png`

Notebook `03_train_model_A.ipynb` (section 8b) generates the confusion-matrix and per-class metrics PNGs for ResNet-50 directly into `results/plots/` with the exact filenames the LaTeX expects.

---

## Notes

- `scripts/extract_rar.py` from the prior build is **no longer used** — the dataset is already extracted on Drive. Safe to delete.
- All figures are saved as both PNG (200–300 DPI) and PDF (vector, with editable text, `pdf.fonttype = 42`) so they can be dropped into Overleaf without rasterization artifacts.
- An early EfficientNet-B0 run was lost to a Colab session crash before any checkpoint was written — this is what motivated the per-improvement Drive checkpointing described under **Crash recovery**.
