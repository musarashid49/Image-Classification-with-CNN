# Pakistani Politicians Image Classification (CNN)

16-class facial image classification of Pakistani political and military figures, using transfer learning with two pretrained CNN backbones. Target test accuracy: **≥90%**.

| Model A | Model B |
|---|---|
| ResNet-50 (ImageNet) | EfficientNet-B2 (ImageNet) |

Final deliverables: trained model checkpoints, evaluation metrics, IEEE-styled figures (PNG + PDF), and a model-comparison report.

---

## Classes (16)

`Asif_Ali_Zardari, Asim_Munir, Bilawal_Bhutto, Hamza_Shahbaz, Imran_Khan, Maryam_Nawaz, Maulana_Fazal_ur_Rehman, Mehmood_Khan_Achakzai, Mehmood_Qureshi, Nawaz_Sharif, Pervez_Khattak, Qamar_Javed_Bajwa, Sheikh_Rasheed_Ahmed, Shehbaz_Sharif, Sirajul_Haq, Yousaf_Raza_Gillani`

Dataset: ~1,428 images, split 75 / 15 / 10 (train / val / test) with a minimum of 5 test images per class enforced.

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
│   └── figures/                   # Copies of figures for Overleaf
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

## Crash recovery (this is the key change from the prior build)

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

---

## First-time GitHub setup (foolproof)

The repo's default branch is **`main`**. If your local clone is on `master`, rename it before the first push:

```bash
# from inside the repo root
git branch -m master main
git fetch origin
git branch -u origin/main main   # only if origin/main already exists
git remote set-head origin -a    # only if origin/main already exists
```

If `origin/main` does **not** yet exist on GitHub, do the first push like this:

```bash
git add .
git commit -m "Rebuild: crash-resilient training pipeline + 6 notebooks"
git push -u origin main
```

Then on GitHub: **Settings → Branches → Default branch → change to `main`** if it was previously `master`. After that, `master` can be deleted on GitHub.

For subsequent commits from a Colab notebook (after training a model):

```python
%cd /content/<repo-folder>
!git config user.email "you@example.com"
!git config user.name  "Your Name"
!git add results/metrics/ results/plots/ report/figures/
!git commit -m "Add ResNet-50 model card and figures"
!git push origin main
```

You will be prompted for your GitHub username and a **personal access token** (not your password) the first time. Generate one at GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → `repo` scope.

---

## Hardware

Training is done on Google Colab with an A100 GPU. ResNet-50 takes ~10 minutes for 35 epochs at batch size 32; EfficientNet-B2 takes ~15 minutes.

---

## Notes

- `scripts/extract_rar.py` from the prior build is **no longer used** — the dataset is already extracted on Drive. Safe to delete.
- All figures are saved as both PNG (300 DPI) and PDF (vector, with editable text, `pdf.fonttype = 42`) so they can be dropped into Overleaf without rasterization artifacts.
