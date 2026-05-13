# Pakistani Politicians Image Classification

Multi-class facial image classification of **16 Pakistani public figures** using pretrained CNNs (ResNet-50, EfficientNet-B0/B2).

> Academic project — Deep Learning course assignment.

---

## 📋 Project Overview

| Item | Detail |
|---|---|
| Task | 16-class facial image classification |
| Models | ResNet-50, EfficientNet-B0 (+ optional B2) |
| Dataset | Self-collected, 80–250 images per class |
| Split | 75% Train / 15% Val / 10% Test |
| Framework | PyTorch + torchvision |
| Training | Google Colab A100 |

---

## 📁 Repository Structure

```
pk_politicians_clf/
├── config/
│   └── config.py               ← All hyperparameters & paths (edit here first)
├── src/
│   ├── transforms.py           ← Train/val/test augmentation pipelines
│   ├── dataset.py              ← ImageFolder DataLoaders
│   ├── models.py               ← ResNet50 / EfficientNet factory
│   ├── train.py                ← Training loop with early stopping
│   ├── evaluate.py             ← Metrics, confusion matrix, plots
│   └── utils.py                ← Seed, Drive helpers, experiment logger
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_split_and_verify.ipynb
│   ├── 03_train_resnet50.ipynb
│   ├── 04_train_efficientnet.ipynb
│   └── 05_evaluation_and_plots.ipynb
├── scripts/
│   ├── split_dataset.py        ← CLI: split raw → train/val/test
│   ├── verify_dataset.py       ← CLI: quality control checks
│   └── export_results.py       ← CLI: build summary CSVs
├── experiments/
│   └── experiment_log.json     ← Auto-generated run history
├── results/
│   ├── plots/                  ← Saved PNG figures (gitignored binaries)
│   ├── metrics/                ← JSON reports & CSV tables
│   └── checkpoints/            ← Best model .pth files (gitignored)
├── data/                       ← Gitignored — stored on Google Drive
├── report/figures/             ← Copy plots here for IEEE report
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/pk_politicians_clf.git
cd pk_politicians_clf
pip install -r requirements.txt
```

### 2. Configure paths
Edit `config/config.py`:
```python
DRIVE_DATASET_PATH = "/content/drive/MyDrive/YOUR_DATASET_FOLDER"
DRIVE_RESULTS_PATH = "/content/drive/MyDrive/YOUR_RESULTS_FOLDER"
```

### 3. Run notebooks in order (on Colab A100)
| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_exploration.ipynb` | Inspect raw dataset, detect corrupt images |
| 02 | `02_split_and_verify.ipynb` | 75/15/10 split + leakage check |
| 03 | `03_train_resnet50.ipynb` | Train & evaluate ResNet-50 |
| 04 | `04_train_efficientnet.ipynb` | Train & evaluate EfficientNet-B0 |
| 05 | `05_evaluation_and_plots.ipynb` | Final comparison & report figures |

---

## 🧑‍🤝‍🧑 Classes (16)

| Index | Folder Name |
|---|---|
| 0 | asif_ali_zardari |
| 1 | bilawal_bhutto |
| 2 | chaudhry_shujaat |
| 3 | fazlur_rehman |
| 4 | imran_khan |
| 5 | ishaq_dar |
| 6 | khawaja_asif |
| 7 | maryam_nawaz |
| 8 | moeed_yusuf |
| 9 | nawaz_sharif |
| 10 | pervaiz_elahi |
| 11 | pervez_musharraf |
| 12 | raheel_sharif |
| 13 | shehbaz_sharif |
| 14 | siraj_ul_haq |
| 15 | zahid_hamid |

---

## 📊 Evaluation Targets

- ≥ 90% overall test accuracy
- Per-class precision, recall, F1
- Confusion matrix heatmap
- Training vs. validation loss/accuracy curves
- Top-5 misclassified samples

---

## ⚠️ Important Rules

- **Augmentation is only applied to training data** — val and test use deterministic transforms.
- **Split before augmentation** — the `split_dataset.py` script handles raw images only.
- **No data leakage** — verified by `src/utils.py::verify_no_leakage()`.
- Dataset images are **not committed to GitHub** — stored on Google Drive.
- Model checkpoints are **not committed** — stored on Google Drive.

---

## 📄 Report

IEEE-format report written in Overleaf. Figures exported from `results/plots/` → `report/figures/`.

---

## 🛠 Tech Stack

`PyTorch` · `torchvision` · `scikit-learn` · `matplotlib` · `seaborn` · `Pillow` · `pandas`
