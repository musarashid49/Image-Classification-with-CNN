"""
Evaluation: compute metrics on a DataLoader and produce IEEE-quality plots.

Every plotting function saves to BOTH:
  - `results/plots/<model>_<fig>.{png,pdf}`     (committed to GitHub)
  - `report/figures/<model>_<fig>.{png,pdf}`    (used by Overleaf)

Plots use serif fonts, 300 DPI for PNG, and vector PDF. Colour palettes
are colourblind-friendly (`viridis`/`tab10`) and print well in greyscale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from torch.utils.data import DataLoader

from src.utils import ensure_dir, write_json


# ---------------------------------------------------------------------------
# IEEE-style matplotlib defaults
# ---------------------------------------------------------------------------
def apply_ieee_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,    # TrueType so text stays editable in Overleaf
        "ps.fonttype": 42,
    })


def _save_both(fig, name: str, plots_dir: Path, report_dir: Path) -> Dict[str, Path]:
    """Save a figure as PNG and PDF in plots_dir and (mirror) report_dir."""
    ensure_dir(plots_dir)
    ensure_dir(report_dir)
    paths = {}
    for ext in ("png", "pdf"):
        p1 = plots_dir / f"{name}.{ext}"
        p2 = report_dir / f"{name}.{ext}"
        fig.savefig(p1)
        fig.savefig(p2)
        paths[ext] = p1
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_on_loader(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred, y_prob).
    y_prob has shape (N, num_classes).
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = [
        {
            "class": cn,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cn in enumerate(class_names)
    ]
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
    }


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str],
    out_path: Path | str,
) -> None:
    txt = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
        digits=4,
    )
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(txt)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    plots_dir: Path,
    report_dir: Path,
) -> Dict[str, Path]:
    """Training & validation loss + accuracy, side by side."""
    apply_ieee_style()
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"], marker="s", label="val", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"{model_name}: loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train", linewidth=1.5)
    axes[1].plot(epochs, history["val_acc"], marker="s", label="val", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name}: accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return _save_both(fig, f"{model_name}_training_curves", plots_dir, report_dir)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    class_names: Sequence[str],
    model_name: str,
    plots_dir: Path,
    report_dir: Path,
    normalize: bool = False,
) -> Dict[str, Path]:
    """Large enough to read all 16 class labels without overlap."""
    apply_ieee_style()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_disp = np.nan_to_num(cm_disp)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm_disp,
        annot=True, fmt=fmt, cmap="Blues",
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        cbar=True, square=False, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title_suffix = " (row-normalised)" if normalize else ""
    ax.set_title(f"{model_name}: confusion matrix{title_suffix}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    suffix = "_norm" if normalize else ""
    return _save_both(fig, f"{model_name}_confusion_matrix{suffix}", plots_dir, report_dir)


def plot_per_class_metrics(
    metrics: Dict, model_name: str,
    plots_dir: Path, report_dir: Path,
) -> Dict[str, Path]:
    """Grouped horizontal bar chart of per-class precision/recall/F1."""
    apply_ieee_style()
    rows = metrics["per_class"]
    classes = [r["class"] for r in rows]
    p = [r["precision"] for r in rows]
    r_ = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    y = np.arange(len(classes))
    bar_h = 0.27
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(classes) + 2))
    ax.barh(y - bar_h, p, height=bar_h, label="precision")
    ax.barh(y, r_, height=bar_h, label="recall")
    ax.barh(y + bar_h, f1, height=bar_h, label="F1")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.set_title(f"{model_name}: per-class precision / recall / F1")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_both(fig, f"{model_name}_per_class_metrics", plots_dir, report_dir)


def plot_top_misclassified(
    dataset, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    class_names: Sequence[str], model_name: str,
    plots_dir: Path, report_dir: Path, top_k: int = 5,
) -> Dict[str, Path]:
    """
    Top-K most confidently wrong predictions, as a 1xK grid showing the image
    with predicted class + confidence vs true class.

    `dataset` must be an ImageFolder whose .samples[i] is (path, label).
    """
    apply_ieee_style()
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        # Make a placeholder figure so the file always exists.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassifications.", ha="center", va="center")
        return _save_both(fig, f"{model_name}_top_misclassified", plots_dir, report_dir)

    conf_of_pred = y_prob[wrong, y_pred[wrong]]
    order = np.argsort(-conf_of_pred)
    pick = wrong[order[:top_k]]

    k = len(pick)
    fig, axes = plt.subplots(1, k, figsize=(3.2 * k, 3.6))
    if k == 1:
        axes = [axes]
    for ax, idx in zip(axes, pick):
        path, _ = dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        true_cls = class_names[y_true[idx]]
        pred_cls = class_names[y_pred[idx]]
        conf = y_prob[idx, y_pred[idx]]
        ax.set_title(
            f"pred: {pred_cls}\n({conf*100:.1f}%)\ntrue: {true_cls}",
            fontsize=9,
        )
    fig.suptitle(f"{model_name}: top-{top_k} most-confident misclassifications", y=1.02)
    fig.tight_layout()
    return _save_both(fig, f"{model_name}_top_misclassified", plots_dir, report_dir)


def plot_model_comparison(
    metrics_by_model: Dict[str, Dict],
    plots_dir: Path,
    report_dir: Path,
    fig_name: str = "model_comparison",
) -> Dict[str, Path]:
    """
    Grouped bar chart comparing top-line metrics across models.
    `metrics_by_model` is {model_name: metrics_dict_from_compute_metrics()}.
    """
    apply_ieee_style()
    models = list(metrics_by_model.keys())
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    x = np.arange(len(keys))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(models):
        vals = [metrics_by_model[m][k] for k in keys]
        bars = ax.bar(x + i * width, vals, width=width, label=m)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([k.replace("_", " ") for k in keys], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on test set")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_both(fig, fig_name, plots_dir, report_dir)


# ---------------------------------------------------------------------------
# Convenience: run full evaluation and save everything
# ---------------------------------------------------------------------------
def evaluate_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    model_name: str,
    plots_dir: Path,
    report_dir: Path,
    metrics_dir: Path,
    history: Optional[Dict[str, List[float]]] = None,
) -> Dict:
    """
    Run full evaluation pipeline on test_loader. Saves:
        - per-class metrics & classification report
        - training curves (if `history` given)
        - confusion matrix (raw + normalised)
        - per-class metrics chart
        - top-5 misclassified
        - metrics JSON

    Returns the metrics dict.
    """
    y_true, y_pred, y_prob = predict_on_loader(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)

    ensure_dir(metrics_dir)
    write_json(metrics, metrics_dir / f"{model_name}_test_metrics.json")
    save_classification_report(
        y_true, y_pred, class_names,
        metrics_dir / f"{model_name}_classification_report.txt",
    )
    np.savez(
        metrics_dir / f"{model_name}_predictions.npz",
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
    )

    if history is not None and len(history.get("train_loss", [])) > 0:
        plot_training_curves(history, model_name, plots_dir, report_dir)

    plot_confusion_matrix(y_true, y_pred, class_names, model_name,
                          plots_dir, report_dir, normalize=False)
    plot_confusion_matrix(y_true, y_pred, class_names, model_name,
                          plots_dir, report_dir, normalize=True)
    plot_per_class_metrics(metrics, model_name, plots_dir, report_dir)
    plot_top_misclassified(
        test_loader.dataset, y_true, y_pred, y_prob,
        class_names, model_name, plots_dir, report_dir, top_k=5,
    )
    return metrics
