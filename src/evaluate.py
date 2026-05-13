"""
src/evaluate.py
===============
Comprehensive evaluation utilities:
  - per-class precision / recall / F1
  - confusion matrix heatmap
  - training vs validation curves
  - top-5 misclassified samples grid
  - model comparison table

All plots are saved to results/plots/ and returned for notebook display.
"""

import os, json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import PLOTS_DIR, METRICS_DIR, CLASS_NAMES


os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  Inference helper
# ─────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Runs the model on all batches in loader.

    Returns
    -------
    y_true  : np.ndarray  (N,)
    y_pred  : np.ndarray  (N,)
    images  : list of tensors  (for misclassification display)
    """
    model.eval()
    all_preds, all_labels, all_images = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
            all_images.extend(imgs.cpu())

    return (
        np.concatenate([l.numpy() for l in all_labels]),
        np.concatenate([p.numpy() for p in all_preds]),
        all_images,
    )


# ─────────────────────────────────────────────
# 2.  Classification report
# ─────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "model",
    save: bool = True,
) -> Dict:
    """
    Prints and saves the full classification report.
    Returns a dict with accuracy, per-class metrics, and raw arrays.
    """
    y_true, y_pred, images = run_inference(model, loader, device)
    acc = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    print(f"\n── Classification Report [{model_name}] ──────────────────")
    print(report_str)
    print(f"  Overall Accuracy: {acc:.4f}")

    if save:
        path = os.path.join(METRICS_DIR, f"{model_name}_report.json")
        with open(path, "w") as f:
            json.dump({"accuracy": acc, "report": report_dict}, f, indent=2)
        print(f"  Report saved → {path}")

    return {
        "accuracy": acc,
        "report":   report_dict,
        "y_true":   y_true,
        "y_pred":   y_pred,
        "images":   images,
    }


# ─────────────────────────────────────────────
# 3.  Confusion matrix heatmap
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "model",
    normalize: bool = True,
    save: bool = True,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", cm.max()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
        fig.savefig(path, dpi=150)
        print(f"  Confusion matrix saved → {path}")
    return fig


# ─────────────────────────────────────────────
# 4.  Training / validation curves
# ─────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    model_name: str = "model",
    save: bool = True,
) -> plt.Figure:
    """
    Plots loss and accuracy curves from a history dict
    (keys: train_loss, val_loss, train_acc, val_acc).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", ms=3)
    axes[0].set_title(f"Loss Curves — {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="o", ms=3)
    axes[1].set_title(f"Accuracy Curves — {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Training History — {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_training_curves.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Training curves saved → {path}")
    return fig


# ─────────────────────────────────────────────
# 5.  Top-5 misclassified samples
# ─────────────────────────────────────────────

def plot_misclassified(
    images: List,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    model_name: str = "model",
    n: int = 5,
    imagenet_mean = (0.485, 0.456, 0.406),
    imagenet_std  = (0.229, 0.224, 0.225),
    save: bool = True,
) -> plt.Figure:
    """Plots a grid of the first `n` misclassified samples."""
    wrong_idx = np.where(y_true != y_pred)[0]
    n = min(n, len(wrong_idx))

    if n == 0:
        print("  No misclassifications found — perfect score!")
        return None

    mean = np.array(imagenet_mean)[:, None, None]
    std  = np.array(imagenet_std)[:, None, None]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, wrong_idx[:n]):
        img = images[idx].numpy()          # (3, H, W)
        img = img * std + mean             # de-normalise
        img = np.clip(img.transpose(1, 2, 0), 0, 1)  # (H, W, 3)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            fontsize=9, color="red",
        )

    plt.suptitle(f"Top-{n} Misclassifications — {model_name}", fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"{model_name}_misclassified.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Misclassified grid saved → {path}")
    return fig


# ─────────────────────────────────────────────
# 6.  Model comparison table
# ─────────────────────────────────────────────

def plot_model_comparison(
    results: Dict[str, Dict],
    save: bool = True,
) -> plt.Figure:
    """
    results = {
        "ResNet50":        {"accuracy": 0.92, "macro_f1": 0.91},
        "EfficientNet-B0": {"accuracy": 0.94, "macro_f1": 0.93},
    }
    """
    model_names = list(results.keys())
    accs  = [results[m]["accuracy"]  for m in model_names]
    f1s   = [results[m]["macro_f1"]  for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1s,  width, label="Macro F1", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Accuracy vs Macro F1", fontsize=13)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "model_comparison.png")
        fig.savefig(path, dpi=150)
        print(f"  Comparison chart saved → {path}")
    return fig
