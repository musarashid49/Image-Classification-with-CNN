"""
src/train.py
============
Training and validation loop with:
  - early stopping
  - ReduceLROnPlateau scheduler
  - per-epoch metric logging
  - best-model checkpoint saving

Usage (from a Colab notebook):
    from src.train import Trainer
    trainer = Trainer(model, loaders, config_overrides={})
    history = trainer.run()
"""

import os, time, copy, json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, EARLY_STOP_PAT,
    CHECKPOINTS_DIR, METRICS_DIR,
)


class EarlyStopping:
    """Stops training when val loss does not improve for `patience` epochs."""
    def __init__(self, patience: int = EARLY_STOP_PAT, delta: float = 1e-4):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Encapsulates the full training loop.

    Parameters
    ----------
    model        : nn.Module
    loaders      : dict  {"train": DataLoader, "val": DataLoader}
    model_name   : str   used for checkpoint filenames
    config_overrides : dict  optional, overrides any config default
    """

    def __init__(
        self,
        model,
        loaders: dict,
        model_name: str = "model",
        config_overrides: dict = None,
    ):
        self.model      = model
        self.loaders    = loaders
        self.model_name = model_name
        cfg = config_overrides or {}

        self.epochs    = cfg.get("num_epochs",    NUM_EPOCHS)
        self.lr        = cfg.get("learning_rate", LEARNING_RATE)
        self.wd        = cfg.get("weight_decay",  WEIGHT_DECAY)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir  = cfg.get("checkpoints_dir", CHECKPOINTS_DIR)
        self.metrics_dir = cfg.get("metrics_dir",   METRICS_DIR)

        os.makedirs(self.ckpt_dir,   exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=self.wd,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=LR_PATIENCE, factor=0.5, verbose=True
        )
        self.early_stop = EarlyStopping(patience=EARLY_STOP_PAT)

        self.model.to(self.device)
        print(f"Training on: {self.device}")

    # ──────────────────────────────────────────────────
    def _run_epoch(self, split: str):
        is_train = (split == "train")
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        running_loss, running_correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for images, labels in tqdm(loader, desc=f"  {split}", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                preds = outputs.argmax(dim=1)
                running_loss    += loss.item() * images.size(0)
                running_correct += (preds == labels).sum().item()
                total           += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = running_correct / total
        return epoch_loss, epoch_acc

    # ──────────────────────────────────────────────────
    def run(self) -> dict:
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        best_val_acc  = 0.0
        best_weights  = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch("train")
            vl_loss, vl_acc = self._run_epoch("val")
            elapsed = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(vl_loss)
            history["val_acc"].append(vl_acc)

            self.scheduler.step(vl_loss)

            print(
                f"Epoch [{epoch:>3}/{self.epochs}]  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
                f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Save best checkpoint
            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_weights = copy.deepcopy(self.model.state_dict())
                ckpt_path = os.path.join(
                    self.ckpt_dir, f"{self.model_name}_best.pth"
                )
                torch.save({
                    "epoch":      epoch,
                    "model_state_dict": best_weights,
                    "val_acc":    best_val_acc,
                    "val_loss":   vl_loss,
                }, ckpt_path)
                print(f"  ✓ Checkpoint saved  (val_acc={best_val_acc:.4f})")

            # Early stopping
            if self.early_stop(vl_loss):
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

        # Restore best weights
        self.model.load_state_dict(best_weights)

        # Persist history to JSON
        hist_path = os.path.join(
            self.metrics_dir, f"{self.model_name}_history.json"
        )
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining complete. Best val_acc = {best_val_acc:.4f}")
        print(f"History saved → {hist_path}")

        return history
