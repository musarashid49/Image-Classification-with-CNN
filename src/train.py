"""
Trainer with crash-resilient per-epoch Drive saves and resume support.

Design decisions (from handoff items 3, 4, 8):
  * Every epoch, append a row to <model>_epoch_log.csv on Drive.
  * Every epoch, overwrite <model>_history.json on Drive.
  * On every val_acc improvement, save full training state to BOTH
    local Colab disk AND Drive (immediately, not at end).
  * Save a `_final.pth` at end of run.
  * Resume from any checkpoint via Trainer(resume_from=...).

A "full training state" checkpoint contains:
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - epoch (0-indexed, last completed)
  - best_val_acc
  - history dict
  - model_name, training_config (for self-describing files)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.utils import (
    append_csv_row,
    copy_to_drive,
    ensure_dir,
    format_metrics,
    write_json,
)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str = "model"
    epochs: int = 30
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 7
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip: float = 1.0
    epoch_save_every: int = 1   # save history/csv this often
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: TrainConfig,
        param_groups: Optional[List[Dict]] = None,
        local_results_dir: Path | str = "/content/results",
        drive_results_dir: Path | str = "/content/drive/MyDrive/pk_politicians_results",
        resume_from: Optional[Path | str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

        # Optimizer (param_groups give per-group LR)
        if param_groups is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr_head,
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(param_groups)

        # Scheduler: ReduceLROnPlateau on val_acc (max mode).
        # NOTE: no `verbose=` argument — removed in modern torch (handoff #6).
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
        )

        # Paths
        self.local_results_dir = ensure_dir(local_results_dir)
        self.local_ckpt_dir = ensure_dir(self.local_results_dir / "checkpoints")
        self.local_logs_dir = ensure_dir(self.local_results_dir / "logs")

        self.drive_results_dir = Path(drive_results_dir)
        self.drive_ckpt_dir = ensure_dir(self.drive_results_dir / "checkpoints")
        self.drive_logs_dir = ensure_dir(self.drive_results_dir / "logs")

        self.best_path_local = self.local_ckpt_dir / f"{cfg.model_name}_best.pth"
        self.best_path_drive = self.drive_ckpt_dir / f"{cfg.model_name}_best.pth"
        self.final_path_local = self.local_ckpt_dir / f"{cfg.model_name}_final.pth"
        self.final_path_drive = self.drive_ckpt_dir / f"{cfg.model_name}_final.pth"
        self.history_path_local = self.local_logs_dir / f"{cfg.model_name}_history.json"
        self.history_path_drive = self.drive_logs_dir / f"{cfg.model_name}_history.json"
        self.epoch_csv_local = self.local_logs_dir / f"{cfg.model_name}_epoch_log.csv"
        self.epoch_csv_drive = self.drive_logs_dir / f"{cfg.model_name}_epoch_log.csv"

        # State
        self.start_epoch = 0
        self.best_val_acc = -float("inf")
        self.epochs_since_improve = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr_head": [], "lr_backbone": [],
        }

        if resume_from is not None:
            self._load_state(resume_from)

    # ---------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        t0 = time.time()
        print(f"[trainer] starting {self.cfg.model_name} on {self.device}")
        if self.start_epoch > 0:
            print(f"[trainer] resuming from epoch {self.start_epoch} "
                  f"(best_val_acc={self.best_val_acc:.4f})")

        if self.start_epoch >= self.cfg.epochs:
            print(f"[trainer] start_epoch ({self.start_epoch}) >= epochs "
                  f"({self.cfg.epochs}); nothing to train.")
            return {
                "best_val_acc": self.best_val_acc,
                "history": self.history,
                "duration_seconds": 0.0,
                "epochs_completed": self.start_epoch,
            }

        epoch = self.start_epoch
        val_acc = self.best_val_acc
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t_epoch = time.time()
            train_loss, train_acc = self._run_one_epoch(epoch, train=True)
            val_loss, val_acc = self._run_one_epoch(epoch, train=False)

            self.scheduler.step(val_acc)

            lr_bb = self.optimizer.param_groups[0]["lr"]
            lr_hd = self.optimizer.param_groups[-1]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr_head"].append(lr_hd)
            self.history["lr_backbone"].append(lr_bb)

            epoch_time = time.time() - t_epoch
            metrics_row = {
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "lr_head": lr_hd, "lr_backbone": lr_bb,
                "epoch_seconds": round(epoch_time, 2),
            }
            print(f"[epoch {epoch+1}/{self.cfg.epochs}] "
                  f"{format_metrics(metrics_row)}")

            # ---- Persist every epoch (this is the crash-resilience layer) ----
            append_csv_row(self.epoch_csv_local, metrics_row)
            append_csv_row(self.epoch_csv_drive, metrics_row)
            write_json(self.history, self.history_path_local)
            write_json(self.history, self.history_path_drive)

            # ---- Save best checkpoint when val improves ----
            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.epochs_since_improve = 0
                self._save_state(self.best_path_local, epoch, val_acc)
                copy_to_drive(self.best_path_local, self.best_path_drive)
                print(f"  ↑ new best val_acc={val_acc:.4f} — "
                      f"saved to Drive: {self.best_path_drive.name}")
            else:
                self.epochs_since_improve += 1
                print(f"  no improvement (best={self.best_val_acc:.4f}, "
                      f"patience {self.epochs_since_improve}/"
                      f"{self.cfg.early_stopping_patience})")
                if self.epochs_since_improve >= self.cfg.early_stopping_patience:
                    print(f"[trainer] early stopping at epoch {epoch+1}")
                    break

        # ---- End of training: save _final ----
        self._save_state(self.final_path_local, epoch, val_acc)
        copy_to_drive(self.final_path_local, self.final_path_drive)
        write_json(self.history, self.history_path_local)
        write_json(self.history, self.history_path_drive)

        duration = time.time() - t0
        print(f"[trainer] done. best_val_acc={self.best_val_acc:.4f} "
              f"in {duration:.1f}s")
        return {
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "duration_seconds": duration,
            "epochs_completed": epoch + 1,
        }

    # ---------------------------------------------------------------------
    # One epoch
    # ---------------------------------------------------------------------
    def _run_one_epoch(self, epoch: int, train: bool):
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        desc = f"{'train' if train else 'val  '} {epoch+1}"
        pbar = tqdm(loader, desc=desc, leave=False)
        torch.set_grad_enabled(train)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            if train:
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_n += labels.size(0)
                total_loss += loss.item() * labels.size(0)

            pbar.set_postfix(
                loss=total_loss / max(total_n, 1),
                acc=total_correct / max(total_n, 1),
            )

        torch.set_grad_enabled(True)
        return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

    # ---------------------------------------------------------------------
    # Checkpoint I/O — FULL STATE
    # ---------------------------------------------------------------------
    def _save_state(self, path: Path, epoch: int, val_acc: float) -> None:
        ensure_dir(Path(path).parent)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "model_name": self.cfg.model_name,
            "training_config": asdict(self.cfg),
        }, path)

    def _load_state(self, path: Path | str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt["best_val_acc"]
        self.history = ckpt.get("history", self.history)
        print(f"[trainer] loaded checkpoint from {path} — "
              f"resuming at epoch {self.start_epoch}")


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def load_model_for_inference(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device,
) -> nn.Module:
    """Load a checkpoint's model_state_dict into `model` and put on device."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
