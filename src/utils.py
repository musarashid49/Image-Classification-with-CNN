"""
Utilities: reproducibility, Drive sync, model cards.

Model cards live in the GitHub repo at `results/metrics/<model>_model_card.md`
and `.json`. They are the durable record of "which model achieved what".
"""
from __future__ import annotations

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch. Pass deterministic=True for strict but
    slower CUDA reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_copy(src: Path | str, dst: Path | str) -> Path:
    """Copy src -> dst, creating parent dirs. Returns dst."""
    src, dst = Path(src), Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Drive sync (call from notebooks running on Colab)
# ---------------------------------------------------------------------------
def mount_drive(force_remount: bool = False) -> None:
    """Mount Google Drive in Colab. No-op outside Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        print("[mount_drive] not running on Colab — skipping.")
        return
    drive.mount("/content/drive", force_remount=force_remount)


def copy_to_drive(local_path: Path | str, drive_path: Path | str) -> Path:
    """Copy a file from local Colab storage to Drive. Returns the Drive path."""
    return safe_copy(local_path, drive_path)


# ---------------------------------------------------------------------------
# JSON / CSV helpers (used for live epoch logging)
# ---------------------------------------------------------------------------
def write_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def append_csv_row(path: Path | str, row: Dict[str, Any]) -> None:
    """Append one row to a CSV file. Writes a header on first write."""
    import csv
    path = Path(path)
    ensure_dir(path.parent)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Model card (the durable per-model record)
# ---------------------------------------------------------------------------
def save_model_card(
    model_name: str,
    model_info: Dict,
    training_config: Dict,
    final_metrics: Dict,
    training_duration_seconds: float,
    checkpoint_drive_path: str,
    output_dir: Path | str,
    notebook_name: str = "",
    notes: str = "",
) -> Dict[str, Path]:
    """
    Write both <model>_model_card.json and <model>_model_card.md to output_dir.
    `output_dir` should be inside the GitHub repo so cards get committed.
    """
    output_dir = ensure_dir(output_dir)
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    payload = {
        "model_name": model_name,
        "timestamp_utc": timestamp,
        "notebook": notebook_name,
        "model_info": model_info,
        "training_config": training_config,
        "final_metrics": final_metrics,
        "training_duration_seconds": round(training_duration_seconds, 2),
        "training_duration_human": _human_duration(training_duration_seconds),
        "checkpoint_drive_path": checkpoint_drive_path,
        "notes": notes,
    }

    json_path = output_dir / f"{model_name}_model_card.json"
    md_path = output_dir / f"{model_name}_model_card.md"

    write_json(payload, json_path)
    md_path.write_text(_render_model_card_md(payload))

    return {"json": json_path, "md": md_path}


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _render_model_card_md(p: Dict) -> str:
    info = p["model_info"]
    cfg = p["training_config"]
    m = p["final_metrics"]

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    cfg_rows = "\n".join(f"| `{k}` | {_fmt(v)} |" for k, v in cfg.items())
    metric_rows = "\n".join(f"| `{k}` | {_fmt(v)} |" for k, v in m.items())

    return f"""# Model Card — `{p["model_name"]}`

**Timestamp (UTC):** {p["timestamp_utc"]}
**Produced by:** `{p["notebook"]}`
**Training duration:** {p["training_duration_human"]}
**Checkpoint on Drive:** `{p["checkpoint_drive_path"]}`

## Architecture
- Backbone: `{info.get("model_name", "?")}`
- Input size: {info.get("input_size", "?")}×{info.get("input_size", "?")}
- Total parameters: {info.get("total_params", "?"):,} ({info.get("total_params_M", "?")}M)
- Trainable parameters: {info.get("trainable_params", "?"):,}

## Training configuration
| key | value |
|---|---|
{cfg_rows}

## Final test metrics
| metric | value |
|---|---|
{metric_rows}

## Notes
{p["notes"] or "_(none)_"}
"""


# ---------------------------------------------------------------------------
# Experiment log (lightweight, append-only)
# ---------------------------------------------------------------------------
class ExperimentLogger:
    """Append-only JSON-lines logger for run summaries."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def log(self, entry: Dict[str, Any]) -> None:
        entry = {"timestamp_utc": datetime.utcnow().isoformat() + "Z", **entry}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def format_metrics(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in metrics.items())


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_info() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name} ({mem_gb:.1f} GB)"
