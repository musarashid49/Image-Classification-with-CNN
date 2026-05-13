"""
scripts/export_results.py
=========================
Collects all metrics JSONs and plot PNGs from results/ and
assembles a single summary CSV + comparison table printout.

Run after all models are evaluated:
    !python scripts/export_results.py
"""

import os, json, glob
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import METRICS_DIR, PLOTS_DIR, CLASS_NAMES


def load_report(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_summary_table() -> pd.DataFrame:
    report_files = glob.glob(os.path.join(METRICS_DIR, "*_report.json"))
    if not report_files:
        print(f"  No report JSONs found in {METRICS_DIR}")
        return pd.DataFrame()

    rows = []
    for path in sorted(report_files):
        model_name = os.path.basename(path).replace("_report.json", "")
        data = load_report(path)
        acc  = data.get("accuracy", 0.0)
        rep  = data.get("report",   {})

        macro = rep.get("macro avg", {})
        row = {
            "model":     model_name,
            "accuracy":  round(acc, 4),
            "precision": round(macro.get("precision", 0), 4),
            "recall":    round(macro.get("recall",    0), 4),
            "f1_macro":  round(macro.get("f1-score",  0), 4),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    return df


def build_per_class_table() -> pd.DataFrame:
    report_files = glob.glob(os.path.join(METRICS_DIR, "*_report.json"))
    if not report_files:
        return pd.DataFrame()

    rows = []
    for path in sorted(report_files):
        model_name = os.path.basename(path).replace("_report.json", "")
        data = load_report(path)
        rep  = data.get("report", {})
        for cls in CLASS_NAMES:
            cls_data = rep.get(cls, {})
            rows.append({
                "model":     model_name,
                "class":     cls,
                "precision": round(cls_data.get("precision", 0), 4),
                "recall":    round(cls_data.get("recall",    0), 4),
                "f1":        round(cls_data.get("f1-score",  0), 4),
                "support":   cls_data.get("support", 0),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("\n── Model Summary ───────────────────────────────────────────────")
    summary = build_summary_table()
    if not summary.empty:
        print(summary.to_string(index=False))
        out = os.path.join(METRICS_DIR, "summary_table.csv")
        summary.to_csv(out, index=False)
        print(f"\n  Saved → {out}")

    print("\n── Per-Class Metrics ────────────────────────────────────────────")
    per_class = build_per_class_table()
    if not per_class.empty:
        print(per_class.to_string(index=False))
        out2 = os.path.join(METRICS_DIR, "per_class_metrics.csv")
        per_class.to_csv(out2, index=False)
        print(f"\n  Saved → {out2}")

    print("\n── Available Plots ──────────────────────────────────────────────")
    plots = glob.glob(os.path.join(PLOTS_DIR, "*.png"))
    for p in sorted(plots):
        print(f"  {p}")
