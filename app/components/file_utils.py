"""Path resolution and safe file loading for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def get_project_root() -> Path:
    """Locate ML project root (contains ``DATASET_ieee-cis-elliptic``). ``processed_data`` may be empty until stage 1."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "DATASET_ieee-cis-elliptic").is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "DATASET_ieee-cis-elliptic").is_dir():
        return cwd
    # Running from app/ only
    for p in cwd.parents:
        if (p / "DATASET_ieee-cis-elliptic").is_dir():
            return p
    return cwd


def processed_dir() -> Path:
    return get_project_root() / "processed_data"


def reports_figures_dir() -> Path:
    return get_project_root() / "reports" / "figures"


def reports_dir() -> Path:
    return get_project_root() / "reports"


def report_docs_dir() -> Path:
    return get_project_root() / "REPORT"


def lit_review_dir() -> Path:
    return get_project_root() / "Lit_Review"


def ref_dir() -> Path:
    return get_project_root() / "Ref"


def main_py_path() -> Path:
    return get_project_root() / "main.py"


def safe_read_csv(path: Path, nrows: int | None = 5000, **kwargs: Any) -> pd.DataFrame | None:
    try:
        if not path.is_file():
            return None
        return pd.read_csv(path, nrows=nrows, low_memory=False, **kwargs)
    except Exception:
        return None


def safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_text(path: Path, max_chars: int = 50_000) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    except Exception:
        return None


def list_images_recursive(root: Path, extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        for ext in extensions:
            out.extend(root.rglob(f"*{ext}"))
        return sorted(set(out))
    except Exception:
        return []


def list_pdfs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(root.rglob("*.pdf"))
    except Exception:
        return []


def list_csvs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    try:
        return sorted(root.glob("*.csv"))
    except Exception:
        return []


# Expected artifact names (pipeline outputs)
ARTIFACTS = {
    "stage1": [
        "ieee_train_merged_cleaned.csv",
        "ieee_train_eda_ready.csv",
        "elliptic_transactions_cleaned.csv",
        "preprocessing_config.json",
        "ieee_missing_top20_summary.csv",
    ],
    "stage2": ["gbdt_preds.csv"],
    "stage3": ["hybrid_dnn_anomaly_preds.csv"],
    "stage4": [
        "final_hybrid_comparison_metrics.csv",
        "final_hybrid_scores.csv",
        "final_hybrid_threshold.txt",
    ],
}
