"""Overview / landing content (shared by Home and page 1)."""

from __future__ import annotations

import streamlit as st

from components import file_utils as fu
from components import stage_utils as su
from components.cards import kpi_row
from components.styling import (
    dataset_cards,
    executive_summary,
    hero,
    highlight_list_plain,
    presentation_footer,
    section_title,
    stage_timeline,
)


def render_overview() -> None:
    hero(
        "Hybrid Fraud Detection System",
        "A reproducible pipeline: tabular gradient boosting, neural scoring, anomaly detection, and fusion on IEEE-CIS—"
        "plus an optional Elliptic graph stage (GCN + tabular baselines) aligned with the academic report.",
        badge="Capstone · ML pipeline dashboard",
    )

    ok, total = su.count_artifacts_found()
    pct = int(100 * ok / total) if total else 0
    kpi_row(
        [
            ("Pipeline depth", "5 stages", "IEEE 1–4 · Elliptic graph (5) optional"),
            ("Data sources", "2 datasets", "IEEE-CIS · Elliptic Bitcoin"),
            ("Model families", "6+", "Baselines · GBDT · DNN/MLP · IF · Fusion"),
            ("Artifacts ready", f"{ok} / {total}", f"≈ {pct}% of expected outputs"),
            ("Workspace", fu.get_project_root().name, "Auto-resolved project root"),
        ]
    )

    executive_summary(
        "This project addresses highly imbalanced fraud classification by chaining transparent preprocessing, "
        "strong tree-based baselines, complementary neural and anomaly scores, and a weighted fusion layer with "
        "validation-tuned operating points—suitable for academic review and reproducible benchmarking."
    )

    section_title("Data foundation")
    dataset_cards(
        [
            (
                "Tabular · Primary",
                "IEEE-CIS Fraud Detection",
                "Hundreds of transaction and identity attributes joined on TransactionID. "
                "Severe missingness handled via sparse-column removal and imputation; target isFraud drives supervised learning.",
            ),
            (
                "Graph-enriched",
                "Elliptic Bitcoin Transactions",
                "Local features plus edge-list statistics (degrees, ratios) for licit vs illicit vs unknown flows—"
                "supports graph-aware signals alongside tabular models.",
            ),
        ]
    )

    section_title("Processing timeline")
    stage_timeline(
        [
            ("Ingest & clean", "Merge, drop sparse cols, impute, engineer time/amount + graph features"),
            ("GBDT stack", "Baselines + LightGBM/XGBoost; optional SMOTE, SHAP, tuning; gbdt_preds.csv"),
            ("Deep + anomaly", "Attention DNN + plain MLP baseline + Isolation Forest; hybrid stage-3 score"),
            ("Fusion & eval", "Normalize, weight, tune threshold; final metrics & report tables"),
            ("Elliptic graph", "Optional: GCN on edgelist + FraudGT-style MLP & LR/RF (temporal split)"),
        ]
    )

    section_title("Stage deliverables")
    highlight_list_plain(
        [
            (
                1,
                "Cleaned IEEE and Elliptic tables, EDA figures, preprocessing_config.json, and missing-value audit.",
            ),
            (
                2,
                "gbdt_preds.csv — full-dataset GBDT probabilities aligned for downstream merge.",
            ),
            (
                3,
                "hybrid_dnn_anomaly_preds.csv — DNN/MLP probabilities, IF anomaly channel, and stage-3 hybrid.",
            ),
            (
                4,
                "final_hybrid_*.csv / threshold, report_table_*.csv — fusion + paper-ready tables.",
            ),
            (
                5,
                "elliptic_graph_experiments.csv — GCN (requires PyTorch) + deep tabular baselines on Elliptic.",
            ),
        ]
    )

    with st.expander("Reviewer notes — paths & reproducibility"):
        st.markdown(
            f"""
| Path | Purpose |
|------|---------|
| `{fu.get_project_root()}` | Project root |
| `{fu.processed_dir()}` | CSV + JSON artifacts |
| `{fu.figures_dir()}` | Saved EDA / model figures (`figures/`) |

Run **`python main.py`** from the project root (Python **3.10–3.12**). Dashboard is read-only except **Run Pipeline**.
"""
        )

    presentation_footer(
        "Fraud Analytics Dashboard · Read-only views of saved artifacts · For demos, use full-screen or Streamlit’s print-friendly browser view"
    )
