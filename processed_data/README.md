# Processed outputs (local — not in Git)

CSV and JSON artifacts are **generated** by the pipeline (`python main.py`).

Examples:

- `ieee_train_eda_ready.csv`, `elliptic_transactions_cleaned.csv`
- `gbdt_preds.csv`, `hybrid_dnn_anomaly_preds.csv`
- `final_hybrid_comparison_metrics.csv`, `final_hybrid_scores.csv`, `final_hybrid_threshold.txt`
- `preprocessing_config.json`

After cloning, run stages 1–4 to populate this directory. The Streamlit dashboard reads these files if present.
