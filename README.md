# Hybrid Fraud Detection — IEEE-CIS & Elliptic

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end **machine learning pipeline** and **Streamlit dashboard** for fraud detection: **gradient boosting**, **attention / plain neural nets**, **isolation-forest anomaly scores**, **weighted fusion** on **IEEE-CIS** transactions, plus optional **graph convolution (GCN)** and **tabular baselines** on the **Elliptic** Bitcoin dataset.

**Course / submission use:** clone → add data locally → run `main.py` → inspect `processed_data/` and the dashboard. Raw data and generated artifacts are **not** committed to Git (see `.gitignore`).

---

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Pipeline (stages 1–5)](#pipeline-stages-15)
- [Command-line options](#command-line-options)
- [Dashboard](#dashboard)
- [Outputs & report tables](#outputs--report-tables)
- [Methodology notes](#methodology-notes)
- [Pushing to GitHub](#pushing-to-github)
- [License](#license)

---

## Features

| Area | What you get |
|------|----------------|
| **IEEE-CIS** | Preprocessing, EDA, LR/RF/DT + **LightGBM/XGBoost**, optional **SMOTE**, **SHAP**, **hyperparameter search** |
| **Deep learning** | **Attention DNN** (TensorFlow) + **plain MLP** baseline; **Isolation Forest** anomaly channel |
| **Fusion** | Normalized scores, weighted fusion, threshold tuning, **report CSVs** for papers |
| **Elliptic** | Optional **2-layer GCN** (PyTorch), **FraudGT-style MLP**, LR/RF, **temporal** split |
| **UI** | Multi-page **Streamlit** app over `processed_data/` artifacts |

---

## Repository layout

```
├── main.py                 # CLI entry: run pipeline stages
├── requirements.txt        # Full dependency stack (TF, torch, dashboard, …)
├── pyproject.toml          # Package metadata; pip install -e .
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # Short notes for contributors
├── docs/
│   └── SUBMISSION_CHECKLIST.md   # pre-push checklist for courses
├── src/fraud_ml/           # Core package (stages, reporting, config)
├── app/                    # Streamlit multipage dashboard
├── manuscript/           # Markdown / LaTeX report sources (tracked)
├── DATASET_ieee-cis-elliptic/   # Place Kaggle data here (gitignored — see README inside)
├── processed_data/         # Generated CSVs/JSON (gitignored)
└── figures/                # PNG figures from pipeline (gitignored)
```

**Naming:** **`manuscript/`** = your written report (Markdown/LaTeX). **`figures/`** = auto-generated pipeline plots. No more `REPORT/` vs `reports/` overlap.

---

## Requirements

| | |
|--|--|
| **Python** | **3.10.x – 3.12.x** (recommended: **3.11**) |
| **Avoid** | **3.13+** for full TensorFlow wheels |
| **Disk** | Enough space for IEEE + Elliptic CSVs and processed outputs |
| **GPU** | Not required (CPU training is supported) |

---

## Quick start

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git
cd YOUR_REPO

python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the full pipeline (stages **1–5**; Stage 5 needs **PyTorch** for GCN metrics):

```bash
python main.py
```

IEEE-only through fusion (skip Elliptic graph stage):

```bash
python main.py --skip-elliptic-graph
```

**GCN:** install PyTorch, then `python main.py --stage 5` (or full `main.py`).

---

## Datasets

Download and place files under **`DATASET_ieee-cis-elliptic/`** as described in:

**[`DATASET_ieee-cis-elliptic/README.md`](DATASET_ieee-cis-elliptic/README.md)**

- **IEEE-CIS Fraud Detection** — [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)  
- **Elliptic Data Set** — [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

This directory is **gitignored**; reviewers clone the repo and add data locally.

---

## Pipeline (stages 1–5)

| Stage | Module | Main outputs (`processed_data/`) |
|-------|--------|----------------------------------|
| 1 | `stage01_data` | Cleaned IEEE/Elliptic CSVs, `preprocessing_config.json` |
| 2 | `stage02_gbdt` | `gbdt_preds.csv`, optional SHAP figure, experiment config |
| 3 | `stage03_deep_anomaly` | `hybrid_dnn_anomaly_preds.csv`, DNN baseline metrics |
| 4 | `stage04_fusion` | `final_hybrid_*.csv`, threshold, `report_table_*.csv` |
| 5 | `stage05_elliptic_graph` | `elliptic_graph_experiments.csv` (GCN + tabular baselines) |

Figures go to **`figures/`** when plotting is enabled.

---

## Command-line options

```bash
python main.py --help
```

| Flag | Purpose |
|------|---------|
| `--stage N` | Run only stage `1`–`5` |
| `--no-plots` | Skip saving figures |
| `--skip-elliptic-graph` | Run stages 1–4 only |
| `--split temporal` | Time-ordered splits where applicable (IEEE `TransactionDT`, fusion) |
| `--smote` | SMOTE on IEEE training matrix (stage 2) |
| `--tune-gbdt` | Randomized search on LightGBM (stage 2) |

Module equivalent: `python -m fraud_ml.pipeline.run_all` (same flags via `run_all`).

---

## Dashboard

```bash
python -m streamlit run app/app.py
```

| Page | Role |
|------|------|
| Home / Overview | KPIs, timeline |
| 2–3 | Data, EDA |
| 4–6 | GBDT, Deep/Anomaly, Fusion + **report tables** |
| 7 | Documents |
| 8 | Run pipeline (subprocess) |
| 9 | Elliptic graph results |

On Windows, `run_dashboard.bat` runs Streamlit from the project folder.

---

## Outputs & report tables

After Stage 4 (and optional Stage 5):

- **`final_hybrid_comparison_metrics.csv`**, **`final_hybrid_scores.csv`**, **`final_hybrid_threshold.txt`**
- **`report_table_1_ieee_cis.csv`**, **`report_table_2_elliptic.csv`**, **`report_table_3_ablation.csv`**
- **`elliptic_graph_experiments.csv`** (Stage 5)

Additional write-ups live under **`manuscript/`** (Markdown / LaTeX).

---

## Methodology notes

Optional experiments (SMOTE, temporal split, SHAP, tuning, plain MLP vs attention, GCN) are implemented in code; enable them with the CLI flags above. The **fusion** step still uses the **attention DNN** branch for IEEE scores unless you change `stage04`. **FraudGT-style** and **GNN** rows refer to implemented baselines (deep MLP / 2-layer GCN), not third-party proprietary systems.

| Report claim | Code / flag |
|--------------|-------------|
| GNN on Elliptic | Stage 5 GCN (`pip install torch`) |
| DNN without attention | Stage 3 plain MLP + `stage03_ieee_dnn_baselines.csv` |
| FraudGT-style baseline | Stage 5 deep MLP on Elliptic |
| SMOTE | `--smote` |
| SHAP | Stage 2 → `figures/stage02_shap_summary.png` |
| Temporal split | `--split temporal` |
| GBDT tuning | `--tune-gbdt` |

---

## Pushing to GitHub

1. **Verify** nothing huge is tracked:

   ```bash
   git status
   git ls-files | findstr /I "\.csv"   # Windows; should NOT list dataset/processed paths if ignored
   ```

2. If data was accidentally committed:

   ```bash
   git rm -r --cached DATASET_ieee-cis-elliptic/ processed_data/ figures/ 2>nul
   git commit -m "Stop tracking data and generated outputs"
   ```

3. **First push:**

   ```bash
   git init
   git add .
   git status
   git commit -m "Initial commit: fraud detection pipeline and dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

See **`docs/SUBMISSION_CHECKLIST.md`** before submitting to a course.

---

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).

Dataset files from Kaggle remain subject to their **respective licenses**; this repo does not redistribute them.

---

## Acknowledgments

- **IEEE-CIS Fraud Detection** and **Elliptic** datasets via Kaggle / original publishers.  
- Stack: **scikit-learn**, **LightGBM** / **XGBoost**, **TensorFlow**, **PyTorch** (optional GCN), **Streamlit**, **SHAP**, **imbalanced-learn**.
