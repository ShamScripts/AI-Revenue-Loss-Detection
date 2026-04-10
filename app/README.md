# Fraud Analytics Dashboard

## Run

From the project root that contains `app/` and `processed_data/`:

```bash
python -m streamlit run app/app.py
```

(Use `python -m streamlit` on Windows if `streamlit` alone is not found.)

## Layout

```
app/
├── app.py                 # Entry: Home + sidebar + global CSS
├── assets/custom.css      # Premium theme overrides
├── components/
│   ├── figure_captions.py # Titles + one-line text for EDA/plot PNGs
│   ├── cards.py           # KPI cards, badges, panels
│   ├── charts.py          # Plotly helpers (optional)
│   ├── file_utils.py      # Paths, safe CSV/JSON/text, artifact discovery
│   ├── overview_content.py# Shared landing content
│   ├── stage_utils.py     # Artifact presence / counts
│   ├── styling.py         # CSS inject, hero, section titles
│   └── tables.py          # DataFrame previews
└── pages/
    ├── 1_Overview.py
    ├── 2_Dataset_and_Preprocessing.py
    ├── 3_EDA.py
    ├── 4_GBDT_Baselines.py
    ├── 5_Deep_Anomaly.py
    ├── 6_Fusion_and_Final_Results.py
    ├── 7_Reports_and_Documents.py
    └── 8_Run_Pipeline.py
```

## Page → pipeline mapping

| Page | Stage | Key files |
|------|-------|-----------|
| 2 | 1 | `ieee_train_eda_ready.csv`, `preprocessing_config.json`, `ieee_missing_top20_summary.csv` |
| 3 | 1 (EDA plots) | `reports/figures/*.png` |
| 4 | 2 | `gbdt_preds.csv` |
| 5 | 3 | `hybrid_dnn_anomaly_preds.csv` |
| 6 | 4 | `final_hybrid_comparison_metrics.csv`, `final_hybrid_scores.csv`, `final_hybrid_threshold.txt` |

## Extending

- Add new artifact names in `components/file_utils.py` → `ARTIFACTS`.
- Add plots: drop PNGs into `reports/figures/` — they appear on **EDA** automatically.
- For new metrics CSVs, add a section in `pages/6_Fusion_and_Final_Results.py` with `safe_read_csv`.
- Keep `st.set_page_config` only in `app.py` (Streamlit requirement).

## Troubleshooting

- **Wrong paths:** Run Streamlit from the directory that contains both `app/` and `DATASET_ieee-cis-elliptic/`.
- **Import errors:** Ensure `pip install streamlit plotly` (or full `requirements.txt`).
