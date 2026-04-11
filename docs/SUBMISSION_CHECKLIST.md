# Submission checklist (GitHub)

Use this before sharing the repository with instructors or reviewers.

- [ ] **Python 3.10–3.12** venv; `pip install -r requirements.txt` succeeds.
- [ ] **Data**: `DATASET_ieee-cis-elliptic/` populated locally (folder stays gitignored).
- [ ] **Pipeline**: `python main.py` (or `--skip-elliptic-graph` if omitting Stage 5) completes.
- [ ] **Optional**: `pip install torch` then `python main.py --stage 5` for Elliptic GCN metrics.
- [ ] **Git**: `git status` shows **no** large CSVs, `processed_data/*.csv`, or dataset files staged.
- [ ] **README** at repo root is up to date; **LICENSE** present if required by course.
- [ ] Remove machine-specific paths or secrets (none should be in code).

**Do not push:** `processed_data/` outputs, `figures/*.png`, raw Kaggle CSVs, or personal PDF reports—`.gitignore` should exclude them.
