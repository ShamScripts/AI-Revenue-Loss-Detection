"""Run pipeline stages 1–4 sequentially or a single stage."""

from __future__ import annotations

import argparse
from pathlib import Path


def run_stages(
    project_root: Path | None = None,
    stage: int | None = None,
    save_plots: bool = True,
) -> int:
    stages = [stage] if stage else [1, 2, 3, 4]
    for s in stages:
        if s == 1:
            from fraud_ml.pipeline.stage01_data import run as run1

            run1(project_root=project_root, save_plots=save_plots)
        elif s == 2:
            from fraud_ml.pipeline.stage02_gbdt import run as run2

            run2(project_root=project_root, save_plots=save_plots)
        elif s == 3:
            from fraud_ml.pipeline.stage03_deep_anomaly import run as run3

            run3(project_root=project_root, save_plots=save_plots)
        else:
            from fraud_ml.pipeline.stage04_fusion import run as run4

            run4(project_root=project_root, save_plots=save_plots)

    print("\n[DONE] Pipeline stage(s) completed. Artifacts under processed_data/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fraud_ml pipeline stages.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Project root (folder with DATASET_ieee-cis-elliptic). Default: auto-detect / cwd.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only this stage (1–4). Default: run all in order.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving figures under reports/figures (faster, headless).",
    )
    args = parser.parse_args()
    root = args.project_dir.resolve() if args.project_dir else None
    save_plots = not args.no_plots
    return run_stages(project_root=root, stage=args.stage, save_plots=save_plots)


if __name__ == "__main__":
    raise SystemExit(main())
