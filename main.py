"""Project entry: run the Python pipeline (stages 1–4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fraud_ml pipeline (stages 1–4).")
    parser.add_argument(
        "--project-dir",
        default=".",
        type=Path,
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only pipeline stage 1–4. Default: run all in order.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving figures under reports/figures.",
    )
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()

    from fraud_ml.pipeline.run_all import run_stages

    return run_stages(
        project_root=project_dir,
        stage=args.stage,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
