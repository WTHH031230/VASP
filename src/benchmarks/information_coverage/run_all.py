"""Run the complete full-100 benchmark pipeline in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun and overwrite all model-generated stages.",
    )
    parser.add_argument(
        "--skip-chinese",
        action="store_true",
        help="Skip Chinese translation/report generation.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    run("prepare_handoff.py")
    common_args = ["--workers", str(args.workers)]
    if args.force:
        common_args.append("--force")
    run("run_reference_extraction.py", *common_args)
    run(
        "run_evaluations.py",
        "--variants",
        "official",
        "rs",
        "whh",
        *common_args,
    )
    run("compare_full100.py")
    if not args.skip_chinese:
        translation_args = ["--workers", str(args.workers)]
        if args.force:
            translation_args.append("--force")
        run("generate_chinese_reports.py", *translation_args)
    run("validate_full100.py")
    run("prepare_handoff.py")
    run("prepare_handoff.py", "--verify-only")
    print("Full benchmark pipeline completed and verified")


if __name__ == "__main__":
    main()
