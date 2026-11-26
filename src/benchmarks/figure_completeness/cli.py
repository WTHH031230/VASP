"""CLI for a reproducible pilot/full comparison of frozen Figure bboxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_catalog, read_json
from .runner import RunConfig, run_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]


def default_methods() -> dict[str, Path]:
    marker = read_json(REPO_ROOT / "src/pipeline/asset_extraction/MAINLINE.json")
    ours = REPO_ROOT / marker["latest_evaluation"] / "single_pass-ap" / "ours"
    return {
        "ours": ours,
        "p2p-doclayout": REPO_ROOT / "results/exp/native_figure_only_gt_grouped_ap_body_latest_gt_20260903/p2p-doclayout",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/exp/figure-completeness-ours-p2p-20260908")
    parser.add_argument("--method", action="append", default=[], metavar="NAME=EVAL_DIR",
                        help="Repeatable saved grouped-AP evaluation root; default: current ours + P2P")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="medium", choices=("none", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/render only; zero model calls")
    parser.add_argument("--report-only", action="store_true", help="Refresh reports only; zero model calls")
    args = parser.parse_args(argv)
    try:
        config = RunConfig(model=args.model, effort=args.effort, jobs=args.jobs, timeout=args.timeout,
                           max_attempts=args.max_attempts, dpi=args.dpi, pilot_size=args.pilot_size,
                           seed=args.seed, codex_bin=args.codex_bin)
        config.validate()
        methods = {}
        for value in args.method:
            name, separator, directory = value.partition("=")
            if not separator or not directory or name in methods:
                raise ValueError(f"invalid/duplicate --method: {value}")
            methods[name] = Path(directory).resolve()
        if not methods:
            methods = default_methods()
        catalog = read_catalog(args.dataset_root.resolve(), methods)
        print(json.dumps({"paper_count": catalog["paper_count"], "gt_count": len(catalog["figures"]),
                          "methods": {name: str(path) for name, path in methods.items()}}, ensure_ascii=False), flush=True)
        report = run_experiment(args.output_root.resolve(), catalog, config, phase=args.phase,
                                prepare_only=args.prepare_only, report_only=args.report_only)
        print(json.dumps(report[args.phase], ensure_ascii=False, indent=2), flush=True)
        return 0 if args.prepare_only or args.report_only or report[args.phase]["finished"] else 1
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
