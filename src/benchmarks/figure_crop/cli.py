"""Command line interface for the full-paper Figure crop benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import EvaluationConfig, evaluate_figure_crops


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gt-root", type=Path, default=Path("results/figure-caption-pairs"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--method", default=None)
    parser.add_argument("--scope", choices=["body", "appendix", "all"], default="body")
    parser.add_argument("--prediction-json", default="assets/meta/figures.json")
    parser.add_argument(
        "--prediction-space",
        choices=["pdf", "normalized1000", "page_pixels", "auto"],
        default="pdf",
    )
    parser.add_argument("--prediction-page-base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--detection-iou", type=float, default=0.5)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = EvaluationConfig(
        gt_root=args.gt_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        method=args.method or args.prediction_root.name,
        scope=args.scope,
        prediction_json=args.prediction_json,
        prediction_space=args.prediction_space,
        prediction_page_base=args.prediction_page_base,
        detection_iou=args.detection_iou,
        coverage_threshold=args.coverage_threshold,
        contamination_threshold=args.contamination_threshold,
    )
    summary = evaluate_figure_crops(config)
    print(
        json.dumps(
            {
                "method": summary["method"],
                "dataset": summary["dataset"],
                "metrics": summary["metrics"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
