"""Evaluate native Figure-only boxes with OmniDocBench COCODetection.

This is the raw detection track: native Figure categories are selected before
scoring, boxes are never assigned, merged, resized, or otherwise changed using
GT, and the official OmniDocBench mmeval.COCODetection implementation computes
mAP, AP50, and AP75. Ground-truth appendix boundaries are used only to select
the requested document scope.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from . import omni_evaluator as omni
from .native_gt_grouped_evaluator import (
    Config,
    DEFAULT_METHODS,
    METHOD_DISPLAY_NAMES,
    _assign_prediction,
    _build_context,
    _load_predictions,
    _metric_records,
    _page_in_scope,
    _prediction_source,
    _scope_accepts,
)


SCHEMA_VERSION = "dp-p2p.native-figure-only-ap.v1"
FIGURE_AP_DISPLAY_NAMES = {**METHOD_DISPLAY_NAMES, "ours": "paper2assets"}

# These are method-native semantic branches, not labels inferred from GT.
FIGURE_SOURCE_TYPES = {
    "ours": frozenset({"figure"}),
    "researchstudio-pdf-agent": frozenset(
        {
            "researchstudio-pdf-agent-verified-crop",
            "researchstudio-pdf-agent-page-render-recovery",
        }
    ),
    "posterforest": frozenset({"posterforest-docling-group"}),
    "postergen": frozenset(
        {
            "postergen-marker-image",
            "postergen-marker-figure",
        }
    ),
    "paper2poster": frozenset({"paper2poster-docling-picture"}),
    "p2p-doclayout": frozenset({"p2p-doclayout-figure"}),
    "postermeld": frozenset(
        {
            "mineru-image",
            "mineru-chart",
            "mineru-figure",
            "postermeld-mineru-pdf_bbox_render",
        }
    ),
}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _figure_types(method: str) -> frozenset[str]:
    try:
        return FIGURE_SOURCE_TYPES[method]
    except KeyError as error:
        raise ValueError(f"no native Figure type policy for {method}") from error


def evaluate_method(config: Config, context: Any, method: str) -> dict[str, Any]:
    output_dir = config.output_root / method
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_types = _figure_types(method)
    boxes_by_page: dict[
        tuple[int, int], list[tuple[float, float, float, float]]
    ] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    source_type_counts: Counter[str] = Counter()
    source_prediction_count = 0
    figure_prediction_count = 0
    excluded_non_figure_count = 0
    scored_prediction_count = 0
    scored_associated_count = 0
    scored_unassociated_count = 0
    ignored_out_of_scope_associated_count = 0
    ignored_out_of_scope_unassociated_count = 0

    for paper_index, paper_dir in enumerate(context.papers):
        predictions = _load_predictions(
            config,
            method,
            paper_dir.name,
            context.page_counts[paper_index],
        )
        targets = context.ownership_ground_truth_by_paper[paper_index]
        appendix_start = context.appendix_start_pages[paper_index]

        for prediction in predictions:
            source_prediction_count += 1
            source_type = prediction.source_type or "<untyped>"
            source_type_counts[source_type] += 1
            base_row = {
                "paper_index": paper_index,
                "paper_title": paper_dir.name,
                "prediction_id": prediction.prediction_id,
                "source_type": source_type,
                "page": prediction.page,
                "bbox_x0": prediction.bbox[0],
                "bbox_y0": prediction.bbox[1],
                "bbox_x1": prediction.bbox[2],
                "bbox_y1": prediction.bbox[3],
            }

            if source_type not in allowed_types:
                excluded_non_figure_count += 1
                rows.append(
                    {
                        **base_row,
                        "matched_scope_for_filtering": "",
                        "status": "excluded_native_non_figure_type",
                    }
                )
                continue

            figure_prediction_count += 1
            assignment = _assign_prediction(prediction, targets)
            if assignment is not None:
                target = targets[assignment.gt_index]
                target_scope = "appendix" if target.in_appendix else "body"
                if not _scope_accepts(target.in_appendix, config.scope):
                    ignored_out_of_scope_associated_count += 1
                    rows.append(
                        {
                            **base_row,
                            "matched_scope_for_filtering": target_scope,
                            "status": "ignored_out_of_scope_associated",
                        }
                    )
                    continue
                status = "scored_associated"
                scored_associated_count += 1
                matched_scope = target_scope
            else:
                if not _page_in_scope(
                    prediction.page,
                    config.scope,
                    appendix_start,
                ):
                    ignored_out_of_scope_unassociated_count += 1
                    rows.append(
                        {
                            **base_row,
                            "matched_scope_for_filtering": "",
                            "status": "ignored_out_of_scope_unassociated",
                        }
                    )
                    continue
                status = "scored_unassociated"
                scored_unassociated_count += 1
                matched_scope = config.scope

            # Raw protocol: preserve the native box exactly. No GT grouping or
            # matching result is passed to the metric.
            boxes_by_page[(paper_index, prediction.page)].append(prediction.bbox)
            scored_prediction_count += 1
            rows.append(
                {
                    **base_row,
                    "matched_scope_for_filtering": matched_scope,
                    "status": status,
                }
            )

    metric_predictions, omni_results = _metric_records(context, boxes_by_page)
    metrics = omni._official_coco_metric(
        metric_predictions,
        context.metric_ground_truths,
    )
    source_method = _prediction_source(method)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_display_name": FIGURE_AP_DISPLAY_NAMES.get(method, method),
        "prediction_source_method": source_method,
        "protocol": {
            "source": "OmniDocBench-v1.5-layout-detection",
            "metric_implementation": "mmeval.COCODetection",
            "category": "figure",
            "native_figure_source_types": sorted(allowed_types),
            "bbox_transform": "none",
            "gt_grouping": False,
            "gt_matching_for_scoring": False,
            "scope_filter": (
                "body/appendix selection only; appendix boundary and same-page "
                "cross-boundary ownership come from GT"
            ),
            "unassociated_in_scope_prediction": "scored as a normal prediction",
            "prediction_score_policy": "constant-1.0-no-native-confidence",
        },
        "config": {
            "scope": config.scope,
            "gt_root": str(config.gt_root.resolve()),
            "dataset_root": str(config.dataset_root.resolve()),
            "prediction_root": str(config.prediction_root.resolve()),
        },
        "dataset": {
            "paper_count": len(context.papers),
            "page_count": context.page_count,
            "gt_figure_count": context.gt_count,
        },
        "predictions": {
            "source_prediction_count": source_prediction_count,
            "source_type_counts": dict(sorted(source_type_counts.items())),
            "figure_prediction_count_all_scopes": figure_prediction_count,
            "excluded_native_non_figure_count": excluded_non_figure_count,
            "scored_prediction_count": scored_prediction_count,
            "scored_associated_count": scored_associated_count,
            "scored_unassociated_count": scored_unassociated_count,
            "ignored_out_of_scope_associated_count": (
                ignored_out_of_scope_associated_count
            ),
            "ignored_out_of_scope_unassociated_count": (
                ignored_out_of_scope_unassociated_count
            ),
        },
        "metrics": metrics,
    }
    omni._write_json(output_dir / "summary.json", summary)
    omni._write_json(output_dir / "omni_ground_truth.json", context.omni_ground_truth)
    omni._write_json(
        output_dir / "omni_prediction.json",
        {"results": omni_results, "categories": {"0": "figure"}},
    )
    _write_csv(output_dir / "per_prediction_scope.csv", rows)
    return summary


def _comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    predictions = summary["predictions"]
    return {
        "method": summary["method_display_name"],
        "method_id": summary["method"],
        "prediction_source_method": summary["prediction_source_method"],
        "gt_figure_count": summary["dataset"]["gt_figure_count"],
        "figure_candidates_all_scopes": predictions[
            "figure_prediction_count_all_scopes"
        ],
        "scored_body_candidates": predictions["scored_prediction_count"],
        "excluded_non_figure": predictions["excluded_native_non_figure_count"],
        "mAP": float(metrics["bbox_mAP"]) * 100.0,
        "AP50": float(metrics["bbox_mAP_50"]) * 100.0,
        "AP75": float(metrics["bbox_mAP_75"]) * 100.0,
    }


def _write_root_outputs(
    output_root: Path,
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [_comparison_row(summary) for summary in summaries]
    rows.sort(key=lambda row: row["mAP"], reverse=True)
    omni._write_json(output_root / "comparison.json", rows)
    _write_csv(output_root / "comparison.csv", rows)
    lines = [
        "# Figure-Only Native Bbox AP (Main Text)",
        "",
        "| Method | mAP | AP50 | AP75 |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {row['method']} | {row['mAP']:.2f} | "
            f"{row['AP50']:.2f} | {row['AP75']:.2f} |"
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "Uses each method's native Figure branch only, without GT-grouped aggregation or bbox transformation.",
            "",
        ]
    )
    (output_root / "figure_only_ap_table.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("results/figure-caption-pairs"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("results/exp/native_bbox"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp/native_figure_only_ap_body"),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=sorted(METHOD_DISPLAY_NAMES),
        help="method to evaluate; repeat for multiple methods (default: all poster methods)",
    )
    parser.add_argument(
        "--scope",
        choices=["body", "appendix", "all"],
        default="body",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config(
        gt_root=args.gt_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_root=args.output_root,
        scope=args.scope,
    )
    config.validate()
    config.output_root.mkdir(parents=True, exist_ok=True)
    context = _build_context(config)
    methods = args.method or list(DEFAULT_METHODS)
    summaries = []
    for index, method in enumerate(methods, start=1):
        print(f"[{index}/{len(methods)}] evaluating {method}", flush=True)
        summary = evaluate_method(config, context, method)
        summaries.append(summary)
        metrics = summary["metrics"]
        print(
            f"  mAP={float(metrics['bbox_mAP']) * 100.0:.2f} "
            f"AP50={float(metrics['bbox_mAP_50']) * 100.0:.2f} "
            f"AP75={float(metrics['bbox_mAP_75']) * 100.0:.2f}",
            flush=True,
        )
    rows = _write_root_outputs(config.output_root, summaries)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

