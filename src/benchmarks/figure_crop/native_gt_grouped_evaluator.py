"""Evaluate native visual boxes with a GT-conditioned Figure crop protocol.

Each method prediction already has a zero-based page index and a top-left
``bbox_1000``. Predictions are treated as an untyped visual candidate pool:
method-provided Figure/Table labels are never used to include or exclude a
box. A prediction is assigned to at most one same-page GT Figure when either
box contains the other's center. Predictions assigned to the same GT are
replaced by their outer union.

The primary crop-quality report is GT-centric. Every GT Figure contributes
one IoU value; an uncovered GT contributes zero, while candidates that cannot
be associated with a Figure are audited but do not affect crop quality. A
separate candidate-audit report describes proposal volume, fragmentation, and
unassociated candidates without calling them false positives: the Figure-only
GT cannot distinguish valid Tables from meaningless visual crops.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import fitz
import numpy as np

from . import omni_evaluator as omni


SCHEMA_VERSION = "dp-p2p.gt-conditioned-figure-crop.v1"
NATIVE_SCHEMA_VERSION = "dp-p2p.native-figure-bbox.v1"
RECALL_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
METHOD_DISPLAY_NAMES = {
    "ours": "Ours (Sol Medium)",
    "postermeld": "PosterMELD (MinerU)",
    "researchstudio-pdf": "ResearchStudio–PDF",
    "researchstudio-pdf-agent": "ResearchStudio (PDF Agent)",
    "p2p-doclayout": "P2P (DocLayout)",
    "paper2poster": "Paper2Poster (Docling)",
    "posterforest": "PosterForest (Docling)",
    "postergen": "PosterGen (Marker)",
}

# PosterMELD preserves MinerU Figure page indices and bboxes. Its parser only
# separates Figure/Table items and re-renders the same source-PDF rectangle, so
# this bbox benchmark shares the controlled Jiatian MinerU extraction.
METHOD_PREDICTION_SOURCES = {"postermeld": "mineru"}

DEFAULT_METHODS = (
    "ours",
    "researchstudio-pdf-agent",
    "posterforest",
    "postergen",
    "paper2poster",
    "p2p-doclayout",
    "postermeld",
)


@dataclass(frozen=True)
class Config:
    gt_root: Path
    dataset_root: Path
    prediction_root: Path
    output_root: Path
    scope: str = "body"

    def validate(self) -> None:
        if self.scope not in {"body", "appendix", "all"}:
            raise ValueError(f"unsupported scope: {self.scope}")
        if not self.gt_root.is_dir():
            raise FileNotFoundError(self.gt_root)
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(self.dataset_root)
        if not self.prediction_root.is_dir():
            raise FileNotFoundError(self.prediction_root)


@dataclass(frozen=True)
class GroundTruth:
    paper_index: int
    paper_title: str
    local_index: int
    figure_id: str
    label: str
    page: int
    bbox: tuple[float, float, float, float]
    in_appendix: bool = False


@dataclass(frozen=True)
class Prediction:
    order: int
    prediction_id: str
    label: str
    source_type: str
    crop: str
    page: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Assignment:
    gt_index: int
    relation: str
    intersection: float
    containment: float
    iou: float


@dataclass(frozen=True)
class DatasetContext:
    papers: tuple[Path, ...]
    page_counts: tuple[int, ...]
    page_metric_ids: dict[tuple[int, int], int]
    ground_truth_by_paper: dict[int, tuple[GroundTruth, ...]]
    ownership_ground_truth_by_paper: dict[int, tuple[GroundTruth, ...]]
    appendix_start_pages: tuple[int | None, ...]
    metric_ground_truths: list[dict[str, Any]]
    omni_ground_truth: list[dict[str, Any]]
    gt_count: int
    page_count: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    omni._write_json(path, value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _as_box(
    value: Any,
    *,
    context: str,
    clip_page_boundary: bool = False,
) -> tuple[float, float, float, float]:
    box = omni._as_box(value)
    if box is None:
        raise ValueError(f"invalid bbox in {context}: {value!r}")
    clipped = omni._clip_box(box)
    if clipped is None:
        raise ValueError(f"bbox outside normalized page in {context}: {value!r}")
    if (
        not clip_page_boundary
        and any(abs(left - right) > 1e-3 for left, right in zip(box, clipped))
    ):
        raise ValueError(f"bbox exceeds normalized page in {context}: {value!r}")
    return clipped


def _scope_accepts(in_appendix: bool, scope: str) -> bool:
    if scope == "all":
        return True
    return in_appendix if scope == "appendix" else not in_appendix


def _page_in_scope(page: int, scope: str, appendix_start: int | None) -> bool:
    """Return whether an otherwise-unassigned prediction belongs to the scope."""

    if scope == "all":
        return True
    if scope == "body":
        return appendix_start is None or page < appendix_start
    return appendix_start is not None and page >= appendix_start


def _prediction_source(method: str) -> str:
    return METHOD_PREDICTION_SOURCES.get(method, method)


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _intersection(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0])))
    height = max(0.0, min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1])))
    return width * height


def _center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (
        float(box[1]) + float(box[3])
    ) / 2.0


def _contains_point(box: Sequence[float], point: tuple[float, float]) -> bool:
    return (
        float(box[0]) <= point[0] <= float(box[2])
        and float(box[1]) <= point[1] <= float(box[3])
    )


def _assign_prediction(
    prediction: Prediction,
    ground_truths: Sequence[GroundTruth],
) -> Assignment | None:
    """Assign one prediction to at most one same-page GT without thresholds."""

    prediction_center = _center(prediction.bbox)
    prediction_area = _area(prediction.bbox)
    candidates: list[tuple[tuple[float, float, float, int], Assignment]] = []
    for gt_index, target in enumerate(ground_truths):
        if prediction.page != target.page:
            continue
        gt_center = _center(target.bbox)
        prediction_center_in_gt = _contains_point(target.bbox, prediction_center)
        gt_center_in_prediction = _contains_point(prediction.bbox, gt_center)
        if not prediction_center_in_gt and not gt_center_in_prediction:
            continue
        intersection = _intersection(prediction.bbox, target.bbox)
        if intersection <= 0.0:
            continue
        target_area = _area(target.bbox)
        containment = intersection / max(1e-9, min(prediction_area, target_area))
        iou = intersection / max(1e-9, prediction_area + target_area - intersection)
        if prediction_center_in_gt and gt_center_in_prediction:
            relation = "mutual_center_containment"
        elif prediction_center_in_gt:
            relation = "prediction_center_in_gt"
        else:
            relation = "gt_center_in_prediction"
        assignment = Assignment(
            gt_index=gt_index,
            relation=relation,
            intersection=intersection,
            containment=containment,
            iou=iou,
        )
        # Deterministic best ownership: strongest containment, then IoU, then
        # absolute overlap, then the earlier GT annotation.
        rank = (containment, iou, intersection, -gt_index)
        candidates.append((rank, assignment))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _union_boxes(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    if not boxes:
        raise ValueError("cannot union an empty box sequence")
    merged = (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )
    clipped = omni._clip_box(merged)
    if clipped is None:
        raise ValueError(f"invalid merged bbox: {merged!r}")
    return clipped


def _gt_conditioned_metrics(ious: Sequence[float]) -> dict[str, Any]:
    """Summarize one grouped-crop IoU per GT, with missing GT represented by 0."""

    if not ious:
        raise ValueError("GT-conditioned metrics require at least one GT Figure")
    bounded = [min(1.0, max(0.0, float(value))) for value in ious]
    recall_by_threshold = {
        f"{threshold:.2f}": sum(value >= threshold for value in bounded) / len(bounded)
        for threshold in RECALL_THRESHOLDS
    }
    return {
        "gt_figure_count": len(bounded),
        "mean_iou": sum(bounded) / len(bounded),
        "recall_50": recall_by_threshold["0.50"],
        "recall_75": recall_by_threshold["0.75"],
        "mean_recall_50_95": (
            sum(recall_by_threshold.values()) / len(recall_by_threshold)
        ),
        "recall_by_iou_threshold": recall_by_threshold,
    }


def _dataset_papers(dataset_root: Path) -> tuple[Path, ...]:
    papers = tuple(
        sorted(
            directory
            for directory in dataset_root.iterdir()
            if directory.is_dir() and (directory / "paper.pdf").is_file()
        )
    )
    if not papers:
        raise FileNotFoundError(f"no <paper>/paper.pdf under {dataset_root}")
    return papers



def _load_gt_documents(gt_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Load GT paper documents from legacy */pairs_with_context.json or VASP-data annotations_948."""
    if gt_root.is_file():
        doc = _read_json(gt_root)
        if isinstance(doc, dict) and isinstance(doc.get("papers"), list):
            return [(gt_root, item) for item in doc["papers"] if isinstance(item, dict)]
        return [(gt_root, doc)]
    legacy = sorted(gt_root.glob("*/pairs_with_context.json"))
    if legacy:
        return [(p, _read_json(p)) for p in legacy]
    bundle = gt_root / "annotations_948.json"
    if bundle.is_file():
        doc = _read_json(bundle)
        if isinstance(doc, dict) and isinstance(doc.get("papers"), list):
            return [(bundle, item) for item in doc["papers"] if isinstance(item, dict)]
    per_dir = gt_root / "per_paper" if (gt_root / "per_paper").is_dir() else gt_root
    per_files = sorted(p for p in per_dir.glob("*.json") if p.name != "annotations_948.json")
    if per_files:
        return [(p, _read_json(p)) for p in per_files]
    raise FileNotFoundError(
        f"no pairs_with_context.json or annotations_948.json / per_paper/*.json under {gt_root}"
    )


def _load_ground_truth(
    config: Config,
    papers: Sequence[Path],
    *,
    scope: str | None = None,
) -> dict[int, tuple[GroundTruth, ...]]:
    selected_scope = config.scope if scope is None else scope
    gt_docs = _load_gt_documents(config.gt_root)
    by_paper: dict[int, tuple[GroundTruth, ...]] = {}
    for path, document in gt_docs:
        paper_index = document.get("paper_index")
        if isinstance(paper_index, bool) or not isinstance(paper_index, int):
            raise ValueError(f"invalid paper_index in {path}")
        if not 0 <= paper_index < len(papers):
            raise ValueError(f"paper_index outside dataset in {path}: {paper_index}")
        figures: list[GroundTruth] = []
        for pair in (document.get("pairs") if "pairs" in document else document.get("figures", [])):
            if not isinstance(pair, dict):
                continue
            in_appendix = bool(pair.get("in_appendix", False))
            if not _scope_accepts(in_appendix, selected_scope):
                continue
            page = pair.get("page")
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ValueError(f"invalid GT page in {path}: {page!r}")
            figures.append(
                GroundTruth(
                    paper_index=paper_index,
                    paper_title=papers[paper_index].name,
                    local_index=len(figures),
                    figure_id=str(pair.get("figure_id") or f"figure_{len(figures):03d}"),
                    label=str(pair.get("caption_label") or ""),
                    page=page,
                    bbox=_as_box(
                        pair.get("bbox"),
                        context=str(path),
                        clip_page_boundary=True,
                    ),
                    in_appendix=in_appendix,
                )
            )
        if paper_index in by_paper:
            raise ValueError(f"duplicate GT paper_index {paper_index}")
        by_paper[paper_index] = tuple(figures)
    missing = sorted(set(range(len(papers))) - set(by_paper))
    if missing:
        raise ValueError(f"GT missing paper indices: {missing}")
    return by_paper


def _build_context(config: Config) -> DatasetContext:
    papers = _dataset_papers(config.dataset_root)
    ground_truth_by_paper = _load_ground_truth(config, papers, scope=config.scope)
    ownership_ground_truth_by_paper = _load_ground_truth(config, papers, scope="all")
    appendix_start_pages = tuple(
        min((target.page for target in ownership_ground_truth_by_paper[index] if target.in_appendix), default=None)
        for index in range(len(papers))
    )
    page_counts: list[int] = []
    page_metric_ids: dict[tuple[int, int], int] = {}
    metric_ground_truths: list[dict[str, Any]] = []
    omni_ground_truth: list[dict[str, Any]] = []
    metric_index = 0
    for paper_index, paper_dir in enumerate(papers):
        with fitz.open(paper_dir / "paper.pdf") as pdf:
            page_count = len(pdf)
        page_counts.append(page_count)
        for target in ownership_ground_truth_by_paper[paper_index]:
            if target.page > page_count:
                raise ValueError(
                    f"GT page {target.page} exceeds {page_count}: {paper_dir.name}"
                )
        figures_by_page: dict[int, list[GroundTruth]] = defaultdict(list)
        for target in ground_truth_by_paper[paper_index]:
            if target.page > page_count:
                raise ValueError(
                    f"GT page {target.page} exceeds {page_count}: {paper_dir.name}"
                )
            figures_by_page[target.page].append(target)
        for page in range(1, page_count + 1):
            page_metric_ids[(paper_index, page)] = metric_index
            targets = figures_by_page.get(page, [])
            boxes = np.asarray(
                [target.bbox for target in targets], dtype=np.float32
            ).reshape(-1, 4)
            metric_ground_truths.append(
                {
                    "img_id": metric_index,
                    "width": int(omni.PAGE_CANVAS),
                    "height": int(omni.PAGE_CANVAS),
                    "bboxes": boxes,
                    "labels": np.zeros(len(targets), dtype=np.int64),
                    "ignore_flags": np.zeros(len(targets), dtype=bool),
                }
            )
            layout_dets = []
            for target in targets:
                x0, y0, x1, y1 = target.bbox
                layout_dets.append(
                    {
                        "category_type": "figure",
                        "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
                        "ignore": False,
                        "anno_id": target.figure_id,
                    }
                )
            omni_ground_truth.append(
                {
                    "page_info": {
                        "page_no": page - 1,
                        "image_path": omni._page_name(paper_index, page) + ".jpg",
                        "width": int(omni.PAGE_CANVAS),
                        "height": int(omni.PAGE_CANVAS),
                        "page_attribute": {
                            "data_source": "academic_literature",
                            "derived_subset": "native_gt_grouped_figure_bbox",
                        },
                    },
                    "layout_dets": layout_dets,
                }
            )
            metric_index += 1
    gt_count = sum(len(items) for items in ground_truth_by_paper.values())
    return DatasetContext(
        papers=tuple(papers),
        page_counts=tuple(page_counts),
        page_metric_ids=page_metric_ids,
        ground_truth_by_paper=ground_truth_by_paper,
        ownership_ground_truth_by_paper=ownership_ground_truth_by_paper,
        appendix_start_pages=appendix_start_pages,
        metric_ground_truths=metric_ground_truths,
        omni_ground_truth=omni_ground_truth,
        gt_count=gt_count,
        page_count=metric_index,
    )


def _load_predictions(
    config: Config,
    method: str,
    paper_title: str,
    page_count: int,
) -> tuple[Prediction, ...]:
    source_method = _prediction_source(method)
    path = config.prediction_root / source_method / paper_title / "predictions.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    document = _read_json(path)
    if document.get("schema_version") != NATIVE_SCHEMA_VERSION:
        raise ValueError(f"unexpected native schema in {path}")
    if document.get("method") != source_method:
        raise ValueError(f"method mismatch in {path}")
    if document.get("paper_title") != paper_title:
        raise ValueError(f"paper title mismatch in {path}")
    if document.get("page_indexing") != "zero_based":
        raise ValueError(f"non-zero-based page indexing in {path}")
    records = document.get("predictions")
    if not isinstance(records, list):
        raise ValueError(f"predictions is not a list in {path}")
    if document.get("prediction_count") != len(records):
        raise ValueError(f"prediction_count mismatch in {path}")
    predictions: list[Prediction] = []
    for order, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"prediction[{order}] is not an object in {path}")
        page_index = record.get("page_index")
        if isinstance(page_index, bool) or not isinstance(page_index, int):
            raise ValueError(f"invalid page_index prediction[{order}] in {path}")
        page = page_index + 1
        if not 1 <= page <= page_count:
            raise ValueError(f"page outside PDF prediction[{order}] in {path}: {page}")
        predictions.append(
            Prediction(
                order=order,
                prediction_id=str(record.get("prediction_id") or f"prediction:{order}"),
                label=str(record.get("label") or ""),
                source_type=str(record.get("source_type") or ""),
                crop=str(record.get("crop") or ""),
                page=page,
                bbox=_as_box(record.get("bbox_1000"), context=f"{path}:{order}"),
            )
        )
    return tuple(predictions)


def _metric_records(
    context: DatasetContext,
    boxes_by_page: dict[tuple[int, int], list[tuple[float, float, float, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric: list[dict[str, Any]] = []
    omni_results: list[dict[str, Any]] = []
    for paper_index, page_count in enumerate(context.page_counts):
        for page in range(1, page_count + 1):
            boxes = boxes_by_page.get((paper_index, page), [])
            metric.append(
                {
                    "img_id": context.page_metric_ids[(paper_index, page)],
                    "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    "scores": np.ones(len(boxes), dtype=np.float32),
                    "labels": np.zeros(len(boxes), dtype=np.int64),
                }
            )
            for bbox in boxes:
                omni_results.append(
                    {
                        "image_name": omni._page_name(paper_index, page),
                        "bbox": list(bbox),
                        "category_id": 0,
                        "score": 1.0,
                    }
                )
    return metric, omni_results


def evaluate_method(
    config: Config,
    context: DatasetContext,
    method: str,
) -> dict[str, Any]:
    output_dir = config.output_root / method
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_boxes_by_page: dict[
        tuple[int, int], list[tuple[float, float, float, float]]
    ] = defaultdict(list)
    grouped_entries_by_page: dict[
        tuple[int, int], list[tuple[int, tuple[float, float, float, float]]]
    ] = defaultdict(list)
    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    relation_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    in_scope_source_type_counts: Counter[str] = Counter()
    source_prediction_count = 0
    assigned_component_count = 0
    unassigned_component_count = 0
    ignored_out_of_scope_assigned_count = 0
    ignored_out_of_scope_unassigned_count = 0
    group_count = 0
    multi_component_group_count = 0
    components_in_multi_groups = 0
    covered_gt_keys: set[tuple[int, str]] = set()

    def component_row(
        paper_index: int,
        paper_title: str,
        prediction: Prediction,
        status: str,
        *,
        target: GroundTruth | None = None,
        assignment: Assignment | None = None,
    ) -> dict[str, Any]:
        return {
            "paper_index": paper_index,
            "paper_title": paper_title,
            "prediction_id": prediction.prediction_id,
            "prediction_label": prediction.label,
            "source_type": prediction.source_type,
            "crop": prediction.crop,
            "page": prediction.page,
            "bbox_x0": prediction.bbox[0],
            "bbox_y0": prediction.bbox[1],
            "bbox_x1": prediction.bbox[2],
            "bbox_y1": prediction.bbox[3],
            "matched_gt_id": "" if target is None else target.figure_id,
            "matched_gt_label": "" if target is None else target.label,
            "matched_gt_scope": (
                "" if target is None else ("appendix" if target.in_appendix else "body")
            ),
            "ownership_relation": "" if assignment is None else assignment.relation,
            "intersection_area": 0.0 if assignment is None else assignment.intersection,
            "containment": 0.0 if assignment is None else assignment.containment,
            "iou_before_grouping": 0.0 if assignment is None else assignment.iou,
            "status": status,
        }

    for paper_index, paper_dir in enumerate(context.papers):
        predictions = _load_predictions(
            config,
            method,
            paper_dir.name,
            context.page_counts[paper_index],
        )
        source_prediction_count += len(predictions)
        targets = context.ownership_ground_truth_by_paper[paper_index]
        appendix_start = context.appendix_start_pages[paper_index]
        assignments: dict[int, list[tuple[Prediction, Assignment]]] = defaultdict(list)

        for prediction in predictions:
            source_type = prediction.source_type or "<untyped>"
            source_type_counts[source_type] += 1
            assignment = _assign_prediction(prediction, targets)

            if assignment is None:
                if not _page_in_scope(prediction.page, config.scope, appendix_start):
                    ignored_out_of_scope_unassigned_count += 1
                    component_rows.append(
                        component_row(
                            paper_index,
                            paper_dir.name,
                            prediction,
                            "ignored_out_of_scope_unassociated",
                        )
                    )
                    continue
                raw_boxes_by_page[(paper_index, prediction.page)].append(prediction.bbox)
                in_scope_source_type_counts[source_type] += 1
                unassigned_component_count += 1
                component_rows.append(
                    component_row(
                        paper_index,
                        paper_dir.name,
                        prediction,
                        "unassociated_in_scope_candidate",
                    )
                )
                continue

            target = targets[assignment.gt_index]
            if not _scope_accepts(target.in_appendix, config.scope):
                ignored_out_of_scope_assigned_count += 1
                component_rows.append(
                    component_row(
                        paper_index,
                        paper_dir.name,
                        prediction,
                        "ignored_out_of_scope_associated",
                        target=target,
                        assignment=assignment,
                    )
                )
                continue

            raw_boxes_by_page[(paper_index, prediction.page)].append(prediction.bbox)
            in_scope_source_type_counts[source_type] += 1
            assignments[assignment.gt_index].append((prediction, assignment))
            assigned_component_count += 1
            relation_counts[assignment.relation] += 1
            component_rows.append(
                component_row(
                    paper_index,
                    paper_dir.name,
                    prediction,
                    "associated_with_figure_gt",
                    target=target,
                    assignment=assignment,
                )
            )

        for gt_index, members in assignments.items():
            target = targets[gt_index]
            merged = _union_boxes([member[0].bbox for member in members])
            first_order = min(member[0].order for member in members)
            grouped_entries_by_page[(paper_index, target.page)].append(
                (first_order, merged)
            )
            group_count += 1
            covered_gt_keys.add((paper_index, target.figure_id))
            if len(members) > 1:
                multi_component_group_count += 1
                components_in_multi_groups += len(members)
            intersection = _intersection(merged, target.bbox)
            merged_area = _area(merged)
            target_area = _area(target.bbox)
            merged_iou = intersection / max(
                1e-9, merged_area + target_area - intersection
            )
            group_rows.append(
                {
                    "paper_index": paper_index,
                    "paper_title": paper_dir.name,
                    "gt_local_index": target.local_index,
                    "matched_gt_id": target.figure_id,
                    "matched_gt_label": target.label,
                    "page": target.page,
                    "component_count": len(members),
                    "component_ids": " | ".join(
                        member[0].prediction_id for member in members
                    ),
                    "bbox_x0": merged[0],
                    "bbox_y0": merged[1],
                    "bbox_x1": merged[2],
                    "bbox_y1": merged[3],
                    "gt_x0": target.bbox[0],
                    "gt_y0": target.bbox[1],
                    "gt_x1": target.bbox[2],
                    "gt_y1": target.bbox[3],
                    "grouped_iou": merged_iou,
                    "status": "grouped" if len(members) > 1 else "singleton",
                }
            )

    for paper_index, targets in context.ground_truth_by_paper.items():
        for target in targets:
            if (paper_index, target.figure_id) in covered_gt_keys:
                continue
            group_rows.append(
                {
                    "paper_index": paper_index,
                    "paper_title": target.paper_title,
                    "gt_local_index": target.local_index,
                    "matched_gt_id": target.figure_id,
                    "matched_gt_label": target.label,
                    "page": target.page,
                    "component_count": 0,
                    "component_ids": "",
                    "bbox_x0": "",
                    "bbox_y0": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "gt_x0": target.bbox[0],
                    "gt_y0": target.bbox[1],
                    "gt_x1": target.bbox[2],
                    "gt_y1": target.bbox[3],
                    "grouped_iou": 0.0,
                    "status": "missing",
                }
            )

    group_rows.sort(
        key=lambda row: (
            int(row["paper_index"]),
            int(row["page"]),
            int(row["gt_local_index"]),
        )
    )
    if len(group_rows) != context.gt_count:
        raise AssertionError(
            f"expected one crop-quality row per GT: {len(group_rows)} != {context.gt_count}"
        )

    grouped_boxes_by_page: dict[
        tuple[int, int], list[tuple[float, float, float, float]]
    ] = {}
    for key, entries in grouped_entries_by_page.items():
        grouped_boxes_by_page[key] = [
            bbox for _order, bbox in sorted(entries, key=lambda item: item[0])
        ]

    _, raw_omni_results = _metric_records(context, raw_boxes_by_page)
    _, grouped_omni_results = _metric_records(context, grouped_boxes_by_page)
    crop_quality = _gt_conditioned_metrics(
        [float(row["grouped_iou"]) for row in group_rows]
    )
    in_scope_candidate_count = assigned_component_count + unassigned_component_count
    collapsed_count = assigned_component_count - group_count
    missing_gt_count = context.gt_count - group_count
    candidate_audit = {
        "in_scope_visual_candidate_count": in_scope_candidate_count,
        "figure_associated_component_count": assigned_component_count,
        "unassociated_candidate_count": unassigned_component_count,
        "unassociated_candidate_fraction": (
            unassigned_component_count / in_scope_candidate_count
            if in_scope_candidate_count
            else 0.0
        ),
        "gt_with_associated_component_count": group_count,
        "gt_without_associated_component_count": missing_gt_count,
        "all_candidates_per_covered_gt": (
            in_scope_candidate_count / group_count if group_count else 0.0
        ),
        "associated_components_per_covered_gt": (
            assigned_component_count / group_count if group_count else 0.0
        ),
        "multi_component_group_count": multi_component_group_count,
        "components_in_multi_component_groups": components_in_multi_groups,
        "components_collapsed_by_grouping": collapsed_count,
        "ignored_out_of_scope_associated_count": ignored_out_of_scope_assigned_count,
        "ignored_out_of_scope_unassociated_count": ignored_out_of_scope_unassigned_count,
        "source_type_counts_all_scopes": dict(sorted(source_type_counts.items())),
        "source_type_counts_in_scope": dict(sorted(in_scope_source_type_counts.items())),
    }
    source_method = _prediction_source(method)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "method_display_name": METHOD_DISPLAY_NAMES.get(method, method),
        "prediction_source_method": source_method,
        "protocol": {
            "metric_family": "GT-conditioned Figure crop quality",
            "coordinate_space": "top_left_0_1000",
            "raw_input": "all native visual bboxes emitted by the prediction source",
            "method_category_policy": (
                "ignore all method-provided semantic categories; no Figure/Table filtering"
            ),
            "grouping_policy": (
                "same-page prediction assigned to at most one GT when either box "
                "contains the other's center; same-GT components become outer union"
            ),
            "ambiguous_assignment_policy": (
                "max intersection/min-area containment, then IoU, intersection, GT order"
            ),
            "scope_assignment_policy": (
                "associate against body+appendix GT first, then retain requested scope"
            ),
            "unassociated_prediction_policy": (
                "audited but excluded from crop-quality metrics because Figure-only GT "
                "cannot distinguish valid non-Figure assets from meaningless crops"
            ),
            "uncovered_gt_policy": "contributes IoU 0 to every primary metric",
            "primary_metrics": {
                "mean_iou": "mean grouped IoU over every GT, including zero for missing GT",
                "recall_50": "fraction of all GT with grouped IoU >= 0.50",
                "recall_75": "fraction of all GT with grouped IoU >= 0.75",
                "mean_recall_50_95": (
                    "mean GT recall at IoU thresholds 0.50:0.05:0.95"
                ),
            },
            "is_coco_average_precision": False,
            "gt_assisted": True,
            "intended_use": "crop/geometry comparison, not end-to-end semantic precision",
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
            "source_prediction_count_all_scopes": source_prediction_count,
            "in_scope_visual_candidate_count": in_scope_candidate_count,
        },
        "grouping": {
            "assigned_component_count": assigned_component_count,
            "unassigned_component_count": unassigned_component_count,
            "ignored_out_of_scope_assigned_count": ignored_out_of_scope_assigned_count,
            "ignored_out_of_scope_unassigned_count": ignored_out_of_scope_unassigned_count,
            "gt_group_count": group_count,
            "multi_component_group_count": multi_component_group_count,
            "components_in_multi_component_groups": components_in_multi_groups,
            "components_collapsed_by_grouping": collapsed_count,
            "gt_with_assigned_component_count": group_count,
            "gt_without_assigned_component_count": missing_gt_count,
            "ownership_relation_counts": dict(relation_counts),
        },
        "crop_quality": crop_quality,
        "candidate_audit": candidate_audit,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "omni_ground_truth.json", context.omni_ground_truth)
    _write_json(
        output_dir / "visual_candidates_in_scope.json",
        {"results": raw_omni_results, "categories": {"0": "visual_candidate"}},
    )
    _write_json(
        output_dir / "gt_conditioned_figure_groups.json",
        {"results": grouped_omni_results, "categories": {"0": "figure"}},
    )
    _write_csv(output_dir / "per_component_assignment.csv", component_rows)
    _write_csv(output_dir / "per_gt_figure.csv", group_rows)
    return summary


def _crop_quality_row(summary: dict[str, Any]) -> dict[str, Any]:
    quality = summary["crop_quality"]
    return {
        "method": summary["method_display_name"],
        "method_id": summary["method"],
        "prediction_source_method": summary["prediction_source_method"],
        "gt_figure_count": quality["gt_figure_count"],
        "mIoU": quality["mean_iou"] * 100.0,
        "R50": quality["recall_50"] * 100.0,
        "R75": quality["recall_75"] * 100.0,
        "mAR_50_95": quality["mean_recall_50_95"] * 100.0,
    }


def _candidate_audit_row(summary: dict[str, Any]) -> dict[str, Any]:
    audit = summary["candidate_audit"]
    return {
        "method": summary["method_display_name"],
        "method_id": summary["method"],
        "prediction_source_method": summary["prediction_source_method"],
        "visual_candidates": audit["in_scope_visual_candidate_count"],
        "figure_associated_components": audit["figure_associated_component_count"],
        "unassociated_candidates": audit["unassociated_candidate_count"],
        "unassociated_fraction": audit["unassociated_candidate_fraction"],
        "covered_gt": audit["gt_with_associated_component_count"],
        "missing_gt": audit["gt_without_associated_component_count"],
        "multi_component_gt": audit["multi_component_group_count"],
        "all_candidates_per_covered_gt": audit["all_candidates_per_covered_gt"],
        "associated_components_per_covered_gt": audit[
            "associated_components_per_covered_gt"
        ],
    }


def _comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {**_crop_quality_row(summary), **_candidate_audit_row(summary)}


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if index == 0 else "---:" for index in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _write_comparison(output_root: Path, summaries: Sequence[dict[str, Any]]) -> None:
    quality_rows = [_crop_quality_row(summary) for summary in summaries]
    quality_rows.sort(key=lambda row: row["mAR_50_95"], reverse=True)
    order = {row["method_id"]: index for index, row in enumerate(quality_rows)}
    audit_rows = [_candidate_audit_row(summary) for summary in summaries]
    audit_rows.sort(key=lambda row: order[row["method_id"]])
    combined_rows = [_comparison_row(summary) for summary in summaries]
    combined_rows.sort(key=lambda row: order[row["method_id"]])

    _write_json(output_root / "crop_quality.json", quality_rows)
    _write_csv(output_root / "crop_quality.csv", quality_rows)
    _write_json(output_root / "candidate_audit.json", audit_rows)
    _write_csv(output_root / "candidate_audit.csv", audit_rows)
    _write_json(output_root / "comparison.json", combined_rows)
    _write_csv(output_root / "comparison.csv", combined_rows)

    quality_markdown = (
        "# Table 1: GT-Conditioned Figure Crop Quality (Main Text)\n\n"
        + _markdown_table(
            ("Method", "mIoU", "R@50", "R@75", "mAR@50:95"),
            [
                (
                    row["method"],
                    f'{row["mIoU"]:.2f}',
                    f'{row["R50"]:.2f}',
                    f'{row["R75"]:.2f}',
                    f'{row["mAR_50_95"]:.2f}',
                )
                for row in quality_rows
            ],
        )
        + "\nEach GT figure contributes one aggregated-box IoU; uncovered GT figures receive 0. Unassociated candidates are excluded from this table.\n"
    )
    (output_root / "crop_quality_table.md").write_text(
        quality_markdown, encoding="utf-8"
    )

    audit_markdown = (
        "# Table 2: Visual Candidates and Figure Association Audit (Main Text)\n\n"
        + _markdown_table(
            (
                "Method",
                "Visual Candidates",
                "Figure-Associated Components",
                "Unassociated Candidates",
                "Covered GT",
                "Missing GT",
                "Multi-Component GT",
                "Components / Covered GT",
            ),
            [
                (
                    row["method"],
                    row["visual_candidates"],
                    row["figure_associated_components"],
                    row["unassociated_candidates"],
                    row["covered_gt"],
                    row["missing_gt"],
                    row["multi_component_gt"],
                    f'{row["associated_components_per_covered_gt"]:.2f}',
                )
                for row in audit_rows
            ],
        )
        + (
            "\nAll methods use all native visual candidates without reading Figure/Table class labels. "
            "Unassociated candidates are audited for completeness and are not equivalent to false positives.\n"
        )
    )
    (output_root / "candidate_audit_table.md").write_text(
        audit_markdown, encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root", type=Path, default=Path("results/figure-caption-pairs")
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
        default=Path("results/exp/gt_conditioned_figure_crop"),
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=sorted(METHOD_DISPLAY_NAMES),
        help="method to evaluate; repeat for multiple methods (default: all)",
    )
    parser.add_argument("--scope", choices=["body", "appendix", "all"], default="body")
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
    methods = args.method or list(DEFAULT_METHODS)
    context = _build_context(config)
    summaries = []
    for index, method in enumerate(methods, start=1):
        print(f"[{index}/{len(methods)}] evaluating {method}", flush=True)
        summary = evaluate_method(config, context, method)
        summaries.append(summary)
        metrics = summary["crop_quality"]
        print(
            f"  mIoU={metrics['mean_iou']:.6f} "
            f"R50={metrics['recall_50']:.6f} "
            f"R75={metrics['recall_75']:.6f} "
            f"mAR={metrics['mean_recall_50_95']:.6f}",
            flush=True,
        )
    _write_comparison(config.output_root, summaries)
    print(json.dumps([_comparison_row(item) for item in summaries], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
