"""Evaluate full-paper Figure crops against the derived GT bundle.

The GT stores boxes in an independently normalized 1000 x 1000 page coordinate
system. Active paper2assets predictions store boxes in PDF points. This module
normalizes predictions, performs same-page maximum-IoU matching, and reports
both geometric crop quality and label-only detection.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import fitz


SCHEMA_VERSION = "dp-p2p.figure-crop-benchmark.v1"
GT_CANVAS_SIZE = 1000.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned x0, y0, x1, y1 box."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def intersection(self, other: "Box") -> float:
        width = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        height = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        return width * height

    def iou(self, other: "Box") -> float:
        intersection = self.intersection(other)
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class GroundTruthFigure:
    paper_index: int
    paper_title: str
    figure_id: str
    label: str
    page: int
    bbox: Box
    figure_path: str
    in_appendix: bool


@dataclass(frozen=True)
class PredictionFigure:
    paper_index: int
    paper_title: str
    prediction_index: int
    label: str
    page: int | None
    bbox: Box | None
    file: str
    asset_path: str
    asset_exists: bool
    invalid_reason: str
    coordinate_space: str


@dataclass(frozen=True)
class PaperSpec:
    paper_index: int
    paper_title: str
    pdf_path: Path
    gt_json_path: Path
    figures: tuple[GroundTruthFigure, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    gt_root: Path
    dataset_root: Path
    prediction_root: Path
    output_dir: Path
    method: str = "prediction"
    scope: str = "body"
    prediction_json: str = "assets/meta/figures.json"
    prediction_space: str = "pdf"
    prediction_page_base: int = 1
    detection_iou: float = 0.5
    coverage_threshold: float = 0.95
    contamination_threshold: float = 0.10

    def validate(self) -> None:
        if self.scope not in {"body", "appendix", "all"}:
            raise ValueError(f"unsupported scope: {self.scope}")
        if self.prediction_space not in {
            "pdf",
            "normalized1000",
            "page_pixels",
            "auto",
        }:
            raise ValueError(f"unsupported prediction space: {self.prediction_space}")
        if self.prediction_page_base not in {0, 1}:
            raise ValueError("prediction_page_base must be 0 or 1")
        for name, value in (
            ("detection_iou", self.detection_iou),
            ("coverage_threshold", self.coverage_threshold),
            ("contamination_threshold", self.contamination_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _summary_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _parse_box(raw: Any) -> Box | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw
    ):
        return None
    box = Box(*(float(value) for value in raw))
    if not all(math.isfinite(value) for value in box.as_list()) or box.area <= 0.0:
        return None
    return box


def _sanitize_title(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", title)


def _normalized_label(label: str) -> str:
    text = " ".join(str(label or "").strip().split()).casefold()
    match = re.search(
        r"\bfig(?:ure)?\.?\s*([a-z]*\d+(?:\.\d+)*(?:[a-z])?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).casefold()
    return text


def _scope_accepts(in_appendix: bool, scope: str) -> bool:
    return scope == "all" or (scope == "appendix") == in_appendix



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


def _load_paper_specs(
    config: EvaluationConfig,
) -> tuple[list[PaperSpec], list[str], int]:
    gt_docs = _load_gt_documents(config.gt_root)
    dataset_dirs = sorted(
        path
        for path in config.dataset_root.iterdir()
        if path.is_dir() and (path / "paper.pdf").is_file()
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"no <paper>/paper.pdf under {config.dataset_root}")

    warnings: list[str] = []
    seen_indices: set[int] = set()
    specs: list[PaperSpec] = []
    missing_gt_images = 0
    for gt_path, document in gt_docs:
        if not isinstance(document, dict):
            raise ValueError(f"GT must be an object: {gt_path}")
        paper_index = document.get("paper_index")
        if isinstance(paper_index, bool) or not isinstance(paper_index, int):
            raise ValueError(f"invalid paper_index in {gt_path}")
        if paper_index in seen_indices:
            raise ValueError(f"duplicate paper_index {paper_index}")
        seen_indices.add(paper_index)
        if not 0 <= paper_index < len(dataset_dirs):
            raise ValueError(f"paper_index {paper_index} outside dataset range")

        dataset_dir = dataset_dirs[paper_index]
        paper_title = dataset_dir.name
        expected_gt_name = f"{paper_index:03d}_{_sanitize_title(paper_title)}"
        if gt_path.parent.name != expected_gt_name:
            warnings.append(
                f"paper {paper_index:03d}: GT directory {gt_path.parent.name!r} "
                f"does not match {expected_gt_name!r}"
            )

        raw_pairs = document.get("pairs") if "pairs" in document else document.get("figures")
        if not isinstance(raw_pairs, list):
            raise ValueError(f"GT pairs must be a list: {gt_path}")
        declared_count = document.get("n_figures")
        if isinstance(declared_count, int) and declared_count != len(raw_pairs):
            warnings.append(
                f"paper {paper_index:03d}: n_figures={declared_count}, "
                f"pairs={len(raw_pairs)}"
            )

        figures: list[GroundTruthFigure] = []
        for pair_index, pair in enumerate(raw_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"invalid GT pair {pair_index} in {gt_path}")
            in_appendix = bool(pair.get("in_appendix", False))
            if not _scope_accepts(in_appendix, config.scope):
                continue
            bbox = _parse_box(pair.get("bbox"))
            page = pair.get("page")
            if (
                bbox is None
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                raise ValueError(
                    f"invalid GT geometry for pair {pair_index} in {gt_path}"
                )
            figure_path = str(pair.get("figure_path") or "")
            if not figure_path or not (gt_path.parent / figure_path).is_file():
                missing_gt_images += 1
            figures.append(
                GroundTruthFigure(
                    paper_index=paper_index,
                    paper_title=paper_title,
                    figure_id=str(
                        pair.get("figure_id") or f"pair_{pair_index:03d}"
                    ),
                    label=str(pair.get("caption_label") or ""),
                    page=page,
                    bbox=bbox,
                    figure_path=figure_path,
                    in_appendix=in_appendix,
                )
            )
        specs.append(
            PaperSpec(
                paper_index=paper_index,
                paper_title=paper_title,
                pdf_path=dataset_dir / "paper.pdf",
                gt_json_path=gt_path,
                figures=tuple(figures),
            )
        )

    specs.sort(key=lambda item: item.paper_index)
    if len(specs) != len(dataset_dirs):
        warnings.append(
            f"GT covers {len(specs)} papers while dataset contains "
            f"{len(dataset_dirs)} papers"
        )
    return specs, warnings, missing_gt_images


def _resolve_asset_path(
    paper_root: Path, prediction_path: Path, raw_file: str
) -> Path:
    if not raw_file:
        return paper_root / "__missing_figure_file__"
    value = Path(raw_file)
    candidates = [
        paper_root / value,
        paper_root / "assets" / value,
        prediction_path.parent.parent / value,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _prediction_coordinate_space(
    requested: str,
    raw: dict[str, Any],
    bbox: Box,
    pdf_width: float,
    pdf_height: float,
) -> str:
    if requested != "auto":
        return requested
    declared = raw.get("bbox_space") or raw.get("coordinate_space")
    if declared in {"pdf", "normalized1000", "page_pixels"}:
        return str(declared)
    page_width = raw.get("page_width")
    page_height = raw.get("page_height")
    if (
        isinstance(page_width, (int, float))
        and not isinstance(page_width, bool)
        and isinstance(page_height, (int, float))
        and not isinstance(page_height, bool)
        and page_width > 0
        and page_height > 0
        and (bbox.x1 > pdf_width * 1.1 or bbox.y1 > pdf_height * 1.1)
    ):
        return "page_pixels"
    return "pdf"


def _normalize_prediction_box(
    bbox: Box,
    raw: dict[str, Any],
    page_width: float,
    page_height: float,
    requested_space: str,
) -> tuple[Box | None, str, str]:
    space = _prediction_coordinate_space(
        requested_space, raw, bbox, page_width, page_height
    )
    if space == "normalized1000":
        return bbox, space, ""
    if space == "pdf":
        return (
            Box(
                bbox.x0 / page_width * GT_CANVAS_SIZE,
                bbox.y0 / page_height * GT_CANVAS_SIZE,
                bbox.x1 / page_width * GT_CANVAS_SIZE,
                bbox.y1 / page_height * GT_CANVAS_SIZE,
            ),
            space,
            "",
        )
    pixel_width = raw.get("page_width")
    pixel_height = raw.get("page_height")
    if (
        isinstance(pixel_width, bool)
        or isinstance(pixel_height, bool)
        or not isinstance(pixel_width, (int, float))
        or not isinstance(pixel_height, (int, float))
        or pixel_width <= 0
        or pixel_height <= 0
    ):
        return (
            None,
            space,
            "page_pixels bbox requires positive page_width/page_height",
        )
    return (
        Box(
            bbox.x0 / float(pixel_width) * GT_CANVAS_SIZE,
            bbox.y0 / float(pixel_height) * GT_CANVAS_SIZE,
            bbox.x1 / float(pixel_width) * GT_CANVAS_SIZE,
            bbox.y1 / float(pixel_height) * GT_CANVAS_SIZE,
        ),
        space,
        "",
    )


def _load_predictions(
    spec: PaperSpec, config: EvaluationConfig
) -> tuple[list[PredictionFigure], bool, str]:
    paper_root = config.prediction_root / spec.paper_title
    prediction_path = paper_root / config.prediction_json
    if not prediction_path.is_file():
        return [], False, f"missing {prediction_path}"
    try:
        document = _read_json(prediction_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [], True, f"cannot read {prediction_path}: {exc}"
    raw_predictions = document.get("figures") if isinstance(document, dict) else document
    if not isinstance(raw_predictions, list):
        return [], True, f"prediction JSON is not a list: {prediction_path}"

    predictions: list[PredictionFigure] = []
    with fitz.open(spec.pdf_path) as pdf:
        for prediction_index, raw in enumerate(raw_predictions):
            if not isinstance(raw, dict):
                predictions.append(
                    PredictionFigure(
                        spec.paper_index,
                        spec.paper_title,
                        prediction_index,
                        "",
                        None,
                        None,
                        "",
                        "",
                        False,
                        "prediction entry is not an object",
                        "",
                    )
                )
                continue

            label = str(raw.get("label") or raw.get("caption_label") or "")
            raw_file = str(raw.get("file") or "")
            asset_path = _resolve_asset_path(
                paper_root, prediction_path, raw_file
            )
            page_value = raw.get("page")
            page = None
            invalid_reason = ""
            if isinstance(page_value, bool) or not isinstance(page_value, int):
                invalid_reason = "missing or invalid page"
            else:
                page = page_value + (1 if config.prediction_page_base == 0 else 0)
                if not 1 <= page <= len(pdf):
                    invalid_reason = f"page {page} outside PDF range 1..{len(pdf)}"

            raw_bbox = _parse_box(raw.get("bbox"))
            normalized_bbox = None
            coordinate_space = ""
            if raw_bbox is None:
                invalid_reason = invalid_reason or "missing or invalid bbox"
            elif page is not None and not invalid_reason:
                page_rect = pdf[page - 1].rect
                (
                    normalized_bbox,
                    coordinate_space,
                    bbox_error,
                ) = _normalize_prediction_box(
                    raw_bbox,
                    raw,
                    float(page_rect.width),
                    float(page_rect.height),
                    config.prediction_space,
                )
                invalid_reason = bbox_error

            predictions.append(
                PredictionFigure(
                    paper_index=spec.paper_index,
                    paper_title=spec.paper_title,
                    prediction_index=prediction_index,
                    label=label,
                    page=page,
                    bbox=normalized_bbox,
                    file=raw_file,
                    asset_path=str(asset_path),
                    asset_exists=asset_path.is_file(),
                    invalid_reason=invalid_reason,
                    coordinate_space=coordinate_space,
                )
            )
    return predictions, True, ""


def _maximum_weight_matching(
    weights: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    """Return a maximum-total-weight rectangular assignment in O(n^3)."""
    if not weights or not weights[0]:
        return []
    rows = len(weights)
    columns = len(weights[0])
    if any(len(row) != columns for row in weights):
        raise ValueError("ragged weight matrix")

    transposed = rows > columns
    matrix = (
        [
            [weights[row][column] for row in range(rows)]
            for column in range(columns)
        ]
        if transposed
        else [list(row) for row in weights]
    )
    n = len(matrix)
    m = len(matrix[0])
    maximum = max(max(row) for row in matrix)
    costs = [[maximum - value for value in row] for row in matrix]

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_values = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < min_values[j]:
                    min_values[j] = current
                    way[j] = j0
                if min_values[j] < delta:
                    delta = min_values[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_values[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for column in range(1, m + 1):
        if p[column] == 0:
            continue
        row_index = p[column] - 1
        column_index = column - 1
        pairs.append(
            (column_index, row_index)
            if transposed
            else (row_index, column_index)
        )
    return sorted(pairs)


def _spatial_matches(
    ground_truth: Sequence[GroundTruthFigure],
    predictions: Sequence[PredictionFigure],
) -> list[tuple[int, int, float]]:
    gt_by_page: dict[int, list[int]] = defaultdict(list)
    prediction_by_page: dict[int, list[int]] = defaultdict(list)
    for index, figure in enumerate(ground_truth):
        gt_by_page[figure.page].append(index)
    for index, figure in enumerate(predictions):
        if (
            figure.page is not None
            and figure.bbox is not None
            and not figure.invalid_reason
        ):
            prediction_by_page[figure.page].append(index)

    matches: list[tuple[int, int, float]] = []
    for page in sorted(set(gt_by_page) | set(prediction_by_page)):
        gt_indices = gt_by_page.get(page, [])
        prediction_indices = prediction_by_page.get(page, [])
        if not gt_indices or not prediction_indices:
            continue
        weights = [
            [
                ground_truth[gt_index].bbox.iou(
                    predictions[prediction_index].bbox
                )
                for prediction_index in prediction_indices
            ]
            for gt_index in gt_indices
        ]
        for gt_local, prediction_local in _maximum_weight_matching(weights):
            iou = weights[gt_local][prediction_local]
            if iou > 0.0:
                matches.append(
                    (
                        gt_indices[gt_local],
                        prediction_indices[prediction_local],
                        iou,
                    )
                )
    return matches


def _label_true_positives(
    ground_truth: Sequence[GroundTruthFigure],
    predictions: Sequence[PredictionFigure],
) -> int:
    gt_counter = Counter(
        normalized
        for item in ground_truth
        if (normalized := _normalized_label(item.label))
    )
    prediction_counter = Counter(
        normalized
        for item in predictions
        if (normalized := _normalized_label(item.label))
    )
    return sum((gt_counter & prediction_counter).values())


def _json_box(box: Box | None) -> str:
    if box is None:
        return ""
    return json.dumps(box.as_list(), separators=(",", ":"))


def _paper_metrics(
    gt_count: int,
    prediction_count: int,
    detection_true_positives: int,
    usable_true_positives: int,
    label_true_positives: int,
) -> dict[str, float | int]:
    detection_precision = _safe_div(detection_true_positives, prediction_count)
    detection_recall = _safe_div(detection_true_positives, gt_count)
    usable_precision = _safe_div(usable_true_positives, prediction_count)
    usable_recall = _safe_div(usable_true_positives, gt_count)
    label_precision = _safe_div(label_true_positives, prediction_count)
    label_recall = _safe_div(label_true_positives, gt_count)
    return {
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "detection_tp": detection_true_positives,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": _f1(detection_precision, detection_recall),
        "usable_tp": usable_true_positives,
        "usable_precision": usable_precision,
        "usable_recall": usable_recall,
        "usable_f1": _f1(usable_precision, usable_recall),
        "label_tp": label_true_positives,
        "label_precision": label_precision,
        "label_recall": label_recall,
        "label_f1": _f1(label_precision, label_recall),
    }


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    metrics = summary["metrics"]
    detection = metrics["spatial_detection"]
    usable = metrics["usable_crop"]
    labels = metrics["label_detection"]
    geometry = metrics["geometry"]
    dataset = summary["dataset"]
    threshold = summary["config"]["detection_iou"]
    lines = [
        f"# Figure Crop Benchmark: {summary['method']}",
        "",
        f"Scope: {summary['config']['scope']}  ",
        f"Papers: {dataset['paper_count']}  ",
        f"GT Figures: {dataset['gt_figure_count']}  ",
        f"Prediction entries: {dataset['prediction_count']}  ",
        f"Geometrically evaluable predictions: {dataset['valid_geometry_count']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Spatial Precision @ IoU {threshold:.2f} | {detection['precision']:.4f} |",
        f"| Spatial Recall @ IoU {threshold:.2f} | {detection['recall']:.4f} |",
        f"| Spatial F1 | {detection['f1']:.4f} |",
        f"| Usable Crop Precision | {usable['precision']:.4f} |",
        f"| Usable Crop Recall | {usable['recall']:.4f} |",
        f"| Usable Crop F1 | {usable['f1']:.4f} |",
        f"| Label Precision | {labels['precision']:.4f} |",
        f"| Label Recall | {labels['recall']:.4f} |",
        f"| Label F1 | {labels['f1']:.4f} |",
        f"| Mean matched IoU | {_format_optional(geometry['iou']['mean'])} |",
        f"| Mean matched Coverage | {_format_optional(geometry['coverage']['mean'])} |",
        "| Mean matched Contamination | "
        f"{_format_optional(geometry['contamination']['mean'])} |",
        "",
        "A usable crop must have "
        f"coverage >= {summary['config']['coverage_threshold']:.2f}, "
        f"contamination <= {summary['config']['contamination_threshold']:.2f}, "
        "and an existing image asset.",
    ]
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


FIGURE_FIELDS = [
    "paper_index",
    "paper_title",
    "status",
    "gt_figure_id",
    "gt_label",
    "page",
    "gt_bbox",
    "gt_figure_path",
    "prediction_index",
    "prediction_label",
    "prediction_bbox",
    "prediction_file",
    "asset_exists",
    "iou",
    "coverage",
    "contamination",
    "label_match",
    "detection_success",
    "usable",
    "invalid_reason",
]


def _matched_row(
    spec: PaperSpec,
    gt: GroundTruthFigure,
    prediction: PredictionFigure,
    iou: float,
    coverage: float,
    contamination: float,
    detection_success: bool,
    usable: bool,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "paper_index": spec.paper_index,
        "paper_title": spec.paper_title,
        "status": "usable" if usable else "+".join(reasons) or "matched",
        "gt_figure_id": gt.figure_id,
        "gt_label": gt.label,
        "page": gt.page,
        "gt_bbox": _json_box(gt.bbox),
        "gt_figure_path": str(spec.gt_json_path.parent / gt.figure_path),
        "prediction_index": prediction.prediction_index,
        "prediction_label": prediction.label,
        "prediction_bbox": _json_box(prediction.bbox),
        "prediction_file": prediction.asset_path,
        "asset_exists": prediction.asset_exists,
        "iou": iou,
        "coverage": coverage,
        "contamination": contamination,
        "label_match": _normalized_label(gt.label)
        == _normalized_label(prediction.label),
        "detection_success": detection_success,
        "usable": usable,
        "invalid_reason": "",
    }


def _missed_row(spec: PaperSpec, gt: GroundTruthFigure) -> dict[str, Any]:
    return {
        "paper_index": spec.paper_index,
        "paper_title": spec.paper_title,
        "status": "missed",
        "gt_figure_id": gt.figure_id,
        "gt_label": gt.label,
        "page": gt.page,
        "gt_bbox": _json_box(gt.bbox),
        "gt_figure_path": str(spec.gt_json_path.parent / gt.figure_path),
        "prediction_index": "",
        "prediction_label": "",
        "prediction_bbox": "",
        "prediction_file": "",
        "asset_exists": False,
        "iou": 0.0,
        "coverage": 0.0,
        "contamination": "",
        "label_match": False,
        "detection_success": False,
        "usable": False,
        "invalid_reason": "",
    }


def _unmatched_prediction_row(
    spec: PaperSpec, prediction: PredictionFigure
) -> dict[str, Any]:
    return {
        "paper_index": spec.paper_index,
        "paper_title": spec.paper_title,
        "status": (
            "invalid_prediction"
            if prediction.invalid_reason
            else "false_positive"
        ),
        "gt_figure_id": "",
        "gt_label": "",
        "page": prediction.page if prediction.page is not None else "",
        "gt_bbox": "",
        "gt_figure_path": "",
        "prediction_index": prediction.prediction_index,
        "prediction_label": prediction.label,
        "prediction_bbox": _json_box(prediction.bbox),
        "prediction_file": prediction.asset_path,
        "asset_exists": prediction.asset_exists,
        "iou": 0.0,
        "coverage": "",
        "contamination": 1.0,
        "label_match": False,
        "detection_success": False,
        "usable": False,
        "invalid_reason": prediction.invalid_reason,
    }


def evaluate_figure_crops(config: EvaluationConfig) -> dict[str, Any]:
    """Run the benchmark and write JSON, CSV, and Markdown reports."""
    config.validate()
    specs, warnings, missing_gt_images = _load_paper_specs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    per_figure_rows: list[dict[str, Any]] = []
    per_paper_rows: list[dict[str, Any]] = []
    all_ious: list[float] = []
    all_coverages: list[float] = []
    all_contaminations: list[float] = []
    coordinate_spaces: Counter[str] = Counter()

    total_gt = 0
    total_predictions = 0
    total_valid_geometry = 0
    total_existing_assets = 0
    total_detection_tp = 0
    total_usable_tp = 0
    total_label_tp = 0
    prediction_files_found = 0
    prediction_file_errors = 0

    for spec in specs:
        predictions, prediction_file_found, prediction_error = _load_predictions(
            spec, config
        )
        if prediction_file_found:
            prediction_files_found += 1
        if prediction_error:
            prediction_file_errors += 1
            warnings.append(f"paper {spec.paper_index:03d}: {prediction_error}")

        valid_predictions = [
            item
            for item in predictions
            if item.bbox is not None and not item.invalid_reason
        ]
        for item in valid_predictions:
            coordinate_spaces[item.coordinate_space] += 1
        matches = _spatial_matches(spec.figures, valid_predictions)
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        paper_detection_tp = 0
        paper_usable_tp = 0

        for gt_index, prediction_index, iou in matches:
            gt = spec.figures[gt_index]
            prediction = valid_predictions[prediction_index]
            matched_gt.add(gt_index)
            matched_predictions.add(prediction_index)
            if prediction.bbox is None:
                raise AssertionError("valid prediction has no bbox")
            intersection = gt.bbox.intersection(prediction.bbox)
            coverage = _safe_div(intersection, gt.bbox.area)
            contamination = 1.0 - _safe_div(
                intersection, prediction.bbox.area
            )
            detection_success = iou >= config.detection_iou
            usable = (
                coverage >= config.coverage_threshold
                and contamination <= config.contamination_threshold
                and prediction.asset_exists
            )
            if detection_success:
                paper_detection_tp += 1
            if usable:
                paper_usable_tp += 1
            all_ious.append(iou)
            all_coverages.append(coverage)
            all_contaminations.append(contamination)

            reasons: list[str] = []
            if not detection_success:
                reasons.append("low_iou")
            if coverage < config.coverage_threshold:
                reasons.append("truncated")
            if contamination > config.contamination_threshold:
                reasons.append("contaminated")
            if not prediction.asset_exists:
                reasons.append("missing_asset")
            per_figure_rows.append(
                _matched_row(
                    spec,
                    gt,
                    prediction,
                    iou,
                    coverage,
                    contamination,
                    detection_success,
                    usable,
                    reasons,
                )
            )

        for gt_index, gt in enumerate(spec.figures):
            if gt_index not in matched_gt:
                per_figure_rows.append(_missed_row(spec, gt))

        valid_index_by_prediction_index = {
            item.prediction_index: index
            for index, item in enumerate(valid_predictions)
        }
        for prediction in predictions:
            valid_index = valid_index_by_prediction_index.get(
                prediction.prediction_index
            )
            if valid_index is not None and valid_index in matched_predictions:
                continue
            per_figure_rows.append(
                _unmatched_prediction_row(spec, prediction)
            )

        paper_label_tp = _label_true_positives(spec.figures, predictions)
        paper = _paper_metrics(
            len(spec.figures),
            len(predictions),
            paper_detection_tp,
            paper_usable_tp,
            paper_label_tp,
        )
        per_paper_rows.append(
            {
                "paper_index": spec.paper_index,
                "paper_title": spec.paper_title,
                "prediction_file_found": prediction_file_found,
                "valid_geometry_count": len(valid_predictions),
                "invalid_prediction_count": len(predictions)
                - len(valid_predictions),
                **paper,
            }
        )

        total_gt += len(spec.figures)
        total_predictions += len(predictions)
        total_valid_geometry += len(valid_predictions)
        total_existing_assets += sum(item.asset_exists for item in predictions)
        total_detection_tp += paper_detection_tp
        total_usable_tp += paper_usable_tp
        total_label_tp += paper_label_tp

    detection_precision = _safe_div(total_detection_tp, total_predictions)
    detection_recall = _safe_div(total_detection_tp, total_gt)
    usable_precision = _safe_div(total_usable_tp, total_predictions)
    usable_recall = _safe_div(total_usable_tp, total_gt)
    label_precision = _safe_div(total_label_tp, total_predictions)
    label_recall = _safe_div(total_label_tp, total_gt)

    macro_fields = (
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "usable_precision",
        "usable_recall",
        "usable_f1",
        "label_precision",
        "label_recall",
        "label_f1",
    )
    macro = {
        field: (
            mean(float(row[field]) for row in per_paper_rows)
            if per_paper_rows
            else 0.0
        )
        for field in macro_fields
    }

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": config.method,
        "paths": {
            "gt_root": str(config.gt_root),
            "dataset_root": str(config.dataset_root),
            "prediction_root": str(config.prediction_root),
            "prediction_json": config.prediction_json,
            "output_dir": str(config.output_dir),
        },
        "config": {
            "scope": config.scope,
            "prediction_space": config.prediction_space,
            "prediction_page_base": config.prediction_page_base,
            "detection_iou": config.detection_iou,
            "coverage_threshold": config.coverage_threshold,
            "contamination_threshold": config.contamination_threshold,
        },
        "dataset": {
            "paper_count": len(specs),
            "gt_figure_count": total_gt,
            "missing_gt_image_count": missing_gt_images,
            "prediction_file_count": prediction_files_found,
            "prediction_file_error_count": prediction_file_errors,
            "prediction_count": total_predictions,
            "valid_geometry_count": total_valid_geometry,
            "invalid_geometry_count": total_predictions
            - total_valid_geometry,
            "geometry_evaluable_fraction": _safe_div(
                total_valid_geometry, total_predictions
            ),
            "existing_asset_count": total_existing_assets,
            "coordinate_spaces": dict(sorted(coordinate_spaces.items())),
        },
        "metrics": {
            "spatial_detection": {
                "true_positives": total_detection_tp,
                "false_positives": total_predictions - total_detection_tp,
                "false_negatives": total_gt - total_detection_tp,
                "precision": detection_precision,
                "recall": detection_recall,
                "f1": _f1(detection_precision, detection_recall),
            },
            "usable_crop": {
                "true_positives": total_usable_tp,
                "false_positives": total_predictions - total_usable_tp,
                "false_negatives": total_gt - total_usable_tp,
                "precision": usable_precision,
                "recall": usable_recall,
                "f1": _f1(usable_precision, usable_recall),
            },
            "label_detection": {
                "true_positives": total_label_tp,
                "false_positives": total_predictions - total_label_tp,
                "false_negatives": total_gt - total_label_tp,
                "precision": label_precision,
                "recall": label_recall,
                "f1": _f1(label_precision, label_recall),
            },
            "geometry": {
                "overlapping_match_count": len(all_ious),
                "iou": _summary_stats(all_ious),
                "coverage": _summary_stats(all_coverages),
                "contamination": _summary_stats(all_contaminations),
                "mean_coverage_over_all_gt": _safe_div(
                    sum(all_coverages), total_gt
                ),
                "mean_contamination_over_all_predictions": _safe_div(
                    sum(all_contaminations)
                    + total_predictions
                    - len(all_contaminations),
                    total_predictions,
                ),
            },
            "macro_per_paper": macro,
        },
        "warnings": warnings,
    }

    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, config.output_dir / "summary.md")
    _write_csv(
        config.output_dir / "per_paper.csv",
        per_paper_rows,
        list(per_paper_rows[0].keys()) if per_paper_rows else [],
    )
    _write_csv(
        config.output_dir / "per_figure.csv",
        per_figure_rows,
        FIGURE_FIELDS,
    )
    return summary
