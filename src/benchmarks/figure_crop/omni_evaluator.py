"""Evaluate final Figure assets with OmniDocBench's COCODet protocol.

OmniDocBench layout detection expects page-space bounding boxes, categories,
and confidence scores.  Some asset extractors retain that geometry while
others only retain final crop images.  This module converts native boxes when
available and otherwise registers a crop back to its source PDF page without
consulting benchmark ground truth.  The resulting predictions are evaluated
with the same ``mmeval.COCODetection`` class used by OmniDocBench v1.5.

Inputs:
  - ``dataset/<paper>/paper.pdf`` source documents;
  - ``results/figure-caption-pairs/*/pairs_with_context.json`` derived Figure
    annotations on a normalized 1000 x 1000 page canvas;
  - one heterogeneous asset result root supported by ``image_evaluator``.

Outputs:
  - Omni simple-format prediction JSON;
  - Omni-style ground-truth JSON;
  - per-asset geometry/localization CSV;
  - official COCODet mAP/mAR values in ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import fitz
import numpy as np

from .image_evaluator import (
    FIGURE_TYPES,
    ImageSource,
    PredictionCrop,
    _dataset_dirs,
    _load_predictions,
    _manifest_entries,
    _normalize_label,
    _read_json,
    _scope_accepts,
    _zip_content_list,
)


SCHEMA_VERSION = "dp-p2p.omnidocbench-figure-crop.v1"
OMNI_PROTOCOL = "OmniDocBench-v1.5-layout-detection-COCODet"
PAGE_CANVAS = 1000.0


@dataclass(frozen=True)
class NativeGeometry:
    """Geometry retained by the method under test."""

    page: int
    bbox: tuple[float, float, float, float]
    coordinate_space: str
    page_width: float | None = None
    page_height: float | None = None


@dataclass(frozen=True)
class FeatureImage:
    """SIFT features and dimensions for a crop or rendered PDF page."""

    width: int
    height: int
    keypoints: tuple[Any, ...]
    descriptors: np.ndarray | None
    error: str = ""


@dataclass(frozen=True)
class Localization:
    """A crop boundary projected onto a normalized PDF page."""

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    score: float = 0.0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    source: str = "failed"
    reason: str = ""


@dataclass(frozen=True)
class OmniEvaluationConfig:
    """Configuration for the derived Figure-only Omni evaluation."""

    gt_root: Path
    dataset_root: Path
    prediction_root: Path
    output_dir: Path
    method: str
    prediction_format: str = "auto"
    scope: str = "body"
    geometry_source: str = "auto"
    render_max_dim: int = 1600
    crop_max_dim: int = 1600
    max_features: int = 4000
    ratio_test: float = 0.75
    min_inliers: int = 6
    min_inlier_ratio: float = 0.25
    ransac_threshold: float = 5.0

    def validate(self) -> None:
        if self.prediction_format not in {
            "auto",
            "bundle",
            "marker",
            "mineru",
            "p2p-doclayout",
        }:
            raise ValueError(f"unsupported prediction format: {self.prediction_format}")
        if self.scope not in {"body", "appendix", "all"}:
            raise ValueError(f"unsupported scope: {self.scope}")
        if self.geometry_source not in {"auto", "native", "register"}:
            raise ValueError(f"unsupported geometry source: {self.geometry_source}")
        if self.render_max_dim < 512 or self.crop_max_dim < 128:
            raise ValueError("image dimensions are too small for PDF registration")
        if self.max_features < 500:
            raise ValueError("max_features must be at least 500")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least four")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be between zero and one")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between zero and one")


def _as_box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        return None
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _clip_box(
    box: Sequence[float],
    width: float = PAGE_CANVAS,
    height: float = PAGE_CANVAS,
) -> tuple[float, float, float, float] | None:
    x0 = min(width, max(0.0, float(box[0])))
    y0 = min(height, max(0.0, float(box[1])))
    x1 = min(width, max(0.0, float(box[2])))
    y1 = min(height, max(0.0, float(box[3])))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "paper_index",
        "paper_title",
        "prediction_id",
        "prediction_label",
        "source_type",
        "page_hint",
        "page",
        "bbox_x0",
        "bbox_y0",
        "bbox_x1",
        "bbox_y1",
        "geometry_source",
        "locator_score",
        "good_matches",
        "inliers",
        "inlier_ratio",
        "status",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _page_name(paper_index: int, page: int) -> str:
    return f"paper_{paper_index:03d}.pdf_{page - 1}"


def _gt_documents(
    config: OmniEvaluationConfig,
    datasets: Sequence[Path],
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    paths = sorted(config.gt_root.glob("*/pairs_with_context.json"))
    if not paths:
        raise FileNotFoundError(f"no GT manifests under {config.gt_root}")
    by_paper: dict[int, list[dict[str, Any]]] = {}
    total = 0
    for path in paths:
        document = _read_json(path)
        paper_index = document.get("paper_index")
        if isinstance(paper_index, bool) or not isinstance(paper_index, int):
            raise ValueError(f"invalid paper_index in {path}")
        if not 0 <= paper_index < len(datasets):
            raise ValueError(f"paper_index {paper_index} outside dataset")
        figures: list[dict[str, Any]] = []
        for pair in document.get("pairs", []):
            if not isinstance(pair, dict):
                continue
            if not _scope_accepts(bool(pair.get("in_appendix", False)), config.scope):
                continue
            page = pair.get("page")
            bbox = _as_box(pair.get("bbox"))
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ValueError(f"invalid GT page in {path}")
            if bbox is None:
                raise ValueError(f"invalid GT bbox in {path}")
            clipped = _clip_box(bbox)
            if clipped is None:
                raise ValueError(f"GT bbox outside page canvas in {path}")
            figures.append(
                {
                    "page": page,
                    "bbox": clipped,
                    "figure_id": str(pair.get("figure_id") or ""),
                    "label": str(pair.get("caption_label") or ""),
                }
            )
        by_paper[paper_index] = figures
        total += len(figures)
    return by_paper, total


def _native_geometries(
    prediction_format: str,
    paper_dir: Path,
) -> list[NativeGeometry | None]:
    """Load native boxes in exactly the same order as image predictions."""

    if prediction_format == "bundle":
        path = paper_dir / "assets" / "meta" / "figures.json"
        if not path.is_file():
            return []
        geometries: list[NativeGeometry | None] = []
        for _, entry in _manifest_entries(_read_json(path)):
            page = entry.get("page")
            bbox = _as_box(entry.get("bbox"))
            if (
                bbox is None
                or isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                geometries.append(None)
                continue
            declared = str(
                entry.get("bbox_space") or entry.get("coordinate_space") or "pdf"
            )
            page_width = entry.get("page_width")
            page_height = entry.get("page_height")
            geometries.append(
                NativeGeometry(
                    page=page,
                    bbox=bbox,
                    coordinate_space=declared,
                    page_width=(
                        float(page_width)
                        if isinstance(page_width, (int, float))
                        and not isinstance(page_width, bool)
                        else None
                    ),
                    page_height=(
                        float(page_height)
                        if isinstance(page_height, (int, float))
                        and not isinstance(page_height, bool)
                        else None
                    ),
                )
            )
        return geometries

    if prediction_format == "marker":
        path = paper_dir / "assets" / "figures.json"
        if not path.is_file():
            return []
        geometries = []
        for _, entry in _manifest_entries(_read_json(path)):
            page_index = entry.get("page_idx")
            bbox = _as_box(entry.get("bbox"))
            if (
                bbox is None
                or isinstance(page_index, bool)
                or not isinstance(page_index, int)
                or page_index < 0
            ):
                geometries.append(None)
            else:
                geometries.append(
                    NativeGeometry(
                        page=page_index + 1,
                        bbox=bbox,
                        coordinate_space="normalized1000",
                    )
                )
        return geometries

    if prediction_format == "mineru":
        archive_path = paper_dir / "mineru_result.zip"
        if not archive_path.is_file():
            return []
        geometries = []
        with zipfile.ZipFile(archive_path) as archive:
            member_root, blocks = _zip_content_list(archive)
            names = set(archive.namelist())
            for block in blocks:
                block_type = str(block.get("type") or "")
                image_path = str(block.get("img_path") or "")
                if block_type not in FIGURE_TYPES or not image_path:
                    continue
                if str(Path(member_root) / image_path) not in names:
                    continue
                page_index = block.get("page_idx")
                bbox = _as_box(block.get("bbox"))
                if (
                    bbox is None
                    or isinstance(page_index, bool)
                    or not isinstance(page_index, int)
                    or page_index < 0
                ):
                    geometries.append(None)
                else:
                    geometries.append(
                        NativeGeometry(
                            page=page_index + 1,
                            bbox=bbox,
                            coordinate_space="normalized1000",
                        )
                    )
        return geometries

    if prediction_format == "p2p-doclayout":
        path = paper_dir / "paper_figures.json"
        if not path.is_file():
            return []
        document = _read_json(path)
        return [None for value in document if isinstance(value, str)]

    raise ValueError(f"unsupported prediction format: {prediction_format}")


def _normalize_native_geometry(
    geometry: NativeGeometry,
    pdf: fitz.Document,
) -> Localization:
    if not 1 <= geometry.page <= len(pdf):
        return Localization(reason=f"native page {geometry.page} outside PDF")
    x0, y0, x1, y1 = geometry.bbox
    if geometry.coordinate_space == "normalized1000":
        normalized = geometry.bbox
    elif geometry.coordinate_space == "page_pixels":
        if not geometry.page_width or not geometry.page_height:
            return Localization(reason="page_pixels geometry lacks page dimensions")
        normalized = (
            x0 * PAGE_CANVAS / geometry.page_width,
            y0 * PAGE_CANVAS / geometry.page_height,
            x1 * PAGE_CANVAS / geometry.page_width,
            y1 * PAGE_CANVAS / geometry.page_height,
        )
    elif geometry.coordinate_space == "pdf":
        rect = pdf[geometry.page - 1].rect
        normalized = (
            x0 * PAGE_CANVAS / rect.width,
            y0 * PAGE_CANVAS / rect.height,
            x1 * PAGE_CANVAS / rect.width,
            y1 * PAGE_CANVAS / rect.height,
        )
    else:
        return Localization(reason=f"unsupported native space {geometry.coordinate_space}")
    clipped = _clip_box(normalized)
    if clipped is None:
        return Localization(reason="native bbox outside normalized page")
    return Localization(
        page=geometry.page,
        bbox=clipped,
        score=1.0,
        source=f"native:{geometry.coordinate_space}",
    )


def _sift_feature(image: np.ndarray, max_features: int) -> FeatureImage:
    if image.size == 0:
        return FeatureImage(0, 0, (), None, "empty image")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return FeatureImage(
        width=width,
        height=height,
        keypoints=tuple(keypoints or ()),
        descriptors=descriptors,
    )


def _crop_feature(
    source: ImageSource,
    config: OmniEvaluationConfig,
) -> FeatureImage:
    try:
        encoded = np.frombuffer(source.read(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError("image decoder returned no pixels")
        height, width = image.shape
        scale = min(1.0, config.crop_max_dim / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return _sift_feature(image, config.max_features)
    except Exception as error:  # noqa: BLE001 - recorded in benchmark output
        return FeatureImage(0, 0, (), None, f"{type(error).__name__}: {error}")


class PdfFeatureIndex:
    """Lazy rendered-page text and SIFT feature index for one PDF."""

    def __init__(self, pdf: fitz.Document, config: OmniEvaluationConfig) -> None:
        self.pdf = pdf
        self.config = config
        self._features: dict[int, FeatureImage] = {}
        self._texts: dict[int, str] = {}

    def text(self, page: int) -> str:
        if page not in self._texts:
            self._texts[page] = self.pdf[page - 1].get_text("text")
        return self._texts[page]

    def feature(self, page: int) -> FeatureImage:
        if page in self._features:
            return self._features[page]
        pdf_page = self.pdf[page - 1]
        rect = pdf_page.rect
        zoom = self.config.render_max_dim / max(rect.width, rect.height)
        pixmap = pdf_page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            colorspace=fitz.csGRAY,
            alpha=False,
        )
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
        )
        feature = _sift_feature(image, self.config.max_features)
        self._features[page] = feature
        return feature


def _candidate_pages(
    prediction: PredictionCrop,
    index: PdfFeatureIndex,
) -> list[int]:
    page_count = len(index.pdf)
    candidates: list[int] = []
    if (
        prediction.page is not None
        and not isinstance(prediction.page, bool)
        and 1 <= prediction.page <= page_count
    ):
        candidates.append(prediction.page)
    label = _normalize_label(prediction.label)
    if label:
        escaped = re.escape(label)
        pattern = re.compile(
            rf"\bfig(?:ure)?\.?\s*{escaped}(?![\d.])",
            flags=re.IGNORECASE,
        )
        for page in range(1, page_count + 1):
            if pattern.search(index.text(page)) and page not in candidates:
                candidates.append(page)
    return candidates


def _register_to_page(
    crop: FeatureImage,
    page: FeatureImage,
    page_number: int,
    config: OmniEvaluationConfig,
) -> Localization:
    if crop.error or page.error:
        return Localization(reason=crop.error or page.error)
    if crop.descriptors is None or page.descriptors is None:
        return Localization(reason="missing SIFT descriptors")
    if len(crop.keypoints) < 4 or len(page.keypoints) < 4:
        return Localization(reason="too few SIFT keypoints")
    try:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            crop.descriptors,
            page.descriptors,
            k=2,
        )
    except cv2.error as error:
        return Localization(reason=f"SIFT matcher failed: {error}")
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < config.ratio_test * pair[1].distance
    ]
    if len(good) < config.min_inliers:
        return Localization(
            good_matches=len(good),
            reason="too few ratio-test matches",
        )
    source_points = np.float32(
        [crop.keypoints[item.queryIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [page.keypoints[item.trainIdx].pt for item in good]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        config.ransac_threshold,
    )
    if homography is None or mask is None:
        return Localization(good_matches=len(good), reason="homography failed")
    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / len(good)
    if inliers < config.min_inliers:
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="too few homography inliers",
        )
    if inlier_ratio < config.min_inlier_ratio:
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="homography inlier ratio too small",
        )
    boundary = np.float32(
        [
            [0, 0],
            [crop.width, 0],
            [crop.width, crop.height],
            [0, crop.height],
        ]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(boundary, homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="non-finite projected boundary",
        )
    if not cv2.isContourConvex(projected.astype(np.float32)):
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="non-convex projected boundary",
        )
    projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    page_area = float(page.width * page.height)
    area_ratio = projected_area / page_area if page_area else 0.0
    if not 1e-5 <= area_ratio <= 1.25:
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="implausible projected area",
        )
    x0 = float(np.min(projected[:, 0])) * PAGE_CANVAS / page.width
    y0 = float(np.min(projected[:, 1])) * PAGE_CANVAS / page.height
    x1 = float(np.max(projected[:, 0])) * PAGE_CANVAS / page.width
    y1 = float(np.max(projected[:, 1])) * PAGE_CANVAS / page.height
    bbox = _clip_box((x0, y0, x1, y1))
    if bbox is None:
        return Localization(
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            reason="projected crop lies outside PDF page",
        )
    score = inlier_ratio * min(1.0, inliers / 20.0)
    return Localization(
        page=page_number,
        bbox=bbox,
        score=score,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        source="pdf-registration",
    )


def _best_localization(
    prediction: PredictionCrop,
    index: PdfFeatureIndex,
    config: OmniEvaluationConfig,
) -> Localization:
    crop = _crop_feature(prediction.image, config)
    if crop.error:
        return Localization(reason=crop.error)
    candidates = _candidate_pages(prediction, index)
    searched: set[int] = set()
    best = Localization(reason="no valid PDF registration")

    def search(pages: Iterable[int]) -> None:
        nonlocal best
        for page in pages:
            if page in searched:
                continue
            searched.add(page)
            registration = _register_to_page(
                crop,
                index.feature(page),
                page,
                config,
            )
            if registration.bbox is not None and registration.score > best.score:
                best = registration

    search(candidates)
    if best.bbox is None:
        search(range(1, len(index.pdf) + 1))
    return best


def _official_coco_metric(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from mmeval import COCODetection
    except ImportError as error:
        raise RuntimeError(
            "Omni COCODet dependencies are missing; install "
            "src/benchmark/figure_crop/requirements-omni.txt"
        ) from error
    metric = COCODetection(
        dataset_meta={"CLASSES": ("figure",)},
        metric=["bbox"],
        classwise=True,
    )
    return _jsonable(metric(predictions=predictions, groundtruths=ground_truths))


def evaluate_omni_figure_crops(config: OmniEvaluationConfig) -> dict[str, Any]:
    """Convert assets and run OmniDocBench's official COCODet metric."""

    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _dataset_dirs(config.dataset_root)
    gt_by_paper, total_gt = _gt_documents(config, datasets)

    metric_ground_truths: list[dict[str, Any]] = []
    metric_predictions: list[dict[str, Any]] = []
    omni_ground_truth: list[dict[str, Any]] = []
    omni_results: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    geometry_sources: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    prediction_count = 0
    localized_count = 0
    failed_count = 0
    page_record_index = 0

    for paper_index in sorted(gt_by_paper):
        paper_dir = datasets[paper_index]
        paper_title = paper_dir.name
        prediction_format, predictions = _load_predictions(config, paper_title)
        native = _native_geometries(prediction_format, config.prediction_root / paper_title)
        if len(native) != len(predictions):
            raise ValueError(
                f"native/prediction length mismatch for {paper_title}: "
                f"{len(native)} != {len(predictions)}"
            )
        formats[prediction_format] += 1
        figures_by_page: dict[int, list[dict[str, Any]]] = {}
        for figure in gt_by_paper[paper_index]:
            figures_by_page.setdefault(int(figure["page"]), []).append(figure)

        with fitz.open(paper_dir / "paper.pdf") as pdf:
            page_to_metric_index: dict[int, int] = {}
            for page in range(1, len(pdf) + 1):
                page_to_metric_index[page] = page_record_index
                page_figures = figures_by_page.get(page, [])
                boxes = np.asarray(
                    [figure["bbox"] for figure in page_figures],
                    dtype=np.float32,
                ).reshape(-1, 4)
                labels = np.zeros(len(page_figures), dtype=np.int64)
                metric_ground_truths.append(
                    {
                        "img_id": page_record_index,
                        "width": int(PAGE_CANVAS),
                        "height": int(PAGE_CANVAS),
                        "bboxes": boxes,
                        "labels": labels,
                        "ignore_flags": np.zeros(len(page_figures), dtype=bool),
                    }
                )
                layout_dets = []
                for figure in page_figures:
                    x0, y0, x1, y1 = figure["bbox"]
                    layout_dets.append(
                        {
                            "category_type": "figure",
                            "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
                            "ignore": False,
                            "anno_id": figure["figure_id"],
                        }
                    )
                omni_ground_truth.append(
                    {
                        "page_info": {
                            "page_no": page - 1,
                            "image_path": _page_name(paper_index, page) + ".jpg",
                            "width": int(PAGE_CANVAS),
                            "height": int(PAGE_CANVAS),
                            "page_attribute": {
                                "data_source": "academic_literature",
                                "derived_subset": "figure_crop",
                            },
                        },
                        "layout_dets": layout_dets,
                    }
                )
                page_record_index += 1

            per_page_predictions: dict[int, list[tuple[Localization, int]]] = {
                page: [] for page in range(1, len(pdf) + 1)
            }
            page_index: PdfFeatureIndex | None = None
            for prediction_index, prediction in enumerate(predictions):
                prediction_count += 1
                localization = Localization(reason="native geometry unavailable")
                if config.geometry_source in {"auto", "native"} and native[prediction_index]:
                    localization = _normalize_native_geometry(
                        native[prediction_index],  # type: ignore[arg-type]
                        pdf,
                    )
                if (
                    localization.bbox is None
                    and config.geometry_source in {"auto", "register"}
                ):
                    if page_index is None:
                        page_index = PdfFeatureIndex(pdf, config)
                    localization = _best_localization(prediction, page_index, config)

                if localization.bbox is not None and localization.page is not None:
                    localized_count += 1
                    status = "localized"
                    effective = localization
                else:
                    failed_count += 1
                    status = "failed_dummy_fp"
                    fallback_page = (
                        prediction.page
                        if prediction.page is not None
                        and not isinstance(prediction.page, bool)
                        and 1 <= prediction.page <= len(pdf)
                        else 1
                    )
                    effective = Localization(
                        page=fallback_page,
                        bbox=(0.0, 0.0, 1.0, 1.0),
                        source="failed-dummy",
                        reason=localization.reason,
                    )
                geometry_sources[effective.source] += 1
                per_page_predictions[effective.page].append(
                    (effective, prediction_index)  # type: ignore[index]
                )
                x0, y0, x1, y1 = effective.bbox  # type: ignore[misc]
                asset_rows.append(
                    {
                        "paper_index": paper_index,
                        "paper_title": paper_title,
                        "prediction_id": prediction.prediction_id,
                        "prediction_label": prediction.label,
                        "source_type": prediction.source_type,
                        "page_hint": prediction.page if prediction.page is not None else "",
                        "page": effective.page,
                        "bbox_x0": x0,
                        "bbox_y0": y0,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "geometry_source": effective.source,
                        "locator_score": localization.score,
                        "good_matches": localization.good_matches,
                        "inliers": localization.inliers,
                        "inlier_ratio": localization.inlier_ratio,
                        "status": status,
                        "reason": localization.reason,
                    }
                )

            for page in range(1, len(pdf) + 1):
                records = per_page_predictions[page]
                boxes = np.asarray(
                    [record[0].bbox for record in records],
                    dtype=np.float32,
                ).reshape(-1, 4)
                scores = np.ones(len(records), dtype=np.float32)
                labels = np.zeros(len(records), dtype=np.int64)
                metric_predictions.append(
                    {
                        "img_id": page_to_metric_index[page],
                        "bboxes": boxes,
                        "scores": scores,
                        "labels": labels,
                    }
                )
                for localization, _ in records:
                    omni_results.append(
                        {
                            "image_name": _page_name(paper_index, page),
                            "bbox": list(localization.bbox or (0, 0, 1, 1)),
                            "category_id": 0,
                            # Asset pipelines do not expose calibrated detector
                            # confidences.  A tied score is the only method-neutral
                            # bridge to Omni's required prediction schema.
                            "score": 1.0,
                        }
                    )

    official_metrics = _official_coco_metric(metric_predictions, metric_ground_truths)
    prediction_document = {
        "results": omni_results,
        "categories": {"0": "figure"},
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": config.method,
        "protocol": {
            "name": OMNI_PROTOCOL,
            "metric_implementation": "mmeval.COCODetection",
            "categories": ["figure"],
            "prediction_score_policy": "constant-1.0-no-native-confidence",
            "geometry_policy": config.geometry_source,
            "gt_canvas": [int(PAGE_CANVAS), int(PAGE_CANVAS)],
        },
        "config": {
            "scope": config.scope,
            "prediction_format": config.prediction_format,
            "render_max_dim": config.render_max_dim,
            "crop_max_dim": config.crop_max_dim,
            "max_features": config.max_features,
            "ratio_test": config.ratio_test,
            "min_inliers": config.min_inliers,
            "min_inlier_ratio": config.min_inlier_ratio,
            "ransac_threshold": config.ransac_threshold,
        },
        "dataset": {
            "paper_count": len(gt_by_paper),
            "page_count": len(metric_ground_truths),
            "gt_figure_count": total_gt,
            "prediction_count": prediction_count,
            "formats": dict(formats),
        },
        "geometry": {
            "localized": localized_count,
            "failed": failed_count,
            "localized_fraction": (
                localized_count / prediction_count if prediction_count else 0.0
            ),
            "sources": dict(geometry_sources),
        },
        "metrics": official_metrics,
    }
    _write_json(config.output_dir / "omni_ground_truth.json", omni_ground_truth)
    _write_json(config.output_dir / "omni_prediction.json", prediction_document)
    _write_json(config.output_dir / "summary.json", summary)
    _write_csv(config.output_dir / "per_asset_geometry.csv", asset_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate final Figure crops with OmniDocBench COCODet",
    )
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--prediction-format",
        choices=["auto", "bundle", "marker", "mineru", "p2p-doclayout"],
        default="auto",
    )
    parser.add_argument(
        "--geometry-source",
        choices=["auto", "native", "register"],
        default="auto",
        help="auto uses native boxes first and PDF registration otherwise",
    )
    parser.add_argument("--gt-root", type=Path, default=Path("results/figure-caption-pairs"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--scope", choices=["body", "appendix", "all"], default="body")
    parser.add_argument("--render-max-dim", type=int, default=1600)
    parser.add_argument("--crop-max-dim", type=int, default=1600)
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = OmniEvaluationConfig(
        gt_root=args.gt_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        method=args.method,
        prediction_format=args.prediction_format,
        scope=args.scope,
        geometry_source=args.geometry_source,
        render_max_dim=args.render_max_dim,
        crop_max_dim=args.crop_max_dim,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        ransac_threshold=args.ransac_threshold,
    )
    summary = evaluate_omni_figure_crops(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
