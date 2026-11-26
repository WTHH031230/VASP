"""Evaluate final Figure crop images across heterogeneous baseline formats.

The derived GT bundle contains one image crop per logical Figure.  Baselines do
not share a geometry schema: some retain PDF boxes, some only retain cropped
images, and Research Studio may use original arXiv graphics.  This evaluator
therefore registers each predicted image directly to the GT crop with SIFT and
a robust homography.  The projected prediction boundary provides comparable
content coverage and contamination measurements without requiring a saved PDF
box from the method under test.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import cv2
import numpy as np

from .evaluator import _maximum_weight_matching


SCHEMA_VERSION = "dp-p2p.figure-crop-image-benchmark.v1"
FIGURE_TYPES = {"image", "chart"}


@dataclass(frozen=True)
class ImageSource:
    """Lazy image bytes from a file, zip member, or base64 payload."""

    kind: str
    value: str
    member: str = ""

    def read(self) -> bytes:
        if self.kind == "file":
            return Path(self.value).read_bytes()
        if self.kind == "zip":
            with zipfile.ZipFile(self.value) as archive:
                return archive.read(self.member)
        if self.kind == "base64":
            return base64.b64decode(self.value, validate=True)
        raise ValueError(f"unsupported image source: {self.kind}")


@dataclass(frozen=True)
class GroundTruthCrop:
    paper_index: int
    paper_title: str
    figure_id: str
    label: str
    page: int
    image_path: Path
    in_appendix: bool


@dataclass(frozen=True)
class PredictionCrop:
    prediction_id: str
    label: str
    page: int | None
    source_type: str
    image: ImageSource


@dataclass(frozen=True)
class ImageFeature:
    width: int
    height: int
    keypoints: tuple[Any, ...]
    descriptors: np.ndarray | None
    error: str = ""


@dataclass(frozen=True)
class Registration:
    score: float = 0.0
    base_confidence: float = 0.0
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    coverage: float = 0.0
    contamination: float = 1.0
    projected_area_ratio: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ImageEvaluationConfig:
    gt_root: Path
    dataset_root: Path
    prediction_root: Path
    output_dir: Path
    method: str
    prediction_format: str = "auto"
    scope: str = "body"
    max_image_dim: int = 900
    max_features: int = 2500
    ratio_test: float = 0.75
    min_inliers: int = 6
    match_threshold: float = 0.15
    coverage_threshold: float = 0.95
    contamination_threshold: float = 0.10

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
        if self.max_image_dim < 128 or self.max_features < 100:
            raise ValueError("image feature limits are too small")
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least 4")
        for name, value in (
            ("ratio_test", self.ratio_test),
            ("match_threshold", self.match_threshold),
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


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _normalize_label(value: str) -> str:
    text = " ".join(str(value or "").split()).casefold()
    match = re.search(
        r"\bfig(?:ure)?\.?\s*([a-z]*\d+(?:\.\d+)*(?:[a-z])?)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).casefold() if match else ""


def _scope_accepts(in_appendix: bool, scope: str) -> bool:
    return scope == "all" or (scope == "appendix") == in_appendix


def _dataset_dirs(dataset_root: Path) -> list[Path]:
    directories = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and (path / "paper.pdf").is_file()
    )
    if not directories:
        raise FileNotFoundError(f"no <paper>/paper.pdf under {dataset_root}")
    return directories


def _load_ground_truth(
    config: ImageEvaluationConfig,
) -> tuple[dict[int, list[GroundTruthCrop]], dict[int, str]]:
    datasets = _dataset_dirs(config.dataset_root)
    by_paper: dict[int, list[GroundTruthCrop]] = {}
    titles: dict[int, str] = {}
    paths = sorted(config.gt_root.glob("*/pairs_with_context.json"))
    if not paths:
        raise FileNotFoundError(f"no GT manifests under {config.gt_root}")
    for path in paths:
        document = _read_json(path)
        paper_index = document.get("paper_index")
        if isinstance(paper_index, bool) or not isinstance(paper_index, int):
            raise ValueError(f"invalid paper_index in {path}")
        if not 0 <= paper_index < len(datasets):
            raise ValueError(f"paper_index {paper_index} outside dataset")
        title = datasets[paper_index].name
        titles[paper_index] = title
        figures: list[GroundTruthCrop] = []
        for offset, pair in enumerate(document.get("pairs", [])):
            in_appendix = bool(pair.get("in_appendix", False))
            if not _scope_accepts(in_appendix, config.scope):
                continue
            image_path = path.parent / str(pair.get("figure_path") or "")
            if not image_path.is_file():
                raise FileNotFoundError(f"missing GT crop: {image_path}")
            page = pair.get("page")
            if isinstance(page, bool) or not isinstance(page, int):
                raise ValueError(f"invalid GT page in {path}")
            figures.append(
                GroundTruthCrop(
                    paper_index=paper_index,
                    paper_title=title,
                    figure_id=str(pair.get("figure_id") or f"figure_{offset:03d}"),
                    label=str(pair.get("caption_label") or ""),
                    page=page,
                    image_path=image_path,
                    in_appendix=in_appendix,
                )
            )
        by_paper[paper_index] = figures
    return by_paper, titles


def _resolve_asset(paper_dir: Path, raw_path: str) -> Path:
    value = Path(raw_path)
    candidates = [
        paper_dir / value,
        paper_dir / "assets" / value,
        paper_dir / "assets" / "figures" / value.name,
        paper_dir / "assets" / value.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _manifest_entries(document: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(document, list):
        return [(str(index), value) for index, value in enumerate(document) if isinstance(value, dict)]
    if isinstance(document, dict):
        return [(str(key), value) for key, value in document.items() if isinstance(value, dict)]
    return []


def _load_bundle(paper_dir: Path) -> list[PredictionCrop]:
    manifest = paper_dir / "assets" / "meta" / "figures.json"
    if not manifest.is_file():
        return []
    predictions: list[PredictionCrop] = []
    for key, entry in _manifest_entries(_read_json(manifest)):
        raw_path = str(entry.get("file") or entry.get("path") or "")
        asset = _resolve_asset(paper_dir, raw_path)
        page = entry.get("page")
        if isinstance(page, bool) or not isinstance(page, int):
            page = None
        label = str(
            entry.get("caption_label")
            or entry.get("label")
            or entry.get("caption")
            or ""
        )
        predictions.append(
            PredictionCrop(
                prediction_id=f"{manifest.name}:{key}",
                label=label,
                page=page,
                source_type=str(entry.get("source") or "bundle"),
                image=ImageSource("file", str(asset)),
            )
        )
    return predictions


def _load_marker(paper_dir: Path) -> list[PredictionCrop]:
    manifest = paper_dir / "assets" / "figures.json"
    if not manifest.is_file():
        return []
    predictions: list[PredictionCrop] = []
    for key, entry in _manifest_entries(_read_json(manifest)):
        raw_path = str(entry.get("path") or entry.get("file") or "")
        asset = _resolve_asset(paper_dir, raw_path)
        predictions.append(
            PredictionCrop(
                prediction_id=f"{manifest.name}:{key}",
                label=str(entry.get("caption") or ""),
                page=None,
                source_type="marker",
                image=ImageSource("file", str(asset)),
            )
        )
    return predictions


def _load_p2p_doclayout(paper_dir: Path) -> list[PredictionCrop]:
    manifest = paper_dir / "paper_figures.json"
    if not manifest.is_file():
        return []
    document = _read_json(manifest)
    if not isinstance(document, list):
        return []
    return [
        PredictionCrop(
            prediction_id=f"{manifest.name}:{index}",
            label="",
            page=None,
            source_type="p2p-doclayout",
            image=ImageSource("base64", value),
        )
        for index, value in enumerate(document)
        if isinstance(value, str)
    ]


def _zip_content_list(archive: zipfile.ZipFile) -> tuple[str, list[dict[str, Any]]]:
    candidates = sorted(
        name
        for name in archive.namelist()
        if name.endswith("_content_list.json") and not name.endswith("_content_list_v2.json")
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one MinerU content list, found {len(candidates)}")
    member = candidates[0]
    document = json.loads(archive.read(member))
    if not isinstance(document, list):
        raise ValueError("MinerU content list must be a list")
    return str(Path(member).parent), [value for value in document if isinstance(value, dict)]


def _load_mineru(paper_dir: Path) -> list[PredictionCrop]:
    archive_path = paper_dir / "mineru_result.zip"
    if not archive_path.is_file():
        return []
    predictions: list[PredictionCrop] = []
    with zipfile.ZipFile(archive_path) as archive:
        member_root, blocks = _zip_content_list(archive)
        names = set(archive.namelist())
        for index, block in enumerate(blocks):
            block_type = str(block.get("type") or "")
            image_path = str(block.get("img_path") or "")
            if block_type not in FIGURE_TYPES or not image_path:
                continue
            member = str(Path(member_root) / image_path)
            if member not in names:
                continue
            captions = block.get("image_caption") or block.get("chart_caption") or []
            if isinstance(captions, str):
                label = captions
            elif isinstance(captions, list):
                label = " ".join(str(value) for value in captions)
            else:
                label = ""
            page = block.get("page_idx")
            page = page + 1 if isinstance(page, int) and not isinstance(page, bool) else None
            predictions.append(
                PredictionCrop(
                    prediction_id=f"{archive_path.name}:{index}",
                    label=label,
                    page=page,
                    source_type=f"mineru-{block_type}",
                    image=ImageSource("zip", str(archive_path), member),
                )
            )
    return predictions


def _detect_format(prediction_root: Path, title: str) -> str:
    paper_dir = prediction_root / title
    if (paper_dir / "assets" / "meta" / "figures.json").is_file():
        return "bundle"
    if (paper_dir / "assets" / "figures.json").is_file():
        return "marker"
    if (paper_dir / "mineru_result.zip").is_file():
        return "mineru"
    if (paper_dir / "paper_figures.json").is_file():
        return "p2p-doclayout"
    return "bundle"


def _load_predictions(
    config: ImageEvaluationConfig,
    title: str,
) -> tuple[str, list[PredictionCrop]]:
    paper_dir = config.prediction_root / title
    prediction_format = config.prediction_format
    if prediction_format == "auto":
        prediction_format = _detect_format(config.prediction_root, title)
    loaders = {
        "bundle": _load_bundle,
        "marker": _load_marker,
        "mineru": _load_mineru,
        "p2p-doclayout": _load_p2p_doclayout,
    }
    return prediction_format, loaders[prediction_format](paper_dir)


def _extract_feature(source: ImageSource, config: ImageEvaluationConfig) -> ImageFeature:
    try:
        array = np.frombuffer(source.read(), dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError("image decoder returned no pixels")
        height, width = image.shape
        scale = min(1.0, config.max_image_dim / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        sift = cv2.SIFT_create(
            nfeatures=config.max_features,
            contrastThreshold=0.02,
        )
        keypoints, descriptors = sift.detectAndCompute(image, None)
        resized_height, resized_width = image.shape
        return ImageFeature(
            width=resized_width,
            height=resized_height,
            keypoints=tuple(keypoints or ()),
            descriptors=descriptors,
        )
    except Exception as error:  # noqa: BLE001 - recorded as benchmark data
        return ImageFeature(0, 0, (), None, f"{type(error).__name__}: {error}")


def _registration(
    ground_truth: ImageFeature,
    prediction: ImageFeature,
    config: ImageEvaluationConfig,
) -> Registration:
    if ground_truth.error or prediction.error:
        return Registration(reason=ground_truth.error or prediction.error)
    if ground_truth.descriptors is None or prediction.descriptors is None:
        return Registration(reason="missing descriptors")
    if len(ground_truth.keypoints) < 4 or len(prediction.keypoints) < 4:
        return Registration(reason="too few keypoints")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        prediction.descriptors,
        ground_truth.descriptors,
        k=2,
    )
    good = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < config.ratio_test * second.distance
    ]
    if len(good) < config.min_inliers:
        return Registration(good_matches=len(good), reason="too few ratio-test matches")
    source_points = np.float32(
        [prediction.keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [ground_truth.keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        4.0,
    )
    if homography is None or mask is None:
        return Registration(good_matches=len(good), reason="homography failed")
    inliers = int(mask.sum())
    if inliers < config.min_inliers:
        return Registration(
            good_matches=len(good),
            inliers=inliers,
            reason="too few homography inliers",
        )
    corners = np.float32(
        [[0, 0], [prediction.width, 0], [prediction.width, prediction.height], [0, prediction.height]]
    ).reshape(1, 4, 2)
    projected = cv2.perspectiveTransform(corners, homography)[0]
    if not np.isfinite(projected).all() or not cv2.isContourConvex(projected):
        return Registration(
            good_matches=len(good),
            inliers=inliers,
            reason="invalid projected boundary",
        )
    projected_area = abs(float(cv2.contourArea(projected)))
    gt_area = float(ground_truth.width * ground_truth.height)
    area_ratio = _safe_div(projected_area, gt_area)
    if not 0.005 <= area_ratio <= 100.0:
        return Registration(
            good_matches=len(good),
            inliers=inliers,
            projected_area_ratio=area_ratio,
            reason="implausible projected area",
        )
    gt_boundary = np.float32(
        [[0, 0], [ground_truth.width, 0], [ground_truth.width, ground_truth.height], [0, ground_truth.height]]
    )
    intersection, _ = cv2.intersectConvexConvex(
        projected.astype(np.float32),
        gt_boundary,
    )
    intersection = max(0.0, float(intersection))
    coverage = min(1.0, _safe_div(intersection, gt_area))
    contamination = min(1.0, max(0.0, 1.0 - _safe_div(intersection, projected_area)))
    inlier_ratio = _safe_div(inliers, len(good))
    base_confidence = inlier_ratio * min(1.0, inliers / 20.0)
    score = base_confidence * math.sqrt(coverage) * math.sqrt(1.0 - contamination)
    return Registration(
        score=score,
        base_confidence=base_confidence,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        coverage=coverage,
        contamination=contamination,
        projected_area_ratio=area_ratio,
    )


def _metric_counts(true_positive: int, predictions: int, targets: int) -> dict[str, float | int]:
    precision = _safe_div(true_positive, predictions)
    recall = _safe_div(true_positive, targets)
    return {
        "true_positive": true_positive,
        "false_positive": predictions - true_positive,
        "false_negative": targets - true_positive,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_image_crops(config: ImageEvaluationConfig) -> dict[str, Any]:
    """Run image-level matching and write JSON/CSV reports."""

    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    gt_by_paper, titles = _load_ground_truth(config)
    rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    formats: Counter[str] = Counter()
    source_types: Counter[str] = Counter()
    total_gt = 0
    total_predictions = 0
    decoded_predictions = 0
    detected = 0
    usable = 0
    coverages: list[float] = []
    contaminations: list[float] = []
    scores: list[float] = []
    inliers: list[float] = []

    for paper_index in sorted(gt_by_paper):
        title = titles[paper_index]
        ground_truths = gt_by_paper[paper_index]
        prediction_format, predictions = _load_predictions(config, title)
        formats[prediction_format] += 1
        source_types.update(prediction.source_type for prediction in predictions)
        gt_features = [
            _extract_feature(ImageSource("file", str(item.image_path)), config)
            for item in ground_truths
        ]
        prediction_features = [
            _extract_feature(item.image, config) for item in predictions
        ]
        total_gt += len(ground_truths)
        total_predictions += len(predictions)
        decoded_predictions += sum(not feature.error for feature in prediction_features)

        registrations: list[list[Registration]] = []
        weights: list[list[float]] = []
        for gt_index, ground_truth in enumerate(ground_truths):
            registration_row: list[Registration] = []
            weight_row: list[float] = []
            gt_label = _normalize_label(ground_truth.label)
            for prediction_index, prediction in enumerate(predictions):
                if prediction.page is not None and prediction.page != ground_truth.page:
                    registration = Registration(reason="page mismatch")
                else:
                    registration = _registration(
                        gt_features[gt_index],
                        prediction_features[prediction_index],
                        config,
                    )
                weight = registration.score
                prediction_label = _normalize_label(prediction.label)
                if weight > 0.0 and gt_label and prediction_label == gt_label:
                    weight += 0.025
                registration_row.append(registration)
                weight_row.append(weight)
            registrations.append(registration_row)
            weights.append(weight_row)

        assigned = _maximum_weight_matching(weights)
        matched_predictions: set[int] = set()
        paper_detected = 0
        paper_usable = 0
        for gt_index, prediction_index in assigned:
            registration = registrations[gt_index][prediction_index]
            if registration.score < config.match_threshold:
                continue
            ground_truth = ground_truths[gt_index]
            prediction = predictions[prediction_index]
            matched_predictions.add(prediction_index)
            is_usable = (
                registration.coverage >= config.coverage_threshold
                and registration.contamination <= config.contamination_threshold
            )
            detected += 1
            paper_detected += 1
            usable += int(is_usable)
            paper_usable += int(is_usable)
            coverages.append(registration.coverage)
            contaminations.append(registration.contamination)
            scores.append(registration.score)
            inliers.append(float(registration.inliers))
            rows.append(
                {
                    "paper_index": paper_index,
                    "paper_title": title,
                    "gt_figure_id": ground_truth.figure_id,
                    "gt_label": ground_truth.label,
                    "gt_page": ground_truth.page,
                    "prediction_id": prediction.prediction_id,
                    "prediction_label": prediction.label,
                    "prediction_page": prediction.page if prediction.page is not None else "",
                    "source_type": prediction.source_type,
                    "score": registration.score,
                    "inliers": registration.inliers,
                    "coverage": registration.coverage,
                    "contamination": registration.contamination,
                    "detected": True,
                    "usable": is_usable,
                    "status": "usable" if is_usable else "matched_low_quality",
                    "reason": registration.reason,
                }
            )

        matched_gt = {
            gt_index
            for gt_index, prediction_index in assigned
            if registrations[gt_index][prediction_index].score >= config.match_threshold
        }
        for gt_index, ground_truth in enumerate(ground_truths):
            if gt_index in matched_gt:
                continue
            rows.append(
                {
                    "paper_index": paper_index,
                    "paper_title": title,
                    "gt_figure_id": ground_truth.figure_id,
                    "gt_label": ground_truth.label,
                    "gt_page": ground_truth.page,
                    "prediction_id": "",
                    "prediction_label": "",
                    "prediction_page": "",
                    "source_type": "",
                    "score": 0.0,
                    "inliers": 0,
                    "coverage": 0.0,
                    "contamination": 1.0,
                    "detected": False,
                    "usable": False,
                    "status": "missed",
                    "reason": "no confident image registration",
                }
            )
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue
            feature = prediction_features[prediction_index]
            rows.append(
                {
                    "paper_index": paper_index,
                    "paper_title": title,
                    "gt_figure_id": "",
                    "gt_label": "",
                    "gt_page": "",
                    "prediction_id": prediction.prediction_id,
                    "prediction_label": prediction.label,
                    "prediction_page": prediction.page if prediction.page is not None else "",
                    "source_type": prediction.source_type,
                    "score": 0.0,
                    "inliers": 0,
                    "coverage": 0.0,
                    "contamination": 1.0,
                    "detected": False,
                    "usable": False,
                    "status": "unreadable" if feature.error else "false_positive",
                    "reason": feature.error or "no confident GT assignment",
                }
            )
        paper_rows.append(
            {
                "paper_index": paper_index,
                "paper_title": title,
                "prediction_format": prediction_format,
                "gt_count": len(ground_truths),
                "prediction_count": len(predictions),
                "decoded_prediction_count": sum(
                    not feature.error for feature in prediction_features
                ),
                "detected_count": paper_detected,
                "usable_count": paper_usable,
                "detection_recall": _safe_div(paper_detected, len(ground_truths)),
                "usable_recall": _safe_div(paper_usable, len(ground_truths)),
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "method": config.method,
        "prediction_format": config.prediction_format,
        "scope": config.scope,
        "config": {
            "max_image_dim": config.max_image_dim,
            "max_features": config.max_features,
            "ratio_test": config.ratio_test,
            "min_inliers": config.min_inliers,
            "match_threshold": config.match_threshold,
            "coverage_threshold": config.coverage_threshold,
            "contamination_threshold": config.contamination_threshold,
        },
        "dataset": {
            "paper_count": len(gt_by_paper),
            "gt_crop_count": total_gt,
            "prediction_count": total_predictions,
            "decoded_prediction_count": decoded_predictions,
            "formats": dict(formats),
            "prediction_source_types": dict(source_types),
        },
        "metrics": {
            "detection": _metric_counts(detected, total_predictions, total_gt),
            "usable_crop": _metric_counts(usable, total_predictions, total_gt),
            "matched_crop_quality": {
                "coverage": _stats(coverages),
                "contamination": _stats(contaminations),
                "registration_score": _stats(scores),
                "homography_inliers": _stats(inliers),
            },
            "asset_readability": {
                "readable": decoded_predictions,
                "unreadable": total_predictions - decoded_predictions,
                "readable_fraction": _safe_div(decoded_predictions, total_predictions),
            },
        },
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "paper_index",
        "paper_title",
        "gt_figure_id",
        "gt_label",
        "gt_page",
        "prediction_id",
        "prediction_label",
        "prediction_page",
        "source_type",
        "score",
        "inliers",
        "coverage",
        "contamination",
        "detected",
        "usable",
        "status",
        "reason",
    ]
    _write_csv(config.output_dir / "per_figure.csv", rows, fields)
    _write_csv(
        config.output_dir / "per_paper.csv",
        paper_rows,
        list(paper_rows[0]) if paper_rows else [],
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gt-root", type=Path, default=Path("results/figure-caption-pairs"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--method", default=None)
    parser.add_argument(
        "--prediction-format",
        choices=["auto", "bundle", "marker", "mineru", "p2p-doclayout"],
        default="auto",
    )
    parser.add_argument("--scope", choices=["body", "appendix", "all"], default="body")
    parser.add_argument("--max-image-dim", type=int, default=900)
    parser.add_argument("--max-features", type=int, default=2500)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--match-threshold", type=float, default=0.15)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--contamination-threshold", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ImageEvaluationConfig(
        gt_root=args.gt_root,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        output_dir=args.output_dir,
        method=args.method or args.prediction_root.name,
        prediction_format=args.prediction_format,
        scope=args.scope,
        max_image_dim=args.max_image_dim,
        max_features=args.max_features,
        ratio_test=args.ratio_test,
        min_inliers=args.min_inliers,
        match_threshold=args.match_threshold,
        coverage_threshold=args.coverage_threshold,
        contamination_threshold=args.contamination_threshold,
    )
    summary = evaluate_image_crops(config)
    print(
        json.dumps(
            {
                "method": summary["method"],
                "prediction_format": summary["prediction_format"],
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
