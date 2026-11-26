"""Read saved GT/grouped-bbox evaluations without rerunning ownership matching."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


GROUP_SCHEMA = "dp-p2p.native-figure-only-gt-grouped-ap.v1"


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_box(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"invalid bbox: {values!r}")
    box = [float(value) for value in values]
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"non-finite bbox: {values!r}")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"empty bbox: {values!r}")
    if box[2] <= 0 or box[3] <= 0 or box[0] >= 1000 or box[1] >= 1000:
        raise ValueError(f"bbox outside page: {values!r}")
    return box


def read_catalog(dataset_root: Path, method_roots: dict[str, Path]) -> dict[str, Any]:
    """Validate identical frozen GT and emit every GT x method, including misses."""

    if not method_roots:
        raise ValueError("at least one method is required")
    papers = sorted(
        path for path in dataset_root.iterdir()
        if path.is_dir() and (path / "paper.pdf").is_file()
    )
    if not papers:
        raise ValueError(f"no papers in {dataset_root}")
    sources: dict[str, Any] = {}
    groups: dict[str, dict[tuple[int, str], Any]] = {}
    gt_document = None
    gt_hash = None
    for method, root in method_roots.items():
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", method):
            raise ValueError(f"unsafe method name: {method!r}")
        summary = read_json(root / "summary.json")
        if summary.get("schema_version") != GROUP_SCHEMA:
            raise ValueError(f"not the frozen Figure-only grouped-AP protocol: {root}")
        if summary["config"]["scope"] != "body":
            raise ValueError(f"expected body-only GT: {root}")
        if summary["dataset"]["paper_count"] != len(papers):
            raise ValueError(f"dataset paper count changed: {root}")
        document = read_json(root / "omni_ground_truth.json")
        current_gt_hash = digest(document)
        if gt_hash is not None and current_gt_hash != gt_hash:
            raise ValueError("methods do not have identical saved ground truth")
        gt_document, gt_hash = document, current_gt_hash
        sources[method] = {
            "evaluation_root": str(root.resolve()),
            "summary": summary,
            "file_sha256": {
                name: file_sha256(root / name)
                for name in ("summary.json", "per_gt_group.csv", "omni_ground_truth.json")
            },
        }
        groups[method] = {}
        with (root / "per_gt_group.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                paper_index = int(row["paper_index"])
                if not 0 <= paper_index < len(papers):
                    raise ValueError(f"invalid paper index: {row}")
                if row["paper_title"] != papers[paper_index].name:
                    raise ValueError(f"paper ordering changed: {row['paper_title']}")
                key = (paper_index, row["matched_gt_id"])
                if key in groups[method]:
                    raise ValueError(f"duplicate method/GT group: {method}, {key}")
                count = int(row["component_count"])
                if count < 1 or row["status"] not in {"gt_singleton", "gt_grouped"}:
                    raise ValueError(f"invalid saved group: {row}")
                groups[method][key] = {
                    "bbox": checked_box([row[f"bbox_{axis}"] for axis in ("x0", "y0", "x1", "y1")]),
                    "page": int(row["page"]),
                    "component_count": count,
                    "component_ids": row["component_ids"],
                    "label": row["matched_gt_label"],
                }
        if len(groups[method]) != summary["predictions"]["gt_group_count"]:
            raise ValueError(f"group count disagrees with summary: {method}")

    figures: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    seen_pages: set[tuple[int, int]] = set()
    for record in gt_document:
        info = record["page_info"]
        match = re.fullmatch(r"paper_(\d+)\.pdf_(\d+)\.jpg", info["image_path"])
        if match is None or info["width"] != 1000 or info["height"] != 1000:
            raise ValueError(f"unsupported saved GT page format: {info}")
        paper_index, page_zero = map(int, match.groups())
        if not 0 <= paper_index < len(papers) or info["page_no"] != page_zero:
            raise ValueError(f"inconsistent saved GT page: {info}")
        page_key = (paper_index, page_zero)
        if page_key in seen_pages:
            raise ValueError(f"duplicate saved GT page: {info}")
        seen_pages.add(page_key)
        for target in record["layout_dets"]:
            if target["category_type"] != "figure" or target.get("ignore"):
                raise ValueError("expected non-ignored Figure-only ground truth")
            figure_id = str(target["anno_id"])
            key = (paper_index, figure_id)
            if key in seen:
                raise ValueError(f"duplicate GT Figure: {key}")
            seen.add(key)
            polygon = target["poly"]
            if len(polygon) != 8:
                raise ValueError(f"invalid saved GT polygon: {polygon}")
            gt_box = checked_box([
                min(polygon[0::2]), min(polygon[1::2]),
                max(polygon[0::2]), max(polygon[1::2]),
            ])
            predictions = {}
            for method in method_roots:
                prediction = groups[method].get(key)
                if prediction is not None and prediction["page"] != page_zero + 1:
                    raise ValueError(f"group and GT page disagree: {method}, {key}")
                predictions[method] = prediction
            label = next((p["label"] for p in predictions.values() if p), figure_id)
            figures.append({
                "gt_key": f"p{paper_index:03d}_{digest(figure_id)[:12]}",
                "paper_index": paper_index,
                "paper_title": papers[paper_index].name,
                "pdf": str((papers[paper_index] / "paper.pdf").resolve()),
                "figure_id": figure_id,
                "label": label,
                "page": page_zero + 1,
                "gt_bbox": gt_box,
                "predictions": predictions,
            })
    figures.sort(key=lambda row: (row["paper_index"], row["page"], row["figure_id"]))
    for method, source in sources.items():
        if set(groups[method]) - seen:
            raise ValueError(f"groups reference unknown GT: {method}")
        if source["summary"]["dataset"]["gt_figure_count"] != len(figures):
            raise ValueError(f"GT count disagrees with summary: {method}")
        missing = len(figures) - len(groups[method])
        if missing != source["summary"]["predictions"]["gt_without_group_count"]:
            raise ValueError(f"missing-GT count disagrees with summary: {method}")
    return {
        "dataset_root": str(dataset_root.resolve()),
        "sources": sources,
        "gt_sha256": gt_hash,
        "paper_count": len(papers),
        "pdf_sha256": {
            str((paper / "paper.pdf").resolve()): file_sha256(paper / "paper.pdf")
            for paper in papers
        },
        "figures": figures,
    }
