"""Derive exact Table / Algorithm bands from membership and PDF rules.

The paper's single semantic call decides which stable source-line IDs belong to
each numbered Table or Algorithm.  This module never asks the model to rank
geometry.  It unions the exact member bboxes, then selects the nearest
horizontal PDF stroke above and below that union which spans the member text.
Missing rules fall back only to the member bbox edge; a remote rule is never
used to complete a pair.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

import fitz

from outer_rule_candidates import _drawing_segments, _merge_segments
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from family_interface import family_name, namespaced_num


SCHEMA_VERSION = "dp-p2p.deterministic-structured-bands.v2"


def canonical_element_num(value):
    """Return the stable number used to match ``Table N`` anchors."""
    if value is None:
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"^(?:table|tab\.?|algorithm|alg\.?)\s*", "", text)
    return text.strip(" \t\r\n:：.()[]{}")


def _bbox(values):
    rects = [fitz.Rect(value) for value in values]
    return fitz.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )


def _role_map(segments, valid_ids):
    valid = set(valid_ids)
    roles = {}
    for segment in segments:
        for line_id in range(
                segment.get("start", 0), segment.get("end", -1) + 1):
            if line_id in valid:
                roles[line_id] = segment.get("role", "other")
    return roles


def _horizontal_rules(page):
    raw = []
    for drawing in page.get_drawings():
        raw.extend(_drawing_segments(drawing))
    rules = []
    min_width = max(18.0, 0.04 * page.rect.width)
    for item in _merge_segments(raw):
        if item.get("orientation") != "horizontal":
            continue
        rect = fitz.Rect(item["rect"]) & page.rect
        if (rect.is_empty or rect.width < min_width or rect.height > 4.0
                or not all(math.isfinite(value) for value in tuple(rect))):
            continue
        rules.append({
            "bbox": rect,
            "source_seqnos": item.get("source_seqnos", []),
        })
    rules.sort(key=lambda item: (
        item["bbox"].y0, item["bbox"].x0,
        item["bbox"].y1, item["bbox"].x1))
    return rules


def _line_payload(item):
    if item is None:
        return None
    return {
        "bbox": [round(value, 4) for value in item["bbox"]],
        "source_seqnos": item.get("source_seqnos", []),
    }


def _foreign_figure_lines(band, page_number, member_ids, rows_by_id, roles):
    foreign = []
    for line_id, row in rows_by_id.items():
        try:
            row_page = int(row.get("page"))
        except (TypeError, ValueError):
            continue
        if (row_page != page_number or line_id in member_ids
                or roles.get(line_id) not in {
                "figure_content", "subfig_caption"}):
            continue
        rect = fitz.Rect(row["bbox"])
        centre = fitz.Point((rect.x0 + rect.x1) / 2,
                            (rect.y0 + rect.y1) / 2)
        if centre in band:
            foreign.append(line_id)
    return sorted(foreign)


def derive_table_bands(doc, source_lines, segments, structured,
                       allowed_pages=None):
    """Return deterministic Table and Algorithm bands.

    The historical public function name is retained so frozen callers remain
    compatible.  ``element_type`` in every record is authoritative.
    """
    rows_by_id = {
        int(row["id"]): row for row in source_lines
        if isinstance(row, dict) and "id" in row and "bbox" in row
    }
    roles = _role_map(segments, rows_by_id)
    allowed = set(allowed_pages) if allowed_pages is not None else None
    rules_by_page = {}
    bands = []

    groups = []
    if isinstance(structured, dict):
        groups.extend(("table", item) for item in structured.get("tables", []))
        groups.extend(("algorithm", item)
                      for item in structured.get("algorithms", []))
    for element_type, element in groups:
        if not isinstance(element, dict):
            continue
        member_ids = []
        for value in element.get("source_line_ids", []):
            try:
                line_id = int(value)
            except (TypeError, ValueError):
                continue
            if line_id in rows_by_id:
                member_ids.append(line_id)
        member_ids = sorted(set(member_ids))
        # A Table title is an external caption and must not enlarge its body
        # band.  An Algorithm title is normally printed inside the ruled box,
        # so every semantically-owned Algorithm line participates.
        content_ids = ([line_id for line_id in member_ids
                        if roles.get(line_id) != "table_title"]
                       if element_type == "table" else list(member_ids))
        pages = defaultdict(list)
        for line_id in content_ids:
            try:
                page_number = int(rows_by_id[line_id]["page"])
            except (KeyError, TypeError, ValueError):
                continue
            if allowed is None or page_number - 1 in allowed:
                pages[page_number].append(line_id)

        if not pages:
            bands.append({
                "element_type": element_type,
                "element_num": element.get("num"),
                "canonical_num": canonical_element_num(namespaced_num(element.get("num"), family_name(element.get("family"),element_type),element_type.title())),
                "family": family_name(element.get("family"),element_type),
                "page": None,
                "source_line_ids": member_ids,
                "content_line_ids": [],
                "active": False,
                "status": "no_non_caption_member_lines",
            })
            continue

        for page_number, page_content_ids in sorted(pages.items()):
            page_index = page_number - 1
            if not (0 <= page_index < len(doc)):
                continue
            page = doc[page_index]
            content = _bbox(
                rows_by_id[line_id]["bbox"] for line_id in page_content_ids)
            heights = [
                fitz.Rect(rows_by_id[line_id]["bbox"]).height
                for line_id in page_content_ids
                if fitz.Rect(rows_by_id[line_id]["bbox"]).height > 0
            ]
            median_height = statistics.median(heights) if heights else 8.0
            x_tolerance = max(2.0, 0.04 * content.width)
            y_tolerance = max(0.75, 0.25 * median_height)
            max_gap = max(6.0, 2.25 * median_height)

            if page_index not in rules_by_page:
                rules_by_page[page_index] = _horizontal_rules(page)
            covering = [
                item for item in rules_by_page[page_index]
                if (item["bbox"].x0 <= content.x0 + x_tolerance
                    and item["bbox"].x1 >= content.x1 - x_tolerance)
            ]
            top_options = [
                item for item in covering
                if item["bbox"].y1 <= content.y0 + y_tolerance
                and content.y0 - item["bbox"].y1 <= max_gap
            ]
            bottom_options = [
                item for item in covering
                if item["bbox"].y0 >= content.y1 - y_tolerance
                and item["bbox"].y0 - content.y1 <= max_gap
            ]
            top_rule = max(
                top_options, key=lambda item: item["bbox"].y1,
                default=None)
            bottom_rule = min(
                bottom_options, key=lambda item: item["bbox"].y0,
                default=None)

            x0 = min(
                [content.x0]
                + ([top_rule["bbox"].x0] if top_rule else [])
                + ([bottom_rule["bbox"].x0] if bottom_rule else []))
            x1 = max(
                [content.x1]
                + ([top_rule["bbox"].x1] if top_rule else [])
                + ([bottom_rule["bbox"].x1] if bottom_rule else []))
            y0 = top_rule["bbox"].y0 if top_rule else content.y0
            y1 = bottom_rule["bbox"].y1 if bottom_rule else content.y1
            band = fitz.Rect(x0, y0, x1, y1) & page.rect
            foreign_ids = _foreign_figure_lines(
                band,page_number,set(page_content_ids),rows_by_id,roles)
            active = (not band.is_empty and band.width > 2 and band.height > 2
                      and not foreign_ids)
            geometry_source = (
                "top_and_bottom_rules" if top_rule and bottom_rule
                else "top_rule_and_member_bbox" if top_rule
                else "member_bbox_and_bottom_rule" if bottom_rule
                else "member_bbox_only")
            bands.append({
                "element_type": element_type,
                "element_num": element.get("num"),
                "canonical_num": canonical_element_num(namespaced_num(element.get("num"), family_name(element.get("family"),element_type),element_type.title())),
                "family": family_name(element.get("family"),element_type),
                "page": page_number,
                "source_line_ids": member_ids,
                "content_line_ids": page_content_ids,
                "content_bbox": [round(value, 4) for value in content],
                "top_rule": _line_payload(top_rule),
                "bottom_rule": _line_payload(bottom_rule),
                "band_bbox": [round(value, 4) for value in band],
                "median_member_line_height": round(median_height, 4),
                "max_rule_gap": round(max_gap, 4),
                "geometry_source": geometry_source,
                "foreign_figure_line_ids": foreign_ids,
                "active": active,
                "status": "ok" if active else (
                    "foreign_figure_membership_conflict" if foreign_ids
                    else "invalid_band"),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": (
            "LLM supplies Table/Algorithm source_line_ids only; exact member "
            "bboxes and nearest spanning PDF rules determine geometry; "
            "missing rules fall back to member bbox edges"),
        "bands": bands,
        "active_count": sum(item.get("active") is True for item in bands),
        "conflict_count": sum(
            item.get("status") == "foreign_figure_membership_conflict"
            for item in bands),
    }


def index_active_table_bands(payload, doc):
    """Index validated active records for crop geometry."""
    by_page = defaultdict(list)
    by_element = {}
    for item in payload.get("bands", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict) or item.get("active") is not True:
            continue
        try:
            page_number = int(item["page"])
            page_index = page_number - 1
            band = fitz.Rect(item["band_bbox"]) & doc[page_index].rect
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if band.is_empty or not all(math.isfinite(value) for value in tuple(band)):
            continue
        top_payload = item.get("top_rule") or {}
        bottom_payload = item.get("bottom_rule") or {}
        top = fitz.Rect(top_payload.get("bbox", (
            band.x0, band.y0, band.x1, band.y0)))
        bottom = fitz.Rect(bottom_payload.get("bbox", (
            band.x0, band.y1, band.x1, band.y1)))
        record = {
            **item,
            "band": band,
            "top": top,
            "bottom": bottom,
        }
        by_page[page_index].append(record)
        number = canonical_element_num(item.get("canonical_num",
                                                item.get("element_num")))
        if number:
            element_type = item.get("element_type", "table")
            by_element[(page_index, element_type, number)] = record
    return dict(by_page), by_element


def exclude_table_owned_ink(pool, records):
    """Remove ink already inside an immutable semantic Table band."""
    kept = []
    for value in pool:
        rect = fitz.Rect(value)
        area = max(0.0, rect.width) * max(0.0, rect.height)
        centre = fitz.Point((rect.x0 + rect.x1) / 2,
                            (rect.y0 + rect.y1) / 2)
        owned = False
        for record in records:
            band = record["band"]
            overlap = rect & band
            overlap_area = (max(0.0, overlap.width)
                            * max(0.0, overlap.height))
            if centre in band or (area > 0 and overlap_area >= 0.5 * area):
                owned = True
                break
        if not owned:
            kept.append(rect)
    return kept
