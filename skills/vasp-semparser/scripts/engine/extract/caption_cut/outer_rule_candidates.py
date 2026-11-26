"""Detect only plausible page-outer horizontal / vertical boundary rules.

This module deliberately does *not* assign internal table, equation, algorithm,
or Figure rules.  It presents a small set of long strokes near the four page
sides to the paper's single semantic call.  Geometry later activates only a
high-confidence rule selected as the page's top, bottom, left, or right legal
content boundary.
"""
from __future__ import annotations

import math
from collections import defaultdict

import fitz


MIN_AXIS_LENGTH = 8.0
AXIS_TOLERANCE = 0.75
HORIZONTAL_MIN_PAGE_RATIO = 0.18
VERTICAL_MIN_PAGE_RATIO = 0.12
OUTER_ZONE = 0.42
MAX_CANDIDATES_PER_SIDE = 12
CONFIDENCE_LEVELS = {"high", "medium", "low"}
SIDES = ("top", "bottom", "left", "right")


def _axis_rect(p0, p1, width):
    x0, x1 = sorted((float(p0.x), float(p1.x)))
    y0, y1 = sorted((float(p0.y), float(p1.y)))
    dx, dy = x1 - x0, y1 - y0
    tolerance = max(AXIS_TOLERANCE, 1.5 * width)
    if dy <= tolerance and dx >= MIN_AXIS_LENGTH:
        y = (y0 + y1) / 2
        return "horizontal", fitz.Rect(
            x0, y - width / 2, x1, y + width / 2)
    if dx <= tolerance and dy >= MIN_AXIS_LENGTH:
        x = (x0 + x1) / 2
        return "vertical", fitz.Rect(
            x - width / 2, y0, x + width / 2, y1)
    return None


def _drawing_segments(drawing):
    """Extract exact painted bboxes for H/V strokes and thin rectangles."""
    width = max(0.1, float(drawing.get("width") or 0.1))
    segments = []
    for item in drawing.get("items") or []:
        kind = item[0] if item else None
        if kind == "l":
            parsed = _axis_rect(item[1], item[2], width)
            if parsed:
                orientation, rect = parsed
                segments.append({"orientation": orientation, "rect": rect})
        elif kind == "re":
            rect = fitz.Rect(item[1])
            if rect.width >= MIN_AXIS_LENGTH and rect.height <= 3.0:
                segments.append({"orientation": "horizontal", "rect": rect})
            elif rect.height >= MIN_AXIS_LENGTH and rect.width <= 3.0:
                segments.append({"orientation": "vertical", "rect": rect})
            elif (drawing.get("fill") is None
                  and min(rect.width, rect.height) >= MIN_AXIS_LENGTH):
                segments.extend([
                    {"orientation": "horizontal", "rect": fitz.Rect(
                        rect.x0, rect.y0 - width / 2,
                        rect.x1, rect.y0 + width / 2)},
                    {"orientation": "horizontal", "rect": fitz.Rect(
                        rect.x0, rect.y1 - width / 2,
                        rect.x1, rect.y1 + width / 2)},
                    {"orientation": "vertical", "rect": fitz.Rect(
                        rect.x0 - width / 2, rect.y0,
                        rect.x0 + width / 2, rect.y1)},
                    {"orientation": "vertical", "rect": fitz.Rect(
                        rect.x1 - width / 2, rect.y0,
                        rect.x1 + width / 2, rect.y1)},
                ])
    seqno = drawing.get("seqno")
    for segment in segments:
        segment["seqno"] = seqno
    return segments


def _merge_segments(segments):
    """Merge duplicate/overlapping collinear primitives without padding."""
    merged = []
    for orientation in ("horizontal", "vertical"):
        axis = []
        for item in segments:
            if item["orientation"] != orientation:
                continue
            rect = fitz.Rect(item["rect"])
            coordinate = ((rect.y0 + rect.y1) / 2
                          if orientation == "horizontal"
                          else (rect.x0 + rect.x1) / 2)
            lo = rect.x0 if orientation == "horizontal" else rect.y0
            hi = rect.x1 if orientation == "horizontal" else rect.y1
            axis.append((coordinate, lo, hi, rect, item.get("seqno")))
        axis.sort(key=lambda value: (value[0], value[1], value[2]))
        current = []
        for coordinate, lo, hi, rect, seqno in axis:
            match = None
            for group in reversed(current):
                if group["coordinate"] < coordinate - 1.0:
                    break
                if (abs(group["coordinate"] - coordinate) <= 1.0
                        and lo <= group["hi"] + 1.0
                        and hi >= group["lo"] - 1.0):
                    match = group
                    break
            if match is None:
                current.append({
                    "coordinate": coordinate,
                    "lo": lo,
                    "hi": hi,
                    "rect": fitz.Rect(rect),
                    "seqnos": {seqno} if isinstance(seqno, int) else set(),
                })
            else:
                match["lo"] = min(match["lo"], lo)
                match["hi"] = max(match["hi"], hi)
                match["rect"] |= rect
                if isinstance(seqno, int):
                    match["seqnos"].add(seqno)
        for group in current:
            merged.append({
                "orientation": orientation,
                "rect": group["rect"],
                "source_seqnos": sorted(group["seqnos"]),
            })
    return merged


def _rect_distance(left, right):
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return math.hypot(dx, dy)


def _context(rect, rows):
    row_boxes = [(row, fitz.Rect(row["bbox"])) for row in rows]
    nearest = sorted(
        ((_rect_distance(rect, box), row) for row, box in row_boxes),
        key=lambda pair: pair[0],
    )
    above = sorted(
        ((rect.y0 - box.y1, row) for row, box in row_boxes
         if box.y1 <= rect.y1 + 1
         and min(box.x1, rect.x1) > max(box.x0, rect.x0)),
        key=lambda pair: pair[0],
    )
    below = sorted(
        ((box.y0 - rect.y1, row) for row, box in row_boxes
         if box.y0 >= rect.y0 - 1
         and min(box.x1, rect.x1) > max(box.x0, rect.x0)),
        key=lambda pair: pair[0],
    )
    left = sorted(
        ((rect.x0 - box.x1, row) for row, box in row_boxes
         if box.x1 <= rect.x1 + 1
         and min(box.y1, rect.y1) > max(box.y0, rect.y0)),
        key=lambda pair: pair[0],
    )
    right = sorted(
        ((box.x0 - rect.x1, row) for row, box in row_boxes
         if box.x0 >= rect.x0 - 1
         and min(box.y1, rect.y1) > max(box.y0, rect.y0)),
        key=lambda pair: pair[0],
    )
    return {
        "nearest_line_ids": [row["id"] for _gap, row in nearest[:8]],
        "above_line_ids": [row["id"] for _gap, row in above[:3]],
        "below_line_ids": [row["id"] for _gap, row in below[:3]],
        "left_line_ids": [row["id"] for _gap, row in left[:3]],
        "right_line_ids": [row["id"] for _gap, row in right[:3]],
    }


def _candidate_side(orientation, rect, page_rect):
    if orientation == "horizontal":
        if rect.width / page_rect.width < HORIZONTAL_MIN_PAGE_RATIO:
            return None
        ratio = ((rect.y0 + rect.y1) / 2 - page_rect.y0) / page_rect.height
        if ratio <= OUTER_ZONE:
            return "top"
        if ratio >= 1.0 - OUTER_ZONE:
            return "bottom"
        return None
    if rect.height / page_rect.height < VERTICAL_MIN_PAGE_RATIO:
        return None
    ratio = ((rect.x0 + rect.x1) / 2 - page_rect.x0) / page_rect.width
    if ratio <= OUTER_ZONE:
        return "left"
    if ratio >= 1.0 - OUTER_ZONE:
        return "right"
    return None


def _rank(side, rect, page_rect):
    if side in {"top", "bottom"}:
        length = rect.width / page_rect.width
        coordinate = ((rect.y0 + rect.y1) / 2 - page_rect.y0) / page_rect.height
        edge_distance = coordinate if side == "top" else 1.0 - coordinate
    else:
        length = rect.height / page_rect.height
        coordinate = ((rect.x0 + rect.x1) / 2 - page_rect.x0) / page_rect.width
        edge_distance = coordinate if side == "left" else 1.0 - coordinate
    return (length + 0.25 * (1.0 - edge_distance), length, -edge_distance)


def extract_outer_boundary_candidates(doc, source_lines):
    """Return a bounded candidate list for the four page-outer sides.

    Repetition is computed before the per-side cap. Otherwise longer plot/grid
    rules can evict a shorter repeated header separator before the semantic
    selector ever sees it.
    """
    by_page = defaultdict(list)
    for row in source_lines:
        by_page[int(row["page"]) - 1].append(row)
    provisional = []
    for page_index in sorted(by_page):
        page = doc[page_index]
        raw = []
        for drawing in page.get_drawings():
            raw.extend(_drawing_segments(drawing))
        merged = _merge_segments(raw)
        for item in merged:
            rect = fitz.Rect(item["rect"]) & page.rect
            if rect.is_empty:
                continue
            side = _candidate_side(item["orientation"], rect, page.rect)
            if side is None:
                continue
            provisional.append({
                "page_index": page_index,
                "side": side,
                "item": item,
                "rect": rect,
                "rank": _rank(side, rect, page.rect),
                "page_x_ratio": rect.x0 / page.rect.width,
                "page_y_ratio": rect.y0 / page.rect.height,
                "page_width_ratio": rect.width / page.rect.width,
                "page_height_ratio": rect.height / page.rect.height,
            })

    for candidate in provisional:
        repeats = []
        for other in provisional:
            if (other["page_index"] == candidate["page_index"]
                    or other["side"] != candidate["side"]):
                continue
            if (abs(other["page_x_ratio"] - candidate["page_x_ratio"]) <= 0.01
                    and abs(other["page_y_ratio"] - candidate["page_y_ratio"]) <= 0.01
                    and abs(other["page_width_ratio"] - candidate["page_width_ratio"]) <= 0.02
                    and abs(other["page_height_ratio"] - candidate["page_height_ratio"]) <= 0.02):
                repeats.append(other["page_index"] + 1)
        candidate["repeat_pages"] = sorted(set(
            [candidate["page_index"] + 1, *repeats]))
        candidate["repeat_count"] = len(candidate["repeat_pages"])

    repeat_threshold = max(2, math.ceil(0.3 * max(1, len(by_page))))
    candidates = []
    for page_index in sorted(by_page):
        page = doc[page_index]
        kept = []
        for side in SIDES:
            values = [value for value in provisional
                      if value["page_index"] == page_index
                      and value["side"] == side]
            values.sort(key=lambda value: (
                value["repeat_count"] >= repeat_threshold,
                value["repeat_count"],
                *value["rank"],
                value["rect"].x0, value["rect"].y0,
                value["rect"].x1, value["rect"].y1,
            ), reverse=True)
            kept.extend(values[:MAX_CANDIDATES_PER_SIDE])
        kept.sort(key=lambda value: (
            SIDES.index(value["side"]), value["rect"].y0,
            value["rect"].x0, value["rect"].y1,
            value["rect"].x1))
        counters = defaultdict(int)
        for value in kept:
            side = value["side"]
            item = value["item"]
            rect = value["rect"]
            counters[side] += 1
            orientation = item["orientation"]
            candidates.append({
                "id": f"p{page_index + 1}_{side[0]}{counters[side]}",
                "page": page_index + 1,
                "candidate_side": side,
                "orientation": orientation,
                "bbox": [round(value, 4) for value in rect],
                "page_x_ratio": round(value["page_x_ratio"], 4),
                "page_y_ratio": round(value["page_y_ratio"], 4),
                "page_width_ratio": round(value["page_width_ratio"], 4),
                "page_height_ratio": round(value["page_height_ratio"], 4),
                "source_seqnos": item["source_seqnos"],
                **_context(rect, by_page[page_index]),
                "repeat_pages": value["repeat_pages"],
                "repeat_count": value["repeat_count"],
                "repeat_threshold": repeat_threshold,
            })
    return candidates


def prompt_candidates(candidates):
    """Remove PDF drawing internals that do not help semantic selection."""
    return [
        {key: value for key, value in candidate.items()
         if key != "source_seqnos"}
        for candidate in candidates
    ]


def normalize_page_boundaries(raw, candidates, content_pages):
    """Validate selections and mark only high-confidence rules as active."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    by_page = {}
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            continue
        by_page.setdefault(page, item)
    pages = []
    active_count = 0
    selected_count = 0
    for page in sorted(set(int(value) for value in content_pages)):
        item = by_page.get(page, {})
        normalized = {"page": page}
        used = set()
        for side in SIDES:
            value = item.get(side)
            if isinstance(value, str):
                value = {"candidate_id": value}
            if not isinstance(value, dict):
                value = {}
            candidate_id = value.get("candidate_id")
            candidate = by_id.get(candidate_id)
            valid = bool(
                candidate
                and candidate["page"] == page
                and candidate["candidate_side"] == side
                and candidate_id not in used)
            confidence = value.get("confidence")
            if confidence not in CONFIDENCE_LEVELS:
                confidence = "low"
            if not valid:
                candidate_id = None
                candidate = None
                confidence = "low"
            else:
                used.add(candidate_id)
                selected_count += 1
            active = bool(candidate and confidence == "high")
            active_count += int(active)
            normalized[side] = {
                "candidate_id": candidate_id,
                "bbox": candidate["bbox"] if candidate else None,
                "confidence": confidence,
                "active": active,
                "reason": str(value.get("reason") or "")[:500],
            }
        pages.append(normalized)
    return {
        "schema_version": "dp-p2p.page-outer-boundaries.v1",
        "fallback_policy": "physical-page-edge-per-missing-non-high-or-invalid-side",
        "pages": pages,
        "candidate_count": len(candidates),
        "selected_count": selected_count,
        "active_count": active_count,
    }
