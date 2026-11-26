"""Generate and validate local top/bottom rule pairs for structured elements.

The candidates are exact horizontal PDF drawing primitives.  The semantic model
only selects a precomputed pair for a Table, Algorithm, or genuinely ruled
Equation; it never invents geometry.  Only validated high-confidence pairs are
active downstream.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import fitz

from outer_rule_candidates import _drawing_segments, _merge_segments


MIN_WIDTH_PT = 28.0
MIN_WIDTH_RATIO = 0.08
MAX_RULE_HEIGHT_PT = 4.0
MIN_PAIR_GAP_PT = 4.0
MAX_PAIR_HEIGHT_RATIO = 0.55
MIN_X_OVERLAP = 0.68
MIN_WIDTH_SIMILARITY = 0.55
MAX_PAIRS_PER_PAGE = 10
MAX_PAIRS_PER_PAPER = 64
CONFIDENCE_LEVELS = {"high", "medium", "low"}
ELEMENT_TYPES = {"table", "algorithm", "equation"}

_STRUCTURE_HINT_RE = re.compile(
    r"^\s*(?:(?:Table|Tab\.?|Algorithm|Alg\.?|Equation|Eq\.?)\s*"
    r"[A-Za-z0-9.()\-]+|\([A-Za-z]?\d+(?:\.\d+)*\)\s*$)",
    re.IGNORECASE,
)


def _axis_overlap(left: fitz.Rect, right: fitz.Rect) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))


def _line_distance_y(rect: fitz.Rect, row: fitz.Rect) -> float:
    return max(rect.y0 - row.y1, row.y0 - rect.y1, 0.0)


def _row_context(top: fitz.Rect, bottom: fitz.Rect, rows: list[dict]) -> dict:
    inside = []
    nearby = []
    hints = []
    table_hints = []
    outside_table_hints = []
    x0 = min(top.x0, bottom.x0)
    x1 = max(top.x1, bottom.x1)
    band = fitz.Rect(x0, top.y0, x1, bottom.y1)
    local_height = max(18.0, min(54.0, 0.18 * band.height + 12.0))
    hint_height = max(local_height, 120.0)
    for row in rows:
        try:
            box = fitz.Rect(row["bbox"])
        except Exception:
            continue
        overlap = max(0.0, min(box.x1, x1) - max(box.x0, x0))
        if overlap <= 0:
            continue
        center_y = (box.y0 + box.y1) / 2
        is_inside = top.y1 <= center_y <= bottom.y0
        distance = min(_line_distance_y(top, box), _line_distance_y(bottom, box))
        if is_inside:
            inside.append(int(row["id"]))
        if is_inside or distance <= local_height:
            nearby.append(int(row["id"]))
        if ((is_inside or distance <= hint_height)
                and _STRUCTURE_HINT_RE.match(str(row.get("text") or ""))):
            line_id=int(row["id"])
            hints.append(line_id)
            if re.match(r"^\s*(?:Table|Tab\.?)\b",str(row.get("text") or ""),re.I):
                table_hints.append(line_id)
                if center_y<top.y0 or center_y>bottom.y1:
                    outside_table_hints.append(line_id)
    return {
        "inside_line_ids": inside[:48],
        "inside_line_count": len(inside),
        "nearby_line_ids": nearby[:64],
        "hint_line_ids": hints[:16],
        "table_hint_line_ids": table_hints[:16],
        "outside_table_hint_line_ids": outside_table_hints[:16],
    }


def _candidate_score(candidate: dict) -> tuple:
    return (
        bool(candidate["outside_table_hint_line_ids"]),
        bool(candidate["table_hint_line_ids"]),
        bool(candidate["hint_line_ids"]),
        min(candidate["inside_line_count"], 256),
        candidate["x_overlap_ratio"],
        candidate["width_similarity"],
        -candidate["page_height_ratio"],
    )


def extract_structured_rule_pair_candidates(doc, source_lines):
    """Return a bounded list of plausible local horizontal-rule pairs."""
    rows_by_page = defaultdict(list)
    for row in source_lines:
        try:
            rows_by_page[int(row["page"]) - 1].append(row)
        except (KeyError, TypeError, ValueError):
            continue

    all_candidates = []
    for page_index in sorted(rows_by_page):
        page = doc[page_index]
        raw = []
        for drawing in page.get_drawings():
            raw.extend(_drawing_segments(drawing))
        horizontal = []
        for item in _merge_segments(raw):
            if item["orientation"] != "horizontal":
                continue
            rect = fitz.Rect(item["rect"]) & page.rect
            if (rect.is_empty or rect.height > MAX_RULE_HEIGHT_PT
                    or rect.width < MIN_WIDTH_PT
                    or rect.width / page.rect.width < MIN_WIDTH_RATIO):
                continue
            horizontal.append((rect, item.get("source_seqnos", [])))
        horizontal.sort(key=lambda value: (
            value[0].y0, value[0].x0, value[0].x1))

        page_candidates = []
        for top_index, (top, top_seqnos) in enumerate(horizontal):
            for bottom, bottom_seqnos in horizontal[top_index + 1:]:
                vertical_gap = bottom.y0 - top.y1
                if vertical_gap < MIN_PAIR_GAP_PT:
                    continue
                if vertical_gap > MAX_PAIR_HEIGHT_RATIO * page.rect.height:
                    break
                overlap = _axis_overlap(top, bottom)
                min_width = min(top.width, bottom.width)
                max_width = max(top.width, bottom.width)
                overlap_ratio = overlap / max(min_width, 1e-6)
                width_similarity = min_width / max(max_width, 1e-6)
                if (overlap_ratio < MIN_X_OVERLAP
                        or width_similarity < MIN_WIDTH_SIMILARITY):
                    continue
                context = _row_context(
                    top, bottom, rows_by_page[page_index])
                # Without an explicit nearby Table / Algorithm / Equation label,
                # the same call cannot attribute a geometric pair reliably.
                if (not context["inside_line_ids"]
                        or not context["hint_line_ids"]):
                    continue
                band = fitz.Rect(
                    min(top.x0, bottom.x0), top.y0,
                    max(top.x1, bottom.x1), bottom.y1)
                candidate = {
                    "page": page_index + 1,
                    "top_bbox": [round(value, 4) for value in top],
                    "bottom_bbox": [round(value, 4) for value in bottom],
                    "band_bbox": [round(value, 4) for value in band],
                    "page_x_ratio": round(band.x0 / page.rect.width, 4),
                    "page_y_ratio": round(band.y0 / page.rect.height, 4),
                    "page_width_ratio": round(band.width / page.rect.width, 4),
                    "page_height_ratio": round(band.height / page.rect.height, 4),
                    "x_overlap_ratio": round(overlap_ratio, 4),
                    "width_similarity": round(width_similarity, 4),
                    "vertical_gap": round(vertical_gap, 4),
                    "top_source_seqnos": top_seqnos,
                    "bottom_source_seqnos": bottom_seqnos,
                    **context,
                }
                page_candidates.append(candidate)

        page_candidates.sort(key=_candidate_score, reverse=True)
        kept = page_candidates[:MAX_PAIRS_PER_PAGE]
        kept.sort(key=lambda value: (
            value["top_bbox"][1], value["top_bbox"][0],
            value["bottom_bbox"][1], value["bottom_bbox"][0]))
        for index, candidate in enumerate(kept, 1):
            candidate["id"] = f"p{page_index + 1}_sp{index}"
            all_candidates.append(candidate)

    if len(all_candidates) > MAX_PAIRS_PER_PAPER:
        ranked = sorted(all_candidates, key=_candidate_score, reverse=True)
        keep_ids = {candidate["id"] for candidate in ranked[:MAX_PAIRS_PER_PAPER]}
        all_candidates = [candidate for candidate in all_candidates
                          if candidate["id"] in keep_ids]
    return all_candidates


def prompt_structured_rule_candidates(candidates):
    """Serialize compact evidence while preserving exact geometry.

    Full line-id lists remain in the on-disk candidate contract used by the
    validator.  The prompt only needs contiguous id ranges plus explicit hint
    ids because the numbered paper text is already present above it.
    """
    def ranges(values):
        ordered=sorted(set(values))
        if not ordered:
            return []
        out=[]; start=previous=ordered[0]
        for value in ordered[1:]:
            if value==previous+1:
                previous=value
                continue
            out.append([start,previous]); start=previous=value
        out.append([start,previous])
        return out

    hidden={
        "top_source_seqnos","bottom_source_seqnos",
        "inside_line_ids","nearby_line_ids",
    }
    compact=[]
    for candidate in candidates:
        item={key:value for key,value in candidate.items() if key not in hidden}
        item["inside_line_id_ranges"]=ranges(candidate["inside_line_ids"])
        item["nearby_line_id_ranges"]=ranges(candidate["nearby_line_ids"])
        compact.append(item)
    return compact


def _canonical_num(value):
    if value is None:
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"^(?:table|tab\.?|algorithm|alg\.?|equation|eq\.?)\s*", "", text)
    return text.strip(" \t\r\n:：.()[]{}")


def _structured_index(structured, source_doc):
    page_by_id = {}
    for row in source_doc:
        try:
            page_by_id[int(row["id"])] = int(row["page"])
        except (KeyError, TypeError, ValueError):
            continue
    index = defaultdict(list)
    for element_type, key, number_key in (
            ("table", "tables", "num"),
            ("algorithm", "algorithms", "num"),
            ("equation", "equations", "label")):
        for item in structured.get(key, []) if isinstance(structured, dict) else []:
            number = _canonical_num(item.get(number_key))
            source_ids = {
                int(value) for value in item.get("source_line_ids", [])
                if isinstance(value, int)
            }
            pages = {page_by_id[value] for value in source_ids if value in page_by_id}
            index[(element_type, number)].append({
                "source_line_ids": source_ids,
                "pages": pages,
            })
    return index


def normalize_structured_rule_pairs(raw, candidates, structured, source_doc):
    """Validate selections; only high-confidence evidenced pairs become active."""
    by_id = {candidate["id"]: candidate for candidate in candidates}
    structured_index = _structured_index(structured, source_doc)
    normalized = []
    used_candidates = set()
    selected_count = 0
    active_count = 0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        element_type = str(item.get("element_type") or "").strip().casefold()
        element_num = item.get("element_num")
        canonical_num = _canonical_num(element_num)
        candidate_id = item.get("pair_candidate_id")
        candidate = by_id.get(candidate_id)
        confidence = item.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            confidence = "low"
        try:
            requested_page = int(item.get("page"))
        except (TypeError, ValueError):
            requested_page = None
        matches = structured_index.get((element_type, canonical_num), [])
        source_ids = set()
        structured_pages = set()
        for match in matches:
            source_ids |= match["source_line_ids"]
            structured_pages |= match["pages"]
        evidence_ids = set(candidate.get("inside_line_ids", [])) if candidate else set()
        evidence_ids |= set(candidate.get("nearby_line_ids", [])) if candidate else set()
        evidence_ids |= set(candidate.get("hint_line_ids", [])) if candidate else set()
        evidence_overlap = sorted(source_ids & evidence_ids)
        valid = bool(
            element_type in ELEMENT_TYPES
            and canonical_num
            and candidate
            and candidate_id not in used_candidates
            and requested_page == candidate["page"]
            and matches
            and (not structured_pages or requested_page in structured_pages)
            and evidence_overlap
        )
        if valid:
            used_candidates.add(candidate_id)
            selected_count += 1
        else:
            candidate_id = None
            candidate = None
            confidence = "low"
        active = bool(candidate and confidence == "high")
        active_count += int(active)
        normalized.append({
            "element_type": element_type if element_type in ELEMENT_TYPES else "",
            "element_num": element_num,
            "canonical_num": canonical_num,
            "page": requested_page,
            "pair_candidate_id": candidate_id,
            "top_bbox": candidate["top_bbox"] if candidate else None,
            "bottom_bbox": candidate["bottom_bbox"] if candidate else None,
            "band_bbox": candidate["band_bbox"] if candidate else None,
            "confidence": confidence,
            "active": active,
            "evidence_overlap_line_ids": evidence_overlap,
            "reason": str(item.get("reason") or "")[:500],
        })
    return {
        "schema_version": "dp-p2p.structured-rule-pairs.v1",
        "policy": "high-confidence-exact-pair-with-structured-source-evidence",
        "pairs": normalized,
        "candidate_count": len(candidates),
        "selected_count": selected_count,
        "active_count": active_count,
    }
