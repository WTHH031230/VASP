#!/usr/bin/env python3
"""Full caption extraction: expand caption first lines downward or look up segments2.json.

When `--seg segments2.json` is provided and readable, assembles captions directly from
the semantic role segmentation (zero extra LLM calls). Otherwise falls back to geometric
continuation from `textgap.find_captions`, collecting same-column lines downward until:
  1) A new Figure/Table caption starts;
  2) A likely section heading appears;
  3) Vertical gap exceeds the intra-caption line spacing threshold;
  4) Character-weighted dominant font size shifts noticeably from the caption's first line.

Usage: python3 extract_captions.py <paper.pdf> <outdir> [--seg segments2.json]
Output: <outdir>/captions.json
    [{"page": N, "label": "Figure 1", "text": "Figure 1: ...", "bbox": [x0,y0,x1,y1]}, ...]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extract"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from family_interface import family_name
from textgap import (  # noqa: E402
    page_lines, find_captions, detect_columns, col_of,
    content_document_lines, load_source_document_lines,
)

# Extended start-of-caption regex (includes Table to avoid absorbing adjacent table captions).
NEXT_CAPTION_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table)\s+(?:[A-Z]\.?)?\d+(?:\.\d+)*\s*[:.]",
    re.IGNORECASE,
)

# Section heading heuristic: numbered headings ("3.1 Method") or standard unnumbered headings.
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+){0,3}\.?\s+[A-Z]")
NAMED_HEADING_RE = re.compile(
    r"^(abstract|introduction|related work|background|preliminaries|"
    r"method|methods|methodology|experiments?|experimental results|"
    r"results|discussion|conclusions?|limitations|broader impact|"
    r"ethics statement|references|bibliography|acknowledge?ments?|"
    r"appendix|supplementary(?: material)?)\s*$",
    re.IGNORECASE,
)
HEADING_MAX_LEN = 40
HEADING_MAX_WORDS = 6

# Calibrated line-gap thresholds (intra-caption ~0.8-2.1pt vs paragraph boundary ~7-22pt):
# yields ~4.5pt cutoff for typical 9-10pt text.
GAP_MIN_PT = 3.5
GAP_HEIGHT_MULT = 0.45
SIZE_DIFF_MIN = 0.6     # pt: typical caption-vs-body font delta is ~1pt; weighted spans filter sub/superscripts
MAX_CONTINUATION_LINES = 30  # safety cap to avoid runaway continuation in degenerate layouts
SAME_ROW_TOL = 2.0      # pt: tolerance for merging PyMuPDF split spans on the same visual line
MIN_REAL_COLUMN_WIDTH = 80.0  # pt: real text columns exceed ~150pt; ignore narrow spurious columns
MIN_HORIZ_OVERLAP = 0.15  # min horizontal overlap ratio fallback when column detection degrades to 1 column


def _looks_like_heading(txt: str) -> bool:
    t = txt.strip()
    if not t or len(t) > HEADING_MAX_LEN or len(t.split()) > HEADING_MAX_WORDS:
        return False
    if NUMBERED_HEADING_RE.match(t):
        return True
    if NAMED_HEADING_RE.match(t):
        return True
    return False


def _is_big_gap(gap: float, height: float) -> bool:
    return gap > max(GAP_MIN_PT, GAP_HEIGHT_MULT * max(height, 1.0))


def _horiz_overlap_frac(a, b) -> float:
    """Horizontal overlap between rects a and b divided by the narrower width (0..1)."""
    ov = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    narrower = min(a.width, b.width)
    return ov / narrower if narrower > 0 else 0.0


def _size_shifted(candidate_size: float, baseline_size: float) -> bool:
    if not candidate_size or not baseline_size:
        return False
    return abs(candidate_size - baseline_size) >= SIZE_DIFF_MIN


def _rect_key(r):
    return (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))


def _line_font_sizes(page):
    """Map each line bbox on the page to its character-weighted dominant font size."""
    out = {}
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            key = _rect_key(fitz.Rect(ln["bbox"]))
            counts = {}
            for sp in ln.get("spans", []):
                n = len(sp.get("text", ""))
                if n == 0:
                    continue
                counts[sp["size"]] = counts.get(sp["size"], 0) + n
            if counts:
                out[key] = max(counts.items(), key=lambda kv: kv[1])[0]
    return out


def collect_page_captions(page, page_num):
    """Extract {page, label, text, bbox} per caption on a page by expanding downward."""
    lines = page_lines(page)
    if not lines:
        return []
    cols = detect_columns(page, lines)
    caps = find_captions(page, lines)
    if not caps:
        return []

# Map rect 4-tuples to original line text for exact lookup of caption first lines.
    line_text = {}
    for r, txt in lines:
        line_text.setdefault(_rect_key(r), txt)
    line_size = _line_font_sizes(page)

    out = []
    for label, cap_rect in caps:
        ci, col_rect = col_of(cap_rect, cols)
        if col_rect.width < MIN_REAL_COLUMN_WIDTH:
# Ignore narrow spurious columns on figure-heavy sparse pages to avoid dropping continuation lines.
            ci = -1
        cap_key = _rect_key(cap_rect)
        first_txt = line_text.get(cap_key, label)
        baseline_size = line_size.get(cap_key)

        # Candidate continuation lines: below the first line or trailing same-row fragments
        # to its right, sorted by (y0, x0). Allow full-width lines so cross-column headings stop expansion.
        below = []
        for r, txt in lines:
            key = _rect_key(r)
            if key == cap_key:
                continue  # first line already added to parts[0]
            same_row = abs(r.y0 - cap_rect.y0) <= SAME_ROW_TOL
            if same_row:
                if r.x0 < cap_rect.x1 - 1:  # same row but left of or overlapping the first fragment
                    continue
                below.append((r, txt))
                continue
            if r.y0 <= cap_rect.y0 + 0.5:
                continue  # strictly above the first line
            rci, _ = col_of(r, cols)
            if not (rci == ci or rci == -1 or ci == -1):
                continue
            below.append((r, txt))
        below.sort(key=lambda t: (t[0].y0, t[0].x0))

        degenerate_cols = len(cols) <= 1
        parts = [first_txt]
        block = fitz.Rect(cap_rect)
        prev_bottom = cap_rect.y1
        for r, txt in below:
            cand_size = line_size.get(_rect_key(r))
            gap = r.y0 - prev_bottom
# Prefer declared font size over geometric bbox height for the line-gap cutoff.
            if _is_big_gap(gap, cand_size or r.height):
                break
            if NEXT_CAPTION_RE.match(txt):
                break
            if _looks_like_heading(txt):
                break
            if _size_shifted(cand_size, baseline_size):
                break
            if degenerate_cols and abs(r.y0 - cap_rect.y0) > SAME_ROW_TOL \
                    and _horiz_overlap_frac(r, block) < MIN_HORIZ_OVERLAP:
                break  # horizontal overlap guard prevents absorbing lines from an adjacent column
            parts.append(txt)
            block.include_rect(r)
            prev_bottom = r.y1
            if len(parts) - 1 >= MAX_CONTINUATION_LINES:
                break

        text = " ".join(p.strip() for p in parts if p.strip())
        text = re.sub(r"\s+", " ", text).strip()
        out.append({
            "page": page_num,
            "label": label,
            "text": text,
            "bbox": [round(block.x0, 1), round(block.y0, 1), round(block.x1, 1), round(block.y1, 1)],
        })
    return out


def extract_captions(pdf_path: Path):
    doc = fitz.open(pdf_path)
    try:
        results = []
        for pi in range(len(doc)):
            results.extend(collect_page_captions(doc[pi], pi + 1))
        return results
    finally:
        doc.close()


# ---------------- segments2.json lookup mode (preferred) ----------------
# Mirrors the exact body/appendix page selection of segment_roles.py so global line IDs align.

def _enumerate_lines(doc, include_appendix=False, source_lines_path=None):
    """Mirror page selection and global line numbering from segment_roles.py."""
    if source_lines_path is not None and Path(source_lines_path).exists():
        return load_source_document_lines(
            source_lines_path, include_appendix=include_appendix,
            page_count=len(doc))
    lines = []
    pages = content_document_lines(doc, include_appendix=include_appendix)
    for pi, rows in enumerate(pages):
        for r, t, section in rows:
            lines.append((len(lines), pi, r, t, section))
    return lines


def captions_from_segments(pdf_path: Path, seg_path: Path, include_appendix=False,
                           source_lines_path=None):
    """Assemble captions from segments2.json, grouping by (kind, num, section)."""
    segs = json.loads(Path(seg_path).read_text(encoding="utf-8"))
    doc = fitz.open(pdf_path)
    try:
        source_lines_path = (Path(source_lines_path) if source_lines_path is not None
                             else Path(seg_path).with_name("source_lines.json"))
        lines = _enumerate_lines(
            doc, include_appendix=include_appendix,
            source_lines_path=source_lines_path)
        line_by_id = {g: (pi, r, t, section) for g, pi, r, t, section in lines}

        # (kind, num, section) -> set of global line IDs in the caption group.
        groups = {}
        for s in segs:
            role = s.get("role")
            if role == "figure_caption":
                kind = "Figure"
            elif role == "table_title":
                kind = "Table"
            else:
                continue
            num = s.get("num")
            kind = family_name(s.get("family"), role)
            if num is None:
                # Unnumbered main captions are not asset anchors (consistent with caption_cut); skip
                print(f"[extract_captions] skipping {role} segment without num "
                      f"start={s.get('start')} end={s.get('end')}", file=sys.stderr)
                continue
            segment_ids = [
                g for g in range(s.get("start", 0), s.get("end", -1) + 1)
                if g in line_by_id
            ]
            if not segment_ids:
                continue
            section = line_by_id[segment_ids[0]][3]
            key = (kind, num, section)
            groups.setdefault(key, set()).update(segment_ids)

        out = []
        for (kind, num, section), ids in groups.items():
            ordered = sorted(ids)  # reading order by global_id
            if not ordered:
                continue
            label = f"{kind} {num}"
            first_pi = line_by_id[ordered[0]][0]

            pages_in_group = {line_by_id[g][0] for g in ordered}
            if len(pages_in_group) > 1:
                print(f"[extract_captions] {label} spans pages "
                      f"{sorted(p + 1 for p in pages_in_group)}; "
                      f"using first page {first_pi + 1} for page/bbox", file=sys.stderr)

# Concatenate all lines in reading order and normalize whitespace.
            raw_parts = [line_by_id[g][2] for g in ordered]
            text = " ".join(p.strip() for p in raw_parts if p.strip())
            text = re.sub(r"\s+", " ", text).strip()

            # Union line bboxes on the first page of the caption group.
            block = None
            for g in ordered:
                pi, r, _t, _section = line_by_id[g]
                if pi != first_pi:
                    continue
                block = fitz.Rect(r) if block is None else (block.include_rect(r) or block)
            item = {
                "page": first_pi + 1,
                "label": label,
                "family": kind,
                "text": text,
                "bbox": [round(block.x0, 1), round(block.y0, 1),
                         round(block.x1, 1), round(block.y1, 1)],
            }
            if include_appendix:
                item["section"] = section
            out.append(item)

        out.sort(key=lambda c: (c["page"], c["bbox"][1]))
        return out
    finally:
        doc.close()


def main():
    ap = argparse.ArgumentParser(description="Extract full multi-line captions from a paper PDF")
    ap.add_argument("pdf", help="Path to paper PDF")
    ap.add_argument("outdir", help="Output directory")
    ap.add_argument("--seg", help="Path to segments2.json for lookup mode (falls back to geometric mode if absent)")
    ap.add_argument("--source-lines", help="Path to source_lines.json paired with segments2 (default: sibling file)")
    ap.add_argument("--include-appendix", action="store_true",
                    help="Include appendix/supplementary pages in lookup mode; References are always skipped")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    captions = None
    if args.seg:
        seg_path = Path(args.seg)
        try:
            captions = captions_from_segments(
                pdf_path, seg_path, include_appendix=args.include_appendix,
                source_lines_path=args.source_lines)
            scope = "body+appendix" if args.include_appendix else "body"
            print(f"[extract_captions] lookup mode: {scope} (References excluded)", file=sys.stderr)
        except Exception as e:  # Fall back to geometric extraction on any load/parse error
            print(f"[extract_captions] failed to load --seg ({e}); falling back to geometric mode", file=sys.stderr)
    if captions is None:
        captions = extract_captions(pdf_path)
    out_path = outdir / "captions.json"
    out_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[extract_captions] {len(captions)} captions -> {out_path}")
    for c in captions[:5]:
        snippet = c["text"][:80]
        more = "..." if len(c["text"]) > 80 else ""
        print(f"  p{c['page']} {c['label']:10s} {snippet}{more}")
    if len(captions) > 5:
        print(f"  ... total {len(captions)}")


if __name__ == "__main__":
    main()
