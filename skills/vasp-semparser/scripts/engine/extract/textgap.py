#!/usr/bin/env python3
"""TextGap figure locator: localize figure bounding boxes via text-bracket + column detection.

Core idea:
  - Figure captions ("Figure N:") determine figure count, page, and identity.
  - Vertical bounds sit between the body text above and the caption below.
  - Column detection determines horizontal bounds.

Three-tier upper-bound fallback:
  1) Text bracketing: bottom of the lowest regular body line above the caption.
  2) Rule 1 (running header): repeated cross-page running-header bottom as top floor.
  3) Rule 2 (graphic top): tighten to the top of graphic primitives when large tables or
     whitespace separate the body text above from the figure.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageChops, ImageDraw

# Match only line-initial figure captions ("Figure N:", "Fig. A.1.", "Figure S2:").
# Keeps numbering as a string to support appendix formats.
FIGURE_ID_RE = r"(?:[A-Z]\.?)?\d+(?:\.\d+)*"
CAPTION_RE = re.compile(
    rf"^\s*(?P<label>(?:Figure|Fig\.?)\s+{FIGURE_ID_RE})\s*[:.]",
    re.IGNORECASE,
)

ZOOM = 6.0          # Figure render scale (72 * 6 = 432 dpi)
OVERLAY_ZOOM = 2.0  # Full-page debug overlay render scale
PAD_TOP = 24        # Top padding (pt) when Rule 2 tightens to graphic top (keeps panel titles/legends)


# ---------------- Text lines & column detection ----------------
def page_lines(page):
    """All non-empty text lines on the page: [(fitz.Rect, text), ...]."""
    out = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            txt = "".join(sp.get("text", "") for sp in ln.get("spans", []))
            if txt.strip():
                out.append((fitz.Rect(ln["bbox"]), txt))
    return out


def detect_columns(page, lines):
    """Detect 1 or 2 column x-intervals from the text-line left-edge histogram."""
    pw = page.rect.width
    full = [fitz.Rect(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1)]
    x0s = [int(r.x0) for r, _ in lines if 50 <= r.width <= 0.85 * pw]
    if len(x0s) < 8:
        return full
    buckets = Counter((x // 5) * 5 for x in x0s)
    total = sum(buckets.values())
    peaks = sorted(x for x, c in buckets.items() if c >= 0.10 * total)
    if not peaks:
        return full
    merged = []
    for p in peaks:
        if not merged or p - merged[-1] >= 30:
            merged.append(p)
    if len(merged) > 2:  # Academic papers have at most 2 columns; extra peaks are figure/table noise
        merged = sorted(sorted(merged, key=lambda x: -buckets[x])[:2])
    cols = []
    for i, left in enumerate(merged):
        right = merged[i + 1] - 4 if i + 1 < len(merged) else page.rect.x1
        cols.append(fitz.Rect(left, page.rect.y0, right, page.rect.y1))
    return cols


def col_of(rect, cols):
    """Assign rect to a column: returns (index, col_rect), or (-1, full_text_rect) for full-width."""
    if len(cols) <= 1:
        return -1, cols[0]
    text_area = fitz.Rect(cols[0].x0, cols[0].y0, cols[-1].x1, cols[-1].y1)
    if rect.width > 0.75 * (cols[-1].x1 - cols[0].x0):
        return -1, text_area
    best, bi = 0.0, None
    for i, c in enumerate(cols):
        ov = max(0.0, min(rect.x1, c.x1) - max(rect.x0, c.x0))
        if ov > best:
            best, bi = ov, i
    return (bi, cols[bi]) if bi is not None else (-1, text_area)


# ---------------- Caption localization ----------------
def find_captions(page, lines):
    """Return y-sorted figure captions [(label, cap_line_rect), ...] matching line-initial captions."""
    out, seen = [], set()
    for r, txt in lines:
        m = CAPTION_RE.match(txt)
        if not m:
            continue
        label = re.sub(r"^(?:Figure|Fig\.?)", "Figure", m.group("label"), flags=re.IGNORECASE)
        label = re.sub(r"\s+", " ", label).strip()
        if label in seen:
            continue
        seen.add(label)
        out.append((label, fitz.Rect(r)))
    out.sort(key=lambda t: t[1].y0)
    return out


def is_body_line(r, txt, col_rect):
    """Regular body line: wide enough, left-aligned to column margin, and not a caption."""
    cw = col_rect.width
    if CAPTION_RE.match(txt):
        return False
    if r.width < 0.55 * cw:
        return False
    if abs(r.x0 - col_rect.x0) > 0.10 * cw:
        return False
    return True


REFERENCE_HEADINGS = {"references", "bibliography"}
APPENDIX_HEADINGS = {
    "appendix",
    "appendices",
    "supplement",
    "supplementary",
    "supplementary material",
    "supplementary materials",
    "supplementary information",
}
APPENDIX_FLOAT_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?|Table)\s+(?:[A-Z]\.?\d+(?:\.\d+)*)\s*[:：]",
    re.IGNORECASE,
)


def _section_heading_kind(text):
    """Conservatively recognize a standalone References/Appendix heading."""
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    if normalized in REFERENCE_HEADINGS:
        return "references"
    if normalized in APPENDIX_HEADINGS:
        return "appendix"
    # Accept compact labels such as "Appendix A" and "Appendix A: Proofs", while
    # rejecting prose references such as "Appendix A for more details".
    if re.fullmatch(r"appendix\s+[a-z0-9]+[.:]?", normalized):
        return "appendix"
    if re.fullmatch(r"appendix\s+[a-z0-9]+\s*[:\-–—]\s*[^();]{1,70}", normalized):
        return "appendix"
    return None


def find_section_starts(doc):
    """Return zero-based starts for References and Appendix/Supplement.

    The two boundaries are intentionally independent: many papers place an
    appendix after their bibliography, so a single cutoff would discard useful
    appendix figures or, if removed, feed reference pages to the semantic model.
    """
    starts = {"references": None, "appendix": None}
    for pi in range(len(doc)):
        if pi < 2:
            continue
        for _rect, text in page_lines(doc[pi]):
            kind = _section_heading_kind(text)
            if kind and starts[kind] is None:
                starts[kind] = pi
        if all(value is not None for value in starts.values()):
            break
    return starts


def sectioned_document_lines(doc):
    """Return PDF lines tagged with their page-internal document section.

    PyMuPDF preserves the PDF's extraction order, which is normally column
    reading order. Switching state at the actual standalone section-heading
    line therefore keeps body figures that occur before ``References`` on the
    same page, including two-column pages where a simple horizontal y cutoff
    would be wrong. The conservative heading recognizer remains the sole
    section-transition signal.

    The return value is a page-aligned list whose entries contain
    ``(rect, text, section)`` tuples. Section state carries across pages and
    can switch more than once (for example body -> references -> appendix).
    """
    pages = []
    section = "body"
    for pi in range(len(doc)):
        raw_rows=[(fitz.Rect(rect),text) for rect,text in page_lines(doc[pi])]
        appendix_heading_index=next(
            (index for index,(_rect,text) in enumerate(raw_rows)
             if pi>=2 and _section_heading_kind(text)=="appendix"),
            None,
        )
        # A leading appendix float can be painted before the printed Appendix
        # heading on the same page. If state carried from the previous page is
        # References, extraction order alone would discard that float. An
        # explicit appendix-labelled caption is strong evidence to retain the
        # page prefix; final citations remain references without such an anchor.
        appendix_prefix=(
            section=="references"
            and appendix_heading_index is not None
            and any(APPENDIX_FLOAT_RE.match(text)
                    for _rect,text in raw_rows[:appendix_heading_index])
        )
        tagged = []
        for index,(rect,text) in enumerate(raw_rows):
            kind = _section_heading_kind(text) if pi >= 2 else None
            if kind:
                section = kind
            row_section=("appendix" if appendix_prefix
                         and index<appendix_heading_index else section)
            tagged.append((fitz.Rect(rect), text, row_section))
        pages.append(tagged)
    return pages


def content_document_lines(doc, include_appendix=False):
    """Return page-aligned non-bibliography lines with section tags."""
    allowed = {"body", "appendix"} if include_appendix else {"body"}
    return [
        [
            (rect, text, section)
            for rect, text, section in rows
            if section in allowed
        ]
        for rows in sectioned_document_lines(doc)
    ]


def load_source_document_lines(path, include_appendix=False, page_count=None):
    """Load the stable line IDs emitted beside ``segments2.json``.

    ``segments2`` ranges refer to these IDs, not to a fresh enumeration of the
    current PDF extractor output. Reading the cached rows is therefore required
    when extraction code changes line grouping between the semantic pass and
    deterministic consumers such as caption cutting.

    Returns ``(id, page_index, rect, text, section)`` tuples ordered by ID.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("source_lines.json must contain a list")
    allowed = {"body", "appendix"} if include_appendix else {"body"}
    rows = []
    seen = set()
    for pos, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"source line {pos} is not an object")
        line_id = item.get("id")
        page = item.get("page")
        bbox = item.get("bbox")
        text = item.get("text")
        if isinstance(line_id, bool) or not isinstance(line_id, int):
            raise ValueError(f"source line {pos} has invalid id")
        if line_id in seen:
            raise ValueError(f"duplicate source line id {line_id}")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"source line {line_id} has invalid page")
        if page_count is not None and page > page_count:
            raise ValueError(f"source line {line_id} points past the PDF")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"source line {line_id} has invalid bbox")
        if not isinstance(text, str):
            raise ValueError(f"source line {line_id} has invalid text")
        section = item.get("section", "body")
        seen.add(line_id)
        if section not in allowed:
            continue
        rect = fitz.Rect(bbox)
        if not all(math.isfinite(value) for value in rect) or not rect.is_valid:
            raise ValueError(f"source line {line_id} has non-finite bbox")
        rows.append((line_id, page - 1, rect, text, section))
    rows.sort(key=lambda row: row[0])
    return rows


def classify_document_pages(doc):
    """Classify every PDF page as body, references, or appendix."""
    starts = find_section_starts(doc)
    refs = starts["references"]
    appendix = starts["appendix"]
    sections = []
    for pi in range(len(doc)):
        if refs is None and appendix is None:
            section = "body"
        elif refs is None:
            section = "appendix" if pi >= appendix else "body"
        elif appendix is None:
            section = "references" if pi >= refs else "body"
        elif refs < appendix:
            section = "body" if pi < refs else ("references" if pi < appendix else "appendix")
        else:
            section = "body" if pi < appendix else ("appendix" if pi < refs else "references")
        sections.append(section)
    return sections


def content_page_indices(doc, include_appendix=False):
    """Return pages containing eligible lines, never bibliography-only pages."""
    return [
        pi
        for pi, rows in enumerate(
            content_document_lines(doc, include_appendix=include_appendix)
        )
        if rows
    ]


def find_appendix_start(doc):
    """Backward-compatible first non-body page (1-indexed), or ``None``."""
    starts = [value for value in find_section_starts(doc).values() if value is not None]
    return min(starts) + 1 if starts else None


# ---------------- Rule 1: Running-header band detection ----------------
def detect_header_band(doc, top_frac=0.14, min_pages_frac=0.30):
    """Detect cross-page repeated top-of-page text (running header); return its bottom y (or 0).

    Uses only the topmost repeated line to avoid misclassifying body text as header chrome.
    """
    counts = defaultdict(list)  # normalized text -> [(y0, y1), ...]
    for pi in range(len(doc)):
        page = doc[pi]
        ph = page.rect.height
        for b in page.get_text("dict").get("blocks", []):
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                r = fitz.Rect(ln["bbox"])
                if r.y0 > ph * top_frac:
                    continue
                txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if len(txt) < 5:
                    continue
                # Filter pure numeric/symbol strings (e.g. "14.4±2.0") from figures/tables
                alpha = sum(1 for c in txt if c.isalpha())
                if alpha < 3:
                    continue
                counts[re.sub(r"\d+", "#", txt.lower())].append((r.y0, r.y1))
    thresh = max(2, len(doc) * min_pages_frac)
    repeated = [rows for rows in counts.values() if len(rows) >= thresh]
    if not repeated:
        return 0.0

    def med(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    top_hdr = min(repeated, key=lambda rows: med([y0 for y0, _ in rows]))
    return med([y1 for _, y1 in top_hdr])


# ---------------- Rule 2: Graphic primitive top ----------------
def collect_graphic_rects(page):
    """Collect page graphic rectangles (raster + vector), filtering hairlines and large white fills."""
    pa = page.rect.width * page.rect.height
    rects = []
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                if r.width >= 8 and r.height >= 8:
                    rects.append(fitz.Rect(r))
        except Exception:
            pass
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if not r:
                continue
            r = fitz.Rect(r)
            if r.width < 8 or r.height < 8:
                continue
            area = r.width * r.height
            fill = d.get("fill")
            is_white = fill is not None and len(fill) >= 3 and all(c >= 0.92 for c in fill[:3])
            if area > 0.40 * pa:          # full-page background fill
                continue
            if is_white and area > 0.15 * pa:  # decorative white background
                continue
            rects.append(r)
    except Exception:
        pass
    return rects


def fig_top_above(grects, cap_top, col_rect, floor):
    """Minimum y0 of graphic elements in the same column within (floor, cap_top), or None."""
    tops = []
    for r in grects:
        cx = (r.x0 + r.x1) / 2
        if not (col_rect.x0 - 5 <= cx <= col_rect.x1 + 5):
            continue
        if r.y0 >= cap_top - 2 or r.y1 <= floor:
            continue
        tops.append(r.y0)
    return min(tops) if tops else None


# ---------------- Core: Text bracket + two rules ----------------
def locate_page(page, header_bottom=0.0):
    """Compute figure bboxes on a page via text bracketing and Rules 1-2; return (results, cols, caps)."""
    lines = page_lines(page)
    cols = detect_columns(page, lines)
    caps = find_captions(page, lines)
    grects = collect_graphic_rects(page)
    results = []
    for i, (label, cap_rect) in enumerate(caps):
        ci, col_rect = col_of(cap_rect, cols)
        cap_top = cap_rect.y0

        # Floor: bottom of the nearest caption above in the same column
        floor = page.rect.y0
        for j in range(i):
            cj, _ = col_of(caps[j][1], cols)
            if ci == -1 or cj == -1 or cj == ci:
                floor = max(floor, caps[j][1].y1)

        # Text bracket: bottom of the lowest regular body line above the caption in the same column
        body_top = floor
        for r, txt in lines:
            if r.y1 > cap_top - 2 or r.y1 < floor:
                continue
            rci, _ = col_of(r, cols)
            if not (ci == -1 or rci == -1 or rci == ci):
                continue
            if is_body_line(r, txt, col_rect):
                body_top = max(body_top, r.y1)

        # Start from the text-bracket bottom and tighten downward via Rules 1 and 2;
        # rules only push the upper bound down and never clip into the figure.
        y_top, top_src = body_top, "body"
        # Rule 1 (running header): fallback for top-of-page figures with no body text above
        if body_top < header_bottom <= cap_top - 20:
            y_top, top_src = header_bottom, "hdr"
        # Rule 2 (graphic top): tighten when a table or large gap separates body text from the figure
        ftop = fig_top_above(grects, cap_top, col_rect, floor)
        if ftop is not None and (ftop - body_top) > 40:
            gt = max(floor, ftop - PAD_TOP)
            if gt > y_top and gt <= cap_top - 5:
                y_top, top_src = gt, "fig"

        bbox = fitz.Rect(col_rect.x0, y_top + 2, col_rect.x1, cap_top - 1)
        note = "degenerate" if (bbox.height < 20 or bbox.width < 20) else ""
        results.append({
            "label": label,
            "bbox": [round(bbox.x0, 1), round(bbox.y0, 1), round(bbox.x1, 1), round(bbox.y1, 1)],
            "column": "full" if ci == -1 else f"col-{ci}",
            "top_src": top_src,
            "note": note,
        })
    return results, cols, caps


# ---------------- Rendering + autotrim + overlay ----------------
def autotrim(img, pad=4):
    """Trim pure-white borders using PIL (no numpy dependency)."""
    rgb = img.convert("RGB")
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255)))
    bb = diff.getbbox()
    if not bb:
        return img
    x0, y0, x1, y1 = bb
    W, H = rgb.size
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)))


def render_bbox(page, bbox, out_png):
    clip = fitz.Rect(*bbox)
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=clip, alpha=False)
    img = autotrim(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    img.save(out_png)
    return img.size


def render_overlay(page, results, out_png):
    z = OVERLAY_ZOOM
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    dr = ImageDraw.Draw(img)
    for r in results:
        x0, y0, x1, y1 = [v * z for v in r["bbox"]]
        color = (240, 160, 0) if r["note"] else (220, 30, 30)
        dr.rectangle([x0, y0, x1, y1], outline=color, width=3)
        dr.text((x0 + 4, max(0, y0 - 12)), f"{r['label']} [{r['top_src']}]", fill=color)
    img.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("outdir")
    args = ap.parse_args()
    out = Path(args.outdir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "overlay").mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    appx = find_appendix_start(doc)
    hb = detect_header_band(doc)
    manifest = []
    for pi in range(len(doc)):
        page = doc[pi]
        results, cols, caps = locate_page(page, hb)
        if not results:
            continue
        for r in results:
            r["page"] = pi + 1
            r["n_columns"] = len(cols)
            r["section"] = "appendix" if (appx and pi + 1 >= appx) else "body"
            if not r["note"] and r["section"] == "body":
                safe = re.sub(r"\W+", "", r["label"]).lower()
                fname = f"page{pi + 1}_{safe}.png"
                w, h = render_bbox(page, r["bbox"], out / "figures" / fname)
                r["file"] = f"figures/{fname}"
                r["render_wh"] = [w, h]
            manifest.append(r)
        if not appx or pi + 1 < appx:  # draw overlays for body pages only
            render_overlay(page, results, out / "overlay" / f"page{pi + 1}.png")

    (out / "figures.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    body = [m for m in manifest if m["section"] == "body"]
    appx_n = len(manifest) - len(body)
    degen = sum(1 for m in body if m["note"])
    n_fig = sum(1 for m in body if m["top_src"] == "fig")
    print(f"[textgap] {len(manifest)} figures — body {len(body)} / appendix {appx_n}, "
          f"body_fallback={degen}, rule2_tightened={n_fig}, appendix_start={appx}, "
          f"header_bottom={round(hb, 1)}, total_pages={len(doc)} -> {out}")
    for m in body:
        bb = [round(v) for v in m["bbox"]]
        print(f"  p{m['page']} {m['label']:10s} {m['column']:6s} top={m['top_src']:4s} bbox={bb} {m.get('note', '')}")


if __name__ == "__main__":
    main()
