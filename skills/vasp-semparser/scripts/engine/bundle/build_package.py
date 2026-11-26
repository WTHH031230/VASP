#!/usr/bin/env python3
"""Build one neutral paper-asset bundle.

Output layout::

    <out>/manifest.json
    <out>/assets/meta/text.txt
    <out>/assets/meta/captions.json
    <out>/assets/meta/captions_full.json
    <out>/assets/meta/metadata.json
    <out>/assets/meta/segments2.json
    <out>/assets/meta/source_lines.json
    <out>/assets/meta/structured_elements.json
    <out>/assets/meta/figures.json
    <out>/assets/meta/appendix_figures.json
    <out>/assets/figures/*.png
    <out>/assets/appendix_figures/*.png
    <out>/assets/logos/*.{png,jpg,svg,gif}
    <out>/assets/qr/*.png

The public figure directories contain only Figure crops. Appendix figures are a
separate, secondary pool. Table crops produced by caption_cut remain diagnostic
intermediates; tables, equations, and algorithms are published as structured data
for native re-rendering by a later poster pipeline.

The Figure cropper and its manifest geometry are preserved. When a PDF is supplied,
this packager deterministically derives each figure's page-column assignment from
the existing text-gap detector. Captions supplied as a list are normalized to a
label-to-text mapping while the lossless list is retained as captions_full.json.

Core assets are text, captions, metadata, figures, line roles, and structured
elements. Missing core inputs produce an INCOMPLETE manifest and exit status 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LAYOUT_VERSION = "v4-assets"
SCHEMA_VERSION = "dp-p2p.pipeline.bundle.v4"



# ---------- Utility helpers ----------

def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_png_size(path: Path) -> tuple[int, int]:
    """Read PNG pixel dimensions via PIL, falling back to raw IHDR header parsing."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        pass
    try:
        data = path.read_bytes()[:33]
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return (w, h)
    except Exception:
        pass
    return (0, 0)


def place(src: Path, dst: Path, *, symlink: bool) -> None:
    """Place a file at dst (copy2 by default; relative symlink with --symlink)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(rel)
    else:
        shutil.copy2(src, dst)



BRAND_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif"}


def place_optional_asset_dir(
    src_dir: Path | None,
    dst_dir: Path,
    *,
    allowed_extensions: set[str],
    symlink: bool,
) -> list[str]:
    """Synchronize one best-effort asset directory and remove stale files."""
    selected = [] if not src_dir or not src_dir.is_dir() else [
        path for path in sorted(src_dir.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    selected_names = {path.name for path in selected}
    if dst_dir.exists():
        for stale in dst_dir.iterdir():
            if stale.is_file() and stale.name not in selected_names:
                stale.unlink()
    if selected:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for source in selected:
            place(source, dst_dir / source.name, symlink=symlink)
    return sorted(selected_names)


def place_figure_dir(src_dir: Path | None, dst_dir: Path, sidecar: Path | None,
                     *, symlink: bool) -> list[str]:
    """Sync Figure PNGs only; Table crops stay upstream for diagnostics."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir or not src_dir.is_dir():
        return []

    allowed = None
    doc = load_json(sidecar) if sidecar else None
    if isinstance(doc, list):
        typed = [it for it in doc if isinstance(it, dict)
                 and (it.get("kind") in ("figure", "table") or it.get("label"))]
        if typed:
            allowed = {
                Path(str(it.get("file", ""))).name
                for it in typed
                if it.get("kind") == "figure"
                or (not it.get("kind") and str(it.get("label", "")).startswith("Figure"))
            }

    selected = [
        src for src in sorted(src_dir.glob("*.png"))
        if src.is_file() and (allowed is None or src.name in allowed)
    ]
    selected_names = {src.name for src in selected}

    # assets/figures is managed exclusively by the bundler; remove stale Table/Figure crops.
    for stale in dst_dir.glob("*.png"):
        if (stale.is_file() or stale.is_symlink()) and stale.name not in selected_names:
            stale.unlink()

    for src in selected:
        place(src, dst_dir / src.name, symlink=symlink)
    return sorted(selected_names)


# ---------- Deterministic figures.json column assignment (via extract/textgap) ----------

def _load_textgap_cols():
    """Lazily import page_lines/detect_columns from extract/textgap when --pdf is given.

    Uses only its pure geometric column detection; import failure falls back to resolver=None.
    """
    extract_dir = Path(__file__).resolve().parent.parent / "extract"
    if str(extract_dir) not in sys.path:
        sys.path.insert(0, str(extract_dir))
    from textgap import detect_columns, page_lines  # noqa: E402
    return page_lines, detect_columns


def derive_column(x0: float, x1: float, cols: list) -> tuple[str, int | None]:
    """Assign figure horizontal span [x0, x1] to a column by majority overlap.

    Overhang of a few pt into the gutter is normal, so strict containment misclassifies
    single-column figures as full-width. A column is occupied if overlap exceeds half
    its width: >=2 columns -> "full", 1 column -> "col-i", 0 columns -> "full".
    Single-column pages always return ("full", max(1, len(cols))).
    """
    if len(cols) <= 1:
        return "full", max(1, len(cols))
    occupied = [
        i for i, c in enumerate(cols)
        if c.x1 > c.x0 and (min(x1, c.x1) - max(x0, c.x0)) > (c.x1 - c.x0) / 2.0
    ]
    if len(occupied) == 1:
        return f"col-{occupied[0]}", len(cols)
    return "full", len(cols)


class ColumnResolver:
    """Cache per-page detect_columns results to derive column/num_columns for each figure.

    Created only when --pdf exists and textgap imports cleanly; resolve() returns
    (None, None) on out-of-range pages or missing bboxes so the caller falls back to sidecar.
    """

    def __init__(self, pdf_path: Path):
        import fitz  # lazy import: not needed when --pdf is omitted
        self._doc = fitz.open(str(pdf_path))
        self._page_lines, self._detect_columns = _load_textgap_cols()
        self._cache: dict[int, list] = {}

    def _cols_for_page(self, page: int) -> list:
        if page not in self._cache:
            pg = self._doc[page - 1]
            self._cache[page] = self._detect_columns(pg, self._page_lines(pg))
        return self._cache[page]

    def resolve(self, page: Any, bbox: Any) -> tuple[str | None, int | None]:
        if not isinstance(page, int) or not (1 <= page <= len(self._doc)):
            return None, None
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            return None, None
        try:
            return derive_column(float(bbox[0]), float(bbox[2]), self._cols_for_page(page))
        except Exception:
            return None, None

    def close(self) -> None:
        try:
            self._doc.close()
        except Exception:
            pass


# ---------- figures.json normalization ----------

def resolve_figures_source(figures_dir: Path | None) -> tuple[Path | None, Path | None]:
    """Discover the PNG directory and optional sidecar JSON under --figures-dir."""
    if not figures_dir or not figures_dir.exists():
        return None, None
    sub = figures_dir / "figures"
    png_dir = sub if sub.is_dir() else figures_dir
    sidecar = None
    for base in (figures_dir, png_dir):
        for name in ("figures.json", "manifest.json"):
            cand = base / name
            if cand.exists():
                sidecar = cand
                break
        if sidecar:
            break
    return png_dir, sidecar


def resolve_appendix_source(figures_dir: Path | None) -> Path | None:
    """Return the optional appendix Figure crop directory."""
    if not figures_dir:
        return None
    path = figures_dir / "appendix_figures"
    return path if path.is_dir() else None


def resolve_segments_source(figures_dir: Path | None, png_dir: Path | None) -> Path | None:
    """Locate segments2.json in --figures-dir or its figures/ subdirectory; return None if absent."""
    if not figures_dir:
        return None
    for base in (figures_dir, png_dir):
        if base:
            cand = base / "segments2.json"
            if cand.is_file():
                return cand
    return None


def resolve_structured_source(figures_dir: Path | None, png_dir: Path | None) -> Path | None:
    """Locate co-produced structured_elements.json in the caption-cut directory."""
    if not figures_dir:
        return None
    for base in (figures_dir, png_dir):
        if base:
            cand = base / "structured_elements.json"
            if cand.is_file():
                return cand
    return None

def resolve_source_lines_source(figures_dir: Path | None, png_dir: Path | None) -> Path | None:
    """Locate source_lines.json mapping source_line_ids to original text, page, and bbox."""
    if not figures_dir:
        return None
    for base in (figures_dir, png_dir):
        if base:
            cand = base / "source_lines.json"
            if cand.is_file():
                return cand
    return None


def _assign_columns(entries: list[dict], resolver: ColumnResolver | None) -> None:
    """Populate column/num_columns on each figures.json entry in place and drop legacy layout key.

    Derives deterministically from PDF when resolver + page + bbox are available; otherwise
    falls back to sidecar column/layout, or "full" / None with a stderr warning.
    """
    for e in entries:
        page = e.get("page")
        # _bbox is temporary for column derivation and popped before writing output
        bbox = e.pop("_bbox", None)
        if bbox is None:
            bbox = e.get("bbox")
        col, ncol = (None, None)
        if resolver is not None:
            col, ncol = resolver.resolve(page, bbox)
        if col is None:
            if e.get("column"):
                col, ncol = e["column"], e.get("num_columns")
            elif e.get("layout"):
                col, ncol = e["layout"], e.get("num_columns")
            else:
                col, ncol = "full", None
                print(f"[build_package] WARNING: unable to derive column (file={e.get('file')}, "
                      f"page={page}); falling back to column=full / num_columns=null", file=sys.stderr)
        e["column"] = col
        e["num_columns"] = ncol
        e.pop("layout", None)  # downstream consumes column; drop legacy layout field


def build_figures_json(png_dir: Path | None, sidecar: Path | None, copied_names: list[str],
                       resolver: ColumnResolver | None, asset_subdir="figures") -> list[dict]:
    if not copied_names:
        return []
    name_set = set(copied_names)
    entries: list[dict] = []

    if sidecar is not None and sidecar.name == "figures.json":
        doc = load_json(sidecar)
        if isinstance(doc, list):
            entries = [e for e in doc if isinstance(e, dict) and Path(str(e.get("file", ""))).name in name_set]

    if not entries and sidecar is not None and sidecar.name == "manifest.json":
        doc = load_json(sidecar)
        if isinstance(doc, list):
            for item in doc:
                if not isinstance(item, dict) or item.get("degen") or not item.get("file"):
                    continue
                fname = Path(str(item["file"])).name
                if fname not in name_set:
                    continue
                w, h = read_png_size(png_dir / fname) if png_dir else (0, 0)
                entry = {
                    "file": f"{asset_subdir}/{fname}",
                    "width": w,
                    "height": h,
                    "page": item.get("page"),
                    "_bbox": item.get("bbox"),  # temporary field popped by _assign_columns
                }
                if item.get("kind"):
                    entry["kind"] = item["kind"]
                if item.get("label"):
                    entry["label"] = item["label"]
                for field in ("section", "priority", "family"):
                    if item.get(field):
                        entry[field] = item[field]
                entries.append(entry)

    if not entries:
        # No usable sidecar (or empty after normalization): synthesize entries from copied PNGs
        for fname in copied_names:
            w, h = read_png_size(png_dir / fname) if png_dir else (0, 0)
            entries.append({"file": f"{asset_subdir}/{fname}", "width": w, "height": h, "page": None})

    _assign_columns(entries, resolver)
    if sidecar is not None and sidecar.name == "manifest.json":
        source_doc = load_json(sidecar)
        source_by_name = (
            {Path(str(it.get("file", ""))).name: it for it in source_doc
             if isinstance(it, dict) and it.get("file")}
            if isinstance(source_doc, list) else {}
        )
        for entry in entries:
            entry["bbox"] = source_by_name.get(Path(str(entry.get("file", ""))).name, {}).get("bbox")
    return entries


# ---------- Main entry point ----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-dir", type=Path, help="Directory containing figure crops and sidecars")
    ap.add_argument("--text", type=Path, help="Path to text.txt")
    ap.add_argument("--captions", type=Path, help="Path to captions.json")
    ap.add_argument("--metadata", type=Path, help="Path to neutral metadata.json")
    ap.add_argument("--logos-dir", type=Path, help="Optional directory of institution logos (best-effort)")
    ap.add_argument("--qr-dir", type=Path, help="Optional directory of QR PNGs (best-effort)")
    ap.add_argument("--pdf", type=Path, help="Optional source paper PDF (for SHA-256 and column detection)")
    ap.add_argument("--out", required=True, type=Path, help="Output bundle root directory")
    ap.add_argument("--symlink", action="store_true",
                    help="Use relative symlinks instead of copying files (saves disk space)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out = args.out
    meta_dir = out / "assets" / "meta"
    figures_dst = out / "assets" / "figures"
    appendix_figures_dst = out / "assets" / "appendix_figures"
    logos_dst = out / "assets" / "logos"
    qr_dst = out / "assets" / "qr"

    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[build_package] Failed to create output directory {out}: {e}", file=sys.stderr)
        return 2

    missing_core: list[str] = []
    files_map: dict[str, str] = {}

    # ---- Core single-file assets (warn on missing; captions handled separately) ----
    core_files = [
        ("text", args.text, meta_dir / "text.txt", "--text"),
        ("metadata", args.metadata, meta_dir / "metadata.json", "--metadata"),
    ]
    for key, src, dst, flag in core_files:
        if src and src.exists():
            place(src, dst, symlink=args.symlink)
            files_map[key] = dst.relative_to(out).as_posix()
            print(f"[build_package] {key} <- {src}", file=sys.stderr)
        else:
            missing_core.append(key)
            detail = f" ({src})" if src else " (not provided)"
            print(f"[build_package] WARNING: {flag} missing or not found{detail}; "
                  f"omitting from assets/meta/", file=sys.stderr)

    # ---- optional neutral identity/link assets; never make the bundle incomplete ----
    copied_logos = place_optional_asset_dir(
        args.logos_dir, logos_dst, allowed_extensions=BRAND_EXTENSIONS,
        symlink=args.symlink)
    copied_qr = place_optional_asset_dir(
        args.qr_dir, qr_dst, allowed_extensions={".png"}, symlink=args.symlink)
    if copied_logos:
        files_map["logos_dir"] = "assets/logos"
    if copied_qr:
        files_map["qr_dir"] = "assets/qr"
    print(
        f"[build_package] brand assets: logos={len(copied_logos)} qr={len(copied_qr)}",
        file=sys.stderr,
    )

    # ---- captions: normalize list -> dict + captions_full; keep dict verbatim ----
    captions_count = 0
    cap_dst = meta_dir / "captions.json"
    if args.captions and args.captions.exists():
        meta_dir.mkdir(parents=True, exist_ok=True)
        doc = load_json(args.captions)
        if isinstance(doc, list):
            cap_map: dict[str, str] = {}
            for it in doc:
                if not isinstance(it, dict):
                    continue
                label = it.get("label")
                if label is None or str(label) in cap_map:  # keep first occurrence per label
                    continue
                cap_map[str(label)] = it.get("text", "")
            cap_dst.write_text(json.dumps(cap_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            full_dst = meta_dir / "captions_full.json"
            full_dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            files_map["captions"] = cap_dst.relative_to(out).as_posix()
            files_map["captions_full"] = full_dst.relative_to(out).as_posix()
            captions_count = len(cap_map)
            print(f"[build_package] captions <- {args.captions} "
                  f"(list->dict {captions_count} entries + captions_full)", file=sys.stderr)
        elif isinstance(doc, dict):
            cap_dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            files_map["captions"] = cap_dst.relative_to(out).as_posix()
            captions_count = len(doc)
            print(f"[build_package] captions <- {args.captions} (dict {captions_count} entries, verbatim)",
                  file=sys.stderr)
        else:
            missing_core.append("captions")
            print(f"[build_package] WARNING: --captions parse failed or non-list/dict ({args.captions}); "
                  f"omitting captions.json", file=sys.stderr)
    else:
        missing_core.append("captions")
        detail = f" ({args.captions})" if args.captions else " (not provided)"
        print(f"[build_package] WARNING: --captions missing or not found{detail}; "
              f"omitting captions.json", file=sys.stderr)

    # ---- figures (auto-detect PNG directory and sidecar layout) ----
    png_dir, sidecar = resolve_figures_source(args.figures_dir)
    copied_figs = place_figure_dir(png_dir, figures_dst, sidecar, symlink=args.symlink)
    sidecar_doc = load_json(sidecar) if sidecar else None
    explicit_zero_figures = isinstance(sidecar_doc, list) and not any(
        isinstance(it, dict) and (it.get("kind") == "figure"
        or str(it.get("label", "")).startswith("Figure")) for it in sidecar_doc)
    if not args.figures_dir:
        missing_core.append("figures")
        print("[build_package] WARNING: --figures-dir not provided; assets/figures/ will be empty", file=sys.stderr)
    elif not copied_figs and not explicit_zero_figures:
        missing_core.append("figures")
        print(f"[build_package] WARNING: no PNGs found under --figures-dir ({args.figures_dir})", file=sys.stderr)
    # Column resolver: enabled only when --pdf exists and textgap imports; otherwise falls back to sidecar
    resolver = None
    if args.pdf and args.pdf.exists():
        try:
            resolver = ColumnResolver(args.pdf)
        except Exception as e:
            print(f"[build_package] WARNING: failed to initialize column resolver ({e}); "
                  f"falling back to sidecar column", file=sys.stderr)
    figures_json = build_figures_json(png_dir, sidecar, copied_figs, resolver)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "figures.json").write_text(
        json.dumps(figures_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    files_map["figures"] = "assets/meta/figures.json"
    files_map["figures_dir"] = "assets/figures"
    sidecar_note = f"sidecar={sidecar.name}" if sidecar else "no sidecar; synthesized from PNGs"
    print(f"[build_package] figures <- {args.figures_dir}  ({len(copied_figs)} png, {sidecar_note})",
          file=sys.stderr)

    # ---- appendix figures: optional secondary pool (missing does not mark INCOMPLETE) ----
    appendix_png_dir = resolve_appendix_source(args.figures_dir)
    copied_appendix_figs = place_figure_dir(
        appendix_png_dir, appendix_figures_dst, sidecar, symlink=args.symlink)
    appendix_figures_json = build_figures_json(
        appendix_png_dir, sidecar, copied_appendix_figs, resolver,
        asset_subdir="appendix_figures")
    if appendix_png_dir is not None:
        (meta_dir / "appendix_figures.json").write_text(
            json.dumps(appendix_figures_json, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        files_map["appendix_figures"] = "assets/meta/appendix_figures.json"
        files_map["appendix_figures_dir"] = "assets/appendix_figures"
        print(f"[build_package] appendix_figures <- {appendix_png_dir} "
              f"({len(copied_appendix_figs)} png)", file=sys.stderr)
    if resolver is not None:
        resolver.close()

    # ---- segments2.json: full-document line-role segments (core asset) ----
    #      Must co-exist with structured_elements.json to avoid stale partial caches.
    segments_count = 0
    seg_src = resolve_segments_source(args.figures_dir, png_dir)
    if seg_src is not None:
        seg_dst = meta_dir / "segments2.json"
        place(seg_src, seg_dst, symlink=args.symlink)
        files_map["segments2"] = seg_dst.relative_to(out).as_posix()
        seg_doc = load_json(seg_dst)
        segments_count = len(seg_doc) if isinstance(seg_doc, list) else 0
        print(f"[build_package] segments2 <- {seg_src}  ({segments_count} segments)", file=sys.stderr)
    else:
        missing_core.append("segments2")
        print("[build_package] WARNING: segments2.json not found under --figures-dir",
              file=sys.stderr)

    # ---- source_lines.json: preserves auditable provenance for source_line_ids ----
    source_lines_count = 0
    source_lines_src = resolve_source_lines_source(args.figures_dir, png_dir)
    if source_lines_src is not None:
        source_lines_dst = meta_dir / "source_lines.json"
        place(source_lines_src, source_lines_dst, symlink=args.symlink)
        source_lines_doc = load_json(source_lines_dst)
        if isinstance(source_lines_doc, list):
            files_map["source_lines"] = source_lines_dst.relative_to(out).as_posix()
            source_lines_count = len(source_lines_doc)
            print(f"[build_package] source_lines <- {source_lines_src}  "
                  f"({source_lines_count} lines)", file=sys.stderr)
        else:
            missing_core.append("source_lines")
            print("[build_package] WARNING: source_lines.json is not a valid list", file=sys.stderr)
    else:
        missing_core.append("source_lines")
        print("[build_package] WARNING: source_lines.json not found under --figures-dir", file=sys.stderr)


    # ---- structured_elements.json: Tables/Equations/Algorithms from the semantic pass ----
    structured_counts = {"tables": 0, "equations": 0, "algorithms": 0}
    structured_src = resolve_structured_source(args.figures_dir, png_dir)
    if structured_src is not None:
        structured_dst = meta_dir / "structured_elements.json"
        place(structured_src, structured_dst, symlink=args.symlink)
        structured_doc = load_json(structured_dst)
        if isinstance(structured_doc, dict):
            files_map["structured_elements"] = structured_dst.relative_to(out).as_posix()
            for key in structured_counts:
                items = structured_doc.get(key)
                structured_counts[key] = len(items) if isinstance(items, list) else 0
            print(f"[build_package] structured_elements <- {structured_src}  "
                  f"(tables={structured_counts['tables']} equations={structured_counts['equations']} "
                  f"algorithms={structured_counts['algorithms']})", file=sys.stderr)
        else:
            missing_core.append("structured_elements")
            print("[build_package] WARNING: structured_elements.json is not a valid object", file=sys.stderr)
    else:
        missing_core.append("structured_elements")
        print("[build_package] WARNING: structured_elements.json not found under --figures-dir", file=sys.stderr)


    # Lossless family catalogue, independent of any poster or experiment name.
    family_catalog = {
        "schema": "dp-p2p.family-catalog.v1",
        "figures": figures_json + appendix_figures_json,
        "structured_elements": load_json(meta_dir / "structured_elements.json"),
    }
    (meta_dir / "family_catalog.json").write_text(
        json.dumps(family_catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files_map["family_catalog"] = "assets/meta/family_catalog.json"

    # ---- manifest.json ----
    metadata_doc = load_json(meta_dir / "metadata.json") if (meta_dir / "metadata.json").exists() else None
    source: dict[str, Any] = {"pdf": None, "filename": None, "bytes": None, "sha256": None}
    if args.pdf:
        if args.pdf.exists():
            source = {
                "pdf": str(args.pdf),
                "filename": args.pdf.name,
                "bytes": args.pdf.stat().st_size,
                "sha256": sha256_file(args.pdf),
            }
        else:
            print(f"[build_package] WARNING: --pdf not found ({args.pdf}); leaving sha256 empty in manifest",
                  file=sys.stderr)

    text_path = meta_dir / "text.txt"
    try:
        text_chars = len(text_path.read_text(encoding="utf-8")) if text_path.exists() else 0
    except Exception:
        text_chars = 0

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "layout": LAYOUT_VERSION,
        "created_at": utc_now(),
        "source": source,
        "files": files_map,
        "counts": {
            "figures": len(copied_figs),
            "appendix_figures": len(copied_appendix_figs),
            "captions": captions_count,
            "segments": segments_count,
            "tables": structured_counts["tables"],
            "equations": structured_counts["equations"],
            "algorithms": structured_counts["algorithms"],
            "logos": len(copied_logos),
            "qr": len(copied_qr),
            "text_chars": text_chars,
            "source_lines": source_lines_count,
            "metadata_populated_fields": (
                len([k for k, v in metadata_doc.items() if v]) if isinstance(metadata_doc, dict) else 0
            ),
        },
        "incomplete": sorted(missing_core),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    status = "OK" if not missing_core else f"INCOMPLETE(missing {','.join(missing_core)})"
    print(f"[build_package] {status} -> {out}  (figures={len(copied_figs)})")
    return 0 if not missing_core else 1


if __name__ == "__main__":
    raise SystemExit(main())
