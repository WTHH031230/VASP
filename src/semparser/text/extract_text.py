#!/usr/bin/env python3
"""Full-text extraction: extract PDF text with PyMuPDF, delimited by \f (form feed).

Follows the native pdftotext pagination convention: every page (including the
last page) ends with a trailing \f, so an N-page PDF contains N form-feed
characters and downstream `text.split("\f")` logic behaves identically.

If `--ocr-cache` is provided (pointing to `content/raw.md`, its parent
`content/` directory, or the paper cache root), the Marker OCR markdown is also
copied verbatim to `text_marker.md` for reference.

Usage:
    python3 extract_text.py <paper.pdf> <outdir> [--ocr-cache <path-to-raw.md-or-cache-dir>]

Outputs:
    <outdir>/text.txt          - PyMuPDF full text with per-page \f delimiters (always produced)
    <outdir>/text_marker.md    - Marker markdown full text (only if --ocr-cache resolves raw.md)
"""
import argparse
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_fitz(pdf_path: Path) -> str:
    """Extract per-page text with trailing \\f (matching pdftotext pagination)."""
    doc = fitz.open(pdf_path)
    try:
        parts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "".join(p + "\f" for p in parts)


def find_marker_raw_md(ocr_cache: Path) -> "Path | None":
    """Resolve --ocr-cache from raw.md directly, its content/ dir, or the paper cache root.

    Supported layouts:
      <cache>/<slug>/content/raw.md   (direct file path)
      <cache>/<slug>/content          (content directory)
      <cache>/<slug>                  (paper cache root)
    """
    if ocr_cache.is_file():
        return ocr_cache
    for candidate in (ocr_cache / "raw.md", ocr_cache / "content" / "raw.md"):
        if candidate.is_file():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract full text from a paper PDF (PyMuPDF + optional Marker cache)")
    ap.add_argument("pdf", help="Path to paper PDF")
    ap.add_argument("outdir", help="Output directory")
    ap.add_argument(
        "--ocr-cache", default=None, metavar="PATH",
        help="Optional Marker cache path: raw.md directly, its content/ directory, "
             "or the paper cache root directory.",
    )
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = extract_text_fitz(pdf_path)
    text_path = outdir / "text.txt"
    text_path.write_text(text, encoding="utf-8")
    n_pages = text.count("\f")
    print(f"[extract_text] fitz: {len(text)} chars, {n_pages} pages -> {text_path}")

    if args.ocr_cache:
        marker_src = find_marker_raw_md(Path(args.ocr_cache))
        if marker_src is None:
            print(f"[extract_text] WARNING: raw.md not found under --ocr-cache={args.ocr_cache}; skipping marker text")
        else:
            marker_text = marker_src.read_text(encoding="utf-8")
            marker_path = outdir / "text_marker.md"
            marker_path.write_text(marker_text, encoding="utf-8")
            print(f"[extract_text] marker: {len(marker_text)} chars (from {marker_src}) -> {marker_path}")
    else:
        print("[extract_text] --ocr-cache not provided; skipping marker text")


if __name__ == "__main__":
    main()
