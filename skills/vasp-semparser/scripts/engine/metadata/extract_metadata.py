#!/usr/bin/env python3
"""Deterministically normalize metadata emitted by the semantic LLM call.

Inputs:
  <paper.pdf>              PDF used only for the first-page arXiv stamp fallback.
  <metadata_candidate.json> Raw metadata object emitted by ``segment_roles.py``.
  <outdir>                 Destination for normalized ``metadata.json``.

This stage never invokes an LLM. Missing values remain empty rather than being
guessed; the only synthesized value is the canonical arXiv URL when an id is known.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import fitz


_ARXIV_STAMP_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")


def scan_arxiv_id(first_page_text):
    """Return this paper's arXiv id from the first-page stamp, or ``""``."""
    if not first_page_text:
        return ""
    match = _ARXIV_STAMP_RE.search(first_page_text)
    return match.group(1) if match else ""


def _author_index_map(raw_map, authors, institutes):
    """Clean author-to-institute 1-indexed mappings without inventing links."""
    n_institutes = len(institutes)
    cleaned = {}
    if isinstance(raw_map, dict):
        for name, indexes in raw_map.items():
            if not (isinstance(name, str) and name.strip()):
                continue
            if not isinstance(indexes, list):
                continue
            seen = set()
            valid = []
            for index in indexes:
                if isinstance(index, bool):
                    continue
                if isinstance(index, str) and index.strip().isdigit():
                    index = int(index.strip())
                if (
                    isinstance(index, int)
                    and 1 <= index <= n_institutes
                    and index not in seen
                ):
                    seen.add(index)
                    valid.append(index)
            if valid:
                cleaned[name.strip()] = valid

    if cleaned:
        return cleaned
    if n_institutes == 1:
        return {author: [1] for author in authors}
    if n_institutes == 0:
        return {}
    print(
        "[extract_metadata] WARNING: multi-institute paper but no valid "
        "author_index_map — leaving it empty",
        file=sys.stderr,
    )
    return {}


def normalize(data, arxiv_id_hint=""):
    """Normalize the single-call metadata candidate into the public schema."""
    data = data if isinstance(data, dict) else {}

    def clean_string(key):
        value = data.get(key, "")
        return value.strip() if isinstance(value, str) else ""

    def clean_list(key):
        value = data.get(key, [])
        if not isinstance(value, list):
            return []
        return [
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]

    venue = clean_string("venue")
    if re.fullmatch(r"(?i)\s*(arxiv(\s+preprint)?|preprint)\s*", venue or ""):
        venue = ""

    arxiv_id = clean_string("arxiv_id") or arxiv_id_hint.strip()
    paper_url = clean_string("paper_url")
    if not paper_url and arxiv_id:
        paper_url = f"https://arxiv.org/abs/{arxiv_id}"

    authors = clean_list("authors")
    institutes = clean_list("institutes")
    return {
        "title": clean_string("title"),
        "authors": authors,
        "author_index_map": _author_index_map(
            data.get("author_index_map"), authors, institutes
        ),
        "institutes": institutes,
        "venue": venue,
        "paper_url": paper_url,
        "code_url": clean_string("code_url"),
        "arxiv_id": arxiv_id,
        "project_url": clean_string("project_url"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("metadata_candidate")
    parser.add_argument("outdir")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    candidate_path = Path(args.metadata_candidate)
    outdir = Path(args.outdir)
    if not pdf_path.exists():
        raise SystemExit(f"[extract_metadata] pdf not found: {pdf_path}")
    if not candidate_path.exists():
        raise SystemExit(
            f"[extract_metadata] metadata candidate not found: {candidate_path}"
        )

    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[extract_metadata] invalid metadata candidate: {exc}"
        ) from exc

    document = fitz.open(str(pdf_path))
    if len(document) == 0:
        raise SystemExit(f"[extract_metadata] pdf has 0 pages: {pdf_path}")
    first_page_text = document[0].get_text("text")
    document.close()

    metadata = normalize(
        candidate,
        arxiv_id_hint=scan_arxiv_id(first_page_text),
    )
    if not metadata["title"]:
        raise SystemExit("[extract_metadata] metadata candidate has no title")

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "metadata.json"
    out_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[extract_metadata] \"{metadata['title'][:60]}\" | "
        f"authors={len(metadata['authors'])} "
        f"institutes={len(metadata['institutes'])} "
        f"authmap={len(metadata['author_index_map'])}/{len(metadata['authors'])} "
        f"venue={metadata['venue'] or '(empty)'} "
        f"arxiv_id={metadata['arxiv_id'] or '(none)'} -> {out_path}"
    )


if __name__ == "__main__":
    main()
