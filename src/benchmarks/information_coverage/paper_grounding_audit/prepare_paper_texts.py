from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import os
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = BENCH_ROOT / "paper_grounding_audit"
REPO_ROOT = BENCH_ROOT.parents[3]
PDF_ROOT = Path(os.environ.get("VASP_PDF_ROOT", str(REPO_ROOT / "dataset")))
PAGE_ROOT = AUDIT_ROOT / "paper_pages"
TEXT_ROOT = AUDIT_ROOT / "paper_text"
MANIFEST_PATH = AUDIT_ROOT / "paper_manifest.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover() -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in PDF_ROOT.glob("*.pdf"):
        match = re.match(r"^(\d+)_", path.name)
        if not match:
            raise RuntimeError(f"PDF does not start with a numeric ID: {path.name}")
        poster_id = int(match.group(1))
        if poster_id in found:
            raise RuntimeError(f"Duplicate PDF ID {poster_id}: {path.name}")
        found[poster_id] = path
    expected = set(range(100))
    if set(found) != expected:
        raise RuntimeError(
            f"PDF IDs must be exactly 0-99; missing={sorted(expected-set(found))}, "
            f"extra={sorted(set(found)-expected)}"
        )
    return found


def clean_page_text(value: str) -> str:
    """Drop txtwrite vector/font dumps while preserving normal prose and equations."""
    cleaned = []
    omitted = 0
    for line in value.splitlines():
        if len(line) > 4000:
            omitted += 1
            continue
        cleaned.append(line.rstrip())
    if omitted:
        cleaned.append(
            f"[OMITTED {omitted} overlong txtwrite vector/font-encoding lines]"
        )
    return "\n".join(cleaned).strip()


def extract_one(poster_id: int, pdf_path: Path, force: bool) -> dict[str, object]:
    page_dir = PAGE_ROOT / str(poster_id)
    text_path = TEXT_ROOT / f"{poster_id}.txt"
    existing_pages = sorted(page_dir.glob("page-*.txt"))
    if force or not existing_pages or not text_path.is_file():
        if force and page_dir.exists():
            shutil.rmtree(page_dir)
        page_dir.mkdir(parents=True, exist_ok=True)
        TEXT_ROOT.mkdir(parents=True, exist_ok=True)
        command = [
            "gs",
            "-q",
            "-sDEVICE=txtwrite",
            "-o",
            str(page_dir / "page-%03d.txt"),
            str(pdf_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ghostscript failed for ID {poster_id}: {completed.stdout}"
            )
        existing_pages = sorted(page_dir.glob("page-*.txt"))
        if not existing_pages:
            raise RuntimeError(f"No text pages extracted for ID {poster_id}")
    sections = []
    for page_number, page_path in enumerate(existing_pages, start=1):
        raw_text = page_path.read_text(encoding="utf-8", errors="replace")
        page_text = clean_page_text(raw_text)
        sections.append(f"===== PDF PAGE {page_number} =====\n{page_text}\n")
    text_path.write_text("\n".join(sections), encoding="utf-8")

    text = text_path.read_text(encoding="utf-8", errors="replace")
    return {
        "poster_id": str(poster_id),
        "pdf": pdf_path.relative_to(REPO_ROOT).as_posix(),
        "pdf_filename": pdf_path.name,
        "pdf_sha256": sha256(pdf_path),
        "page_count": len(existing_pages),
        "text": text_path.relative_to(REPO_ROOT).as_posix(),
        "text_characters": len(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pdfs = discover()
    records = []
    for poster_id in range(100):
        record = extract_one(poster_id, pdfs[poster_id], args.force)
        records.append(record)
        print(
            f"[{poster_id}] pages={record['page_count']} "
            f"chars={record['text_characters']}"
        )
    write_json(
        MANIFEST_PATH,
        {
            "paper_count": 100,
            "mapping_policy": "numeric filename prefix 0-99",
            "text_extractor": "Ghostscript txtwrite, one file per PDF page",
            "papers": records,
        },
    )
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
