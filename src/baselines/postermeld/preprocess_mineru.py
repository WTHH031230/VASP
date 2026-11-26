"""Populate the local PosterMELD MinerU bridge cache for a paper dataset."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("audit_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_one(paper: Path, output: Path, base_url: str, poll_seconds: float) -> dict[str, object]:
    status_path = output / "preprocess.json"
    if status_path.is_file():
        try:
            cached = json.loads(status_path.read_text(encoding="utf-8"))
            if cached.get("status") == "completed" and (output / "mineru_result.zip").is_file():
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    started = time.monotonic()
    payload = {
        "enable_formula": True,
        "enable_table": True,
        "language": "en",
        "model_version": "vlm",
        "files": [{"name": paper.name, "is_ocr": False, "data_id": paper.stem[:64]}],
    }
    response = requests.post(f"{base_url}/api/v4/file-urls/batch", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()["data"]
    batch_id = str(data["batch_id"])
    upload_url = str(data["file_urls"][0]["url"])
    with paper.open("rb") as handle:
        upload = requests.put(upload_url, data=handle, timeout=120)
    upload.raise_for_status()
    while True:
        result_response = requests.get(
            f"{base_url}/api/v4/extract-results/batch/{batch_id}", timeout=30
        )
        result_response.raise_for_status()
        result = result_response.json()["data"]["extract_result"][0]
        state = str(result.get("state") or "").lower()
        if state == "done":
            break
        if state == "failed":
            raise RuntimeError(str(result.get("err_msg") or "MinerU failed"))
        time.sleep(poll_seconds)
    archive = requests.get(str(result["full_zip_url"]), timeout=600)
    archive.raise_for_status()
    output.mkdir(parents=True, exist_ok=True)
    zip_path = output / "mineru_result.zip"
    temporary = output / ".mineru_result.zip.tmp"
    temporary.write_bytes(archive.content)
    temporary.replace(zip_path)
    record = {
        "status": "completed",
        "paper": str(paper),
        "batch_id": batch_id,
        "duration_seconds": round(time.monotonic() - started, 3),
        "zip_bytes": zip_path.stat().st_size,
    }
    _atomic_json(status_path, record)
    return record


def main() -> int:
    args = _parser().parse_args()
    papers = sorted(args.dataset_root.expanduser().resolve().glob("*/paper.pdf"))[: args.limit]
    if len(papers) != args.limit:
        raise SystemExit(f"Expected {args.limit} papers, found {len(papers)}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-count must be positive and --shard-index must be in range")
    selected = [
        (index, paper)
        for index, paper in enumerate(papers, 1)
        if (index - 1) % args.shard_count == args.shard_index
    ]
    audit_root = args.audit_root.expanduser().resolve()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _run_one,
                paper,
                audit_root / paper.parent.name,
                args.base_url.rstrip("/"),
                args.poll_seconds,
            ): (index, paper)
            for index, paper in selected
        }
        for future in as_completed(futures):
            index, paper = futures[future]
            try:
                record = future.result()
                print(
                    f"[{index:03d}/{len(papers)}] completed "
                    f"{record.get('duration_seconds', 0)}s {paper.parent.name}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                _atomic_json(
                    audit_root / paper.parent.name / "preprocess.json",
                    {"status": "failed", "paper": str(paper), "error": repr(exc)},
                )
                print(f"[{index:03d}/{len(papers)}] failed {paper.parent.name}: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
