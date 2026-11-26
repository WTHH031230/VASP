"""Resumable parallel batch runner for the Codex-backed PosterGen baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().with_name("run.py")
DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument(
        "--preparse-cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "gpu_cache" / "postergen",
        help="Per-paper Marker GPU preparse caches; disables PDF parsing during generation",
    )
    parser.add_argument("--codex-model", default="Qwen3.8-Flash-Next-FP8")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--paper-timeout", type=int, default=10800)
    parser.add_argument("--parser-lock", type=Path, default=Path("/tmp/postergen-marker-gpu.lock"))
    parser.add_argument("--no-parser-lock", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--poster-width", type=float, default=54.0)
    parser.add_argument("--poster-height", type=float, default=36.0)
    parser.add_argument("--logo", type=Path)
    parser.add_argument("--aff-logo", type=Path)
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-global-ledger", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    return parser


def _papers(dataset_root: Path, start_index: int, limit: int) -> list[Path]:
    found = sorted(dataset_root.glob("*/paper.pdf"), key=lambda path: path.parent.name)
    if start_index < 1 or limit < 1:
        raise SystemExit("--start-index and --limit must be positive")
    selected = found[start_index - 1 : start_index - 1 + limit]
    if len(selected) != limit:
        raise SystemExit(
            f"Requested {limit} papers from index {start_index}, found only {len(selected)} "
            f"under {dataset_root}"
        )
    return selected


def _is_complete(output_dir: Path) -> bool:
    return (
        (output_dir / "adapter_run.json").is_file()
        and any(output_dir.glob("*.pptx"))
        and any(output_dir.glob("*.png"))
    )


def _metrics(output_dir: Path) -> dict:
    try:
        manifest = json.loads((output_dir / "adapter_run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    usage = manifest.get("codex_usage") or {}
    keys = (
        "call_count",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "estimated_api_cost_usd",
    )
    result = {key: usage[key] for key in keys if key in usage}
    if "duration_seconds" in manifest:
        result["generation_duration_seconds"] = manifest["duration_seconds"]
    return result


def _write_status(path: Path, records: dict[int, dict]) -> None:
    ordered = [records[index] for index in sorted(records)]
    measured = [
        record
        for record in ordered
        if record["status"] in {"success", "skipped"}
        and "estimated_api_cost_usd" in record
    ]
    total_seconds = sum(record.get("generation_duration_seconds", 0) for record in measured)
    total_cost = sum(record.get("estimated_api_cost_usd", 0) for record in measured)
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(ordered),
        "success": sum(record["status"] == "success" for record in ordered),
        "skipped": sum(record["status"] == "skipped" for record in ordered),
        "failed": sum(record["status"] == "failed" for record in ordered),
        "running": sum(record["status"] == "running" for record in ordered),
        "pending": sum(record["status"] == "pending" for record in ordered),
        "metrics": {
            "measured_papers": len(measured),
            "total_generation_seconds": round(total_seconds, 2),
            "mean_generation_seconds": round(total_seconds / len(measured), 2) if measured else 0,
            "total_codex_calls": sum(record.get("call_count", 0) for record in measured),
            "total_input_tokens": sum(record.get("input_tokens", 0) for record in measured),
            "total_cached_input_tokens": sum(record.get("cached_input_tokens", 0) for record in measured),
            "total_cache_write_input_tokens": sum(record.get("cache_write_input_tokens", 0) for record in measured),
            "total_output_tokens": sum(record.get("output_tokens", 0) for record in measured),
            "total_estimated_api_cost_usd": round(total_cost, 6),
            "mean_estimated_api_cost_usd": round(total_cost / len(measured), 6) if measured else 0,
            "cost_basis": "gpt-5.6-sol standard API-equivalent estimate",
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        },
        "records": ordered,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _command(args: argparse.Namespace, paper: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--paper",
        str(paper),
        "--output-dir",
        str(output_dir),
        "--codex-model",
        args.codex_model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--timeout",
        str(args.timeout),
        "--poster-width",
        str(args.poster_width),
        "--poster-height",
        str(args.poster_height),
        "--usage-ledger",
        str(args.usage_ledger.expanduser()),
        "--parser-lock",
        str(args.parser_lock.expanduser()),
    ]
    if args.preparse_cache_root is not None:
        command.extend(
            (
                "--preparse-cache",
                str(args.preparse_cache_root.expanduser().resolve() / paper.parent.name),
                "--no-parser-lock",
            )
        )
    if args.logo:
        command.extend(("--logo", str(args.logo.expanduser().resolve())))
    if args.aff_logo:
        command.extend(("--aff-logo", str(args.aff_logo.expanduser().resolve())))
    if args.no_global_ledger:
        command.append("--no-global-ledger")
    if args.no_parser_lock:
        command.append("--no-parser-lock")
    if args.no_traces:
        command.append("--no-traces")
    return command


def main() -> int:
    args = _parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_logs"
    log_root.mkdir(exist_ok=True)
    status_path = args.status_file.expanduser().resolve() if args.status_file else output_root / "batch_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    papers = _papers(dataset_root, args.start_index, args.limit)
    records: dict[int, dict] = {}
    lock = threading.Lock()

    for index, paper in enumerate(papers, start=args.start_index):
        output_dir = output_root / paper.parent.name
        status = "skipped" if _is_complete(output_dir) else "pending"
        records[index] = {
            "index": index,
            "paper": paper.parent.name,
            "status": status,
            **(_metrics(output_dir) if status == "skipped" else {}),
        }
    _write_status(status_path, records)

    def update(index: int, record: dict) -> None:
        with lock:
            records[index] = record
            _write_status(status_path, records)

    def run_one(index: int, paper: Path) -> dict:
        name = paper.parent.name
        output_dir = output_root / name
        log_path = log_root / f"{index:03d}_{name}.log"
        update(index, {"index": index, "paper": name, "status": "running", "log": str(log_path)})
        started = time.monotonic()
        returncode = -1
        error = None
        for attempt in range(1, args.retries + 2):
            print(f"[batch {index:03d}] attempt {attempt}: {name}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n===== attempt {attempt} =====\n")
                log.flush()
                try:
                    completed = subprocess.run(
                        _command(args, paper, output_dir),
                        cwd=REPOSITORY_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=args.paper_timeout,
                        check=False,
                    )
                    returncode = completed.returncode
                    error = None
                except subprocess.TimeoutExpired:
                    returncode = 124
                    error = f"paper timeout after {args.paper_timeout}s"
            if returncode == 0 and _is_complete(output_dir):
                break
            print(f"[batch {index:03d}] attempt {attempt} failed: exit={returncode}", flush=True)

        success = returncode == 0 and _is_complete(output_dir)
        record = {
            "index": index,
            "paper": name,
            "status": "success" if success else "failed",
            "duration_seconds": round(time.monotonic() - started, 2),
            "returncode": returncode,
            "log": str(log_path),
        }
        if success:
            record.update(_metrics(output_dir))
        if error:
            record["error"] = error
        update(index, record)
        print(
            f"[batch {index:03d}] {record['status']} in {record['duration_seconds']:.1f}s "
            f"calls={record.get('call_count', 0)} cost=${record.get('estimated_api_cost_usd', 0):.4f}",
            flush=True,
        )
        return record

    pending = [
        (index, paper)
        for index, paper in enumerate(papers, start=args.start_index)
        if records[index]["status"] == "pending"
    ]
    print(
        f"[batch] papers={len(papers)} pending={len(pending)} jobs={args.jobs} "
        f"model={args.codex_model} effort={args.reasoning_effort}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, index, paper): index for index, paper in pending}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                index = futures[future]
                current = records[index]
                update(index, {**current, "status": "failed", "error": repr(exc)})

    failed = [record for record in records.values() if record["status"] == "failed"]
    print(
        f"[batch] finished: complete={len(records) - len(failed)}, failed={len(failed)}, "
        f"status={status_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
