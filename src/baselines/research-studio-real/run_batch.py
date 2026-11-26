"""Parallel runner for the research-studio-real Codex baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().with_name("run.py")
DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"
REQUIRED_DELIVERABLES = ("poster.html", "poster.pdf", "poster.png", "poster.pptx")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPOSITORY_ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--codex-model", default="Qwen3.8-Flash-Next-FP8")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout", type=int, default=10_800)
    parser.add_argument("--paper-timeout", type=int, default=11_400)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--status-file", type=Path)
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
            f"Requested {limit} papers from index {start_index}, found only "
            f"{len(selected)} under {dataset_root}"
        )
    return selected


def _manifest(output: Path) -> dict[str, Any]:
    try:
        return json.loads((output / "adapter_run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_complete(output: Path) -> bool:
    manifest = _manifest(output)
    return manifest.get("status") == "success" and all(
        (output / name).is_file() for name in REQUIRED_DELIVERABLES
    )


def _metrics(output: Path) -> dict[str, Any]:
    manifest = _manifest(output)
    usage = manifest.get("codex_usage") or {}
    totals = manifest.get("attempt_totals") or {}
    result: dict[str, Any] = {
        key: usage[key]
        for key in (
            "call_count",
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "estimated_api_cost_usd",
        )
        if key in usage
    }
    if "duration_seconds" in manifest:
        result["generation_duration_seconds"] = manifest["duration_seconds"]
    if totals:
        result["attempt_count"] = totals.get("attempt_count", 0)
        result["spent_api_cost_usd"] = totals.get("estimated_api_cost_usd", 0)
        result["spent_input_tokens"] = totals.get("input_tokens", 0)
        result["spent_output_tokens"] = totals.get("output_tokens", 0)
    return result


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _write_status(
    path: Path,
    records: dict[int, dict[str, Any]],
    batch_started: datetime,
) -> None:
    ordered = [records[index] for index in sorted(records)]
    measured = [
        record
        for record in ordered
        if record["status"] in {"success", "skipped"}
        and "estimated_api_cost_usd" in record
    ]
    durations = [float(record.get("generation_duration_seconds", 0)) for record in measured]
    delivered_costs = [float(record.get("estimated_api_cost_usd", 0)) for record in measured]
    all_spend = sum(float(record.get("spent_api_cost_usd", 0)) for record in ordered)
    now = datetime.now(timezone.utc)
    summary = {
        "schema_version": 1,
        "batch_started_at": batch_started.isoformat(),
        "updated_at": now.isoformat(),
        "batch_elapsed_seconds": round((now - batch_started).total_seconds(), 2),
        "total": len(ordered),
        "success": sum(record["status"] == "success" for record in ordered),
        "skipped": sum(record["status"] == "skipped" for record in ordered),
        "failed": sum(record["status"] == "failed" for record in ordered),
        "running": sum(record["status"] == "running" for record in ordered),
        "pending": sum(record["status"] == "pending" for record in ordered),
        "metrics": {
            "measured_posters": len(measured),
            "total_generation_seconds": round(sum(durations), 2),
            "mean_generation_seconds": round(statistics.fmean(durations), 2) if durations else 0,
            "median_generation_seconds": round(statistics.median(durations), 2) if durations else 0,
            "p95_generation_seconds": round(_percentile(durations, 0.95), 2),
            "total_codex_calls_for_delivered_posters": sum(
                int(record.get("call_count", 0)) for record in measured
            ),
            "total_input_tokens_for_delivered_posters": sum(
                int(record.get("input_tokens", 0)) for record in measured
            ),
            "total_output_tokens_for_delivered_posters": sum(
                int(record.get("output_tokens", 0)) for record in measured
            ),
            "total_delivered_api_equivalent_cost_usd": round(sum(delivered_costs), 6),
            "mean_delivered_api_equivalent_cost_usd": round(
                statistics.fmean(delivered_costs), 6
            ) if delivered_costs else 0,
            "total_spend_including_retries_api_equivalent_usd": round(all_spend, 6),
            "mean_spend_per_requested_paper_api_equivalent_usd": round(
                all_spend / len(ordered), 6
            ) if ordered else 0,
            "cost_basis": "gpt-5.6-sol API-equivalent estimate; Codex subscription usage may not be billed as API usage",
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        },
        "records": ordered,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _command(args: argparse.Namespace, paper: Path, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--paper",
        str(paper),
        "--output-dir",
        str(output),
        "--codex-model",
        args.codex_model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--timeout",
        str(args.timeout),
        "--usage-ledger",
        str(args.usage_ledger.expanduser()),
        "--resume",
    ]
    if args.no_global_ledger:
        command.append("--no-global-ledger")
    if args.no_traces:
        command.append("--no-traces")
    return command


def main() -> int:
    args = _parser().parse_args()
    if args.jobs < 1 or args.retries < 0:
        raise SystemExit("--jobs must be positive and --retries cannot be negative")
    dataset = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_logs"
    log_root.mkdir(exist_ok=True)
    status_path = (
        args.status_file.expanduser().resolve()
        if args.status_file
        else output_root / "batch_status.json"
    )
    papers = _papers(dataset, args.start_index, args.limit)
    batch_started = datetime.now(timezone.utc)
    records: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()
    for index, paper in enumerate(papers, start=args.start_index):
        output = output_root / paper.parent.name
        status = "skipped" if _is_complete(output) else "pending"
        records[index] = {
            "index": index,
            "paper": paper.parent.name,
            "status": status,
            **(_metrics(output) if status == "skipped" else {}),
        }
    _write_status(status_path, records, batch_started)

    def update(index: int, record: dict[str, Any]) -> None:
        with lock:
            records[index] = record
            _write_status(status_path, records, batch_started)

    def run_one(index: int, paper: Path) -> dict[str, Any]:
        name = paper.parent.name
        output = output_root / name
        log_path = log_root / f"{index:03d}_{name}.log"
        update(index, {"index": index, "paper": name, "status": "running", "log": str(log_path)})
        started = time.monotonic()
        returncode = -1
        error = None
        for attempt in range(1, args.retries + 2):
            print(
                f"[research-studio-real {index:03d}] attempt {attempt}: {name}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n===== batch attempt {attempt} =====\n")
                log.flush()
                try:
                    completed = subprocess.run(
                        _command(args, paper, output),
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
            if returncode == 0 and _is_complete(output):
                break
            print(
                f"[research-studio-real {index:03d}] attempt {attempt} "
                f"failed: exit={returncode}",
                flush=True,
            )
        success = returncode == 0 and _is_complete(output)
        record: dict[str, Any] = {
            "index": index,
            "paper": name,
            "status": "success" if success else "failed",
            "batch_attempt_duration_seconds": round(time.monotonic() - started, 2),
            "returncode": returncode,
            "log": str(log_path),
            **_metrics(output),
        }
        if error:
            record["error"] = error
        update(index, record)
        print(
            f"[research-studio-real {index:03d}] {record['status']} "
            f"generation={record.get('generation_duration_seconds', 0):.1f}s "
            f"cost=${record.get('estimated_api_cost_usd', 0):.4f} "
            f"spent=${record.get('spent_api_cost_usd', 0):.4f}",
            flush=True,
        )
        return record

    pending = [
        (index, paper)
        for index, paper in enumerate(papers, start=args.start_index)
        if records[index]["status"] == "pending"
    ]
    print(
        f"[research-studio-real] papers={len(papers)} "
        f"pending={len(pending)} jobs={args.jobs} "
        f"model={args.codex_model} effort={args.reasoning_effort} output={output_root}",
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
        f"[research-studio-real] finished "
        f"complete={len(records) - len(failed)} failed={len(failed)} "
        f"status={status_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
