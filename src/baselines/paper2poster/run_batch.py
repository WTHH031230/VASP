"""Resumable parallel batch runner for the Codex-backed Paper2Poster baseline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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
        default=REPOSITORY_ROOT / "gpu_cache" / "paper2poster",
        help="Per-paper GPU preparse caches; disables PDF parsing during generation",
    )
    parser.add_argument("--codex-model", default="Qwen3.8-Flash-Next-FP8")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--jobs",
        type=int,
        default=3,
        help="Concurrent papers; each paper also uses --max-workers model calls",
    )
    parser.add_argument(
        "--global-codex-slots",
        type=int,
        default=6,
        help="Maximum codex exec processes shared by every paper worker",
    )
    parser.add_argument(
        "--global-codex-slot-dir",
        type=Path,
        help="Cross-process lock directory (default: OUTPUT/.codex_slots)",
    )
    parser.add_argument(
        "--parse-jobs",
        type=int,
        default=3,
        help="Concurrent GPU PDF parsers across all paper workers",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per Codex call")
    parser.add_argument(
        "--model-retries",
        type=int,
        default=3,
        help="Retries for each failed Codex transport/model turn",
    )
    parser.add_argument("--paper-timeout", type=int, default=14400)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument("--no-commenter", action="store_true")
    return parser


def _paper_dirs(dataset_root: Path, start_index: int, limit: int) -> list[Path]:
    papers = sorted(dataset_root.glob("*/paper.pdf"), key=lambda path: path.parent.name)
    if start_index < 1:
        raise SystemExit("--start-index must be at least 1")
    start = start_index - 1
    end = start + limit
    if end > len(papers):
        raise SystemExit(
            f"Requested paper range {start_index}..{end}, but only found "
            f"{len(papers)} papers under {dataset_root}"
        )
    return papers[start:end]


def _is_complete(output_dir: Path) -> bool:
    return (
        (output_dir / "adapter_run.json").is_file()
        and any(output_dir.glob("*.pptx"))
        and (any(output_dir.glob("*.png")) or any(output_dir.glob("*.jpg")))
    )


def _result_metrics(output_dir: Path) -> dict:
    manifest_path = output_dir / "adapter_run.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    metrics = {key: usage[key] for key in keys if key in usage}
    if "poster_generation_seconds" in manifest:
        metrics["generation_duration_seconds"] = manifest[
            "poster_generation_seconds"
        ]
    elif "duration_seconds" in manifest:
        metrics["generation_duration_seconds"] = manifest["duration_seconds"]
    for source, target in (
        ("gpu_preparse_seconds", "gpu_preparse_seconds"),
        ("total_pipeline_seconds", "total_pipeline_seconds"),
    ):
        if manifest.get(source) is not None:
            metrics[target] = manifest[source]
    return metrics


def _write_status(
    path: Path,
    records: list[dict],
    requested_total: int | None = None,
) -> None:
    measured = [
        record
        for record in records
        if record["status"] in {"success", "skipped"}
        and "estimated_api_cost_usd" in record
    ]
    records = sorted(records, key=lambda record: int(record["index"]))
    total = requested_total if requested_total is not None else len(records)
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "completed": len(records),
        "remaining": max(0, total - len(records)),
        "success": sum(record["status"] == "success" for record in records),
        "skipped": sum(record["status"] == "skipped" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "metrics": {
            "measured_papers": len(measured),
            "total_generation_seconds": round(
                sum(record.get("generation_duration_seconds", 0) for record in measured),
                2,
            ),
            "mean_generation_seconds": round(
                sum(record.get("generation_duration_seconds", 0) for record in measured)
                / len(measured),
                2,
            ) if measured else 0,
            "staged_papers": sum(
                "gpu_preparse_seconds" in record for record in measured
            ),
            "mean_gpu_preparse_seconds_staged": round(
                sum(
                    record.get("gpu_preparse_seconds", 0)
                    for record in measured
                    if "gpu_preparse_seconds" in record
                )
                / max(
                    1,
                    sum("gpu_preparse_seconds" in record for record in measured),
                ),
                2,
            ),
            "mean_poster_generation_seconds_staged": round(
                sum(
                    record.get("generation_duration_seconds", 0)
                    for record in measured
                    if "gpu_preparse_seconds" in record
                )
                / max(
                    1,
                    sum("gpu_preparse_seconds" in record for record in measured),
                ),
                2,
            ),
            "mean_total_pipeline_seconds_staged": round(
                sum(
                    record.get("total_pipeline_seconds", 0)
                    for record in measured
                    if "gpu_preparse_seconds" in record
                )
                / max(
                    1,
                    sum("gpu_preparse_seconds" in record for record in measured),
                ),
                2,
            ),
            "total_codex_calls": sum(record.get("call_count", 0) for record in measured),
            "total_input_tokens": sum(record.get("input_tokens", 0) for record in measured),
            "total_cached_input_tokens": sum(
                record.get("cached_input_tokens", 0) for record in measured
            ),
            "total_cache_write_input_tokens": sum(
                record.get("cache_write_input_tokens", 0) for record in measured
            ),
            "total_output_tokens": sum(record.get("output_tokens", 0) for record in measured),
            "total_estimated_api_cost_usd": round(
                sum(record["estimated_api_cost_usd"] for record in measured),
                6,
            ),
            "mean_estimated_api_cost_usd": round(
                sum(record["estimated_api_cost_usd"] for record in measured)
                / len(measured),
                6,
            ) if measured else 0,
            "cost_basis": "gpt-5.6-sol API-equivalent estimate",
            "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        },
        "records": records,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_paper(
    args: argparse.Namespace,
    index: int,
    batch_end: int,
    paper: Path,
    output_root: Path,
    log_root: Path,
) -> dict:
    name = paper.parent.name
    output_dir = output_root / name
    log_path = log_root / f"{index:03d}_{name}.log"
    command = [
        sys.executable,
        "-m",
        "src.baselines.paper2poster.run",
        "--paper",
        str(paper),
        "--output-dir",
        str(output_dir),
        "--reproduction-profile",
        "paper-v2",
        "--codex-model",
        args.codex_model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--max-workers",
        str(args.max_workers),
        "--timeout",
        str(args.timeout),
        "--model-retries",
        str(args.model_retries),
        "--force",
    ]
    if args.preparse_cache_root is not None:
        command.extend(
            (
                "--preparse-cache",
                str(args.preparse_cache_root / name),
            )
        )
    if args.no_traces:
        command.append("--no-traces")
    if args.no_commenter:
        command.append("--no-commenter")

    started = time.monotonic()
    success = False
    returncode = -1
    error = None
    for attempt in range(1, args.retries + 2):
        print(
            f"[batch {index:03d}/{batch_end}] start attempt {attempt}: {name}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== attempt {attempt} =====\n")
            log.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env={
                        **os.environ,
                        "P2P_PAPER2POSTER_PARSE_JOBS": str(args.parse_jobs),
                        "P2P_PAPER2POSTER_PARSE_SLOT_DIR": str(
                            output_root / ".parse_slots"
                        ),
                        "P2P_CODEX_GLOBAL_SLOTS": str(args.global_codex_slots),
                        "P2P_CODEX_GLOBAL_SLOT_DIR": str(
                            args.global_codex_slot_dir
                        ),
                    },
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
            success = True
            break
        print(
            f"[batch {index:03d}/{batch_end}] attempt {attempt} failed "
            f"(exit={returncode}); log={log_path}",
            flush=True,
        )

    duration = round(time.monotonic() - started, 2)
    status = "success" if success else "failed"
    record = {
        "index": index,
        "paper": name,
        "status": status,
        "duration_seconds": duration,
        "returncode": returncode,
        "log": str(log_path),
    }
    if success:
        record.update(_result_metrics(output_dir))
    if error:
        record["error"] = error
    print(
        f"[batch {index:03d}/{batch_end}] {status} in {duration:.1f}s: {name}",
        flush=True,
    )
    if success:
        print(
            f"[batch {index:03d}/{batch_end}] calls={record.get('call_count', 0)} "
            f"api-equivalent=${record.get('estimated_api_cost_usd', 0):.4f}",
            flush=True,
        )
    return record


def main() -> int:
    args = _parser().parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.parse_jobs < 1:
        raise SystemExit("--parse-jobs must be at least 1")
    if args.global_codex_slots < 1:
        raise SystemExit("--global-codex-slots must be at least 1")
    if args.model_retries < 0:
        raise SystemExit("--model-retries cannot be negative")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.preparse_cache_root is not None:
        args.preparse_cache_root = args.preparse_cache_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.global_codex_slot_dir = (
        args.global_codex_slot_dir.expanduser().resolve()
        if args.global_codex_slot_dir is not None
        else output_root / ".codex_slots"
    )
    args.global_codex_slot_dir.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "_logs"
    log_root.mkdir(exist_ok=True)
    status_path = (
        args.status_file.expanduser().resolve()
        if args.status_file
        else output_root / "batch_status.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    papers = _paper_dirs(dataset_root, args.start_index, args.limit)
    batch_end = args.start_index + len(papers) - 1
    records: list[dict] = []
    pending: list[tuple[int, Path]] = []

    print(
        f"[batch] papers={len(papers)} paper_jobs={args.jobs} "
        f"section_workers={args.max_workers} "
        f"global_codex_slots={args.global_codex_slots} model={args.codex_model} "
        f"effort={args.reasoning_effort} output={output_root} "
        f"preparse_cache={args.preparse_cache_root or 'disabled'}",
        flush=True,
    )
    for index, paper in enumerate(papers, start=args.start_index):
        name = paper.parent.name
        output_dir = output_root / name
        if _is_complete(output_dir):
            print(f"[batch {index:03d}/{batch_end}] skip complete: {name}", flush=True)
            records.append(
                {
                    "index": index,
                    "paper": name,
                    "status": "skipped",
                    **_result_metrics(output_dir),
                }
            )
        else:
            pending.append((index, paper))
    _write_status(status_path, records, requested_total=len(papers))

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                _run_paper,
                args,
                index,
                batch_end,
                paper,
                output_root,
                log_root,
            ): (index, paper)
            for index, paper in pending
        }
        for future in as_completed(futures):
            index, paper = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "index": index,
                    "paper": paper.parent.name,
                    "status": "failed",
                    "returncode": -1,
                    "error": f"batch worker error: {type(exc).__name__}: {exc}",
                }
                print(
                    f"[batch {index:03d}/{batch_end}] worker failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            records.append(record)
            _write_status(status_path, records, requested_total=len(papers))

    failed = [record for record in records if record["status"] == "failed"]
    print(
        f"[batch] finished: success/skipped={len(records) - len(failed)}, "
        f"failed={len(failed)}, status={status_path}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
