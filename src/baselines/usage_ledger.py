"""Backfill project-preserved Codex Exec traces into the global usage ledger."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"
PRICING = {
    "input": 5.0,
    "cached": 0.5,
    "cache_write": 6.25,
    "output": 30.0,
    "threshold": 272_000,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--application", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _events(path: Path) -> tuple[str | None, dict[str, int]]:
    thread_id = None
    usage: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    return thread_id, usage


def _cost_record(call_id: int, usage: dict[str, int], duration: float, images: int) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", 0))
    cached = min(input_tokens, int(usage.get("cached_input_tokens", 0)))
    non_cached = input_tokens - cached
    cache_write = min(non_cached, int(usage.get("cache_write_input_tokens", 0)))
    uncached = non_cached - cache_write
    output = int(usage.get("output_tokens", 0))
    long_context = input_tokens > PRICING["threshold"]
    input_multiplier = 2.0 if long_context else 1.0
    output_multiplier = 1.5 if long_context else 1.0
    cost = (
        (uncached * PRICING["input"] + cached * PRICING["cached"] + cache_write * PRICING["cache_write"])
        * input_multiplier
        + output * PRICING["output"] * output_multiplier
    ) / 1_000_000
    return {
        "call_id": call_id,
        "duration_seconds": round(duration, 3),
        "image_count": images,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "uncached_input_tokens": uncached,
        "output_tokens": output,
        "long_context_pricing": long_context,
        "estimated_api_cost_usd": round(cost, 8),
    }


def _known_threads(ledger: Path) -> set[str]:
    known: set[str] = set()
    if not ledger.is_file():
        return known
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            thread_id = json.loads(line).get("thread_id")
        except json.JSONDecodeError:
            continue
        if thread_id:
            known.add(thread_id)
    return known


def main() -> int:
    args = _parser().parse_args()
    root = args.results_root.expanduser().resolve()
    ledger = args.ledger.expanduser()
    known = _known_threads(ledger)
    entries: list[dict[str, Any]] = []
    skipped = 0

    for manifest_path in sorted(root.glob("*/adapter_run.json")):
        result_dir = manifest_path.parent
        trace_dir = result_dir / "codex_traces"
        if not trace_dir.is_dir():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        usage_calls = {
            int(record["call_id"]): record
            for record in (manifest.get("codex_usage") or {}).get("calls", [])
            if "call_id" in record
        }
        for event_path in sorted(trace_dir.glob("call_*.events.jsonl")):
            call_text = event_path.name.removeprefix("call_").split(".", 1)[0]
            if not call_text.isdigit():
                continue
            thread_id, usage = _events(event_path)
            if not thread_id or not usage:
                continue
            if thread_id in known:
                skipped += 1
                continue
            call_id = int(call_text)
            prior = usage_calls.get(call_id, {})
            image_count = sum(1 for _ in trace_dir.glob(f"call_{call_id:04d}_image_*"))
            record = _cost_record(call_id, usage, float(prior.get("duration_seconds", 0)), image_count)
            entries.append(
                {
                    "schema_version": 1,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "source": "trace_backfill",
                    "source_completed_at": manifest.get("completed_at"),
                    "backend": "codex exec",
                    "ephemeral": True,
                    "application": args.application,
                    "model": manifest.get("codex_model", "unknown"),
                    "reasoning_effort": manifest.get("reasoning_effort", "unknown"),
                    "thread_id": thread_id,
                    "cwd": None,
                    "trace_dir": str(trace_dir),
                    "result_dir": str(result_dir),
                    **record,
                }
            )
            known.add(thread_id)

    print(f"backfill candidates={len(entries)} already_present={skipped} ledger={ledger}")
    if args.dry_run or not entries:
        return 0
    ledger.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with ledger.open("a", encoding="utf-8") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        for entry in entries:
            output.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
        fcntl.flock(output.fileno(), fcntl.LOCK_UN)
    print(f"appended={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
