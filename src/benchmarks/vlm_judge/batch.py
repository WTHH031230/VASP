"""Parallel, resumable batch runner for poster VLM judging."""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .judge import JudgeConfig, JudgeError, backend_display, judge_poster
from .reports import write_benchmark_reports
from .rubrics import ALL_ASPECTS


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def discover_posters(
    input_root: Path,
    poster_name: str,
    expected_paper_root: Path | None = None,
) -> tuple[list[tuple[str, Path]], list[dict[str, str]]]:
    """Discover exact poster filenames without silently selecting figure PNGs."""

    pairs: list[tuple[str, Path]] = []
    missing: list[dict[str, str]] = []
    if expected_paper_root is not None:
        expected_names = sorted(
            path.name
            for path in expected_paper_root.iterdir()
            if path.is_dir()
            and not path.name.startswith((".", "_"))
            and (path / "paper.pdf").is_file()
        )
        subdirs = [input_root / name for name in expected_names]
    else:
        subdirs = sorted(
            path
            for path in input_root.iterdir()
            if path.is_dir() and not path.name.startswith((".", "_"))
        )
    if subdirs:
        for paper_dir in subdirs:
            candidates = [paper_dir / poster_name]
            title_named = paper_dir / f"{paper_dir.name}.png"
            if title_named not in candidates:
                candidates.append(title_named)
            image = next((path for path in candidates if path.is_file()), None)
            if image is None:
                root_pngs = sorted(paper_dir.glob("*.png"))
                if len(root_pngs) == 1:
                    image = root_pngs[0]
            if image is None:
                missing.append(
                    {
                        "paper_id": paper_dir.name,
                        "error": (
                            f"no unambiguous final poster PNG; tried {poster_name!r}, "
                            f"{paper_dir.name + '.png'!r}, and a single root-level PNG"
                        ),
                    }
                )
            else:
                pairs.append((paper_dir.name, image.resolve()))
    else:
        for image in sorted(input_root.glob("*.png")):
            pairs.append((image.stem, image.resolve()))
    return pairs, missing


def _load_success(path: Path, config: JudgeConfig, image: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "backend": backend_display(config),
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "call_mode": config.call_mode,
        "scope": config.scope,
        "tool_policy": (
            "disabled-fail-closed"
            if config.enforce_no_tools
            else "agent-tools-allowed"
        ),
    }
    actual = value.get("configuration") or {}
    if value.get("status") != "success":
        return None
    if any(actual.get(key) != expected_value for key, expected_value in expected.items()):
        return None
    if [str(image.resolve())] != value.get("images"):
        return None
    return value


def _usage_from(value: dict[str, Any]) -> dict[str, Any]:
    """Read usage from the backend-neutral field or a legacy backend field."""

    return (
        value.get("judge_usage")
        or value.get("claude_usage")
        or value.get("codex_usage")
        or {}
    )


def _record_from_result(
    paper_id: str,
    result: dict[str, Any],
    *,
    skipped: bool,
) -> dict[str, Any]:
    usage = _usage_from(result)
    return {
        "paper_id": paper_id,
        "status": "success",
        "skipped": skipped,
        "error": None,
        "generation_seconds": result.get("generation_seconds"),
        "estimated_api_cost_usd": usage.get("estimated_api_cost_usd", 0),
        "call_count": usage.get("call_count", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "aesthetic_average": result.get("aesthetic_average"),
        "information_average": result.get("information_average"),
        "overall_average": result.get("overall_average"),
        "scores": {
            aspect: (result.get("results") or {}).get(aspect, {}).get("score")
            for aspect in ALL_ASPECTS
        },
    }


def _record_from_failure(paper_id: str, failure: dict[str, Any], error: str) -> dict[str, Any]:
    usage = _usage_from(failure)
    return {
        "paper_id": paper_id,
        "status": "failed",
        "skipped": False,
        "error": error,
        "generation_seconds": failure.get("generation_seconds"),
        "estimated_api_cost_usd": usage.get("estimated_api_cost_usd", 0),
        "call_count": usage.get("call_count", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_input_tokens": usage.get("cached_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "aesthetic_average": None,
        "information_average": None,
        "overall_average": None,
        "scores": {},
    }


def _summary(records: list[dict[str, Any]], config: JudgeConfig) -> dict[str, Any]:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] == "failed"]

    def mean(key: str) -> float | None:
        values = [
            float(record[key])
            for record in successes
            if record.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    aspect_means = {}
    for aspect in ALL_ASPECTS:
        values = [
            float(record["scores"][aspect])
            for record in successes
            if record.get("scores", {}).get(aspect) is not None
        ]
        aspect_means[aspect] = sum(values) / len(values) if values else None

    return {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "configuration": {
            "backend": backend_display(config),
            "tool_policy": (
                "disabled-fail-closed"
                if config.enforce_no_tools
                else "agent-tools-allowed"
            ),
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "call_mode": config.call_mode,
            "scope": config.scope,
            "aspect_jobs": config.aspect_jobs,
        },
        "total": len(records),
        "success": len(successes),
        "failed": len(failures),
        "skipped": sum(bool(record.get("skipped")) for record in successes),
        "total_generation_seconds": round(
            sum(float(record.get("generation_seconds") or 0) for record in successes),
            3,
        ),
        "mean_generation_seconds": mean("generation_seconds"),
        "total_api_cost_usd": round(
            sum(float(record.get("estimated_api_cost_usd") or 0) for record in records),
            8,
        ),
        "mean_api_cost_usd": (
            sum(float(record.get("estimated_api_cost_usd") or 0) for record in records)
            / len(successes)
            if successes
            else None
        ),
        "total_calls": sum(int(record.get("call_count") or 0) for record in records),
        "input_tokens": sum(int(record.get("input_tokens") or 0) for record in records),
        "cached_input_tokens": sum(
            int(record.get("cached_input_tokens") or 0) for record in records
        ),
        "output_tokens": sum(int(record.get("output_tokens") or 0) for record in records),
        "metric_means": {
            **aspect_means,
            "aesthetic_average": mean("aesthetic_average"),
            "information_average": mean("information_average"),
            "overall_average": mean("overall_average"),
        },
        "failures": [
            {"paper_id": record["paper_id"], "error": record.get("error")}
            for record in failures
        ],
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "paper_id",
        "status",
        "skipped",
        *ALL_ASPECTS,
        "aesthetic_average",
        "information_average",
        "overall_average",
        "generation_seconds",
        "call_count",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "estimated_api_cost_usd",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["paper_id"]):
            row = {key: record.get(key) for key in columns}
            for aspect in ALL_ASPECTS:
                row[aspect] = record.get("scores", {}).get(aspect)
            writer.writerow(row)


def run_batch(
    input_root: str | Path,
    output_root: str | Path,
    *,
    config: JudgeConfig,
    poster_name: str = "poster.png",
    expected_paper_root: str | Path | None = None,
    jobs: int = 4,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate a directory of paper subdirectories and persist progress."""

    source_root = Path(input_root).expanduser().resolve()
    target_root = Path(output_root).expanduser().resolve()
    expected_root = (
        Path(expected_paper_root).expanduser().resolve()
        if expected_paper_root is not None
        else None
    )
    if not source_root.is_dir():
        raise ValueError(f"input root is not a directory: {source_root}")
    if expected_root is not None and not expected_root.is_dir():
        raise ValueError(f"expected paper root is not a directory: {expected_root}")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    target_root.mkdir(parents=True, exist_ok=True)

    pairs, missing = discover_posters(source_root, poster_name, expected_root)
    if limit is not None:
        pairs = pairs[:limit]
        allowed = {paper_id for paper_id, _ in pairs}
        missing = [record for record in missing if record["paper_id"] in allowed]

    started_at = _utc_now()
    started = time.monotonic()
    records: list[dict[str, Any]] = [
        {
            "paper_id": item["paper_id"],
            "status": "failed",
            "skipped": False,
            "error": item["error"],
            "scores": {},
        }
        for item in missing
    ]
    pending: list[tuple[str, Path]] = []
    for paper_id, image in pairs:
        existing = None
        if not force:
            existing = _load_success(
                target_root / paper_id / "judge_result.json",
                config,
                image,
            )
        if existing is not None:
            records.append(_record_from_result(paper_id, existing, skipped=True))
        else:
            pending.append((paper_id, image))

    status_path = target_root / "batch_status.json"

    def write_progress() -> None:
        payload = {
            "schema_version": 1,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "input_root": str(source_root),
            "expected_paper_root": str(expected_root) if expected_root else None,
            "output_root": str(target_root),
            "total": len(pairs) + len(missing),
            "completed": len(records),
            "pending": max(0, len(pairs) + len(missing) - len(records)),
            "records": sorted(records, key=lambda item: item["paper_id"]),
        }
        _atomic_json(status_path, payload)

    write_progress()

    def run_one(paper_id: str, image: Path) -> dict[str, Any]:
        paper_output = target_root / paper_id
        try:
            result = judge_poster(
                [image],
                paper_id=paper_id,
                output_dir=paper_output,
                config=config,
            )
            return _record_from_result(paper_id, result, skipped=False)
        except JudgeError as exc:
            failure_path = paper_output / "judge_failure.json"
            try:
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failure = {}
            return _record_from_failure(paper_id, failure, str(exc))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_one, paper_id, image): paper_id
            for paper_id, image in pending
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[batch] {len(records)}/{len(pairs) + len(missing)} "
                f"{record['paper_id']}: {record['status']}",
                flush=True,
            )
            write_progress()

    records.sort(key=lambda item: item["paper_id"])
    summary = _summary(records, config)
    summary["wall_seconds"] = round(time.monotonic() - started, 3)
    _atomic_json(target_root / "summary.json", summary)
    _write_csv(target_root / "results.csv", records)
    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "input_root": str(source_root),
            "expected_paper_root": str(expected_root) if expected_root else None,
            "output_root": str(target_root),
            "total": len(records),
            "completed": len(records),
            "pending": 0,
            "records": records,
        },
    )
    write_benchmark_reports(target_root, source_root, records, summary)
    return summary
