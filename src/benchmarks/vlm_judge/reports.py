"""Write benchmark-facing per-paper and average JSON reports."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rubrics import ALL_ASPECTS

JSON_NAN = "NaN"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else JSON_NAN


def _source_generation(
    source_root: Path,
    paper_id: str,
) -> dict[str, float | str]:
    """Load original poster-generation time and cost when recorded."""

    paper_root = source_root / paper_id
    adapter_path = paper_root / "adapter_run.json"
    harness_path = paper_root / "run_manifest.json"
    run: dict[str, Any] = {}
    source_kind: str | None = None
    for kind, path in (("adapter", adapter_path), ("harness", harness_path)):
        if not path.is_file():
            continue
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict):
            run = candidate
            source_kind = kind
            break

    if source_kind == "adapter":
        duration = _number(run.get("duration_seconds"))
        usage = run.get("codex_usage")
        cost = (
            _number(usage.get("estimated_api_cost_usd"))
            if isinstance(usage, dict)
            else None
        )
    elif source_kind == "harness":
        duration = _number(run.get("total_wall_seconds"))
        cost = _number(run.get("api_equivalent_cost_usd"))
    else:
        duration = None
        cost = None
    return {
        "duration_seconds": duration if duration is not None else JSON_NAN,
        "api_cost_usd": cost if cost is not None else JSON_NAN,
    }


def _paper_report(
    record: dict[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    return {
        "schema_version": 1,
        "paper_id": record["paper_id"],
        "status": record.get("status"),
        "error": record.get("error"),
        "metrics": {
            **{aspect: scores.get(aspect) for aspect in ALL_ASPECTS},
            "aesthetic_average": record.get("aesthetic_average"),
            "information_average": record.get("information_average"),
            "overall_average": record.get("overall_average"),
        },
        "evaluation": {
            "duration_seconds": record.get("generation_seconds"),
            "api_cost_usd": record.get("estimated_api_cost_usd"),
            "call_count": record.get("call_count"),
            "input_tokens": record.get("input_tokens"),
            "cached_input_tokens": record.get("cached_input_tokens"),
            "output_tokens": record.get("output_tokens"),
        },
        "poster_generation": _source_generation(source_root, record["paper_id"]),
    }


def write_benchmark_reports(
    output_root: Path,
    source_root: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write individual reports, ``per_paper.json``, and ``average.json``."""

    papers = [_paper_report(record, source_root) for record in records]
    for paper in papers:
        _atomic_json(
            output_root / paper["paper_id"] / "benchmark_result.json",
            paper,
        )

    _atomic_json(
        output_root / "per_paper.json",
        {
            "schema_version": 1,
            "updated_at": _utc_now(),
            "input_root": str(source_root),
            "total": len(papers),
            "papers": papers,
        },
    )

    successful = [paper for paper in papers if paper["status"] == "success"]
    generation_seconds = [
        number
        for paper in successful
        if (number := _number(paper["poster_generation"]["duration_seconds"]))
        is not None
    ]
    generation_costs = [
        number
        for paper in successful
        if (number := _number(paper["poster_generation"]["api_cost_usd"])) is not None
    ]
    _atomic_json(
        output_root / "average.json",
        {
            "schema_version": 1,
            "updated_at": _utc_now(),
            "configuration": summary["configuration"],
            "total": summary["total"],
            "success": summary["success"],
            "failed": summary["failed"],
            "metrics": summary["metric_means"],
            "evaluation": {
                "average_duration_seconds": summary["mean_generation_seconds"],
                "total_duration_seconds": summary["total_generation_seconds"],
                "average_api_cost_usd": summary["mean_api_cost_usd"],
                "total_api_cost_usd": summary["total_api_cost_usd"],
            },
            "poster_generation": {
                "average_duration_seconds": _mean(generation_seconds),
                "average_api_cost_usd": _mean(generation_costs),
                "recorded_count": len(generation_seconds),
            },
        },
    )
