"""Paper2Poster-compatible six-dimensional poster judge."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .codex_exec import (
    DEFAULT_LEDGER,
    CodexExecClient,
    CodexExecConfig,
    CodexExecError,
)
from .claude_cli import (
    DEFAULT_CLAUDE_LEDGER,
    ClaudeCLIClient,
    ClaudeCLIConfig,
    ClaudeCLIError,
)
from .rubrics import (
    AESTHETIC_ASPECTS,
    INFORMATION_ASPECTS,
    aspects_for_scope,
    combined_prompt,
    independent_prompt,
    validate_aspect_result,
    validate_combined_result,
)
from .schemas import combined_schema, single_aspect_schema


class JudgeError(RuntimeError):
    """Raised when a poster cannot be judged completely."""


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for one poster evaluation."""

    backend: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    call_mode: str = "independent"
    scope: str = "all"
    codex_bin: str = "codex"
    claude_bin: str = "claude"
    timeout_s: int = 900
    retries: int = 2
    aspect_jobs: int = 1
    ledger_path: Path | None = DEFAULT_LEDGER
    extra_codex_args: tuple[str, ...] = ()
    extra_claude_args: tuple[str, ...] = ()
    enforce_no_tools: bool = True


def backend_display(config: JudgeConfig) -> str:
    """Stable persisted name for the selected model backend."""

    if config.backend == "codex":
        return "codex exec (API-style)"
    if config.backend == "claude-cli":
        return "claude CLI (API-style image message)"
    raise JudgeError(f"unsupported backend: {config.backend!r}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _averages(results: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    aesthetic = [
        float(results[aspect]["score"])
        for aspect in AESTHETIC_ASPECTS
        if aspect in results
    ]
    information = [
        float(results[aspect]["score"])
        for aspect in INFORMATION_ASPECTS
        if aspect in results
    ]
    all_scores = [float(item["score"]) for item in results.values()]
    return {
        "aesthetic_average": _mean(aesthetic),
        "information_average": _mean(information),
        "overall_average": _mean(all_scores),
    }


def judge_poster(
    images: Sequence[str | Path],
    *,
    paper_id: str,
    output_dir: str | Path | None = None,
    config: JudgeConfig | None = None,
) -> dict[str, Any]:
    """Judge one poster and optionally persist ``judge_result.json``.

    ``independent`` mode makes one isolated model call per aspect, matching the
    upstream Paper2Poster evaluation. ``combined`` mode preserves all six
    rubrics but requests the selected aspects in one lower-cost call.
    """

    config = config or JudgeConfig()
    backend_name = backend_display(config)
    if config.call_mode not in {"independent", "combined"}:
        raise JudgeError(f"unsupported call mode: {config.call_mode!r}")
    if config.timeout_s <= 0:
        raise JudgeError("timeout_s must be positive")
    if config.retries < 0:
        raise JudgeError("retries must be non-negative")
    if config.aspect_jobs < 1:
        raise JudgeError("aspect_jobs must be at least 1")
    try:
        aspects = aspects_for_scope(config.scope)
    except ValueError as exc:
        raise JudgeError(str(exc)) from exc

    image_paths = [Path(image).expanduser().resolve() for image in images]
    if not image_paths:
        raise JudgeError("at least one poster image is required")
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise JudgeError(f"missing poster images: {missing}")

    resolved_output = Path(output_dir).expanduser().resolve() if output_dir else None
    started_at = _utc_now()
    trace_tag = started_at.replace(":", "").replace("+", "_").replace(".", "_")
    trace_family = "codex_traces" if config.backend == "codex" else "claude_traces"
    trace_dir = (
        resolved_output / trace_family / f"{trace_tag}_{os.getpid()}"
        if resolved_output
        else None
    )
    if config.backend == "codex":
        client = CodexExecClient(
            CodexExecConfig(
                executable=config.codex_bin,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                timeout_s=config.timeout_s,
                max_retries=config.retries,
                trace_dir=trace_dir,
                ledger_path=config.ledger_path,
                result_dir=resolved_output,
                paper_id=paper_id,
                extra_args=config.extra_codex_args,
                enforce_no_tools=config.enforce_no_tools,
            )
        )
    else:
        ledger_path = config.ledger_path
        if ledger_path == DEFAULT_LEDGER:
            ledger_path = DEFAULT_CLAUDE_LEDGER
        client = ClaudeCLIClient(
            ClaudeCLIConfig(
                executable=config.claude_bin,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                timeout_s=config.timeout_s,
                max_retries=config.retries,
                trace_dir=trace_dir,
                ledger_path=ledger_path,
                result_dir=resolved_output,
                paper_id=paper_id,
                extra_args=config.extra_claude_args,
                enforce_no_tools=config.enforce_no_tools,
            )
        )

    started = time.monotonic()
    results: dict[str, dict[str, Any]] = {}
    try:
        if config.call_mode == "independent":
            def judge_aspect(aspect: str) -> dict[str, Any]:
                return client.invoke_json(
                    prompt=independent_prompt(aspect, len(image_paths)),
                    images=image_paths,
                    response_schema=single_aspect_schema(),
                    validator=validate_aspect_result,
                    label=aspect,
                )

            workers = min(config.aspect_jobs, len(aspects))
            if workers == 1:
                for aspect in aspects:
                    results[aspect] = judge_aspect(aspect)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        aspect: pool.submit(judge_aspect, aspect)
                        for aspect in aspects
                    }
                    for aspect in aspects:
                        results[aspect] = futures[aspect].result()
        else:
            combined = client.invoke_json(
                prompt=combined_prompt(aspects, len(image_paths)),
                images=image_paths,
                response_schema=combined_schema(aspects),
                validator=lambda value: validate_combined_result(value, aspects),
                label=f"combined:{config.scope}",
            )
            results.update(combined)
    except (CodexExecError, ClaudeCLIError) as exc:
        usage = client.summary()
        failure = {
            "schema_version": 1,
            "status": "failed",
            "paper_id": paper_id,
            "images": [str(path) for path in image_paths],
            "error": str(exc),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "generation_seconds": round(time.monotonic() - started, 3),
            "judge_usage": usage,
            **(
                {"codex_usage": usage}
                if config.backend == "codex"
                else {"claude_usage": usage}
            ),
            "configuration": {
                "backend": backend_name,
                "ephemeral": True,
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
                "trace_dir": str(trace_dir) if trace_dir else None,
            },
        }
        if resolved_output is not None:
            _atomic_json(resolved_output / "judge_failure.json", failure)
        raise JudgeError(str(exc)) from exc

    usage = client.summary()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "success",
        "paper_id": paper_id,
        "images": [str(path) for path in image_paths],
        **_averages(results),
        "results": results,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "generation_seconds": round(time.monotonic() - started, 3),
        "judge_usage": usage,
        **(
            {"codex_usage": usage}
            if config.backend == "codex"
            else {"claude_usage": usage}
        ),
        "configuration": {
            "backend": backend_name,
            "ephemeral": True,
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
            "timeout_s": config.timeout_s,
            "retries": config.retries,
            "trace_dir": str(trace_dir) if trace_dir else None,
        },
    }
    if resolved_output is not None:
        _atomic_json(resolved_output / "judge_result.json", result)
        failure_path = resolved_output / "judge_failure.json"
        if failure_path.exists():
            failure_path.unlink()
    return result
