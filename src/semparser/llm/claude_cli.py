"""Tool-free Claude CLI backend with direct API-style message input."""

from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from .codex_exec import _append_ledger, _render_prompt, _write_trace


class ClaudeCLIError(RuntimeError):
    """Raised when Claude CLI fails or emits no terminal result."""


def is_claude_model(model: str) -> bool:
    """Return whether ``model`` should be routed to Claude CLI."""

    lowered = model.casefold()
    return lowered.startswith("claude-") or lowered.startswith(("opus", "sonnet"))


def message_jsonl(prompt: str, images: Sequence[Path]) -> str:
    """Build one stream-json user message containing direct image blocks."""

    content: list[dict[str, Any]] = []
    for path in images:
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        if media_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
            raise ClaudeCLIError(f"unsupported Claude image media type: {media_type}")
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                },
            }
        )
    content.append({"type": "text", "text": prompt})
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}},
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def parse_stream_result(stdout: str) -> dict[str, Any]:
    """Return the terminal result from Claude stream-json output."""

    result: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is None:
        raise ClaudeCLIError("Claude CLI emitted no terminal result event")
    return result


def usage_record(
    payload: dict[str, Any],
    duration_seconds: float,
    *,
    model: str,
    effort: str,
) -> dict[str, Any]:
    """Normalize Claude CLI's measured token and cost fields."""

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    uncached = int(usage.get("input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    return {
        "backend": "claude CLI (API-style message)",
        "model": model,
        "reasoning_effort": effort,
        "thread_id": payload.get("session_id"),
        "session_id": payload.get("session_id"),
        "duration_seconds": round(duration_seconds, 3),
        "input_tokens": uncached + cache_write + cached,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "uncached_input_tokens": uncached,
        "output_tokens": output,
        "estimated_api_cost_usd": round(float(payload.get("total_cost_usd") or 0), 8),
        "pricing": {
            "source": "claude CLI total_cost_usd",
            "note": (
                "API-equivalent value reported by Claude Code CLI; subscription "
                "usage is not necessarily billed as API usage."
            ),
        },
        "model_usage": payload.get("modelUsage") or {},
    }


def _draft7_schema(value: Any) -> Any:
    """Convert the pipeline's self-contained 2020-12 schema to Draft 7 syntax."""

    if isinstance(value, list):
        return [_draft7_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    converted: dict[str, Any] = {}
    for key, item in value.items():
        if key == "$schema":
            continue
        converted["definitions" if key == "$defs" else key] = _draft7_schema(item)
    reference = converted.get("$ref")
    if isinstance(reference, str):
        converted["$ref"] = reference.replace("#/$defs/", "#/definitions/")
    return converted


def base_command(*, model: str, effort: str, output_schema: Path | None) -> list[str]:
    """Construct the common isolated, tool-free Claude CLI invocation."""

    command = [
        "claude",
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--effort",
        effort,
        "--permission-mode",
        "dontAsk",
        "--safe-mode",
        "--restricted",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--disable-slash-commands",
        "--tools",
        "",
    ]
    if output_schema is not None:
        schema = _draft7_schema(json.loads(output_schema.read_text(encoding="utf-8")))
        command += [
            "--json-schema",
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        ]
    return command


def response_text(payload: dict[str, Any]) -> str:
    """Extract a plain JSON/text answer from a terminal Claude result."""

    if payload.get("is_error"):
        raise ClaudeCLIError(str(payload.get("result") or payload))
    denials = payload.get("permission_denials") or []
    if denials:
        raise ClaudeCLIError(f"Claude attempted forbidden tools: {denials}")
    structured = payload.get("structured_output")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ClaudeCLIError("Claude CLI result has no response text")
    return result


def call_claude(
    prompt: str,
    *,
    model: str = "claude-opus-5",
    effort: str = "high",
    timeout: int = 1800,
    cwd: Path | None = None,
    trace_dir: Path,
    call_name: str,
    result_dir: Path,
    images: list[Path] | None = None,
    output_schema: Path | None = None,
    application: str = "asset_extraction",
) -> tuple[str | None, dict[str, Any] | str]:
    """Run one stateless, tool-free Claude call and return ``(answer, usage)``."""

    cwd = (cwd or Path.cwd()).resolve()
    trace_dir = trace_dir.resolve()
    result_dir = result_dir.resolve()
    resolved_images = [path.resolve() for path in (images or [])]
    missing = [path for path in resolved_images if not path.is_file()]
    if missing:
        return None, "missing image input(s): " + ", ".join(map(str, missing))
    resolved_schema = output_schema.resolve() if output_schema else None
    if resolved_schema is not None and not resolved_schema.is_file():
        return None, f"missing output schema: {resolved_schema}"
    rendered = _render_prompt(prompt, application)
    try:
        stdin = message_jsonl(rendered, resolved_images)
        command = base_command(model=model, effort=effort, output_schema=resolved_schema)
    except (OSError, ValueError, ClaudeCLIError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    command += ["--no-session-persistence"]
    print(f"[{application}/claude] {call_name}: model={model}, effort={effort}", flush=True)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    _write_trace(trace_dir, call_name, rendered, completed.stdout, completed.stderr, command)
    if completed.returncode != 0:
        return None, (
            f"claude exited with status {completed.returncode}: "
            f"{completed.stderr.strip()[-2000:]}"
        )
    try:
        payload = parse_stream_result(completed.stdout)
        answer = response_text(payload)
    except ClaudeCLIError as exc:
        return None, str(exc)
    record = usage_record(
        payload,
        time.monotonic() - started,
        model=model,
        effort=effort,
    )
    _append_ledger(
        record,
        application=application,
        cwd=cwd,
        trace_dir=trace_dir,
        result_dir=result_dir,
    )
    print(
        f"[{application}/claude] {call_name}: {record['duration_seconds']:.1f}s, "
        f"in={record['input_tokens']}, cached={record['cached_input_tokens']}, "
        f"out={record['output_tokens']}, "
        f"api-equivalent=${record['estimated_api_cost_usd']:.4f}",
        flush=True,
    )
    return answer, record
