"""Tool-free Claude CLI backend using direct multimodal message input."""

from __future__ import annotations

import base64
import itertools
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


class ClaudeCLIError(RuntimeError):
    """Raised when all Claude CLI attempts fail or return invalid output."""


DEFAULT_CLAUDE_LEDGER = (
    Path.home() / ".claude" / "usage-ledger" / "vlm-judge.jsonl"
)

Validator = Callable[[Any], Any]


@dataclass(frozen=True)
class ClaudeCLIConfig:
    """Runtime settings for one poster's Claude calls."""

    executable: str = "claude"
    model: str = "claude-opus-5"
    reasoning_effort: str = "high"
    timeout_s: int = 900
    max_retries: int = 2
    trace_dir: Path | None = None
    ledger_path: Path | None = DEFAULT_CLAUDE_LEDGER
    result_dir: Path | None = None
    paper_id: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    enforce_no_tools: bool = True


def _parse_json_object(text: str) -> Any:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    return json.loads(value)


def _parse_stream_result(stdout: str) -> dict[str, Any]:
    """Return the terminal Claude result event from stream-json output."""

    result: dict[str, Any] | None = None
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        event = json.loads(raw_line)
        if not isinstance(event, dict):
            raise TypeError("Claude stream event is not a JSON object")
        if event.get("type") == "result":
            result = event
    if result is None:
        raise ValueError("Claude stream emitted no terminal result event")
    return result


def _message(prompt: str, images: Sequence[Path]) -> str:
    """Build one Claude stream-json user message with direct image blocks."""

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
        {
            "type": "user",
            "message": {"role": "user", "content": content},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"


def _usage_record(
    *,
    call_id: int,
    label: str,
    payload: dict[str, Any],
    duration_seconds: float,
    image_count: int,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    uncached = int(usage.get("input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    cached = int(usage.get("cache_read_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    cost = float(payload.get("total_cost_usd") or 0)
    model_usage = payload.get("modelUsage")
    return {
        "call_id": call_id,
        "label": label,
        "status": status,
        "error": error,
        "duration_seconds": round(duration_seconds, 3),
        "image_count": image_count,
        "input_tokens": uncached + cache_write + cached,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "uncached_input_tokens": uncached,
        "output_tokens": output,
        "estimated_api_cost_usd": round(cost, 8),
        "tool_event_count": 0,
        "unexpected_tool_events": [],
        "session_id": payload.get("session_id"),
        "model_usage": model_usage if isinstance(model_usage, dict) else {},
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all Claude attempts, including retries."""

    copied = [dict(record) for record in records]
    return {
        "call_count": len(copied),
        "successful_calls": sum(record["status"] == "success" for record in copied),
        "failed_calls": sum(record["status"] != "success" for record in copied),
        "claude_call_seconds": round(
            sum(float(record["duration_seconds"]) for record in copied), 3
        ),
        "input_tokens": sum(int(record["input_tokens"]) for record in copied),
        "cached_input_tokens": sum(
            int(record["cached_input_tokens"]) for record in copied
        ),
        "cache_write_input_tokens": sum(
            int(record["cache_write_input_tokens"]) for record in copied
        ),
        "uncached_input_tokens": sum(
            int(record["uncached_input_tokens"]) for record in copied
        ),
        "output_tokens": sum(int(record["output_tokens"]) for record in copied),
        "tool_event_count": 0,
        "estimated_api_cost_usd": round(
            sum(float(record["estimated_api_cost_usd"]) for record in copied), 8
        ),
        "pricing": {
            "source": "claude CLI total_cost_usd",
            "note": (
                "API-equivalent value reported by Claude Code CLI; subscription "
                "usage is not necessarily billed as API usage."
            ),
        },
        "calls": copied,
    }


class ClaudeCLIClient:
    """Run schema-constrained, tool-free Claude calls for one poster."""

    def __init__(self, config: ClaudeCLIConfig) -> None:
        self.config = config
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        if config.trace_dir is not None:
            config.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records]

    def summary(self) -> dict[str, Any]:
        return summarize_records(self.records)

    def invoke_json(
        self,
        *,
        prompt: str,
        images: Sequence[Path],
        response_schema: dict[str, Any],
        validator: Validator,
        label: str,
    ) -> Any:
        """Call Claude and return a validated JSON value, retrying failures."""

        image_paths = [Path(path).expanduser().resolve() for path in images]
        if not image_paths:
            raise ClaudeCLIError("at least one image is required")
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise ClaudeCLIError(f"image files do not exist: {missing}")

        input_message = _message(prompt, image_paths)
        last_error = "unknown failure"
        attempts_made = 0
        for attempt in range(1, self.config.max_retries + 2):
            attempts_made = attempt
            call_id = self._next_call_id()
            command = self._command(response_schema)
            started = time.monotonic()
            stdout = ""
            stderr = ""
            payload: dict[str, Any] = {}
            try:
                completed = subprocess.run(
                    command,
                    input=input_message,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout_s,
                    check=False,
                )
                stdout = completed.stdout
                stderr = completed.stderr
                returncode = completed.returncode
            except FileNotFoundError as exc:
                raise ClaudeCLIError(
                    f"cannot find Claude executable {self.config.executable!r}"
                ) from exc
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                returncode = None

            status = "success"
            error = None
            validated = None
            if returncode is None:
                status = "timeout"
                error = f"timed out after {self.config.timeout_s}s"
            elif returncode != 0:
                status = "command_failed"
                error = f"exit {returncode}: {stderr.strip()[-1600:]}"
            else:
                try:
                    payload = _parse_stream_result(stdout)
                    if payload.get("is_error"):
                        raise ValueError(str(payload.get("result") or payload))
                    response = payload.get("structured_output")
                    if response is None:
                        result_text = payload.get("result")
                        if not isinstance(result_text, str) or not result_text.strip():
                            raise ValueError("Claude output has no result")
                        response = _parse_json_object(result_text)
                    validated = validator(response)
                    if self.config.enforce_no_tools:
                        denials = payload.get("permission_denials") or []
                        if denials:
                            raise ValueError(f"Claude attempted forbidden tools: {denials}")
                except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                    status = "invalid_response"
                    error = str(exc)

            record = _usage_record(
                call_id=call_id,
                label=label,
                payload=payload,
                duration_seconds=time.monotonic() - started,
                image_count=len(image_paths),
                status=status,
                error=error,
            )
            self._save_trace(
                call_id=call_id,
                label=label,
                prompt=prompt,
                schema=response_schema,
                command=command,
                stdout=stdout,
                stderr=stderr,
                record=record,
            )
            self._record(record)
            self._print_record(record)
            if status == "success":
                return validated
            last_error = error or status

        raise ClaudeCLIError(
            f"{label} failed after {attempts_made} attempt(s): {last_error}"
        )

    def _next_call_id(self) -> int:
        with self._lock:
            return next(self._ids)

    def _command(self, response_schema: dict[str, Any]) -> list[str]:
        command = [
            self.config.executable,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.config.model,
            "--effort",
            self.config.reasoning_effort,
            "--json-schema",
            json.dumps(response_schema, ensure_ascii=False, separators=(",", ":")),
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
            "--safe-mode",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
        ]
        command.extend(self.config.extra_args)
        if self.config.enforce_no_tools:
            command.extend(("--tools", ""))
        return command

    def _save_trace(
        self,
        *,
        call_id: int,
        label: str,
        prompt: str,
        schema: dict[str, Any],
        command: Sequence[str],
        stdout: str,
        stderr: str,
        record: dict[str, Any],
    ) -> None:
        trace_dir = self.config.trace_dir
        if trace_dir is None:
            return
        stem = trace_dir / f"call_{call_id:04d}"
        stem.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        stem.with_suffix(".schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stem.with_suffix(".response.json").write_text(stdout, encoding="utf-8")
        stem.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
        stem.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "command": list(command),
                    "usage": record,
                    "note": "Prompt and base64 images are supplied as stream-json stdin.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(dict(record))
        ledger_path = self.config.ledger_path
        if ledger_path is None:
            return
        entry = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "backend": "claude CLI (API-style image message)",
            "ephemeral": True,
            "tool_policy": "disabled" if self.config.enforce_no_tools else "allowed",
            "application": "benchmark/vlm_judge",
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "paper_id": self.config.paper_id,
            "trace_dir": str(self.config.trace_dir) if self.config.trace_dir else None,
            "result_dir": str(self.config.result_dir) if self.config.result_dir else None,
            **record,
        }
        try:
            target = ledger_path.expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as output:
                try:
                    import fcntl

                    fcntl.flock(output.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                output.write(json.dumps(entry, ensure_ascii=False) + "\n")
                output.flush()
                try:
                    os.fsync(output.fileno())
                except OSError:
                    pass
        except OSError as exc:
            print(f"[vlm-judge] warning: cannot append Claude ledger: {exc}", file=sys.stderr)

    @staticmethod
    def _print_record(record: dict[str, Any]) -> None:
        print(
            "[vlm-judge/claude] call {call_id:04d} {label}: {status}, "
            "{duration_seconds:.1f}s, in={input_tokens}, cached={cached_input_tokens}, "
            "out={output_tokens}, api-equivalent=${estimated_api_cost_usd:.4f}".format(
                **record
            ),
            flush=True,
        )
