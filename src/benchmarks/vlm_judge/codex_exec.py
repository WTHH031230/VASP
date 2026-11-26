"""Tool-free, API-style ``codex exec`` backend with audit and accounting."""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


class CodexExecError(RuntimeError):
    """Raised when all attempts fail or return an invalid response."""


DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"
PRICING = {
    "profile": "gpt-5.6-sol",
    "currency": "USD",
    "unit_tokens": 1_000_000,
    "input_per_million": 5.0,
    "cached_input_per_million": 0.5,
    "cache_write_input_per_million": 6.25,
    "output_per_million": 30.0,
    "long_context_threshold_tokens": 272_000,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "note": (
        "API-equivalent estimate. ChatGPT-authenticated Codex CLI usage is not "
        "necessarily billed as API usage."
    ),
}

Validator = Callable[[Any], Any]


# Keep the evaluator equivalent to a plain multimodal model API call: the
# poster is model input, and the only permitted output is the JSON judgment.
API_STYLE_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode",
    "code_mode_host",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "multi_agent_v2",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_snapshot",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "standalone_web_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "unified_exec",
    "workspace_dependencies",
)
API_STYLE_CONFIG_OVERRIDES = (
    'agents.enabled=false',
    'approval_policy="never"',
    'web_search="disabled"',
)
# ``codex exec`` can emit a non-fatal ``error`` item for a capability that was
# deliberately disabled (for example, code-mode host). This is runtime
# diagnostics, not evidence that the model invoked a tool. A fatal diagnostic
# still fails when it produces no valid final agent message, while real tool
# attempts retain their concrete item types and remain fail-closed.
NON_TOOL_ITEM_TYPES = {"agent_message", "reasoning", "error"}


@dataclass(frozen=True)
class CodexExecConfig:
    """Runtime settings for one poster's Codex calls."""

    executable: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "medium"
    timeout_s: int = 900
    max_retries: int = 2
    trace_dir: Path | None = None
    ledger_path: Path | None = DEFAULT_LEDGER
    result_dir: Path | None = None
    paper_id: str | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    enforce_no_tools: bool = True


@dataclass
class _RawEvents:
    final_message: str | None
    usage: dict[str, int]
    thread_id: str | None
    parse_errors: list[str]
    unexpected_items: list[dict[str, Any]]


def _parse_events(stdout: str) -> _RawEvents:
    final_message = None
    usage: dict[str, int] = {}
    thread_id = None
    parse_errors: list[str] = []
    unexpected_items: list[dict[str, Any]] = []
    unexpected_keys: set[tuple[str, str, str]] = set()
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            parse_errors.append(raw_line[-500:])
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        item = event.get("item") or {}
        event_type = str(event.get("type") or "")
        item_type = str(item.get("type") or "")
        if (
            event_type in {"item.started", "item.completed"}
            and item_type not in NON_TOOL_ITEM_TYPES
        ):
            key = (
                str(item.get("id") or ""),
                item_type,
                str(
                    item.get("command")
                    or item.get("name")
                    or item.get("server")
                    or ""
                ),
            )
            if key not in unexpected_keys:
                unexpected_keys.add(key)
                unexpected_items.append(
                    {
                        "id": item.get("id"),
                        "type": item_type or "unknown",
                        "name": item.get("name"),
                        "command": item.get("command"),
                        "server": item.get("server"),
                        "status": item.get("status"),
                    }
                )
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            final_message = item.get("text", "")
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    return _RawEvents(
        final_message,
        usage,
        thread_id,
        parse_errors,
        unexpected_items,
    )


def _parse_json_object(text: str) -> Any:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL)
    if fenced:
        value = fenced.group(1)
    return json.loads(value)


def _usage_record(
    call_id: int,
    label: str,
    usage: dict[str, int],
    duration_seconds: float,
    image_count: int,
    status: str,
    error: str | None,
    unexpected_items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", 0))
    cached = min(input_tokens, int(usage.get("cached_input_tokens", 0)))
    non_cached = input_tokens - cached
    cache_write = min(
        non_cached,
        int(usage.get("cache_write_input_tokens", 0)),
    )
    uncached = non_cached - cache_write
    output_tokens = int(usage.get("output_tokens", 0))
    long_context = input_tokens > PRICING["long_context_threshold_tokens"]
    input_multiplier = PRICING["long_context_input_multiplier"] if long_context else 1.0
    output_multiplier = PRICING["long_context_output_multiplier"] if long_context else 1.0
    cost = (
        (
            uncached * PRICING["input_per_million"]
            + cached * PRICING["cached_input_per_million"]
            + cache_write * PRICING["cache_write_input_per_million"]
        )
        * input_multiplier
        + output_tokens * PRICING["output_per_million"] * output_multiplier
    ) / PRICING["unit_tokens"]
    return {
        "call_id": call_id,
        "label": label,
        "status": status,
        "error": error,
        "duration_seconds": round(duration_seconds, 3),
        "image_count": image_count,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "uncached_input_tokens": uncached,
        "output_tokens": output_tokens,
        "long_context_pricing": long_context,
        "estimated_api_cost_usd": round(cost, 8),
        "tool_event_count": len(unexpected_items),
        "unexpected_tool_events": list(unexpected_items),
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all attempts, including invalid responses and retries."""

    copied = [dict(record) for record in records]
    return {
        "call_count": len(copied),
        "successful_calls": sum(record["status"] == "success" for record in copied),
        "failed_calls": sum(record["status"] != "success" for record in copied),
        "codex_call_seconds": round(
            sum(float(record["duration_seconds"]) for record in copied),
            3,
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
        "tool_event_count": sum(
            int(record.get("tool_event_count", 0)) for record in copied
        ),
        "estimated_api_cost_usd": round(
            sum(float(record["estimated_api_cost_usd"]) for record in copied),
            8,
        ),
        "pricing": dict(PRICING),
        "calls": copied,
    }


class CodexExecClient:
    """Run schema-constrained, ephemeral Codex calls for one poster."""

    def __init__(self, config: CodexExecConfig) -> None:
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
        """Call Codex and return a validated JSON value, retrying failures."""

        image_paths = [Path(path).expanduser().resolve() for path in images]
        if not image_paths:
            raise CodexExecError("at least one image is required")
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise CodexExecError(f"image files do not exist: {missing}")

        last_error = "unknown failure"
        attempts_made = 0
        for attempt in range(1, self.config.max_retries + 2):
            attempts_made = attempt
            call_id = self._next_call_id()
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="p2p-vlm-judge-") as temp_name:
                temp_dir = Path(temp_name)
                schema_path = temp_dir / "response.schema.json"
                schema_path.write_text(
                    json.dumps(response_schema, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                staged_images = self._stage_images(temp_dir, image_paths)
                command = self._command(schema_path, staged_images)
                try:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        cwd=temp_dir,
                        timeout=self.config.timeout_s,
                        check=False,
                    )
                    stdout = completed.stdout
                    stderr = completed.stderr
                    returncode = completed.returncode
                except FileNotFoundError as exc:
                    raise CodexExecError(
                        f"cannot find Codex executable {self.config.executable!r}"
                    ) from exc
                except subprocess.TimeoutExpired as exc:
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    if isinstance(stdout, bytes):
                        stdout = stdout.decode("utf-8", errors="replace")
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    returncode = None

                events = _parse_events(stdout)
                status = "success"
                error = None
                validated = None
                if returncode is None:
                    status = "timeout"
                    error = f"timed out after {self.config.timeout_s}s"
                elif returncode != 0:
                    status = "command_failed"
                    error = f"exit {returncode}: {stderr.strip()[-1600:]}"
                elif self.config.enforce_no_tools and events.parse_errors:
                    status = "event_stream_parse_error"
                    error = "non-JSON stdout prevents complete tool-event auditing"
                elif self.config.enforce_no_tools and events.unexpected_items:
                    status = "tool_use_detected"
                    kinds = ", ".join(
                        sorted({item["type"] for item in events.unexpected_items})
                    )
                    error = f"forbidden tool event(s) detected: {kinds}"
                elif events.final_message is None:
                    status = "missing_final_message"
                    error = "codex exec emitted no final agent message"
                else:
                    try:
                        validated = validator(_parse_json_object(events.final_message))
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                        status = "invalid_response"
                        error = str(exc)

                record = _usage_record(
                    call_id=call_id,
                    label=label,
                    usage=events.usage,
                    duration_seconds=time.monotonic() - started,
                    image_count=len(image_paths),
                    status=status,
                    error=error,
                    unexpected_items=events.unexpected_items,
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
                    thread_id=events.thread_id,
                )
                self._record(record, events.thread_id)
                self._print_record(record)
                if status == "success":
                    return validated
                last_error = error or status
                if status in {"event_stream_parse_error", "tool_use_detected"}:
                    break

        raise CodexExecError(
            f"{label} failed after {attempts_made} attempt(s): {last_error}"
        )

    def _next_call_id(self) -> int:
        with self._lock:
            return next(self._ids)

    def _command(self, schema_path: Path, images: Sequence[Path]) -> list[str]:
        command = [
            self.config.executable,
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--output-schema",
            str(schema_path),
            "--model",
            self.config.model,
            "--config",
            f'model_reasoning_effort="{self.config.reasoning_effort}"',
        ]
        for image_path in images:
            command.extend(("--image", str(image_path)))
        command.extend(self.config.extra_args)
        if self.config.enforce_no_tools:
            for feature in API_STYLE_DISABLED_FEATURES:
                command.extend(("--disable", feature))
            for override in API_STYLE_CONFIG_OVERRIDES:
                command.extend(("--config", override))
            # Apply the locked-down overrides after user-supplied extra args and
            # reject unknown config keys, so the no-tool policy cannot drift.
            command.append("--strict-config")
        command.append("-")
        return command

    @staticmethod
    def _stage_images(temp_dir: Path, images: Sequence[Path]) -> list[Path]:
        """Copy images to comma-free paths accepted by the Codex CLI.

        The Codex CLI treats commas inside each --image value as separators,
        even when subprocess receives the path as one correctly quoted argument.
        Paper titles commonly contain commas, so pass stable temporary filenames
        instead of exposing the original path to the CLI parser.
        """

        staged: list[Path] = []
        for index, source in enumerate(images, start=1):
            suffix = source.suffix.lower()
            if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix) is None:
                suffix = ".img"
            target = temp_dir / "image_{:04d}{}".format(index, suffix)
            shutil.copy2(source, target)
            staged.append(target)
        return staged

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
        thread_id: str | None,
    ) -> None:
        trace_dir = self.config.trace_dir
        if trace_dir is None:
            return
        stem = trace_dir / f"call_{call_id:04d}"
        stem.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        stem.with_suffix(".schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        stem.with_suffix(".events.jsonl").write_text(stdout, encoding="utf-8")
        stem.with_suffix(".stderr.txt").write_text(stderr, encoding="utf-8")
        stem.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "label": label,
                    "command": list(command),
                    "thread_id": thread_id,
                    "usage": record,
                    "note": "The prompt is supplied on stdin.",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _record(self, record: dict[str, Any], thread_id: str | None) -> None:
        with self._lock:
            self._records.append(dict(record))
        ledger_path = self.config.ledger_path
        if ledger_path is None:
            return
        entry = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "backend": "codex exec (API-style)",
            "ephemeral": True,
            "tool_policy": (
                "disabled-fail-closed"
                if self.config.enforce_no_tools
                else "agent-tools-allowed"
            ),
            "application": "benchmark/vlm_judge",
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "paper_id": self.config.paper_id,
            "thread_id": thread_id,
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
                output.write(
                    json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                output.flush()
                try:
                    os.fsync(output.fileno())
                except OSError:
                    pass
                try:
                    import fcntl

                    fcntl.flock(output.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
        except OSError as exc:
            print(f"[vlm-judge] warning: cannot append ledger: {exc}", file=sys.stderr)

    @staticmethod
    def _print_record(record: dict[str, Any]) -> None:
        print(
            "[vlm-judge] call {call_id:04d} {label}: {status}, {duration_seconds:.1f}s, "
            "in={input_tokens}, cached={cached_input_tokens}, out={output_tokens}, "
            "api-equivalent=${estimated_api_cost_usd:.4f}".format(**record),
            flush=True,
        )
