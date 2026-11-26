"""Persistent ``codex exec`` session used by the poster design workbench.

Unlike the response-only asset-extraction adapter, this backend gives one Codex
session an isolated writable directory and resumes that same session with each
browser screenshot.  Every turn is traced and costed independently.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.semparser.llm.codex_exec import _usage_record


class DesignerSessionError(RuntimeError):
    """Raised when a persistent designer turn cannot be completed."""


def _parse_events(stdout: str) -> tuple[str, dict[str, int], str | None]:
    final_message: str | None = None
    usage: dict[str, int] = {}
    thread_id: str | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            final_message = str(item.get("text", ""))
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    if final_message is None:
        raise DesignerSessionError("codex exec emitted no final agent message")
    return final_message, usage, thread_id


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class DesignerTurn:
    """Result of one initial or resumed designer turn."""

    name: str
    message: str
    usage: dict[str, Any]
    thread_id: str


class PersistentCodexDesigner:
    """Run and resume one Codex designer inside a writable run workspace."""

    backend = "codex exec"

    def __init__(
        self,
        *,
        workspace: Path,
        trace_root: Path,
        model: str,
        effort: str,
        timeout: int | None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.trace_root = trace_root.resolve()
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.thread_id: str | None = None

    def _base_options(self) -> list[str]:
        return [
            "--json",
            "--skip-git-repo-check",
            "--ignore-user-config",
            "--ignore-rules",
            "--disable",
            "multi_agent",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.effort}"',
        ]

    def _run(
        self,
        *,
        name: str,
        prompt: str,
        images: list[Path],
        output_schema: Path,
        resume: bool,
    ) -> DesignerTurn:
        resolved_images = [path.resolve() for path in images]
        resolved_schema = output_schema.resolve()
        if not resolved_schema.is_file():
            raise DesignerSessionError(f"missing output schema: {resolved_schema}")
        missing = [path for path in resolved_images if not path.is_file()]
        if missing:
            raise DesignerSessionError(
                "missing designer image input(s): " + ", ".join(map(str, missing))
            )
        if resume:
            if self.thread_id is None:
                raise DesignerSessionError("cannot resume before the initial designer turn")
            command = ["codex", "exec", "resume", *self._base_options()]
            for image in resolved_images:
                command += ["--image", str(image)]
            command += ["--output-schema", str(resolved_schema), self.thread_id, "-"]
        else:
            command = [
                "codex",
                "exec",
                *self._base_options(),
                "--sandbox",
                "read-only",
                "--cd",
                str(self.workspace),
            ]
            if resolved_images:
                command += ["--image", *map(str, resolved_images)]
            command += ["--output-schema", str(resolved_schema), "-"]

        turn_dir = self.trace_root / name
        turn_dir.mkdir(parents=True, exist_ok=False)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        _write_json(
            turn_dir / "command.json",
            {
                "command": command,
                "prompt_transport": "stdin",
                "resume": resume,
                "expected_thread_id": self.thread_id,
                "images": [str(path) for path in resolved_images],
                "output_schema": str(resolved_schema),
            },
        )
        print(
            f"[paper2poster/designer] {name}: model={self.model}, "
            f"effort={self.effort}, resume={resume}",
            flush=True,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.workspace),
                timeout=self.timeout,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise DesignerSessionError(f"{type(exc).__name__}: {exc}") from exc
        duration = time.monotonic() - started
        (turn_dir / "events.jsonl").write_text(completed.stdout, encoding="utf-8")
        (turn_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise DesignerSessionError(
                f"codex exec exited with status {completed.returncode}: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        message, raw_usage, emitted_thread_id = _parse_events(completed.stdout)
        actual_thread_id = emitted_thread_id or self.thread_id
        if actual_thread_id is None:
            raise DesignerSessionError("codex exec did not report a persistent thread ID")
        if self.thread_id is not None and actual_thread_id != self.thread_id:
            raise DesignerSessionError(
                f"resume changed thread ID: {self.thread_id} -> {actual_thread_id}"
            )
        self.thread_id = actual_thread_id
        usage = _usage_record(
            raw_usage,
            duration,
            model=self.model,
            effort=self.effort,
            thread_id=self.thread_id,
        )
        usage["persistent_session"] = True
        _write_json(turn_dir / "usage.json", usage)
        (turn_dir / "last_message.txt").write_text(message, encoding="utf-8")
        _write_json(
            self.trace_root / "session.json",
            {
                "thread_id": self.thread_id,
                "model": self.model,
                "reasoning_effort": self.effort,
                "workspace": str(self.workspace),
                "latest_turn": name,
            },
        )
        print(
            f"[paper2poster/designer] {name}: {usage['duration_seconds']:.1f}s, "
            f"in={usage['input_tokens']}, cached={usage['cached_input_tokens']}, "
            f"out={usage['output_tokens']}, "
            f"api-equivalent=${usage['estimated_api_cost_usd']:.4f}",
            flush=True,
        )
        return DesignerTurn(name, message, usage, self.thread_id)

    def start(
        self, *, name: str, prompt: str, images: list[Path], output_schema: Path
    ) -> DesignerTurn:
        """Start a persisted designer session."""

        if self.thread_id is not None:
            raise DesignerSessionError("designer session has already started")
        return self._run(
            name=name,
            prompt=prompt,
            images=images,
            output_schema=output_schema,
            resume=False,
        )

    def resume(
        self, *, name: str, prompt: str, images: list[Path], output_schema: Path
    ) -> DesignerTurn:
        """Resume the same designer session with new visual feedback."""

        return self._run(
            name=name,
            prompt=prompt,
            images=images,
            output_schema=output_schema,
            resume=True,
        )
