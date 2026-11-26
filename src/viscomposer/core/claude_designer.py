"""Persistent tool-free Claude session for the poster design workbench."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

from src.semparser.llm.claude_cli import (
    base_command,
    message_jsonl,
    parse_stream_result,
    response_text,
    usage_record,
)
from src.viscomposer.core.codex_designer import (
    DesignerSessionError,
    DesignerTurn,
    _write_json,
)


class PersistentClaudeDesigner:
    """Start and resume one Claude designer with direct multimodal messages."""

    backend = "claude CLI (API-style message)"

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
        if resume and self.thread_id is None:
            raise DesignerSessionError("cannot resume before the initial designer turn")
        if not resume and self.thread_id is not None:
            raise DesignerSessionError("designer session has already started")

        expected_session = self.thread_id or str(uuid.uuid4())
        try:
            command = base_command(
                model=self.model,
                effort=self.effort,
                output_schema=resolved_schema,
            )
            command += ["--resume", expected_session] if resume else [
                "--session-id",
                expected_session,
            ]
            stdin = message_jsonl(prompt, resolved_images)
        except (OSError, ValueError, RuntimeError) as exc:
            raise DesignerSessionError(f"{type(exc).__name__}: {exc}") from exc

        turn_dir = self.trace_root / name
        turn_dir.mkdir(parents=True, exist_ok=False)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        _write_json(
            turn_dir / "command.json",
            {
                "command": command,
                "prompt_transport": "stream-json stdin",
                "image_transport": "base64 message blocks",
                "resume": resume,
                "expected_thread_id": expected_session,
                "images": [str(path) for path in resolved_images],
                "output_schema": str(resolved_schema),
                "tools": "disabled",
            },
        )
        print(
            f"[paper2poster/designer] {name}: backend=claude, model={self.model}, "
            f"effort={self.effort}, resume={resume}",
            flush=True,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=stdin,
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
                f"claude exited with status {completed.returncode}: "
                f"{completed.stderr.strip()[-2000:]}"
            )
        try:
            payload = parse_stream_result(completed.stdout)
            message = response_text(payload)
        except RuntimeError as exc:
            raise DesignerSessionError(str(exc)) from exc
        emitted_session = payload.get("session_id")
        actual_session = str(emitted_session or expected_session)
        if actual_session != expected_session:
            raise DesignerSessionError(
                f"resume changed session ID: {expected_session} -> {actual_session}"
            )
        self.thread_id = actual_session
        usage = usage_record(
            payload,
            duration,
            model=self.model,
            effort=self.effort,
        )
        usage["persistent_session"] = True
        _write_json(turn_dir / "usage.json", usage)
        (turn_dir / "last_message.txt").write_text(message, encoding="utf-8")
        _write_json(
            self.trace_root / "session.json",
            {
                "thread_id": self.thread_id,
                "backend": self.backend,
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
        """Start a persisted Claude designer session."""

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
        """Resume the same Claude designer session with visual feedback."""

        return self._run(
            name=name,
            prompt=prompt,
            images=images,
            output_schema=output_schema,
            resume=True,
        )
