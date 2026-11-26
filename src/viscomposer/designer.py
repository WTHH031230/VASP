"""Designer backends for the pi_poster runner.

PiDesigner keeps one pi process per paper in RPC mode, so every turn continues the same
session (the persistent-session design of the codex harness). Each turn selects its
submit tool with /poster-turn, sends the prompt and images, waits until pi settles, and
returns the arguments of the accepted submit call. Token usage is priced per LLM call
with the codex harness's own API-equivalent table, so costs stay comparable.

ReplayDesigner returns stored main-batch responses (pruned to the arm's schema) for
offline end-to-end checks without a model.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.viscomposer.arms import SEM, VF, Arm
from src.viscomposer.gate import NODE_FIELDS, STATE_FIELDS, owns
from src.semparser.llm.codex_exec import _usage_record

PI_DIR = Path(__file__).resolve().parent / "pi"
PI_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "pi"
SYSTEM_PROMPT = (
    "You are a research-poster designer working inside a host-controlled harness. "
    "The user messages contain the task, the inputs, and the response contract."
)
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class DesignerError(RuntimeError):
    """A turn ended without an accepted structured submission."""


def _image(path: Path) -> dict[str, str]:
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        raise DesignerError(f"unsupported image type: {path}")
    return {"type": "image", "data": base64.b64encode(path.read_bytes()).decode("ascii"), "mimeType": mime}


class PiDesigner:
    backend = "pi rpc"

    def __init__(self, *, workspace: Path, trace_root: Path, schema_dir: Path, session_dir: Path,
                 model: str, effort: str, provider: str = "openai-compat", pricing_model: str = "gpt-5.6-sol",
                 pi_bin: str | None = None, timeout: float | None = None) -> None:
        self.trace_root = Path(trace_root)
        self.model, self.effort, self.pricing_model, self.timeout = model, effort, pricing_model, timeout
        default_pi = os.environ.get("PI_POSTER_PI_BIN", str(PI_DIR / "node_modules" / ".bin" / "pi"))
        agent_dir = os.environ.get("PI_POSTER_AGENT_DIR", str(PI_CONFIG_DIR))
        self.pi_bin = pi_bin or default_pi
        self.command = [
            self.pi_bin,
            "--mode", "rpc", "--provider", provider, "--model", model, "--thinking", effort,
            "--no-builtin-tools", "--no-extensions", "-e", str(PI_DIR / "poster_ext.ts"),
            "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files",
            "--session-dir", str(session_dir), "--system-prompt", SYSTEM_PROMPT,
        ]
        env = {**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1",
               "PI_POSTER_SCHEMA_DIR": str(schema_dir)}
        proxy = os.environ.get("PI_POSTER_PROXY", "")
        if proxy:
            for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
                env.setdefault(k, proxy)
            env.setdefault("NODE_USE_ENV_PROXY", "1")
        self._stderr = open(self.trace_root / "pi_stderr.log", "ab")
        # pi appends <cwd> to the system prompt; one fixed directory keeps run and arm
        # names out of the model input. No file tools are exposed, so cwd is otherwise unused.
        neutral = PI_DIR / "cwd"
        neutral.mkdir(exist_ok=True)
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._request = 0
        if provider == "openai":
            lock_path = Path(agent_dir) / "startup.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    self.proc = subprocess.Popen(self.command, cwd=neutral, env=env, stdin=subprocess.PIPE,
                                                 stdout=subprocess.PIPE, stderr=self._stderr)
                    threading.Thread(target=self._read, daemon=True).start()
                    state = self._command({"type": "get_state"}, None, [])
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        else:
            self.proc = subprocess.Popen(self.command, cwd=neutral, env=env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self._stderr)
            threading.Thread(target=self._read, daemon=True).start()
            state = self._command({"type": "get_state"}, None, [])
        self.thread_id = (state.get("data") or {}).get("sessionId")

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:  # binary pipe: records split on b"\n" only (RPC framing)
            self._lines.put(raw)
        self._lines.put(None)

    def _next(self, deadline: float | None) -> dict[str, Any]:
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            raw = self._lines.get(timeout=wait)
        except queue.Empty as exc:
            raise DesignerError("pi turn timed out") from exc
        if raw is None:
            raise DesignerError(f"pi exited with code {self.proc.poll()}")
        return json.loads(raw.decode("utf-8").rstrip("\r\n"))

    def _command(self, payload: dict[str, Any], deadline: float | None,
                 sink: list[dict[str, Any]]) -> dict[str, Any]:
        self._request += 1
        request_id = f"req-{self._request}"
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps({"id": request_id, **payload}, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        while True:
            event = self._next(deadline)
            sink.append(event)
            if event.get("type") == "response" and event.get("id") == request_id:
                if not event.get("success"):
                    raise DesignerError(f"pi rejected {payload['type']}: {event.get('error')}")
                return event

    def _usage(self, calls: list[dict[str, Any]], duration: float, attempts: int) -> dict[str, Any]:
        records = []
        reported = {"input": 0, "cacheRead": 0, "cacheWrite": 0, "output": 0, "reasoning": 0}
        for message in calls:
            usage = message.get("usage") or {}
            for key in reported:
                reported[key] += int(usage.get(key) or 0)
            read, write = int(usage.get("cacheRead") or 0), int(usage.get("cacheWrite") or 0)
            # Some proxies report prompt tokens as cacheWrite. OpenAI list prices have no
            # cache-write charge, so they are priced as uncached input, as codex runs were.
            records.append(_usage_record(
                {"input_tokens": int(usage.get("input") or 0) + read + write,
                 "cached_input_tokens": read, "cache_write_input_tokens": 0,
                 "output_tokens": int(usage.get("output") or 0)},
                0.0, model=self.pricing_model, effort=self.effort, thread_id=self.thread_id))
        keys = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                "uncached_input_tokens", "output_tokens")
        total = {key: sum(int(r[key]) for r in records) for key in keys}
        return {
            "backend": self.backend, "model": self.model, "reasoning_effort": self.effort,
            "thread_id": self.thread_id, "duration_seconds": round(duration, 3),
            "llm_calls": len(records), "tool_call_attempts": attempts, **total,
            "estimated_api_cost_usd": round(sum(float(r["estimated_api_cost_usd"]) for r in records), 8),
            "long_context_calls": sum(bool(r["long_context_pricing"]) for r in records),
            "provider_reported_tokens": reported,
            "pricing": records[0]["pricing"] if records else None,
        }

    def _turn(self, name: str, kind: str, prompt: str, images: list[Path]) -> SimpleNamespace:
        tool = "submit_poster" if kind == "initial" else "submit_revision"
        turn_dir = self.trace_root / name
        turn_dir.mkdir(parents=True, exist_ok=False)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (turn_dir / "command.json").write_text(json.dumps(
            {"command": self.command, "tool": tool, "images": [str(p) for p in images]}, indent=2))
        events: list[dict[str, Any]] = []
        started = time.monotonic()
        deadline = None if self.timeout is None else started + self.timeout
        try:
            self._command({"type": "prompt", "message": f"/poster-turn {kind}"}, deadline, events)
            mark = len(events)
            self._command({"type": "prompt", "message": prompt, "images": [_image(p) for p in images]},
                          deadline, events)
            settled = any(e.get("type") == "agent_settled" for e in events[mark:])
            while not settled:
                events.append(self._next(deadline))
                settled = events[-1].get("type") == "agent_settled"
        finally:
            with (turn_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
                for event in events:  # streaming deltas are counted, not stored
                    if event.get("type") != "message_update":
                        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.write(json.dumps({"type": "stream_update_count", "count": sum(
                    e.get("type") == "message_update" for e in events)}) + "\n")
        ends = {e.get("toolCallId"): e for e in events if e.get("type") == "tool_execution_end"}
        starts = [e for e in events if e.get("type") == "tool_execution_start"]
        accepted = [e for e in starts if e.get("toolName") == tool
                    and not ends.get(e.get("toolCallId"), {}).get("isError", True)]
        calls = [e["message"] for e in events if e.get("type") == "message_end"
                 and (e.get("message") or {}).get("role") == "assistant"]
        usage = self._usage(calls, time.monotonic() - started, len(starts))
        (turn_dir / "usage.json").write_text(json.dumps(usage, indent=2))
        if not accepted:
            raise DesignerError(f"{name}: no accepted {tool} call")
        return SimpleNamespace(name=name, message=json.dumps(accepted[-1]["args"], ensure_ascii=False),
                               usage=usage)

    def start(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._turn(name, "initial", prompt, images)

    def resume(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._turn(name, "revision", prompt, images)

    def close(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            stats = self._command({"type": "get_session_stats"}, time.monotonic() + 30, []).get("data") or {}
        except DesignerError:
            pass
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._stderr.close()
        (self.trace_root / "session_stats.json").write_text(json.dumps(stats, indent=2))
        return stats


def prune_response(response: dict[str, Any], arm: Arm) -> dict[str, Any]:
    """Drop the fields an arm's schema does not have (replay of full-arm responses)."""
    out = dict(response)
    if not arm.declares_state:
        out.pop("design_state", None)
    if not arm.has(SEM):
        out.pop("critique", None)
    if "design_state" in out:
        state = {k: v for k, v in out["design_state"].items() if owns(arm, STATE_FIELDS.get(k, SEM))}
        state["nodes"] = [{k: v for k, v in node.items() if owns(arm, NODE_FIELDS.get(k, SEM))}
                          for node in state.get("nodes", [])]
        out["design_state"] = state
    return out


class ReplayDesigner:
    backend = "replay of stored Sol main-batch responses"

    def __init__(self, *, source_traces: Path, trace_root: Path, arm: Arm) -> None:
        self.source, self.trace_root, self.arm = Path(source_traces), Path(trace_root), arm
        self.thread_id = "replay"

    def _turn(self, name: str, prompt: str, images: list[Path]) -> SimpleNamespace:
        response = json.loads((self.source / name / "structured_response.json").read_text(encoding="utf-8"))
        turn_dir = self.trace_root / name
        turn_dir.mkdir(parents=True, exist_ok=False)
        (turn_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (turn_dir / "command.json").write_text(json.dumps({"images": [str(p) for p in images]}, indent=2))
        usage = {"backend": self.backend, "thread_id": self.thread_id, "estimated_api_cost_usd": 0.0}
        return SimpleNamespace(name=name, message=json.dumps(prune_response(response, self.arm)), usage=usage)

    def start(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._turn(name, prompt, images)

    def resume(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._turn(name, prompt, images)

    def close(self) -> dict[str, Any]:
        return {}


class AgyDesigner:
    """Thin Antigravity CLI (`agy -p`) designer supporting multi-turn start/resume."""

    backend = "agy cli"

    def __init__(
        self,
        *,
        workspace: Path,
        trace_root: Path,
        schema_dir: Path,
        model: str,
        effort: str = "medium",
        timeout: float | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.trace_root = Path(trace_root)
        self.schema_dir = Path(schema_dir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.agy_bin = os.environ.get("AGY_BIN", "agy")
        self.conversation_id: str | None = None

    def _run_turn(self, name: str, kind: str, prompt: str, images: list[Path], output_schema: Path | None) -> SimpleNamespace:
        turn_dir = self.trace_root / name
        turn_dir.mkdir(parents=True, exist_ok=True)
        schema_path = output_schema or (
            self.schema_dir / ("initial_response.schema.json" if kind == "initial" else "revision_response.schema.json")
        )
        img_lines = ""
        if images:
            listed = "\n".join(f"- {Path(p).resolve()}" for p in images)
            img_lines = (
                "\n\nVisual inputs (inspect each image using `read_file` before answering; do not write files):\n"
                f"{listed}\n"
            )
        full_prompt = prompt + img_lines
        cmd = [
            self.agy_bin,
            "-p",
            full_prompt,
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--json-schema",
            str(schema_path),
            "--disable-slash-commands",
        ]
        if self.conversation_id:
            cmd.extend(["--conversation", self.conversation_id])
        if self.timeout:
            cmd.extend(["--print-timeout", f"{max(1, int(self.timeout))}s"])

        env = {**os.environ, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}
        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=str(self.workspace),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        duration = round(time.monotonic() - t0, 3)
        (turn_dir / "stdout.jsonl").write_text(proc.stdout, encoding="utf-8")
        (turn_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

        final_text = ""
        for raw in proc.stdout.splitlines():
            raw = raw.strip()
            if not raw.startswith("{"):
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            cid = ev.get("conversation_id") or ev.get("session_id")
            if cid and not self.conversation_id:
                self.conversation_id = str(cid)
            if ev.get("type") == "result":
                res = ev.get("result")
                final_text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
            elif ev.get("type") == "assistant":
                msg = ev.get("message") or {}
                for blk in msg.get("content") or []:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        final_text = str(blk.get("text") or "")

        if not final_text:
            raise DesignerError(f"agy turn {name} returned no output (rc={proc.returncode}): {proc.stderr[:400]}")
        (turn_dir / "final_message.json").write_text(final_text, encoding="utf-8")
        usage = {
            "backend": self.backend,
            "model": self.model,
            "effort": self.effort,
            "duration_s": duration,
            "conversation_id": self.conversation_id,
        }
        (turn_dir / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
        return SimpleNamespace(name=name, message=final_text, usage=usage)

    def start(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._run_turn(name, "initial", prompt, images, Path(output_schema) if output_schema else None)

    def resume(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        return self._run_turn(name, "revision", prompt, images, Path(output_schema) if output_schema else None)

    def close(self) -> dict[str, Any]:
        return {"backend": self.backend, "conversation_id": self.conversation_id}


class PersistentCLIDesignerWrapper:
    """Thin wrapper adapting PersistentCodexDesigner / PersistentClaudeDesigner to run.py interface."""

    def __init__(self, inner: Any, schema_dir: Path, backend_name: str) -> None:
        self.inner = inner
        self.schema_dir = Path(schema_dir)
        self.backend = backend_name

    def start(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        schema = Path(output_schema) if output_schema else (self.schema_dir / "initial_response.schema.json")
        res = self.inner.start(name=name, prompt=prompt, images=images, output_schema=schema)
        return SimpleNamespace(name=name, message=res.message, usage=res.usage)

    def resume(self, *, name: str, prompt: str, images: list[Path], output_schema: Any = None) -> SimpleNamespace:
        schema = Path(output_schema) if output_schema else (self.schema_dir / "revision_response.schema.json")
        res = self.inner.resume(name=name, prompt=prompt, images=images, output_schema=schema)
        return SimpleNamespace(name=name, message=res.message, usage=res.usage)

    def close(self) -> dict[str, Any]:
        return {"backend": self.backend, "session_id": getattr(self.inner, "session_id", None)}
