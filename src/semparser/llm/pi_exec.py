"""Thin Pi CLI (`pi -p --mode json`) backend for SemParser (supports pi-api and pi-chatgpt)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PI_BIN = REPO_ROOT / "src" / "viscomposer" / "pi" / "node_modules" / ".bin" / "pi"
DEFAULT_PI_CONFIG = REPO_ROOT / "configs" / "pi"


def _extract_pi_jsonl_text(stdout: str) -> tuple[str, str | None]:
    text = ""
    err: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") == "assistant":
                if msg.get("stopReason") == "error" and msg.get("errorMessage"):
                    err = str(msg["errorMessage"])
                parts = [
                    str(b.get("text") or "")
                    for b in (msg.get("content") or [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                if parts:
                    text = "\n".join(parts).strip()
    return text, err


def call_pi(
    prompt: str,
    *,
    model: str = "gpt-5.6-sol",
    effort: str = "medium",
    provider: str = "openai-compat",
    chatgpt_login: bool = False,
    timeout: float | None = 600.0,
    cwd: Path | None = None,
    trace_dir: Path | None = None,
    call_name: str = "semantic",
    result_dir: Path | None = None,
    output_schema: Path | None = None,
    application: str = "asset_extraction",
) -> tuple[str, dict[str, Any]]:
    pi_bin = os.environ.get("PI_POSTER_PI_BIN", str(DEFAULT_PI_BIN))
    effective_provider = "openai-codex" if chatgpt_login and provider == "openai-compat" else provider
    agent_dir = os.environ.get("PI_POSTER_AGENT_DIR", str(DEFAULT_PI_CONFIG))

    full_prompt = prompt
    if output_schema is not None:
        schema_txt = Path(output_schema).read_text(encoding="utf-8")
        full_prompt = (
            f"{prompt}\n\nReturn ONLY a single valid JSON object conforming strictly to this JSON Schema:\n"
            f"```json\n{schema_txt}\n```"
        )

    cmd = [
        pi_bin,
        "-p",
        "--mode",
        "json",
        "--provider",
        effective_provider,
        "--model",
        model,
        "--thinking",
        effort,
        "--no-builtin-tools",
        "--no-extensions",
        "--no-mcp",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
        "--no-session",
        "--",
        full_prompt,
    ]
    env = {**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1"}
    proxy = os.environ.get("PI_POSTER_PROXY", "")
    if proxy:
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            env.setdefault(k, proxy)
        env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
        env.setdefault("NODE_USE_ENV_PROXY", "1")

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vasp_pi_sem_") as tmpdir:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or tmpdir),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    elapsed = round(time.monotonic() - t0, 3)

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{call_name}.stdout.jsonl").write_text(proc.stdout, encoding="utf-8")
        (trace_dir / f"{call_name}.stderr.log").write_text(proc.stderr, encoding="utf-8")

    raw_text, pi_err = _extract_pi_jsonl_text(proc.stdout)
    if not raw_text:
        raise RuntimeError(f"pi failed (rc={proc.returncode}): {pi_err or proc.stderr[:500]}")

    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    usage = {
        "backend": "pi-chatgpt" if chatgpt_login else "pi-api",
        "provider": effective_provider,
        "model": model,
        "effort": effort,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "application": application,
    }
    return raw_text, usage
