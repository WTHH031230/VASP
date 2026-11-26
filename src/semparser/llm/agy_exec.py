"""Thin Antigravity CLI (`agy -p`) backend for SemParser."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def call_agy(
    prompt: str,
    *,
    model: str = "gemini-4-argon-low",
    effort: str = "medium",
    timeout: float | None = 600.0,
    cwd: Path | None = None,
    trace_dir: Path | None = None,
    call_name: str = "semantic",
    result_dir: Path | None = None,
    output_schema: Path | None = None,
    application: str = "asset_extraction",
) -> tuple[str, dict[str, Any]]:
    agy_bin = os.environ.get("AGY_BIN", "agy")
    cmd = [
        agy_bin,
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--disable-slash-commands",
    ]
    if output_schema is not None:
        cmd.extend(["--json-schema", str(output_schema)])
    if timeout:
        cmd.extend(["--print-timeout", f"{max(1, int(timeout))}s"])

    env = {**os.environ, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vasp_agy_") as tmpdir:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or tmpdir),
            env=env,
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

    raw_text = ""
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            res = ev.get("result")
            raw_text = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
        elif ev.get("type") == "assistant":
            msg = ev.get("message") or {}
            for blk in msg.get("content") or []:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    raw_text = str(blk.get("text") or "")

    if proc.returncode != 0 and not raw_text:
        raise RuntimeError(f"agy exited with {proc.returncode}: {proc.stderr[:500]}")

    usage = {
        "backend": "agy",
        "model": model,
        "effort": effort,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "application": application,
    }
    return raw_text, usage
