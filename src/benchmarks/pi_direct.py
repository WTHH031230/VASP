"""Clean stateless Pi CLI bridge for OpenAI-shaped multimodal payloads."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PI_BIN = REPO_ROOT / "src" / "viscomposer" / "pi" / "node_modules" / ".bin" / "pi"
DEFAULT_PI_AGENT_DIR = REPO_ROOT / "configs" / "pi"


def call_pi_from_openai_payload(payload: Mapping[str, Any], timeout: float = 480.0) -> dict[str, Any]:
    """Execute one stateless Pi CLI call from an OpenAI-format messages payload."""
    model = str(payload.get("model") or "gpt-5.6-sol")
    effort = str(payload.get("reasoning_effort") or "medium")
    if effort == "minimal":
        effort = "low"
    provider = os.environ.get("PI_POSTER_PROVIDER", "openai-compat")
    pi_bin = os.environ.get("PI_POSTER_PI_BIN", str(DEFAULT_PI_BIN))
    agent_dir = os.environ.get("PI_POSTER_AGENT_DIR", str(DEFAULT_PI_AGENT_DIR))

    system_parts: list[str] = []
    user_parts: list[str] = []
    image_blobs: list[tuple[str, bytes]] = []

    for msg in payload.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        system_parts.append(str(blk.get("text", "")))
        elif role == "user":
            if isinstance(content, str):
                user_parts.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "text":
                        user_parts.append(str(blk.get("text", "")))
                    elif blk.get("type") == "image_url":
                        url = (blk.get("image_url") or {}).get("url", "")
                        if url.startswith("data:image/"):
                            header, _, b64 = url.partition(",")
                            fmt = header.split("/")[1].split(";")[0].lower()
                            ext = ".jpg" if fmt in ("jpeg", "jpg") else f".{fmt}"
                            image_blobs.append((ext, base64.b64decode(b64)))

    resp_fmt = payload.get("response_format")
    if isinstance(resp_fmt, dict) and resp_fmt.get("type") == "json_schema":
        schema_obj = (resp_fmt.get("json_schema") or {}).get("schema")
        if isinstance(schema_obj, dict):
            system_parts.append("OUTPUT JSON SCHEMA:\n" + json.dumps(schema_obj, ensure_ascii=False))

    system_body = "\n\n".join(p for p in system_parts if p).strip()
    system_prompt = (
        "EVALUATION-ONLY TASK\n"
        "The attached image is the only evidence. Do not call tools or use outside knowledge.\n\n"
        + system_body
    ).strip()
    user_prompt = "\n\n".join(p for p in user_parts if p).strip()

    with tempfile.TemporaryDirectory(prefix="vasp_pi_eval_") as tmp:
        work_dir = Path(tmp)
        img_args: list[str] = []
        for idx, (ext, blob) in enumerate(image_blobs):
            if len(blob) > 2_400_000:
                import io
                from PIL import Image
                with Image.open(io.BytesIO(blob)) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    if max(w, h) > 4096:
                        scale = 4096.0 / float(max(w, h))
                        im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=92, optimize=True)
                    blob = buf.getvalue()
                    ext = ".jpg"
            img_path = work_dir / f"image_{idx:02d}{ext}"
            img_path.write_bytes(blob)
            img_args.append(f"@{img_path.resolve()}")

        cmd = [
            pi_bin, "-p", "--mode", "json",
            "--provider", provider, "--model", model, "--thinking", effort,
            "--no-builtin-tools", "--no-extensions", "--no-mcp",
            "--no-skills", "--no-prompt-templates", "--no-themes",
            "--no-context-files", "--no-session",
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        cmd.extend(img_args)
        cmd.extend(["--", user_prompt])

        env = {**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1"}
        proxy = os.environ.get("PI_POSTER_PROXY", "")
        if proxy:
            for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
                env.setdefault(k, proxy)
            env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
            env.setdefault("NODE_USE_ENV_PROXY", "1")

        proc = subprocess.run(cmd, cwd=str(work_dir), env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout, check=False)

    text = ""
    usage: dict[str, Any] = {}
    err_msg: str | None = None
    for line in proc.stdout.splitlines():
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
                    err_msg = str(msg["errorMessage"])
                parts = [str(b.get("text") or "") for b in (msg.get("content") or [])
                         if isinstance(b, dict) and b.get("type") == "text"]
                if parts:
                    text = "\n".join(parts).strip()
                if isinstance(msg.get("usage"), dict):
                    usage = msg["usage"]
    if not text:
        raise RuntimeError(f"pi_direct failed (rc={proc.returncode}): {err_msg or proc.stderr[:400]}")
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": int(usage.get("input", 0) or 0),
            "completion_tokens": int(usage.get("output", 0) or 0),
            "total_tokens": int(usage.get("totalTokens", 0) or 0),
        },
        "model": model,
    }
