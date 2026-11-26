"""Unified thin 6-backend VLM caller for VASP benchmarks.

Supports:
  - codex      : Codex CLI (`codex exec -i ... --output-schema ...`)
  - claude     : Claude Code CLI (`claude -p --json-schema ...`)
  - agy        : Antigravity CLI (`agy -p --output-format stream-json --json-schema ...`)
  - pi-api     : Pi CLI (`pi --print --provider openai-compat ...` with configs/pi/models.json)
  - pi-chatgpt : Pi CLI (`pi --print --provider openai-codex ...` via ChatGPT OAuth login)
  - vertex     : Google Vertex AI REST API (`call.mjs` with GOOGLE_CLOUD_PROJECT / GOOGLE_APPLICATION_CREDENTIALS)

All benchmark prompts, rubrics, and JSON schemas are loaded verbatim from the original
benchmark modules; this layer only dispatches the model call.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PI_BIN = REPO_ROOT / "src" / "viscomposer" / "pi" / "node_modules" / ".bin" / "pi"
DEFAULT_PI_CONFIG = REPO_ROOT / "configs" / "pi"
BACKENDS = ("codex", "claude", "agy", "pi-api", "pi-chatgpt", "vertex")
VERTEX_CALL_MJS = REPO_ROOT / "src" / "benchmarks" / "cross_family" / "call.mjs"
_VERTEX_STRIP_KEYS = {"$schema", "title", "default", "minLength", "pattern"}


def _strip_vertex_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: ({p: _strip_vertex_schema(s) for p, s in v.items()} if k == "properties" else _strip_vertex_schema(v))
            for k, v in node.items()
            if k == "properties" or k not in _VERTEX_STRIP_KEYS
        }
    if isinstance(node, list):
        return [_strip_vertex_schema(x) for x in node]
    return node


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(raw[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"Could not parse JSON object from model response: {raw[:400]}")


def call_vlm_json(
    *,
    backend: str,
    prompt: str,
    images: Sequence[Path],
    schema: dict[str, Any],
    model: str = "gpt-5.6-sol",
    effort: str = "medium",
    provider: str = "openai-compat",
    timeout: float = 300.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}; choose from {BACKENDS}")

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"vasp_eval_{backend}_") as tmp:
        tmp_dir = Path(tmp)
        schema_path = tmp_dir / "schema.json"
        schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

        staged_imgs: list[Path] = []
        for idx, img in enumerate(images):
            src = Path(img).resolve()
            suffix = src.suffix.lower() or ".png"
            dst = tmp_dir / f"image_{idx}{suffix}"
            shutil.copy2(src, dst)
            staged_imgs.append(dst)

        if backend == "codex":
            out_path = tmp_dir / "last_message.json"
            cmd = [
                os.environ.get("CODEX_BIN", "codex"),
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "-m",
                model,
                "-c",
                f"model_reasoning_effort=\"{effort}\"",
                "--output-schema",
                str(schema_path),
                "-o",
                str(out_path),
            ]
            for img in staged_imgs:
                cmd.extend(["-i", str(img)])
            cmd.append(prompt)
            proc = subprocess.run(cmd, cwd=str(tmp_dir), capture_output=True, text=True, timeout=timeout, check=False)
            raw = out_path.read_text(encoding="utf-8") if out_path.exists() else proc.stdout
            if proc.returncode != 0 and not raw.strip():
                raise RuntimeError(f"codex failed ({proc.returncode}): {proc.stderr[:400]}")

        elif backend == "claude":
            img_note = "\n".join(f"- {p}" for p in staged_imgs)
            full_prompt = f"{prompt}\n\nInspect the following image(s) with Read before answering:\n{img_note}" if staged_imgs else prompt
            cmd = [
                os.environ.get("CLAUDE_BIN", "claude"),
                "-p",
                full_prompt,
                "--model",
                model,
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(schema, ensure_ascii=False),
            ]
            proc = subprocess.run(cmd, cwd=str(tmp_dir), capture_output=True, text=True, timeout=timeout, check=False)
            raw = proc.stdout
            if proc.returncode != 0 and not raw.strip():
                raise RuntimeError(f"claude failed ({proc.returncode}): {proc.stderr[:400]}")
            try:
                wrapper = json.loads(raw)
                if isinstance(wrapper, dict) and "structured_output" in wrapper:
                    raw = json.dumps(wrapper["structured_output"], ensure_ascii=False)
                elif isinstance(wrapper, dict) and "result" in wrapper:
                    raw = wrapper["result"] if isinstance(wrapper["result"], str) else json.dumps(wrapper["result"], ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        elif backend == "agy":
            img_note = "\n".join(f"- {p}" for p in staged_imgs)
            full_prompt = (
                f"{prompt}\n\nCall `read_file` on each image below before answering; do not call other tools:\n{img_note}"
                if staged_imgs
                else prompt
            )
            cmd = [
                os.environ.get("AGY_BIN", "agy"),
                "-p",
                full_prompt,
                "--model",
                model,
                "--output-format",
                "stream-json",
                "--json-schema",
                str(schema_path),
                "--disable-slash-commands",
                "--print-timeout",
                f"{max(1, int(timeout))}s",
            ]
            env = {**os.environ, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}
            proc = subprocess.run(cmd, cwd=str(tmp_dir), env=env, capture_output=True, text=True, timeout=timeout, check=False)
            raw = ""
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
                    raw = res if isinstance(res, str) else json.dumps(res, ensure_ascii=False)
                elif ev.get("type") == "assistant":
                    for blk in (ev.get("message") or {}).get("content") or []:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            raw = str(blk.get("text") or "")
            if proc.returncode != 0 and not raw.strip():
                raise RuntimeError(f"agy failed ({proc.returncode}): {proc.stderr[:400]}")

        elif backend == "vertex":
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
            if not project:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required for --backend vertex")
            req_images = []
            for img in staged_imgs:
                mime, _ = mimetypes.guess_type(str(img))
                req_images.append({
                    "mime_type": mime or "image/png",
                    "data_b64": base64.b64encode(img.read_bytes()).decode("ascii"),
                })
            req_path = tmp_dir / "vertex_request.json"
            res_path = tmp_dir / "vertex_response.json"
            req_path.write_text(
                json.dumps(
                    {
                        "system": "EVALUATION-ONLY TASK\nThe attached image is the only evidence. Do not call tools or use outside knowledge.",
                        "user": prompt,
                        "images": req_images,
                        "schema": _strip_vertex_schema(schema),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            thinking = {
                "low": "LOW",
                "medium": "MEDIUM",
                "high": "HIGH",
                "xhigh": "HIGH",
                "max": "HIGH",
            }.get(effort.lower(), "MEDIUM")
            env = {
                **os.environ,
                "GOOGLE_CLOUD_PROJECT": project,
                "GOOGLE_CLOUD_LOCATION": os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                "MODEL": model,
                "THINKING": thinking,
                "PI_NODE_MODULES": str(REPO_ROOT / "src" / "viscomposer" / "pi" / "node_modules"),
            }
            proc = subprocess.run(
                ["node", str(VERTEX_CALL_MJS), str(req_path), str(res_path)],
                cwd=str(tmp_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0 or not res_path.is_file():
                raise RuntimeError(f"vertex failed ({proc.returncode}): {(proc.stderr or proc.stdout)[:400]}")
            raw = str(json.loads(res_path.read_text(encoding="utf-8")).get("text") or "")

        else:  # pi-api or pi-chatgpt
            pi_bin = os.environ.get("PI_POSTER_PI_BIN", str(DEFAULT_PI_BIN))
            agent_dir = os.environ.get("PI_POSTER_AGENT_DIR", str(DEFAULT_PI_CONFIG))
            eff_provider = "openai-codex" if (backend == "pi-chatgpt" and provider == "openai-compat") else provider
            schema_txt = json.dumps(schema, ensure_ascii=False)
            full_prompt = f"{prompt}\n\nReturn ONLY a single JSON object matching this JSON Schema:\n```json\n{schema_txt}\n```"
            cmd = [
                pi_bin,
                "-p",
                "--mode",
                "json",
                "--provider",
                eff_provider,
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
            ]
            for img in staged_imgs:
                cmd.append(f"@{img}")
            cmd.extend(["--", full_prompt])
            env = {**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1"}
            proxy = os.environ.get("PI_POSTER_PROXY", "")
            if proxy:
                for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
                    env.setdefault(k, proxy)
                env.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
                env.setdefault("NODE_USE_ENV_PROXY", "1")
            proc = subprocess.run(cmd, cwd=str(tmp_dir), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False)
            raw = ""
            pi_err = None
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
                            pi_err = str(msg["errorMessage"])
                        parts = [str(b.get("text") or "") for b in (msg.get("content") or []) if isinstance(b, dict) and b.get("type") == "text"]
                        if parts:
                            raw = "\n".join(parts).strip()
            if not raw.strip():
                raise RuntimeError(f"pi failed ({proc.returncode}): {pi_err or proc.stderr[:400]}")

    parsed = _extract_json_object(raw)
    meta = {
        "backend": backend,
        "model": model,
        "effort": effort,
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    return parsed, meta
