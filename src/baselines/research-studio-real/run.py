"""Run ResearchStudio paper2assets -> paper2poster through one Codex Exec agent.

Inputs are one paper PDF and a result directory. Outputs are the ResearchStudio
asset bundle, poster HTML/PDF/PNG/PPTX, Codex traces, and ``adapter_run.json``.
The ResearchStudio checkout is mounted read-only through project skill links;
this adapter never edits ``third_party``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = REPOSITORY_ROOT / "third_party" / "ResearchStudio"
DEFAULT_LEDGER = Path.home() / ".codex" / "usage-ledger" / "ephemeral.jsonl"
REQUIRED_SKILLS = ("paper2assets", "paper2poster")
REQUIRED_DELIVERABLES = ("poster.html", "poster.pdf", "poster.png", "poster.pptx")
CONTROL_FILES = {
    "adapter_run.json",
    "codex_attempts.jsonl",
    "codex_traces",
    "codex_traces_sandbox_failure",
}
TRANSIENT_COPY_NAMES = {"_render"}
REQUIRED_ASSETS = (
    "manifest.json",
    "assets/meta/text.txt",
    "assets/meta/captions.json",
    "assets/meta/figures.json",
    "assets/meta/metadata.json",
    "assets/meta/paper_spec.md",
    "assets/meta/sections.json",
)
PRICING = {
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
        "API-equivalent estimate for gpt-5.6-sol. Codex CLI usage authenticated "
        "through ChatGPT may not be billed as API usage."
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True, help="Input paper PDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default="Qwen3.8-Flash-Next-FP8")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10_800,
        help="Maximum seconds for the complete Codex agent run (default: 10800)",
    )
    parser.add_argument("--usage-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-global-ledger", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing partial ResearchStudio bundle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output directory before starting",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs, skills, executable, and dependencies only",
    )
    return parser


def is_claude_model(model: str | None) -> bool:
    if not model:
        return False
    lowered = model.casefold()
    return lowered.startswith(("claude-", "opus", "sonnet"))


def is_qwen_model(model: str | None) -> bool:
    if not model:
        return False
    lowered = model.casefold()
    return lowered.startswith("qwen") or "qwen3" in lowered


DEFAULT_PI_BIN = Path(
    str(Path(__file__).resolve().parents[2] / "viscomposer" / "pi" / "node_modules" / ".bin" / "pi")
)


def load_qwen_config() -> dict[str, str]:
    import ast

    values: dict[str, str] = {}
    source_path = Path(
        os.environ.get(
            "QWEN_SOURCE_SCRIPT",
        )
    )
    if source_path.is_file():
        for node in ast.parse(source_path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Call
            ):
                continue
            call = node.value
            if getattr(call.func, "attr", None) == "getenv" and len(call.args) >= 2:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                        "base_url",
                        "model_name",
                        "api_key",
                    ):
                        values[target.id] = str(ast.literal_eval(call.args[1]))

    env_file = os.environ.get("QWEN_ENV_FILE", "").strip()
    if env_file and Path(env_file).is_file():
        for raw_line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in ("QWEN_PROXY_URL", "QWEN_BASE_URL") and v:
                values["base_url"] = v
            elif k in ("QWEN_PROXY_KEY", "QWEN_API_KEY") and v:
                values["api_key"] = v
            elif k in ("QWEN_PROXY_MODEL", "QWEN_MODEL") and v:
                values["model_name"] = v

    if os.environ.get("QWEN_BASE_URL"):
        values["base_url"] = os.environ["QWEN_BASE_URL"]
    if os.environ.get("QWEN_API_KEY"):
        values["api_key"] = os.environ["QWEN_API_KEY"]
    if os.environ.get("QWEN_MODEL"):
        values["model_name"] = os.environ["QWEN_MODEL"]

    base_url = values.get("base_url") or "http://192.168.198.191:8015/v1"
    api_key = values.get("api_key") or ""
    model = values.get("model_name") or "Qwen3.8-Flash-Next-FP8"
    if not api_key:
        raise SystemExit(
            "Missing Qwen API key. Set QWEN_ENV_FILE or QWEN_API_KEY."
        )
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _skill_root(upstream: Path) -> Path:
    return upstream / "ResearchStudio-Reel" / "skills"


def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    paper = args.paper.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    if not paper.is_file() or paper.suffix.lower() != ".pdf":
        raise SystemExit(f"--paper must point to an existing PDF: {paper}")
    missing = [
        str(_skill_root(upstream) / name / "SKILL.md")
        for name in REQUIRED_SKILLS
        if not (_skill_root(upstream) / name / "SKILL.md").is_file()
    ]
    if missing:
        raise SystemExit("ResearchStudio checkout is incomplete; missing:\n  " + "\n  ".join(missing))
    if is_qwen_model(args.codex_model):
        load_qwen_config()
        if args.codex_bin == "codex":
            args.codex_bin = str(DEFAULT_PI_BIN) if DEFAULT_PI_BIN.is_file() else "pi"
    elif is_claude_model(args.codex_model) and args.codex_bin == "codex":
        args.codex_bin = "claude"
    if shutil.which(args.codex_bin) is None and not Path(args.codex_bin).is_file():
        raise SystemExit(f"Cannot find CLI executable: {args.codex_bin!r}")
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    if args.force and args.resume:
        raise SystemExit("--force and --resume are mutually exclusive")
    if output in {REPOSITORY_ROOT, upstream, _skill_root(upstream)}:
        raise SystemExit("--output-dir cannot be the repository or ResearchStudio checkout")
    if output.exists() and not output.is_dir():
        raise SystemExit(f"--output-dir exists but is not a directory: {output}")
    if output.exists() and any(output.iterdir()) and not (args.resume or args.force):
        raise SystemExit(
            f"Output directory is not empty: {output}\n"
            "Pass --resume for a partial bundle or --force to replace it."
        )
    return paper, output, upstream


def _dependency_report() -> dict[str, bool]:
    executables = ("pdftotext", "soffice")
    return {name: shutil.which(name) is not None for name in executables}


def _prepare_output(output: Path, force: bool) -> None:
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    """Exclude disposable render trees and links into the private runtime."""
    root = Path(directory)
    return {
        name
        for name in names
        if name in TRANSIENT_COPY_NAMES or (root / name).is_symlink()
    }


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_symlink() or source.name in TRANSIENT_COPY_NAMES:
        return
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=_copy_ignore,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _prepare_staging_bundle(paper: Path, output: Path, bundle: Path) -> Path:
    """Create one agent-private bundle without exposing the shared result root."""
    bundle.mkdir(parents=True)
    for source in output.iterdir():
        if source.name in CONTROL_FILES:
            continue
        _copy_path(source, bundle / source.name)
    local_paper = bundle / "paper.pdf"
    if not local_paper.exists() or local_paper.stat().st_size != paper.stat().st_size:
        shutil.copy2(paper, local_paper)
    return local_paper


def _sync_staging_bundle(bundle: Path, output: Path) -> None:
    """Promote one isolated bundle; never remove a sibling result directory."""
    output.mkdir(parents=True, exist_ok=True)
    for source in bundle.iterdir():
        _copy_path(source, output / source.name)


def _prepare_runtime(
    runtime_root: Path,
    skill_root: Path,
    output: Path,
    model: str | None = None,
) -> Path:
    links = runtime_root / ".codex" / "skills"
    claude_links = runtime_root / ".claude" / "skills"
    links.mkdir(parents=True)
    claude_links.mkdir(parents=True)
    for name in REQUIRED_SKILLS:
        shutil.copytree(skill_root / name, links / name)
        shutil.copytree(skill_root / name, claude_links / name)
    instructions = (
        "# Isolated research-studio-real baseline runtime\n\n"
        "Use only the installed paper2assets and paper2poster skills. Their project "
        "copies are disposable runtime inputs; do not edit them. Write generated "
        f"files only inside {output}. Never inspect, kill, or modify sibling processes.\n"
    )
    (runtime_root / "AGENTS.md").write_text(instructions, encoding="utf-8")
    (runtime_root / "CLAUDE.md").write_text(instructions, encoding="utf-8")
    if is_qwen_model(model):
        cfg = load_qwen_config()
        model_id = model or cfg["model"]
        pi_agent_dir = runtime_root / ".pi_agent"
        pi_agent_dir.mkdir(parents=True, exist_ok=True)
        (runtime_root / "_pi_sessions").mkdir(parents=True, exist_ok=True)
        models_payload = {
            "providers": {
                "qwen": {
                    "baseUrl": cfg["base_url"],
                    "api": "openai-completions",
                    "apiKey": "!printenv QWEN_API_KEY",
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "reasoning": True,
                            "input": ["text", "image"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": 983616 if "3.5" in model_id else 262144,
                            "maxTokens": 65536 if "3.5" in model_id else 131072,
                            "compat": {"supportsDeveloperRole": False},
                        }
                    ],
                }
            }
        }
        (pi_agent_dir / "models.json").write_text(
            json.dumps(models_payload, indent=2), encoding="utf-8"
        )
        (pi_agent_dir / "settings.json").write_text(
            json.dumps({"compaction": {"enabled": True}}, indent=2),
            encoding="utf-8",
        )
    return claude_links


def _prompt(local_paper: Path, output: Path, skill_root: Path) -> str:
    assets_skill = skill_root / "paper2assets"
    poster_skill = skill_root / "paper2poster"
    return f"""Run the complete ResearchStudio paper-to-poster pipeline for exactly one paper.

Input PDF: {local_paper}
Bundle/output directory: {output}
Python executable for all skill scripts: {Path(sys.executable).resolve()}

Use the installed $paper2assets skill first and then the installed $paper2poster skill.
Read each SKILL.md completely and follow its required references, validation loops, and
final deliverables gate. Generate all four final files: poster.html, poster.pdf,
poster.png, and poster.pptx. Reuse valid cached intermediate assets when present.

The skills are read-only upstream inputs:
- paper2assets root: {assets_skill}
- paper2poster root: {poster_skill}
Never edit either skill directory or any other third_party path. Any command example
that uses ~/.claude/skills/paper2assets or ~/.claude/skills/paper2poster must instead
use the corresponding absolute root above. Write only inside the bundle/output
directory. Do not generate blog, video, reel, or unrelated files.

This is one member of a parallel batch. Never run ps, pkill, kill, killall, or inspect
sibling jobs. Do not ask the user questions; make reasonable paper-grounded choices.
If an optional network lookup fails, use the skill's deterministic/local fallback and
continue. Before finishing, run the paper2poster final verification and deliverables
checks. Return a concise completion summary only after the files exist.
"""


def _command(args: argparse.Namespace, runtime_root: Path) -> list[str]:
    if is_qwen_model(args.codex_model):
        thinking = (
            "low"
            if (args.reasoning_effort or "").lower() in ("minimal", "low")
            else "medium"
        )
        return [
            args.codex_bin,
            "-p",
            "--mode",
            "json",
            "--provider",
            "qwen",
            "--model",
            args.codex_model,
            "--thinking",
            thinking,
            "--session-dir",
            str(runtime_root / "_pi_sessions"),
            "--no-extensions",
            "--no-themes",
            "--no-prompt-templates",
            "--offline",
            "--tools",
            "read,bash,edit,write",
            "--skill",
            str(runtime_root / ".claude" / "skills" / "paper2assets"),
            "--skill",
            str(runtime_root / ".claude" / "skills" / "paper2poster"),
            "--system-prompt",
            str(runtime_root / "CLAUDE.md"),
        ]
    if is_claude_model(args.codex_model):
        effort = args.reasoning_effort or "high"
        if effort == "minimal":
            effort = "low"
        return [
            args.codex_bin,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--add-dir",
            str(runtime_root),
            "--model",
            args.codex_model,
            "--effort",
            effort,
        ]
    command = [
        args.codex_bin,
        "exec",
        "--json",
        "--color",
        "never",
        "--cd",
        str(runtime_root),
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        args.codex_model,
        "--config",
        f'model_reasoning_effort="{args.reasoning_effort}"',
    ]
    command.append("-")
    return command


def _parse_events(text: str) -> tuple[str | None, dict[str, int], str | None]:
    final_message: str | None = None
    usage: dict[str, int] = {}
    thread_id: str | None = None
    pi_uncached = 0
    pi_cached = 0
    pi_cache_write = 0
    pi_output = 0
    saw_pi_usage = False
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "session":
            thread_id = event.get("id") or thread_id
        if event.get("type") == "message_end":
            msg = event.get("message") or {}
            if msg.get("role") == "assistant":
                for block in msg.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        txt = str(block.get("text") or "").strip()
                        if txt:
                            final_message = txt
                raw_u = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}
                if raw_u:
                    saw_pi_usage = True
                    pi_uncached += int(raw_u.get("input") or 0)
                    pi_cached += int(raw_u.get("cacheRead") or 0)
                    pi_cache_write += int(raw_u.get("cacheWrite") or 0)
                    pi_output += int(raw_u.get("output") or 0)
        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            final_message = str(item.get("text", ""))
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
        if event.get("type") == "result":
            if isinstance(event.get("result"), str):
                final_message = event["result"]
            thread_id = event.get("session_id") or thread_id
            raw_u = event.get("usage") if isinstance(event.get("usage"), dict) else {}
            uncached = int(raw_u.get("input_tokens") or 0)
            cache_write = int(raw_u.get("cache_creation_input_tokens") or 0)
            cached = int(raw_u.get("cache_read_input_tokens") or 0)
            out_tok = int(raw_u.get("output_tokens") or 0)
            usage = {
                "input_tokens": uncached + cache_write + cached,
                "cached_input_tokens": cached,
                "cache_write_input_tokens": cache_write,
                "uncached_input_tokens": uncached,
                "output_tokens": out_tok,
            }
            if event.get("total_cost_usd") is not None:
                usage["total_cost_usd"] = float(event["total_cost_usd"])
    if saw_pi_usage and not usage:
        usage = {
            "input_tokens": pi_uncached + pi_cache_write + pi_cached,
            "cached_input_tokens": pi_cached,
            "cache_write_input_tokens": pi_cache_write,
            "uncached_input_tokens": pi_uncached,
            "output_tokens": pi_output,
        }
    return final_message, usage, thread_id


def _usage_record(usage: dict[str, int], duration_seconds: float) -> dict[str, Any]:
    input_tokens = int(usage.get("input_tokens", 0))
    cached = min(input_tokens, int(usage.get("cached_input_tokens", 0)))
    non_cached = max(0, input_tokens - cached)
    cache_write = min(non_cached, int(usage.get("cache_write_input_tokens", 0)))
    uncached = non_cached - cache_write
    output_tokens = int(usage.get("output_tokens", 0))
    is_long = input_tokens > PRICING["long_context_threshold_tokens"]
    input_multiplier = PRICING["long_context_input_multiplier"] if is_long else 1.0
    output_multiplier = PRICING["long_context_output_multiplier"] if is_long else 1.0
    if "total_cost_usd" in usage and usage["total_cost_usd"] is not None:
        cost = float(usage["total_cost_usd"])
    else:
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
        "call_count": 1,
        "codex_call_seconds": round(duration_seconds, 3),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "uncached_input_tokens": uncached,
        "output_tokens": output_tokens,
        "long_context_pricing": is_long,
        "estimated_api_cost_usd": round(cost, 8),
        "pricing": dict(PRICING),
    }


def _global_ledger_entry(
    *,
    attempt_id: int,
    completed_at: datetime,
    model: str,
    reasoning_effort: str,
    thread_id: str | None,
    output: Path,
    trace_dir: Path | None,
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact per-call schema used by the other Codex baselines."""

    return {
        "schema_version": 1,
        "recorded_at": completed_at.isoformat(),
        "backend": "codex exec",
        "ephemeral": True,
        "application": "research-studio-real",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "thread_id": thread_id,
        "cwd": None,
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
        "result_dir": str(output),
        "call_id": attempt_id,
        "duration_seconds": usage["codex_call_seconds"],
        "image_count": 0,
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "cache_write_input_tokens": usage["cache_write_input_tokens"],
        "uncached_input_tokens": usage["uncached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "long_context_pricing": usage["long_context_pricing"],
        "estimated_api_cost_usd": usage["estimated_api_cost_usd"],
    }


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as output:
        try:
            import fcntl

            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        output.write(line)
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


def _attempt_count(output: Path) -> int:
    path = output / "codex_attempts.jsonl"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _deliverables(output: Path) -> dict[str, bool]:
    paths = (*REQUIRED_DELIVERABLES, *REQUIRED_ASSETS)
    return {relative: (output / relative).is_file() for relative in paths}


def _attempt_totals(output: Path) -> dict[str, Any]:
    path = output / "codex_attempts.jsonl"
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "attempt_count": len(records),
        "recorded_call_count": sum(
            int((record.get("codex_usage") or {}).get("call_count", 0))
            for record in records
        ),
        "input_tokens": sum(
            int((record.get("codex_usage") or {}).get("input_tokens", 0))
            for record in records
        ),
        "output_tokens": sum(
            int((record.get("codex_usage") or {}).get("output_tokens", 0))
            for record in records
        ),
        "estimated_api_cost_usd": round(
            sum(
                float((record.get("codex_usage") or {}).get("estimated_api_cost_usd", 0))
                for record in records
            ),
            8,
        ),
    }


def _run_codex(
    command: Sequence[str],
    prompt: str,
    events_path: Path,
    stderr_path: Path,
    timeout: int,
    cwd: Path | None = None,
    use_qwen: bool = False,
    bundle: Path | None = None,
) -> tuple[int, float, str | None]:
    started = time.monotonic()
    timed_out = None
    env = os.environ.copy()
    vendor_dir = str(REPOSITORY_ROOT / "vendor")
    env["PYTHONPATH"] = (
        f"{vendor_dir}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else vendor_dir
    )
    py_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{py_bin_dir}:/usr/local/bin:{env.get('PATH', '')}"
    if use_qwen:
        cfg = load_qwen_config()
        env["QWEN_API_KEY"] = cfg["api_key"]
        env["PI_OFFLINE"] = "1"
        if cwd is not None:
            env["PI_CODING_AGENT_DIR"] = str(cwd / ".pi_agent")
        for proxy_key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            env.pop(proxy_key, None)

    def _run_once(
        cmd_args: list[str],
        stdin_text: str | None,
        rem_timeout: float,
        append: bool = False,
    ) -> tuple[int, str | None]:
        mode = "a" if append else "w"
        t_err = None
        with events_path.open(mode, encoding="utf-8") as events, stderr_path.open(
            mode, encoding="utf-8"
        ) as errors:
            process = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=events,
                stderr=errors,
                text=True,
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                start_new_session=True,
            )
            try:
                process.communicate(stdin_text, timeout=max(30.0, rem_timeout))
            except subprocess.TimeoutExpired:
                t_err = f"codex exec timed out after {timeout}s"
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        process.kill()
                    process.wait()
        return process.returncode, t_err

    if use_qwen:
        rc, timed_out = _run_once([*command, "--", prompt], None, float(timeout), append=False)
        cont_idx = 0
        while (
            rc == 0
            and timed_out is None
            and bundle is not None
            and cont_idx < 4
        ):
            missing = [
                name for name, exists in _deliverables(bundle).items() if not exists
            ]
            if not missing:
                break
            elapsed = time.monotonic() - started
            rem = float(timeout) - elapsed
            if rem <= 60.0:
                break
            cont_idx += 1
            cont_prompt = (
                f"Continuation check {cont_idx}: The following required deliverables "
                f"in {bundle} are still missing: {', '.join(missing)}. "
                "Do not stop yet. Immediately execute the remaining steps of the "
                "paper2assets and paper2poster skills using bash/write/edit tools "
                "until all required files (manifest.json, assets/meta/*, poster.html, "
                "poster.pdf, poster.png, poster.pptx) exist in the bundle directory."
            )
            rc, timed_out = _run_once(
                [*command, "-c", "--", cont_prompt],
                None,
                rem,
                append=True,
            )
        return rc, time.monotonic() - started, timed_out

    rc, timed_out = _run_once(list(command), prompt, float(timeout), append=False)
    return rc, time.monotonic() - started, timed_out


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    paper, output, upstream = _validate(args)
    dependencies = _dependency_report()
    if args.check:
        print(
            json.dumps(
                {
                    "paper": str(paper),
                    "output_dir": str(output),
                    "upstream_root": str(upstream),
                    "skill_root": str(_skill_root(upstream)),
                    "codex_bin": shutil.which(args.codex_bin),
                    "dependencies": dependencies,
                },
                indent=2,
            )
        )
        return 0 if all(dependencies.values()) else 1
    missing_dependencies = [name for name, present in dependencies.items() if not present]
    if missing_dependencies:
        raise SystemExit("Missing required executables: " + ", ".join(missing_dependencies))

    _prepare_output(output, args.force)
    if args.resume and (output / "adapter_run.json").is_file():
        try:
            _prev = json.loads((output / "adapter_run.json").read_text(encoding="utf-8"))
        except Exception:
            _prev = {}
        if _prev.get("status") == "success" and all(_deliverables(output).values()):
            print(f"[research-studio-real/codex] already succeeded, skipping: {output}", flush=True)
            return 0
    if args.resume:
        try:
            _ps = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
            _my_pid = os.getpid()
            for _line in _ps.splitlines():
                if "research-studio-real/run.py" in _line and str(output) in _line:
                    _p = int(_line.split()[0])
                    if _p != _my_pid:
                        print(f"[research-studio-real/codex] sibling PID {_p} already running on {output}, skipping", flush=True)
                        return 0
        except Exception:
            pass
    attempt_id = _attempt_count(output) + 1
    trace_root = output / "codex_traces"
    persistent_stem = trace_root / f"call_{attempt_id:04d}"

    started_at = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(
        prefix="research-studio-real-runtime-"
    ) as temporary:
        runtime_root = Path(temporary)
        bundle = runtime_root / "bundle"
        local_paper = _prepare_staging_bundle(paper, output, bundle)
        runtime_skill_root = _prepare_runtime(
            runtime_root, _skill_root(upstream), bundle, model=args.codex_model
        )
        prompt = _prompt(local_paper, bundle, runtime_skill_root)
        runtime_trace_dir = runtime_root / "_trace"
        runtime_trace_dir.mkdir()
        runtime_stem = runtime_trace_dir / f"call_{attempt_id:04d}"
        runtime_stem.with_suffix(".prompt.txt").write_text(
            prompt, encoding="utf-8"
        )
        command = _command(args, runtime_root)
        runtime_stem.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "command": list(command),
                    "runtime_root": str(runtime_root),
                    "staging_bundle": str(bundle),
                    "final_output": str(output),
                    "note": (
                        "Prompt is supplied on stdin; skill copies and the writable "
                        "bundle are isolated per call."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[research-studio-real/codex] attempt={attempt_id} "
            f"model={args.codex_model} effort={args.reasoning_effort} "
            f"output={output}",
            flush=True,
        )
        returncode, duration, timeout_error = _run_codex(
            command,
            prompt,
            runtime_stem.with_suffix(".events.jsonl"),
            runtime_stem.with_suffix(".stderr.txt"),
            args.timeout,
            cwd=runtime_root,
            use_qwen=is_qwen_model(args.codex_model),
            bundle=bundle,
        )
        events = runtime_stem.with_suffix(".events.jsonl").read_text(
            encoding="utf-8"
        )
        final_message, raw_usage, thread_id = _parse_events(events)
        if final_message:
            final_message = final_message.replace(str(bundle), str(output))
        _sync_staging_bundle(bundle, output)
        if not args.no_traces:
            trace_root.mkdir(parents=True, exist_ok=True)
            for trace_file in runtime_trace_dir.iterdir():
                shutil.copy2(trace_file, trace_root / trace_file.name)
    usage = _usage_record(raw_usage, duration)
    files = _deliverables(output)
    success = all(files.values()) and (returncode == 0 or returncode in (143, -15))
    error = timeout_error
    if error is None and returncode != 0:
        error = f"codex exec exited with status {returncode}"
    if error is None and not all(files.values()):
        missing = [name for name, exists in files.items() if not exists]
        error = "missing required outputs: " + ", ".join(missing)

    completed_at = datetime.now(timezone.utc)
    attempt = {
        "schema_version": 1,
        "attempt": attempt_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(duration, 3),
        "status": "success" if success else "failed",
        "returncode": returncode,
        "error": error,
        "model": args.codex_model,
        "reasoning_effort": args.reasoning_effort,
        "thread_id": thread_id,
        "codex_usage": usage,
        "files": files,
    }
    _append_jsonl(output / "codex_attempts.jsonl", attempt)
    if not args.no_global_ledger and raw_usage:
        _append_jsonl(
            args.usage_ledger.expanduser(),
            _global_ledger_entry(
                attempt_id=attempt_id,
                completed_at=completed_at,
                model=args.codex_model,
                reasoning_effort=args.reasoning_effort,
                thread_id=thread_id,
                output=output,
                trace_dir=None if args.no_traces else trace_root,
                usage=usage,
            ),
        )
    manifest = {
        **attempt,
        "paper": str(paper),
        "output_dir": str(output),
        "upstream_root": str(upstream),
        "third_party_read_only": True,
        "codex_final_message": final_message,
        "staging_isolation": True,
        "attempt_totals": _attempt_totals(output),
    }
    _atomic_json(output / "adapter_run.json", manifest)
    if final_message and not args.no_traces:
        persistent_stem.with_suffix(".final.txt").write_text(
            final_message, encoding="utf-8"
        )
    print(
        f"[research-studio-real/codex] status={manifest['status']} "
        f"duration={duration:.1f}s "
        f"cost=${usage['estimated_api_cost_usd']:.4f}",
        flush=True,
    )
    if error:
        print(
            f"[research-studio-real/codex] {error}", file=sys.stderr, flush=True
        )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
