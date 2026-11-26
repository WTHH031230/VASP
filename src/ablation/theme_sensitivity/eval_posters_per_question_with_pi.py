#!/usr/bin/env python3
"""Evaluate posters with one stateless Pi CLI invocation per atomic VisionBench question.

Uses Pi logged in via Codex/ChatGPT OAuth (`PI_CODING_AGENT_DIR`), with the exact
startup.lock serialization, proxy configuration, and priority service-tier extension
from the Pi CLI runtime.

Rubrics, prompts, question stimuli (`DIST01` distance view), response validation,
report layout (`schema_version: "2.0"`), and aggregation are imported unchanged from
the pinned `vision_bench` package next to this script.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
VB_DIR = REPO_ROOT / "src" / "benchmarks" / "vision_bench"
if str(VB_DIR) not in sys.path:
    sys.path.insert(0, str(VB_DIR))

from evaluator import (  # type: ignore  # noqa: E402
    VISUAL_EVALUATION_DIMENSIONS,
    aggregate_results,
    build_system_prompt,
    build_user_prompt,
    extract_json_object,
    load_rubrics,
    validate_dimension_response,
)
from eval_poster import (  # type: ignore  # noqa: E402
    select_questions,
    write_json,
    write_text,
)
from poster_stimuli import prepare_question_poster  # type: ignore  # noqa: E402


def poster_key(poster: Path) -> str:
    return poster.stem


def completed_ids(report: Mapping[str, Any]) -> set[str]:
    done: set[str] = set()
    for dim_block in (report.get("dimensions") or {}).values():
        if not isinstance(dim_block, Mapping):
            continue
        for item in dim_block.get("results") or []:
            if isinstance(item, Mapping) and item.get("id"):
                done.add(str(item["id"]))
    return done


def normalize_report(report: dict[str, Any], rubrics: Mapping[str, Any], dimensions: list[str]) -> dict[str, Any]:
    dims_out = report.setdefault("dimensions", {})
    ordered_rubrics = {d: rubrics[d] for d in dimensions if d in rubrics}
    report["summary"] = aggregate_results(dims_out, ordered_rubrics)
    return report


DEFAULT_PI_BIN = REPO_ROOT / "src" / "viscomposer" / "pi" / "node_modules" / ".bin" / "pi"
DEFAULT_PI_AGENT_DIR = REPO_ROOT / "configs" / "pi"
DEFAULT_PI_CWD = REPO_ROOT / "src" / "viscomposer" / "pi" / "cwd"
DEFAULT_REQUEST_EXT: Path | None = None
DEFAULT_PROXY = ""

PROTOCOL_NAME = "p2p_atomic_judgement_pi_oauth_per_question_v1"

EVAL_PREAMBLE = (
    "EVALUATION-ONLY TASK\n"
    "The attached image is the only evidence. Do not call tools or use outside "
    "knowledge. Do not discuss how to improve the poster. Complete only the "
    "structured visual judgement requested below and output only one valid JSON object."
)

FATAL_STDERR_MARKERS = (
    "unknown option",
    "unrecognized option",
    "authentication required",
    "authentication failed",
    "invalid_state",
    "not logged in",
    "model not found",
)


class FatalPiError(RuntimeError):
    """Non-retryable configuration, CLI option, or authentication error."""


class TransportError(RuntimeError):
    """Retryable transport/provider/process error during a Pi invocation."""


class InvalidOutput(ValueError):
    """The model answered, but the output failed JSON extraction or schema validation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poster", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", default="openai", help="Pi provider name (default: openai)")
    parser.add_argument("--model", default="gpt-5.6-sol", help="Model id (default: gpt-5.6-sol)")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--service-tier", choices=("priority", "default", ""), default="priority")
    parser.add_argument("--pi-bin", type=Path, default=DEFAULT_PI_BIN)
    parser.add_argument("--pi-agent-dir", type=Path, default=DEFAULT_PI_AGENT_DIR)
    parser.add_argument("--pi-cwd", type=Path, default=DEFAULT_PI_CWD)
    parser.add_argument("--request-extension", type=Path, default=DEFAULT_REQUEST_EXT)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-transport-retries", type=int, default=5)
    parser.add_argument("--invalid-output-retries", type=int, default=2)
    parser.add_argument("--rubric-dir", default=str(SNAPSHOT_DIR / "rubrics"))
    parser.add_argument(
        "--output-schema",
        default=str(SNAPSHOT_DIR / "schemas" / "single_question_response.schema.json"),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        choices=VISUAL_EVALUATION_DIMENSIONS,
        default=list(VISUAL_EVALUATION_DIMENSIONS),
    )
    parser.add_argument("--question-ids", nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_clean_image(source: Path, question_dir: Path, question_id: str) -> tuple[Path, str]:
    """Expose image through a delimiter- and space-free path for Pi @file attachment."""
    token = hashlib.sha256(str(question_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    staging_dir = question_dir.parent.parent / "_pi_image_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".png"
    staged = staging_dir / f"{token}_{question_id}{suffix}"
    if staged.is_file():
        staged.unlink()
    try:
        os.link(source, staged)
        return staged, "hardlink"
    except OSError:
        shutil.copy2(source, staged)
        return staged, "copy"


def build_pi_env(args: argparse.Namespace, question_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PI_CODING_AGENT_DIR"] = str(args.pi_agent_dir.resolve())
    env["PI_OFFLINE"] = "1"
    env["DPCHEM_TASK_DIR"] = str(question_dir.resolve())
    env["DPCHEM_PI_SERVICE_TIER"] = args.service_tier
    if args.proxy:
        env.update(
            HTTPS_PROXY=args.proxy,
            HTTP_PROXY=args.proxy,
            https_proxy=args.proxy,
            http_proxy=args.proxy,
            NO_PROXY="localhost,127.0.0.1,::1",
            no_proxy="localhost,127.0.0.1,::1",
            NODE_USE_ENV_PROXY="1",
        )
    return env


def build_pi_command(
    *,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    staged_image: Path,
    session_id: str,
) -> list[str]:
    cmd = [
        str(args.pi_bin.resolve()),
        "-p",
        "--mode",
        "json",
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--thinking",
        args.reasoning_effort,
        "--no-builtin-tools",
        "--no-extensions",
        "--no-mcp",
    ]
    if args.service_tier and args.request_extension.is_file():
        cmd.extend(["-e", str(args.request_extension.resolve())])
    cmd.extend(
        [
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-session",
            "--session-id",
            session_id,
            "--system-prompt",
            system_prompt,
            f"@{staged_image.resolve()}",
            "--",
            user_prompt,
        ]
    )
    return cmd


def run_pi_serialized(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> int:
    """Serialize Pi settings/OAuth startup via startup.lock, then allow concurrent model calls."""
    deadline = time.monotonic() + timeout
    lock_path = Path(env["PI_CODING_AGENT_DIR"]) / "startup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen[bytes] | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("a", encoding="utf-8") as err_f:
            with lock_path.open("a", encoding="utf-8") as lock_f:
                while True:
                    try:
                        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(cmd, timeout)
                        time.sleep(0.02)
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(cwd),
                        env=env,
                        stdout=out_f,
                        stderr=err_f,
                        stdin=subprocess.DEVNULL,
                    )
                    with stdout_path.open("r", encoding="utf-8", errors="replace") as events_f:
                        pending = ""
                        while proc.poll() is None:
                            pending += events_f.readline()
                            if pending.endswith("\n"):
                                try:
                                    if json.loads(pending).get("type") == "session":
                                        break
                                except (ValueError, AttributeError):
                                    pass
                                pending = ""
                            if time.monotonic() >= deadline:
                                raise subprocess.TimeoutExpired(cmd, timeout)
                            time.sleep(0.02)
                finally:
                    fcntl.flock(lock_f, fcntl.LOCK_UN)

            while True:
                code = proc.poll()
                if code is not None:
                    return code
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                time.sleep(0.05)
    except BaseException:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        raise


def parse_pi_session_events(
    session_path: Path,
    stderr_path: Path,
    rubric: Mapping[str, Any],
    question: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Extract the final assistant message, usage, and validated judgement from a Pi JSONL stream."""
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
    low_stderr = stderr_text.lower()
    for marker in FATAL_STDERR_MARKERS:
        if marker in low_stderr:
            safe_err = re.sub(r"[A-Za-z0-9_\-]{28,}", "<redacted>", stderr_text)[:400]
            raise FatalPiError(f"Fatal Pi CLI/auth error ({marker}): {safe_err}")

    assistant_messages: list[dict[str, Any]] = []
    usage_totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0}
    raw_lines = session_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "message_end":
            msg = ev.get("message") or {}
            if msg.get("role") == "assistant":
                assistant_messages.append(msg)
                u = msg.get("usage") or {}
                usage_totals["input"] += int(u.get("input") or 0)
                usage_totals["output"] += int(u.get("output") or 0)
                usage_totals["cache_read"] += int(u.get("cacheRead") or 0)
                usage_totals["cache_write"] += int(u.get("cacheWrite") or 0)
                usage_totals["reasoning"] += int(u.get("reasoning") or 0)

    if not assistant_messages:
        safe_err = re.sub(r"[A-Za-z0-9_\-]{28,}", "<redacted>", stderr_text)[:300]
        raise TransportError(f"No assistant message_end in {session_path} (stderr={safe_err})")

    last_msg = assistant_messages[-1]
    stop_reason = last_msg.get("stopReason")
    error_msg = last_msg.get("errorMessage") or ""
    if stop_reason == "error" or error_msg:
        safe_err = re.sub(r"[A-Za-z0-9_\-]{28,}", "<redacted>", str(error_msg))[:400]
        for marker in FATAL_STDERR_MARKERS:
            if marker in safe_err.lower():
                raise FatalPiError(f"Fatal Pi provider error ({marker}): {safe_err}")
        raise TransportError(f"Pi assistant stopReason={stop_reason!r} error={safe_err}")

    if stop_reason not in (None, "stop", "end_turn"):
        raise InvalidOutput(f"Unexpected stopReason={stop_reason!r}")

    content = last_msg.get("content")
    try:
        parsed = extract_json_object(content)
        normalized = validate_dimension_response(parsed, rubric, [question])
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidOutput(str(exc)) from exc

    if isinstance(content, list):
        text_out = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
    elif isinstance(content, str):
        text_out = content.strip()
    else:
        text_out = json.dumps(parsed, ensure_ascii=False)

    meta = {
        "response_id": last_msg.get("responseId"),
        "model_returned": last_msg.get("model"),
        "provider_returned": last_msg.get("provider"),
        "api_returned": last_msg.get("api"),
        "stop_reason": stop_reason,
        "raw_stop_reason": last_msg.get("rawStopReason"),
        "usage": usage_totals,
        "assistant_turns": len(assistant_messages),
    }
    return normalized, text_out, meta


def run_one(
    *,
    poster: Path,
    dimension: str,
    rubric: Mapping[str, Any],
    question: Mapping[str, Any],
    call_index: int,
    run_dir: Path,
    args: argparse.Namespace,
    abort_event: threading.Event,
    abort_reason: list[str],
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    if abort_event.is_set():
        raise FatalPiError(f"Aborted due to earlier fatal error: {abort_reason[0] if abort_reason else 'fatal'}")

    question_id = str(question["id"])
    question_dir = run_dir / f"{call_index:02d}_{question_id}"
    question_dir.mkdir(parents=True, exist_ok=True)

    prepared_poster, input_transform = prepare_question_poster(poster, question, question_dir)
    staged_image, staging_mode = stage_clean_image(prepared_poster, question_dir, question_id)

    system_prompt = EVAL_PREAMBLE + "\n\n" + build_system_prompt(rubric)
    user_prompt = build_user_prompt(rubric, [question])

    write_text(question_dir / "system_prompt.txt", system_prompt)
    write_text(question_dir / "user_prompt.txt", user_prompt)

    cache_session_id = re.sub(r"[^A-Za-z0-9._-]", "_", f"vb-pi-{dimension}-{question_id}")
    env = build_pi_env(args, question_dir)
    cmd = build_pi_command(
        args=args,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        staged_image=staged_image,
        session_id=cache_session_id,
    )

    base_record = {
        "question_id": question_id,
        "dimension": dimension,
        "call_index": call_index,
        "independent_request": True,
        "stateless": True,
        "provider_override": args.provider,
        "model_override": args.model,
        "reasoning_effort_override": args.reasoning_effort,
        "service_tier": args.service_tier,
        "evaluation_image_path": str(prepared_poster.resolve()),
        "evaluation_image_sha256": sha256_bytes(prepared_poster.read_bytes()),
        "staged_image_path": str(staged_image.resolve()),
        "image_staging_mode": staging_mode,
        "input_transform": input_transform,
    }
    if args.dry_run:
        return dimension, question_id, {}, {**base_record, "status": "planned"}

    started_at = utc_now()
    attempts: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []

    for ask_index in range(args.invalid_output_retries + 1):
        decoded_meta: dict[str, Any] | None = None
        normalized: dict[str, Any] | None = None
        text_out: str = ""

        for transport_try in range(args.max_transport_retries + 1):
            if abort_event.is_set():
                raise FatalPiError(f"Aborted: {abort_reason[0] if abort_reason else 'fatal'}")
            t0 = time.monotonic()
            session_path = question_dir / f"session_ask{ask_index}_try{transport_try}.jsonl"
            stderr_path = question_dir / f"stderr_ask{ask_index}_try{transport_try}.log"
            attempt_rec: dict[str, Any] = {
                "ask": ask_index,
                "transport_try": transport_try,
                "started_at": utc_now(),
                "session_path": str(session_path.resolve()),
            }
            attempts.append(attempt_rec)
            try:
                exit_code = run_pi_serialized(
                    cmd,
                    cwd=args.pi_cwd,
                    env=env,
                    stdout_path=session_path,
                    stderr_path=stderr_path,
                    timeout=args.timeout,
                )
                attempt_rec["exit_code"] = exit_code
                attempt_rec["latency_s"] = round(time.monotonic() - t0, 3)
                normalized, text_out, decoded_meta = parse_pi_session_events(
                    session_path, stderr_path, rubric, question
                )
                if exit_code != 0:
                    raise TransportError(f"Pi exited with code {exit_code}")
                attempt_rec["outcome"] = "ok"
                break
            except FatalPiError as exc:
                attempt_rec["latency_s"] = round(time.monotonic() - t0, 3)
                attempt_rec["outcome"] = "fatal_error"
                attempt_rec["error"] = str(exc)
                if not abort_reason:
                    abort_reason.append(str(exc))
                abort_event.set()
                raise
            except subprocess.TimeoutExpired:
                attempt_rec["latency_s"] = round(time.monotonic() - t0, 3)
                attempt_rec["outcome"] = "timeout"
                attempt_rec["error"] = f"Timeout after {args.timeout}s"
            except TransportError as exc:
                attempt_rec["latency_s"] = round(time.monotonic() - t0, 3)
                attempt_rec["outcome"] = "transport_error"
                attempt_rec["error"] = str(exc)
            except InvalidOutput as exc:
                attempt_rec["latency_s"] = round(time.monotonic() - t0, 3)
                attempt_rec["outcome"] = "invalid_output"
                attempt_rec["error"] = str(exc)
                invalid_reasons.append(str(exc))
                break

            if transport_try == args.max_transport_retries:
                raise RuntimeError(
                    f"Transport retries exhausted after {args.max_transport_retries + 1} tries "
                    f"for {poster.stem} {question_id}: {attempts[-1].get('error')}"
                )
            delay = min(90.0, 4.0 * (2 ** transport_try)) + random.uniform(0.0, 2.0)
            time.sleep(delay)

        if normalized is not None and decoded_meta is not None:
            last_message_path = question_dir / "last_message.json"
            write_text(last_message_path, text_out)
            result = normalized["results"][0]
            result["independent_question_session"] = True
            run_record = {
                **base_record,
                "status": "completed",
                "automatic_retry": len(attempts) > 1,
                "invalid_output_reasks": ask_index,
                "invalid_output_reasons": invalid_reasons,
                "pi_attempts": attempts,
                "created_at": started_at,
                "completed_at": utc_now(),
                **decoded_meta,
                "last_message_path": str(last_message_path.resolve()),
                "dimension_summary": normalized.get("dimension_summary", ""),
            }
            return dimension, question_id, result, run_record

    raise RuntimeError(
        f"Invalid output after {args.invalid_output_retries + 1} asks for {poster.stem} {question_id}: "
        f"{invalid_reasons}"
    )


def new_report(*, poster: Path, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "created_at": utc_now(),
        "protocol": {
            "name": PROTOCOL_NAME,
            "evaluation_system": "visual_evaluation",
            "evaluation_system_title": "Visual Presentation Evaluation",
            "backend": "pi_coding_agent_oauth",
            "provider": args.provider,
            "service_tier": args.service_tier,
            "message_layout": (
                "system=EVAL_PREAMBLE+evaluator.build_system_prompt(rubric); "
                "user=[@stimulus_image, evaluator.build_user_prompt(rubric,[question])]"
            ),
            "independent_dialogues": True,
            "independent_request_per_question": True,
            "one_dialogue_per_question": True,
            "stateless_requests": True,
            "question_specific_stimuli": True,
            "verdicts": ["PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE"],
        },
        "model_override": args.model,
        "reasoning_effort_override": args.reasoning_effort,
        "poster_images": [str(poster.resolve())],
        "poster_sha256": sha256_bytes(poster.read_bytes()),
        "requested_dimensions": list(args.dimensions),
        "requested_question_ids": list(args.question_ids or []),
        "run_dir": str(run_dir.resolve()),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "dimensions": {},
        "aggregate": {},
        "pi_runs": {},
        "errors": {},
        "status": "running",
    }


def relabel_summaries(report: dict[str, Any]) -> None:
    for block in report.get("dimensions", {}).values():
        block["dimension_summary"] = (
            "Each atomic judgement was produced by a separate stateless Pi "
            "process; consult per-question evidence."
        )


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if not args.dry_run:
        if not args.pi_bin.is_file():
            raise FileNotFoundError(f"Pi binary not found: {args.pi_bin}")
        if not args.pi_agent_dir.is_dir():
            raise FileNotFoundError(f"Pi agent dir not found: {args.pi_agent_dir}")
        args.pi_cwd.mkdir(parents=True, exist_ok=True)

    posters = [Path(path).resolve() for path in args.poster]
    for poster in posters:
        if not poster.is_file():
            raise FileNotFoundError(poster)
    keys = [poster_key(poster) for poster in posters]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate poster stems")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rubrics = load_rubrics(args.rubric_dir, dimensions=args.dimensions)
    rubrics, args.dimensions = select_questions(rubrics, list(args.dimensions), args.question_ids)

    reports: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, Path] = {}
    tasks: list[tuple[Path, str, Mapping[str, Any], Mapping[str, Any], int]] = []

    for poster in posters:
        key = poster_key(poster)
        output_path = output_dir / f"{key}.json"
        run_dir = output_dir / f"{key}_pi_runs"
        current_sha = sha256_bytes(poster.read_bytes())
        if args.resume and output_path.is_file():
            report = json.loads(output_path.read_text(encoding="utf-8"))
            proto = report.get("protocol") or {}
            existing = (
                report.get("model_override"),
                report.get("reasoning_effort_override"),
                proto.get("name"),
                proto.get("provider"),
                report.get("poster_sha256"),
            )
            wanted = (
                args.model,
                args.reasoning_effort,
                PROTOCOL_NAME,
                args.provider,
                current_sha,
            )
            if existing != wanted:
                raise ValueError(
                    f"--resume settings {wanted} differ from existing report {output_path}: {existing}"
                )
            report.setdefault("pi_runs", {})
            report.setdefault("errors", {})
        else:
            report = new_report(poster=poster, args=args, run_dir=run_dir)
        reports[key] = report
        output_paths[key] = output_path
        done = completed_ids(report)
        call_index = 0
        for dimension in args.dimensions:
            rubric = rubrics[dimension]
            for question in rubric["questions"]:
                call_index += 1
                if question["id"] in done:
                    continue
                tasks.append((poster, dimension, rubric, question, call_index))
        normalize_report(report, rubrics, list(args.dimensions))
        relabel_summaries(report)
        write_json(output_path, report)

    total = len(tasks)
    print(f"planned {total} per-question Pi calls for {len(posters)} posters", flush=True)
    abort_event = threading.Event()
    abort_reason: list[str] = []

    if total > 0:
        future_context: dict[Future[Any], tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for poster, dimension, rubric, question, call_index in tasks:
                key = poster_key(poster)
                future = executor.submit(
                    run_one,
                    poster=poster,
                    dimension=dimension,
                    rubric=rubric,
                    question=question,
                    call_index=call_index,
                    run_dir=Path(reports[key]["run_dir"]),
                    args=args,
                    abort_event=abort_event,
                    abort_reason=abort_reason,
                )
                future_context[future] = (key, str(question["id"]))

            finished = 0
            for future in as_completed(future_context):
                key, expected_id = future_context[future]
                report = reports[key]
                try:
                    dimension, question_id, result, run_record = future.result()
                    if question_id != expected_id:
                        raise RuntimeError(f"Expected {expected_id}, worker returned {question_id}")
                    if not args.dry_run:
                        block = report["dimensions"][dimension]
                        block["results"] = [
                            item for item in block["results"] if item["id"] != question_id
                        ]
                        block["results"].append(result)
                        report["errors"].pop(question_id, None)
                    report["pi_runs"][question_id] = run_record
                except Exception as exc:
                    report["errors"][expected_id] = {"type": type(exc).__name__, "message": str(exc)}
                normalize_report(report, rubrics, list(args.dimensions))
                relabel_summaries(report)
                report["status"] = "dry_run" if args.dry_run else "running"
                write_json(output_paths[key], report)
                finished += 1
                status = report["pi_runs"].get(expected_id, {}).get("status", "failed")
                if expected_id in report["errors"]:
                    status = "failed: " + report["errors"][expected_id]["message"][:160]
                print(f"[{finished}/{total}] {key} {expected_id}: {status}", flush=True)

    incomplete = 0
    for key, report in reports.items():
        normalize_report(report, rubrics, list(args.dimensions))
        relabel_summaries(report)
        expected_count = sum(len(rubrics[d]["questions"]) for d in args.dimensions)
        result_count = sum(len(report["dimensions"][d]["results"]) for d in args.dimensions)
        complete = result_count == expected_count and not report["errors"]
        report["status"] = "dry_run" if args.dry_run else "completed" if complete else "partial"
        write_json(output_paths[key], report)
        incomplete += 0 if complete else 1
    print(
        f"reports: {len(reports) - incomplete} complete, {incomplete} incomplete",
        flush=True,
    )
    if abort_event.is_set():
        print(f"FATAL: {abort_reason[0] if abort_reason else 'aborted'}", file=sys.stderr, flush=True)
        return 2
    return 1 if incomplete and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
