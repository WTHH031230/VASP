"""One semantic call per attempt, strict validation, and bounded auditable retries.

Only absent output or invalid machine format can trigger a second call. Accepted
responses are reused after downstream failures; coverage never triggers a retry.
This module imports only active pipeline code, never an experiment or backup.
"""

import fcntl
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from family_interface import normalize_families
from llm.codex_exec import _render_prompt
from semantic_prompt import build_prompt
from semantic_schema import FormatError, build_schema, parse_response, to_legacy
from semantic_validation import build_spec, validate_semantic

PRODUCTS = (
    "segments2.json", "structured_elements.json", "metadata_candidate.json",
    "source_lines.json", "outer_boundary_candidates.json", "page_outer_boundaries.json",
    "semantic_wire.json", "semantic_legacy.json", "semantic_corrections.json",
    "semantic_quality.json", "semantic_usage.json", "semantic_complete.json",
    "semantic_request.json", "semantic_attempts.json",
)


def read(path):
    return Path(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read(path))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def archive_semantic(output):
    """An explicit --force preserves previous attempts instead of overwriting them."""
    paths = [output / name for name in (*PRODUCTS, "attempts", "codex_trace", "claude_trace")
             if (output / name).exists()]
    if not paths:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    archive = output / "semantic_history" / stamp
    archive.mkdir(parents=True)
    for path in paths:
        shutil.move(str(path), archive / path.name)
    print(f"Preserved previous semantic outputs: {archive}", flush=True)


def attempt_summary(output):
    rows = []
    for path in sorted((output / "attempts").glob("semantic-*")):
        usage = read_json(path / "usage.json") if (path / "usage.json").exists() else None
        rows.append({
            "attempt": path.name,
            "accepted": (path / "validation.json").is_file(),
            "format_failure": (path / "validation_error.json").is_file(),
            "call_failure": (path / "failure.json").is_file(),
            "usage": usage if isinstance(usage, dict) else None,
            "path": str(path.resolve()),
        })
    summary = {"call_count": len(rows), "calls": rows,
               "format_failures": sum(r["format_failure"] for r in rows),
               "call_failures": sum(r["call_failure"] for r in rows),
               "calls_without_usage": sum(r["usage"] is None for r in rows),
               "first_attempt_accepted": bool(rows and rows[0]["accepted"])}
    write_json(output / "semantic_attempts.json", summary)
    return summary


def transport_error(attempt, usage):
    parts = [str(usage)]
    events = attempt / "trace/semantic.events.jsonl"
    if events.exists():
        for line in read(events).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in ("error", "turn.failed"):
                parts.append(str(event.get("message", event.get("error", ""))))
    return "\n".join(parts)


def configuration_error(error):
    return any(token in error.lower() for token in (
        "invalid_json_schema", "401", "unauthorized", "model_not_found",
        "filenotfounderror", "missing output schema", "permission denied",
    ))


def extract_semantic(source, candidates, pages, numbered, boundaries, output, args, helpers):
    """Publish legacy-compatible assets only after a complete response validates."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "semantic.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _extract_locked(source, candidates, pages, numbered, boundaries,
                               output, args, helpers)


def _extract_locked(source, candidates, pages, numbered, boundaries, output, args, helpers):
    if not source:
        raise ValueError("no numbered content lines; no model call was made")
    if not 1 <= args.max_attempts <= 2 or args.timeout <= 0:
        raise ValueError("max_attempts must be 1..2 and timeout must be positive")
    base_task = helpers.PROMPT.split("\nLines:\n__LINES__", 1)[0]
    spec = build_spec(base_task)
    schema = build_schema(len(source), spec["roles"])
    prompt = build_prompt(base_task, numbered, boundaries)
    request = {
        "schema_version": "dp-p2p.single-pass-semantic.v1",
        "model": args.model, "effort": args.effort,
        "timeout": args.timeout, "max_attempts": args.max_attempts,
        "source_sha256": digest(json.dumps(source, ensure_ascii=False, sort_keys=True)),
        "candidates_sha256": digest(json.dumps(candidates, ensure_ascii=False, sort_keys=True)),
        "prompt_sha256": digest(prompt), "schema_sha256": digest(json.dumps(schema, sort_keys=True)),
        "wrapper_sha256": digest(_render_prompt("")),
    }
    if args.force:
        archive_semantic(output)
    request_path = output / "semantic_request.json"
    attempts_root = output / "attempts"
    if request_path.exists() and read_json(request_path) != request:
        raise RuntimeError("semantic request changed; use a new output directory or explicit --force")
    if not request_path.exists() and attempts_root.exists() and any(attempts_root.iterdir()):
        raise RuntimeError("unidentified previous attempts; use a new output directory or explicit --force")
    if not request_path.exists() and any((output / name).exists() for name in PRODUCTS[:3]):
        raise RuntimeError("existing legacy outputs do not match this request; preserve them or use explicit --force")
    marker = output / "semantic_complete.json"
    if request_path.exists() and marker.exists() and all((output / name).is_file() for name in PRODUCTS[:6]):
        return read_json(marker)
    write_json(request_path, request)
    write_json(output / "source_lines.json", source)
    write_json(output / "outer_boundary_candidates.json", {
        "schema_version": "dp-p2p.outer-boundary-candidates.v1", "candidates": candidates})
    attempts_root.mkdir(exist_ok=True)
    feedback = None
    for number in range(1, args.max_attempts + 1):
        attempt = attempts_root / f"semantic-{number:03d}"
        if (attempt / "validation_error.json").exists():
            feedback = read_json(attempt / "validation_error.json")["error"]
            continue
        if (attempt / "failure.json").exists():
            if configuration_error(read_json(attempt / "failure.json")["error"]):
                attempt_summary(output)
                return {"state": "configuration_failed", "attempts": number}
            continue
        if (attempt / "host_failure.json").exists():
            raise RuntimeError(f"host failure is not a model retry: {attempt}")
        actual_prompt = prompt
        if feedback:
            actual_prompt += ("\nFORMAT VALIDATOR FEEDBACK:\n" + feedback +
                              "\nReturn the complete extraction under the same schema.\n")
        if (attempt / "response.txt").is_file():
            raw = read(attempt / "response.txt")
            usage = read_json(attempt / "usage.json") if (attempt / "usage.json").exists() else None
        elif attempt.exists():
            # A crashed in-flight attempt still consumes its original budget.
            write_json(attempt / "failure.json", {"kind": "call_failure", "error": "interrupted without output"})
            attempt_summary(output)
            continue
        else:
            attempt.mkdir()
            (attempt / "input.prompt.txt").write_text(actual_prompt, encoding="utf-8")
            schema_path = attempt / "response.schema.json"
            write_json(schema_path, schema)
            started = time.monotonic()
            write_json(attempt / "status.json", {"state": "running",
                       "started_utc": datetime.now(timezone.utc).isoformat()})
            backend = getattr(args, "backend", "auto") or "auto"
            if backend == "agy":
                from llm.agy_exec import call_agy
                call_model = call_agy
            elif backend == "pi-api":
                from llm.pi_exec import call_pi
                call_model = lambda p, **kw: call_pi(p, provider=getattr(args, "provider", "openai-compat"), chatgpt_login=False, **kw)
            elif backend == "pi-chatgpt":
                from llm.pi_exec import call_pi
                call_model = lambda p, **kw: call_pi(p, provider=getattr(args, "provider", "openai-codex"), chatgpt_login=True, **kw)
            elif backend == "claude":
                call_model = helpers.call_claude
            elif backend == "codex":
                call_model = helpers.call_codex
            else:
                call_model = helpers.call_claude if helpers.is_claude_model(args.model) else helpers.call_codex
            try:
                raw, usage = call_model(
                    actual_prompt, model=args.model, effort=args.effort, timeout=args.timeout,
                    cwd=helpers._PIPE, trace_dir=attempt / "trace", call_name="semantic",
                    result_dir=output, output_schema=schema_path, application="asset_extraction")
            except Exception as exc:
                write_json(attempt / "host_failure.json", {"error": repr(exc)})
                attempt_summary(output)
                raise
            if raw is not None:
                (attempt / "response.txt").write_text(raw, encoding="utf-8")
            write_json(attempt / "usage.json", usage)
            write_json(attempt / "status.json", {"state": "returned" if raw is not None else "failed",
                       "elapsed_seconds": time.monotonic() - started})
        if raw is None:
            error = transport_error(attempt, usage)
            write_json(attempt / "failure.json", {"kind": "call_failure", "error": error})
            attempt_summary(output)
            print(f"CALL FAILURE attempt={number}: {error[:300]}", flush=True)
            if configuration_error(error):
                return {"state": "configuration_failed", "attempts": number}
            continue
        try:
            wire = parse_response(raw, schema)
            obj = to_legacy(wire)
            quality = validate_semantic(obj, len(source), spec)
            structured = helpers.normalize_structured(obj, len(source))
            segments, corrections = helpers.enforce_explicit_figure_caption_invariants(obj["segments"], source)
            segments, structured = normalize_families(segments, structured)
        except (FormatError, ValueError, TypeError, KeyError) as exc:
            feedback = str(exc)
            write_json(attempt / "validation_error.json", {"kind": "invalid_format", "error": feedback})
            attempt_summary(output)
            print(f"FORMAT FAILURE attempt={number}: {feedback[:250]}", flush=True)
            continue
        write_json(attempt / "validation.json", {"state": "accepted", "raw_json_and_schema_valid": True})
        values = {"semantic_wire": wire, "semantic_legacy": obj, "segments2": segments,
                  "structured_elements": structured, "metadata_candidate": obj["metadata"],
                  "page_outer_boundaries": helpers.normalize_page_boundaries(obj["page_outer_boundaries"], candidates, pages),
                  "semantic_corrections": corrections, "semantic_quality": quality, "semantic_usage": usage}
        for name, value in values.items():
            write_json(output / f"{name}.json", value)
        summary = attempt_summary(output)
        result = {"state": "accepted", "attempts": number,
                  "first_attempt_accepted": summary["first_attempt_accepted"],
                  "source_lines": len(source), "segments": len(segments), "quality": quality}
        write_json(marker, result)
        return result
    attempt_summary(output)
    return {"state": "failed", "attempts": args.max_attempts}
