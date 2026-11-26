#!/usr/bin/env python3
"""Evaluate one or more posters with one fresh Codex session per question."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluator import (  # type: ignore
        VISUAL_EVALUATION_DIMENSIONS,
        aggregate_dimension,
        aggregate_results,
        build_system_prompt,
        build_user_prompt,
        extract_json_object,
        load_rubrics,
        question_ids,
        rubric_sha256,
        validate_dimension_response,
    )
    from poster_stimuli import prepare_question_poster  # type: ignore
else:
    from .evaluator import (
        VISUAL_EVALUATION_DIMENSIONS,
        aggregate_dimension,
        aggregate_results,
        build_system_prompt,
        build_user_prompt,
        extract_json_object,
        load_rubrics,
        question_ids,
        rubric_sha256,
        validate_dimension_response,
    )
    from .poster_stimuli import prepare_question_poster


def targeted_prompt(
    rubric: Mapping[str, Any], question: Mapping[str, Any]
) -> str:
    """Build the prompt for one atomic question in its own session."""

    return "\n\n".join(
        [
            "The attached poster image is the only evidence. Do not inspect "
            "files, run shell commands, browse the web, call tools, or use "
            "outside knowledge. Judge only the single requested atomic question.",
            build_system_prompt(rubric),
            build_user_prompt(rubric, [question]),
            f"Return exactly one result with ID {question['id']}. Your final "
            "response must satisfy the supplied JSON Schema.",
        ]
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate posters with one independent Codex process per question."
    )
    parser.add_argument("--poster", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Codex model override. The default is the Codex configuration.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default=None,
        help="Optional model_reasoning_effort override.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--ignore-user-config", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--codex-cwd",
        default=None,
        help=(
            "Working directory for codex exec. "
            "The default is an empty temporary directory, so the session cannot read this package."
        ),
    )
    parser.add_argument("--rubric-dir", default=str(script_dir / "rubrics"))
    parser.add_argument(
        "--output-schema",
        default=str(script_dir.parents[2] / "configs" / "schemas" / "vision_bench" / "single_question_response.schema.json"),
    )
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Run only these question IDs. Already stored answers are kept.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing answers and run only requested questions that are missing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_questions(
    rubrics: Mapping[str, Mapping[str, Any]],
    dimensions: list[str],
    question_ids_arg: list[str] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return rubric copies limited to the questions this invocation will run."""

    selected_rubrics = {
        dimension: {**rubric, "questions": list(rubric["questions"])}
        for dimension, rubric in rubrics.items()
    }
    if not question_ids_arg:
        return selected_rubrics, list(dimensions)

    requested = list(dict.fromkeys(question_ids_arg))
    question_dimensions = {
        str(question["id"]): dimension
        for dimension, rubric in rubrics.items()
        for question in rubric["questions"]
    }
    unknown = [question_id for question_id in requested if question_id not in question_dimensions]
    if unknown:
        raise ValueError(f"Unknown --question-ids: {unknown}")
    outside_dimensions = [
        question_id
        for question_id in requested
        if question_dimensions[question_id] not in dimensions
    ]
    if outside_dimensions:
        raise ValueError(
            "Requested question IDs are outside the visual-evaluation rubrics: "
            f"{outside_dimensions}"
        )

    requested_set = set(requested)
    for rubric in selected_rubrics.values():
        rubric["questions"] = [
            question
            for question in rubric["questions"]
            if str(question["id"]) in requested_set
        ]
    active_dimensions = [
        dimension
        for dimension in dimensions
        if selected_rubrics[dimension]["questions"]
    ]
    return selected_rubrics, active_dimensions


def existing_report_action(report: Mapping[str, Any], *, resume: bool) -> str:
    """Choose skip, resume, or replace for a report that is already on disk.

    A dry run writes a report with no answers. A later real run replaces it.
    A report that already stores answers or errors is kept unless ``--resume``
    is set, so those answers cannot be deleted by accident.
    """

    if resume:
        return "resume"
    if report.get("results") or report.get("errors"):
        return "skip"
    return "replace"


def questions_to_run(
    requested_ids: Sequence[str],
    stored_results: Mapping[str, Any],
) -> list[str]:
    """Return requested questions that do not already have a stored answer."""

    return [question_id for question_id in requested_ids if question_id not in stored_results]


def ensure_resume_compatible(
    report: Mapping[str, Any],
    *,
    fingerprint: str,
    model: str | None,
    reasoning_effort: str | None,
) -> None:
    """Refuse to mix answers from a different rubric or model setting."""

    if report.get("rubric_sha256") != fingerprint:
        raise ValueError(
            "Rubric hash does not match this report. "
            "Remove the report to start over."
        )
    if report.get("model") != model or report.get("reasoning_effort") != reasoning_effort:
        raise ValueError(
            "Model settings do not match this report. "
            "Pass the same --model and --reasoning-effort, or start a new output file."
        )


def make_codex_cwd(explicit: str | None) -> tuple[Path, bool]:
    """Return a working directory that does not contain the rubric package."""

    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Codex working directory does not exist: {path}")
        return path.resolve(), False
    return Path(tempfile.mkdtemp(prefix="visionbench_codex_")), True


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def resolve_codex_binary(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"Codex executable does not exist: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if not resolved:
        raise FileNotFoundError(
            f"Could not find {value!r} on PATH. Install Codex CLI or pass --codex-bin."
        )
    return resolved


def build_codex_command(
    args: SimpleNamespace,
    *,
    codex_binary: str,
    last_message_path: Path,
) -> list[str]:
    """Build one shell-free ``codex exec`` command for a single question."""

    command = [
        codex_binary,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--image",
        *[str(Path(path).resolve()) for path in args.poster],
        "--skip-git-repo-check",
        "--color",
        "never",
        "--json",
        "--cd",
        str(Path(args.codex_cwd).resolve()),
        "--output-schema",
        str(Path(args.output_schema).resolve()),
        "--output-last-message",
        str(last_message_path.resolve()),
    ]
    if args.ignore_user_config:
        command.append("--ignore-user-config")
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.model:
        command.extend(["--model", args.model])
    if args.reasoning_effort:
        command.extend(
            ["--config", f'model_reasoning_effort="{args.reasoning_effort}"']
        )
    command.append("-")
    return command


def command_for_display(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def run_codex_process(
    command: Sequence[str],
    *,
    prompt: str,
    timeout: float,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        write_text(stdout_path, stdout)
        write_text(stderr_path, stderr + f"\nTimed out after {timeout} seconds.\n")
        raise RuntimeError(f"Codex process timed out after {timeout} seconds") from exc

    write_text(stdout_path, completed.stdout)
    write_text(stderr_path, completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex exited with code {completed.returncode}; see {stderr_path}"
        )
    return completed


def stage_codex_image(
    source: Path,
    question_dir: Path,
    question_id: str,
) -> tuple[Path, str]:
    """Copy the image to a comma-free path. Codex splits ``--image`` on commas."""

    token = hashlib.sha256(str(question_dir.resolve()).encode("utf-8")).hexdigest()[:20]
    staging_dir = question_dir.parent.parent / "_codex_image_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".img"
    staged = staging_dir / f"{token}_{question_id}{suffix}"
    if staged.is_file():
        return staged, "existing"
    try:
        os.link(source, staged)
        mode = "hardlink"
    except OSError:
        shutil.copy2(source, staged)
        mode = "copy"
    return staged, mode


def new_report(
    *,
    poster: Path,
    fingerprint: str,
    model: str | None,
    reasoning_effort: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "3.0",
        "created_at": now,
        "updated_at": now,
        "poster": str(poster.resolve()),
        "rubric_sha256": fingerprint,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "run_dir": str(run_dir.resolve()),
        "results": {},
        "dimensions": {},
        "runs": {},
        "errors": {},
        "aggregate": {},
        "status": "running",
    }


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"Report must be a JSON object: {path}")
    report.setdefault("results", {})
    report.setdefault("dimensions", {})
    report.setdefault("runs", {})
    report.setdefault("errors", {})
    return report


def refresh_report(
    report: dict[str, Any],
    rubrics: Mapping[str, Mapping[str, Any]],
    *,
    status: str,
) -> None:
    results = report.get("results") or {}
    dims: dict[str, Any] = {}
    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        rubric = rubrics.get(dimension)
        if rubric is None:
            continue
        ordered = [
            results[str(question["id"])]
            for question in rubric["questions"]
            if str(question["id"]) in results
        ]
        block: dict[str, Any] = {
            "dimension": dimension,
            "dimension_title": rubric["dimension_title"],
            "results": ordered,
            "dimension_summary": (
                "Each atomic judgement was produced independently; consult per-question evidence."
            ),
        }
        block["aggregate"] = aggregate_dimension(block)
        dims[dimension] = block
    report["dimensions"] = dims
    report["aggregate"] = aggregate_results(results, rubrics)
    report["status"] = status
    report["updated_at"] = datetime.now(timezone.utc).isoformat()


def call_index_by_question(
    rubrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    index: dict[str, int] = {}
    number = 0
    for dimension in VISUAL_EVALUATION_DIMENSIONS:
        for question in rubrics[dimension]["questions"]:
            number += 1
            index[str(question["id"])] = number
    return index


def run_one(
    *,
    poster: Path,
    dimension: str,
    rubric: Mapping[str, Any],
    question: Mapping[str, Any],
    call_index: int,
    run_dir: Path,
    args: argparse.Namespace,
    codex_binary: str,
    codex_cwd: Path,
) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    question_id = str(question["id"])
    question_dir = run_dir / f"{call_index:02d}_{question_id}"
    question_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = question_dir / "prompt.txt"
    command_path = question_dir / "command.txt"
    last_message_path = question_dir / "last_message.json"
    events_path = question_dir / "events.jsonl"
    stderr_path = question_dir / "stderr.txt"

    prepared_poster, input_transform = prepare_question_poster(
        poster, question, question_dir
    )
    evaluation_poster, image_staging_mode = stage_codex_image(
        prepared_poster, question_dir, question_id
    )
    call_args = SimpleNamespace(
        poster=[str(evaluation_poster)],
        codex_cwd=str(codex_cwd),
        output_schema=args.output_schema,
        ignore_user_config=args.ignore_user_config,
        profile=args.profile,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    prompt = targeted_prompt(rubric, question)
    command = build_codex_command(
        call_args,
        codex_binary=codex_binary,
        last_message_path=last_message_path,
    )
    write_text(prompt_path, prompt)
    write_text(command_path, command_for_display(command) + "\n")
    started_at = datetime.now(timezone.utc)
    run_record: dict[str, Any] = {
        "question_id": question_id,
        "dimension": dimension,
        "call_index": call_index,
        "prompt_path": str(prompt_path.resolve()),
        "command_path": str(command_path.resolve()),
        "evaluation_image_path": str(evaluation_poster.resolve()),
        "prepared_image_path": str(prepared_poster.resolve()),
        "image_staging_mode": image_staging_mode,
        "input_transform": input_transform,
    }
    if args.dry_run:
        run_record["status"] = "planned"
        return question_id, None, run_record

    completed = run_codex_process(
        command,
        prompt=prompt,
        timeout=args.timeout,
        stdout_path=events_path,
        stderr_path=stderr_path,
    )
    if not last_message_path.is_file():
        raise RuntimeError(f"Codex did not write {last_message_path}")
    parsed = extract_json_object(last_message_path.read_text(encoding="utf-8"))
    normalized = validate_dimension_response(parsed, rubric, [question])
    result = normalized["results"][0]
    run_record.update(
        {
            "status": "completed",
            "created_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "codex_returncode": completed.returncode,
            "last_message_path": str(last_message_path.resolve()),
            "events_path": str(events_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
            "dimension_summary": normalized["dimension_summary"],
        }
    )
    return question_id, result, run_record


def canonical_status(report: Mapping[str, Any], all_ids: Sequence[str], *, dry_run: bool) -> str:
    if dry_run:
        return "dry_run"
    results = report.get("results", {})
    if report.get("errors"):
        return "partial"
    if all(question_id in results for question_id in all_ids):
        return "completed"
    return "partial"


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    posters = [Path(path).resolve() for path in args.poster]
    for poster in posters:
        if not poster.is_file():
            raise FileNotFoundError(poster)
    stems = [poster.stem for poster in posters]
    if len(stems) != len(set(stems)):
        raise ValueError("Two posters share the same filename stem; their reports would collide.")
    if not Path(args.output_schema).is_file():
        raise FileNotFoundError(args.output_schema)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rubrics = load_rubrics(args.rubric_dir)
    fingerprint = rubric_sha256(args.rubric_dir)
    selected, _dimensions = select_questions(
        rubrics,
        list(VISUAL_EVALUATION_DIMENSIONS),
        args.question_ids,
    )
    requested_ids = question_ids(selected)
    indexes = call_index_by_question(rubrics)
    all_ids = question_ids(rubrics)

    reports: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, Path] = {}
    tasks: list[tuple[Path, str, Mapping[str, Any], Mapping[str, Any], int]] = []
    for poster in posters:
        key = poster.stem
        output_path = output_dir / f"{key}.json"
        run_dir = output_dir / f"{key}_runs"
        if output_path.is_file():
            loaded = load_report(output_path)
            action = existing_report_action(loaded, resume=args.resume)
            if action == "skip":
                print(
                    f"skip {output_path}: already has answers "
                    "(pass --resume to fill missing questions)"
                )
                continue
            if action == "resume":
                report = loaded
                report.setdefault("run_dir", str(run_dir.resolve()))
                ensure_resume_compatible(
                    report,
                    fingerprint=fingerprint,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                )
            else:
                report = new_report(
                    poster=poster,
                    fingerprint=fingerprint,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    run_dir=run_dir,
                )
        else:
            report = new_report(
                poster=poster,
                fingerprint=fingerprint,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                run_dir=run_dir,
            )
        reports[key] = report
        output_paths[key] = output_path
        pending = set(questions_to_run(requested_ids, report["results"]))
        for dimension in VISUAL_EVALUATION_DIMENSIONS:
            rubric = selected[dimension]
            for question in rubric["questions"]:
                question_id = str(question["id"])
                if question_id not in pending:
                    continue
                tasks.append(
                    (poster, dimension, rubrics[dimension], question, indexes[question_id])
                )
        refresh_report(
            report,
            rubrics,
            status=canonical_status(report, all_ids, dry_run=args.dry_run),
        )
        write_json(output_path, report)

    if not tasks:
        print(f"no questions to run for {len(posters)} poster(s)")
        return 0

    codex_cwd, delete_cwd = make_codex_cwd(args.codex_cwd)
    codex_binary = args.codex_bin if args.dry_run else resolve_codex_binary(args.codex_bin)
    print(
        f"planned {len(tasks)} per-question session(s) "
        f"for {len(reports)} poster(s)"
    )
    failed = False
    try:
        future_context: dict[Future[Any], tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for poster, dimension, rubric, question, call_index in tasks:
                key = poster.stem
                future = executor.submit(
                    run_one,
                    poster=poster,
                    dimension=dimension,
                    rubric=rubric,
                    question=question,
                    call_index=call_index,
                    run_dir=Path(reports[key]["run_dir"]),
                    args=args,
                    codex_binary=codex_binary,
                    codex_cwd=codex_cwd,
                )
                future_context[future] = (key, str(question["id"]))

            finished = 0
            total = len(tasks)
            for future in as_completed(future_context):
                key, expected_id = future_context[future]
                report = reports[key]
                try:
                    question_id, result, run_record = future.result()
                    if question_id != expected_id:
                        raise RuntimeError(
                            f"Expected {expected_id}, worker returned {question_id}"
                        )
                    report["runs"][question_id] = run_record
                    if result is not None:
                        report["results"][question_id] = result
                        report["errors"].pop(question_id, None)
                except Exception as exc:
                    failed = True
                    report["errors"][expected_id] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                refresh_report(
                    report,
                    rubrics,
                    status=canonical_status(report, all_ids, dry_run=args.dry_run),
                )
                write_json(output_paths[key], report)
                finished += 1
                verdict = report["runs"].get(expected_id, {}).get("status", "failed")
                print(f"[{finished}/{total}] {key} {expected_id}: {verdict}", flush=True)
    finally:
        if delete_cwd:
            shutil.rmtree(codex_cwd, ignore_errors=True)

    for key, report in reports.items():
        refresh_report(
            report,
            rubrics,
            status=canonical_status(report, all_ids, dry_run=args.dry_run),
        )
        write_json(output_paths[key], report)
        present = len(report["results"])
        print(
            f"saved {output_paths[key]} "
            f"({present}/{len(all_ids)}, status={report['status']})"
        )
    return 1 if failed and not args.dry_run else 0


def poster_key(poster: Path) -> str:
    """Stable identifier for a poster image path."""
    return poster.stem


def completed_ids(report: Mapping[str, Any]) -> set[str]:
    """Return all question IDs already recorded in a VisionBench report."""
    done: set[str] = set()
    for dim_block in (report.get("dimensions") or {}).values():
        if not isinstance(dim_block, Mapping):
            continue
        for item in dim_block.get("results") or []:
            if isinstance(item, Mapping) and item.get("id"):
                done.add(str(item["id"]))
    return done


def normalize_report(
    report: dict[str, Any],
    rubrics: Mapping[str, Any],
    dimensions: list[str] | tuple[str, ...] = VISUAL_EVALUATION_DIMENSIONS,
) -> dict[str, Any]:
    """Recompute per-dimension aggregates and equal-group summaries on a VisionBench report."""
    dims_out = report.setdefault("dimensions", {})
    flat_results: dict[str, Mapping[str, Any]] = {}
    for dimension in dimensions:
        rubric = rubrics.get(dimension)
        if rubric is None:
            continue
        existing_block = dims_out.get(dimension)
        existing_results = (
            existing_block.get("results") or []
            if isinstance(existing_block, Mapping)
            else []
        )
        by_id = {
            str(item["id"]): item
            for item in existing_results
            if isinstance(item, Mapping) and item.get("id")
        }
        ordered = [
            by_id[str(question["id"])]
            for question in rubric["questions"]
            if str(question["id"]) in by_id
        ]
        summary_text = (
            existing_block.get("dimension_summary")
            if isinstance(existing_block, Mapping)
            else None
        )
        block: dict[str, Any] = {
            "dimension": dimension,
            "dimension_title": rubric["dimension_title"],
            "results": ordered,
            "dimension_summary": (
                summary_text
                or "Each atomic judgement was produced independently; consult per-question evidence."
            ),
        }
        block["aggregate"] = aggregate_dimension(block)
        dims_out[dimension] = block
        for item in ordered:
            flat_results[str(item["id"])] = item
    ordered_rubrics = {d: rubrics[d] for d in dimensions if d in rubrics}
    summary = aggregate_results(flat_results, ordered_rubrics)
    report["aggregate"] = summary
    report["summary"] = summary
    return report


if __name__ == "__main__":
    raise SystemExit(main())
