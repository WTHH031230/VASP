from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BENCH_ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = BENCH_ROOT / "paper_grounding_audit"
REPO_ROOT = BENCH_ROOT.parents[3]
REFERENCE_PATH = Path(os.environ.get("VASP_DATA_DIR", str(REPO_ROOT.parent / "VASP-data"))) / "key_info_coverage_700" / "reference_information_units_before.json"
MANIFEST_PATH = AUDIT_ROOT / "paper_manifest.json"
PROMPT_PATH = BENCH_ROOT / "prompts" / "audit_prompt.md"
SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "information_coverage" / "audit_output_schema.json"
RESULT_ROOT = AUDIT_ROOT / "results"
LOG_ROOT = AUDIT_ROOT / "logs"
SUMMARY_PATH = AUDIT_ROOT / "summary.json"
REPORT_PATH = AUDIT_ROOT / "REPORT.md"
ISSUES_PATH = AUDIT_ROOT / "issues.csv"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    references = read_json(REFERENCE_PATH)["posters"]
    papers = read_json(MANIFEST_PATH)["papers"]
    if [row["poster_id"] for row in references] != [str(i) for i in range(100)]:
        raise RuntimeError("Reference IDs are not exactly 0-99")
    if [row["poster_id"] for row in papers] != [str(i) for i in range(100)]:
        raise RuntimeError("Paper IDs are not exactly 0-99")
    return references, {row["poster_id"]: row for row in papers}


def validate_result(reference: dict[str, Any], result: dict[str, Any]) -> None:
    poster_id = reference["poster_id"]
    if result.get("poster_id") != poster_id:
        raise ValueError(f"poster_id mismatch for {poster_id}")
    units = result.get("units")
    if not isinstance(units, list) or len(units) != 7:
        raise ValueError(f"poster {poster_id} must contain seven audited units")
    expected = [unit["unit_id"] for unit in reference["information_units"]]
    actual = [unit.get("unit_id") for unit in units]
    if actual != expected:
        raise ValueError(f"unit IDs mismatch for {poster_id}: {actual}")
    if all(unit.get("verdict") == "UNREADABLE" for unit in units):
        raise ValueError(
            f"poster {poster_id} returned all UNREADABLE; treat as an execution/read failure"
        )


def build_prompt(
    template: str, reference: dict[str, Any], paper: dict[str, Any]
) -> str:
    payload = {
        "poster_id": reference["poster_id"],
        "benchmark_title": reference["title"],
        "paper_text_file": paper["text"],
        "instruction": (
            "Read the complete paper_text_file with local read/search tools before judging. "
            "Determine paper identity only from the PDF text, especially PDF PAGE 1; do not "
            "infer its title from any filesystem path. The page markers in that file are the "
            "authoritative PDF page numbers."
        ),
        "information_units": reference["information_units"],
    }
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def run_one(
    codex: str,
    template: str,
    reference: dict[str, Any],
    paper: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    poster_id = reference["poster_id"]
    result_path = RESULT_ROOT / f"{poster_id}.json"
    if result_path.is_file() and not force:
        existing = read_json(result_path)
        validate_result(reference, existing)
        return {"poster_id": poster_id, "status": "reused"}

    prompt = build_prompt(template, reference, paper)
    poster_log_root = LOG_ROOT / poster_id
    poster_log_root.mkdir(parents=True, exist_ok=True)
    errors = []
    prior_attempts = []
    for path in poster_log_root.glob("attempt_*.json"):
        match = re.fullmatch(r"attempt_(\d+)\.json", path.name)
        if match:
            prior_attempts.append(int(match.group(1)))
    first_attempt = max(prior_attempts, default=0) + 1
    for attempt in range(first_attempt, first_attempt + 3):
        last_message = poster_log_root / f"attempt_{attempt}_last_message.json"
        log_path = poster_log_root / f"attempt_{attempt}.json"
        command = [
            codex,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "-C",
            str(REPO_ROOT),
            "-m",
            MODEL,
            "-c",
            f'model_reasoning_effort="{REASONING_EFFORT}"',
            "--output-schema",
            str(SCHEMA_PATH),
            "--output-last-message",
            str(last_message),
            "-",
        ]
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
        timed_out = False
        try:
            combined_output, _ = process.communicate(prompt, timeout=1200)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                combined_output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                combined_output, _ = process.communicate()
            combined_output += "\n[TIMEOUT: process group terminated after 1200 seconds]\n"
            returncode = 124
        elapsed = round(time.perf_counter() - started, 3)
        write_json(
            log_path,
            {
                "poster_id": poster_id,
                "stage": "paper_grounding_audit",
                "attempt": attempt,
                "command": command,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "paper_pdf": paper["pdf"],
                "paper_sha256": paper["pdf_sha256"],
                "paper_text": paper["text"],
                "prompt": prompt,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "combined_output": combined_output,
            },
        )
        try:
            if timed_out:
                raise RuntimeError("codex exec timed out after 1200 seconds")
            if returncode != 0:
                raise RuntimeError(f"codex exec returned {returncode}")
            result = read_json(last_message)
            validate_result(reference, result)
            result["benchmark_title"] = reference["title"]
            result["pdf_filename"] = paper["pdf_filename"]
            result["metadata"] = {
                "paper_pdf": paper["pdf"],
                "paper_sha256": paper["pdf_sha256"],
                "paper_text": paper["text"],
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "session_policy": "fresh ephemeral codex exec; never resumed",
                "attempt": attempt,
            }
            write_json(result_path, result)
            return {
                "poster_id": poster_id,
                "status": "completed",
                "attempt": attempt,
                "elapsed_seconds": elapsed,
            }
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            write_json(
                poster_log_root / f"attempt_{attempt}_error.json",
                {"error": errors[-1]},
            )
    raise RuntimeError(f"poster {poster_id} failed: {'; '.join(errors)}")


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def summarize(references: list[dict[str, Any]]) -> None:
    ref_by_id = {row["poster_id"]: row for row in references}
    verdict_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    alignment_counts: Counter[str] = Counter()
    problem_posters = set()
    issue_rows = []
    all_results = []
    for poster_id in map(str, range(100)):
        result = read_json(RESULT_ROOT / f"{poster_id}.json")
        validate_result(ref_by_id[poster_id], result)
        all_results.append(result)
        alignment_counts[result["paper_alignment"]] += 1
        original_by_id = {
            unit["unit_id"]: unit
            for unit in ref_by_id[poster_id]["information_units"]
        }
        for audit in result["units"]:
            verdict_counts[audit["verdict"]] += 1
            issue_counts[audit["issue_type"]] += 1
            if audit["verdict"] != "SUPPORTED":
                problem_posters.add(poster_id)
                original = original_by_id[audit["unit_id"]]
                evidence = " / ".join(
                    f"p.{item['page']} {item['location']}: {item['quote']}"
                    for item in audit["evidence"]
                )
                issue_rows.append(
                    {
                        "poster_id": poster_id,
                        "paper_title": result["paper_title_detected"],
                        "paper_alignment": result["paper_alignment"],
                        "unit_id": audit["unit_id"],
                        "topic": original["topic"],
                        "original_statement": original["statement"],
                        "verdict": audit["verdict"],
                        "issue_type": audit["issue_type"],
                        "confidence": audit["confidence"],
                        "reasoning": audit["reasoning"],
                        "paper_evidence": evidence,
                        "recommended_statement": audit["recommended_statement"],
                    }
                )

    summary = {
        "paper_count": 100,
        "unit_count": 700,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "issue_type_counts": dict(sorted(issue_counts.items())),
        "paper_alignment_counts": dict(sorted(alignment_counts.items())),
        "posters_with_non_supported_units": len(problem_posters),
        "non_supported_unit_count": len(issue_rows),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
    }
    write_json(SUMMARY_PATH, summary)

    fieldnames = [
        "poster_id",
        "paper_title",
        "paper_alignment",
        "unit_id",
        "topic",
        "original_statement",
        "verdict",
        "issue_type",
        "confidence",
        "reasoning",
        "paper_evidence",
        "recommended_statement",
    ]
    with ISSUES_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(issue_rows)

    lines = [
        "# Paper-Grounding Audit of Official-Poster Information Units",
        "",
        "This report audits all 700 information units extracted from the 100 official posters against their corresponding source paper PDFs.",
        "",
        "## Summary",
        "",
        f"- Papers: {summary['paper_count']}; information units: {summary['unit_count']}.",
        f"- Supported: {verdict_counts['SUPPORTED']}.",
        f"- Supported with narrowing: {verdict_counts['SUPPORTED_WITH_NARROWING']}.",
        f"- Not found in paper: {verdict_counts['NOT_FOUND']}.",
        f"- Contradicted by paper: {verdict_counts['CONTRADICTED']}.",
        f"- Unreadable: {verdict_counts['UNREADABLE']}.",
        f"- Papers with at least one non-fully-supported unit: {len(problem_posters)}.",
        "",
        "Note: `SUPPORTED_WITH_NARROWING` indicates that the core claim is supported by the paper, but the original unit includes overly strong, broad, specific, or compound additions.",
        "",
        "## Flagged Information Units",
        "",
    ]
    for poster_id in map(str, range(100)):
        rows = [row for row in issue_rows if row["poster_id"] == poster_id]
        if not rows:
            continue
        result = all_results[int(poster_id)]
        lines.extend(
            [
                f"### {poster_id} | {result['paper_title_detected']}",
                "",
                f"Paper alignment: `{result['paper_alignment']}`. {result['overall_assessment']}",
                "",
            ]
        )
        for row in rows:
            lines.extend(
                [
                    f"#### {row['unit_id']} | {row['topic']}",
                    "",
                    f"- Verdict: `{row['verdict']}` / `{row['issue_type']}` / confidence `{row['confidence']}`",
                    f"- Original statement: {row['original_statement']}",
                    f"- Reasoning: {row['reasoning']}",
                    f"- Paper evidence: {row['paper_evidence'] or 'Insufficient evidence found in paper'}",
                    f"- Recommended statement: {row['recommended_statement'] or 'Remove unit or rewrite after manual review'}",
                    "",
                ]
            )
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    references, papers = load_inputs()
    if args.summarize_only:
        summarize(references)
        return
    selected_ids = set(args.ids) if args.ids else set(range(100))
    if any(value < 0 or value > 99 for value in selected_ids):
        raise ValueError("--ids values must be between 0 and 99")
    selected = [row for row in references if int(row["poster_id"]) in selected_ids]
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex executable not found")
    template = PROMPT_PATH.read_text(encoding="utf-8")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                codex,
                template,
                reference,
                papers[reference["poster_id"]],
                args.force,
            ): reference
            for reference in selected
        }
        for future in as_completed(futures):
            reference = futures[future]
            try:
                record = future.result()
                print(
                    f"[{record['poster_id']}] {record['status']} "
                    f"attempt={record.get('attempt', '-')} "
                    f"elapsed={record.get('elapsed_seconds', '-')}"
                )
            except Exception as exc:
                message = (
                    f"{reference['poster_id']}: {type(exc).__name__}: {exc}"
                )
                failures.append(message)
                print(f"[{reference['poster_id']}] FAILED: {exc}")
    if failures:
        write_json(AUDIT_ROOT / "failures.json", {"failures": failures})
        raise RuntimeError("; ".join(failures))
    if selected_ids == set(range(100)):
        summarize(references)
        print(f"Wrote {SUMMARY_PATH}, {ISSUES_PATH}, and {REPORT_PATH}")


if __name__ == "__main__":
    main()
