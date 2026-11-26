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

if __package__ in {None, ""}:
    import run_audit as base
else:
    from . import run_audit as base


BENCH_ROOT = Path(__file__).resolve().parent
AUDIT_ROOT = BENCH_ROOT / "paper_grounding_audit"
REPO_ROOT = BENCH_ROOT.parents[3]
PROMPT_PATH = BENCH_ROOT / "prompts" / "adjudication_prompt.md"
SCHEMA_PATH = REPO_ROOT / "configs" / "schemas" / "information_coverage" / "adjudication_output_schema.json"
RESULT_ROOT = AUDIT_ROOT / "adjudication" / "results"
LOG_ROOT = AUDIT_ROOT / "adjudication" / "logs"
SUMMARY_PATH = AUDIT_ROOT / "adjudication_summary.json"
ISSUES_PATH = AUDIT_ROOT / "FINAL_ISSUES.csv"
REPORT_PATH = AUDIT_ROOT / "FINAL_REPORT.md"
CRITICAL_PATH = AUDIT_ROOT / "CRITICAL_ISSUES.md"


def flagged_units(initial: dict[str, Any]) -> list[dict[str, Any]]:
    return [unit for unit in initial["units"] if unit["verdict"] != "SUPPORTED"]


def validate_result(
    poster_id: str, expected_ids: list[str], result: dict[str, Any]
) -> None:
    if result.get("poster_id") != poster_id:
        raise ValueError(f"poster_id mismatch for {poster_id}")
    if result.get("paper_read_status") != "READ":
        raise ValueError(f"paper {poster_id} was not successfully read")
    reviews = result.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError(f"reviews missing for {poster_id}")
    actual_ids = [row.get("unit_id") for row in reviews]
    if actual_ids != expected_ids:
        raise ValueError(
            f"review unit IDs mismatch for {poster_id}: {actual_ids} != {expected_ids}"
        )
    for row in reviews:
        if row["decision"] == "DISMISS" and row["final_verdict"] != "SUPPORTED":
            raise ValueError(f"DISMISS must yield SUPPORTED for {row['unit_id']}")
        if row["final_verdict"] == "SUPPORTED" and row["final_issue_type"] != "NONE":
            raise ValueError(f"SUPPORTED must use NONE for {row['unit_id']}")


def build_prompt(
    template: str,
    reference: dict[str, Any],
    paper: dict[str, Any],
    initial: dict[str, Any],
) -> tuple[str, list[str]]:
    original_by_id = {
        unit["unit_id"]: unit for unit in reference["information_units"]
    }
    flags = flagged_units(initial)
    expected_ids = [row["unit_id"] for row in flags]
    payload = {
        "poster_id": reference["poster_id"],
        "benchmark_title": reference["title"],
        "paper_text_file": paper["text"],
        "instruction": (
            "Read the paper text locally. Its ===== PDF PAGE N ===== markers define page numbers. "
            "Determine the final status of only the supplied flagged units."
        ),
        "flagged_units": [
            {
                "original_unit": original_by_id[row["unit_id"]],
                "first_pass_audit": row,
            }
            for row in flags
        ],
    }
    prompt = template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    return prompt, expected_ids


def execute(
    command: list[str], prompt: str, timeout_seconds: int
) -> tuple[int, float, str, bool]:
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
        output, _ = process.communicate(prompt, timeout=timeout_seconds)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, _ = process.communicate()
        output += f"\n[TIMEOUT after {timeout_seconds} seconds]\n"
        returncode = 124
    elapsed = round(time.perf_counter() - started, 3)
    return returncode, elapsed, output, timed_out


def run_one(
    codex: str,
    template: str,
    reference: dict[str, Any],
    paper: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    poster_id = reference["poster_id"]
    initial = base.read_json(base.RESULT_ROOT / f"{poster_id}.json")
    flags = flagged_units(initial)
    if not flags:
        return {"poster_id": poster_id, "status": "no_flags"}
    prompt, expected_ids = build_prompt(template, reference, paper, initial)
    result_path = RESULT_ROOT / f"{poster_id}.json"
    if result_path.is_file() and not force:
        existing = base.read_json(result_path)
        validate_result(poster_id, expected_ids, existing)
        return {"poster_id": poster_id, "status": "reused"}

    log_root = LOG_ROOT / poster_id
    log_root.mkdir(parents=True, exist_ok=True)
    prior_attempts = []
    for path in log_root.glob("attempt_*.json"):
        match = re.fullmatch(r"attempt_(\d+)\.json", path.name)
        if match:
            prior_attempts.append(int(match.group(1)))
    first_attempt = max(prior_attempts, default=0) + 1
    errors = []
    for attempt in range(first_attempt, first_attempt + 3):
        last_message = log_root / f"attempt_{attempt}_last_message.json"
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
            base.MODEL,
            "-c",
            f'model_reasoning_effort="{base.REASONING_EFFORT}"',
            "--output-schema",
            str(SCHEMA_PATH),
            "--output-last-message",
            str(last_message),
            "-",
        ]
        returncode, elapsed, output, timed_out = execute(command, prompt, 1200)
        base.write_json(
            log_root / f"attempt_{attempt}.json",
            {
                "poster_id": poster_id,
                "stage": "paper_grounding_adjudication",
                "attempt": attempt,
                "command": command,
                "model": base.MODEL,
                "reasoning_effort": base.REASONING_EFFORT,
                "paper_pdf": paper["pdf"],
                "paper_sha256": paper["pdf_sha256"],
                "paper_text": paper["text"],
                "prompt": prompt,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "combined_output": output,
            },
        )
        try:
            if timed_out:
                raise RuntimeError("adjudication timed out")
            if returncode != 0:
                raise RuntimeError(f"codex exec returned {returncode}")
            result = base.read_json(last_message)
            validate_result(poster_id, expected_ids, result)
            result["metadata"] = {
                "paper_pdf": paper["pdf"],
                "paper_sha256": paper["pdf_sha256"],
                "model": base.MODEL,
                "reasoning_effort": base.REASONING_EFFORT,
                "attempt": attempt,
            }
            base.write_json(result_path, result)
            return {
                "poster_id": poster_id,
                "status": "completed",
                "attempt": attempt,
                "elapsed_seconds": elapsed,
            }
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            errors.append(error)
            base.write_json(log_root / f"attempt_{attempt}_error.json", {"error": error})
    raise RuntimeError(f"poster {poster_id} failed: {'; '.join(errors)}")


def aggregate(references: list[dict[str, Any]]) -> None:
    verdict_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    final_rows = []
    changed = 0
    posters_with_issues = set()
    for reference in references:
        poster_id = reference["poster_id"]
        initial = base.read_json(base.RESULT_ROOT / f"{poster_id}.json")
        original_by_id = {
            unit["unit_id"]: unit for unit in reference["information_units"]
        }
        flags = flagged_units(initial)
        review_by_id: dict[str, dict[str, Any]] = {}
        if flags:
            adjudication = base.read_json(RESULT_ROOT / f"{poster_id}.json")
            expected_ids = [unit["unit_id"] for unit in flags]
            validate_result(poster_id, expected_ids, adjudication)
            review_by_id = {row["unit_id"]: row for row in adjudication["reviews"]}
        for first in initial["units"]:
            unit_id = first["unit_id"]
            if unit_id in review_by_id:
                review = review_by_id[unit_id]
                verdict = review["final_verdict"]
                issue_type = review["final_issue_type"]
                reasoning = review["reasoning"]
                evidence_items = review["evidence"]
                recommendation = review["recommended_statement"]
                confidence = review["confidence"]
                decision = review["decision"]
                decision_counts[decision] += 1
                if verdict != first["verdict"] or issue_type != first["issue_type"]:
                    changed += 1
            else:
                verdict = first["verdict"]
                issue_type = first["issue_type"]
                reasoning = first["reasoning"]
                evidence_items = first["evidence"]
                recommendation = first["recommended_statement"]
                confidence = first["confidence"]
                decision = "NOT_REVIEWED_FULLY_SUPPORTED"
            verdict_counts[verdict] += 1
            issue_counts[issue_type] += 1
            if verdict == "SUPPORTED":
                continue
            posters_with_issues.add(poster_id)
            original = original_by_id[unit_id]
            evidence = " / ".join(
                f"p.{item['page']} {item['location']}: {item['quote']}"
                for item in evidence_items
            )
            final_rows.append(
                {
                    "poster_id": poster_id,
                    "paper_title": initial["paper_title_detected"],
                    "paper_alignment": initial["paper_alignment"],
                    "unit_id": unit_id,
                    "topic": original["topic"],
                    "original_statement": original["statement"],
                    "first_pass_verdict": first["verdict"],
                    "adjudication_decision": decision,
                    "final_verdict": verdict,
                    "final_issue_type": issue_type,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "paper_evidence": evidence,
                    "recommended_statement": recommendation,
                }
            )

    summary = {
        "paper_count": 100,
        "unit_count": 700,
        "first_pass_flag_count": sum(decision_counts.values()),
        "adjudication_decision_counts": dict(sorted(decision_counts.items())),
        "changed_classification_count": changed,
        "final_verdict_counts": dict(sorted(verdict_counts.items())),
        "final_issue_type_counts": dict(sorted(issue_counts.items())),
        "final_issue_unit_count": len(final_rows),
        "posters_with_final_issues": len(posters_with_issues),
        "model": base.MODEL,
        "reasoning_effort": base.REASONING_EFFORT,
    }
    base.write_json(SUMMARY_PATH, summary)

    fieldnames = [
        "poster_id",
        "paper_title",
        "paper_alignment",
        "unit_id",
        "topic",
        "original_statement",
        "first_pass_verdict",
        "adjudication_decision",
        "final_verdict",
        "final_issue_type",
        "confidence",
        "reasoning",
        "paper_evidence",
        "recommended_statement",
    ]
    with ISSUES_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    lines = [
        "# Paper-Grounding Audit of Official-Poster Information Units (Adjudicated)",
        "",
        "This report audits all 700 official-poster information units against their corresponding source paper PDFs and independently adjudicates every first-pass flagged unit.",
        "",
        "## Final Summary",
        "",
        f"- Supported: {verdict_counts['SUPPORTED']}.",
        f"- Supported with narrowing: {verdict_counts['SUPPORTED_WITH_NARROWING']}.",
        f"- Not found in paper: {verdict_counts['NOT_FOUND']}.",
        f"- Contradicted by paper: {verdict_counts['CONTRADICTED']}.",
        f"- Final flagged units: {len(final_rows)} across {len(posters_with_issues)} papers.",
        f"- Second-pass decisions: CONFIRM={decision_counts['CONFIRM']}; REVISE={decision_counts['REVISE']}; DISMISS={decision_counts['DISMISS']}.",
        "",
        "## Final Flagged Units",
        "",
    ]
    current_poster = None
    for row in final_rows:
        if row["poster_id"] != current_poster:
            current_poster = row["poster_id"]
            lines.extend(
                [
                    f"### {row['poster_id']} | {row['paper_title']}",
                    "",
                    f"Paper alignment: `{row['paper_alignment']}`",
                    "",
                ]
            )
        lines.extend(
            [
                f"#### {row['unit_id']} | {row['topic']}",
                "",
                f"- Final verdict: `{row['final_verdict']}` / `{row['final_issue_type']}` / `{row['confidence']}`",
                f"- Adjudication decision: `{row['adjudication_decision']}` (first pass: `{row['first_pass_verdict']}`)",
                f"- Original statement: {row['original_statement']}",
                f"- Adjudication reasoning: {row['reasoning']}",
                f"- Paper evidence: {row['paper_evidence'] or 'Insufficient evidence found in paper'}",
                f"- Recommended statement: {row['recommended_statement'] or 'Remove unit or rewrite after manual review'}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    critical_rows = [
        row
        for row in final_rows
        if row["final_verdict"] in {"NOT_FOUND", "CONTRADICTED"}
    ]
    critical_lines = [
        "# Information Units Not Found In or Contradicted By Source Papers",
        "",
        f"Total {len(critical_rows)}: NOT_FOUND = "
        f"{sum(row['final_verdict'] == 'NOT_FOUND' for row in critical_rows)}, "
        f"CONTRADICTED = {sum(row['final_verdict'] == 'CONTRADICTED' for row in critical_rows)}.",
        "",
    ]
    for row in critical_rows:
        critical_lines.extend(
            [
                f"## {row['unit_id']} | {row['topic']}",
                "",
                f"- Paper: {row['poster_id']} | {row['paper_title']}",
                f"- Verdict: `{row['final_verdict']}` / `{row['final_issue_type']}`",
                f"- Original statement: {row['original_statement']}",
                f"- Assessment: {row['reasoning']}",
                f"- Paper evidence: {row['paper_evidence'] or 'Insufficient evidence found in paper'}",
                f"- Recommendation: {row['recommended_statement'] or 'Remove unit'}",
                "",
            ]
        )
    CRITICAL_PATH.write_text(
        "\n".join(critical_lines).rstrip() + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    references, papers = base.load_inputs()
    if args.aggregate_only:
        aggregate(references)
        return
    flagged_references = [
        row
        for row in references
        if flagged_units(base.read_json(base.RESULT_ROOT / f"{row['poster_id']}.json"))
    ]
    selected_ids = set(args.ids) if args.ids else {
        int(row["poster_id"]) for row in flagged_references
    }
    selected = [
        row for row in flagged_references if int(row["poster_id"]) in selected_ids
    ]
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
                    f"elapsed={record.get('elapsed_seconds', '-')}",
                    flush=True,
                )
            except Exception as exc:
                message = f"{reference['poster_id']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[{reference['poster_id']}] FAILED: {exc}", flush=True)
    if failures:
        base.write_json(AUDIT_ROOT / "adjudication_failures.json", {"failures": failures})
        raise RuntimeError("; ".join(failures))
    if selected_ids == {int(row["poster_id"]) for row in flagged_references}:
        aggregate(references)
        print(
            f"Wrote {SUMMARY_PATH}, {ISSUES_PATH}, {REPORT_PATH}, and {CRITICAL_PATH}"
        )


if __name__ == "__main__":
    main()
