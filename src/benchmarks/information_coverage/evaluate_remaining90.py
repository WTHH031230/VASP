from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common


REFERENCE_PATH = common.BENCH_ROOT / "reference_information_units_10_99.json"
SCHEMA_PATH = common.BENCH_ROOT.parents[2] / "configs" / "schemas" / "information_coverage" / "evaluation_output_schema.json"
PROMPTS = {
    "official": common.BENCH_ROOT / "official_selfcheck_prompt.md",
    "rs": common.BENCH_ROOT / "evaluation_prompt.md",
    "whh": common.BENCH_ROOT / "evaluation_prompt.md",
}


def load_posters() -> list[dict[str, Any]]:
    posters = common.read_json(REFERENCE_PATH)["posters"]
    if [item["poster_id"] for item in posters] != [str(i) for i in range(10, 100)]:
        raise RuntimeError("Reference poster IDs are not 10-99")
    manifest = {
        item["poster_id"]: item for item in common.ensure_manifest()
    }
    for poster in posters:
        common.validate_units(poster, poster)
        mapped = manifest[poster["poster_id"]]
        poster["official_poster"] = mapped["official_poster"]
        poster["rs_poster"] = mapped["rs_poster"]
        poster["whh_poster"] = mapped["whh_poster"]
    return posters


def target_image(poster_id: str, variant: str, poster: dict[str, Any]) -> Path:
    if variant == "official":
        staged = (
            common.BENCH_ROOT
            / "reference_extraction"
            / "staged_official_images"
            / f"{poster_id}.png"
        )
    else:
        staged = common.BENCH_ROOT / "staged_images" / variant / f"{poster_id}.png"
    if staged.is_file():
        return staged
    fallback = common.ROOT / poster[f"{variant}_poster"]
    if not fallback.is_file():
        raise FileNotFoundError(fallback)
    return fallback


def build_prompt(template: str, poster: dict[str, Any]) -> str:
    payload = {
        "poster_id": poster["poster_id"],
        "title": poster["title"],
        "reference_information_units": poster["information_units"],
    }
    return template.rstrip() + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )


def run_one(
    codex_prefix: list[str],
    template: str,
    poster: dict[str, Any],
    variant: str,
    result_root: Path,
    log_root: Path,
    force: bool,
) -> dict[str, Any]:
    poster_id = poster["poster_id"]
    result_path = result_root / f"{poster_id}.json"
    if result_path.is_file() and not force:
        existing = common.read_json(result_path)
        common.validate_evaluation(poster, existing)
        return {
            "poster_id": poster_id,
            "status": "reused",
            "missing_count": existing["missing_count"],
            "attempt": None,
        }

    image_path = target_image(poster_id, variant, poster)
    prompt = build_prompt(template, poster)
    poster_log_root = log_root / poster_id
    common.clear_attempt_artifacts(poster_log_root, force)
    errors: list[str] = []
    for attempt in range(1, 3):
        last_message = poster_log_root / f"attempt_{attempt}_last_message.json"
        returncode, elapsed, _ = common.run_codex(
            codex_prefix=codex_prefix,
            prompt=prompt,
            image_path=image_path,
            schema_path=SCHEMA_PATH,
            last_message=last_message,
            log_path=poster_log_root / f"attempt_{attempt}.json",
            poster_id=poster_id,
            stage=f"evaluation_{variant}",
            attempt=attempt,
        )
        try:
            if returncode != 0:
                raise RuntimeError(f"codex exec returned {returncode}")
            if not last_message.is_file():
                raise RuntimeError("missing --output-last-message")
            result = common.read_json(last_message)
            common.validate_evaluation(poster, result)
            result["metadata"] = {
                "target_variant": variant,
                "target_image": str(image_path.resolve()),
                "model": common.MODEL,
                "reasoning_effort": common.REASONING_EFFORT,
                "session_policy": "fresh ephemeral codex exec; never resumed",
                "attempt": attempt,
            }
            common.write_json(result_path, result)
            return {
                "poster_id": poster_id,
                "status": "completed",
                "missing_count": result["missing_count"],
                "attempt": attempt,
                "elapsed_seconds": elapsed,
            }
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            common.write_json(
                poster_log_root / f"attempt_{attempt}_error.json",
                {"error": errors[-1]},
            )
    raise RuntimeError(f"poster {poster_id} failed: {'; '.join(errors)}")


def aggregate(posters: list[dict[str, Any]], variant: str, stage_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    suspect_units: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for poster in posters:
        result = common.read_json(stage_root / "results" / f"{poster['poster_id']}.json")
        common.validate_evaluation(poster, result)
        missing_assessments = [
            item for item in result["assessments"] if item["verdict"] == "NOT_COVERED"
        ]
        rows.append(
            {
                "poster_id": poster["poster_id"],
                "title": poster["title"],
                "covered_count": 7 - len(missing_assessments),
                "missing_count": len(missing_assessments),
                "missing_unit_ids": [item["unit_id"] for item in missing_assessments],
                "result_file": f"results/{poster['poster_id']}.json",
            }
        )
        records.append(
            {
                "poster_id": poster["poster_id"],
                "status": "completed",
                "attempt": result.get("metadata", {}).get("attempt"),
            }
        )
        if variant == "official":
            unit_map = {unit["unit_id"]: unit for unit in poster["information_units"]}
            for assessment in missing_assessments:
                suspect_units.append(
                    {
                        "poster_id": poster["poster_id"],
                        "title": poster["title"],
                        "unit": unit_map[assessment["unit_id"]],
                        "judge_evidence": assessment["evidence"],
                        "judge_rationale": assessment["rationale"],
                    }
                )

    total_missing = sum(row["missing_count"] for row in rows)
    summary = {
        "experiment": (
            "official-poster self-coverage, posters 10-99"
            if variant == "official"
            else f"{variant.upper()} generated-poster information coverage, posters 10-99"
        ),
        "variant": variant,
        "model": common.MODEL,
        "reasoning_effort": common.REASONING_EFFORT,
        "session_policy": "one fresh ephemeral codex exec invocation per poster; no resume",
        "poster_count": 90,
        "reference_units_total": 630,
        "covered_total": 630 - total_missing,
        "missing_total": total_missing,
        "coverage_rate": (630 - total_missing) / 630,
        "complete_coverage_posters": sum(row["missing_count"] == 0 for row in rows),
        "suspect_reference_units": suspect_units if variant == "official" else [],
        "posters": rows,
        "run_records": records,
    }
    common.write_json(stage_root / "summary.json", summary)

    label = "Official" if variant == "official" else variant.upper()
    lines = [
        f"# {label} information-coverage results, posters 10-99",
        "",
        f"- Model: `{common.MODEL}`",
        f"- Reasoning effort: `{common.REASONING_EFFORT}`",
        "- Session policy: one fresh `codex exec --ephemeral` invocation per poster; no resume",
        f"- Coverage: {summary['covered_total']} / 630 ({summary['coverage_rate']:.2%})",
        f"- Missing: {summary['missing_total']} / 630",
        f"- Posters with all seven units covered: {summary['complete_coverage_posters']} / 90",
        "",
        "| ID | Paper | Covered | Missing | Missing unit IDs |",
        "|---:|---|---:|---:|---|",
    ]
    for row in rows:
        title = row["title"].replace("|", "\\|")
        missing = ", ".join(row["missing_unit_ids"]) or "none"
        lines.append(
            f"| {row['poster_id']} | {title} | {row['covered_count']} | "
            f"{row['missing_count']} | {missing} |"
        )
    if suspect_units:
        lines.extend(["", "## Suspect reference units", ""])
        for item in suspect_units:
            lines.extend(
                [
                    f"### {item['unit']['unit_id']} — {item['unit']['topic']}",
                    "",
                    f"- Statement: {item['unit']['statement']}",
                    f"- Evidence: {item['judge_evidence']}",
                    f"- Rationale: {item['judge_rationale']}",
                    "",
                ]
            )
    (stage_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("official", "rs", "whh"), required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int)
    args = parser.parse_args()
    all_posters = load_posters()
    selected_ids = set(args.ids) if args.ids else set(range(10, 100))
    if any(value < 10 or value > 99 for value in selected_ids):
        raise ValueError("--ids values must be between 10 and 99")
    posters = [item for item in all_posters if int(item["poster_id"]) in selected_ids]

    stage_root = common.BENCH_ROOT / "evaluations" / args.variant
    result_root = stage_root / "results"
    log_root = stage_root / "logs"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    failure_path = stage_root / "failures.json"
    if args.force and failure_path.exists():
        failure_path.unlink()

    codex_prefix = common.resolve_codex()
    template = PROMPTS[args.variant].read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one,
                codex_prefix,
                template,
                poster,
                args.variant,
                result_root,
                log_root,
                args.force,
            ): poster
            for poster in posters
        }
        for future in as_completed(futures):
            poster = futures[future]
            try:
                record = future.result()
                records.append(record)
                print(
                    f"[{poster['poster_id']}] {record['status']} "
                    f"missing={record['missing_count']} attempt={record.get('attempt', '-')}"
                )
            except Exception as exc:
                message = f"{poster['poster_id']}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[{poster['poster_id']}] FAILED: {exc}")
    if failures:
        common.write_json(failure_path, {"failures": failures})
        raise RuntimeError("; ".join(failures))
    aggregate(all_posters, args.variant, stage_root)
    print(f"Wrote {stage_root / 'summary.json'}")


if __name__ == "__main__":
    main()
