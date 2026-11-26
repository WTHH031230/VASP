from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common


FIRST10_RS_ROOT = common.BENCH_ROOT.parent / "first10_rs"
FIRST10_WHH_ROOT = common.BENCH_ROOT.parent / "first10_whh"
REFERENCE_PATH = common.BENCH_ROOT / "reference_information_units_full100.json"
ALIGNMENT_FLAGS_PATH = common.BENCH_ROOT / "data_alignment_flags.json"


SOURCE_ROOTS = {
    "official": FIRST10_RS_ROOT / "official_selfcheck",
    "rs": FIRST10_RS_ROOT,
    "whh": FIRST10_WHH_ROOT,
}


def copy_first10() -> None:
    for variant, source_root in SOURCE_ROOTS.items():
        destination_root = common.BENCH_ROOT / "evaluations" / variant
        result_destination = destination_root / "results"
        log_destination = destination_root / "logs"
        result_destination.mkdir(parents=True, exist_ok=True)
        log_destination.mkdir(parents=True, exist_ok=True)
        for poster_id in range(10):
            source_result = source_root / "results" / f"{poster_id}.json"
            if not source_result.is_file():
                raise FileNotFoundError(source_result)
            shutil.copy2(source_result, result_destination / f"{poster_id}.json")

            source_log = source_root / "logs" / str(poster_id)
            destination_log = log_destination / str(poster_id)
            if not source_log.is_dir():
                raise FileNotFoundError(source_log)
            shutil.copytree(source_log, destination_log, dirs_exist_ok=True)


def create_variant_summary(
    variant: str, posters: list[dict[str, Any]]
) -> dict[str, Any]:
    stage_root = common.BENCH_ROOT / "evaluations" / variant
    rows: list[dict[str, Any]] = []
    for poster in posters:
        poster_id = poster["poster_id"]
        result = common.read_json(stage_root / "results" / f"{poster_id}.json")
        common.validate_evaluation(poster, result)
        missing_ids = [
            item["unit_id"]
            for item in result["assessments"]
            if item["verdict"] == "NOT_COVERED"
        ]
        rows.append(
            {
                "poster_id": poster_id,
                "title": poster["title"],
                "covered_count": 7 - len(missing_ids),
                "missing_count": len(missing_ids),
                "missing_unit_ids": missing_ids,
                "result_file": f"results/{poster_id}.json",
            }
        )
    missing_total = sum(row["missing_count"] for row in rows)
    alignment_flags = common.read_json(ALIGNMENT_FLAGS_PATH).get("flags", [])
    flagged_ids = sorted(
        (item["poster_id"] for item in alignment_flags), key=int
    )
    summary = {
        "experiment": (
            "official-poster self-coverage, full 100"
            if variant == "official"
            else f"{variant.upper()} generated-poster information coverage, full 100"
        ),
        "variant": variant,
        "model": common.MODEL,
        "reasoning_effort": common.REASONING_EFFORT,
        "session_policy": "one fresh ephemeral codex exec invocation per poster; no resume",
        "poster_count": 100,
        "reference_units_total": 700,
        "covered_total": 700 - missing_total,
        "missing_total": missing_total,
        "coverage_rate": (700 - missing_total) / 700,
        "complete_coverage_posters": sum(row["missing_count"] == 0 for row in rows),
        "data_alignment_note": (
            f"Primary paired metrics exclude unresolved poster IDs: {', '.join(flagged_ids)}."
            if variant in {"rs", "whh"} and flagged_ids
            else None
        ),
        "posters": rows,
    }
    common.write_json(stage_root / "summary_full100.json", summary)

    label = "Official" if variant == "official" else variant.upper()
    lines = [
        f"# {label} information-coverage results, full 100",
        "",
        f"- Model: `{common.MODEL}`",
        f"- Reasoning effort: `{common.REASONING_EFFORT}`",
        "- Result and log directories contain poster IDs 0-99",
        f"- Coverage: {summary['covered_total']} / 700 ({summary['coverage_rate']:.2%})",
        f"- Missing: {summary['missing_total']} / 700",
        f"- Complete coverage: {summary['complete_coverage_posters']} / 100 posters",
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
    (stage_root / "summary_full100.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    reference = common.read_json(REFERENCE_PATH)
    posters = reference["posters"]
    if [item["poster_id"] for item in posters] != [str(i) for i in range(100)]:
        raise RuntimeError("Reference IDs are not 0-99")
    copy_first10()
    summaries = {
        variant: create_variant_summary(variant, posters)
        for variant in ("official", "rs", "whh")
    }
    manifest = {
        "status": "complete",
        "policy": "Original first-10 files are preserved; copies are consolidated under full100/evaluations.",
        "reference_file": "reference_information_units_full100.json",
        "poster_ids": "0-99",
        "variants": {
            variant: {
                "result_directory": f"evaluations/{variant}/results",
                "log_directory": f"evaluations/{variant}/logs",
                "result_files": 100,
                "log_directories": 100,
                "summary": f"evaluations/{variant}/summary_full100.json",
                "missing_total": summary["missing_total"],
            }
            for variant, summary in summaries.items()
        },
    }
    common.write_json(common.BENCH_ROOT / "consolidation_manifest.json", manifest)
    readme = """# Full-100 information-coverage benchmark

This directory is the unified entry point for all 100 posters.

- `reference_information_units_full100.json`: reference units for poster IDs 0-99.
- `evaluations/official/results/`: official self-check results for IDs 0-99.
- `evaluations/rs/results/`: RS results for IDs 0-99.
- `evaluations/whh/results/`: WHH results for IDs 0-99.
- Each corresponding `logs/` directory also contains IDs 0-99.
- `comparison_summary.md` and `comparison_summary.json`: paired RS/WHH comparison.
- `validation_report.json`: end-to-end structural and session validation.

The older `first10_rs` and `first10_whh` directories are preserved as historical source artifacts. The copies under this directory are the canonical unified view.
"""
    (common.BENCH_ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
