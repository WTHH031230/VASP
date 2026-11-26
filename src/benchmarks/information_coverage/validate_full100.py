from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common


def evaluation_paths(variant: str, poster_id: int) -> tuple[Path, Path]:
    root = common.BENCH_ROOT / "evaluations" / variant
    return root / "results" / f"{poster_id}.json", root / "logs" / str(poster_id)


def validate_log(log_path: Path, expected_stage_fragment: str | None = None) -> None:
    log = common.read_json(log_path)
    if log.get("returncode") != 0:
        raise RuntimeError(f"nonzero return code: {log_path}")
    if log.get("model") != common.MODEL:
        raise RuntimeError(f"model mismatch: {log_path}")
    if log.get("reasoning_effort") != common.REASONING_EFFORT:
        raise RuntimeError(f"reasoning mismatch: {log_path}")
    command = log.get("command", [])
    if "--ephemeral" not in command or "resume" in command:
        raise RuntimeError(f"session-policy mismatch: {log_path}")
    if expected_stage_fragment is not None:
        stage = str(log.get("stage", ""))
        if expected_stage_fragment not in stage:
            raise RuntimeError(f"stage mismatch: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-baseline",
        action="store_true",
        help="Also require the archived RS=158 and WHH=125 missing-unit totals.",
    )
    args = parser.parse_args()
    reference = common.read_json(
        common.BENCH_ROOT / "reference_information_units_full100.json"
    )
    posters = reference["posters"]
    if len(posters) != 100:
        raise RuntimeError("reference poster count is not 100")
    poster_map: dict[int, dict[str, Any]] = {}
    for expected_id, poster in enumerate(posters):
        if poster["poster_id"] != str(expected_id):
            raise RuntimeError(f"reference ID mismatch at {expected_id}")
        common.validate_units(poster, poster)
        poster_map[expected_id] = poster

    extraction_sessions = 0
    extraction_logs_unavailable: list[int] = []
    for poster_id in range(100):
        result_path = (
            common.BENCH_ROOT
            / "reference_extraction"
            / "results"
            / f"{poster_id}.json"
        )
        result = common.read_json(result_path)
        common.validate_units(poster_map[poster_id], result)
        attempt = result.get("metadata", {}).get("attempt")
        log_path = (
            common.BENCH_ROOT
            / "reference_extraction"
            / "logs"
            / str(poster_id)
            / f"attempt_{attempt}.json"
        )
        if isinstance(attempt, int) and log_path.is_file():
            validate_log(log_path, "reference_extraction")
            extraction_sessions += 1
        else:
            extraction_logs_unavailable.append(poster_id)

    evaluation_sessions = 0
    variant_missing: dict[str, int] = {"official": 0, "rs": 0, "whh": 0}
    for variant in ("official", "rs", "whh"):
        for poster_id in range(100):
            result_path, log_root = evaluation_paths(variant, poster_id)
            result = common.read_json(result_path)
            common.validate_evaluation(poster_map[poster_id], result)
            variant_missing[variant] += result["missing_count"]
            attempt = int(result.get("metadata", {}).get("attempt", 1))
            validate_log(log_root / f"attempt_{attempt}.json")
            evaluation_sessions += 1

    if variant_missing["official"] != 0:
        raise RuntimeError(
            f"official self-check contains missing units: {variant_missing['official']}"
        )
    if args.expect_baseline and variant_missing != {
        "official": 0,
        "rs": 158,
        "whh": 125,
    }:
        raise RuntimeError(f"unexpected missing totals: {variant_missing}")

    comparison = common.read_json(common.BENCH_ROOT / "comparison_summary.json")
    if comparison["rs"]["missing_units"] != variant_missing["rs"]:
        raise RuntimeError("RS comparison total mismatch")
    if comparison["whh"]["missing_units"] != variant_missing["whh"]:
        raise RuntimeError("WHH comparison total mismatch")
    if comparison["official_selfcheck"]["covered"] != 700:
        raise RuntimeError("official selfcheck summary mismatch")

    failure_files = [
        path
        for path in common.BENCH_ROOT.rglob("failures.json")
        if path.is_file()
    ]
    report = {
        "status": "PASS",
        "reference_posters": 100,
        "reference_units": 700,
        "official_selfcheck_missing": variant_missing["official"],
        "rs_missing": variant_missing["rs"],
        "whh_missing": variant_missing["whh"],
        "validated_reference_extraction_sessions": extraction_sessions,
        "reference_extraction_logs_unavailable_for_ids": extraction_logs_unavailable,
        "validated_evaluation_sessions": evaluation_sessions,
        "validated_successful_sessions_total": extraction_sessions
        + evaluation_sessions,
        "model": common.MODEL,
        "reasoning_effort": common.REASONING_EFFORT,
        "baseline_expectation_checked": args.expect_baseline,
        "session_policy": "all validated logs contain --ephemeral and no resume",
        "failure_files": [str(path) for path in failure_files],
    }
    if failure_files:
        raise RuntimeError(f"stale failure files found: {failure_files}")
    common.write_json(common.BENCH_ROOT / "validation_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
