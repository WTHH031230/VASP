from __future__ import annotations

import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
import benchmark_common as common


REFERENCE_PATH = common.BENCH_ROOT / "reference_information_units_full100.json"
ALIGNMENT_FLAGS_PATH = common.BENCH_ROOT / "data_alignment_flags.json"
def result_path(variant: str, poster_id: int) -> Path:
    return common.BENCH_ROOT / "evaluations" / variant / "results" / f"{poster_id}.json"


def exact_two_sided_binomial_p(successes: int, failures: int) -> float:
    n = successes + failures
    if n == 0:
        return 1.0
    k = min(successes, failures)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * lower)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_mean_ci(values: list[int], samples: int = 20000) -> list[float]:
    rng = random.Random(20260805)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    means.sort()
    return [percentile(means, 0.025), percentile(means, 0.975)]


def variant_metrics(missing_counts: list[int]) -> dict[str, Any]:
    total_missing = sum(missing_counts)
    return {
        "posters": len(missing_counts),
        "reference_units": len(missing_counts) * 7,
        "covered_units": len(missing_counts) * 7 - total_missing,
        "missing_units": total_missing,
        "coverage_rate": 1 - total_missing / (len(missing_counts) * 7),
        "mean_missing_per_poster": statistics.mean(missing_counts),
        "median_missing_per_poster": statistics.median(missing_counts),
        "stdev_missing_per_poster": statistics.pstdev(missing_counts),
        "complete_coverage_posters": sum(value == 0 for value in missing_counts),
        "complete_coverage_rate": sum(value == 0 for value in missing_counts)
        / len(missing_counts),
        "posters_missing_at_least_four": sum(value >= 4 for value in missing_counts),
        "maximum_missing": max(missing_counts),
        "missing_count_distribution": {
            str(value): missing_counts.count(value) for value in range(8)
        },
    }


def validate_official_selfcheck() -> dict[str, Any]:
    missing = 0
    count = 0
    for poster_id in range(10):
        path = (
            common.BENCH_ROOT
            / "evaluations"
            / "official"
            / "results"
            / f"{poster_id}.json"
        )
        result = common.read_json(path)
        missing += result["missing_count"]
        count += len(result["assessments"])
    for poster_id in range(10, 100):
        path = (
            common.BENCH_ROOT
            / "evaluations"
            / "official"
            / "results"
            / f"{poster_id}.json"
        )
        result = common.read_json(path)
        missing += result["missing_count"]
        count += len(result["assessments"])
    if count != 700 or missing != 0:
        raise RuntimeError(f"official selfcheck mismatch: count={count}, missing={missing}")
    return {"covered": 700, "total": 700, "coverage_rate": 1.0}


def main() -> None:
    reference = common.read_json(REFERENCE_PATH)
    posters = reference["posters"]
    if [item["poster_id"] for item in posters] != [str(i) for i in range(100)]:
        raise RuntimeError("full reference IDs are not 0-99")

    rows: list[dict[str, Any]] = []
    type_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"units": 0, "rs_missing": 0, "whh_missing": 0}
    )
    unit_contingency = Counter()
    for poster in posters:
        poster_id = int(poster["poster_id"])
        rs_result = common.read_json(result_path("rs", poster_id))
        whh_result = common.read_json(result_path("whh", poster_id))
        common.validate_evaluation(poster, rs_result)
        common.validate_evaluation(poster, whh_result)
        rs_missing_ids = {
            item["unit_id"]
            for item in rs_result["assessments"]
            if item["verdict"] == "NOT_COVERED"
        }
        whh_missing_ids = {
            item["unit_id"]
            for item in whh_result["assessments"]
            if item["verdict"] == "NOT_COVERED"
        }
        rs_missing = len(rs_missing_ids)
        whh_missing = len(whh_missing_ids)
        difference = rs_missing - whh_missing
        rows.append(
            {
                "poster_id": str(poster_id),
                "title": poster["title"],
                "rs_covered": 7 - rs_missing,
                "rs_missing": rs_missing,
                "rs_missing_unit_ids": sorted(rs_missing_ids),
                "whh_covered": 7 - whh_missing,
                "whh_missing": whh_missing,
                "whh_missing_unit_ids": sorted(whh_missing_ids),
                "rs_minus_whh_missing": difference,
                "winner": "WHH" if difference > 0 else "RS" if difference < 0 else "tie",
                "missing_in_both": sorted(rs_missing_ids & whh_missing_ids),
                "missing_only_rs": sorted(rs_missing_ids - whh_missing_ids),
                "missing_only_whh": sorted(whh_missing_ids - rs_missing_ids),
                "rs_result": str(result_path("rs", poster_id)),
                "whh_result": str(result_path("whh", poster_id)),
            }
        )
        unit_type = {
            unit["unit_id"]: unit["information_type"].strip().lower()
            for unit in poster["information_units"]
        }
        for unit_id, info_type in unit_type.items():
            type_counts[info_type]["units"] += 1
            rs_is_missing = unit_id in rs_missing_ids
            whh_is_missing = unit_id in whh_missing_ids
            type_counts[info_type]["rs_missing"] += int(rs_is_missing)
            type_counts[info_type]["whh_missing"] += int(whh_is_missing)
            if rs_is_missing and whh_is_missing:
                unit_contingency["missing_both"] += 1
            elif rs_is_missing:
                unit_contingency["missing_only_rs"] += 1
            elif whh_is_missing:
                unit_contingency["missing_only_whh"] += 1
            else:
                unit_contingency["covered_both"] += 1

    rs_counts = [row["rs_missing"] for row in rows]
    whh_counts = [row["whh_missing"] for row in rows]
    paired_differences = [row["rs_minus_whh_missing"] for row in rows]
    whh_wins = sum(value > 0 for value in paired_differences)
    rs_wins = sum(value < 0 for value in paired_differences)
    ties = sum(value == 0 for value in paired_differences)
    type_rows: list[dict[str, Any]] = []
    for info_type, counts in sorted(
        type_counts.items(), key=lambda item: (-item[1]["units"], item[0])
    ):
        type_rows.append(
            {
                "information_type": info_type,
                "units": counts["units"],
                "rs_missing": counts["rs_missing"],
                "rs_coverage_rate": 1 - counts["rs_missing"] / counts["units"],
                "whh_missing": counts["whh_missing"],
                "whh_coverage_rate": 1 - counts["whh_missing"] / counts["units"],
                "whh_minus_rs_coverage_points": (
                    counts["rs_missing"] - counts["whh_missing"]
                )
                / counts["units"],
            }
        )

    alignment_data = common.read_json(ALIGNMENT_FLAGS_PATH)
    alignment_flags = alignment_data.get("flags", [])
    resolved_alignment_flags = alignment_data.get("resolved_flags", [])
    flagged_ids = {item["poster_id"] for item in alignment_flags}
    sensitivity_rows = [row for row in rows if row["poster_id"] not in flagged_ids]
    sensitivity_rs = [row["rs_missing"] for row in sensitivity_rows]
    sensitivity_whh = [row["whh_missing"] for row in sensitivity_rows]
    sensitivity_differences = [
        row["rs_minus_whh_missing"] for row in sensitivity_rows
    ]
    sensitivity_whh_wins = sum(value > 0 for value in sensitivity_differences)
    sensitivity_rs_wins = sum(value < 0 for value in sensitivity_differences)
    if flagged_ids:
        primary_scope = (
            f"{len(sensitivity_rows)} valid paper/official-poster pairs; "
            f"unresolved poster IDs excluded: {', '.join(sorted(flagged_ids, key=int))}"
        )
    else:
        primary_scope = (
            "all 100 supplied paper/official-poster pairs; "
            "no unresolved alignment exclusions"
        )
    summary = {
        "benchmark": "official-poster information coverage, full 100",
        "primary_metric_scope": primary_scope,
        "raw_full100_includes_alignment_anomaly": bool(alignment_flags),
        "model": common.MODEL,
        "reasoning_effort": common.REASONING_EFFORT,
        "reference_units": 700,
        "official_selfcheck": validate_official_selfcheck(),
        "rs": variant_metrics(rs_counts),
        "whh": variant_metrics(whh_counts),
        "paired_comparison": {
            "mean_rs_minus_whh_missing": statistics.mean(paired_differences),
            "median_rs_minus_whh_missing": statistics.median(paired_differences),
            "bootstrap_95pct_ci_for_mean_rs_minus_whh_missing": bootstrap_mean_ci(
                paired_differences
            ),
            "whh_better_posters": whh_wins,
            "rs_better_posters": rs_wins,
            "ties": ties,
            "two_sided_exact_sign_test_p_non_ties": exact_two_sided_binomial_p(
                whh_wins, rs_wins
            ),
        },
        "unit_level_contingency": {
            **unit_contingency,
            "two_sided_exact_mcnemar_p": exact_two_sided_binomial_p(
                unit_contingency["missing_only_rs"],
                unit_contingency["missing_only_whh"],
            ),
        },
        "data_alignment_flags": alignment_flags,
        "resolved_alignment_flags": resolved_alignment_flags,
        "sensitivity_excluding_alignment_flags": {
            "excluded_poster_ids": sorted(flagged_ids, key=int),
            "rs": variant_metrics(sensitivity_rs),
            "whh": variant_metrics(sensitivity_whh),
            "mean_rs_minus_whh_missing": statistics.mean(sensitivity_differences),
            "bootstrap_95pct_ci_for_mean_rs_minus_whh_missing": bootstrap_mean_ci(
                sensitivity_differences
            ),
            "whh_better_posters": sensitivity_whh_wins,
            "rs_better_posters": sensitivity_rs_wins,
            "ties": sum(value == 0 for value in sensitivity_differences),
            "two_sided_exact_sign_test_p_non_ties": exact_two_sided_binomial_p(
                sensitivity_whh_wins, sensitivity_rs_wins
            ),
        },
        "information_type_metrics": type_rows,
        "posters": rows,
    }
    summary["primary_valid_pair_metrics"] = summary[
        "sensitivity_excluding_alignment_flags"
    ]
    common.write_json(common.BENCH_ROOT / "comparison_summary.json", summary)

    with (common.BENCH_ROOT / "per_poster_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "poster_id",
                "title",
                "rs_missing",
                "whh_missing",
                "rs_minus_whh_missing",
                "winner",
                "rs_missing_unit_ids",
                "whh_missing_unit_ids",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["poster_id"],
                    row["title"],
                    row["rs_missing"],
                    row["whh_missing"],
                    row["rs_minus_whh_missing"],
                    row["winner"],
                    ",".join(row["rs_missing_unit_ids"]),
                    ",".join(row["whh_missing_unit_ids"]),
                ]
            )

    with (common.BENCH_ROOT / "per_type_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(type_rows[0]))
        writer.writeheader()
        writer.writerows(type_rows)

    rs = summary["rs"]
    whh = summary["whh"]
    paired = summary["paired_comparison"]
    contingency = summary["unit_level_contingency"]
    sensitivity = summary["sensitivity_excluding_alignment_flags"]
    if flagged_ids:
        alignment_line = (
            "- Unresolved source-alignment flags exclude poster IDs "
            f"{', '.join(sorted(flagged_ids, key=int))} from primary metrics"
        )
        primary_note = (
            "- Excluded unresolved poster IDs: "
            f"{', '.join(sorted(flagged_ids, key=int))}."
        )
    else:
        resolved_ids = sorted(
            (item["poster_id"] for item in resolved_alignment_flags), key=int
        )
        resolved_suffix = (
            f"; corrected and reevaluated poster IDs: {', '.join(resolved_ids)}"
            if resolved_ids
            else ""
        )
        alignment_line = (
            "- Data alignment: all 100 paper/official-poster pairs are valid"
            f"{resolved_suffix}"
        )
        primary_note = "- No unresolved source-alignment exclusions; all 100 pairs are included."
    lines = [
        "# Full-100 poster information-coverage comparison",
        "",
        f"- Judge: `{common.MODEL}` with `{common.REASONING_EFFORT}` reasoning",
        "- Session policy: one fresh `codex exec --ephemeral` invocation per poster and method; no resume",
        "- Reference: 7 official-poster information units per supplied poster, 700 raw units",
        "- Official-poster self-check: 700 / 700 units covered",
        alignment_line,
        "",
        "## Full 100-pair metrics",
        "",
        "| Metric | RS | WHH |",
        "|---|---:|---:|",
        f"| Covered units | {rs['covered_units']} / 700 | {whh['covered_units']} / 700 |",
        f"| Coverage rate | {rs['coverage_rate']:.2%} | {whh['coverage_rate']:.2%} |",
        f"| Missing units | {rs['missing_units']} | {whh['missing_units']} |",
        f"| Mean missing per poster | {rs['mean_missing_per_poster']:.3f} | {whh['mean_missing_per_poster']:.3f} |",
        f"| Median missing per poster | {rs['median_missing_per_poster']:.1f} | {whh['median_missing_per_poster']:.1f} |",
        f"| Complete-coverage posters | {rs['complete_coverage_posters']} / 100 | {whh['complete_coverage_posters']} / 100 |",
        f"| Posters missing >=4 units | {rs['posters_missing_at_least_four']} | {whh['posters_missing_at_least_four']} |",
        "",
        "## Paired comparison",
        "",
        f"- WHH has fewer missing units on {paired['whh_better_posters']} papers; RS on {paired['rs_better_posters']}; {paired['ties']} ties.",
        f"- Mean paired difference `RS missing - WHH missing`: {paired['mean_rs_minus_whh_missing']:.3f} units/paper.",
        f"- Paired bootstrap 95% CI: [{paired['bootstrap_95pct_ci_for_mean_rs_minus_whh_missing'][0]:.3f}, {paired['bootstrap_95pct_ci_for_mean_rs_minus_whh_missing'][1]:.3f}].",
        f"- Exact two-sided sign-test p-value across non-tied papers: {paired['two_sided_exact_sign_test_p_non_ties']:.6g}.",
        f"- Unit-level discordance: RS-only missing {contingency['missing_only_rs']}, WHH-only missing {contingency['missing_only_whh']}, missing in both {contingency['missing_both']}; exact McNemar p={contingency['two_sided_exact_mcnemar_p']:.6g}.",
        "",
        f"## Primary valid-pair metrics ({len(sensitivity_rows)} papers)",
        "",
        primary_note,
        f"- RS coverage {sensitivity['rs']['coverage_rate']:.2%} ({sensitivity['rs']['missing_units']} missing / {sensitivity['rs']['reference_units']}); WHH coverage {sensitivity['whh']['coverage_rate']:.2%} ({sensitivity['whh']['missing_units']} missing / {sensitivity['whh']['reference_units']}).",
        f"- WHH wins {sensitivity['whh_better_posters']} papers, RS wins {sensitivity['rs_better_posters']}, with {sensitivity['ties']} ties; mean paired advantage is {sensitivity['mean_rs_minus_whh_missing']:.3f} fewer missing units for WHH.",
        "",
        "## Missing-count distribution",
        "",
        "| Missing units in a poster | RS papers | WHH papers |",
        "|---:|---:|---:|",
    ]
    for value in range(8):
        lines.append(
            f"| {value} | {rs['missing_count_distribution'][str(value)]} | "
            f"{whh['missing_count_distribution'][str(value)]} |"
        )
    lines.extend(
        [
            "",
            "## Per-paper results",
            "",
            "| ID | Paper | RS missing | WHH missing | RS-WHH | Winner |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        title = row["title"].replace("|", "\\|")
        difference = row["rs_minus_whh_missing"]
        difference_text = f"+{difference}" if difference > 0 else str(difference)
        lines.append(
            f"| {row['poster_id']} | {title} | {row['rs_missing']} | "
            f"{row['whh_missing']} | {difference_text} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "Positive `RS-WHH` means WHH omitted fewer reference units. Statistical tests describe this single deterministic judge run and do not measure run-to-run judge variance.",
        ]
    )
    (common.BENCH_ROOT / "comparison_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {common.BENCH_ROOT / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
