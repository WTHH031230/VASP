#!/usr/bin/env python3
"""Aggregate VisualBench reports with equal group/question macro averaging.

The primary score is deliberately not rubric-weighted: each of the three
groups contributes one third, and questions within a group contribute equally.
Only PASS and FAIL are evaluable; UNCERTAIN and NOT_APPLICABLE are retained in
counts and coverage statistics.

Variant (researcher decision 2026-09-23): every poster is scored. A group with
no evaluable judgement (all UNCERTAIN / NOT_APPLICABLE, e.g. a poster without
any figure) contributes a pass rate of 0 instead of leaving the poster unscored.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUESTION_ORDER = ["AE04", "AE13", "IS01", "ENH01", "DIST01", "COL01",
                  "EN05", "EN07", "AL01", "AL05", "DEN01", "WS01"]
GROUPS = {
    "Figure presentation": ["AE04", "AE13", "IS01"],
    "Visual guidance and emphasis": ["ENH01", "DIST01", "COL01", "EN05", "EN07"],
    "Layout and spatial organization": ["AL01", "AL05", "DEN01", "WS01"],
}
VERDICTS = ("PASS", "FAIL", "UNCERTAIN", "NOT_APPLICABLE")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method-dir", action="append", required=True,
                   metavar="NAME=PATH", help="repeat once per method")
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-md", type=Path, required=True)
    return p.parse_args()


def parse_method_dirs(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--method-dir must be NAME=PATH: {value}")
        name, raw = value.split("=", 1)
        if not name or name in out:
            raise ValueError(f"duplicate/empty method name: {name!r}")
        out[name] = Path(raw).resolve()
    if len(out) < 2:
        raise ValueError("at least two methods are required")
    return out


def flatten(report: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for block in report.get("dimensions", {}).values():
        for item in block.get("results", []):
            qid = str(item.get("id", ""))
            if qid in result:
                raise ValueError(f"duplicate question {qid} in {path}")
            result[qid] = item
    return result


def load_method(directory: Path) -> dict[str, dict[str, Any]]:
    files = sorted(directory.glob("*.json"), key=lambda p: p.stem.casefold())
    if not files:
        raise FileNotFoundError(f"no JSON reports in {directory}")
    records: dict[str, dict[str, Any]] = {}
    for path in files:
        title = path.stem
        if title in records:
            raise ValueError(f"duplicate title {title!r} in {directory}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise ValueError(f"incomplete report (status={report.get('status')!r}): {path}")
        if report.get("errors"):
            raise ValueError(f"report contains errors: {path}")
        results = flatten(report, path)
        if set(results) != set(QUESTION_ORDER):
            raise ValueError(f"unexpected question IDs in {path}: {sorted(results)}")
        for qid, item in results.items():
            if item.get("verdict") not in VERDICTS:
                raise ValueError(f"invalid verdict for {qid} in {path}: {item.get('verdict')}")
        records[title] = {"path": str(path), "report": report, "results": results}
    return records


def item_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(i["verdict"]) for i in items)
    evaluable = counts["PASS"] + counts["FAIL"]
    return {
        "judgement_count": len(items),
        "verdict_counts": {v: counts[v] for v in VERDICTS},
        "evaluable_count": evaluable,
        "coverage": evaluable / len(items) if items else None,
        "pass_rate": counts["PASS"] / evaluable if evaluable else None,
    }


def poster_stats(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    rates: list[float] = []
    for name, qids in GROUPS.items():
        stat = item_stats([results[q] for q in qids])
        if stat["pass_rate"] is None:
            stat["pass_rate"] = 0.0
            stat["missing_scored_as_zero"] = True
        groups[name] = stat
        rates.append(stat["pass_rate"])
    complete = len(rates) == len(GROUPS)
    return {
        "groups": groups,
        "score": statistics.mean(rates) if complete else None,
        "scored": complete,
        "missing_groups": [n for n, s in groups.items() if s["pass_rate"] is None],
    }


def method_stats(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    posters = {title: poster_stats(rec["results"]) for title, rec in records.items()}
    scores = [p["score"] for p in posters.values() if p["score"] is not None]
    all_items = [i for rec in records.values() for i in rec["results"].values()]
    pooled = item_stats(all_items)
    return {
        "poster_count": len(records), "scored_poster_count": len(scores),
        "unscored_poster_count": len(records) - len(scores),
        "mean_poster_equal_group_score": statistics.mean(scores) if scores else None,
        "median_poster_equal_group_score": statistics.median(scores) if scores else None,
        "poster_score_coverage": len(scores) / len(records) if records else None,
        "pooled_verdicts": pooled,
        "posters": posters,
    }


def build_markdown(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    lines = ["# VisionBench equal-group comparison", "",
             "Primary score = mean over posters of (three group pass rates averaged equally).",
             "Within each group: PASS/(PASS+FAIL); UNCERTAIN and NOT_APPLICABLE are excluded from the rate denominator but included in coverage.", "",
             "## Method summary", "",
             "| Method | Poster macro score | Scored posters | Overall judgment coverage | PASS | FAIL | UNCERTAIN | NOT_APPLICABLE | Model / effort |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for m in methods:
        s = summary["method_stats"][m]; p = s["pooled_verdicts"]
        c = p["verdict_counts"]
        score = "—" if s["mean_poster_equal_group_score"] is None else f"{s['mean_poster_equal_group_score']*100:.1f}%"
        meta = summary["metadata"][m]
        lines.append(f"| `{m}` | {score} | {s['scored_poster_count']}/{s['poster_count']} | {p['coverage']*100:.1f}% | {c['PASS']} | {c['FAIL']} | {c['UNCERTAIN']} | {c['NOT_APPLICABLE']} | `{meta['model']}` / `{meta['reasoning_effort']}` |")
    lines += ["", "## Group rates", "", "| Group | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for group in GROUPS:
        vals = []
        for m in methods:
            stats = [x["groups"][group]["pass_rate"] for x in summary["method_stats"][m]["posters"].values() if x["groups"][group]["pass_rate"] is not None]
            vals.append("—" if not stats else f"{statistics.mean(stats)*100:.1f}%")
        lines.append("| " + group + " | " + " | ".join(vals) + " |")
    if summary["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    cli = args(); dirs = parse_method_dirs(cli.method_dir)
    records = {m: load_method(d) for m, d in dirs.items()}
    titles = {m: set(r) for m, r in records.items()}
    expected = next(iter(titles.values()))
    if any(ts != expected for ts in titles.values()):
        raise ValueError("method title sets differ: " + ", ".join(f"{m}={len(ts)}" for m, ts in titles.items()))
    metadata = {}
    for m, rs in records.items():
        vals = {(str(r["report"].get("model_override")), str(r["report"].get("reasoning_effort_override"))) for r in rs.values()}
        if len(vals) != 1: raise ValueError(f"inconsistent model metadata in {m}: {vals}")
        model, effort = next(iter(vals)); metadata[m] = {"model": model, "reasoning_effort": effort}
    warnings = []
    efforts = {x["reasoning_effort"] for x in metadata.values()}
    if "xhigh" in efforts and "medium" in efforts:
        warnings.append("Old canonical runs use reasoning effort xhigh while new runs use medium; scores are not strictly effort-matched.")
    summary = {"schema_version": "1.0", "created_at": datetime.now(timezone.utc).isoformat(), "methods": list(dirs), "poster_count": len(expected), "question_order": QUESTION_ORDER, "groups": GROUPS, "metadata": metadata, "warnings": warnings, "method_stats": {m: method_stats(records[m]) for m in dirs}}
    cli.output_json.parent.mkdir(parents=True, exist_ok=True); cli.output_md.parent.mkdir(parents=True, exist_ok=True)
    cli.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cli.output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(cli.output_json.resolve()); print(cli.output_md.resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
