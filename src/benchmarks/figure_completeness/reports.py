"""All-GT binary rates, paired diagnostics, per-Figure CSV and inspection pages."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from .data import read_json
from .runner import atomic_json, existing_result, utc_now


def summarize_subset(figures: list[dict[str, Any]], results: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    methods = list(figures[0]["predictions"]) if figures else []
    summary = {}
    for method in methods:
        rows = [results.get((method, figure["gt_key"]), {}) for figure in figures]
        judged = [row for row in rows if row.get("status") == "judged"]
        missing = sum(row.get("status") == "missing_detection" for row in rows)
        complete = sum(row.get("complete") == 1 for row in judged)
        incomplete = sum(row.get("complete") == 0 for row in judged)
        errors = sum(row.get("status") == "execution_error" for row in rows)
        pending = len(rows) - len(judged) - missing - errors
        finished = pending == 0 and errors == 0
        summary[method] = {
            "gt_count": len(figures),
            "matched_count": sum(figure["predictions"][method] is not None for figure in figures),
            "complete_count": complete,
            "vlm_incomplete_count": incomplete,
            "whole_figure_missing_count": missing,
            "zero_count": incomplete + missing,
            "judged_count": len(judged),
            "execution_error_count": errors,
            "pending_count": pending,
            "complete_rate": complete / len(figures) if finished and figures else None,
            "finished": finished,
        }
    return {"methods": summary, "finished": bool(summary) and all(row["finished"] for row in summary.values())}


def write_reports(output_root: Path, experiment: dict[str, Any], phase: str, *, gallery: bool = True) -> dict[str, Any]:
    catalog = experiment["identity"]["catalog"]
    figures = catalog["figures"]
    pilot_keys = set(experiment["identity"]["pilot_gt_keys"])
    methods = list(catalog["sources"])
    results = {}
    csv_rows = []
    attempts = []
    for figure in figures:
        for method in methods:
            result = existing_result(output_root, experiment, method, figure)
            if result:
                results[(method, figure["gt_key"])] = result
                attempts.extend(result.get("attempts", []))
            prediction = figure["predictions"][method]
            csv_rows.append({
                "method": method, "gt_key": figure["gt_key"],
                "paper_index": figure["paper_index"], "paper_title": figure["paper_title"],
                "figure_id": figure["figure_id"], "page": figure["page"],
                "pilot": int(figure["gt_key"] in pilot_keys),
                "complete": result.get("complete") if result else None,
                "status": result["status"] if result else "pending",
                "component_count": prediction["component_count"] if prediction else 0,
                "gt_bbox": str(figure["gt_bbox"]),
                "prediction_bbox": str(prediction["bbox"]) if prediction else "",
                "attempt_count": result.get("attempt_count", 0) if result else 0,
                "error": result.get("error") if result else None,
            })
    backend_records = [record for attempt in attempts for record in attempt.get("backend_records", [])]
    known_calls = sum(bool(attempt.get("usage_available")) for attempt in attempts)
    report = {
        "protocol": experiment["identity"]["protocol"],
        "experiment_fingerprint": experiment["fingerprint"],
        "updated_at": utc_now(), "last_phase": phase,
        "model": experiment["identity"]["model"], "effort": experiment["identity"]["effort"],
        "full": summarize_subset(figures, results),
        "pilot": summarize_subset([row for row in figures if row["gt_key"] in pilot_keys], results),
        "common_matched": summarize_subset([
            row for row in figures if all(value is not None for value in row["predictions"].values())
        ], results),
        "usage": {
            "reserved_attempts": len(attempts),
            "backend_calls_recorded": len(backend_records),
            "calls_without_usage": len(attempts) - known_calls,
            "accepted_calls": sum(attempt["status"] == "accepted" for attempt in attempts),
            "failed_attempts": sum(attempt["status"] != "accepted" for attempt in attempts),
            "input_tokens": sum(record["input_tokens"] for record in backend_records),
            "cached_input_tokens": sum(record["cached_input_tokens"] for record in backend_records),
            "output_tokens": sum(record["output_tokens"] for record in backend_records),
            "call_seconds": round(sum(record["duration_seconds"] for record in backend_records), 3),
            "tool_event_count": sum(record.get("tool_event_count", 0) for record in backend_records),
            "note": "Token totals include known usage only; unknown usage is not assumed free. No invoice claim.",
        },
    }
    atomic_json(output_root / "summary.json", report)
    temporary = output_root / ".per_figure.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    temporary.replace(output_root / "per_figure.csv")
    lines = [
        "# Figure completeness", "",
        f"Model: `{report['model']}`; effort: `{report['effort']}`.", "",
        "1 = all reference Figure content preserved; 0 = partial content loss OR a whole-Figure miss.",
        "Rates use every GT Figure as denominator. Unfinished/error calls do not become semantic zeros;",
        "a final rate is withheld until the corresponding subset has no pending/error entries.", "",
        "The saved Figure-only GT ownership/grouping is inherited unchanged. This is frozen-bbox",
        "completeness, not native PNG export fidelity or free-detection attribution accuracy.", "",
    ]
    for subset in ("pilot", "full", "common_matched"):
        lines += [f"## {subset}", "", "| Method | GT | Complete (1) | Partial loss (0) | Whole miss (0) | Pending | Errors | Complete rate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, row in report[subset]["methods"].items():
            rate = "unfinished" if row["complete_rate"] is None else f"{100 * row['complete_rate']:.2f}%"
            lines.append(f"| {method} | {row['gt_count']} | {row['complete_count']} | {row['vlm_incomplete_count']} | "
                         f"{row['whole_figure_missing_count']} | {row['pending_count']} | {row['execution_error_count']} | {rate} |")
        lines.append("")
    lines += [f"Known backend calls: {len(backend_records)}; failed attempts: {report['usage']['failed_attempts']}; "
              f"calls without usage: {report['usage']['calls_without_usage']}.", "",
              "The 100-paper ours input is the 2026-09-08 validated single-pass batch promoted to mainline;",
              "promotion's generic backend-introduction wording was only live-smoke-tested on one paper.", ""]
    (output_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if gallery:
        selected = [row for row in figures if phase == "full" or row["gt_key"] in pilot_keys]
        _write_gallery(output_root / f"review_{phase}.html", output_root, selected, results, methods)
    return report


def _write_gallery(path: Path, output_root: Path, figures: list[dict[str, Any]], results: dict[Any, Any], methods: list[str]) -> None:
    pieces = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Figure completeness review</title>",
              "<style>body{font:16px sans-serif;margin:24px;background:#eee}section{background:white;padding:16px;margin:24px 0}"
              ".images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}img{width:100%;height:auto}"
              "h3{margin:8px 0}a{color:#0645ad}</style><h1>Frozen-bbox completeness review</h1>",
              "<p>1 = complete; 0 = missing. Click an image to inspect its native resolution. Scores are visible only on this review page.</p>"]
    for figure in figures:
        pieces.append(f"<section id='{figure['gt_key']}'><h2>{html.escape(figure['paper_title'])} — "
                      f"{html.escape(figure['label'])}, page {figure['page']}</h2><div class='images'>")
        for name in ["reference", *methods]:
            image = output_root / "inputs" / figure["gt_key"] / f"{name}.png"
            result = results.get((name, figure["gt_key"]), {})
            score = result.get("complete")
            label = "GT reference" if name == "reference" else f"{name}: {score if score is not None else 'pending'}"
            pieces.append(f"<div><h3>{html.escape(label)}</h3>")
            if image.is_file():
                relative = html.escape(str(image.relative_to(output_root)), quote=True)
                pieces.append(f"<a href='{relative}'><img loading='lazy' src='{relative}'></a>")
            else:
                pieces.append(f"<p>{html.escape(str(result.get('status', 'image not prepared')))}</p>")
            pieces.append("</div>")
        pieces.append("</div></section>")
    pieces.append("</html>")
    path.write_text("\n".join(pieces), encoding="utf-8")
