#!/usr/bin/env python3
"""Summarize EXP-024: PaperQuiz raw accuracy per group, by aspect, and paired.

Accuracy is the upstream per-call value (correct / 50). The per-paper average is
the mean of detail and understanding, as upstream averages them. No density
adjustment (it is not in upstream code). API-equivalent cost uses the project
pricing function (src/pipeline/asset_extraction/llm/codex_exec._usage_record,
model gpt-5.6-sol), cached prompt tokens priced as cached input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(os.environ.get("VASP_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
sys.path.insert(0, str(ROOT))
from src.semparser.llm.codex_exec import _usage_record  # noqa: E402

QTYPES = ("detail", "understanding")
ASPECT_NAMES = {
    "detail": {
        "A": "Title & authorship", "B": "Motivation / problem", "C": "Objectives / hypotheses",
        "D": "Datasets / materials", "E": "Methodology", "F": "Key parameters",
        "G": "Evaluation metrics", "H": "Quantitative results", "I": "Qualitative findings / figures",
        "J": "Comparative / ablation results", "K": "Conclusions / contributions",
        "L": "Limitations / future work", "M": "Definitions of terms",
    },
    "understanding": {
        "A": "Domain & background", "B": "Problem / motivation", "C": "Goal / hypothesis",
        "D": "Contributions / novelty", "E": "Overall methodology", "F": "Principal findings",
        "G": "Qualitative insights", "H": "Implications", "I": "Limitations / future work",
        "J": "Main conclusions",
    },
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int = 10000) -> list[float]:
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--groups", nargs="+", default=["pi_full", "researchstudio"])
    args = parser.parse_args()
    exp = Path(args.experiment)
    rng = np.random.default_rng(20260926)

    per_paper: dict[str, dict[str, dict]] = {}
    summary: dict = {"groups": {}, "paired": {}, "notes": [
        "accuracy = upstream compute_accuracy (correct / 50; NA, NA reference and missing answers are wrong)",
        "average = mean of detail and understanding accuracy per paper",
        "no density adjustment; single reader gpt-5.6-sol medium",
    ]}
    for group in args.groups:
        rows = {}
        aspect_tot = {q: defaultdict(lambda: [0, 0]) for q in QTYPES}
        answer_stats = {q: {"answered": 0, "na": 0, "missing": 0, "extra_keys": 0} for q in QTYPES}
        usage_tot = {"calls": 0, "prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0,
                     "reasoning_tokens": 0, "cost_usd": 0.0, "latency_s": []}
        reasks = 0
        for path in sorted((exp / "results" / group).glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            assert report["status"] == "completed", (group, path.name, report["status"])
            paper = report["paper"]
            quiz = json.loads(Path(report["quiz_path"]).read_text(encoding="utf-8"))
            row = {}
            for q in QTYPES:
                result = report["results"][q]
                row[q] = result["accuracy"]
                answers = result["agent_answers"]
                truth = quiz[q]["answers"]
                aspects = quiz[q]["aspects"]
                recount = 0
                for key in truth:
                    pred = answers.get(key)
                    if pred is None:
                        answer_stats[q]["missing"] += 1
                        correct = False
                    else:
                        answer_stats[q]["answered"] += 1
                        na = pred["answer"] == "NA" or pred.get("reference", "NA") == "NA"
                        answer_stats[q]["na"] += int(na)
                        letter = pred["answer"][0].upper() if pred["answer"] else ""
                        correct = (not na) and letter == truth[key].split(".")[0].strip()
                    aspect_tot[q][aspects[key]][0] += int(correct)
                    aspect_tot[q][aspects[key]][1] += 1
                    recount += int(correct)
                # the by-aspect table counts all 50 questions (missing = wrong); it must
                # agree with the upstream per-call accuracy
                assert abs(recount / len(truth) - result["accuracy"]) < 1e-12, (group, paper, q)
                answer_stats[q]["extra_keys"] += len(set(answers) - set(truth))
                run = report["api_runs"][q]
                reasks += run.get("invalid_output_reasks", 0)
                usage = run.get("usage") or {}
                cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
                priced = _usage_record(
                    {"input_tokens": usage.get("prompt_tokens", 0), "cached_input_tokens": cached,
                     "cache_write_input_tokens": 0, "output_tokens": usage.get("completion_tokens", 0)},
                    0.0, model="gpt-5.6-sol", effort="medium", thread_id=None)
                usage_tot["calls"] += 1
                usage_tot["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                usage_tot["cached_prompt_tokens"] += cached
                usage_tot["completion_tokens"] += int(usage.get("completion_tokens", 0))
                usage_tot["reasoning_tokens"] += int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
                usage_tot["cost_usd"] += priced["estimated_api_cost_usd"]
                ok_attempts = [a for a in run.get("http_attempts", []) if a.get("outcome") == "http_ok"]
                if ok_attempts:
                    usage_tot["latency_s"].append(ok_attempts[-1].get("latency_s"))
            row["average"] = (row["detail"] + row["understanding"]) / 2
            rows[paper] = row
        per_paper[group] = rows
        stats = {}
        for key in ("detail", "understanding", "average"):
            values = np.array([r[key] for r in rows.values()])
            stats[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                          "median": float(np.median(values)), "ci95": bootstrap_ci(values, rng)}
        latencies = [x for x in usage_tot.pop("latency_s") if x is not None]
        usage_tot["median_latency_s"] = float(np.median(latencies)) if latencies else None
        usage_tot["cost_usd"] = round(usage_tot["cost_usd"], 4)
        summary["groups"][group] = {
            "n_papers": len(rows),
            "accuracy": stats,
            "by_aspect": {
                q: {f"{a} {ASPECT_NAMES[q].get(a, '?')}": {"correct": c, "total": t, "accuracy": c / t}
                    for a, (c, t) in sorted(aspect_tot[q].items())}
                for q in QTYPES
            },
            "answers": {q: {**s, "na_rate": s["na"] / max(1, s["answered"] + s["missing"])} for q, s in answer_stats.items()},
            "invalid_output_reasks": reasks,
            "usage": usage_tot,
        }
    if len(args.groups) == 2:
        a, b = args.groups
        common = sorted(set(per_paper[a]) & set(per_paper[b]))
        for key in ("detail", "understanding", "average"):
            diff = np.array([per_paper[a][p][key] - per_paper[b][p][key] for p in common])
            nonzero = diff[diff != 0]
            test = wilcoxon(nonzero) if len(nonzero) else None
            summary["paired"][key] = {
                "direction": f"{a} - {b}",
                "n": len(common),
                "mean_difference": float(diff.mean()),
                "ci95": bootstrap_ci(diff, rng),
                "wins": int((diff > 0).sum()), "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum()),
                "wilcoxon_statistic": float(test.statistic) if test else None,
                "wilcoxon_p": float(test.pvalue) if test else None,
            }
    out = exp / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "per_paper.json").write_text(json.dumps(per_paper, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    brief = {g: {k: round(v["mean"] * 100, 2) for k, v in s["accuracy"].items()} for g, s in summary["groups"].items()}
    print(json.dumps({"accuracy_percent": brief, "paired": summary["paired"],
                      "cost_usd": {g: s["usage"]["cost_usd"] for g, s in summary["groups"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
