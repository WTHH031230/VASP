#!/usr/bin/env python3
"""Group scores for EXP-016 and paired comparison with our Sol finals from EXP-014.

Per group: mean over posters of the per-poster six-aspect mean, aesthetic and
information means, per-aspect means, bootstrap 95% CIs and score distributions.
Paired: ours (EXP-014 results/final) minus each baseline on the same 100 papers,
with win/tie/loss counts, signed-rank sums and Wilcoxon p. Run with the p2p env.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ASPECTS = ("aesthetic_element", "aesthetic_engagement", "aesthetic_layout",
           "information_low_level", "information_logic", "information_content")
AESTHETIC, INFORMATION = ASPECTS[:3], ASPECTS[3:]
BOOTSTRAP, SEED = 10000, 20260923


def load(results: Path, group: str) -> tuple[dict[str, dict[str, int]], list[str], Counter]:
    scores, incomplete, tokens = {}, [], Counter()
    for path in sorted((results / group).glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or set(report["results"]) != set(ASPECTS):
            incomplete.append(path.name)
            continue
        scores[report["paper"]] = {a: int(report["results"][a]["score"]) for a in ASPECTS}
        for run in report["api_runs"].values():
            usage = run.get("usage") or {}
            tokens["calls"] += 1
            tokens["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            tokens["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    return scores, incomplete, tokens


def mean_vec(scores: dict, papers: list[str], aspects: tuple[str, ...]) -> np.ndarray:
    return np.array([np.mean([scores[p][a] for a in aspects]) for p in papers], dtype=float)


def ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    boot = values[rng.integers(0, values.size, size=(BOOTSTRAP, values.size))].mean(axis=1)
    return [round(float(np.percentile(boot, 2.5)), 6), round(float(np.percentile(boot, 97.5)), 6)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ours-results", required=True, help="EXP-014 results dir")
    parser.add_argument("--ours-version", default="final")
    args = parser.parse_args()
    exp = Path(args.experiment).resolve()
    manifest = json.loads((exp / "inputs" / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)

    groups = {g: load(exp / "results", g) for g in manifest["groups"]}
    ours, ours_incomplete, _ = load(Path(args.ours_results), args.ours_version)
    summary = {"groups": {}, "paired_ours_minus_baseline": {}, "tokens": {}, "incomplete": {}}
    table = [("ours-sol", ours)] + [(g, groups[g][0]) for g in manifest["groups"]]
    for name, scores in table:
        papers = sorted(scores)
        block = {"n": len(papers)}
        for label, aspects in (("six_dim_mean", ASPECTS), ("aesthetic_mean", AESTHETIC), ("information_mean", INFORMATION)):
            v = mean_vec(scores, papers, aspects)
            block[label] = round(float(v.mean()), 6)
            block[label + "_ci"] = ci(v, rng)
        block["per_aspect"] = {a: round(float(mean_vec(scores, papers, (a,)).mean()), 6) for a in ASPECTS}
        block["distribution"] = {a: dict(sorted(Counter(scores[p][a] for p in papers).items())) for a in ASPECTS}
        summary["groups"][name] = block
    for g in manifest["groups"]:
        base = groups[g][0]
        papers = sorted(set(base) & set(ours))
        entry = {}
        for label, aspects in (("six_dim_mean", ASPECTS), ("aesthetic_mean", AESTHETIC), ("information_mean", INFORMATION), *((a, (a,)) for a in ASPECTS)):
            d = mean_vec(ours, papers, aspects) - mean_vec(base, papers, aspects)
            nonzero = d[d != 0]
            ranks = stats.rankdata(np.abs(nonzero))
            entry[label] = {
                "n": int(d.size), "mean_diff": round(float(d.mean()), 6), "ci": ci(d, rng),
                "ours_higher": int((d > 0).sum()), "tied": int((d == 0).sum()), "ours_lower": int((d < 0).sum()),
                "r_plus": float(ranks[nonzero > 0].sum()), "r_minus": float(ranks[nonzero < 0].sum()),
                "p_wilcoxon": float(stats.wilcoxon(nonzero).pvalue) if nonzero.size else None,
            }
        summary["paired_ours_minus_baseline"][g] = entry
        summary["tokens"][g] = dict(groups[g][2])
        summary["incomplete"][g] = groups[g][1]
    summary["incomplete"]["ours"] = ours_incomplete
    out = exp / "analysis"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["| group | n | six-dim | aesthetic | information | " + " | ".join(ASPECTS) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(ASPECTS)]
    for name, block in sorted(summary["groups"].items(), key=lambda kv: -kv[1]["six_dim_mean"]):
        lines.append(f"| {name} | {block['n']} | {block['six_dim_mean']:.3f} | {block['aesthetic_mean']:.3f} | "
                     f"{block['information_mean']:.3f} | " + " | ".join(f"{block['per_aspect'][a]:.2f}" for a in ASPECTS) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
