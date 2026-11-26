#!/usr/bin/env python3
"""Aggregate VisionBench and geometry metrics across T0 (VASP Default) and T1-T8 themes.

Produces:
1. Official VisionBench equal-group summary (`vb/summary_equal_group_missing_zero.{json,md}`)
2. Full statistical comparison (`analysis/theme_ablation_summary.{json,md}`):
   - Visual Presentation Scores (Poster, Figure, Guidance, Layout)
   - Paired differences vs. T0 with 95% bootstrap CI and paired sign-flip p-values
   - Per-criterion pass rates (IS01, AE13, AL05, WS01, DEN01, COL01, EN05 + all 12 questions)
   - Deterministic geometry & figure-allocation invariants (overflow, largest, area, node_drift_px)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUESTION_ORDER = [
    "AE04", "AE13", "IS01",
    "ENH01", "DIST01", "COL01", "EN05", "EN07",
    "AL01", "AL05", "DEN01", "WS01",
]
TABLE2_CRITERIA = ["IS01", "AE13", "AL05", "WS01", "DEN01", "COL01", "EN05"]
GROUPS = {
    "Figure presentation": ["AE04", "AE13", "IS01"],
    "Visual guidance and emphasis": ["ENH01", "DIST01", "COL01", "EN05", "EN07"],
    "Layout and spatial organization": ["AL01", "AL05", "DEN01", "WS01"],
}
THEME_ORDER = [
    "t0_original",
    "t1_paper2poster",
    "t2_posterforest",
    "t3_postermeld",
    "t4_postergen",
    "t5_researchstudio",
    "t6_p2p",
    "t7_monochrome",
    "t8_beamer",
]
THEME_LABELS = {
    "t0_original": "T0: VASP Default (Pi Eval)",
    "t0_stored_api": "T0 (Ref): VASP Default (Stored API)",
    "t1_paper2poster": "T1: Paper2Poster Style",
    "t2_posterforest": "T2: PosterForest Style",
    "t3_postermeld": "T3: PosterMELD Style",
    "t4_postergen": "T4: PosterGen Style",
    "t5_researchstudio": "T5: ResearchStudio Style",
    "t6_p2p": "T6: P2P Style",
    "t7_monochrome": "T7: Monochrome Wireframe",
    "t8_beamer": "T8: LaTeX Beamer Classic",
}


def paired_bootstrap_ci(
    diffs: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, hi


def paired_sign_flip_pvalue(diffs: list[float], *, seed: int = 42) -> float:
    """Exact two-sided paired sign-flip permutation p-value for n <= 20."""
    non_zero = [d for d in diffs if abs(d) > 1e-12]
    m = len(non_zero)
    if m == 0:
        return 1.0
    obs = abs(sum(non_zero) / m)
    if m <= 20:
        total = 1 << m
        extreme = 0
        for mask in range(total):
            s = 0.0
            for idx in range(m):
                s += non_zero[idx] if (mask & (1 << idx)) else -non_zero[idx]
            if abs(s / m) >= obs - 1e-12:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    n_perm = 20000
    extreme = 0
    for _ in range(n_perm):
        s = sum(d if rng.getrandbits(1) else -d for d in non_zero)
        if abs(s / m) >= obs - 1e-12:
            extreme += 1
    return extreme / n_perm


def compute_geometry_metrics(
    exp_dir: Path,
    theme_id: str,
    titles: list[str],
    t0_rects_by_title: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    run_theme = "t0_original" if theme_id in {"t0_original", "t0_stored_api"} else theme_id
    overflow_free = 0
    largest_ratios: list[float] = []
    total_area_ratios: list[float] = []
    max_drifts: list[float] = []
    valid_count = 0

    for title in titles:
        val_path = exp_dir / "runs" / run_theme / title / "final" / "validation.json"
        if not val_path.is_file():
            continue
        val = json.loads(val_path.read_text(encoding="utf-8"))
        valid_count += 1
        hard_ov = val.get("hardOverflow") or []
        if len(hard_ov) == 0:
            overflow_free += 1
        canvas = val.get("canvas") or [2544, 1696]
        canvas_area = max(1.0, float(canvas[0]) * float(canvas[1]))
        sci_imgs = val.get("scientificImageMetrics") or []
        areas = []
        for img in sci_imgs:
            eff = img.get("effective") or img.get("rect") or {}
            w = float(eff.get("w") or 0.0)
            h = float(eff.get("h") or 0.0)
            if w > 0 and h > 0:
                areas.append(w * h)
        largest_ratios.append((max(areas) / canvas_area) * 100.0 if areas else 0.0)
        total_area_ratios.append((sum(areas) / canvas_area) * 100.0 if areas else 0.0)

        t0_rects = t0_rects_by_title.get(title) or {}
        cur_rects = val.get("nodeRects") or {}
        paper_drift = 0.0
        for nid, r0 in t0_rects.items():
            r1 = cur_rects.get(nid)
            if not r1:
                continue
            for k in ("x", "y", "w", "h"):
                d = abs(float(r1.get(k, 0.0)) - float(r0.get(k, 0.0)))
                if d > paper_drift:
                    paper_drift = d
        max_drifts.append(paper_drift)

    if valid_count == 0:
        return {
            "count": 0,
            "overflow_pass_pct": None,
            "largest_fig_pct": None,
            "total_fig_area_pct": None,
            "max_node_rect_drift_px": None,
        }
    return {
        "count": valid_count,
        "overflow_pass_pct": (overflow_free / valid_count) * 100.0,
        "largest_fig_pct": statistics.mean(largest_ratios),
        "total_fig_area_pct": statistics.mean(total_area_ratios),
        "max_node_rect_drift_px": max(max_drifts) if max_drifts else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--themes",
        nargs="+",
        default=THEME_ORDER,
        help="Ordered theme IDs to include in comparison (first must be baseline T0).",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected number of evaluated posters per theme (defaults to manifest paper_count or 20).",
    )
    return parser.parse_args()


def compute_pi_usage_summary(vb_results_dir: Path, themes: list[str], titles: list[str]) -> dict[str, Any]:
    totals = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0, "latency_s": 0.0}
    per_theme: dict[str, dict[str, Any]] = {}
    for t in themes:
        t_tot = {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0, "latency_s": 0.0}
        for title in titles:
            p = vb_results_dir / t / f"{title}.json"
            if not p.is_file():
                continue
            rep = json.loads(p.read_text(encoding="utf-8"))
            for qid, rec in (rep.get("pi_runs") or {}).items():
                t_tot["calls"] += 1
                u = rec.get("usage") or {}
                t_tot["input"] += int(u.get("input") or 0)
                t_tot["output"] += int(u.get("output") or 0)
                t_tot["cache_read"] += int(u.get("cache_read") or 0)
                t_tot["cache_write"] += int(u.get("cache_write") or 0)
                t_tot["reasoning"] += int(u.get("reasoning") or 0)
                for att in rec.get("pi_attempts") or []:
                    t_tot["latency_s"] += float(att.get("latency_s") or 0.0)
        t_tot["latency_s"] = round(t_tot["latency_s"], 1)
        per_theme[t] = t_tot
        for k in totals:
            totals[k] += t_tot[k]
    totals["latency_s"] = round(totals["latency_s"], 1)
    return {"total": totals, "per_theme": per_theme}


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    code_dir = Path(__file__).resolve().parent
    agg_script = code_dir / "aggregate_equal_group_comparison_missing_zero.py"

    manifest_path = exp_dir / "inputs" / "manifest.json"
    expected_count = args.expected_count
    expected_titles: list[str] | None = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_count is None:
            expected_count = int(manifest.get("paper_count") or 20)
        expected_titles = [str(p["title"]) for p in (manifest.get("papers") or [])]
    if expected_count is None:
        expected_count = 20

    vb_results_dir = exp_dir / "vb" / "results"
    requested_themes = list(args.themes)
    # If t0_original was not re-evaluated via Pi, fall back to t0_stored_api if present
    if (
        requested_themes
        and requested_themes[0] == "t0_original"
        and not ((vb_results_dir / "t0_original").is_dir() and list((vb_results_dir / "t0_original").glob("*.json")))
        and (vb_results_dir / "t0_stored_api").is_dir()
        and list((vb_results_dir / "t0_stored_api").glob("*.json"))
    ):
        requested_themes[0] = "t0_stored_api"

    missing_themes = [
        t for t in requested_themes
        if not ((vb_results_dir / t).is_dir() and len(list((vb_results_dir / t).glob("*.json"))) >= expected_count)
    ]
    if missing_themes:
        raise RuntimeError(
            f"Missing or incomplete theme result directories in {vb_results_dir} "
            f"(expected {expected_count} JSON reports each): {missing_themes}"
        )

    # Also include t0_stored_api at the end as a reference row if t0_original is the primary baseline
    # and all 20 papers are being compared
    eval_themes = list(requested_themes)
    if (
        expected_count == 20
        and eval_themes[0] == "t0_original"
        and "t0_stored_api" not in eval_themes
        and (vb_results_dir / "t0_stored_api").is_dir()
        and len(list((vb_results_dir / "t0_stored_api").glob("*.json"))) == 20
    ):
        eval_themes.append("t0_stored_api")

    vb_json = exp_dir / "vb" / "summary_equal_group_missing_zero.json"
    vb_md = exp_dir / "vb" / "summary_equal_group_missing_zero.md"

    cmd = [sys.executable, str(agg_script)]
    for t in eval_themes:
        cmd.extend(["--method-dir", f"{t}={vb_results_dir / t}"])
    cmd.extend(["--output-json", str(vb_json), "--output-md", str(vb_md)])
    subprocess.run(cmd, check=True)

    vb_summary = json.loads(vb_json.read_text(encoding="utf-8"))
    method_stats = vb_summary["method_stats"]
    titles = sorted(next(iter(method_stats.values()))["posters"].keys())

    t0_rects_by_title: dict[str, dict[str, dict[str, float]]] = {}
    for title in titles:
        vp = exp_dir / "runs" / "t0_original" / title / "final" / "validation.json"
        if vp.is_file():
            t0_rects_by_title[title] = json.loads(vp.read_text(encoding="utf-8")).get("nodeRects") or {}

    per_question_rates: dict[str, dict[str, float | None]] = {}
    for t in eval_themes:
        per_question_rates[t] = {}
        reports_by_title = {}
        for p in sorted((vb_results_dir / t).glob("*.json")):
            rep = json.loads(p.read_text(encoding="utf-8"))
            qmap = {}
            for block in rep.get("dimensions", {}).values():
                for item in block.get("results", []):
                    qmap[str(item["id"])] = str(item["verdict"])
            reports_by_title[p.stem] = qmap
        for qid in QUESTION_ORDER:
            verdicts = [reports_by_title[title][qid] for title in titles if qid in reports_by_title.get(title, {})]
            pos = sum(1 for v in verdicts if v == "PASS")
            neg = sum(1 for v in verdicts if v == "FAIL")
            per_question_rates[t][qid] = (pos / (pos + neg)) * 100.0 if (pos + neg) > 0 else None

    baseline_id = requested_themes[0]
    baseline_posters = method_stats[baseline_id]["posters"]

    theme_rows: list[dict[str, Any]] = []
    for t in eval_themes:
        mstat = method_stats[t]
        posters_map = mstat["posters"]
        poster_score = float(mstat["mean_poster_equal_group_score"]) * 100.0
        fig_score = statistics.mean(
            float(posters_map[title]["groups"]["Figure presentation"]["pass_rate"]) * 100.0
            for title in titles
        )
        guid_score = statistics.mean(
            float(posters_map[title]["groups"]["Visual guidance and emphasis"]["pass_rate"]) * 100.0
            for title in titles
        )
        lay_score = statistics.mean(
            float(posters_map[title]["groups"]["Layout and spatial organization"]["pass_rate"]) * 100.0
            for title in titles
        )

        diffs = [
            (float(posters_map[title]["score"]) - float(baseline_posters[title]["score"])) * 100.0
            for title in titles
        ]
        mean_diff = statistics.mean(diffs) if diffs else 0.0
        sd_diff = statistics.stdev(diffs) if len(diffs) >= 2 else 0.0
        ci_lo, ci_hi = paired_bootstrap_ci(diffs)
        pval = 1.0 if t == baseline_id else paired_sign_flip_pvalue(diffs)

        geom = compute_geometry_metrics(exp_dir, t, titles, t0_rects_by_title)
        theme_rows.append(
            {
                "theme_id": t,
                "label": THEME_LABELS.get(t, t),
                "is_reference_only": (t == "t0_stored_api" and baseline_id == "t0_original"),
                "poster_count": len(titles),
                "poster_score_pct": round(poster_score, 2),
                "figure_score_pct": round(fig_score, 2),
                "guidance_score_pct": round(guid_score, 2),
                "layout_score_pct": round(lay_score, 2),
                "paired_diff_vs_t0_pct": round(mean_diff, 2),
                "paired_diff_sd_pct": round(sd_diff, 2),
                "paired_diff_ci95_pct": [round(ci_lo, 2), round(ci_hi, 2)],
                "paired_sign_flip_pvalue": round(pval, 4),
                "geometry": {
                    "overflow_pass_pct": round(geom["overflow_pass_pct"], 1)
                    if geom["overflow_pass_pct"] is not None
                    else None,
                    "largest_fig_pct": round(geom["largest_fig_pct"], 1)
                    if geom["largest_fig_pct"] is not None
                    else None,
                    "total_fig_area_pct": round(geom["total_fig_area_pct"], 1)
                    if geom["total_fig_area_pct"] is not None
                    else None,
                    "max_node_rect_drift_px": round(geom["max_node_rect_drift_px"], 2)
                    if geom["max_node_rect_drift_px"] is not None
                    else None,
                },
                "criteria_pass_rates_pct": {
                    qid: (round(per_question_rates[t][qid], 1) if per_question_rates[t][qid] is not None else None)
                    for qid in QUESTION_ORDER
                },
            }
        )

    primary_rows = [r for r in theme_rows if not r["is_reference_only"]]
    poster_scores_all = [r["poster_score_pct"] for r in primary_rows]
    cross_theme_summary = {
        "theme_count": len(primary_rows),
        "mean_across_themes_pct": round(statistics.mean(poster_scores_all), 2),
        "sd_across_themes_pct": round(statistics.stdev(poster_scores_all), 2) if len(poster_scores_all) >= 2 else 0.0,
        "min_theme_score_pct": round(min(poster_scores_all), 2),
        "max_theme_score_pct": round(max(poster_scores_all), 2),
        "range_pct": round(max(poster_scores_all) - min(poster_scores_all), 2),
    }

    out_summary = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_theme": baseline_id,
        "poster_count": len(titles),
        "titles": titles,
        "cross_theme_summary": cross_theme_summary,
        "pi_usage_summary": compute_pi_usage_summary(vb_results_dir, eval_themes, titles),
        "themes": theme_rows,
    }

    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_json = analysis_dir / "theme_ablation_summary.json"
    out_md = analysis_dir / "theme_ablation_summary.md"
    out_json.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Visual Theme Sensitivity Ablation (Reviewer P6ve W1)",
        "",
        f"- **Dataset**: 20-paper standard ablation benchmark (`k1` + `e1`), holding VASP (GPT-5.6-Sol) content, asset extraction, and spatial layout fixed.",
        f"- **Cross-theme stability**: Mean Poster score across all {len(primary_rows)} themes = **{cross_theme_summary['mean_across_themes_pct']:.1f}% ± {cross_theme_summary['sd_across_themes_pct']:.1f}%** (min `{cross_theme_summary['min_theme_score_pct']:.1f}%`, max `{cross_theme_summary['max_theme_score_pct']:.1f}%`, total spread `{cross_theme_summary['range_pct']:.1f}` pp).",
        "",
        "## 1. Visual Presentation Scores & Geometry Invariants",
        "",
        "| Theme | Poster | Δ vs. T0 (95% CI) | p-val | Figure | Guidance | Layout | overflow | largest | area | max drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in theme_rows:
        ci_lo, ci_hi = r["paired_diff_ci95_pct"]
        diff_str = "—" if r["theme_id"] == baseline_id else f"{r['paired_diff_vs_t0_pct']:+.1f}% [{ci_lo:+.1f}, {ci_hi:+.1f}]"
        p_str = "—" if r["theme_id"] == baseline_id else f"{r['paired_sign_flip_pvalue']:.3f}"
        g = r["geometry"]
        ov_str = "—" if g["overflow_pass_pct"] is None else f"{g['overflow_pass_pct']:.1f}%"
        lg_str = "—" if g["largest_fig_pct"] is None else f"{g['largest_fig_pct']:.1f}%"
        ar_str = "—" if g["total_fig_area_pct"] is None else f"{g['total_fig_area_pct']:.1f}%"
        dr_str = "—" if g["max_node_rect_drift_px"] is None else f"{g['max_node_rect_drift_px']:.1f}px"
        md_lines.append(
            f"| {r['label']} | **{r['poster_score_pct']:.1f}%** | {diff_str} | {p_str} | "
            f"{r['figure_score_pct']:.1f}% | {r['guidance_score_pct']:.1f}% | {r['layout_score_pct']:.1f}% | "
            f"{ov_str} | {lg_str} | {ar_str} | {dr_str} |"
        )

    md_lines += [
        "",
        "## 2. Table 2 Key Criteria Pass Rates (%)",
        "",
        "| Theme | " + " | ".join(TABLE2_CRITERIA) + " |",
        "|---|" + "---:|" * len(TABLE2_CRITERIA),
    ]
    for r in theme_rows:
        c = r["criteria_pass_rates_pct"]
        vals = ["—" if c[q] is None else f"{c[q]:.1f}%" for q in TABLE2_CRITERIA]
        md_lines.append(f"| {r['label']} | " + " | ".join(vals) + " |")

    md_lines += [
        "",
        "## 3. All 12 VisionBench Atomic Questions Pass Rates (%)",
        "",
        "| Theme | " + " | ".join(QUESTION_ORDER) + " |",
        "|---|" + "---:|" * len(QUESTION_ORDER),
    ]
    for r in theme_rows:
        c = r["criteria_pass_rates_pct"]
        vals = ["—" if c[q] is None else f"{c[q]:.1f}%" for q in QUESTION_ORDER]
        md_lines.append(f"| {r['label']} | " + " | ".join(vals) + " |")

    usage_tot = out_summary["pi_usage_summary"]["total"]
    md_lines += [
        "",
        "## 4. Pi OAuth Token & Call Usage Summary",
        "",
        f"- **Total Pi atomic question calls**: `{usage_tot['calls']}`",
        f"- **Total input tokens**: `{usage_tot['input']:,}` (cache read: `{usage_tot['cache_read']:,}`, uncached input: `{max(0, usage_tot['input'] - usage_tot['cache_read']):,}`)",
        f"- **Total output tokens**: `{usage_tot['output']:,}` (reasoning: `{usage_tot['reasoning']:,}`)",
        f"- **Cumulative worker latency**: `{usage_tot['latency_s']:.1f}s`",
    ]

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        f"[analyze_themes] Pi usage across {len(titles)} poster(s) x {len(eval_themes)} theme(s): "
        f"calls={usage_tot['calls']}, input_tokens={usage_tot['input']:,} "
        f"(cache_read={usage_tot['cache_read']:,}), output_tokens={usage_tot['output']:,} "
        f"(reasoning={usage_tot['reasoning']:,})",
        flush=True,
    )
    print(f"[analyze_themes] Wrote {out_json}", flush=True)
    print(f"[analyze_themes] Wrote {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
