#!/usr/bin/env python3
"""Run all 5 VASP benchmarks (SixDim, PaperQuiz, VisionBench, InfoCoverage, CEG) on PaperX via Codex gpt-5.6-sol medium."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("VASP_REPO_ROOT", str(Path(__file__).resolve().parents[3])))
EXP_DIR = Path(os.environ.get("VASP_OUT_DIR", "./outputs/paperx"))
EVAL_DIR = EXP_DIR / "eval"
PY_POSTERGEN = os.environ.get("VASP_PYTHON", sys.executable)
METHOD = "paperx"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def stage_ready_papers(only_papers: list[str] | None = None) -> list[str]:
    index_map = json.loads((EXP_DIR / "paper_index_map.json").read_text(encoding="utf-8"))
    all_canon = [e["canonical_name"] for e in index_map]
    if only_papers is not None:
        target = only_papers
    else:
        target = [p for p in all_canon if (EXP_DIR / "posters" / p / "poster.png").is_file()]
    if not target:
        raise RuntimeError("No completed PaperX posters found under posters/*/poster.png")

    mdir = EVAL_DIR / "inputs" / METHOD
    mdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"papers": target, "groups": {METHOD: {}}}
    for p in target:
        src = EXP_DIR / "posters" / p / "poster.png"
        dst = mdir / f"{p}.png"
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        manifest["groups"][METHOD][p] = {
            "source": str(src),
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        }
    (EVAL_DIR / "inputs" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"Staged {len(target)} PaperX poster(s) into {mdir}")
    return target


def run_step(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        log(f"Running (attempt {attempt}/3): {cmd[1]} -> {log_path.name}")
        with log_path.open("a", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, check=False)
        if proc.returncode == 0:
            return
        log(f"Step failed with rc={proc.returncode}; retrying in 5s...")
        time.sleep(5)
    raise RuntimeError(f"Evaluation step failed after 3 attempts; see {log_path}")


def paired_bootstrap_ci(vasp_vals: list[float], base_vals: list[float], n_boot: int = 10000, seed: int = 42) -> dict[str, Any]:
    diffs = [v - b for v, b in zip(vasp_vals, base_vals)]
    n = len(diffs)
    mean_diff = statistics.mean(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    w = sum(1 for d in diffs if d > 1e-9)
    t = sum(1 for d in diffs if abs(d) <= 1e-9)
    l = sum(1 for d in diffs if d < -1e-9)
    return {
        "delta_vasp_minus_paperx": round(mean_diff, 2),
        "ci95": [round(lo, 2), round(hi, 2)],
        "WTL": f"{w}W-{t}T-{l}L",
    }


def summarize_results(papers: list[str]) -> Path:
    ic_manifest = json.loads(
        (ROOT / "results/exp/infocoverage-api-9groups-medium-20260924/_code/snapshot/information_coverage_bench/api_nine_methods/manifest.json").read_text(encoding="utf-8")
    )
    fn_to_id = {row["input_filename"]: str(row["poster_id"]) for row in ic_manifest["posters"]}
    poster_ids = [fn_to_id[f"{p}.png"] for p in papers]

    aes_keys = ["aesthetic_element", "aesthetic_layout", "aesthetic_engagement"]
    info_keys = ["information_low_level", "information_logic", "information_content"]
    all_six = aes_keys + info_keys

    def load_sixdim(folder: Path) -> dict[str, Any]:
        aes_l, info_l, ov_l, per_p = [], [], [], {}
        asp = {k: [] for k in all_six}
        for p in papers:
            res = json.loads((folder / f"{p}.json").read_text(encoding="utf-8"))["results"]
            sc = {k: float(res[k]["score"]) for k in all_six}
            for k in all_six:
                asp[k].append(sc[k])
            a = sum(sc[k] for k in aes_keys) / 3.0
            i = sum(sc[k] for k in info_keys) / 3.0
            o = sum(sc[k] for k in all_six) / 6.0
            aes_l.append(a)
            info_l.append(i)
            ov_l.append(o)
            per_p[p] = o
        return {
            "Aes": round(statistics.mean(aes_l), 2),
            "Info": round(statistics.mean(info_l), 2),
            "Overall": round(statistics.mean(ov_l), 2),
            "aspects": {k: round(statistics.mean(v), 2) for k, v in asp.items()},
            "_per_p": per_p,
        }

    def load_paperquiz(folder: Path) -> dict[str, Any]:
        det_l, und_l, ov_l, per_p = [], [], [], {}
        for p in papers:
            r = json.loads((folder / f"{p}.json").read_text(encoding="utf-8"))["results"]
            d = float(r["detail"]["accuracy"]) * 100.0
            u = float(r["understanding"]["accuracy"]) * 100.0
            o = (d + u) / 2.0
            det_l.append(d)
            und_l.append(u)
            ov_l.append(o)
            per_p[p] = o
        return {
            "Detail": round(statistics.mean(det_l), 2),
            "Understanding": round(statistics.mean(und_l), 2),
            "Overall": round(statistics.mean(ov_l), 2),
            "_per_p": per_p,
        }

    def load_visionbench(folder: Path) -> dict[str, Any]:
        fig_fix, guid_fix, lay_fix, ov_fix = [], [], [], []
        ov_excl_na, per_p_fix = [], {}
        na_total = 0
        for p in papers:
            dims = json.loads((folder / f"{p}.json").read_text(encoding="utf-8"))["dimensions"]
            dim_fix = {}
            dim_ena = []
            for dk in ("aesthetic_element", "aesthetic_engagement", "aesthetic_layout"):
                agg = dims[dk]["aggregate"]
                qc = int(agg["question_count"])
                cnt = agg["counts"]
                pa = int(cnt.get("PASS", 0))
                na = int(cnt.get("NOT_APPLICABLE", 0))
                na_total += na
                dim_fix[dk] = pa / qc * 100.0
                if qc - na > 0:
                    dim_ena.append(pa / (qc - na) * 100.0)
            f_v, g_v, l_v = dim_fix["aesthetic_element"], dim_fix["aesthetic_engagement"], dim_fix["aesthetic_layout"]
            o_v = (f_v + g_v + l_v) / 3.0
            fig_fix.append(f_v)
            guid_fix.append(g_v)
            lay_fix.append(l_v)
            ov_fix.append(o_v)
            ov_excl_na.append(statistics.mean(dim_ena))
            per_p_fix[p] = o_v
        return {
            "Figure_Fixed": round(statistics.mean(fig_fix), 2),
            "Guidance_Fixed": round(statistics.mean(guid_fix), 2),
            "Layout_Fixed": round(statistics.mean(lay_fix), 2),
            "Overall_Fixed": round(statistics.mean(ov_fix), 2),
            "Overall_Excl_NA": round(statistics.mean(ov_excl_na), 2),
            "Total_NA": na_total,
            "_per_p": per_p_fix,
        }

    def load_infocov(folder: Path) -> dict[str, Any]:
        miss_l, per_p = [], {}
        for p, pid in zip(papers, poster_ids):
            mc = int(json.loads((folder / f"{pid}.json").read_text(encoding="utf-8"))["missing_count"])
            miss_l.append(mc)
            per_p[p] = (7 - mc) / 7.0 * 100.0
        tot = len(papers) * 7
        cov = (tot - sum(miss_l)) / tot * 100.0
        comp = sum(1 for m in miss_l if m == 0) / len(papers) * 100.0
        return {
            "Coverage_pct": round(cov, 2),
            "Avg_Missing": round(statistics.mean(miss_l), 2),
            "Complete_Posters_pct": round(comp, 2),
            "_per_p": per_p,
        }

    def load_ceg(folder: Path) -> dict[str, Any]:
        pa_sum, to_sum, macro_l, per_p = 0, 0, [], {}
        for p in papers:
            d = json.loads((folder / f"{p}.json").read_text(encoding="utf-8"))
            pa = int(d["passed_figures"])
            to = int(d["total_figures"])
            sc = (pa / to * 100.0) if to > 0 else 0.0
            pa_sum += pa
            to_sum += to
            macro_l.append(sc)
            per_p[p] = sc
        return {
            "Macro_CEG_pct": round(statistics.mean(macro_l), 2),
            "Micro_CEG_pct": round((pa_sum / to_sum * 100.0) if to_sum > 0 else 0.0, 2),
            "Passed_over_Total": f"{pa_sum}/{to_sum}",
            "_per_p": per_p,
        }

    sd_vasp = load_sixdim(ROOT / "results/exp/p2p-sixdim-upstream-api-pi-full-sol-20260926/results/pi_full")
    sd_px = load_sixdim(EVAL_DIR / "sixdim/results" / METHOD)

    pq_vasp = load_paperquiz(ROOT / "results/exp/paperquiz-upstream-api-pi-rs-20260926/results/pi_full")
    pq_px = load_paperquiz(EVAL_DIR / "paperquiz/results" / METHOD)

    vb_vasp = load_visionbench(ROOT / "results/exp/visionbench-api-9groups-medium-20260923/results/gpt5.6-sol-medium")
    vb_px = load_visionbench(EVAL_DIR / "visionbench/results" / METHOD)

    ic_base = ROOT / "results/exp/infocoverage-api-9groups-medium-20260924/_code/snapshot/information_coverage_bench/api_nine_methods/evaluations"
    ic_vasp = load_infocov(ic_base / "gpt5.6-sol-medium/results")
    ic_px = load_infocov(EVAL_DIR / "infocoverage/results" / METHOD)

    ceg_vasp = load_ceg(ROOT / "results/exp/ceg-full100-and-opus20-20261109/results/pi_full")
    ceg_px = load_ceg(EVAL_DIR / "ceg/results" / METHOD)

    def strip_priv(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    summary = {
        "num_papers": len(papers),
        "SixDim": {
            "VASP (GPT-5.6-Sol)": strip_priv(sd_vasp),
            "PaperX (GPT-5.6-Sol)": strip_priv(sd_px),
            "Paired_VASP_vs_PaperX": paired_bootstrap_ci([sd_vasp["_per_p"][p] for p in papers], [sd_px["_per_p"][p] for p in papers]),
        },
        "PaperQuiz": {
            "VASP (GPT-5.6-Sol)": strip_priv(pq_vasp),
            "PaperX (GPT-5.6-Sol)": strip_priv(pq_px),
            "Paired_VASP_vs_PaperX": paired_bootstrap_ci([pq_vasp["_per_p"][p] for p in papers], [pq_px["_per_p"][p] for p in papers]),
        },
        "VisionBench": {
            "VASP (GPT-5.6-Sol)": strip_priv(vb_vasp),
            "PaperX (GPT-5.6-Sol)": strip_priv(vb_px),
            "Paired_VASP_vs_PaperX": paired_bootstrap_ci([vb_vasp["_per_p"][p] for p in papers], [vb_px["_per_p"][p] for p in papers]),
        },
        "InfoCoverage": {
            "VASP (GPT-5.6-Sol)": strip_priv(ic_vasp),
            "PaperX (GPT-5.6-Sol)": strip_priv(ic_px),
            "Paired_VASP_vs_PaperX": paired_bootstrap_ci([ic_vasp["_per_p"][p] for p in papers], [ic_px["_per_p"][p] for p in papers]),
        },
        "CEG": {
            "VASP (GPT-5.6-Sol)": strip_priv(ceg_vasp),
            "PaperX (GPT-5.6-Sol)": strip_priv(ceg_px),
            "Paired_VASP_vs_PaperX": paired_bootstrap_ci([ceg_vasp["_per_p"][p] for p in papers], [ceg_px["_per_p"][p] for p in papers]),
        },
    }
    all_100 = [e["canonical_name"] for e in json.loads((EXP_DIR / "paper_index_map.json").read_text(encoding="utf-8"))]
    failed_papers = [p for p in all_100 if p not in set(papers)]
    if failed_papers:
        all_pids = [fn_to_id[f"{p}.png"] for p in all_100]
        def _full100_vasp_sixdim(folder: Path) -> dict[str, Any]:
            aes_l, info_l, ov_l, per_p = [], [], [], {}
            for p in all_100:
                res = json.loads((folder / f"{p}.json").read_text(encoding="utf-8"))["results"]
                sc = {k: float(res[k]["score"]) for k in all_six}
                a, i, o = sum(sc[k] for k in aes_keys) / 3.0, sum(sc[k] for k in info_keys) / 3.0, sum(sc[k] for k in all_six) / 6.0
                aes_l.append(a); info_l.append(i); ov_l.append(o); per_p[p] = o
            return {"Aes": round(statistics.mean(aes_l), 2), "Info": round(statistics.mean(info_l), 2), "Overall": round(statistics.mean(ov_l), 2), "_per_p": per_p}
        def _full100_px_sixdim(folder: Path, min_score: float = 0.0) -> dict[str, Any]:
            aes_l, info_l, ov_l, per_p = [], [], [], {}
            for p in all_100:
                fp = folder / f"{p}.json"
                if fp.is_file():
                    res = json.loads(fp.read_text(encoding="utf-8"))["results"]
                    sc = {k: float(res[k]["score"]) for k in all_six}
                    a, i, o = sum(sc[k] for k in aes_keys) / 3.0, sum(sc[k] for k in info_keys) / 3.0, sum(sc[k] for k in all_six) / 6.0
                else:
                    a = i = o = min_score
                aes_l.append(a); info_l.append(i); ov_l.append(o); per_p[p] = o
            return {"Aes": round(statistics.mean(aes_l), 2), "Info": round(statistics.mean(info_l), 2), "Overall": round(statistics.mean(ov_l), 2), "_per_p": per_p}
        def _full100_pq(folder: Path) -> dict[str, Any]:
            det_l, und_l, ov_l, per_p = [], [], [], {}
            for p in all_100:
                fp = folder / f"{p}.json"
                if fp.is_file():
                    r = json.loads(fp.read_text(encoding="utf-8"))["results"]
                    d, u = float(r["detail"]["accuracy"]) * 100.0, float(r["understanding"]["accuracy"]) * 100.0
                    o = (d + u) / 2.0
                else:
                    d = u = o = 0.0
                det_l.append(d); und_l.append(u); ov_l.append(o); per_p[p] = o
            return {"Detail": round(statistics.mean(det_l), 2), "Understanding": round(statistics.mean(und_l), 2), "Overall": round(statistics.mean(ov_l), 2), "_per_p": per_p}
        def _full100_vb(folder: Path) -> dict[str, Any]:
            fig_fix, guid_fix, lay_fix, ov_fix, ov_ena, per_p = [], [], [], [], [], {}
            for p in all_100:
                fp = folder / f"{p}.json"
                if fp.is_file():
                    dims = json.loads(fp.read_text(encoding="utf-8"))["dimensions"]
                    dim_fix, dim_ena = {}, []
                    for dk in ("aesthetic_element", "aesthetic_engagement", "aesthetic_layout"):
                        agg = dims[dk]["aggregate"]
                        qc, cnt = int(agg["question_count"]), agg["counts"]
                        pa, na = int(cnt.get("PASS", 0)), int(cnt.get("NOT_APPLICABLE", 0))
                        dim_fix[dk] = pa / qc * 100.0
                        if qc - na > 0:
                            dim_ena.append(pa / (qc - na) * 100.0)
                    f_v, g_v, l_v = dim_fix["aesthetic_element"], dim_fix["aesthetic_engagement"], dim_fix["aesthetic_layout"]
                    o_v, ena_v = (f_v + g_v + l_v) / 3.0, statistics.mean(dim_ena)
                else:
                    f_v = g_v = l_v = o_v = ena_v = 0.0
                fig_fix.append(f_v); guid_fix.append(g_v); lay_fix.append(l_v); ov_fix.append(o_v); ov_ena.append(ena_v); per_p[p] = o_v
            return {"Figure_Fixed": round(statistics.mean(fig_fix), 2), "Guidance_Fixed": round(statistics.mean(guid_fix), 2), "Layout_Fixed": round(statistics.mean(lay_fix), 2), "Overall_Fixed": round(statistics.mean(ov_fix), 2), "Overall_Excl_NA": round(statistics.mean(ov_ena), 2), "_per_p": per_p}
        def _full100_ic(folder: Path) -> dict[str, Any]:
            miss_l, per_p = [], {}
            for p, pid in zip(all_100, all_pids):
                fp = folder / f"{pid}.json"
                mc = int(json.loads(fp.read_text(encoding="utf-8"))["missing_count"]) if fp.is_file() else 7
                miss_l.append(mc); per_p[p] = (7 - mc) / 7.0 * 100.0
            tot = len(all_100) * 7
            return {"Coverage_pct": round((tot - sum(miss_l)) / tot * 100.0, 2), "Avg_Missing": round(statistics.mean(miss_l), 2), "Complete_Posters_pct": round(sum(1 for m in miss_l if m == 0) / len(all_100) * 100.0, 2), "_per_p": per_p}
        def _full100_ceg(folder: Path) -> dict[str, Any]:
            pa_sum, to_sum, macro_l, per_p = 0, 0, [], {}
            for p in all_100:
                fp = folder / f"{p}.json"
                if fp.is_file():
                    d = json.loads(fp.read_text(encoding="utf-8"))
                    pa, to = int(d["passed_figures"]), int(d["total_figures"])
                    sc = (pa / to * 100.0) if to > 0 else 0.0
                    pa_sum += pa; to_sum += to
                else:
                    sc = 0.0
                macro_l.append(sc); per_p[p] = sc
            return {"Macro_CEG_pct": round(statistics.mean(macro_l), 2), "Micro_CEG_pct": round((pa_sum / to_sum * 100.0) if to_sum > 0 else 0.0, 2), "Passed_over_Total": f"{pa_sum}/{to_sum}", "_per_p": per_p}

        sd_v100 = _full100_vasp_sixdim(ROOT / "results/exp/p2p-sixdim-upstream-api-pi-full-sol-20260926/results/pi_full")
        sd_p100_z = _full100_px_sixdim(EVAL_DIR / "sixdim/results" / METHOD, 0.0)
        sd_p100_1 = _full100_px_sixdim(EVAL_DIR / "sixdim/results" / METHOD, 1.0)
        pq_v100 = _full100_pq(ROOT / "results/exp/paperquiz-upstream-api-pi-rs-20260926/results/pi_full")
        pq_p100 = _full100_pq(EVAL_DIR / "paperquiz/results" / METHOD)
        vb_v100 = _full100_vb(ROOT / "results/exp/visionbench-api-9groups-medium-20260923/results/gpt5.6-sol-medium")
        vb_p100 = _full100_vb(EVAL_DIR / "visionbench/results" / METHOD)
        ic_v100 = _full100_ic(ic_base / "gpt5.6-sol-medium/results")
        ic_p100 = _full100_ic(EVAL_DIR / "infocoverage/results" / METHOD)
        ceg_v100 = _full100_ceg(ROOT / "results/exp/ceg-full100-and-opus20-20261109/results/pi_full")
        ceg_p100 = _full100_ceg(EVAL_DIR / "ceg/results" / METHOD)
        summary["failed_generation_papers"] = failed_papers
        summary["Full_100_MissingAsZero"] = {
            "SixDim_Missing0": {"VASP (GPT-5.6-Sol)": strip_priv(sd_v100), "PaperX (GPT-5.6-Sol)": strip_priv(sd_p100_z), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([sd_v100["_per_p"][p] for p in all_100], [sd_p100_z["_per_p"][p] for p in all_100])},
            "SixDim_Missing1": {"VASP (GPT-5.6-Sol)": strip_priv(sd_v100), "PaperX (GPT-5.6-Sol)": strip_priv(sd_p100_1), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([sd_v100["_per_p"][p] for p in all_100], [sd_p100_1["_per_p"][p] for p in all_100])},
            "PaperQuiz": {"VASP (GPT-5.6-Sol)": strip_priv(pq_v100), "PaperX (GPT-5.6-Sol)": strip_priv(pq_p100), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([pq_v100["_per_p"][p] for p in all_100], [pq_p100["_per_p"][p] for p in all_100])},
            "VisionBench": {"VASP (GPT-5.6-Sol)": strip_priv(vb_v100), "PaperX (GPT-5.6-Sol)": strip_priv(vb_p100), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([vb_v100["_per_p"][p] for p in all_100], [vb_p100["_per_p"][p] for p in all_100])},
            "InfoCoverage": {"VASP (GPT-5.6-Sol)": strip_priv(ic_v100), "PaperX (GPT-5.6-Sol)": strip_priv(ic_p100), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([ic_v100["_per_p"][p] for p in all_100], [ic_p100["_per_p"][p] for p in all_100])},
            "CEG": {"VASP (GPT-5.6-Sol)": strip_priv(ceg_v100), "PaperX (GPT-5.6-Sol)": strip_priv(ceg_p100), "Paired_VASP_vs_PaperX": paired_bootstrap_ci([ceg_v100["_per_p"][p] for p in all_100], [ceg_p100["_per_p"][p] for p in all_100])},
        }

    out_file = EVAL_DIR / f"summary_paperx_{len(papers)}papers.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_100 = EVAL_DIR / "summary_paperx_100papers.json"
    out_100.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Saved full 5-benchmark summary -> {out_file} and {out_100}")
    return out_100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    papers = stage_ready_papers()
    if args.summarize_only:
        summarize_results(papers)
        return 0

    key_file = str(EVAL_DIR / "_code" / "dummy_key.txt")
    logs_dir = EVAL_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    vendor = str(ROOT / "backup/rebuttal_packages/opus5-baselines-20-pkg-20261107/vendor")
    code_dir = str(EVAL_DIR / "_code")
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{code_dir}:{vendor}:{old_pp}" if old_pp else f"{code_dir}:{vendor}"
    w_str = str(args.max_workers)

    log(f"[1/5] PaperQuiz ({len(papers)} papers, direct codex exec)")
    run_step([
        PY_POSTERGEN, str(EVAL_DIR / "_code/paperquiz/eval_paperquiz_upstream_api.py"),
        "--input-root", str(EVAL_DIR / "inputs"),
        "--groups", METHOD,
        "--output-dir", str(EVAL_DIR / "paperquiz/results"),
        "--model", "gpt-5.6-sol",
        "--reasoning-effort", "medium",
        "--base-url", "codex-direct://local",
        "--api-key-file", key_file,
        "--max-workers", w_str,
        "--resume",
    ], logs_dir / "paperquiz.log", env)

    log(f"[2/5] InfoCoverage ({len(papers)} papers, direct codex exec)")
    run_step([
        PY_POSTERGEN, str(EVAL_DIR / "_code/infocoverage/eval_infocoverage_20.py"),
        "--base-url", "codex-direct://local",
        "--max-workers", w_str,
        "--methods", METHOD,
        "--output-dir", str(EVAL_DIR / "infocoverage"),
    ], logs_dir / "infocoverage.log", env)

    log(f"[3/5] SixDim VLM-as-Judge ({len(papers)} papers, direct codex exec)")
    run_step([
        PY_POSTERGEN, str(EVAL_DIR / "_code/sixdim/eval_sixdim_upstream_api.py"),
        "--input-root", str(EVAL_DIR / "inputs"),
        "--versions", METHOD,
        "--output-dir", str(EVAL_DIR / "sixdim/results"),
        "--model", "gpt-5.6-sol",
        "--reasoning-effort", "medium",
        "--base-url", "codex-direct://local",
        "--api-key-file", key_file,
        "--max-workers", w_str,
        "--resume",
    ], logs_dir / "sixdim.log", env)

    log(f"[4/5] VisionBench ({len(papers)} papers, native eval_poster.py)")
    pngs = [str(EVAL_DIR / "inputs" / METHOD / f"{p}.png") for p in papers]
    run_step([
        PY_POSTERGEN, str(EVAL_DIR / "_code/visionbench/vision_bench/eval_poster.py"),
        "--poster", *pngs,
        "--output-dir", str(EVAL_DIR / "visionbench/results" / METHOD),
        "--model", "gpt-5.6-sol",
        "--reasoning-effort", "medium",
        "--ignore-user-config",
        "--max-workers", w_str,
        "--resume",
    ], logs_dir / "visionbench.log", env)

    log(f"[5/5] CEG ({len(papers)} papers, direct codex exec)")
    run_step([
        PY_POSTERGEN, str(EVAL_DIR / "_code/eval_ceg_codex.py"),
        "--base-url", "codex-direct://local",
        "--eval-dir", str(EVAL_DIR),
        "--method", METHOD,
        "--max-workers", w_str,
    ], logs_dir / "ceg.log", env)

    out_summary = summarize_results(papers)
    log(f"All 5 benchmarks finished! Summary at {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
