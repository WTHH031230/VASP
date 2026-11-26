#!/usr/bin/env python3
"""Batch runner for the standalone w/o Semantic Verification (Geometry-Only) ablation.

Defaults to the 20-paper ablation benchmark (10 headroom + 10 ext papers).

Usage:
  python -m src.ablation.batch_wo_sv --out results/exp/pi-ablation-wo-sv-20-20261108 [--concurrency 5]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = REPO / "outputs" / "semparser"

DEFAULT_20_PAPERS = [
    "Differentially Private CutMix for Split Learning with Vision Transformer",
    "Inner Classifier-Free Guidance and Its Taylor Expansion for Diffusion Models",
    "Towards Reasoning-Aware Explainable VQA",
    "All in a Row: Compressed Convolution Networks for Graphs",
    "Butterfly Effects of SGD Noise: Error Amplification in Behavior Cloning and Autoregression",
    "CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing",
    "Near-real-time monitoring of global ocean carbon sink",
    "UniTabE: A Universal Pretraining Protocol for Tabular Foundation  Model in Data Science",
    "Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger",
    "[Re] Graph Edit Networks",
    "Quantifying Network Similarity using Graph Cumulants",
    "Collective Certified Robustness against Graph Injection Attacks",
    "DRAGONN: Distributed Randomized Approximate Gradients of Neural Networks",
    "Diving into Underwater: Segment Anything Model Guided Underwater Salient Instance Segmentation and A Large-scale Dataset",
    "Revisiting Robustness in Graph Machine Learning",
    "A Theoretical Analysis of the Test Error of Finite-Rank Kernel Ridge Regression",
    "A View From Somewhere: Human-Centric Face Representations",
    "Accurate and Scalable Estimation of Epistemic Uncertainty for Graph Neural Networks",
    "An Infinite-Width Analysis on the Jacobian-Regularised Training of a Neural Network",
    "CABINET: Content Relevance-based Noise Reduction for Table Question Answering",
]


def run_one(*, group: str, paper: str, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = out / "runs" / group / paper
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        return {
            "arm": "wo_sv",
            "paper": paper,
            "status": "done",
            "attempts": 0,
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        }
    log_dir = out / "logs" / group
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.max_attempts + 1):
        if run_dir.exists():
            failed = out / "failed" / group / f"{paper}__attempt{attempt - 1}_{int(time.time())}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(failed)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "src.ablation.run_wo_sv",
            "--model", args.model,
            "--provider", args.provider,
            "--effort", args.effort,
            "--max-revisions", str(args.max_revisions),
            "--timeout", str(args.timeout),
            "--asset-dir", str(args.assets_root / paper),
            "--output-dir", str(run_dir),
        ]
        with (log_dir / f"{paper}.attempt{attempt}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if code == 0 and manifest_path.is_file():
            return {
                "arm": "wo_sv",
                "paper": paper,
                "status": "done",
                "attempts": attempt,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        time.sleep(min(60, 10 * attempt))
    return {"arm": "wo_sv", "paper": paper, "status": "failed", "attempts": args.max_attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--papers-file", type=Path, default=None)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT, help="Root directory containing per-paper SemParser asset bundles")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", default="openai-compat")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    papers = json.loads(args.papers_file.read_text(encoding="utf-8")) if args.papers_file else DEFAULT_20_PAPERS
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping_path = out / "arm_dirs.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {"wo_sv": "g1"}
    mapping.setdefault("wo_sv", "g1")
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    (out / "papers.json").write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    group = mapping["wo_sv"]
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_one, group=group, paper=paper, out=out, args=args): paper
            for paper in papers
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            m = result.get("manifest") or {}
            print(json.dumps({
                "arm": "wo_sv",
                "paper": result["paper"][:50],
                "status": result["status"],
                "attempts": result["attempts"],
                "selected": m.get("selected_checkpoint"),
                "stop": m.get("stop_reason"),
                "cost": m.get("api_equivalent_cost_usd"),
                "wall": m.get("total_wall_seconds"),
            }, ensure_ascii=False), flush=True)

    summary = {
        "wall_seconds": round(time.monotonic() - started, 1),
        "arms": {
            "wo_sv": {
                "done": sum(r["status"] == "done" for r in results),
                "failed": [r["paper"] for r in results if r["status"] == "failed"],
                "retried": sum(r["attempts"] > 1 for r in results),
                "cost_usd": round(sum(float((r.get("manifest") or {}).get("api_equivalent_cost_usd") or 0) for r in results), 4),
            }
        },
        "runs": [
            {k: v for k, v in r.items() if k != "manifest"} | {
                "selected": (r.get("manifest") or {}).get("selected_checkpoint"),
                "stop": (r.get("manifest") or {}).get("stop_reason"),
                "cost": (r.get("manifest") or {}).get("api_equivalent_cost_usd"),
                "wall": (r.get("manifest") or {}).get("total_wall_seconds"),
            }
            for r in results
        ],
    }
    (out / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    return 0 if all(r["status"] == "done" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
