#!/usr/bin/env python3
"""Batch runner for the standalone w/o Iterative Semantic Verification (Initial SV + GV) ablation.

Defaults to the 20-paper ablation benchmark (10 headroom + 10 ext papers).

Usage:
  python -m src.ablation.batch_wo_iter_sv --out results/exp/pi-ablation-wo-iter-sv-schemeA-20-20261110 --initial-from-full [--concurrency 8]
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

from src.ablation.batch_wo_sv import DEFAULT_20_PAPERS, DEFAULT_ASSETS_ROOT, REPO


def get_full_run_map() -> dict[str, Path]:
    B = REPO / "results/exp/pi-headroom10-final-20260925"
    first10 = json.loads((REPO / "results/exp/pi-headroom10-20260925/papers.json").read_text(encoding="utf-8"))
    ext10 = json.loads((B / "ext/papers.json").read_text(encoding="utf-8"))
    d1 = json.loads((REPO / "results/exp/pi-headroom10-20260925/arm_dirs.json").read_text(encoding="utf-8"))
    de = json.loads((B / "ext/arm_dirs.json").read_text(encoding="utf-8"))
    m: dict[str, Path] = {}
    for p in first10:
        m[p] = REPO / "results/exp/pi-headroom10-20260925/runs" / d1["full"] / p
    for p in ext10:
        m[p] = B / "ext/runs" / de["full"] / p
    return m


def run_one(*, group: str, paper: str, out: Path, args: argparse.Namespace, full_map: dict[str, Path]) -> dict[str, Any]:
    run_dir = out / "runs" / group / paper
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        return {
            "arm": "wo_iter_sv",
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
            sys.executable, "-m", "src.ablation.run_wo_iter_sv",
            "--model", args.model,
            "--provider", args.provider,
            "--effort", args.effort,
            "--max-revisions", str(args.max_revisions),
            "--timeout", str(args.timeout),
            "--asset-dir", str(args.assets_root / paper),
            "--output-dir", str(run_dir),
        ]
        if args.initial_from_full and paper in full_map:
            command.extend(["--initial-from-run", str(full_map[paper])])
        with (log_dir / f"{paper}.attempt{attempt}.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if code == 0 and manifest_path.is_file():
            return {
                "arm": "wo_iter_sv",
                "paper": paper,
                "status": "done",
                "attempts": attempt,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        time.sleep(min(60, 10 * attempt))
    return {"arm": "wo_iter_sv", "paper": paper, "status": "failed", "attempts": args.max_attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--papers-file", type=Path, default=None)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT, help="Root directory containing per-paper SemParser asset bundles")
    parser.add_argument("--initial-from-full", action="store_true", help="Seed Round 0 from VASP (full) stored 01_initial_design")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", default="openai-compat")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    papers = json.loads(args.papers_file.read_text(encoding="utf-8")) if args.papers_file else DEFAULT_20_PAPERS
    full_map = get_full_run_map() if args.initial_from_full else {}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping_path = out / "arm_dirs.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.is_file() else {"wo_iter_sv": "g1"}
    mapping.setdefault("wo_iter_sv", "g1")
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    (out / "papers.json").write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    group = mapping["wo_iter_sv"]
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_one, group=group, paper=paper, out=out, args=args, full_map=full_map): paper
            for paper in papers
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            m = result.get("manifest") or {}
            print(json.dumps({
                "arm": "wo_iter_sv",
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
            "wo_iter_sv": {
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
