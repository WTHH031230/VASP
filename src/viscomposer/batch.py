#!/usr/bin/env python3
"""Run arms x papers on pi.

Run directories use neutral group names (g1..gN) because image paths inside the run
directory enter the prompt; the arm mapping is written to <out>/arm_dirs.json only.
Papers are queued arm-interleaved so every arm runs in the same period. A failed run is
moved to <out>/failed/ and restarted from scratch, up to --max-attempts. Finished runs
(run_manifest.json present) are skipped, so the batch can be resumed.

  python -m src.viscomposer.batch --out DIR --papers-file FILE [--arms a,b,...] [--concurrency 5]
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

from src.viscomposer.arms import ARMS

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS_ROOT = REPO / "outputs" / "semparser"
DEFAULT_ARMS = ("full", "no_assets", "no_vf", "no_sem", "no_basic")


def run_one(*, arm: str, group: str, paper: str, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = out / "runs" / group / paper
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        return {"arm": arm, "paper": paper, "status": "done", "attempts": 0,
                "manifest": json.loads(manifest_path.read_text())}
    log_dir = out / "logs" / group
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.max_attempts + 1):
        if run_dir.exists():
            failed = out / "failed" / group / f"{paper}__attempt{attempt - 1}_{int(time.time())}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            run_dir.rename(failed)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "src.viscomposer.run", "--arm", arm, "--designer", "pi",
                   "--effort", args.effort, "--max-revisions", str(args.max_revisions),
                   "--timeout", str(args.timeout),
                   "--asset-dir", str(args.assets_root / paper),
                   "--output-dir", str(run_dir)]
        if args.provider:
            command.extend(["--provider", args.provider])
        if args.model:
            command.extend(["--model", args.model])
        with (log_dir / f"{paper}.attempt{attempt}.log").open("w") as log:
            code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if code == 0 and manifest_path.is_file():
            return {"arm": arm, "paper": paper, "status": "done", "attempts": attempt,
                    "manifest": json.loads(manifest_path.read_text())}
        time.sleep(min(60, 10 * attempt))
    return {"arm": arm, "paper": paper, "status": "failed", "attempts": args.max_attempts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--papers-file", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT, help="Root directory containing per-paper SemParser asset bundles")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)  # API turns take 1-2 min; a hung request restarts sooner
    args = parser.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        parser.error(f"unknown arms: {unknown}")
    papers = json.loads(args.papers_file.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping_path = out / "arm_dirs.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    for arm in arms:
        mapping.setdefault(arm, f"g{len(mapping) + 1}")
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    (out / "papers.json").write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n")

    jobs = [(arm, paper) for paper in papers for arm in arms]  # arm-interleaved
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, arm=arm, group=mapping[arm], paper=paper, out=out, args=args): (arm, paper)
                   for arm, paper in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            m = result.get("manifest") or {}
            print(json.dumps({"arm": result["arm"], "paper": result["paper"][:50], "status": result["status"],
                              "attempts": result["attempts"], "selected": m.get("selected_checkpoint"),
                              "stop": m.get("stop_reason"), "cost": m.get("api_equivalent_cost_usd"),
                              "wall": m.get("total_wall_seconds")}, ensure_ascii=False), flush=True)
    summary = {
        "wall_seconds": round(time.monotonic() - started, 1),
        "arms": {arm: {
            "done": sum(r["status"] == "done" for r in results if r["arm"] == arm),
            "failed": [r["paper"] for r in results if r["arm"] == arm and r["status"] == "failed"],
            "retried": sum(r["attempts"] > 1 for r in results if r["arm"] == arm),
            "cost_usd": round(sum(float((r.get("manifest") or {}).get("api_equivalent_cost_usd") or 0)
                                  for r in results if r["arm"] == arm), 4),
        } for arm in arms},
        "runs": [{k: v for k, v in r.items() if k != "manifest"} | {
            "selected": (r.get("manifest") or {}).get("selected_checkpoint"),
            "stop": (r.get("manifest") or {}).get("stop_reason"),
            "cost": (r.get("manifest") or {}).get("api_equivalent_cost_usd"),
            "wall": (r.get("manifest") or {}).get("total_wall_seconds"),
        } for r in results],
    }
    (out / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary["arms"], ensure_ascii=False, indent=2))
    return 0 if all(r["status"] == "done" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
