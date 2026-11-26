#!/usr/bin/env python3
"""Concurrent batch runner for single-pass semantic extraction + deterministic caption_cut.

- Each paper gets its own output directory with segments2/source_lines, figures/, and report.json.
- Cache reuse (--seg-root) requires co-produced segments/source/structured/outer-boundary files.
- Summarizes per-paper figure/table counts and invariant flags (I1-I4, R3 adoption, cross-page adoption).

Live LLM runs use isolated codex exec and record call traces/usage per paper;
cached runs (--seg-root) require no LLM calls (~1-2 min for 100 papers).
Usage: python run_caption_batch.py <data_dir> <out_root> [--jobs 12] [--limit N]
       [--seg-root DIR] [--include-appendix]
       [--model gpt-5.6-sol] [--effort medium]
"""
import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_seg_cache(seg_root, name):
    if not seg_root:
        return None
    for key in (name, name[:30]):
        for rel in ((key, "segments2.json"),
                    (key, "elements", "segments2.json"),
                    (key, "assets", "meta", "segments2.json")):
            path = seg_root.joinpath(*rel)
            if path.exists():
                return path
    return None


def run_one(paper_dir, out_root, seg_root, model, effort, include_appendix,
            disable_outer_boundaries,disable_table_bands, timeout=1800, max_attempts=2, crop_semaphore=None):
    outdir = out_root / paper_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    seg = outdir / "segments2.json"
    structured = outdir / "structured_elements.json"
    source_lines = outdir / "source_lines.json"
    metadata_candidate = outdir / "metadata_candidate.json"
    outer_candidates = outdir / "outer_boundary_candidates.json"
    outer_boundaries = outdir / "page_outer_boundaries.json"
    semantic_required = (
        seg,structured,source_lines,metadata_candidate,
        outer_candidates,outer_boundaries)
    if not all(path.exists() for path in semantic_required):
        cache = find_seg_cache(seg_root, paper_dir.name)
        cache_source_lines = cache.with_name("source_lines.json") if cache else None
        cache_structured = cache.with_name("structured_elements.json") if cache else None
        cache_metadata = cache.with_name("metadata_candidate.json") if cache else None
        cache_outer_candidates = cache.with_name("outer_boundary_candidates.json") if cache else None
        cache_outer_boundaries = cache.with_name("page_outer_boundaries.json") if cache else None
        if (cache and cache_source_lines.exists() and cache_structured.exists()
                and cache_metadata.exists()
                and cache_outer_candidates.exists() and cache_outer_boundaries.exists()
                ):
            shutil.copy(cache, seg)
            shutil.copy(cache_source_lines, source_lines)
            shutil.copy(cache_structured, structured)
            shutil.copy(cache_metadata, metadata_candidate)
            shutil.copy(cache_outer_candidates, outer_candidates)
            shutil.copy(cache_outer_boundaries, outer_boundaries)
        else:
            semantic_cmd = [
                sys.executable, str(HERE / "segment_roles.py"),
                str(paper_dir), str(outdir),
                "--model", model, "--effort", effort,
                "--timeout", str(timeout), "--max-attempts", str(max_attempts),
            ]
            if include_appendix:
                semantic_cmd.append("--include-appendix")
            p = subprocess.run(semantic_cmd, capture_output=True, text=True)
            if p.returncode != 0 or not all(path.exists() for path in semantic_required):
                return {"slug": paper_dir.name, "rc": p.returncode, "err": "segment_roles failed: " + (p.stdout or p.stderr)[-160:],
                        "n_fig": 0, "n_tab": 0, "flags": {}, "sec": time.monotonic() - t0}
    crop_cmd = [sys.executable, str(HERE / "caption_cut.py"),
                str(paper_dir), str(outdir)]
    if include_appendix:
        crop_cmd.append("--include-appendix")
    if disable_outer_boundaries:
        crop_cmd.append("--disable-outer-boundaries")
    if disable_table_bands:
        crop_cmd.append("--disable-table-bands")
    with crop_semaphore or threading.BoundedSemaphore(1):
        p = subprocess.run(crop_cmd, capture_output=True, text=True)
    # Count persisted assets instead of parsing human-facing stdout. The
    # caption-cut log format is not an API and may change independently.
    n_fig = 0
    n_tab = 0
    manifest = outdir / "manifest.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            kind = str(item.get("kind", "")).lower()
            n_fig += kind == "figure"
            n_tab += kind == "table"
    rep = outdir / "report.json"
    flags = Counter()
    if rep.exists():
        for fl in json.loads(rep.read_text()).get("flags", []):
            flags[fl["kind"]] += 1
    return {"slug": paper_dir.name, "rc": p.returncode, "err": None if p.returncode == 0 else (p.stderr or p.stdout)[-160:],
            "n_fig": n_fig, "n_tab": n_tab, "flags": dict(flags), "sec": time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("out_root")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--crop-jobs", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seg-root", default=None, help="Root directory with precomputed <paper>/segments2.json to reuse")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--include-appendix", action="store_true")
    ap.add_argument("--disable-outer-boundaries", action="store_true",
                    help="Ablation: reuse semantic cache without applying outer page boundaries")
    ap.add_argument("--disable-table-bands", action="store_true",
                    help="Ablation: reuse semantic cache without applying deterministic Table bands")
    a = ap.parse_args()
    if a.jobs < 1 or not 1 <= a.crop_jobs <= 12 or a.timeout < 1:
        ap.error("jobs must be positive, crop-jobs must be 1..12, timeout must be positive")
    crop_semaphore = threading.BoundedSemaphore(a.crop_jobs)
    data = Path(a.data_dir); out_root = Path(a.out_root); out_root.mkdir(parents=True, exist_ok=True)
    seg_root = Path(a.seg_root) if a.seg_root else None
    papers = sorted([d for d in data.iterdir() if (d / "paper.pdf").exists()])
    if a.limit:
        papers = papers[:a.limit]
    print(f"{len(papers)} papers -> {out_root}  (jobs={a.jobs}, seg-root={seg_root})")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {
            ex.submit(run_one, p, out_root, seg_root, a.model, a.effort,
                      a.include_appendix,a.disable_outer_boundaries,
                      a.disable_table_bands, a.timeout, a.max_attempts, crop_semaphore): p
            for p in papers
        }
        for fu in concurrent.futures.as_completed(futs):
            r = fu.result(); results.append(r)
            tag = "OK " if r["rc"] == 0 and not r["err"] else "ERR"
            print(f"  [{tag}] {r['slug'][:40]:40} fig={r['n_fig']:2} tab={r['n_tab']:2} "
                  f"flags={sum(r['flags'].values())} {r['sec']:.0f}s" + (f"  {r['err']}" if r["err"] else ""))
    ok = [r for r in results if r["rc"] == 0 and not r["err"]]
    tot_flags = Counter()
    for r in results:
        for k, v in r["flags"].items():
            tot_flags[k] += v
    print(f"\n=== Summary: {len(ok)}/{len(results)} succeeded  figure={sum(r['n_fig'] for r in results)} "
          f"table={sum(r['n_tab'] for r in results)} ===")
    print("flags:", dict(tot_flags))
    (out_root / "batch_summary.json").write_text(json.dumps(
        {"results": results, "flags_total": dict(tot_flags)}, ensure_ascii=False, indent=2))
    print(f"→ {out_root}/batch_summary.json")


if __name__ == "__main__":
    main()
