#!/usr/bin/env python3
"""Asset-extraction orchestration.

Each paper is processed as:
  1. one semantic LLM call over the existing numbered text-line input, producing
     line roles, structured tables/equations/algorithms, and metadata;
  2. the existing caption-anchored cropper, whose Figure logic is unchanged;
  3. deterministic text/caption extraction and metadata normalization;
  4. deterministic, best-effort institution-logo retrieval and QR generation;
  5. a poster-agnostic bundle. Table crops remain diagnostic intermediates while
     structured tables are the public asset used for later re-rendering.

Poster specification, layout, and rendering intentionally do not belong here.
Logos and QR codes are neutral identity/link assets and are bundled for later use.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
import threading
from collections import Counter
from pathlib import Path

PIPE = Path(__file__).resolve().parent
SCRIPTS = {
    "semantic": PIPE / "extract" / "caption_cut" / "segment_roles.py",
    "crop": PIPE / "extract" / "caption_cut" / "caption_cut.py",
    "text": PIPE / "text" / "extract_text.py",
    "captions": PIPE / "captions" / "extract_captions.py",
    "metadata": PIPE / "metadata" / "extract_metadata.py",
    "enrich": PIPE / "metadata" / "enrich_metadata.py",
    "logos": PIPE / "logos" / "fetch_logos.py",
    "qr": PIPE / "qr" / "make_qr.py",
    "bundle": PIPE / "bundle" / "build_package.py",
}
MODULE_ORDER = [
    "semantic", "crop", "text", "captions", "metadata", "enrich",
    "logos", "qr", "bundle",
]

# Subprocesses inherit the current environment; LLM submodules invoke isolated codex exec.
ENV = os.environ.copy()


def _nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


def _tail(s: str, n: int = 240) -> str:
    return " ".join((s or "").split())[-n:]


def _supports_flag(script: Path, flag: str) -> bool:
    """Inspect --help so optional flags are forwarded only when supported by the subscript."""
    try:
        p = subprocess.run([sys.executable, str(script), "--help"],
                           capture_output=True, text=True, timeout=30, env=ENV)
    except Exception:
        return False
    return flag in (p.stdout + p.stderr)


def _run(cmd, log_path: Path):
    """Run a module subprocess, teeing stdout/stderr to a log file; return (proc, seconds)."""
    t0 = time.monotonic()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV)
    sec = time.monotonic() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(str(c) for c in cmd) + "\n\n"
        "=== STDOUT ===\n" + (proc.stdout or "") + "\n"
        "=== STDERR ===\n" + (proc.stderr or "") + "\n",
        encoding="utf-8",
    )
    return proc, sec


def _skip(note):
    return {"status": "skip", "seconds": 0.0, "note": note}


def _cached(note):
    return {"status": "cached", "seconds": 0.0, "note": note}


def _exec(name, cmd, log_path, ok_rcs=(0,), note_ok=""):
    """Run a module and classify ok/fail by return code, preserving a stderr summary on failure."""
    proc, sec = _run(cmd, log_path)
    if proc.returncode in ok_rcs:
        return {"status": "ok", "seconds": sec, "note": note_ok}
    err = _tail(proc.stderr) or _tail(proc.stdout) or "(no stderr)"
    return {"status": "fail", "seconds": sec, "note": f"rc={proc.returncode}: {err}"}


def run_paper(paper_dir: Path, out_root: Path, figures_root, opt):
    """Run all modules for one paper and return {"slug", "modules": {module: {status, seconds, note}}}."""
    paper_started = time.monotonic()
    slug = paper_dir.name
    pdf = paper_dir / "paper.pdf"
    work = out_root / "_work" / slug
    logs = work / "logs"
    force = opt["force"]
    def forced(m):
        return ("all" in force or m in force or
                ("semantic" in force and m in {"crop", "captions", "metadata", "enrich", "logos", "qr", "bundle"}))

    mods = {}

    # ---- Single-pass semantic call: line roles + structured elements, followed by caption-cut cropping ----
    if figures_root:
        fig_src = figures_root / slug
        seg_file = fig_src / "segments2.json"
        structured_file = fig_src / "structured_elements.json"
        source_lines_file = fig_src / "source_lines.json"
        metadata_candidate = fig_src / "metadata_candidate.json"
        outer_candidates = fig_src / "outer_boundary_candidates.json"
        outer_boundaries = fig_src / "page_outer_boundaries.json"

        if all(_nonempty(path) for path in (
                seg_file, structured_file, source_lines_file, metadata_candidate,
                outer_candidates, outer_boundaries)):
            mods["semantic"] = _cached("external semantic outputs")
        else:
            mods["semantic"] = _skip(f"external element source incomplete: {fig_src}")

        crop_manifest = fig_src / "manifest.json"
        if _nonempty(crop_manifest):
            mods["crop"] = _cached("external manifest.json")
        else:
            mods["crop"] = _skip(f"external crop source incomplete: {fig_src}")
    else:
        fig_src = work / "elements"
        fig_src.mkdir(parents=True, exist_ok=True)
        seg_file = fig_src / "segments2.json"
        structured_file = fig_src / "structured_elements.json"
        source_lines_file = fig_src / "source_lines.json"
        metadata_candidate = fig_src / "metadata_candidate.json"
        outer_candidates = fig_src / "outer_boundary_candidates.json"
        outer_boundaries = fig_src / "page_outer_boundaries.json"
        crop_manifest = fig_src / "manifest.json"

        if (all(_nonempty(p) for p in (
                seg_file, structured_file, source_lines_file, metadata_candidate,
                outer_candidates, outer_boundaries))
                and not forced("semantic")):
            mods["semantic"] = _cached("semantic outputs")
        elif opt["skip_llm"]:
            mods["semantic"] = _skip("--skip-llm")
        else:
            cmd = [sys.executable, SCRIPTS["semantic"], paper_dir, fig_src,
                   "--model", opt["model"], "--effort", opt["effort"],
                   "--backend", opt.get("backend", "auto"), "--provider", opt.get("provider", "openai-compat"),
                   "--timeout", opt["timeout"], "--max-attempts", opt["max_attempts"]]
            if opt["include_appendix"]:
                cmd.append("--include-appendix")
            if forced("semantic"):
                cmd.append("--force")
            mods["semantic"] = _exec("semantic", cmd, logs / "semantic.log")

        if not all(_nonempty(p) for p in (
                seg_file,structured_file,source_lines_file,outer_boundaries)):
            mods["crop"] = _skip("dep: semantic outputs unavailable")
        elif _nonempty(crop_manifest) and not forced("crop"):
            mods["crop"] = _cached("manifest.json")
        else:
            crop_cmd = [sys.executable, SCRIPTS["crop"], paper_dir, fig_src]
            if opt["include_appendix"]:
                crop_cmd.append("--include-appendix")
            with opt["crop_semaphore"]:
                mods["crop"] = _exec("crop", crop_cmd, logs / "crop.log")

    # ---- text (deterministic, no LLM) ----
    text_out = work / "text"
    text_txt = text_out / "text.txt"
    if _nonempty(text_txt) and not forced("text"):
        mods["text"] = _cached("text.txt")
    else:
        text_out.mkdir(parents=True, exist_ok=True)
        mods["text"] = _exec("text", [sys.executable, SCRIPTS["text"], pdf, text_out],
                             logs / "text.log")

    # ---- captions (deterministic; --seg auto-probed) ----
    cap_out = work / "captions"
    cap_json = cap_out / "captions.json"
    if _nonempty(cap_json) and not forced("captions"):
        mods["captions"] = _cached("captions.json")
    else:
        cap_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, SCRIPTS["captions"], pdf, cap_out]
        note = "geometry"
        if opt["caps_seg"] and seg_file and seg_file.exists():
            cmd += ["--seg", str(seg_file)]
            note = "--seg"
        if opt["include_appendix"]:
            cmd.append("--include-appendix")
            note += "+appendix"
        mods["captions"] = _exec("captions", cmd, logs / "captions.log", note_ok=note)

    # ---- metadata (deterministic; cleans candidate from the semantic pass) ----
    meta_out = work / "metadata"
    meta_json = meta_out / "metadata.json"
    if not _nonempty(metadata_candidate):
        mods["metadata"] = _skip("dep: metadata_candidate.json unavailable")
    elif _nonempty(meta_json) and not forced("metadata"):
        mods["metadata"] = _cached("metadata.json")
    else:
        meta_out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, SCRIPTS["metadata"], pdf, metadata_candidate, meta_out]
        mods["metadata"] = _exec("metadata", cmd, logs / "metadata.log")

    meta_ok = _nonempty(meta_json)

    # ---- enrich (no LLM; fills missing neutral metadata via public scholarly APIs) ----
    # Depends on metadata.json; cache signal is the "_enrich" key (best-effort:
    # if no source matches, _enrich is not written and the step retries next run).
    # --skip-llm does not skip enrich (zero LLM calls), unless metadata is absent.
    if not meta_ok:
        mods["enrich"] = _skip("no metadata")
    else:
        meta_doc = _load_json(meta_json)
        if isinstance(meta_doc, dict) and "_enrich" in meta_doc and not forced("enrich"):
            mods["enrich"] = _cached("metadata._enrich")
        else:
            mods["enrich"] = _exec("enrich", [sys.executable, SCRIPTS["enrich"], meta_json],
                                   logs / "enrich.log")

    # ---- neutral brand/link assets (zero LLM, best-effort) ----
    logos_out = work / "logos"
    qr_out = work / "qr"
    if not meta_ok:
        mods["logos"] = _skip("no metadata")
        mods["qr"] = _skip("no metadata")
    else:
        meta_doc = _load_json(meta_json)
        if isinstance(meta_doc, dict) and "logos" in meta_doc and not forced("logos"):
            mods["logos"] = _cached("metadata.logos")
        else:
            logos_out.mkdir(parents=True, exist_ok=True)
            mods["logos"] = _exec(
                "logos",
                [sys.executable, SCRIPTS["logos"], meta_json, logos_out],
                logs / "logos.log",
            )

        meta_doc = _load_json(meta_json)
        if isinstance(meta_doc, dict) and "qr" in meta_doc and not forced("qr"):
            mods["qr"] = _cached("metadata.qr")
        else:
            qr_out.mkdir(parents=True, exist_ok=True)
            mods["qr"] = _exec(
                "qr",
                [sys.executable, SCRIPTS["qr"], meta_json, qr_out],
                logs / "qr.log",
                ok_rcs=(0, 1),
                note_ok="best-effort",
            )

    # ---- bundle (rc 0=complete / 1=INCOMPLETE both count as ok; only rc 2 is hard fail) ----
    bundle_out = out_root / slug
    manifest = bundle_out / "manifest.json"
    if _nonempty(manifest) and not forced("bundle"):
        mods["bundle"] = _cached("manifest.json")
    else:
        bcmd = [sys.executable, SCRIPTS["bundle"], "--out", bundle_out]
        if fig_src and fig_src.exists():
            bcmd += ["--figures-dir", fig_src]
        if _nonempty(text_txt):
            bcmd += ["--text", text_txt]
        if _nonempty(cap_json):
            bcmd += ["--captions", cap_json]
        if _nonempty(meta_json):
            bcmd += ["--metadata", meta_json]
        if logos_out.is_dir():
            bcmd += ["--logos-dir", logos_out]
        if qr_out.is_dir():
            bcmd += ["--qr-dir", qr_out]
        if pdf.exists():
            bcmd += ["--pdf", pdf]
        proc, sec = _run(bcmd, logs / "bundle.log")
        if proc.returncode == 0:
            mods["bundle"] = {"status": "ok", "seconds": sec, "note": "complete"}
        elif proc.returncode == 1:
            inc = (_load_json(manifest) or {}).get("incomplete") if manifest.exists() else None
            note = f"INCOMPLETE(missing {','.join(inc)})" if inc else "INCOMPLETE"
            mods["bundle"] = {"status": "ok", "seconds": sec, "note": note}
        else:
            mods["bundle"] = {"status": "fail", "seconds": sec,
                              "note": f"rc={proc.returncode}: {_tail(proc.stderr)}"}

    elapsed = time.monotonic() - paper_started
    usage_summary = _write_usage_summary(
        bundle_out,
        [fig_src / "semantic_usage.json"],
        model=opt["model"],
        effort=opt["effort"],
        pipeline_seconds=elapsed,
    )
    return {
        "slug": slug,
        "seconds": round(elapsed, 3),
        "estimated_api_cost_usd": usage_summary["estimated_api_cost_usd"],
        "modules": mods,
    }


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_usage_summary(
    bundle_out: Path,
    usage_paths,
    *,
    model: str,
    effort: str,
    pipeline_seconds: float,
):
    """Account for all attempts; unknown failed-call usage is not billed as zero."""
    calls = []
    for path in usage_paths:
        attempts = _load_json(path.with_name("semantic_attempts.json"))
        if isinstance(attempts, dict):
            for item in attempts.get("calls", []):
                usage = item.get("usage")
                calls.append({"stage": "semantic", "attempt": item["attempt"],
                              "accepted": item["accepted"],
                              "format_failure": item["format_failure"],
                              "call_failure": item["call_failure"],
                              "usage_available": isinstance(usage, dict),
                              **(usage if isinstance(usage, dict) else {})})
        else:
            item = _load_json(path)
            if isinstance(item, dict):
                calls.append({"stage": path.stem.removesuffix("_usage"),
                              "usage_available": True, **item})
    number_fields = (
        "duration_seconds",
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "estimated_api_cost_usd",
    )
    summary = {
        "backend": next((c["backend"] for c in calls if c.get("backend")), "unknown"),
        "model": model,
        "reasoning_effort": effort,
        "pipeline_seconds": round(pipeline_seconds, 3),
        "call_count": len(calls),
        "calls_without_usage": sum(not c["usage_available"] for c in calls),
        "format_failures": sum(c.get("format_failure", False) for c in calls),
        "call_failures": sum(c.get("call_failure", False) for c in calls),
        "cost_note": "API-equivalent estimate for recorded usage only; failed calls without usage are unknown.",
        **{
            field: round(sum(float(call.get(field, 0)) for call in calls), 8)
            if field in ("duration_seconds", "estimated_api_cost_usd")
            else sum(int(call.get(field, 0)) for call in calls)
            for field in number_fields
        },
        "pricing": next((c["pricing"] for c in calls if c.get("pricing")), None),
        "calls": calls,
    }
    bundle_out.mkdir(parents=True, exist_ok=True)
    (bundle_out / "llm_usage.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def print_table(results):
    """Print an aligned summary table to stdout: rows=papers, columns=module statuses."""
    disp = {"ok": "ok", "cached": "cache", "skip": "skip", "fail": "FAIL"}
    slug_w = min(42, max((len(r["slug"]) for r in results), default=4))
    col_w = {m: max(len(m), 5) for m in MODULE_ORDER}
    header = f"{'paper':<{slug_w}}  " + "  ".join(f"{m:<{col_w[m]}}" for m in MODULE_ORDER)
    print("\n" + header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x["slug"]):
        cells = []
        for m in MODULE_ORDER:
            st = r["modules"].get(m, {}).get("status", "-")
            cells.append(f"{disp.get(st, st):<{col_w[m]}}")
        print(f"{r['slug'][:slug_w]:<{slug_w}}  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("out_root")
    ap.add_argument("--figures-root", default=None,
                    help="Optional precomputed elements root (default: <out-root>/_work/<paper>/elements). "
                         "Must contain segments2, structured_elements, metadata_candidate, "
                         "source_lines, and manifest JSON files.")
    ap.add_argument("--papers", default=None, help="Comma-separated title keywords to filter papers (case-insensitive)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--crop-jobs", type=int, default=12, help="Max concurrent cropping workers (independent of LLM jobs)")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-call LLM timeout in seconds")
    ap.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    ap.add_argument("--skip-llm", action="store_true", help="Skip the semantic LLM call")
    ap.add_argument("--include-appendix", action="store_true",
                    help="Include appendix pages in the semantic pass; References are always excluded")
    ap.add_argument("--force", default="", help="Comma-separated module names to force rerun, or all")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--backend", choices=["auto", "codex", "claude", "agy", "pi-api", "pi-chatgpt"], default="auto")
    ap.add_argument("--provider", default="openai-compat")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    if a.jobs < 1 or not 1 <= a.crop_jobs <= 12 or a.timeout < 1:
        ap.error("jobs must be positive, crop-jobs must be 1..12, timeout must be positive")

    data = Path(a.data_dir)
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    figures_root = Path(a.figures_root) if a.figures_root else None

    papers = sorted([d for d in data.iterdir() if (d / "paper.pdf").exists()])
    if a.papers:
        kws = [k.strip().lower() for k in a.papers.split(",") if k.strip()]
        papers = [p for p in papers if any(k in p.name.lower() for k in kws)]
    if a.limit:
        papers = papers[:a.limit]
    if not papers:
        print("No matching papers found (check data_dir / --papers / --limit)", file=sys.stderr)
        return 1

    force = {m.strip() for m in a.force.split(",") if m.strip()}
    opt = {
        "skip_llm": a.skip_llm,
        "include_appendix": a.include_appendix,
        "force": force,
        "model": a.model, "backend": a.backend, "provider": a.provider,
        "effort": a.effort,
        "timeout": a.timeout, "max_attempts": a.max_attempts,
        "crop_semaphore": threading.BoundedSemaphore(a.crop_jobs),
        # Probe optional subscript flags once before spawning workers
        "caps_seg": _supports_flag(SCRIPTS["captions"], "--seg"),
    }

    print(f"{len(papers)} papers -> {out_root}  (jobs={a.jobs}, skip_llm={a.skip_llm}, "
          f"include_appendix={a.include_appendix}, external_elements={figures_root}, "
          f"force={sorted(force) or 'none'})")
    print(f"Capability probe: captions --seg={opt['caps_seg']}  metadata=deterministic")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_paper, p, out_root, figures_root, opt): p for p in papers}
        for fu in concurrent.futures.as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:  # Surface unexpected run_paper exceptions as paper-level failures
                p = futs[fu]
                r = {"slug": p.name, "modules": {"_orchestrator": {
                    "status": "fail", "seconds": 0.0, "note": f"run_paper error: {e!r}"}}}
            results.append(r)
            n_fail = sum(1 for v in r["modules"].values() if v["status"] == "fail")
            tag = "OK " if n_fail == 0 else "ERR"
            print(f"  [{tag}] {r['slug'][:48]:48} " + " ".join(
                f"{m}:{r['modules'][m]['status']}" for m in MODULE_ORDER if m in r["modules"]))

    counts = Counter()
    for r in results:
        for v in r["modules"].values():
            counts[v["status"]] += 1
    fails = [(r["slug"], m, v["note"]) for r in results for m, v in r["modules"].items()
             if v["status"] == "fail"]

    summary = {
        "config": {"data_dir": str(data), "out_root": str(out_root),
                   "figures_root": str(figures_root), "jobs": a.jobs, "skip_llm": a.skip_llm,
                   "include_appendix": a.include_appendix, "force": sorted(force),
                   "model": a.model, "effort": a.effort,
                   "timeout": a.timeout, "max_attempts": a.max_attempts, "crop_jobs": a.crop_jobs,
                   "caps_seg": opt["caps_seg"], "metadata_backend": "deterministic"},
        "n_papers": len(results),
        "counts": dict(counts),
        "papers": sorted(results, key=lambda x: x["slug"]),
    }
    (out_root / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_table(results)
    print(f"\n=== Summary: {len(results)} papers  " + "  ".join(
        f"{k}={counts.get(k, 0)}" for k in ("ok", "cached", "skip", "fail")) + " ===")
    if fails:
        print(f"FAIL details ({len(fails)}):")
        for slug, m, note in fails:
            print(f"  - [{m}] {slug[:44]}: {note}")
    print(f"→ {out_root}/batch_summary.json")

    return 1 if counts.get("fail", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
