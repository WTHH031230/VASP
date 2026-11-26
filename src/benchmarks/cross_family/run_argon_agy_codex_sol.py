#!/usr/bin/env python3
"""Headless Antigravity CLI (`agy -p`) runner for Codex Sol (`gpt5.6-sol-medium`)
on Table 4 (InfoCoverage, 100 calls) and Table 3 (VisionBench, 1200 calls).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

BASE = Path(os.environ.get("VASP_EXP_DIR", str(Path(__file__).resolve().parents[3] / "results" / "exp")))
OUT_DIR = Path(os.environ.get("VASP_OUT_DIR", "./outputs/cross_family"))
STIM_ROOT = Path("/tmp/vasp_cache")
AGY_BIN = "agy"
AGY_MODEL = "gemini-4-argon-low"
PROXY_URL = ""

CODEX_VERSION = "gpt5.6-sol-medium"
CODEX_POSTER_DIR = BASE / "visionbench-api-9groups-medium-20260923/inputs/gpt5.6-sol-medium"

IC_SNAP = BASE / "infocoverage-api-9groups-medium-20260924/_code/snapshot/information_coverage_bench"
_IC_RUNNER = None


def _get_ic_runner():
    global _IC_RUNNER
    if _IC_RUNNER is None:
        spec = importlib.util.spec_from_file_location("ic_base_runner", IC_SNAP / "six_methods/run_six_methods.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IC_RUNNER = mod
    return _IC_RUNNER


VB_SNAP = Path(__file__).resolve().parents[1] / "vision_bench"
if str(VB_SNAP) not in sys.path:
    sys.path.insert(0, str(VB_SNAP))

from evaluator import (  # type: ignore  # noqa: E402
    VISUAL_EVALUATION_DIMENSIONS,
    build_system_prompt,
    build_user_prompt,
    load_rubrics,
    validate_dimension_response,
)
from eval_poster import (  # type: ignore  # noqa: E402
    completed_ids,
    normalize_report,
    poster_key,
)
from poster_stimuli import prepare_question_poster  # type: ignore  # noqa: E402

STRIP_KEYS = {"$schema", "title", "default", "minLength", "pattern"}
STOP_KEYWORDS = (
    "quota",
    "rate limit",
    "rate_limit",
    "exhausted",
    "authentication",
    "unauthorized",
    "permission_denied",
    "http 429",
    "http 403",
    "http 401",
    "credits",
)

_LAUNCH_LOCK = threading.Lock()
_LAST_LAUNCH_TS = 0.0
_CONSEC_FAILS = 0
_CONSEC_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def strip_schema(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: ({p: strip_schema(s) for p, s in v.items()} if k == "properties" else strip_schema(v))
            for k, v in node.items()
            if k == "properties" or k not in STRIP_KEYS
        }
    if isinstance(node, list):
        return [strip_schema(x) for x in node]
    return node


def pace_launch(min_interval: float) -> None:
    global _LAST_LAUNCH_TS
    if min_interval <= 0:
        return
    with _LAUNCH_LOCK:
        now = time.monotonic()
        wait_s = (_LAST_LAUNCH_TS + min_interval) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _LAST_LAUNCH_TS = time.monotonic()


def invoke_agy_once(
    stim_dir: Path,
    poster_src: Path,
    prompt_text: str,
    schema: dict[str, Any],
    min_interval: float,
) -> dict[str, Any]:
    global _CONSEC_FAILS
    local_poster = stim_dir / "poster.png"
    shutil.copy2(poster_src, local_poster)
    schema_path = stim_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    env = {
        **os.environ,
        "HTTPS_PROXY": PROXY_URL,
        "HTTP_PROXY": PROXY_URL,
        "https_proxy": PROXY_URL,
        "http_proxy": PROXY_URL,
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
    }

    pace_launch(min_interval)
    t0 = time.monotonic()
    proc = subprocess.run(
        [
            AGY_BIN,
            "-p",
            prompt_text,
            "--model",
            AGY_MODEL,
            "--output-format",
            "stream-json",
            "--json-schema",
            str(schema_path),
            "--settings",
            '{"useG1Credits":false}',
            "--disable-slash-commands",
            "--print-timeout",
            "5m",
        ],
        cwd=str(stim_dir),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = round(time.monotonic() - t0, 3)

    saw_read_file = False
    tool_errors: list[str] = []
    final_result: dict[str, Any] | None = None

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("event")
        if etype == "step_update":
            su = ev.get("step_update") or {}
            if su.get("step_type") == "tool" and su.get("state") == "DONE":
                tname = su.get("tool_name", "")
                tinfo = su.get("tool_info") or {}
                if "read_file" in tname or "view_file" in tname:
                    if tinfo.get("error"):
                        tool_errors.append(json.dumps(tinfo["error"], ensure_ascii=False))
                    else:
                        saw_read_file = True
        elif etype == "result":
            final_result = ev.get("result")

    combined_err = ""
    if final_result and final_result.get("status") != "SUCCESS":
        combined_err = str(final_result.get("error") or "")
    elif proc.returncode != 0:
        combined_err = (proc.stderr or proc.stdout or f"rc={proc.returncode}")[-800:]

    if combined_err:
        low_err = combined_err.lower()
        if any(k in low_err for k in STOP_KEYWORDS):
            print(f"[FATAL-QUOTA/AUTH] Stopping immediately without retry: {combined_err[:500]}", file=sys.stderr, flush=True)
            os._exit(1)
        with _CONSEC_LOCK:
            _CONSEC_FAILS += 1
            cf = _CONSEC_FAILS
        if cf >= 10:
            print(f"[FATAL] {cf} consecutive agy failures; stopping: {combined_err[:500]}", file=sys.stderr, flush=True)
            os._exit(1)
        raise RuntimeError(f"agy failed: {combined_err[:400]}")

    if not final_result:
        with _CONSEC_LOCK:
            _CONSEC_FAILS += 1
        raise RuntimeError(f"agy produced no result event; stderr={proc.stderr[-300:]}")

    if not saw_read_file:
        with _CONSEC_LOCK:
            _CONSEC_FAILS += 1
        raise RuntimeError(f"Model did not read poster.png via read_file (tool_errors={tool_errors})")

    parsed = final_result.get("structured_output")
    if not isinstance(parsed, dict):
        resp_txt = (final_result.get("response") or "").strip()
        parsed = json.loads(resp_txt)

    with _CONSEC_LOCK:
        _CONSEC_FAILS = 0

    usage = final_result.get("usage") or {}
    return {
        "parsed": parsed,
        "conversation_id": final_result.get("conversation_id"),
        "usage": {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "thinking_tokens": int(usage.get("thinking_tokens", 0)),
            "cache_read_tokens": int(usage.get("cache_read_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        },
        "wall_seconds": elapsed,
        "completed_at": utc_now(),
    }


def run_infocoverage_once(args: argparse.Namespace) -> int:
    ic_out = OUT_DIR / "infocoverage"
    res_root = ic_out / "results"
    ic_manifest = json.loads((IC_SNAP / "api_nine_methods/manifest.json").read_text(encoding="utf-8"))
    fn_to_id = {row["input_filename"]: str(row["poster_id"]) for row in ic_manifest["posters"]}
    ref_posters = {str(p["poster_id"]): p for p in _get_ic_runner().load_reference()}
    template = (IC_SNAP / "full100/evaluation_prompt_api.md").read_text(encoding="utf-8")
    schema = strip_schema(_get_ic_runner().read_json(_get_ic_runner().SCHEMA_PATH))

    tasks: list[tuple[str, str, dict[str, Any], Path, Path]] = []
    for poster in sorted(CODEX_POSTER_DIR.glob("*.png")):
        paper = poster.stem
        pid = fn_to_id[f"{paper}.png"]
        pref = ref_posters[pid]
        rpath = res_root / CODEX_VERSION / f"{pid}.json"
        if rpath.is_file():
            try:
                existing = json.loads(rpath.read_text(encoding="utf-8"))
                _get_ic_runner().common.validate_evaluation(pref, existing)
                continue
            except Exception:
                pass
        tasks.append((paper, pid, pref, poster, rpath))

    if args.smoke:
        tasks = tasks[:1]

    total = len(tasks)
    print(f"[AGY-InfoCoverage] {CODEX_VERSION} | remaining={total} | workers={args.max_workers} | model={AGY_MODEL}", flush=True)
    if total == 0:
        return 0

    STIM_ROOT.mkdir(parents=True, exist_ok=True)
    seq = 0
    seq_lock = threading.Lock()

    def _do_one(item: tuple[str, str, dict[str, Any], Path, Path]) -> tuple[str, int, float]:
        nonlocal seq
        paper, pid, pref, poster, rpath = item
        with seq_lock:
            seq += 1
            my_seq = seq
        stim_dir = STIM_ROOT / f"ic_{os.getpid()}_{my_seq}_{pid}"
        stim_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompt_raw = _get_ic_runner().build_prompt(template, pref)
            poster_p = stim_dir / "poster.png"
            full_prompt = "\n\n".join(
                [
                    "EVALUATION-ONLY KEY-INFORMATION COVERAGE JUDGEMENT",
                    f"1. Call `read_file` ONCE on `{poster_p}` to visually inspect the candidate poster image.",
                    "2. Do NOT inspect any other files, do NOT run shell commands, do NOT search, and do NOT call any other tools.",
                    "3. That candidate poster image is the ONLY visual evidence. Judge all 7 frozen units strictly per the instructions below.",
                    prompt_raw,
                    "Return your final response strictly conforming to the enforced JSON Schema.",
                ]
            )
            img_sha = hashlib.sha256(poster.read_bytes()).hexdigest()
            rec = invoke_agy_once(stim_dir, poster, full_prompt, schema, args.min_interval)
            result = rec["parsed"]
            _get_ic_runner().common.validate_evaluation(pref, result)
            result["metadata"] = {
                "target_method": CODEX_VERSION,
                "target_paper": paper,
                "target_image": str(poster.resolve()),
                "image_sha256": img_sha,
                "backend": "antigravity_cli_headless",
                "model": AGY_MODEL,
                "thinking_level": "LOW",
                "elapsed_seconds": rec["wall_seconds"],
                "tokens": rec["usage"],
                "conversation_id": rec["conversation_id"],
                "completed_at": rec["completed_at"],
            }
            atomic_write_json(rpath, result)
            return pid, int(result["missing_count"]), rec["wall_seconds"]
        finally:
            shutil.rmtree(stim_dir, ignore_errors=True)

    finished = 0
    failed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_do_one, t): t for t in tasks}
        for fut in as_completed(futs):
            paper, pid, _, _, _ = futs[fut]
            finished += 1
            try:
                r_pid, mc, sec = fut.result()
                if args.smoke or finished % 10 == 0 or finished == total:
                    elapsed = time.monotonic() - t0
                    rate = finished / elapsed if elapsed > 0 else 0.0
                    eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0.0
                    print(
                        json.dumps(
                            {
                                "bench": "InfoCoverage",
                                "progress": f"{finished}/{total}",
                                "failed": failed,
                                "rate_req_s": round(rate, 3),
                                "eta_min": round(eta_min, 1),
                                "poster_id": r_pid,
                                "missing_count": mc,
                                "seconds": sec,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                print(f"[AGY-InfoCoverage] failed {pid} ({paper[:30]}): {str(exc)[:300]}", flush=True)
    return failed


def build_vb_prompt(poster_file: Path, dim: str, rubric: Mapping[str, Any], question: Mapping[str, Any]) -> str:
    qid = str(question["id"])
    one_rubric = {**rubric, "questions": [dict(question)]}
    return "\n\n".join(
        [
            "EVALUATION-ONLY TARGETED ATOMIC JUDGEMENT",
            f"1. Call `read_file` ONCE on `{poster_file}` to visually inspect the poster image.",
            "2. Do NOT inspect any other files, do NOT run shell commands, do NOT search, and do NOT call any other tools.",
            "3. That poster image is the ONLY evidence. Judge strictly the single requested atomic question below.",
            build_system_prompt(one_rubric),
            build_user_prompt(one_rubric, [dict(question)]),
            f'Return your final response strictly conforming to the enforced JSON Schema with `dimension` = "{dim}" and `results` containing the single item `{qid}`.',
        ]
    )


def run_visionbench_once(args: argparse.Namespace) -> int:
    vb_out = OUT_DIR / "visionbench"
    res_root = vb_out / "results"
    rubrics = load_rubrics(str(VB_SNAP / "rubrics"), dimensions=list(VISUAL_EVALUATION_DIMENSIONS))
    schema = strip_schema(json.loads((VB_SNAP / "schemas/single_question_response.schema.json").read_text(encoding="utf-8")))

    out_vdir = res_root / CODEX_VERSION
    out_vdir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    tasks: list[tuple[str, Path, str, Mapping[str, Any], Mapping[str, Any], int]] = []

    for poster in sorted(CODEX_POSTER_DIR.glob("*.png")):
        pkey = poster_key(poster)
        rpath = out_vdir / f"{pkey}.json"
        if rpath.is_file():
            report = json.loads(rpath.read_text(encoding="utf-8"))
        else:
            report = {
                "schema_version": "2.0",
                "created_at": utc_now(),
                "protocol": {
                    "name": "p2p_atomic_judgement_agy_argon_per_question_v1",
                    "evaluation_system": "visual_evaluation",
                    "backend": "antigravity_cli_headless",
                    "model": AGY_MODEL,
                    "thinking_level": "LOW",
                },
                "model_override": AGY_MODEL,
                "reasoning_effort_override": "LOW",
                "poster_images": [str(poster.resolve())],
                "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
                "requested_dimensions": list(VISUAL_EVALUATION_DIMENSIONS),
                "requested_question_ids": [],
                "dimensions": {},
                "question_runs": {},
            }
            normalize_report(report, rubrics, list(VISUAL_EVALUATION_DIMENSIONS))
            atomic_write_json(rpath, report)
        reports[pkey] = report
        report_paths[pkey] = rpath
        done = completed_ids(report)
        call_idx = 0
        for dim in VISUAL_EVALUATION_DIMENSIONS:
            rubric = rubrics[dim]
            for q in rubric["questions"]:
                call_idx += 1
                if q["id"] in done:
                    continue
                tasks.append((pkey, poster, dim, rubric, q, call_idx))

    if args.smoke:
        tasks = tasks[:1]

    total = len(tasks)
    print(f"[AGY-VisionBench] {CODEX_VERSION} | remaining={total} | workers={args.max_workers} | model={AGY_MODEL}", flush=True)
    if total == 0:
        return 0

    STIM_ROOT.mkdir(parents=True, exist_ok=True)
    seq = 0
    seq_lock = threading.Lock()

    def _do_one(item: tuple[str, Path, str, Mapping[str, Any], Mapping[str, Any], int]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        nonlocal seq
        pkey, poster, dim, rubric, question, call_idx = item
        qid = str(question["id"])
        with seq_lock:
            seq += 1
            my_seq = seq
        stim_dir = STIM_ROOT / f"vb_{os.getpid()}_{my_seq}_{qid}"
        prep_dir = stim_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)
        try:
            prepared_poster, input_transform = prepare_question_poster(poster, question, prep_dir)
            eval_sha = hashlib.sha256(prepared_poster.read_bytes()).hexdigest()
            prompt_text = build_vb_prompt(stim_dir / "poster.png", dim, rubric, question)
            rec = invoke_agy_once(stim_dir, prepared_poster, prompt_text, schema, args.min_interval)
            normalized = validate_dimension_response(rec["parsed"], rubric, [question])
            result = normalized["results"][0]
            result["independent_question_session"] = True
            run_record = {
                "question_id": qid,
                "dimension": dim,
                "call_index": call_idx,
                "status": "completed",
                "backend": "antigravity_cli_headless",
                "model_override": AGY_MODEL,
                "thinking_level": "LOW",
                "evaluation_image_sha256": eval_sha,
                "input_transform": input_transform,
                "conversation_id": rec["conversation_id"],
                "tokens": rec["usage"],
                "seconds": rec["wall_seconds"],
                "completed_at": rec["completed_at"],
                "dimension_summary": normalized.get("dimension_summary", ""),
            }
            return dim, qid, result, run_record
        finally:
            shutil.rmtree(stim_dir, ignore_errors=True)

    lock = threading.Lock()
    finished = 0
    failed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_do_one, t): t for t in tasks}
        for fut in as_completed(futs):
            pkey, poster, dim, rubric, question, call_idx = futs[fut]
            qid = str(question["id"])
            with lock:
                report = reports[pkey]
                try:
                    r_dim, r_qid, result, run_rec = fut.result()
                    dim_entry = report.setdefault("dimensions", {}).setdefault(r_dim, {"results": []})
                    existing_results = [r for r in dim_entry.get("results", []) if r.get("id") != r_qid]
                    existing_results.append(result)
                    dim_entry["results"] = existing_results
                    report.setdefault("question_runs", {})[r_qid] = run_rec
                    normalize_report(report, rubrics, list(VISUAL_EVALUATION_DIMENSIONS))
                    atomic_write_json(report_paths[pkey], report)
                    verdict_val = result.get("verdict")
                    sec_val = run_rec["seconds"]
                except Exception as exc:
                    failed += 1
                    verdict_val = None
                    sec_val = 0.0
                    print(f"[AGY-VisionBench] failed {CODEX_VERSION}/{pkey}/{qid}: {str(exc)[:300]}", flush=True)
                finished += 1
                if args.smoke or finished % 25 == 0 or finished == total:
                    elapsed = time.monotonic() - t0
                    rate = finished / elapsed if elapsed > 0 else 0.0
                    eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0.0
                    print(
                        json.dumps(
                            {
                                "bench": "VisionBench",
                                "progress": f"{finished}/{total}",
                                "failed": failed,
                                "rate_req_s": round(rate, 3),
                                "eta_min": round(eta_min, 1),
                                "paper": pkey[:32],
                                "qid": qid,
                                "verdict": verdict_val,
                                "seconds": sec_val,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("all", "infocoverage", "visionbench"), default="all")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--min-interval", type=float, default=0.4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.bench in ("all", "infocoverage"):
        if args.smoke:
            if run_infocoverage_once(args) != 0:
                return 2
        else:
            for sweep in range(1, 6):
                rem = run_infocoverage_once(args)
                if rem == 0:
                    break
                print(f"[AGY-InfoCoverage] Sweep {sweep} had {rem} drops; running next sweep...", flush=True)

    if args.bench in ("all", "visionbench"):
        if args.smoke:
            if run_visionbench_once(args) != 0:
                return 2
        else:
            for sweep in range(1, 6):
                rem = run_visionbench_once(args)
                if rem == 0:
                    break
                print(f"[AGY-VisionBench] Sweep {sweep} had {rem} drops; running next sweep...", flush=True)

    print("CODEX_SOL_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
