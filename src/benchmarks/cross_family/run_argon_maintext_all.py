#!/usr/bin/env python3
"""Sequential 8-worker Gemini-4-Argon judge runner for Main-Text Tables:
  1. Table 4: InfoCoverage (9 groups x 100 posters = 900 calls)
  2. Table 1 (right): PaperQuiz (9 groups x 100 posters x 2 splits = 1800 calls)
  3. Table 3: VisionBench (9 groups x 100 posters x 12 questions = 10800 calls)
  (Table 1 left SixDim is already completed in p2p-sixdim-vertex-argon-100-20261107.)

Follows p2p-sixdim-vertex-argon-100-20261107/_code/run_argon_full100.py + call.mjs:
- MODEL = "gemini-4-argon", THINKING = "LOW", MAX_WORKERS = 8
- Zero retries: if call.mjs exits non-zero (e.g. HTTP 429/403), exits immediately via os._exit(1).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from jinja2 import Environment, StrictUndefined
from PIL import Image

PI_RUNTIME_ROOT = Path(os.environ.get("VASP_PI_ROOT", str(Path(__file__).resolve().parents[3] / "src" / "viscomposer" / "pi")))

BASE = Path(os.environ.get("VASP_EXP_DIR", str(Path(__file__).resolve().parents[3] / "results" / "exp")))
OUT_DIR = Path(os.environ.get("VASP_OUT_DIR", "./outputs/cross_family"))
WORK_ROOT = Path("/dev/shm/p2p_argon_maintext_work")

MODEL, THINKING = "gemini-4-argon", "LOW"
MAX_WORKERS = 10
MIN_LAUNCH_INTERVAL = 0.25
DEFAULT_VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
COOLDOWN_429_SECONDS = 90.0
PRICE = {"input": 2.0, "cached": 0.1, "output": 10.0}
STRIP_KEYS = {"$schema", "title", "default", "minLength", "pattern"}

VERSION_DIRS: dict[str, Path] = {
    "pi_full": BASE / "p2p-sixdim-upstream-api-pi-full-sol-20260926/inputs/pi_full",
    "opus5-high": BASE / "p2p-sixdim-upstream-api-opus-gt-20260924/inputs/opus5-high",
    "original_posters": BASE / "p2p-sixdim-upstream-api-opus-gt-20260924/inputs/original_posters",
    "research-studio-real": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/research-studio-real",
    "postergen": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/postergen",
    "paper2poster": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/paper2poster",
    "posterforest": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/posterforest",
    "postermeld": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/postermeld",
    "p2p": BASE / "p2p-sixdim-upstream-api-baselines-20260923/inputs/p2p",
}


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


def load_config(
    project_override: str | None = None,
    credentials_override: str | None = None,
    location_override: str | None = None,
) -> dict[str, Any]:
    project = (project_override or os.environ.get("GOOGLE_CLOUD_PROJECT", "")).strip()
    if not project:
        raise RuntimeError("Set GOOGLE_CLOUD_PROJECT or pass --project for Vertex AI API calls.")
    credentials = (credentials_override or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")).strip()
    location = (location_override or DEFAULT_VERTEX_LOCATION).strip() or "global"
    return {
        "vertex": {"project": project, "location": location, "credentials": credentials},
        "proxy": os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "",
    }


_CONSEC_FAILS = 0
_CONSEC_LOCK = threading.Lock()
_LAUNCH_LOCK = threading.Lock()
_LAST_LAUNCH_TS = 0.0


class WorkerRetire429(RuntimeError):
    pass


def _pace_launch() -> None:
    global _LAST_LAUNCH_TS
    with _LAUNCH_LOCK:
        now = time.monotonic()
        wait_s = (_LAST_LAUNCH_TS + MIN_LAUNCH_INTERVAL) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _LAST_LAUNCH_TS = time.monotonic()


def run_retiring_pool(tasks: list[Any], fn: Any):
    global MAX_WORKERS
    task_q: queue.Queue[Any] = queue.Queue()
    for t in tasks:
        task_q.put(t)
    out_q: queue.Queue[Any] = queue.Queue()
    state_lock = threading.Lock()
    n_start = max(1, min(MAX_WORKERS, len(tasks)))
    active = [n_start]
    retired_429 = [0]
    DONE_SENTINEL = object()

    def _worker(wid: int) -> None:
        global MAX_WORKERS
        try:
            while True:
                try:
                    item = task_q.get_nowait()
                except queue.Empty:
                    return
                try:
                    res = fn(item)
                    out_q.put((item, res, None))
                except WorkerRetire429 as exc:
                    out_q.put((item, None, exc))
                    with state_lock:
                        retired_429[0] += 1
                        MAX_WORKERS = max(0, MAX_WORKERS - 1)
                        all_429 = retired_429[0] >= n_start
                        rem = max(0, n_start - retired_429[0])
                    print(
                        f"[RATE-LIMIT] Worker #{wid} hit HTTP 429 -> retiring ONLY this worker (no retry; worker cap now {rem}/{n_start})",
                        file=sys.stderr,
                        flush=True,
                    )
                    if all_429:
                        print("[FATAL] All workers hit HTTP 429 and retired; stopping process cleanly.", file=sys.stderr, flush=True)
                        os._exit(1)
                    return
                except BaseException as exc:
                    out_q.put((item, None, exc))
        finally:
            with state_lock:
                active[0] -= 1
                if active[0] <= 0:
                    out_q.put(DONE_SENTINEL)

    threads = [threading.Thread(target=_worker, args=(i + 1,), daemon=True) for i in range(n_start)]
    for th in threads:
        th.start()

    while True:
        msg = out_q.get()
        if msg is DONE_SENTINEL:
            break
        yield msg


def call_vertex(out: Path, cfg: dict[str, Any], max_out: str | None = None) -> None:
    vx = cfg["vertex"]
    env = {
        **os.environ,
        "GOOGLE_CLOUD_PROJECT": vx["project"],
        "GOOGLE_CLOUD_LOCATION": vx["location"],
        "MODEL": MODEL,
        "THINKING": THINKING,
        "PI_NODE_MODULES": str(PI_RUNTIME_ROOT / "node_modules"),
    }
    if vx.get("credentials"):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = vx["credentials"]
    if max_out is not None:
        env["MAX_OUT"] = max_out
    if cfg.get("proxy"):
        env.update(
            HTTPS_PROXY=cfg["proxy"],
            HTTP_PROXY=cfg["proxy"],
            NO_PROXY="localhost,127.0.0.1,::1",
            NODE_USE_ENV_PROXY="1",
        )
    _pace_launch()
    log_path = out / "call.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            ["node", str(Path(__file__).with_name("call.mjs")), str(out / "request.json"), str(out / "response.json")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    global _CONSEC_FAILS
    if proc.returncode != 0:
        err_detail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else f"rc={proc.returncode}"
        if any(code in err_detail for code in ("HTTP 403", "HTTP 401")):
            print(f"[FATAL] call.mjs hit auth/policy error: {err_detail[:500]}", file=sys.stderr, flush=True)
            os._exit(1)
        if "HTTP 429" in err_detail or "RESOURCE_EXHAUSTED" in err_detail:
            raise WorkerRetire429(f"HTTP 429: {err_detail[:240]}")
        with _CONSEC_LOCK:
            _CONSEC_FAILS += 1
            cf = _CONSEC_FAILS
        if cf >= 12:
            print(f"[FATAL] {cf} consecutive call.mjs failures; circuit breaker stopping: {err_detail[:500]}", file=sys.stderr, flush=True)
            os._exit(1)
        raise subprocess.CalledProcessError(proc.returncode, "call.mjs")
    with _CONSEC_LOCK:
        _CONSEC_FAILS = 0


def cost(hops: list[dict[str, Any]]) -> tuple[dict[str, int], float]:
    total = {"input": 0, "cached": 0, "output": 0}
    for h in hops:
        u = h.get("usage") or {}
        cached = u.get("cachedContentTokenCount", 0)
        total["input"] += u.get("promptTokenCount", 0) - cached
        total["cached"] += cached
        total["output"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return total, round(sum(total[k] * PRICE[k] for k in total) / 1e6, 4)


IMG_CACHE: dict[str, tuple[str, str, str]] = {}
IMG_LOCK = threading.Lock()


def encode_image(poster_path: Path, use_cache: bool = True) -> tuple[str, str, str]:
    key = str(poster_path)
    if use_cache:
        with IMG_LOCK:
            if key in IMG_CACHE:
                return IMG_CACHE[key]
    raw_bytes = poster_path.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        fmt = img.format or "PNG"
        buf = io.BytesIO()
        img.save(fp=buf, format=fmt)
    val = (f"image/{fmt.lower()}", base64.b64encode(buf.getvalue()).decode("utf-8"), hashlib.sha256(raw_bytes).hexdigest())
    if use_cache:
        with IMG_LOCK:
            if len(IMG_CACHE) > 64:
                IMG_CACHE.clear()
            IMG_CACHE[key] = val
    return val


def ensure_under_limit_pil(img: Image.Image, max_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    for quality in (90, 80, 70, 60, 50):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        new_raw = buf.getvalue()
        if len(new_raw) <= max_bytes:
            return Image.open(io.BytesIO(new_raw))
    w, h = img.size
    img_resized = img.resize((w // 2, h // 2), Image.LANCZOS)
    buf = io.BytesIO()
    img_resized.save(buf, format="JPEG", quality=50)
    new_raw = buf.getvalue()
    if len(new_raw) > max_bytes:
        raise RuntimeError("Could not reduce image under size limit")
    return Image.open(io.BytesIO(new_raw))


def encode_paperquiz_image(poster_path: Path) -> tuple[str, str, str]:
    raw_bytes = poster_path.read_bytes()
    img = ensure_under_limit_pil(Image.open(io.BytesIO(raw_bytes)))
    fmt = (img.format or "JPEG").upper()
    buf = io.BytesIO()
    img.save(fp=buf, format=fmt)
    return (f"image/{fmt.lower()}", base64.b64encode(buf.getvalue()).decode("utf-8"), hashlib.sha256(raw_bytes).hexdigest())


def invoke_argon_once(
    cfg: dict[str, Any],
    sys_prompt: str,
    usr_prompt: str,
    mime_type: str,
    image_b64: str,
    schema: dict[str, Any],
    max_out: str | None = None,
) -> dict[str, Any]:
    work_dir = WORK_ROOT / f"{os.getpid()}_{threading.get_ident()}"
    work_dir.mkdir(parents=True, exist_ok=True)
    req_path = work_dir / "request.json"
    res_path = work_dir / "response.json"
    log_path = work_dir / "call.log"
    atomic_write_json(
        req_path,
        {
            "system": sys_prompt,
            "user": usr_prompt,
            "mime_type": mime_type,
            "image_b64": image_b64,
            "schema": schema,
        },
    )
    start = time.monotonic()
    try:
        call_vertex(work_dir, cfg, max_out=max_out)
        reply = json.loads(res_path.read_text(encoding="utf-8"))
        if reply["finish"] != "STOP":
            raise ValueError(f"The answer ended with {reply['finish']}, not STOP")
        ans = json.loads(reply["text"])
        tokens, usd = cost(reply["hops"])
        return {
            "status": "completed",
            "parsed": ans,
            "raw_text": reply["text"],
            "hops": len(reply["hops"]),
            "finish": reply["finish"],
            "tokens": tokens,
            "cost_usd": usd,
            "model_seconds": reply["seconds"],
            "seconds": round(time.monotonic() - start, 1),
            "completed_at": utc_now(),
        }
    except WorkerRetire429:
        raise
    except Exception as exc:
        err_detail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else str(exc)
        raise RuntimeError(f"{exc}: {err_detail[:300]}") from exc
    finally:
        req_path.unlink(missing_ok=True)
        res_path.unlink(missing_ok=True)


# ==============================================================================
# 1. Table 4: InfoCoverage (9 groups x 100 posters = 900 calls)
# ==============================================================================
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


def run_infocoverage(cfg: dict[str, Any], smoke: bool = False) -> None:
    ic_out = OUT_DIR / "infocoverage"
    res_root = ic_out / "results"
    ic_manifest = json.loads((IC_SNAP / "api_nine_methods/manifest.json").read_text(encoding="utf-8"))
    fn_to_id = {row["input_filename"]: str(row["poster_id"]) for row in ic_manifest["posters"]}
    ref_posters = {str(p["poster_id"]): p for p in _get_ic_runner().load_reference()}
    template = (IC_SNAP / "full100/evaluation_prompt_api.md").read_text(encoding="utf-8")
    schema = strip_schema(_get_ic_runner().read_json(_get_ic_runner().SCHEMA_PATH))

    tasks: list[tuple[str, str, str, dict[str, Any], Path, Path]] = []
    for version, vdir in VERSION_DIRS.items():
        for poster in sorted(vdir.glob("*.png")):
            paper = poster.stem
            pid = fn_to_id[f"{paper}.png"]
            pref = ref_posters[pid]
            rpath = res_root / version / f"{pid}.json"
            if rpath.is_file():
                try:
                    existing = json.loads(rpath.read_text(encoding="utf-8"))
                    _get_ic_runner().common.validate_evaluation(pref, existing)
                    if (existing.get("metadata", {}).get("tokens") or {}).get("input", 0) < 1000:
                        _get_ic_runner().validate_image_grounding(existing)
                    continue
                except Exception:
                    pass
            tasks.append((version, paper, pid, pref, poster, rpath))

    if smoke:
        tasks = tasks[:1]

    total = len(tasks)
    print(f"[InfoCoverage] Total remaining tasks: {total} | workers={MAX_WORKERS} | model={MODEL} ({THINKING})", flush=True)
    if total == 0:
        summarize_infocoverage()
        return

    def _do_one(item: tuple[str, str, str, dict[str, Any], Path, Path]) -> tuple[str, str, str, int, float]:
        version, paper, pid, pref, poster, rpath = item
        prompt = _get_ic_runner().build_prompt(template, pref)
        sys_p, sep, tail = prompt.partition("\n\nINPUT:\n")
        if not sep:
            raise ValueError("rendered prompt lacks INPUT separator")
        usr_p = "INPUT:\n" + tail
        mime_type, image_b64, img_sha = encode_image(poster)
        rec = invoke_argon_once(cfg, sys_p, usr_p, mime_type, image_b64, schema)
        result = rec["parsed"]
        _get_ic_runner().common.validate_evaluation(pref, result)
        if (rec.get("tokens") or {}).get("input", 0) < 1000:
            _get_ic_runner().validate_image_grounding(result)
        result["metadata"] = {
            "target_method": version,
            "target_paper": paper,
            "target_image": str(poster.resolve()),
            "image_sha256": img_sha,
            "model": MODEL,
            "thinking_level": THINKING,
            "elapsed_seconds": rec["seconds"],
            "model_seconds": rec["model_seconds"],
            "tokens": rec["tokens"],
            "cost_usd": rec["cost_usd"],
            "hops": rec["hops"],
            "completed_at": rec["completed_at"],
        }
        atomic_write_json(rpath, result)
        return version, paper, pid, int(result["missing_count"]), rec["cost_usd"]

    finished = 0
    failed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_do_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            finished += 1
            try:
                v, p, pid, mc, usd = fut.result()
            except Exception as exc:
                failed += 1
                print(f"[InfoCoverage] failed {t[0]}/{t[2]} ({t[1][:30]}): {str(exc)[:200]}", flush=True)
                continue
            if finished % 25 == 0 or finished == total or smoke:
                elapsed = time.monotonic() - t0
                rate = finished / elapsed if elapsed > 0 else 0
                eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0
                print(
                    json.dumps(
                        {
                            "bench": "InfoCoverage",
                            "progress": f"{finished}/{total}",
                            "failed": failed,
                            "rate_req_s": round(rate, 2),
                            "eta_min": round(eta_min, 1),
                            "version": v,
                            "poster_id": pid,
                            "missing_count": mc,
                            "cost_usd": usd,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    if not smoke:
        summarize_infocoverage()


def summarize_infocoverage() -> None:
    ic_out = OUT_DIR / "infocoverage"
    res_root = ic_out / "results"
    ref_posters = {str(p["poster_id"]): p for p in _get_ic_runner().load_reference()}
    unit_type_map: dict[tuple[str, str], str] = {}
    for pid, p in ref_posters.items():
        for u in p.get("information_units", p.get("units", [])):
            unit_type_map[(pid, str(u["unit_id"]))] = str(u["information_type"])

    prob_types = {"problem", "background", "motivation", "problem_formulation", "goal", "challenge", "objective", "research_question", "scope"}
    summary: dict[str, Any] = {}
    for version in VERSION_DIRS:
        vdir = res_root / version
        files = sorted(vdir.glob("*.json"))
        if len(files) < 100:
            continue
        missing_counts = []
        type_tot: dict[str, int] = defaultdict(int)
        type_cov: dict[str, int] = defaultdict(int)
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            pid = str(d["poster_id"])
            missing_counts.append(int(d["missing_count"]))
            for a in d["assessments"]:
                ut = unit_type_map[(pid, str(a["unit_id"]))]
                type_tot[ut] += 1
                if a["verdict"] == "COVERED":
                    type_cov[ut] += 1
        tot_u = len(files) * 7
        cov_u = tot_u - sum(missing_counts)
        prob_tot = sum(type_tot[t] for t in prob_types)
        prob_cov = sum(type_cov[t] for t in prob_types)
        summary[version] = {
            "Overall_pct": round(cov_u / tot_u * 100.0, 2),
            "Covered_units": f"{cov_u}/{tot_u}",
            "Full_7of7": sum(1 for mc in missing_counts if mc == 0),
            "Problem_pct": round(prob_cov / prob_tot * 100.0, 1) if prob_tot else None,
            "by_information_type": {t: {"covered": type_cov[t], "total": type_tot[t]} for t in sorted(type_tot)},
        }
    atomic_write_json(ic_out / "summary_100.json", summary)


# ==============================================================================
# 2. Table 1 (right): PaperQuiz (9 groups x 100 posters x 2 splits = 1800 calls)
# ==============================================================================
PQ_BASE = BASE / "paperquiz-upstream-api-pi-rs-20260926"
PQ_TMPL_PATH = PQ_BASE / "_code/upstream_templates/answer_question_from_image.yaml"
PQ_QUIZ_DIR = PQ_BASE / "inputs/quiz"
QUESTION_TYPES = ("detail", "understanding")


def pq_load_prompts(questions: Mapping[str, Any]) -> tuple[str, str]:
    cfg = yaml.safe_load(PQ_TMPL_PATH.read_text(encoding="utf-8"))
    tmpl = Environment(undefined=StrictUndefined).from_string(cfg["template"])
    return cfg["system_prompt"], tmpl.render(**{"questions": questions})


def pq_schema(questions: Mapping[str, Any]) -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string", "enum": ["A", "B", "C", "D", "NA"]},
            "reference": {"type": "string"},
        },
        "required": ["answer", "reference"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {str(k): item for k in questions.keys()},
        "required": [str(k) for k in questions.keys()],
    }


def pq_compute_accuracy(predicted: Mapping[str, Any], ground_truth: Mapping[str, Any], aspects: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    correct_global = 0
    total_global = len(ground_truth)
    total_by_aspect: dict[str, int] = defaultdict(int)
    correct_by_aspect: dict[str, int] = defaultdict(int)
    for q, pred_info in predicted.items():
        letter_pred = str(pred_info.get("answer", "NA"))
        ref = str(pred_info.get("reference", "NA"))
        aspect = aspects.get(q, "Unknown")
        total_by_aspect[aspect] += 1
        if letter_pred == "NA" or ref == "NA":
            continue
        if q in ground_truth:
            letter_gt = str(ground_truth[q]).split(".")[0].strip()
            if len(letter_pred) > 0:
                letter_pred = letter_pred[0].upper()
            if letter_pred == letter_gt:
                correct_global += 1
                correct_by_aspect[aspect] += 1
    overall_accuracy = correct_global / total_global if total_global else 0.0
    aspect_summary = {}
    for aspect, total in total_by_aspect.items():
        correct = correct_by_aspect[aspect]
        acc = correct / total if total else 0.0
        aspect_summary[aspect] = {"total": total, "correct": correct, "accuracy": acc}
    return overall_accuracy, aspect_summary


def run_paperquiz(cfg: dict[str, Any], smoke: bool = False) -> None:
    pq_out = OUT_DIR / "paperquiz"
    res_root = pq_out / "results"
    project_id = cfg["vertex"]["project"]

    reports: dict[tuple[str, str], dict[str, Any]] = {}
    report_paths: dict[tuple[str, str], Path] = {}
    tasks: list[tuple[str, str, Path, Path, str]] = []

    for version, vdir in VERSION_DIRS.items():
        for poster in sorted(vdir.glob("*.png")):
            paper = poster.stem
            quiz_path = PQ_QUIZ_DIR / paper / "o3_qa.json"
            key = (version, paper)
            rpath = res_root / version / f"{paper}.json"
            if rpath.is_file():
                report = json.loads(rpath.read_text(encoding="utf-8"))
                report["errors"] = {}
            else:
                report = {
                    "schema_version": 1,
                    "created_at": utc_now(),
                    "model_override": MODEL,
                    "thinking_level": THINKING,
                    "project_id": project_id,
                    "group": version,
                    "paper": paper,
                    "poster_path": str(poster.resolve()),
                    "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
                    "quiz_path": str(quiz_path.resolve()),
                    "results": {},
                    "api_runs": {},
                    "errors": {},
                    "status": "running",
                }
                atomic_write_json(rpath, report)
            reports[key] = report
            report_paths[key] = rpath
            for qtype in QUESTION_TYPES:
                if qtype not in report.get("results", {}):
                    tasks.append((version, paper, poster, quiz_path, qtype))

    if smoke:
        tasks = tasks[:1]

    total = len(tasks)
    print(f"[PaperQuiz] Total remaining tasks: {total} | workers={MAX_WORKERS} | model={MODEL} ({THINKING})", flush=True)
    if total == 0:
        summarize_paperquiz()
        return

    def _do_one(item: tuple[str, str, Path, Path, str]) -> dict[str, Any]:
        version, paper, poster, quiz_path, qtype = item
        quiz = json.loads(quiz_path.read_text(encoding="utf-8"))
        qa = quiz[qtype]
        sys_p, usr_p = pq_load_prompts(qa["questions"])
        schema = pq_schema(qa["questions"])
        mime_type, image_b64, _ = encode_paperquiz_image(poster)
        rec = invoke_argon_once(cfg, sys_p, usr_p, mime_type, image_b64, schema)
        parsed = rec["parsed"]
        acc, aspect_acc = pq_compute_accuracy(parsed, qa["answers"], qa["aspects"])
        return {
            "qtype": qtype,
            "accuracy": acc,
            "aspect_accuracy": aspect_acc,
            "answers": parsed,
            "api_run": {
                "status": "completed",
                "hops": rec["hops"],
                "finish": rec["finish"],
                "tokens": rec["tokens"],
                "cost_usd": rec["cost_usd"],
                "model_seconds": rec["model_seconds"],
                "seconds": rec["seconds"],
                "completed_at": rec["completed_at"],
            },
        }

    lock = threading.Lock()
    finished = 0
    failed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_do_one, t): t for t in tasks}
        for fut in as_completed(futs):
            v, p, poster, quiz_path, qtype = futs[fut]
            with lock:
                report = reports[(v, p)]
                try:
                    out = fut.result()
                    report["results"][qtype] = {
                        "accuracy": out["accuracy"],
                        "aspect_accuracy": out["aspect_accuracy"],
                        "answers": out["answers"],
                    }
                    report["api_runs"][qtype] = out["api_run"]
                    report["errors"].pop(qtype, None)
                    acc_val = round(out["accuracy"] * 100.0, 2)
                    usd_val = out["api_run"]["cost_usd"]
                except Exception as exc:
                    failed += 1
                    report["errors"][qtype] = {"type": type(exc).__name__, "message": str(exc)}
                    acc_val = None
                    usd_val = 0.0
                    print(f"[PaperQuiz] failed {v}/{p}/{qtype}: {str(exc)[:200]}", flush=True)
                report["qa_result"] = {
                    f"{qt}_accuracy": report["results"].get(qt, {}).get("accuracy") for qt in QUESTION_TYPES
                }
                report["status"] = "completed" if (len(report["results"]) == 2 and not report["errors"]) else "partial"
                atomic_write_json(report_paths[(v, p)], report)
                finished += 1
                if finished % 25 == 0 or finished == total or smoke:
                    elapsed = time.monotonic() - t0
                    rate = finished / elapsed if elapsed > 0 else 0
                    eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0
                    print(
                        json.dumps(
                            {
                                "bench": "PaperQuiz",
                                "progress": f"{finished}/{total}",
                                "failed": failed,
                                "rate_req_s": round(rate, 2),
                                "eta_min": round(eta_min, 1),
                                "version": v,
                                "paper": p[:30],
                                "qtype": qtype,
                                "acc_pct": acc_val,
                                "cost_usd": usd_val,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    if not smoke:
        summarize_paperquiz()


def summarize_paperquiz() -> None:
    pq_out = OUT_DIR / "paperquiz"
    res_root = pq_out / "results"
    summary: dict[str, Any] = {}
    for version in VERSION_DIRS:
        vdir = res_root / version
        files = sorted(vdir.glob("*.json"))
        if len(files) < 100:
            continue
        det_l, und_l, ov_l = [], [], []
        for f in files:
            r = json.loads(f.read_text(encoding="utf-8")).get("results", {})
            if "detail" not in r or "understanding" not in r:
                continue
            d = float(r["detail"]["accuracy"]) * 100.0
            u = float(r["understanding"]["accuracy"]) * 100.0
            det_l.append(d)
            und_l.append(u)
            ov_l.append((d + u) / 2.0)
        if len(ov_l) == 100:
            summary[version] = {
                "Verbatim_Detail": round(statistics.mean(det_l), 2),
                "Interpretive_Understanding": round(statistics.mean(und_l), 2),
                "Overall": round(statistics.mean(ov_l), 2),
            }
    atomic_write_json(pq_out / "summary_100.json", summary)


# ==============================================================================
# 3. Table 3: VisionBench (9 groups x 100 posters x 12 questions = 10800 calls)
# ==============================================================================
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


def run_visionbench(cfg: dict[str, Any], smoke: bool = False) -> None:
    vb_out = OUT_DIR / "visionbench"
    res_root = vb_out / "results"
    rubrics = load_rubrics(str(VB_SNAP / "rubrics"), dimensions=list(VISUAL_EVALUATION_DIMENSIONS))
    schema = strip_schema(json.loads((VB_SNAP / "schemas/single_question_response.schema.json").read_text(encoding="utf-8")))

    reports: dict[tuple[str, str], dict[str, Any]] = {}
    report_paths: dict[tuple[str, str], Path] = {}
    tasks: list[tuple[str, str, Path, str, Mapping[str, Any], Mapping[str, Any], int]] = []

    for version, vdir in VERSION_DIRS.items():
        out_vdir = res_root / version
        out_vdir.mkdir(parents=True, exist_ok=True)
        for poster in sorted(vdir.glob("*.png")):
            pkey = poster_key(poster)
            rpath = out_vdir / f"{pkey}.json"
            if rpath.is_file():
                report = json.loads(rpath.read_text(encoding="utf-8"))
            else:
                report = {
                    "schema_version": "2.0",
                    "created_at": utc_now(),
                    "protocol": {
                        "name": "p2p_atomic_judgement_vertex_argon_per_question_v1",
                        "evaluation_system": "visual_evaluation",
                        "backend": "vertex_ai_stream_generate_content",
                        "model": MODEL,
                        "thinking_level": THINKING,
                    },
                    "model_override": MODEL,
                    "reasoning_effort_override": THINKING,
                    "poster_images": [str(poster.resolve())],
                    "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
                    "requested_dimensions": list(VISUAL_EVALUATION_DIMENSIONS),
                    "requested_question_ids": [],
                    "dimensions": {},
                    "question_runs": {},
                }
                normalize_report(report, rubrics, list(VISUAL_EVALUATION_DIMENSIONS))
                atomic_write_json(rpath, report)
            reports[(version, pkey)] = report
            report_paths[(version, pkey)] = rpath
            done = completed_ids(report)
            call_idx = 0
            for dim in VISUAL_EVALUATION_DIMENSIONS:
                rubric = rubrics[dim]
                for q in rubric["questions"]:
                    call_idx += 1
                    if q["id"] in done:
                        continue
                    tasks.append((version, pkey, poster, dim, rubric, q, call_idx))

    if smoke:
        tasks = tasks[:1]

    total = len(tasks)
    print(f"[VisionBench] Total remaining tasks: {total} | workers={MAX_WORKERS} | model={MODEL} ({THINKING})", flush=True)
    if total == 0:
        summarize_visionbench()
        return 0

    def _do_one(item: tuple[str, str, Path, str, Mapping[str, Any], Mapping[str, Any], int]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        version, pkey, poster, dim, rubric, question, call_idx = item
        qid = str(question["id"])
        stim_dir = WORK_ROOT / f"vb_stim_{os.getpid()}_{threading.get_ident()}"
        stim_dir.mkdir(parents=True, exist_ok=True)
        try:
            prepared_poster, input_transform = prepare_question_poster(poster, question, stim_dir)
            one_rubric = {**rubric, "questions": [dict(question)]}
            sys_p = build_system_prompt(one_rubric)
            usr_p = build_user_prompt(one_rubric, [dict(question)])
            use_cache = (prepared_poster.resolve() == poster.resolve())
            mime_type, image_b64, eval_sha = encode_image(prepared_poster, use_cache=use_cache)
            rec = invoke_argon_once(cfg, sys_p, usr_p, mime_type, image_b64, schema)
            normalized = validate_dimension_response(rec["parsed"], rubric, [question])
            result = normalized["results"][0]
            result["independent_question_session"] = True
            run_record = {
                "question_id": qid,
                "dimension": dim,
                "call_index": call_idx,
                "status": "completed",
                "model_override": MODEL,
                "thinking_level": THINKING,
                "evaluation_image_sha256": eval_sha,
                "input_transform": input_transform,
                "hops": rec["hops"],
                "finish": rec["finish"],
                "tokens": rec["tokens"],
                "cost_usd": rec["cost_usd"],
                "model_seconds": rec["model_seconds"],
                "seconds": rec["seconds"],
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
    for item_t, item_res, item_exc in run_retiring_pool(tasks, _do_one):
        if True:
            v, pkey, poster, dim, rubric, question, call_idx = item_t
            qid = str(question["id"])
            with lock:
                report = reports[(v, pkey)]
                try:
                    if item_exc is not None:
                        raise item_exc
                    r_dim, r_qid, result, run_rec = item_res
                    dim_entry = report.setdefault("dimensions", {}).setdefault(r_dim, {"results": []})
                    existing_results = [r for r in dim_entry.get("results", []) if r.get("id") != r_qid]
                    existing_results.append(result)
                    dim_entry["results"] = existing_results
                    report.setdefault("question_runs", {})[r_qid] = run_rec
                    normalize_report(report, rubrics, list(VISUAL_EVALUATION_DIMENSIONS))
                    verdict_val = result.get("verdict")
                    usd_val = run_rec["cost_usd"]
                except Exception as exc:
                    failed += 1
                    verdict_val = None
                    usd_val = 0.0
                    print(f"[VisionBench] failed {v}/{pkey}/{qid}: {str(exc)[:200]}", flush=True)
                atomic_write_json(report_paths[(v, pkey)], report)
                finished += 1
                if finished % 50 == 0 or finished == total or smoke:
                    elapsed = time.monotonic() - t0
                    rate = finished / elapsed if elapsed > 0 else 0
                    eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0
                    print(
                        json.dumps(
                            {
                                "bench": "VisionBench",
                                "progress": f"{finished}/{total}",
                                "failed": failed,
                                "rate_req_s": round(rate, 2),
                                "eta_min": round(eta_min, 1),
                                "version": v,
                                "paper": pkey[:30],
                                "qid": qid,
                                "verdict": verdict_val,
                                "cost_usd": usd_val,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    if not smoke and failed == 0:
        summarize_visionbench()
    return failed


def summarize_visionbench() -> None:
    vb_out = OUT_DIR / "visionbench"
    res_root = vb_out / "results"
    summary: dict[str, Any] = {}
    for version in VERSION_DIRS:
        vdir = res_root / version
        files = sorted(vdir.glob("*.json"))
        if len(files) < 100:
            continue
        fig_l, guid_l, lay_l, eq_l = [], [], [], []
        complete_count = 0
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            dims = d.get("dimensions", {})
            fpr = dims.get("aesthetic_element", {}).get("aggregate", {}).get("pass_rate")
            gpr = dims.get("aesthetic_engagement", {}).get("aggregate", {}).get("pass_rate")
            lpr = dims.get("aesthetic_layout", {}).get("aggregate", {}).get("pass_rate")
            valid = [pr for pr in (fpr, gpr, lpr) if pr is not None]
            if not valid:
                continue
            complete_count += 1
            eq_l.append(statistics.mean(valid) * 100.0)
            if fpr is not None:
                fig_l.append(fpr * 100.0)
            if gpr is not None:
                guid_l.append(gpr * 100.0)
            if lpr is not None:
                lay_l.append(lpr * 100.0)
        if complete_count == 100:
            summary[version] = {
                "Figure_Presentation": round(statistics.mean(fig_l), 2),
                "Visual_Guidance": round(statistics.mean(guid_l), 2),
                "Layout": round(statistics.mean(lay_l), 2),
                "Overall": round(statistics.mean(eq_l), 2),
            }
    atomic_write_json(vb_out / "summary_100.json", summary)


def main() -> int:
    global MAX_WORKERS, MIN_LAUNCH_INTERVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bench", choices=("all", "infocoverage", "paperquiz", "visionbench"), default="all")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--min-interval", type=float, default=MIN_LAUNCH_INTERVAL)
    ap.add_argument("--project", default=None, help="Vertex AI GCP project ID (default: GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--location", default=None, help="Vertex AI location (default: GOOGLE_CLOUD_LOCATION or global)")
    ap.add_argument("--credentials", default=None, help="Service account JSON path (default: GOOGLE_APPLICATION_CREDENTIALS)")
    args = ap.parse_args()

    MAX_WORKERS = args.max_workers
    MIN_LAUNCH_INTERVAL = args.min_interval
    cfg = load_config(
        project_override=args.project,
        credentials_override=args.credentials,
        location_override=args.location,
    )
    if args.bench in ("all", "infocoverage"):
        run_infocoverage(cfg, smoke=args.smoke)
    if args.bench in ("all", "paperquiz"):
        run_paperquiz(cfg, smoke=args.smoke)
    if args.bench in ("all", "visionbench"):
        if args.smoke:
            run_visionbench(cfg, smoke=True)
        else:
            for sweep in range(1, 6):
                rem_failed = run_visionbench(cfg, smoke=False)
                if rem_failed == 0:
                    break
                print(f"[VisionBench] Sweep {sweep} finished with {rem_failed} transient failures; starting next sweep...", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
