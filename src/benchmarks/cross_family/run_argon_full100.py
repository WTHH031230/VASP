#!/usr/bin/env python3
"""20-worker cross-family VLM-as-Judge runner (Gemini-4-Argon).
Zero retries: if call.mjs exits non-zero (e.g. HTTP 429), the entire process exits immediately via os._exit(1).
"""
import base64
import hashlib
import io
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined
from PIL import Image

PI_RUNTIME_ROOT = Path(os.environ.get("VASP_PI_ROOT", str(Path(__file__).resolve().parents[3] / "src" / "viscomposer" / "pi")))

BASE = Path(os.environ.get("VASP_EXP_DIR", str(Path(__file__).resolve().parents[3] / "results" / "exp")))
OUT_DIR = BASE / "p2p-sixdim-vertex-argon-100-20261107"
RESULTS_DIR = OUT_DIR / "results"
TMPL_DIR = BASE / "p2p-sixdim-upstream-api-pi-full-sol-20260926/_code/upstream_templates"
WORK_ROOT = Path("/dev/shm/p2p_argon_work")

MODEL, THINKING = "gemini-4-argon", "LOW"
MAX_WORKERS = 8
MAX_OUT = "1024"
PRICE = {"input": 2.0, "cached": 0.1, "output": 10.0}

ASPECTS = (
    "aesthetic_element",
    "aesthetic_engagement",
    "aesthetic_layout",
    "information_low_level",
    "information_logic",
    "information_content",
)

VERSION_DIRS = {
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def answer_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["reason", "score"],
    }


def load_config():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI API calls.")
    return {
        "vertex": {
            "project": project,
            "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global").strip() or "global",
            "credentials": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        },
        "proxy": os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "",
    }


def call(out: Path, cfg: dict):
    vx = cfg["vertex"]
    env = {
        **os.environ,
        "GOOGLE_CLOUD_PROJECT": vx["project"],
        "GOOGLE_CLOUD_LOCATION": vx["location"],
        "MODEL": MODEL,
        "THINKING": THINKING,
        "MAX_OUT": MAX_OUT,
        "PI_NODE_MODULES": str(PI_RUNTIME_ROOT / "node_modules"),
    }
    if vx.get("credentials"):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = vx["credentials"]
    if cfg.get("proxy"):
        env.update(
            HTTPS_PROXY=cfg["proxy"],
            HTTP_PROXY=cfg["proxy"],
            NO_PROXY="localhost,127.0.0.1,::1",
            NODE_USE_ENV_PROXY="1",
        )
    with open(out / "call.log", "w", encoding="utf-8") as log:
        subprocess.run(
            ["node", str(Path(__file__).with_name("call.mjs")), str(out / "request.json"), str(out / "response.json")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def cost(hops):
    total = {"input": 0, "cached": 0, "output": 0}
    for h in hops:
        u = h.get("usage") or {}
        cached = u.get("cachedContentTokenCount", 0)
        total["input"] += u.get("promptTokenCount", 0) - cached
        total["cached"] += cached
        total["output"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return total, round(sum(total[k] * PRICE[k] for k in total) / 1e6, 4)


IMG_CACHE = {}
IMG_LOCK = threading.Lock()


def encode_image(poster_path: Path):
    key = str(poster_path)
    with IMG_LOCK:
        if key in IMG_CACHE:
            return IMG_CACHE[key]
    raw_bytes = poster_path.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        fmt = img.format or "PNG"
        buf = io.BytesIO()
        img.save(fp=buf, format=fmt)
    val = (f"image/{fmt.lower()}", base64.b64encode(buf.getvalue()).decode("utf-8"), hashlib.sha256(raw_bytes).hexdigest())
    with IMG_LOCK:
        if len(IMG_CACHE) > 64:
            IMG_CACHE.clear()
        IMG_CACHE[key] = val
    return val


def load_prompt(aspect: str):
    cfg = yaml.safe_load((TMPL_DIR / f"{aspect}_judge.yaml").read_text(encoding="utf-8"))
    tmpl = Environment(undefined=StrictUndefined).from_string(cfg["template"])
    return cfg["system_prompt"], tmpl.render()


def summarize(report: dict):
    scores = {a: item["score"] for a, item in report["results"].items()}
    aes = [scores[a] for a in ASPECTS[:3] if a in scores]
    inf = [scores[a] for a in ASPECTS[3:] if a in scores]
    report["averages"] = {
        "aesthetic_average": sum(aes) / len(aes) if aes else None,
        "information_average": sum(inf) / len(inf) if inf else None,
        "overall_average": sum(scores.values()) / len(scores) if scores else None,
        "aspects_scored": len(scores),
    }


def run_one(cfg: dict, prompts: dict, version: str, paper: str, poster: Path, aspect: str):
    mime_type, image_b64, _ = encode_image(poster)
    sys_prompt, usr_prompt = prompts[aspect]
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
            "schema": answer_schema(),
        },
    )
    start = time.monotonic()
    try:
        call(work_dir, cfg)
        reply = json.loads(res_path.read_text(encoding="utf-8"))
        if reply["finish"] != "STOP":
            raise ValueError(f"The answer ended with {reply['finish']}, not STOP")
        ans = json.loads(reply["text"])
        tokens, usd = cost(reply["hops"])
        return {
            "status": "completed",
            "score": int(ans["score"]),
            "reason": ans["reason"],
            "hops": len(reply["hops"]),
            "finish": reply["finish"],
            "tokens": tokens,
            "cost_usd": usd,
            "model_seconds": reply["seconds"],
            "seconds": round(time.monotonic() - start, 1),
            "completed_at": utc_now(),
        }
    except Exception as exc:
        err_detail = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else str(exc)
        raise RuntimeError(f"{exc}: {err_detail[:300]}") from exc
    finally:
        req_path.unlink(missing_ok=True)
        res_path.unlink(missing_ok=True)


def main():
    cfg = load_config()
    project_id = cfg["vertex"]["project"]
    prompts = {a: load_prompt(a) for a in ASPECTS}

    reports = {}
    report_paths = {}
    tasks = []

    for version, vdir in VERSION_DIRS.items():
        for poster in sorted(vdir.glob("*.png")):
            paper = poster.stem
            key = (version, paper)
            rpath = RESULTS_DIR / version / f"{paper}.json"
            if rpath.exists():
                report = json.loads(rpath.read_text(encoding="utf-8"))
                report["errors"] = {}
            else:
                report = {
                    "schema_version": 1,
                    "created_at": utc_now(),
                    "model_override": MODEL,
                    "thinking_level": THINKING,
                    "project_id": project_id,
                    "version": version,
                    "paper": paper,
                    "poster_path": str(poster.resolve()),
                    "poster_sha256": hashlib.sha256(poster.read_bytes()).hexdigest(),
                    "results": {},
                    "api_runs": {},
                    "errors": {},
                    "status": "running",
                }
                atomic_write_json(rpath, report)
            reports[key] = report
            report_paths[key] = rpath
            for aspect in ASPECTS:
                if aspect not in report.get("results", {}):
                    tasks.append((version, paper, poster, aspect))

    total = len(tasks)
    print(f"Total remaining tasks: {total} | workers={MAX_WORKERS} | project={project_id} | model={MODEL} ({THINKING})", flush=True)
    if total == 0:
        return

    lock = threading.Lock()
    finished = 0
    failed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(run_one, cfg, prompts, v, p, poster, a): (v, p, a) for v, p, poster, a in tasks}
        for fut in as_completed(futs):
            v, p, a = futs[fut]
            with lock:
                report = reports[(v, p)]
                try:
                    rec = fut.result()
                    report["results"][a] = {"score": rec["score"], "reason": rec["reason"]}
                    report["api_runs"][a] = rec
                    report["errors"].pop(a, None)
                    score_val = rec["score"]
                    cost_val = rec["cost_usd"]
                except Exception as exc:
                    failed += 1
                    report["errors"][a] = {"type": type(exc).__name__, "message": str(exc)}
                    score_val = None
                    cost_val = 0.0
                    print(f"failed {v}/{p}/{a}: {str(exc)[:200]}", flush=True)
                summarize(report)
                report["status"] = "completed" if (len(report["results"]) == 6 and not report["errors"]) else "partial"
                atomic_write_json(report_paths[(v, p)], report)
                finished += 1
                if finished % 25 == 0 or finished == total:
                    elapsed = time.monotonic() - t0
                    rate = finished / elapsed if elapsed > 0 else 0
                    eta_min = (total - finished) / rate / 60.0 if rate > 0 else 0
                    print(
                        json.dumps(
                            {
                                "progress": f"{finished}/{total}",
                                "failed": failed,
                                "rate_req_s": round(rate, 2),
                                "eta_min": round(eta_min, 1),
                                "version": v,
                                "paper": p[:30],
                                "aspect": a,
                                "score": score_val,
                                "cost_usd": cost_val,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
