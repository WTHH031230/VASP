#!/usr/bin/env python3
"""Judge posters on the six Paper2Poster dimensions exactly as upstream, over an API.

API backend for EXP-014. Mirrors upstream Paper2Poster
``utils/poster_eval_utils.eval_vlm_as_judge_aspect`` (commit 623d042):

* prompts are the upstream ``utils/prompt_templates/<aspect>_judge.yaml`` files,
  copied verbatim into ``upstream_templates/``
* system message = ``system_prompt``; user message = [jinja2-rendered ``template``,
  poster image], the image re-saved in its own format and base64-encoded as in
  camel ``BaseMessage.to_openai_user_message`` (detail ``auto``)
* no temperature/top_p/response_format (camel drops sampling keys for GPT-5
  models; upstream reads free text); the answer is parsed with upstream
  ``get_json_from_response`` (last ```json block or whole text, json_repair)
* one request per (poster, aspect), no shared state

Only ``reasoning_effort`` is added (researcher's choice). Transport, retry,
redaction and key handling follow the EXP-013 runner. The API key is read from
``--api-key-file`` and is never written anywhere.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests

import json_repair
import yaml
from jinja2 import Environment, StrictUndefined
from PIL import Image

TEMPLATE_DIR = Path(__file__).resolve().parent / "upstream_templates"
ALL_ASPECTS = (
    "aesthetic_element",
    "aesthetic_engagement",
    "aesthetic_layout",
    "information_low_level",
    "information_logic",
    "information_content",
)

RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
FATAL_HTTP = {401, 403, 404}
PROTOCOL_NAME = "paper2poster_upstream_vlm_judge_chat_api_v1"
INVOCATION = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"


class FatalAPIError(RuntimeError):
    """An error that makes every further call pointless (bad key, bad model)."""


class InvalidOutput(ValueError):
    """The model answered, but the answer is not a valid judgement."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory with one sub-directory per poster version holding <paper>.png",
    )
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--papers", nargs="+", default=None, help="Restrict to these paper stems")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Platform model id")
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument(
        "--api-key-file", default=str(Path.home() / ".config" / "openai-compat" / "key")
    )
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="auto")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-transport-retries", type=int, default=6)
    parser.add_argument("--invalid-output-retries", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260923, help="Task order shuffle seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def read_api_key(path: str) -> str:
    key = Path(path).expanduser().read_text(encoding="utf-8").strip()
    if not key or any(ch.isspace() for ch in key):
        raise FatalAPIError(f"API key file is empty or malformed: {path}")
    return key


def load_prompts(aspect: str) -> tuple[str, str]:
    """Return (system_prompt, rendered user template) as upstream builds them."""

    with open(TEMPLATE_DIR / f"{aspect}_judge.yaml", "r") as handle:
        config = yaml.safe_load(handle)
    template = Environment(undefined=StrictUndefined).from_string(config["template"])
    return config["system_prompt"], template.render()


def image_data_uri(path: Path) -> str:
    """camel to_openai_user_message: save the PIL image in its own format."""

    with Image.open(path) as image:
        if image.format is None:
            raise ValueError(f"image format unknown: {path}")
        image_type = image.format.lower()
        buffer = io.BytesIO()
        image.save(fp=buffer, format=image.format)
    return f"data:image/{image_type};base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_payload(*, aspect: str, image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    system_prompt, user_text = load_prompts(aspect)
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_uri(image_path), "detail": args.image_detail}},
                ],
            },
        ],
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    return payload


def get_json_from_response(raw_response: str) -> Any:
    """Upstream utils/src/utils.py get_json_from_response, verbatim logic."""

    response = raw_response.strip()
    l, r = response.rfind("```json"), response.rfind("```")
    try:
        if l == -1 or r == -1:
            response = json_repair.loads(response)
        else:
            response = json_repair.loads(response[l + 7 : r].strip())
        return response
    except Exception as e:
        raise RuntimeError("Failed to parse JSON from response", e)


def redacted_payload(payload: Mapping[str, Any], image_path: Path) -> dict[str, Any]:
    record = copy.deepcopy(dict(payload))
    for message in record["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if block.get("type") != "image_url":
                continue
            header, _, encoded = block["image_url"]["url"].partition(",")
            raw = base64.b64decode(encoded)
            block["image_url"]["url"] = {
                "data_uri_header": header,
                "sha256_of_sent_bytes": sha256_bytes(raw),
                "sent_bytes": len(raw),
                "source_path": str(image_path.resolve()),
                "source_sha256": sha256_bytes(image_path.read_bytes()),
            }
    return record


class RequestBudget:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.sent = 0
        self.lock = threading.Lock()
        self.abort_reason: str | None = None

    def take(self) -> None:
        with self.lock:
            if self.abort_reason:
                raise FatalAPIError(f"aborted: {self.abort_reason}")
            if self.limit is not None and self.sent >= self.limit:
                raise FatalAPIError(f"--max-requests {self.limit} reached")
            self.sent += 1

    def abort(self, reason: str) -> None:
        with self.lock:
            self.abort_reason = self.abort_reason or reason


import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.benchmarks.pi_direct import call_pi_from_openai_payload

def post_with_retries(*, session, endpoint, api_key, payload, args, budget, attempts, call_dir, ask_index):
    for transport_try in range(args.max_transport_retries + 1):
        budget.take()
        started = time.monotonic()
        record = {"ask": ask_index, "transport_try": transport_try, "started_at": utc_now()}
        attempts.append(record)
        try:
            decoded = call_pi_from_openai_payload(payload, timeout=float(args.timeout))
            raw_path = call_dir / f"response_{INVOCATION}_ask{ask_index}_try{transport_try}.txt"
            write_text(raw_path, json.dumps(decoded, ensure_ascii=False))
            record.update({"http_status": 200, "raw_response_path": str(raw_path.resolve()), "latency_s": round(time.monotonic() - started, 3), "outcome": "http_ok"})
            return decoded
        except Exception as exc:
            record.update({"outcome": "transport_error", "error": f"{type(exc).__name__}: {exc}", "latency_s": round(time.monotonic() - started, 3)})
            if transport_try == args.max_transport_retries:
                break
            time.sleep(min(60.0, 4.0 * 2 ** transport_try))
    raise RuntimeError(f"pi direct retries exhausted: {attempts[-1]}")

def _unused_post_with_retries(
    *,
    session: requests.Session,
    endpoint: str,
    api_key: str,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
    budget: RequestBudget,
    attempts: list[dict[str, Any]],
    call_dir: Path,
    ask_index: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for transport_try in range(args.max_transport_retries + 1):
        budget.take()
        started = time.monotonic()
        record: dict[str, Any] = {"ask": ask_index, "transport_try": transport_try, "started_at": utc_now()}
        attempts.append(record)
        retry_after: float | None = None
        try:
            response = session.post(endpoint, data=body, headers=headers, timeout=args.timeout)
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            record.update({"outcome": "transport_error", "error": f"{type(exc).__name__}: {exc}"})
        else:
            record["http_status"] = response.status_code
            record["request_id"] = response.headers.get("x-request-id") or response.headers.get("request-id")
            raw_path = call_dir / f"response_{INVOCATION}_ask{ask_index}_try{transport_try}.txt"
            write_text(raw_path, response.text.replace(api_key, "<redacted>"))
            record["raw_response_path"] = str(raw_path.resolve())
            if response.status_code == 200:
                record["latency_s"] = round(time.monotonic() - started, 3)
                try:
                    decoded = response.json()
                except ValueError as exc:
                    record.update({"outcome": "undecodable_body", "error": str(exc)})
                else:
                    record["outcome"] = "http_ok"
                    return decoded
            else:
                safe_text = response.text.replace(api_key, "<redacted>")
                record.update({"outcome": "http_error", "error": safe_text[:2000]})
                if response.status_code in FATAL_HTTP:
                    budget.abort(f"HTTP {response.status_code}")
                    raise FatalAPIError(f"HTTP {response.status_code}: {safe_text[:500]}")
                if response.status_code not in RETRYABLE_HTTP:
                    raise RuntimeError(f"HTTP {response.status_code} (not retried): {safe_text[:500]}")
                header = response.headers.get("retry-after")
                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        retry_after = None
        record["latency_s"] = round(time.monotonic() - started, 3)
        if transport_try == args.max_transport_retries:
            break
        delay = retry_after if retry_after is not None else min(120.0, 5.0 * 2**transport_try)
        time.sleep(delay + random.uniform(0.0, 2.0))
    raise RuntimeError(f"transport retries exhausted after {args.max_transport_retries + 1} tries")


def parse_judgement(decoded: Mapping[str, Any]) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        choice = decoded["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidOutput(f"no choices[0].message in response: {exc}") from exc
    meta = {"finish_reason": choice.get("finish_reason"), "refusal": message.get("refusal")}
    if meta["refusal"]:
        raise InvalidOutput(f"model refusal: {meta['refusal']}")
    if meta["finish_reason"] not in (None, "stop"):
        raise InvalidOutput(f"finish_reason={meta['finish_reason']!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InvalidOutput("empty or non-text content")
    try:
        parsed = get_json_from_response(content)
    except RuntimeError as exc:
        raise InvalidOutput(str(exc)) from exc
    score = parsed.get("score") if isinstance(parsed, dict) else None
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise InvalidOutput(f"score must be an integer in [1, 5], got {score!r}")
    reason = parsed.get("reason")
    normalized = {"score": score, "reason": reason if isinstance(reason, str) else None}
    return normalized, content, {**meta, "parsed": parsed}


def run_one(
    *,
    poster: Path,
    aspect: str,
    run_dir: Path,
    args: argparse.Namespace,
    api_key: str | None,
    budget: RequestBudget,
    local: threading.local,
) -> dict[str, Any]:
    call_dir = run_dir / aspect
    call_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(aspect=aspect, image_path=poster, args=args)
    request_path = call_dir / "request.json"
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    write_json(request_path, {"endpoint": endpoint, "payload": redacted_payload(payload, poster)})
    write_text(call_dir / "system_prompt.txt", payload["messages"][0]["content"])
    write_text(call_dir / "user_prompt.txt", payload["messages"][1]["content"][0]["text"])
    base_record = {
        "aspect": aspect,
        "independent_request": True,
        "stateless": True,
        "model_override": args.model,
        "reasoning_effort_override": args.reasoning_effort,
        "image_detail": args.image_detail,
        "request_path": str(request_path.resolve()),
        "poster_path": str(poster.resolve()),
        "poster_sha256": sha256_bytes(poster.read_bytes()),
    }
    if args.dry_run:
        return {**base_record, "status": "planned"}
    assert api_key is not None
    if not hasattr(local, "session"):
        local.session = requests.Session()
    started_at = utc_now()
    attempts: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for ask_index in range(args.invalid_output_retries + 1):
        decoded = post_with_retries(
            session=local.session,
            endpoint=endpoint,
            api_key=api_key,
            payload=payload,
            args=args,
            budget=budget,
            attempts=attempts,
            call_dir=call_dir,
            ask_index=ask_index,
        )
        try:
            normalized, text, meta = parse_judgement(decoded)
        except InvalidOutput as exc:
            attempts[-1]["outcome"] = "invalid_output"
            attempts[-1]["error"] = str(exc)
            invalid_reasons.append(str(exc))
            continue
        write_text(call_dir / "last_message.txt", text)
        write_json(call_dir / "parsed_answer.json", meta["parsed"])
        write_json(call_dir / "response.json", decoded)
        return {
            **base_record,
            "status": "completed",
            "score": normalized["score"],
            "reason": normalized["reason"],
            "automatic_retry": len(attempts) > 1,
            "invalid_output_reasks": ask_index,
            "invalid_output_reasons": invalid_reasons,
            "http_attempts": attempts,
            "created_at": started_at,
            "completed_at": utc_now(),
            "response_id": decoded.get("id"),
            "model_returned": decoded.get("model"),
            "system_fingerprint": decoded.get("system_fingerprint"),
            "usage": decoded.get("usage"),
            "finish_reason": meta["finish_reason"],
            "response_path": str((call_dir / "response.json").resolve()),
        }
    raise RuntimeError(f"invalid output after {args.invalid_output_retries + 1} asks: {invalid_reasons}")


def new_report(*, version: str, paper: str, poster: Path, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "protocol": {
            "name": PROTOCOL_NAME,
            "backend": "openai_compatible_chat_completions",
            "api_base_url": args.base_url,
            "message_layout": "system=<aspect>_judge.yaml system_prompt; user=[jinja2-rendered template, poster image (camel encoding)]",
            "upstream": "third_party/Paper2Poster@623d042 utils/poster_eval_utils.eval_vlm_as_judge_aspect",
            "call_mode": "independent",
            "stateless_requests": True,
            "response_format": "none (free text parsed by upstream get_json_from_response; score must be an int in [1, 5])",
            "image_detail": args.image_detail,
            "automatic_retries": {
                "transport": f"up to {args.max_transport_retries} retries on timeout, connection error, HTTP 408/409/425/429/5xx",
                "invalid_output": f"up to {args.invalid_output_retries} fresh re-asks when JSON parsing or validation fails",
            },
        },
        "model_override": args.model,
        "reasoning_effort_override": args.reasoning_effort,
        "version": version,
        "paper": paper,
        "poster_path": str(poster.resolve()),
        "poster_sha256": sha256_bytes(poster.read_bytes()),
        "run_dir": str(run_dir.resolve()),
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "results": {},
        "api_runs": {},
        "errors": {},
        "status": "running",
    }


def summarize(report: dict[str, Any]) -> None:
    scores = {aspect: item["score"] for aspect, item in report["results"].items()}
    aesthetic = [scores[a] for a in ALL_ASPECTS[:3] if a in scores]
    information = [scores[a] for a in ALL_ASPECTS[3:] if a in scores]
    report["averages"] = {
        "aesthetic_average": sum(aesthetic) / len(aesthetic) if aesthetic else None,
        "information_average": sum(information) / len(information) if information else None,
        "overall_average": sum(scores.values()) / len(scores) if scores else None,
        "aspects_scored": len(scores),
    }


def scan_for_key(root: Path, api_key: str) -> list[str]:
    needle = api_key.encode("utf-8")
    return [str(path) for path in root.rglob("*") if path.is_file() and needle in path.read_bytes()]


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = None if args.dry_run else read_api_key(args.api_key_file)

    reports: dict[tuple[str, str], dict[str, Any]] = {}
    report_paths: dict[tuple[str, str], Path] = {}
    tasks: list[tuple[str, str, Path, str]] = []
    wanted = set(args.papers) if args.papers else None
    for version in args.versions:
        for poster in sorted((input_root / version).glob("*.png")):
            paper = poster.stem
            if wanted is not None and paper not in wanted:
                continue
            key = (version, paper)
            report_path = output_dir / version / f"{paper}.json"
            run_dir = output_dir / version / f"{paper}_api_runs"
            if args.resume and report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                settings = (report.get("model_override"), report.get("reasoning_effort_override"),
                            report["protocol"].get("image_detail"), report["protocol"].get("api_base_url"))
                if settings != (args.model, args.reasoning_effort, args.image_detail, args.base_url):
                    raise ValueError(f"--resume settings differ from {report_path}: {settings}")
                if report["poster_sha256"] != sha256_bytes(poster.read_bytes()):
                    raise ValueError(f"poster changed since {report_path} was written")
            else:
                if report_path.exists():
                    raise FileExistsError(f"{report_path} exists; use --resume")
                report = new_report(version=version, paper=paper, poster=poster, run_dir=run_dir, args=args)
            reports[key] = report
            report_paths[key] = report_path
            for aspect in ALL_ASPECTS:
                if aspect not in report["results"]:
                    tasks.append((version, paper, poster, aspect))
            if not args.dry_run:
                write_json(report_path, report)
    if wanted is not None:
        found = {paper for _, paper in reports}
        missing = sorted(wanted - found)
        if missing:
            print(f"requested papers without an input poster: {missing}", flush=True)

    random.Random(args.seed).shuffle(tasks)
    total = len(tasks)
    print(f"planned {total} aspect calls for {len(reports)} posters", flush=True)
    budget = RequestBudget(args.max_requests)
    local = threading.local()
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_one,
                poster=poster,
                aspect=aspect,
                run_dir=Path(reports[(version, paper)]["run_dir"]),
                args=args,
                api_key=api_key,
                budget=budget,
                local=local,
            ): (version, paper, aspect)
            for version, paper, poster, aspect in tasks
        }
        finished = 0
        for future in as_completed(futures):
            version, paper, aspect = futures[future]
            with lock:
                report = reports[(version, paper)]
                try:
                    record = future.result()
                    if record["status"] == "completed":
                        report["results"][aspect] = {"score": record["score"], "reason": record["reason"]}
                        report["errors"].pop(aspect, None)
                    report["api_runs"][aspect] = record
                    status = record["status"]
                except Exception as exc:  # persist every failure for --resume
                    report["errors"][aspect] = {"type": type(exc).__name__, "message": str(exc)}
                    status = "failed: " + str(exc)[:160]
                summarize(report)
                if not args.dry_run:
                    write_json(report_paths[(version, paper)], report)
                finished += 1
                print(f"[{finished}/{total}] {version}/{paper} {aspect}: {status}", flush=True)

    incomplete = 0
    for key, report in reports.items():
        summarize(report)
        complete = len(report["results"]) == len(ALL_ASPECTS) and not report["errors"]
        report["status"] = "dry_run" if args.dry_run else "completed" if complete else "partial"
        if not args.dry_run:
            write_json(report_paths[key], report)
        incomplete += 0 if complete else 1
    print(f"reports: {len(reports) - incomplete} complete, {incomplete} incomplete; HTTP requests sent: {budget.sent}", flush=True)
    if api_key is not None:
        hits = scan_for_key(output_dir, api_key)
        print(f"key scan over {output_dir}: {len(hits)} files contain the key", flush=True)
        if hits:
            return 3
    return 1 if incomplete and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
