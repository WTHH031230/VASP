"""Render frozen bboxes, run bounded independent VLM calls, and resume safely."""

from __future__ import annotations

import fcntl
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from src.benchmarks.vlm_judge.codex_exec import (
    CodexExecClient,
    CodexExecConfig,
    CodexExecError,
    _parse_events,
)

from .data import digest, file_sha256, read_json
from .protocol import PROMPT, PROTOCOL_VERSION, RESPONSE_SCHEMA, validate_response


RETRYABLE = {"timeout", "missing_final_message", "invalid_response", "interrupted"}


@dataclass(frozen=True)
class RunConfig:
    model: str = "gpt-5.6-sol"
    effort: str = "medium"
    dpi: int = 300
    jobs: int = 8
    timeout: int = 900
    max_attempts: int = 2
    pilot_size: int = 20
    seed: int = 20260908
    codex_bin: str = "codex"

    def validate(self) -> None:
        if self.dpi < 72 or self.jobs < 1 or self.timeout < 1:
            raise ValueError("dpi >= 72, jobs >= 1 and timeout >= 1 are required")
        if self.max_attempts not in (1, 2):
            raise ValueError("max_attempts must be 1 or 2")
        if self.pilot_size < 1:
            raise ValueError("pilot_size must be positive")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def item_dir(output_root: Path, method: str, figure: dict[str, Any]) -> Path:
    return output_root / "items" / method / figure["gt_key"]


def task_fingerprint(experiment: dict[str, Any], method: str, figure: dict[str, Any]) -> str:
    return digest({"experiment": experiment["fingerprint"], "method": method, "figure": figure})


def choose_pilot(figures: list[dict[str, Any]], config: RunConfig) -> list[str]:
    common = sorted(
        row["gt_key"] for row in figures
        if all(prediction is not None for prediction in row["predictions"].values())
    )
    if config.pilot_size > len(common):
        raise ValueError(f"pilot needs {config.pilot_size} common GTs; only {len(common)} available")
    return sorted(random.Random(config.seed).sample(common, config.pilot_size))


def ensure_experiment(output_root: Path, catalog: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    """Freeze source files, PDFs, prompt, schema and model before any paid call."""

    config.validate()
    pilot = choose_pilot(catalog["figures"], config)
    identity = {
        "protocol": PROTOCOL_VERSION,
        "catalog": catalog,
        "model": config.model,
        "effort": config.effort,
        "dpi": config.dpi,
        "pymupdf_version": fitz.VersionBind,
        "prompt": PROMPT,
        "response_schema": RESPONSE_SCHEMA,
        "max_attempts": config.max_attempts,
        "timeout": config.timeout,
        "pilot_gt_keys": pilot,
        "seed": config.seed,
        "tool_policy": "disabled-fail-closed",
        "render_policy": "original PDF / fixed grouped bbox / same DPI / no added margin",
        "scoring": "1=complete; 0=missing content or whole-Figure miss; all-GT denominator",
    }
    fingerprint = digest(identity)
    path = output_root / "experiment.json"
    if path.exists():
        experiment = read_json(path)
        if experiment.get("fingerprint") != fingerprint:
            raise ValueError("experiment inputs/configuration changed; use a NEW output root")
        return experiment
    unrelated = [p.name for p in output_root.iterdir() if p.name != ".run.lock"]
    if unrelated:
        raise ValueError(f"refusing to mix into a non-experiment directory: {unrelated[:5]}")
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    source_paths.append(Path(__file__).parents[1] / "vlm_judge" / "codex_exec.py")
    experiment = {
        "fingerprint": fingerprint,
        "created_at": utc_now(),
        "identity": identity,
        "initial_runtime": asdict(config),
        "code_sha256": {str(p.resolve()): file_sha256(p) for p in source_paths},
        "notes": [
            "GT-assisted Figure ownership/grouping is inherited, not rerun.",
            "Measures frozen-bbox completeness, not native exported PNG fidelity.",
            "Execution errors remain unscored; they are never semantic zeros.",
            "Backend dollar estimates use its retained pricing profile, not an actual invoice.",
        ],
    }
    atomic_json(path, experiment)
    (output_root / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    atomic_json(output_root / "response.schema.json", RESPONSE_SCHEMA)
    atomic_json(output_root / "pilot_selection.json", {"seed": config.seed, "gt_keys": pilot})
    return experiment


def _base_result(experiment: dict[str, Any], method: str, figure: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "task_fingerprint": task_fingerprint(experiment, method, figure),
        "method": method,
        "gt_key": figure["gt_key"],
        "paper_title": figure["paper_title"],
        "figure_id": figure["figure_id"],
        "page": figure["page"],
        "updated_at": utc_now(),
    }


def existing_result(output_root: Path, experiment: dict[str, Any], method: str, figure: dict[str, Any]) -> dict[str, Any] | None:
    path = item_dir(output_root, method, figure) / "result.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("task_fingerprint") != task_fingerprint(experiment, method, figure):
        raise ValueError(f"stale result: {path}")
    if result.get("status") in {"judged", "missing_detection"}:
        validate_response({"complete": result.get("complete")})
    if result.get("status") == "missing_detection":
        if figure["predictions"][method] is not None or result["complete"] != 0:
            raise ValueError(f"invalid missing-detection record: {path}")
    return result


def initialize_misses(output_root: Path, experiment: dict[str, Any]) -> None:
    for figure in experiment["identity"]["catalog"]["figures"]:
        for method, prediction in figure["predictions"].items():
            previous = existing_result(output_root, experiment, method, figure)
            if prediction is None and previous is None:
                atomic_json(item_dir(output_root, method, figure) / "result.json", {
                    **_base_result(experiment, method, figure),
                    "status": "missing_detection", "complete": 0,
                    "decision_source": "saved_matching", "attempt_count": 0,
                })


def render_figure(output_root: Path, experiment: dict[str, Any], figure: dict[str, Any], dpi: int) -> dict[str, Any]:
    """Serial PyMuPDF rendering: no re-localization, padding or image resizing."""

    directory = output_root / "inputs" / figure["gt_key"]
    meta_path = directory / "images.json"
    boxes = {"reference": figure["gt_bbox"]}
    boxes.update({method: prediction["bbox"] for method, prediction in figure["predictions"].items() if prediction})
    identity = {
        "pdf_sha256": experiment["identity"]["catalog"]["pdf_sha256"][figure["pdf"]],
        "page": figure["page"], "boxes": boxes, "dpi": dpi,
        "pymupdf_version": fitz.VersionBind,
    }
    render_hash = digest(identity)
    if meta_path.exists():
        metadata = read_json(meta_path)
        if metadata.get("render_fingerprint") != render_hash:
            raise ValueError(f"render configuration changed: {meta_path}")
        for details in metadata["images"].values():
            path = output_root / details["path"]
            if not path.is_file() or file_sha256(path) != details["sha256"]:
                raise ValueError(f"prepared image changed or missing: {path}")
        return metadata
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    with fitz.open(figure["pdf"]) as pdf:
        if not 1 <= figure["page"] <= len(pdf):
            raise ValueError(f"GT page exceeds PDF: {figure}")
        page = pdf[figure["page"] - 1]
        for name, bbox in boxes.items():
            rect = fitz.Rect(
                bbox[0] * page.rect.width / 1000,
                bbox[1] * page.rect.height / 1000,
                bbox[2] * page.rect.width / 1000,
                bbox[3] * page.rect.height / 1000,
            )
            effective = rect & page.rect
            if effective.is_empty:
                raise ValueError(f"bbox has no PDF-page pixels: {figure['gt_key']}, {name}")
            pixmap = page.get_pixmap(dpi=dpi, clip=effective, colorspace=fitz.csRGB, alpha=False)
            path = directory / f"{name}.png"
            temporary = path.with_name(f".{name}.{os.getpid()}.png")
            pixmap.save(temporary)
            temporary.replace(path)
            images[name] = {
                "path": str(path.relative_to(output_root)),
                "sha256": file_sha256(path), "width": pixmap.width, "height": pixmap.height,
                "bbox_normalized": bbox, "bbox_pdf_points": list(rect),
                "effective_page_clip": list(effective),
            }
    metadata = {"render_fingerprint": render_hash, "identity": identity, "images": images}
    atomic_json(meta_path, metadata)
    return metadata


def _recover_attempt(path: Path) -> dict[str, Any]:
    """Recover a saved successful raw reply after interrupted result publication."""

    state_path = path / "state.json"
    state = read_json(state_path)
    if state["status"] != "started":
        return state
    meta_path = path / "trace" / "call_0001.meta.json"
    events_path = path / "trace" / "call_0001.events.jsonl"
    if meta_path.exists() and events_path.exists():
        raw_meta = read_json(meta_path)
        record = raw_meta["usage"]
        events = _parse_events(events_path.read_text(encoding="utf-8"))
        if record["status"] == "success" and not events.unexpected_items and not events.parse_errors:
            response = validate_response(json.loads(events.final_message or ""))
            state.update(status="accepted", response=response)
        else:
            state.update(status=record["status"], error=record.get("error"))
        state["backend_records"] = [record]
        state["usage_available"] = bool(events.usage)
    else:
        state.update(status="interrupted", error="reserved attempt interrupted before durable result")
    state["recovered_at"] = utc_now()
    atomic_json(state_path, state)
    return state


def judge_one(output_root: Path, experiment: dict[str, Any], config: RunConfig, method: str, figure: dict[str, Any]) -> dict[str, Any]:
    """At most two total attempts across resumes; score 0 is an accepted result."""

    previous = existing_result(output_root, experiment, method, figure)
    if previous and previous["status"] in {"judged", "missing_detection"}:
        return previous
    directory = item_dir(output_root, method, figure)
    images = read_json(output_root / "inputs" / figure["gt_key"] / "images.json")["images"]
    paths = [output_root / images[name]["path"] for name in ("reference", method)]
    request_hash = digest({
        "task": task_fingerprint(experiment, method, figure),
        "images": [images[name]["sha256"] for name in ("reference", method)],
    })
    states: list[dict[str, Any]] = []
    attempts_root = directory / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    for attempt_dir in sorted(attempts_root.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.isdigit():
            raise ValueError(f"unexpected attempt entry: {attempt_dir}")
        state = _recover_attempt(attempt_dir)
        if state.get("request_fingerprint") != request_hash:
            raise ValueError(f"stale attempt: {attempt_dir}")
        states.append(state)
    response = next((state["response"] for state in states if state["status"] == "accepted"), None)
    if response is not None:
        response = validate_response(response)
    while response is None and len(states) < config.max_attempts:
        if states and states[-1]["status"] not in RETRYABLE:
            break
        attempt_number = len(states) + 1
        attempt_dir = attempts_root / f"{attempt_number:03d}"
        state = {
            "attempt": attempt_number, "status": "started", "started_at": utc_now(),
            "request_fingerprint": request_hash,
        }
        atomic_json(attempt_dir / "state.json", state)
        client = CodexExecClient(CodexExecConfig(
            executable=config.codex_bin, model=config.model, reasoning_effort=config.effort,
            timeout_s=config.timeout, max_retries=0, trace_dir=attempt_dir / "trace",
            ledger_path=output_root / "backend_usage.jsonl", result_dir=directory,
            paper_id=f"{method}/{figure['gt_key']}", enforce_no_tools=True,
        ))
        try:
            response = client.invoke_json(
                prompt=PROMPT, images=paths, response_schema=RESPONSE_SCHEMA,
                validator=validate_response, label="figure-completeness",
            )
            state.update(status="accepted", response=response)
        except CodexExecError as exc:
            records = client.records
            status = records[-1]["status"] if records else "client_error"
            state.update(status=status, error=str(exc))
        state["backend_records"] = client.records
        events_path = attempt_dir / "trace" / "call_0001.events.jsonl"
        state["usage_available"] = bool(
            _parse_events(events_path.read_text(encoding="utf-8")).usage
        ) if events_path.exists() else False
        state["finished_at"] = utc_now()
        atomic_json(attempt_dir / "state.json", state)
        states.append(state)
    result = {
        **_base_result(experiment, method, figure),
        "request_fingerprint": request_hash,
        "status": "judged" if response is not None else "execution_error",
        "complete": response["complete"] if response is not None else None,
        "decision_source": "vlm", "attempt_count": len(states),
        "error": None if response is not None else states[-1].get("error", "attempt budget exhausted"),
        "image_sha256": [images[name]["sha256"] for name in ("reference", method)],
        "attempts": states,
    }
    atomic_json(directory / "result.json", result)
    return result


def run_experiment(
    output_root: Path, catalog: dict[str, Any], config: RunConfig,
    *, phase: str = "pilot", prepare_only: bool = False, report_only: bool = False,
) -> dict[str, Any]:
    """Lock one experiment; render serially, judge concurrently, save progress."""

    from .reports import write_reports

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another runner owns {output_root}") from exc
        experiment = ensure_experiment(output_root, catalog, config)
        initialize_misses(output_root, experiment)
        pilot = set(experiment["identity"]["pilot_gt_keys"])
        selected = [row for row in catalog["figures"] if phase == "full" or row["gt_key"] in pilot]
        methods = list(catalog["sources"])
        tasks = [(method, figure) for figure in selected for method in methods if figure["predictions"][method]]
        atomic_json(output_root / "run_status.json", {
            "phase": phase, "started_at": utc_now(), "config": asdict(config),
            "selected_gt": len(selected), "matched_tasks": len(tasks),
            "prepare_only": prepare_only, "report_only": report_only,
        })
        print(f"[{phase}] GT={len(selected)} methods={len(methods)} matched_tasks={len(tasks)}", flush=True)
        if not report_only:
            for index, figure in enumerate(selected, start=1):
                render_figure(output_root, experiment, figure, config.dpi)
                if index % 20 == 0 or index == len(selected):
                    print(f"[render] {index}/{len(selected)} GTs prepared", flush=True)
        report = write_reports(output_root, experiment, phase)
        if prepare_only or report_only:
            return report
        pending = []
        for method, figure in tasks:
            result = existing_result(output_root, experiment, method, figure)
            if result is None or result["status"] == "execution_error":
                pending.append((method, figure))
        print(f"[judge] new_or_unfinished={len(pending)} cached={len(tasks) - len(pending)} jobs={config.jobs}", flush=True)
        last_report = time.monotonic()
        with ThreadPoolExecutor(max_workers=config.jobs) as pool:
            futures = {pool.submit(judge_one, output_root, experiment, config, method, figure): (method, figure)
                       for method, figure in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                method, figure = futures[future]
                result = future.result()
                print(f"[judge {index}/{len(pending)}] {method} {figure['gt_key']} "
                      f"{result['status']} complete={result['complete']} attempts={result['attempt_count']}", flush=True)
                if time.monotonic() - last_report > 15 or index % config.jobs == 0:
                    write_reports(output_root, experiment, phase, gallery=False)
                    last_report = time.monotonic()
        report = write_reports(output_root, experiment, phase)
        atomic_json(output_root / "run_status.json", {
            "phase": phase, "finished_at": utc_now(), "config": asdict(config),
            "selected_gt": len(selected), "matched_tasks": len(tasks),
            "complete": report[phase]["finished"],
        })
        return report
