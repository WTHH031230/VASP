"""Offline integrity audit of frozen sources, image pairs and binary judgments."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import fitz

from .data import file_sha256, read_json
from .protocol import PROMPT, RESPONSE_SCHEMA, validate_response
from .runner import atomic_json, existing_result, item_dir, utc_now
from src.benchmarks.vlm_judge.codex_exec import _parse_events


def check_render_warnings(output_root: Path) -> dict[str, Any]:
    """Locate non-fatal PDF warnings without changing any submitted images."""

    experiment = read_json(output_root / "experiment.json")
    figures = experiment["identity"]["catalog"]["figures"]
    warnings = []
    fitz.TOOLS.mupdf_display_errors(False)
    fitz.TOOLS.mupdf_display_warnings(False)
    for index, figure in enumerate(figures, start=1):
        fitz.TOOLS.mupdf_warnings(reset=True)
        with fitz.open(figure["pdf"]) as pdf:
            page = pdf[figure["page"] - 1]
            box = figure["gt_bbox"]
            rect = fitz.Rect(box[0] * page.rect.width / 1000, box[1] * page.rect.height / 1000,
                             box[2] * page.rect.width / 1000, box[3] * page.rect.height / 1000)
            pixmap = page.get_pixmap(dpi=96, clip=rect, colorspace=fitz.csRGB, alpha=False)
            message = fitz.TOOLS.mupdf_warnings(reset=True)
            if message:
                warning = {"gt_key": figure["gt_key"], "paper_title": figure["paper_title"],
                           "pdf": figure["pdf"], "page": figure["page"], "warning": message,
                           "diagnostic_width": pixmap.width, "diagnostic_height": pixmap.height,
                           "gt_bbox": box}
                warnings.append(warning)
                print(json.dumps(warning, ensure_ascii=False), flush=True)
        if index % 100 == 0:
            print(f"[render warning check] {index}/{len(figures)}", flush=True)
    result = {"checked_gt_count": len(figures), "warning_cases": warnings,
              "existing_inputs_modified": False, "model_calls": 0}
    atomic_json(output_root / "render_warning_check.json", result)
    return result


def render_warning_comparisons(output_root: Path) -> dict[str, Any]:
    """Independent Poppler renders for visual QA only; never replace judge input."""

    warnings = read_json(output_root / "render_warning_check.json")["warning_cases"]
    destination = output_root / "render_warning_comparisons"
    destination.mkdir(exist_ok=True)
    rows = []
    for warning in warnings:
        metadata = read_json(output_root / "inputs" / warning["gt_key"] / "images.json")
        reference = metadata["images"]["reference"]
        dpi = metadata["identity"]["dpi"]
        rect = reference["effective_page_clip"]
        x0, y0 = math.floor(rect[0] * dpi / 72), math.floor(rect[1] * dpi / 72)
        width = math.ceil(rect[2] * dpi / 72) - x0
        height = math.ceil(rect[3] * dpi / 72) - y0
        prefix = destination / warning["gt_key"]
        command = ["pdftoppm", "-f", str(warning["page"]), "-l", str(warning["page"]),
                   "-r", str(dpi), "-x", str(x0), "-y", str(y0), "-W", str(width),
                   "-H", str(height), "-singlefile", "-png", warning["pdf"], str(prefix)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        image_path = prefix.with_suffix(".png")
        if completed.returncode or not image_path.is_file():
            raise RuntimeError(f"Poppler QA rendering failed: {completed.stderr}")
        rows.append({**warning, "comparison_image": str(image_path), "stderr": completed.stderr,
                     "comparison_sha256": file_sha256(image_path)})
        print(f"[Poppler QA] {warning['gt_key']}", flush=True)
    result = {"comparisons": rows, "judge_inputs_modified": False, "model_calls": 0}
    atomic_json(destination / "manifest.json", result)
    return result


def audit(output_root: Path, phase: str) -> dict[str, Any]:
    experiment = read_json(output_root / "experiment.json")
    identity = experiment["identity"]
    catalog = identity["catalog"]
    if identity["prompt"] != PROMPT or identity["response_schema"] != RESPONSE_SCHEMA:
        raise ValueError("current protocol differs from the frozen experiment")
    for source in catalog["sources"].values():
        directory = Path(source["evaluation_root"])
        for name, expected in source["file_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError(f"changed evaluation source: {directory / name}")
    for pdf, expected in catalog["pdf_sha256"].items():
        if file_sha256(Path(pdf)) != expected:
            raise ValueError(f"changed PDF: {pdf}")
    pilot = set(identity["pilot_gt_keys"])
    figures = [row for row in catalog["figures"] if phase == "full" or row["gt_key"] in pilot]
    counts = {method: {"complete": 0, "content_missing": 0, "whole_missing": 0} for method in catalog["sources"]}
    call_count = 0
    for figure in figures:
        metadata = read_json(output_root / "inputs" / figure["gt_key"] / "images.json")
        image_info = metadata["images"]
        if metadata["identity"]["page"] != figure["page"]:
            raise ValueError("rendered page differs from frozen GT")
        if image_info["reference"]["bbox_normalized"] != figure["gt_bbox"]:
            raise ValueError("reference bbox changed")
        for details in image_info.values():
            if file_sha256(output_root / details["path"]) != details["sha256"]:
                raise ValueError(f"image changed: {details['path']}")
        for method, prediction in figure["predictions"].items():
            result = existing_result(output_root, experiment, method, figure)
            if not result or result["status"] not in {"judged", "missing_detection"}:
                raise ValueError(f"unresolved judgment: {method}/{figure['gt_key']}")
            if prediction is None:
                if result["status"] != "missing_detection" or result["complete"] != 0 or result["attempt_count"] != 0:
                    raise ValueError("whole-Figure miss was not an automatic zero")
                counts[method]["whole_missing"] += 1
                continue
            if image_info[method]["bbox_normalized"] != prediction["bbox"]:
                raise ValueError("prediction bbox changed")
            expected_images = [image_info[name]["sha256"] for name in ("reference", method)]
            if result["image_sha256"] != expected_images:
                raise ValueError("judgment image fingerprint mismatch")
            score = validate_response({"complete": result["complete"]})["complete"]
            counts[method]["complete" if score else "content_missing"] += 1
            attempts = result["attempts"]
            if not 1 <= len(attempts) <= identity["max_attempts"]:
                raise ValueError("attempt budget violated")
            if any(state["status"] == "accepted" for state in attempts[:-1]):
                raise ValueError("a valid score was retried")
            for state in attempts:
                call_count += 1
                trace = item_dir(output_root, method, figure) / "attempts" / f"{state['attempt']:03d}" / "trace"
                if (trace / "call_0001.prompt.txt").read_text(encoding="utf-8") != PROMPT:
                    raise ValueError("model prompt changed")
                if read_json(trace / "call_0001.schema.json") != RESPONSE_SCHEMA:
                    raise ValueError("model output schema changed")
                meta = read_json(trace / "call_0001.meta.json")
                command = meta["command"]
                if command[command.index("--model") + 1] != identity["model"]:
                    raise ValueError("wrong model")
                if f'model_reasoning_effort="{identity["effort"]}"' not in command:
                    raise ValueError("wrong effort")
                required = ("--ephemeral", "--ignore-user-config", "--ignore-rules", "--strict-config", "--output-schema")
                if not all(flag in command for flag in required) or command.count("--image") != 2:
                    raise ValueError("not the isolated two-image call protocol")
                events = _parse_events((trace / "call_0001.events.jsonl").read_text(encoding="utf-8"))
                if state["status"] == "accepted":
                    if events.unexpected_items or events.parse_errors:
                        raise ValueError("un-auditable/tool-using accepted call")
                    raw = validate_response(json.loads(events.final_message or ""))
                    if raw["complete"] != score:
                        raise ValueError("stored score differs from raw model response")
    changed_sources = [path for path, expected in experiment["code_sha256"].items()
                       if file_sha256(Path(path)) != expected]
    if any(Path(path).name != "reports.py" for path in changed_sources):
        raise ValueError(f"scoring/calling source changed during run: {changed_sources}")
    result = {
        "phase": phase, "audited_at": utc_now(), "passed": True,
        "gt_count": len(figures), "method_counts": counts, "model_attempts": call_count,
        "source_evaluations_and_pdfs_unchanged": True,
        "all_prepared_image_hashes_match": True,
        "all_scores_match_raw_model_responses": True,
        "all_whole_misses_automatic_zero": True,
        "no_valid_score_retried": True,
        "scoring_and_backend_sources_unchanged": True,
        "non_scoring_source_changes_since_prepare": changed_sources,
        "note": "reports.py had trailing whitespace cleaned after prepare and before pilot; scoring code was unchanged.",
    }
    atomic_json(output_root / f"audit_{phase}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--phase", choices=("pilot", "full"), default="full")
    parser.add_argument("--render-check", action="store_true", help="Locate PDF warnings only; no model calls")
    parser.add_argument("--poppler-check", action="store_true", help="Render warning cases independently for QA only")
    args = parser.parse_args()
    if args.render_check:
        result = check_render_warnings(args.output_root.resolve())
    elif args.poppler_check:
        result = render_warning_comparisons(args.output_root.resolve())
    else:
        result = audit(args.output_root.resolve(), args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
