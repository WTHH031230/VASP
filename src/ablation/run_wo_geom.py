#!/usr/bin/env python3
"""Run one paper under the standalone w/o Whitespace/Geometry Verification (SV-Only) ablation arm.

Usage:
  python -m src.ablation.run_wo_geom --asset-dir DIR --output-dir DIR \
      [--model gpt-5.6-sol] [--provider openai-compat] [--effort medium] [--max-revisions 2]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src.viscomposer import compose, gate, schemas
from src.ablation import wo_geom
from src.viscomposer.designer import PI_CONFIG_DIR, PI_DIR, PiDesigner, ReplayDesigner
from src.viscomposer.core.harness import _appendix_contact_sheets, _brand_contact_sheet, _evidence_detail_sheet, _figure_contact_sheet, stage_inputs
from src.viscomposer.core.host_edits import HostEditError, parse_structured_message
from src.viscomposer.core.render_html import (
    PosterContractError,
    render_bundle,
    validate_source_safety,
)
from src.viscomposer.core.workbench import write_json

CODE_DIR = Path(__file__).resolve().parent
ARM = wo_geom.FULL_ARM


def _write_objects(workspace: Path, response: dict[str, Any]) -> None:
    write_json(workspace / "design_state.json", response["design_state"])
    write_json(workspace / "critique.json", response["critique"])


def apply_initial(workspace: Path, response: dict[str, Any]) -> None:
    required = {"poster_html", "style_css", "design_state", "critique", "summary"}
    if set(response) != required:
        raise HostEditError(f"initial designer response fields disagree with contract: {sorted(response)}")
    (workspace / "poster.html").write_text(str(response["poster_html"]), encoding="utf-8")
    (workspace / "style.css").write_text(str(response["style_css"]), encoding="utf-8")
    _write_objects(workspace, response)


def apply_revision(workspace: Path, response: dict[str, Any]) -> list[dict[str, str]]:
    required = {"file_edits", "design_state", "critique", "summary"}
    if set(response) != required:
        raise HostEditError(f"revision designer response fields disagree with contract: {sorted(response)}")
    buffers = {name: (workspace / name).read_text(encoding="utf-8") for name in ("poster.html", "style.css")}
    audit = []
    for index, edit in enumerate(response["file_edits"]):
        name = str(edit.get("file", ""))
        if name not in buffers:
            raise HostEditError(f"edit {index} targets forbidden file: {name}")
        old, new = str(edit.get("old_text", "")), str(edit.get("new_text", ""))
        count = buffers[name].count(old)
        if count != 1:
            raise HostEditError(f"edit {index} old_text occurs {count} times in {name}; expected exactly 1")
        buffers[name] = buffers[name].replace(old, new, 1)
        audit.append({
            "file": name,
            "reason": str(edit.get("reason", "")),
            "old_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
            "new_sha256": hashlib.sha256(new.encode("utf-8")).hexdigest(),
        })
    for name, content in buffers.items():
        (workspace / name).write_text(content, encoding="utf-8")
    _write_objects(workspace, response)
    return audit


def _render_ready(bundle: dict[str, Any]) -> dict[str, Any]:
    plan = {
        "concept": "", "reading_order": [], "territories": [], "nodes": [], "edges": [],
        **(bundle.get("design_plan") or {}),
    }
    return {**bundle, "design_plan": plan}


def _copy_sources(source: Path, target: Path) -> None:
    for name in wo_geom.SOURCE_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", target / "assets")


def render_checkpoint(
    *,
    workspace: Path,
    checkpoint_dir: Path,
    allowed_assets: set[str],
    paper_title: str,
    require_branding: bool | dict[str, Any],
    canvas: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(workspace, checkpoint_dir)
    report: dict[str, Any] = {"status": "contract_error"}
    error, state, html = gate.contract_error(
        workspace, allowed_assets, ARM,
        canvas_width=canvas[0], canvas_height=canvas[1],
    )
    if error is None:
        bundle, _ = gate.bundle_from_workspace(workspace, ARM)
        try:
            report = render_bundle(
                _render_ready(bundle), checkpoint_dir, export_pdf=False,
                canvas_width=canvas[0], canvas_height=canvas[1],
            )
        except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"source contract error: {exc}"
    if error is not None:
        write_json(checkpoint_dir / "validation.json", report)
    if state:
        result = wo_geom.quality_gate_sv_only(
            state=state, html=html, report=report, paper_title=paper_title,
            require_branding=require_branding, contract_error=error,
        )
    else:
        result = {
            "schema_version": 1,
            "accepted": False,
            "issues": [error or "design_state.json is unreadable"],
            "semantic_issues": [error or "design_state.json is unreadable"],
            "semantic_warnings": [],
        }
    write_json(checkpoint_dir / "quality_gate.json", result)
    write_json(workspace / "quality_gate.json", result)
    return report, result


def _record(name: str, directory: Path, report: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    eligible = (directory / "poster.png").is_file() and all(
        (directory / n).is_file() for n in wo_geom.SOURCE_FILES
    )
    return {"name": name, "dir": directory, "report": report, "gate": result, "eligible": eligible}


def export(
    *,
    selected: dict[str, Any],
    output_dir: Path,
    allowed_assets: set[str],
    canvas: tuple[int, int],
    safety_only: bool = False,
) -> dict[str, Any]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(selected["dir"], final_dir)
    bundle, _ = gate.bundle_from_workspace(final_dir, ARM)
    if safety_only:
        validate_source_safety(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
    else:
        gate.validate_bundle(bundle, allowed_assets, ARM, canvas_width=canvas[0], canvas_height=canvas[1])
    report = render_bundle(
        _render_ready(bundle), final_dir, export_pdf=True,
        canvas_width=canvas[0], canvas_height=canvas[1],
    )
    shutil.copy2(selected["dir"] / "quality_gate.json", final_dir / "quality_gate.json")
    for name in ("poster.html", "style.css", "poster.png", "poster.pdf", "validation.json",
                 "quality_gate.json", "design_state.json", "critique.json"):
        if (final_dir / name).is_file():
            shutil.copy2(final_dir / name, output_dir / name)
    return report


def _code_hashes() -> dict[str, str]:
    files = sorted([
        *CODE_DIR.glob("*.py"),
        PI_DIR / "poster_ext.ts",
        PI_CONFIG_DIR / "models.json",
        PI_CONFIG_DIR / "settings.json",
    ])
    return {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    canvas = (args.canvas_width, args.canvas_height)
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    workspace = output_dir / "workspace"
    checkpoints_root = output_dir / "checkpoints"
    traces_root = output_dir / "traces"
    for directory in (workspace, checkpoints_root, traces_root):
        directory.mkdir(parents=True)

    asset_dir = args.asset_dir.resolve()
    context, figure_paths, appendix_figure_paths, allowed_assets = stage_inputs(
        asset_dir=asset_dir, workspace=workspace, brand_dir=None
    )
    contact_sheet = _figure_contact_sheet(
        figure_paths=figure_paths, catalog=context["asset_catalog"],
        output=output_dir / "model_inputs" / "figure_contact_sheet.jpg",
    )
    appendix_sheets = _appendix_contact_sheets(
        figure_paths=appendix_figure_paths, catalog=context["asset_catalog"],
        output_dir=output_dir / "model_inputs",
    )
    brand_sheet = _brand_contact_sheet(
        catalog=context["asset_catalog"], workspace=workspace,
        output=output_dir / "model_inputs" / "brand_contact_sheet.jpg",
    )
    context["visual_input_order"] = [
        {"role": "primary_body_figures", "path": str(contact_sheet), "asset_count": len(figure_paths)},
        *[{"role": "secondary_appendix_figures", "path": str(ap), "page": idx}
          for idx, ap in enumerate(appendix_sheets, start=1)],
        *([{"role": "brand_assets", "path": str(brand_sheet)}] if brand_sheet else []),
    ]

    brand = context.get("brand_requirements") or {}
    require_branding = brand if any(brand.values()) else False
    metadata = context["metadata"]
    paper_title = str(metadata.get("title") or metadata.get("paper_title") or metadata.get("name") or asset_dir.name)
    write_json(output_dir / "model_package.json", compose.package(context, ARM, source="ours"))

    k = args.max_revisions
    initial_schema, revision_schema = wo_geom.response_schemas(max(k, 1))
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir()
    write_json(schema_dir / "initial_response.schema.json", initial_schema)
    write_json(schema_dir / "revision_response.schema.json", revision_schema)
    write_json(schema_dir / "submit_poster.json", schemas.tool_parameters(initial_schema))
    write_json(schema_dir / "submit_revision.json", schemas.tool_parameters(revision_schema))

    if args.designer == "pi":
        designer = PiDesigner(
            workspace=workspace, trace_root=traces_root, schema_dir=schema_dir,
            session_dir=output_dir / "pi_session", model=args.model,
            provider=args.provider, effort=args.effort, timeout=args.timeout,
        )
    else:
        designer = ReplayDesigner(
            source_traces=args.replay_traces.resolve(), trace_root=traces_root, arm=ARM,
        )

    turns: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    stop_reason = "checkpoint_limit"
    try:
        turn = designer.start(
            name="01_initial_design",
            prompt=wo_geom.initial_prompt(
                context, canvas_width=canvas[0], canvas_height=canvas[1], source="ours",
            ),
            images=compose.initial_images(context, ARM),
        )
        turns.append({"name": turn.name, **turn.usage})
        response = parse_structured_message(turn.message)
        write_json(traces_root / "01_initial_design" / "structured_response.json", response)
        apply_initial(workspace, response)
        report, result = render_checkpoint(
            workspace=workspace, checkpoint_dir=checkpoints_root / "01_initial",
            allowed_assets=allowed_assets, paper_title=paper_title,
            require_branding=require_branding, canvas=canvas,
        )
        checkpoints.append(_record("01_initial", checkpoints_root / "01_initial", report, result))

        # Fixed refinement budget K=2 (matching no_sem ablation)
        for index in range(1, k + 1):
            previous = checkpoints[-1]
            screenshot = previous["dir"] / "poster.png"
            detail = _evidence_detail_sheet(
                report=previous["report"], poster=screenshot,
                output=previous["dir"] / "evidence_detail_sheet.jpg",
            )
            name = f"{index + 1:02d}_visual_revision"
            turn = designer.resume(
                name=name,
                prompt=wo_geom.revision_prompt(
                    checkpoint=index, gate_result=previous["gate"],
                    final_checkpoint=index == k,
                    canvas_width=canvas[0], canvas_height=canvas[1],
                ),
                images=compose.revision_images(ARM, screenshot, detail),
            )
            turns.append({"name": turn.name, **turn.usage})
            response = parse_structured_message(turn.message)
            write_json(traces_root / name / "structured_response.json", response)
            edit_error = None
            try:
                audit = apply_revision(workspace, response)
            except HostEditError as exc:
                edit_error, audit = str(exc), []
            write_json(traces_root / name / "edit_audit.json", audit)
            directory = checkpoints_root / f"{index + 1:02d}_revision"
            report, result = render_checkpoint(
                workspace=workspace, checkpoint_dir=directory,
                allowed_assets=allowed_assets, paper_title=paper_title,
                require_branding=require_branding, canvas=canvas,
            )
            if edit_error:
                result["accepted"] = False
                result["issues"].insert(0, f"host edit rejected: {edit_error}")
                write_json(directory / "quality_gate.json", result)
                write_json(workspace / "quality_gate.json", result)
            checkpoints.append(_record(directory.name, directory, report, result))
    finally:
        session_stats = designer.close()

    # Deterministic geometry fit is disabled in w/o Whitespace/Geometry Verification
    fit_audit: dict[str, Any] = {
        "applied": False,
        "reason": "disabled in w/o Whitespace/Geometry Verification (SV-Only)",
    }
    write_json(output_dir / "deterministic_fit.json", fit_audit)

    selected = wo_geom.select_checkpoint(checkpoints)
    fallback = selected is None
    if fallback:
        candidates = [c for c in checkpoints if all((c["dir"] / n).is_file() for n in wo_geom.SOURCE_FILES)]
        selected = candidates[-1] if candidates else None
    final_report = (
        export(selected=selected, output_dir=output_dir, allowed_assets=allowed_assets,
               canvas=canvas, safety_only=fallback)
        if selected else None
    )

    totals_keys = (
        "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
        "uncached_input_tokens", "output_tokens",
    )
    usage = {
        "turns": turns,
        "totals": {
            **{k_: sum(int(t.get(k_, 0)) for t in turns) for k_ in totals_keys},
            "estimated_api_cost_usd": round(sum(float(t.get("estimated_api_cost_usd", 0)) for t in turns), 8),
        },
        "session_stats": session_stats,
    }
    write_json(output_dir / "llm_usage.json", usage)
    pi_version = None
    if args.designer == "pi":
        pi_exe = getattr(designer, "pi_bin", str(PI_DIR / "node_modules" / ".bin" / "pi"))
        pi_version = subprocess.run(
            [pi_exe, "--version"], capture_output=True, text=True
        ).stdout.strip() or None
    manifest = {
        "schema_version": 1,
        "architecture": "pi_poster_ablation_wo_geom_v1",
        "arm": "wo_geom",
        "description": "w/o Whitespace/Geometry Verification (SV-Only Verification)",
        "designer": getattr(designer, "backend", args.designer),
        "model": args.model,
        "reasoning_effort": args.effort,
        "provider": args.provider,
        "pi_version": pi_version,
        "max_revisions": k,
        "canvas": {"width_px": canvas[0], "height_px": canvas[1]},
        "asset_dir": str(asset_dir),
        "thread_id": getattr(designer, "thread_id", None),
        "stop_reason": stop_reason,
        "selected_checkpoint": selected["name"] if selected else None,
        "fallback_export": fallback and selected is not None,
        "deterministic_fit_applied": False,
        "checkpoints": [
            {
                "name": c["name"],
                "eligible": c["eligible"],
                "accepted": c["gate"].get("accepted"),
                "issue_count": len(c["gate"].get("issues") or []),
            }
            for c in checkpoints
        ],
        "final_status": (final_report or {}).get("status"),
        "api_equivalent_cost_usd": usage["totals"]["estimated_api_cost_usd"],
        "total_wall_seconds": round(time.monotonic() - started, 3),
        "code_sha256": _code_hashes(),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--designer", choices=("pi", "replay"), default="pi")
    parser.add_argument("--replay-traces", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", default="openai-compat")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--canvas-width", type=int, default=2544)
    parser.add_argument("--canvas-height", type=int, default=1696)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.designer == "replay" and not args.replay_traces:
        parser.error("--replay-traces is required with --designer replay")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
