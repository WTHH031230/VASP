#!/usr/bin/env python3
"""Run one paper under the standalone w/o Iterative Semantic Verification (Initial SV + GV) ablation arm.

Usage:
  python -m src.ablation.run_wo_iter_sv --asset-dir DIR --output-dir DIR \
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

from src.viscomposer import schemas
from src.ablation import wo_iter_sv
from src.viscomposer.arms import Arm
from src.viscomposer.designer import PI_CONFIG_DIR, PI_DIR, PiDesigner, ReplayDesigner
from src.viscomposer.core.geometry_fit import apply_small_overflow_fit
from src.viscomposer.core.harness import _appendix_contact_sheets, _brand_contact_sheet, _evidence_detail_sheet, _figure_contact_sheet, stage_inputs
from src.viscomposer.core.host_edits import HostEditError, parse_structured_message
from src.viscomposer.core.render_html import (
    PosterContractError,
    render_bundle,
    validate_source_safety,
)
from src.viscomposer.core.workbench import write_json

CODE_DIR = Path(__file__).resolve().parent


def apply_initial(workspace: Path, response: dict[str, Any]) -> dict[str, Any]:
    required = {"poster_html", "style_css", "design_state", "critique", "summary"}
    if set(response) != required:
        raise HostEditError(f"initial designer response fields disagree with contract: {sorted(response)}")
    (workspace / "poster.html").write_text(str(response["poster_html"]), encoding="utf-8")
    (workspace / "style.css").write_text(str(response["style_css"]), encoding="utf-8")
    write_json(workspace / "design_state.json", response["design_state"])
    write_json(workspace / "critique.json", response["critique"])
    return response["design_state"] if isinstance(response["design_state"], dict) else {}


def apply_revision(workspace: Path, response: dict[str, Any]) -> list[dict[str, str]]:
    required = {"file_edits", "summary"}
    if set(response) != required:
        raise HostEditError(f"revision designer response fields disagree with contract: {sorted(response)}")
    buffers = {name: (workspace / name).read_text(encoding="utf-8") for name in wo_iter_sv.REVISION_SOURCE_FILES}
    audit = []
    for index, edit in enumerate(response["file_edits"]):
        name = str(edit.get("file", ""))
        if name in ("design_state.json", "critique.json"):
            continue
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
    return audit


def _copy_sources(source: Path, target: Path) -> None:
    for name in wo_iter_sv.INITIAL_SOURCE_FILES:
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
    initial_state: dict[str, Any] | None,
    canvas: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(workspace, checkpoint_dir)
    report: dict[str, Any] = {"status": "contract_error"}
    err, html = wo_iter_sv.contract_error(
        workspace,
        allowed_assets,
        initial_state=initial_state,
        canvas_width=canvas[0],
        canvas_height=canvas[1],
    )
    if err is None:
        bundle = wo_iter_sv.bundle_from_workspace(workspace, initial_state=initial_state)
        try:
            report = render_bundle(
                bundle,
                checkpoint_dir,
                export_pdf=False,
                canvas_width=canvas[0],
                canvas_height=canvas[1],
            )
        except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            err = f"source contract error: {exc}"
    if err is not None:
        write_json(checkpoint_dir / "validation.json", report)
    result = wo_iter_sv.quality_gate(
        html=html,
        report=report,
        paper_title=paper_title,
        require_branding=require_branding,
        initial_state=initial_state,
        contract_err=err,
    )
    write_json(checkpoint_dir / "quality_gate.json", result)
    write_json(workspace / "quality_gate.json", result)
    return report, result


def _record(name: str, directory: Path, report: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    eligible = (directory / "poster.png").is_file() and all(
        (directory / n).is_file() for n in wo_iter_sv.REVISION_SOURCE_FILES
    )
    return {"name": name, "dir": directory, "report": report, "gate": result, "eligible": eligible}


def export(
    *,
    selected: dict[str, Any],
    output_dir: Path,
    allowed_assets: set[str],
    initial_state: dict[str, Any] | None,
    canvas: tuple[int, int],
    safety_only: bool = False,
) -> dict[str, Any]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(selected["dir"], final_dir)
    bundle = wo_iter_sv.bundle_from_workspace(final_dir, initial_state=initial_state)
    if safety_only:
        validate_source_safety(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
    else:
        wo_iter_sv.validate_bundle(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
    report = render_bundle(
        bundle,
        final_dir,
        export_pdf=True,
        canvas_width=canvas[0],
        canvas_height=canvas[1],
    )
    shutil.copy2(selected["dir"] / "quality_gate.json", final_dir / "quality_gate.json")
    for name in (
        "poster.html",
        "style.css",
        "design_state.json",
        "critique.json",
        "poster.png",
        "poster.pdf",
        "validation.json",
        "quality_gate.json",
    ):
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
    write_json(output_dir / "model_package.json", context)

    k = args.max_revisions
    initial_schema, revision_schema = wo_iter_sv.response_schemas(max(k, 1))
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir()
    write_json(schema_dir / "initial_response.schema.json", initial_schema)
    write_json(schema_dir / "revision_response.schema.json", revision_schema)
    write_json(schema_dir / "submit_poster.json", schemas.tool_parameters(initial_schema))
    write_json(schema_dir / "submit_revision.json", schemas.tool_parameters(revision_schema))

    if args.designer == "replay":
        dummy_arm = Arm("wo_iter_sv", frozenset({"basic", "assets", "vf"}), "iter_geometry_only")
        designer: Any = ReplayDesigner(
            source_traces=args.replay_traces.resolve(), trace_root=traces_root, arm=dummy_arm
        )
    else:
        designer = PiDesigner(
            workspace=workspace,
            trace_root=traces_root,
            schema_dir=schema_dir,
            session_dir=output_dir / "pi_session",
            model=args.model,
            effort=args.effort,
            provider=args.provider,
            timeout=args.timeout,
        )

    turns: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    initial_state: dict[str, Any] = {}
    stop_reason = "checkpoint_limit" if k else "single_pass"
    try:
        if args.initial_from_run is not None:
            seed_run = args.initial_from_run.resolve()
            seed_trace = seed_run / "traces" / "01_initial_design"
            response = json.loads((seed_trace / "structured_response.json").read_text(encoding="utf-8"))
            turn_dir = traces_root / "01_initial_design"
            turn_dir.mkdir(parents=True, exist_ok=False)
            write_json(turn_dir / "structured_response.json", response)
            seed_usage = {}
            if (seed_trace / "usage.json").is_file():
                seed_usage = json.loads((seed_trace / "usage.json").read_text(encoding="utf-8"))
                write_json(turn_dir / "usage.json", seed_usage)
            turns.append({"name": "01_initial_design", **seed_usage})
        else:
            turn = designer.start(
                name="01_initial_design",
                prompt=wo_iter_sv.build_initial_prompt(context, canvas_width=canvas[0], canvas_height=canvas[1]),
                images=wo_iter_sv.initial_images(context),
            )
            turns.append({"name": turn.name, **turn.usage})
            response = parse_structured_message(turn.message)
            write_json(traces_root / turn.name / "structured_response.json", response)

        initial_state = apply_initial(workspace, response)
        report, result = render_checkpoint(
            workspace=workspace,
            checkpoint_dir=checkpoints_root / "01_initial",
            allowed_assets=allowed_assets,
            paper_title=paper_title,
            require_branding=require_branding,
            initial_state=initial_state,
            canvas=canvas,
        )
        checkpoints.append(_record("01_initial", checkpoints_root / "01_initial", report, result))

        for index in range(1, k + 1):
            if wo_iter_sv.should_stop(checkpoints[-1]["gate"]):
                stop_reason = "quality_gate_pass"
                break
            previous = checkpoints[-1]
            screenshot = previous["dir"] / "poster.png"
            detail = _evidence_detail_sheet(
                report=previous["report"],
                poster=screenshot,
                output=previous["dir"] / "evidence_detail_sheet.jpg",
            )
            name = f"{index + 1:02d}_visual_revision"
            current_html = (workspace / "poster.html").read_text(encoding="utf-8")
            current_css = (workspace / "style.css").read_text(encoding="utf-8")
            try:
                turn = designer.resume(
                    name=name,
                    prompt=wo_iter_sv.build_revision_prompt(
                        checkpoint=index,
                        gate=previous["gate"],
                        final_checkpoint=(index == k),
                        canvas_width=canvas[0],
                        canvas_height=canvas[1],
                        current_html=current_html,
                        current_css=current_css,
                    ),
                    images=wo_iter_sv.revision_images(screenshot, detail),
                )
            except Exception as rev_exc:
                stop_reason = f"revision_error:{type(rev_exc).__name__}"
                break
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
                workspace=workspace,
                checkpoint_dir=directory,
                allowed_assets=allowed_assets,
                paper_title=paper_title,
                require_branding=require_branding,
                initial_state=initial_state,
                canvas=canvas,
            )
            if edit_error:
                result["accepted"] = False
                result["issues"].insert(0, f"host edit rejected: {edit_error}")
                write_json(directory / "quality_gate.json", result)
                write_json(workspace / "quality_gate.json", result)
            checkpoints.append(_record(directory.name, directory, report, result))
            if wo_iter_sv.should_stop(result):
                stop_reason = "quality_gate_pass"
                break
    finally:
        session_stats = designer.close()

    prefit = wo_iter_sv.select_checkpoint(checkpoints)
    if prefit is not None and prefit is not checkpoints[-1]:
        for name in wo_iter_sv.REVISION_SOURCE_FILES:
            shutil.copy2(prefit["dir"] / name, workspace / name)
    fit_audit: dict[str, Any] = {
        "applied": False,
        "reason": "quality preconditions not met" if prefit else "no rendered checkpoint",
    }
    if prefit is not None and wo_iter_sv.fit_allowed(prefit["gate"]):
        fit_audit = apply_small_overflow_fit(workspace, prefit["report"])
    write_json(output_dir / "deterministic_fit.json", fit_audit)
    if fit_audit.get("applied"):
        directory = checkpoints_root / f"{len(checkpoints) + 1:02d}_deterministic_fit"
        report, result = render_checkpoint(
            workspace=workspace,
            checkpoint_dir=directory,
            allowed_assets=allowed_assets,
            paper_title=paper_title,
            require_branding=require_branding,
            initial_state=initial_state,
            canvas=canvas,
        )
        checkpoints.append(_record(directory.name, directory, report, result))

    selected = wo_iter_sv.select_checkpoint(checkpoints)
    fallback = selected is None
    if fallback:
        candidates = [c for c in checkpoints if all((c["dir"] / n).is_file() for n in wo_iter_sv.REVISION_SOURCE_FILES)]
        selected = candidates[-1] if candidates else None
    final_report = (
        export(
            selected=selected,
            output_dir=output_dir,
            allowed_assets=allowed_assets,
            initial_state=initial_state,
            canvas=canvas,
            safety_only=fallback,
        )
        if selected
        else None
    )

    totals_keys = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
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
        pi_version = subprocess.run([pi_exe, "--version"], capture_output=True, text=True).stdout.strip() or None
    manifest = {
        "schema_version": 1,
        "architecture": "pi_poster_ablation_wo_iter_sv_v2",
        "arm": "wo_iter_sv",
        "description": "w/o Iterative Semantic Verification (Initial SV + Ordinary Geometric Verification)",
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
        "deterministic_fit_applied": bool(fit_audit.get("applied")),
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
    parser.add_argument("--initial-from-run", type=Path, default=None, help="Seed Round 0 from an existing VASP (full) run directory")
    parser.add_argument("--designer", choices=("pi", "replay"), default="pi")
    parser.add_argument("--replay-traces", type=Path)
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
