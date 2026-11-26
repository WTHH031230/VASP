#!/usr/bin/env python3
"""Run one paper under one arm.

Mirrors src/viscomposer/core/harness.run on SemParser asset directories, with every
module-owned step switched by the arm: prompts, package, images, response schema, gate,
feedback, early stop, selection, deterministic fit. Rendering always measures everything;
the arm decides what the model sees and what the loop uses.

  python -m src.viscomposer.run --arm full --asset-dir DIR --output-dir DIR \\
      [--designer pi|replay] [--replay-traces DIR] [--max-revisions 2]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from src.viscomposer import compose, control, gate, schemas
from src.viscomposer.arms import ALL_MODULES, ARMS, ASSETS, BASIC, MODULES, SEM, VF, Arm
from src.viscomposer.designer import AgyDesigner, PI_CONFIG_DIR, PI_DIR, PersistentCLIDesignerWrapper, PiDesigner, ReplayDesigner
from src.viscomposer.researchstudio import stage_researchstudio
from src.viscomposer.core.geometry_fit import apply_small_overflow_fit
from src.viscomposer.core.harness import _appendix_contact_sheets, _brand_contact_sheet, _evidence_detail_sheet, _figure_contact_sheet, stage_inputs
from src.viscomposer.core.host_edits import HostEditError, parse_structured_message
from src.viscomposer.core.render_html import (
    PosterContractError,
    render_bundle,
    validate_source_safety,
)
from src.viscomposer.core.workbench import load_json, write_json

CODE_DIR = Path(__file__).resolve().parent


# --- host-applied responses ---------------------------------------------------------

def _required(arm: Arm, turn: str) -> set[str]:
    fields = {"poster_html", "style_css", "summary"} if turn == "initial" else {"file_edits", "summary"}
    if arm.declares_state:
        fields.add("design_state")
    if arm.declares_plan:
        fields.add("critique")
    return fields


def apply_initial(workspace: Path, response: dict[str, Any], arm: Arm) -> None:
    required = _required(arm, "initial")
    if set(response) != required:
        raise HostEditError(f"initial designer response fields disagree with contract: {sorted(response)}")
    (workspace / "poster.html").write_text(str(response["poster_html"]), encoding="utf-8")
    (workspace / "style.css").write_text(str(response["style_css"]), encoding="utf-8")
    _write_objects(workspace, response, arm)


def apply_revision(workspace: Path, response: dict[str, Any], arm: Arm) -> list[dict[str, str]]:
    """Atomic unique exact replacements, as host_edits.apply_revision_response."""
    required = _required(arm, "revision")
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
        audit.append({"file": name, "reason": str(edit.get("reason", "")),
                      "old_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
                      "new_sha256": hashlib.sha256(new.encode("utf-8")).hexdigest()})
    for name, content in buffers.items():
        (workspace / name).write_text(content, encoding="utf-8")
    _write_objects(workspace, response, arm)
    return audit


def _write_objects(workspace: Path, response: dict[str, Any], arm: Arm) -> None:
    if arm.declares_state:
        write_json(workspace / "design_state.json", response["design_state"])
    if arm.declares_plan:
        write_json(workspace / "critique.json", response["critique"])


# --- render and gate one checkpoint ----------------------------------------------------

def _render_ready(bundle: dict[str, Any]) -> dict[str, Any]:
    """The renderer reads every plan key; absent module keys become empty."""
    plan = {"concept": "", "reading_order": [], "territories": [], "nodes": [], "edges": [],
            **(bundle.get("design_plan") or {})}
    return {**bundle, "design_plan": plan}


def _copy_sources(source: Path, target: Path, arm: Arm) -> None:
    for name in gate.source_files(arm):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", target / "assets")


def render_checkpoint(*, workspace: Path, checkpoint_dir: Path, allowed_assets: set[str], paper_title: str,
                      require_branding: bool | dict[str, Any], arm: Arm, canvas: tuple[int, int]
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """As workbench.render_checkpoint, with the arm's contract and gate."""
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(workspace, checkpoint_dir, arm)
    report: dict[str, Any] = {"status": "contract_error"}
    error, state, html = gate.contract_error(workspace, allowed_assets, arm,
                                             canvas_width=canvas[0], canvas_height=canvas[1])
    if error is None:
        bundle, _ = gate.bundle_from_workspace(workspace, arm)
        try:
            report = render_bundle(_render_ready(bundle), checkpoint_dir, export_pdf=False,
                                   canvas_width=canvas[0], canvas_height=canvas[1])
        except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"source contract error: {exc}"
    if error is not None:
        write_json(checkpoint_dir / "validation.json", report)
    if state or not arm.declares_state:
        result = gate.quality_gate(state=state, html=html, report=report, paper_title=paper_title,
                                   require_branding=require_branding, arm=arm, contract_error=error)
    else:
        result = {"schema_version": 1, "accepted": False,
                  "issues": [error or "design_state.json is unreadable"], "semantic_warnings": []}
    write_json(checkpoint_dir / "quality_gate.json", result)
    write_json(workspace / "quality_gate.json", result)
    return report, result


def _record(name: str, directory: Path, report: dict[str, Any], result: dict[str, Any], arm: Arm) -> dict[str, Any]:
    eligible = (directory / "poster.png").is_file() and all(
        (directory / name).is_file() for name in gate.source_files(arm))
    return {"name": name, "dir": directory, "report": report, "gate": result, "eligible": eligible}


# --- export ----------------------------------------------------------------------------

def export(*, selected: dict[str, Any], output_dir: Path, allowed_assets: set[str], arm: Arm,
           canvas: tuple[int, int], safety_only: bool = False) -> dict[str, Any]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    _copy_sources(selected["dir"], final_dir, arm)
    bundle, _ = gate.bundle_from_workspace(final_dir, arm)
    if safety_only:
        validate_source_safety(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
    else:
        gate.validate_bundle(bundle, allowed_assets, arm, canvas_width=canvas[0], canvas_height=canvas[1])
    report = render_bundle(_render_ready(bundle), final_dir, export_pdf=True,
                           canvas_width=canvas[0], canvas_height=canvas[1])
    shutil.copy2(selected["dir"] / "quality_gate.json", final_dir / "quality_gate.json")
    for name in ("poster.html", "style.css", "poster.png", "poster.pdf", "validation.json",
                 "quality_gate.json", *gate.source_files(arm)[2:]):
        if (final_dir / name).is_file():
            shutil.copy2(final_dir / name, output_dir / name)
    return report


# --- the loop --------------------------------------------------------------------------

def _code_hashes() -> dict[str, str]:
    files = sorted([*CODE_DIR.glob("*.py"), PI_DIR / "poster_ext.ts", PI_CONFIG_DIR / "models.json",
                    PI_CONFIG_DIR / "settings.json"])
    return {str(p.relative_to(PI_CONFIG_DIR.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    arm = ARMS[args.arm]
    canvas = (args.canvas_width, args.canvas_height)
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    workspace, checkpoints_root, traces_root = (output_dir / "workspace", output_dir / "checkpoints",
                                                output_dir / "traces")
    for directory in (workspace, checkpoints_root, traces_root):
        directory.mkdir(parents=True)
    asset_dir = args.asset_dir.resolve()
    asset_provenance: dict[str, Any] | None = None
    if arm.has(ASSETS):
        source = "ours"
        context, figure_paths, appendix_figure_paths, allowed_assets = stage_inputs(
            asset_dir=asset_dir, workspace=workspace, brand_dir=None)
        contact_sheet = _figure_contact_sheet(
            figure_paths=figure_paths, catalog=context["asset_catalog"],
            output=output_dir / "model_inputs" / "figure_contact_sheet.jpg")
        appendix_sheets = _appendix_contact_sheets(
            figure_paths=appendix_figure_paths, catalog=context["asset_catalog"],
            output_dir=output_dir / "model_inputs")
        brand_sheet = _brand_contact_sheet(
            catalog=context["asset_catalog"], workspace=workspace,
            output=output_dir / "model_inputs" / "brand_contact_sheet.jpg")
        context["visual_input_order"] = [
            {"role": "primary_body_figures", "path": str(contact_sheet), "asset_count": len(figure_paths)},
            *[{"role": "secondary_appendix_figures", "path": str(ap), "page": idx}
              for idx, ap in enumerate(appendix_sheets, start=1)],
            *([{"role": "brand_assets", "path": str(brand_sheet)}] if brand_sheet else []),
        ]
    else:
        source = "researchstudio"
        meta_path = asset_dir / "assets" / "meta" / "metadata.json"
        if not meta_path.is_file():
            meta_path = asset_dir / "metadata.json"
        paper_metadata = load_json(meta_path)
        context, allowed_assets, asset_provenance = stage_researchstudio(
            paper=asset_dir.name, metadata=paper_metadata, workspace=workspace,
            output_dir=output_dir, rs_root=args.rs_root)
    brand = context.get("brand_requirements") or {}
    require_branding = brand if (arm.has(ASSETS) and any(brand.values())) else False
    metadata = context["metadata"]
    paper_title = str(metadata.get("title") or metadata.get("paper_title") or metadata.get("name") or asset_dir.name)
    write_json(output_dir / "model_package.json",
               compose.package(context, arm, source=source))

    k = args.max_revisions if arm.has(BASIC) else 0
    initial_schema, revision_schema = schemas.response_schemas(arm, max(args.max_revisions, 1))
    schema_dir = output_dir / "schemas"
    schema_dir.mkdir()
    write_json(schema_dir / "initial_response.schema.json", initial_schema)
    write_json(schema_dir / "revision_response.schema.json", revision_schema)
    write_json(schema_dir / "submit_poster.json", schemas.tool_parameters(initial_schema))
    write_json(schema_dir / "submit_revision.json", schemas.tool_parameters(revision_schema))

    if args.designer in ("pi", "pi-api", "pi-chatgpt"):
        provider = "openai-codex" if (args.designer == "pi-chatgpt" and args.provider == "openai-compat") else args.provider
        designer = PiDesigner(workspace=workspace, trace_root=traces_root, schema_dir=schema_dir,
                              session_dir=output_dir / "pi_session", model=args.model, effort=args.effort,
                              provider=provider, timeout=args.timeout)
    elif args.designer == "codex":
        from src.viscomposer.core.codex_designer import PersistentCodexDesigner
        designer = PersistentCLIDesignerWrapper(
            PersistentCodexDesigner(workspace=workspace, trace_root=traces_root,
                                    model=args.model, effort=args.effort, timeout=int(args.timeout) if args.timeout else None),
            schema_dir=schema_dir, backend_name="codex cli")
    elif args.designer == "claude":
        from src.viscomposer.core.claude_designer import PersistentClaudeDesigner
        designer = PersistentCLIDesignerWrapper(
            PersistentClaudeDesigner(workspace=workspace, trace_root=traces_root,
                                     model=args.model, effort=args.effort, timeout=int(args.timeout) if args.timeout else None),
            schema_dir=schema_dir, backend_name="claude cli")
    elif args.designer == "agy":
        designer = AgyDesigner(workspace=workspace, trace_root=traces_root, schema_dir=schema_dir,
                               model=args.model, effort=args.effort, timeout=args.timeout)
    else:
        designer = ReplayDesigner(source_traces=args.replay_traces, trace_root=traces_root, arm=arm)

    harness_protocol = "pi" if args.designer in ("pi", "pi-api", "pi-chatgpt", "replay") else "agent"
    turns: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    stop_reason = "checkpoint_limit" if k else "single_pass"
    try:
        turn = designer.start(
            name="01_initial_design",
            prompt=compose.initial_prompt(context, arm, canvas_width=canvas[0], canvas_height=canvas[1],
                                          source=source, harness=harness_protocol),
            images=compose.initial_images(context, arm))
        turns.append({"name": turn.name, **turn.usage})
        response = parse_structured_message(turn.message)
        write_json(traces_root / turn.name / "structured_response.json", response)
        apply_initial(workspace, response, arm)
        report, result = render_checkpoint(workspace=workspace, checkpoint_dir=checkpoints_root / "01_initial",
                                           allowed_assets=allowed_assets, paper_title=paper_title,
                                           require_branding=require_branding, arm=arm, canvas=canvas)
        checkpoints.append(_record("01_initial", checkpoints_root / "01_initial", report, result, arm))

        for index in range(1, k + 1):
            if control.should_stop(checkpoints[-1]["gate"], arm):
                stop_reason = "quality_gate_pass"
                break
            previous = checkpoints[-1]
            screenshot = previous["dir"] / "poster.png"
            detail = (_evidence_detail_sheet(report=previous["report"], poster=screenshot,
                                             output=previous["dir"] / "evidence_detail_sheet.jpg")
                      if arm.has(VF) else None)
            name = f"{index + 1:02d}_visual_revision"
            try:
                turn = designer.resume(
                    name=name,
                    prompt=compose.revision_prompt(arm, checkpoint=index, gate=previous["gate"],
                                                   final_checkpoint=index == k,
                                                   canvas_width=canvas[0], canvas_height=canvas[1],
                                                   harness=harness_protocol),
                    images=compose.revision_images(arm, screenshot, detail))
            except Exception as rev_exc:
                stop_reason = f"revision_error:{type(rev_exc).__name__}"
                break
            turns.append({"name": turn.name, **turn.usage})
            response = parse_structured_message(turn.message)
            write_json(traces_root / name / "structured_response.json", response)
            edit_error = None
            try:
                audit = apply_revision(workspace, response, arm)
            except HostEditError as exc:
                edit_error, audit = str(exc), []
            write_json(traces_root / name / "edit_audit.json", audit)
            directory = checkpoints_root / f"{index + 1:02d}_revision"
            report, result = render_checkpoint(workspace=workspace, checkpoint_dir=directory,
                                               allowed_assets=allowed_assets, paper_title=paper_title,
                                               require_branding=require_branding, arm=arm, canvas=canvas)
            if edit_error:
                result["accepted"] = False
                result["issues"].insert(0, f"host edit rejected: {edit_error}")
                write_json(directory / "quality_gate.json", result)
                write_json(workspace / "quality_gate.json", result)
            checkpoints.append(_record(directory.name, directory, report, result, arm))
            if control.should_stop(result, arm):
                stop_reason = "quality_gate_pass"
                break
    finally:
        session_stats = designer.close()

    prefit = control.select(checkpoints, arm)
    if prefit is not None and prefit is not checkpoints[-1]:
        for name in gate.source_files(arm):
            shutil.copy2(prefit["dir"] / name, workspace / name)
    fit_audit: dict[str, Any] = {"applied": False, "reason": "quality preconditions not met" if prefit
                                 else "no rendered checkpoint"}
    if prefit is not None and control.fit_allowed(prefit["gate"], arm):
        fit_audit = apply_small_overflow_fit(workspace, prefit["report"])
    write_json(output_dir / "deterministic_fit.json", fit_audit)
    if fit_audit.get("applied"):
        directory = checkpoints_root / f"{len(checkpoints) + 1:02d}_deterministic_fit"
        report, result = render_checkpoint(workspace=workspace, checkpoint_dir=directory,
                                           allowed_assets=allowed_assets, paper_title=paper_title,
                                           require_branding=require_branding, arm=arm, canvas=canvas)
        checkpoints.append(_record(directory.name, directory, report, result, arm))

    selected = control.select(checkpoints, arm)
    fallback = selected is None
    if fallback:  # nothing rendered: export the last round after the safety check only
        candidates = [c for c in checkpoints if all((c["dir"] / n).is_file() for n in gate.source_files(arm))]
        selected = candidates[-1] if candidates else None
    final_report = (export(selected=selected, output_dir=output_dir, allowed_assets=allowed_assets, arm=arm,
                           canvas=canvas, safety_only=fallback) if selected else None)

    totals_keys = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                   "uncached_input_tokens", "output_tokens")
    usage = {"turns": turns,
             "totals": {**{k_: sum(int(t.get(k_, 0)) for t in turns) for k_ in totals_keys},
                        "estimated_api_cost_usd": round(sum(float(t.get("estimated_api_cost_usd", 0))
                                                            for t in turns), 8)},
             "session_stats": session_stats}
    write_json(output_dir / "llm_usage.json", usage)
    pi_version = None
    if args.designer == "pi":
        pi_exe = getattr(designer, "pi_bin", str(PI_DIR / "node_modules" / ".bin" / "pi"))
        pi_version = subprocess.run([pi_exe, "--version"],
                                    capture_output=True, text=True).stdout.strip() or None
    manifest = {
        "schema_version": 1,
        "architecture": "pi_poster_modular_v1",
        "arm": arm.name,
        "modules": [m for m in ALL_MODULES if arm.has(m)],
        "replacement": arm.replacement,
        "designer": getattr(designer, "backend", args.designer),
        "harness_protocol": harness_protocol,
        "model": args.model, "reasoning_effort": args.effort, "provider": args.provider,
        "pi_version": pi_version,
        "max_revisions": k,
        "canvas": {"width_px": canvas[0], "height_px": canvas[1]},
        "asset_dir": str(asset_dir),
        "asset_source": source,
        "asset_provenance": asset_provenance,
        "thread_id": getattr(designer, "thread_id", None),
        "stop_reason": stop_reason,
        "selected_checkpoint": selected["name"] if selected else None,
        "fallback_export": fallback and selected is not None,
        "deterministic_fit_applied": bool(fit_audit.get("applied")),
        "checkpoints": [{"name": c["name"], "eligible": c["eligible"], "accepted": c["gate"].get("accepted"),
                         "issue_count": len(c["gate"].get("issues") or [])} for c in checkpoints],
        "final_status": (final_report or {}).get("status"),
        "api_equivalent_cost_usd": usage["totals"]["estimated_api_cost_usd"],
        "total_wall_seconds": round(time.monotonic() - started, 3),
        "code_sha256": _code_hashes(),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--asset-dir", type=Path, required=True, help="SemParser output directory for one paper")
    parser.add_argument("--rs-root", type=Path, default=None, help="ResearchStudio assets root for the no_assets ablation (default: VASP_RS_ROOT)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--designer", choices=("pi", "pi-api", "pi-chatgpt", "codex", "claude", "agy", "replay"), default="pi")
    parser.add_argument("--replay-traces", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", default="openai-compat")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-revisions", type=int, default=2)
    parser.add_argument("--canvas-width", type=int, default=2544)
    parser.add_argument("--canvas-height", type=int, default=1696)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    if args.designer == "replay" and not args.replay_traces:
        parser.error("--replay-traces is required with --designer replay")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
