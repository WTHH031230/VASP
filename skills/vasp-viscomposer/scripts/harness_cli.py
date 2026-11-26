#!/usr/bin/env python3
"""Standalone CLI exposing VASP VisComposer's full deterministic harness to any agent.

Subcommands:
  init    Stage SemParser Asset Package (or custom user figure folder), generate visual contact sheets
          and grounded_context.json, and initialize workspace/.
  check   Run full contract validation, Playwright DOM rendering & measurement, quality_gate predicates,
          zoomed evidence_detail_sheet, checkpoint recording, best-checkpoint selection, and regression diff.
  fit     Apply deterministic small-overflow CSS fit (geometry_fit.py) and record a fitted checkpoint.
  export  Export the selected (or agent-specified) checkpoint to crisp PNG, print-ready PDF, and editable PPTX.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from engine import control, gate  # noqa: E402
from engine.arms import ARMS, Arm  # noqa: E402
from engine.core.contact_sheets import (  # noqa: E402
    appendix_contact_sheets,
    brand_contact_sheet,
    checkpoint_contact_sheet,
    evidence_detail_sheet,
    figure_contact_sheet,
)
from engine.core.export.poster_to_pptx import export_poster_to_pptx  # noqa: E402
from engine.core.geometry_fit import apply_small_overflow_fit  # noqa: E402
from engine.core.render_html import (  # noqa: E402
    PosterContractError,
    render_bundle,
    validate_source_safety,
)
from engine.core.visual_assets import image_palette_profile  # noqa: E402
from engine.core.workbench import load_json, stage_inputs, write_json  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def _render_ready(bundle: dict[str, Any]) -> dict[str, Any]:
    plan = {
        "concept": "",
        "reading_order": [],
        "territories": [],
        "nodes": [],
        "edges": [],
        **(bundle.get("design_plan") or {}),
    }
    return {**bundle, "design_plan": plan}


def _gate_metric_snapshot(gate_res: dict[str, Any]) -> dict[str, float | int]:
    geometry = gate_res.get("geometry") or {}
    fonts = gate_res.get("font_stats") or {}
    territories = geometry.get("territories") or []
    return {
        "semantic_issue_count": len(gate_res.get("semantic_issues") or []),
        "outside_or_missing_count": len(geometry.get("outside") or []) + len(geometry.get("missing_images") or []),
        "hard_overflow_count": len(geometry.get("hard_overflow") or []),
        "overflow_excess_px": float(geometry.get("overflow_excess_px") or 0),
        "empty_shell_count": len(geometry.get("empty_shells") or []),
        "small_scientific_image_count": len(geometry.get("small_scientific_images") or []),
        "below_17px_text_count": int(fonts.get("below_17_count") or 0),
        "max_vertical_void_ratio": round(
            max((float(item.get("largest_vertical_void_ratio") or 0) for item in territories), default=0.0), 4
        ),
        "max_painted_bottom_gap_ratio": round(
            max((float(item.get("painted_bottom_gap_ratio") or 0) for item in territories), default=0.0), 4
        ),
    }


def _stage_custom_assets(
    *, asset_dir: Path, workspace: Path, title: str | None
) -> tuple[dict[str, Any], list[Path], list[Path], set[str]]:
    """Fallback staging for plain folders of custom figures/notes without a SemParser manifest."""
    assets_out = workspace / "assets"
    assets_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SKILL_DIR / "schemas" / "design_state.schema.json", workspace / "design_state.schema.json")

    meta_path = asset_dir / "metadata.json"
    metadata: dict[str, Any] = load_json(meta_path) if meta_path.is_file() else {}
    if title:
        metadata["title"] = title
    metadata.setdefault("title", asset_dir.name)
    metadata.setdefault("authors", [])
    metadata.setdefault("institutes", [])

    paper_text = ""
    for candidate in ("text.txt", "paper.txt", "notes.md", "README.md"):
        p = asset_dir / candidate
        if p.is_file():
            paper_text = p.read_text(encoding="utf-8")
            break

    img_files = sorted(
        [p for p in asset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.casefold(),
    )
    catalog: list[dict[str, Any]] = []
    figure_paths: list[Path] = []
    allowed: set[str] = set()
    for idx, src in enumerate(img_files, start=1):
        rel_name = f"fig_{idx:02d}_{src.name}"
        dst = assets_out / rel_name
        shutil.copy2(src, dst)
        rel_path = f"assets/{rel_name}"
        allowed.add(rel_path)
        w = h = 0
        palette = {}
        if dst.suffix.lower() != ".svg":
            with Image.open(dst) as im:
                w, h = im.size
            try:
                palette = image_palette_profile(dst)
            except Exception:
                palette = {}
        catalog.append({
            "id": f"fig_{idx}",
            "label": src.stem.replace("_", " "),
            "kind": "figure",
            "scope": "body",
            "relative_path": rel_path,
            "width_px": w,
            "height_px": h,
            "aspect_ratio": round(w / h, 3) if h else None,
            "caption": "",
            "palette_profile": palette,
        })
        figure_paths.append(dst)

    context = {
        "metadata": metadata,
        "asset_catalog": catalog,
        "brand_requirements": {"logo": 0, "qr": 0},
        "brand_policy": {},
        "structured_evidence": {"tables": [], "equations": [], "algorithms": []},
        "paper_text": paper_text,
    }
    write_json(workspace / "grounded_context.json", context)
    return context, figure_paths, [], allowed


def _is_semparser_package(asset_dir: Path) -> bool:
    return (asset_dir / "manifest.json").is_file() or (asset_dir / "assets" / "figures").is_dir() or (asset_dir / "assets" / "meta").is_dir()


def cmd_init(args: argparse.Namespace) -> int:
    asset_dir = args.asset_dir.resolve()
    if not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
    work = args.work_dir.resolve()
    workspace = work / "workspace"
    checkpoints_dir = work / "checkpoints"
    inputs_dir = work / "model_inputs"
    for d in (workspace, checkpoints_dir, inputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.custom_assets and _is_semparser_package(asset_dir):
        context, figure_paths, appendix_paths, allowed_assets = stage_inputs(
            asset_dir=asset_dir, workspace=workspace, brand_dir=None
        )
        source_mode = "semparser_package"
    else:
        context, figure_paths, appendix_paths, allowed_assets = _stage_custom_assets(
            asset_dir=asset_dir, workspace=workspace, title=args.title
        )
        source_mode = "custom_assets"

    if args.title:
        context.setdefault("metadata", {})["title"] = args.title
        write_json(workspace / "grounded_context.json", context)

    contact = figure_contact_sheet(
        figure_paths=figure_paths,
        catalog=context.get("asset_catalog") or [],
        output=inputs_dir / "figure_contact_sheet.jpg",
    )
    app_sheets = appendix_contact_sheets(
        figure_paths=appendix_paths,
        catalog=context.get("asset_catalog") or [],
        output_dir=inputs_dir,
    )
    brand_sheet = brand_contact_sheet(
        catalog=context.get("asset_catalog") or [],
        workspace=workspace,
        output=inputs_dir / "brand_contact_sheet.jpg",
    )

    visual_order: list[dict[str, Any]] = []
    if contact:
        visual_order.append({"role": "primary_body_figures", "path": str(contact), "asset_count": len(figure_paths)})
    for idx, ap in enumerate(app_sheets, start=1):
        visual_order.append({"role": "secondary_appendix_figures", "path": str(ap), "page": idx})
    if brand_sheet:
        visual_order.append({"role": "brand_assets", "path": str(brand_sheet)})
    context["visual_input_order"] = visual_order
    write_json(workspace / "grounded_context.json", context)

    metadata = context.get("metadata") or {}
    paper_title = str(metadata.get("title") or metadata.get("paper_title") or asset_dir.name)
    brand_req = context.get("brand_requirements") or {}
    require_branding: bool | dict[str, Any] = brand_req if any(brand_req.values()) else False

    session_cfg = {
        "source_mode": source_mode,
        "asset_dir": str(asset_dir),
        "canvas_width": int(args.canvas_width),
        "canvas_height": int(args.canvas_height),
        "paper_title": paper_title,
        "allowed_assets": sorted(allowed_assets),
        "require_branding": require_branding,
    }
    write_json(work / "session_config.json", session_cfg)
    write_json(checkpoints_dir / "index.json", {"checkpoints": []})

    print(json.dumps({
        "status": "initialized",
        "source_mode": source_mode,
        "work_dir": str(work),
        "workspace": str(workspace),
        "canvas": [session_cfg["canvas_width"], session_cfg["canvas_height"]],
        "paper_title": paper_title,
        "body_figure_count": len(figure_paths),
        "appendix_figure_count": len(appendix_paths),
        "allowed_asset_count": len(allowed_assets),
        "require_branding": require_branding,
        "files": {
            "grounded_context": str(workspace / "grounded_context.json"),
            "design_state_schema": str(workspace / "design_state.schema.json"),
            "visual_input_order": visual_order,
        },
        "expected_workspace_files": [
            str(workspace / "poster.html"),
            str(workspace / "style.css"),
            str(workspace / "design_state.json"),
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def _copy_workspace_sources(workspace: Path, target: Path, arm: Arm) -> None:
    for name in (*gate.source_files(arm), "critique.json"):
        src = workspace / name
        if src.is_file():
            shutil.copy2(src, target / name)
    if (workspace / "assets").is_dir():
        shutil.copytree(workspace / "assets", target / "assets", dirs_exist_ok=True)


def _load_checkpoints(checkpoints_dir: Path) -> list[dict[str, Any]]:
    idx_path = checkpoints_dir / "index.json"
    if not idx_path.is_file():
        return []
    raw = load_json(idx_path).get("checkpoints") or []
    records: list[dict[str, Any]] = []
    for entry in raw:
        cdir = Path(entry["dir"])
        if not cdir.is_dir():
            continue
        rep = load_json(cdir / "validation.json") if (cdir / "validation.json").is_file() else {}
        gres = load_json(cdir / "quality_gate.json") if (cdir / "quality_gate.json").is_file() else {}
        records.append({
            "name": entry["name"],
            "dir": cdir,
            "report": rep,
            "gate": gres,
            "eligible": bool(entry.get("eligible")),
        })
    return records


def _save_checkpoints_index(checkpoints_dir: Path, records: list[dict[str, Any]]) -> None:
    write_json(
        checkpoints_dir / "index.json",
        {
            "checkpoints": [
                {
                    "name": r["name"],
                    "dir": str(r["dir"]),
                    "eligible": bool(r["eligible"]),
                    "accepted": bool((r.get("gate") or {}).get("accepted")),
                    "issue_count": len((r.get("gate") or {}).get("issues") or []),
                }
                for r in records
            ]
        },
    )


def _run_checkpoint(
    *,
    work: Path,
    label: str,
    canvas: tuple[int, int],
    relax_contract: bool = False,
) -> dict[str, Any]:
    arm = ARMS["full"]
    workspace = work / "workspace"
    checkpoints_dir = work / "checkpoints"
    cfg = load_json(work / "session_config.json")
    # Also dynamically include any new local files the agent placed inside workspace/assets/
    allowed_assets = set(cfg.get("allowed_assets") or [])
    if (workspace / "assets").is_dir():
        for p in (workspace / "assets").rglob("*"):
            if p.is_file():
                allowed_assets.add(p.relative_to(workspace).as_posix())

    paper_title = str(cfg.get("paper_title") or "")
    require_branding = cfg.get("require_branding", False)

    prior_records = _load_checkpoints(checkpoints_dir)
    # Auto-seed critique.json if the agent did not author one explicitly
    crit_path = workspace / "critique.json"
    if not crit_path.is_file():
        write_json(crit_path, {
            "checkpoint": len(prior_records),
            "root_causes": [],
            "decisions": [],
            "remaining_risks": [],
        })
    cdir = checkpoints_dir / label
    if cdir.exists():
        shutil.rmtree(cdir)
    cdir.mkdir(parents=True, exist_ok=False)
    _copy_workspace_sources(workspace, cdir, arm)

    prior_gate = prior_records[-1]["gate"] if prior_records else None

    report: dict[str, Any] = {"status": "contract_error"}
    error: str | None = None
    state: dict[str, Any] = {}
    html = (workspace / "poster.html").read_text(encoding="utf-8") if (workspace / "poster.html").is_file() else ""
    css = (workspace / "style.css").read_text(encoding="utf-8") if (workspace / "style.css").is_file() else ""
    if (workspace / "design_state.json").is_file():
        try:
            state = load_json(workspace / "design_state.json")
        except Exception:
            state = {}

    if relax_contract:
        bundle = {"html": html, "css": css, "design_plan": gate.design_plan(state, arm) if state else {}}
        try:
            validate_source_safety(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
            report = render_bundle(_render_ready(bundle), cdir, export_pdf=False, canvas_width=canvas[0], canvas_height=canvas[1])
        except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"render/safety error: {exc}"
    else:
        error, state, html = gate.contract_error(
            workspace, allowed_assets, arm, canvas_width=canvas[0], canvas_height=canvas[1]
        )
        if error is None:
            bundle, _ = gate.bundle_from_workspace(workspace, arm)
            try:
                report = render_bundle(
                    _render_ready(bundle), cdir, export_pdf=False, canvas_width=canvas[0], canvas_height=canvas[1]
                )
            except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                error = f"source contract error: {exc}"

    if error is not None:
        write_json(cdir / "validation.json", report)

    if state:
        result = gate.quality_gate(
            state=state,
            html=html,
            report=report,
            paper_title=paper_title,
            require_branding=require_branding,
            arm=arm,
            contract_error=error,
        )
    else:
        result = {
            "schema_version": 1,
            "accepted": error is None and relax_contract,
            "issues": ([error] if error else ([] if relax_contract else ["design_state.json is missing or unreadable"])),
            "semantic_warnings": [],
            "geometry": {"status": report.get("status"), "hard_overflow": report.get("hardOverflow") or []},
        }

    write_json(cdir / "quality_gate.json", result)
    write_json(workspace / "quality_gate.json", result)

    poster_png = cdir / "poster.png"
    detail_sheet = evidence_detail_sheet(
        report=report,
        poster=poster_png,
        output=cdir / "evidence_detail_sheet.jpg",
    )

    eligible = poster_png.is_file() and (
        relax_contract or all((cdir / name).is_file() for name in gate.source_files(arm))
    )
    record = {"name": label, "dir": cdir, "report": report, "gate": result, "eligible": eligible}
    all_records = [r for r in prior_records if r["name"] != label] + [record]
    _save_checkpoints_index(checkpoints_dir, all_records)
    checkpoint_contact_sheet(all_records, checkpoints_dir / "checkpoint_contact_sheet.jpg")

    best = control.select(all_records, arm)
    regression_audit = None
    if prior_gate:
        prev_snap = _gate_metric_snapshot(prior_gate)
        curr_snap = _gate_metric_snapshot(result)
        regression_audit = {
            "prior_checkpoint": prior_records[-1]["name"],
            "worsened": {k: {"prior": prev_snap[k], "current": curr_snap[k]} for k in prev_snap if float(curr_snap[k]) > float(prev_snap[k]) + 1e-6},
            "improved": {k: {"prior": prev_snap[k], "current": curr_snap[k]} for k in prev_snap if float(curr_snap[k]) < float(prev_snap[k]) - 1e-6},
        }

    return {
        "checkpoint": label,
        "checkpoint_dir": str(cdir),
        "eligible": eligible,
        "accepted": bool(result.get("accepted")),
        "recommended_stop": bool(control.should_stop(result, arm)),
        "fit_allowed": bool(control.fit_allowed(result, arm)),
        "best_checkpoint": best["name"] if best else None,
        "visuals": {
            "poster_png": str(poster_png) if poster_png.is_file() else None,
            "evidence_detail_sheet": str(detail_sheet) if detail_sheet else None,
            "checkpoint_contact_sheet": str(checkpoints_dir / "checkpoint_contact_sheet.jpg"),
        },
        "controller_state": {
            "deterministic_issues": result.get("issues") or [],
            "semantic_warnings": result.get("semantic_warnings") or [],
            "font_stats": result.get("font_stats") or {},
            "far_correspondence": result.get("far_correspondence") or [],
            "stage_centers": result.get("stage_centers") or [],
            "geometry": result.get("geometry") or {},
        },
        "regression_audit": regression_audit,
    }


def cmd_check(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    cfg = load_json(work / "session_config.json")
    canvas = (
        int(args.canvas_width or cfg["canvas_width"]),
        int(args.canvas_height or cfg["canvas_height"]),
    )
    if args.canvas_width or args.canvas_height:
        cfg["canvas_width"], cfg["canvas_height"] = canvas
        write_json(work / "session_config.json", cfg)

    existing = _load_checkpoints(work / "checkpoints")
    label = args.label or (f"01_initial" if not existing else f"{len(existing) + 1:02d}_revision")
    out = _run_checkpoint(work=work, label=label, canvas=canvas, relax_contract=args.relax_contract)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    arm = ARMS["full"]
    cfg = load_json(work / "session_config.json")
    canvas = (int(cfg["canvas_width"]), int(cfg["canvas_height"]))
    workspace = work / "workspace"
    records = _load_checkpoints(work / "checkpoints")
    if not records:
        raise RuntimeError("No checkpoints found; run check first.")

    if args.checkpoint == "best":
        target = control.select(records, arm) or records[-1]
    elif args.checkpoint == "current":
        target = records[-1]
    else:
        matches = [r for r in records if r["name"] == args.checkpoint]
        if not matches:
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        target = matches[0]

    for name in gate.source_files(arm):
        src = target["dir"] / name
        if src.is_file():
            shutil.copy2(src, workspace / name)

    fit_audit = apply_small_overflow_fit(workspace, target["report"])
    write_json(work / "deterministic_fit.json", fit_audit)
    out: dict[str, Any] = {"fit_audit": fit_audit}
    if fit_audit.get("applied"):
        label = f"{len(records) + 1:02d}_deterministic_fit"
        out["checkpoint_result"] = _run_checkpoint(work=work, label=label, canvas=canvas, relax_contract=False)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    arm = ARMS["full"]
    cfg = load_json(work / "session_config.json")
    canvas = (int(cfg["canvas_width"]), int(cfg["canvas_height"]))
    workspace = work / "workspace"
    allowed_assets = set(cfg.get("allowed_assets") or [])
    if (workspace / "assets").is_dir():
        for p in (workspace / "assets").rglob("*"):
            if p.is_file():
                allowed_assets.add(p.relative_to(workspace).as_posix())

    records = _load_checkpoints(work / "checkpoints")
    if args.checkpoint == "current":
        src_dir = workspace
        selected_name = "workspace_current"
    elif args.checkpoint == "best":
        sel = control.select(records, arm)
        if sel is None:
            sel = records[-1] if records else None
        if sel is None:
            src_dir, selected_name = workspace, "workspace_current"
        else:
            src_dir, selected_name = sel["dir"], sel["name"]
    else:
        matches = [r for r in records if r["name"] == args.checkpoint]
        if not matches:
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        src_dir, selected_name = matches[0]["dir"], matches[0]["name"]

    final_dir = (args.out.resolve() if args.out else work / "final")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    _copy_workspace_sources(src_dir, final_dir, arm)
    if not (final_dir / "critique.json").is_file():
        write_json(final_dir / "critique.json", {"checkpoint": 0, "root_causes": [], "decisions": [], "remaining_risks": []})
    if (src_dir / "quality_gate.json").is_file():
        shutil.copy2(src_dir / "quality_gate.json", final_dir / "quality_gate.json")

    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    want_pdf = "pdf" in formats
    want_pptx = "pptx" in formats

    html = (final_dir / "poster.html").read_text(encoding="utf-8")
    css = (final_dir / "style.css").read_text(encoding="utf-8")
    state = load_json(final_dir / "design_state.json") if (final_dir / "design_state.json").is_file() else {}
    if args.safety_only or not state:
        bundle = {"html": html, "css": css, "design_plan": gate.design_plan(state, arm) if state else {}}
        validate_source_safety(bundle, allowed_assets, canvas_width=canvas[0], canvas_height=canvas[1])
    else:
        bundle, _ = gate.bundle_from_workspace(final_dir, arm)
        gate.validate_bundle(bundle, allowed_assets, arm, canvas_width=canvas[0], canvas_height=canvas[1])

    report = render_bundle(
        _render_ready(bundle), final_dir, export_pdf=want_pdf, canvas_width=canvas[0], canvas_height=canvas[1]
    )
    pptx_info = None
    if want_pptx:
        pptx_info = export_poster_to_pptx(
            final_dir / "poster.html",
            final_dir / "poster.pptx",
            canvas_width=canvas[0],
            canvas_height=canvas[1],
        )

    print(json.dumps({
        "status": "exported",
        "selected_checkpoint": selected_name,
        "final_dir": str(final_dir),
        "outputs": {
            "html": str(final_dir / "poster.html"),
            "css": str(final_dir / "style.css"),
            "png": str(final_dir / "poster.png") if (final_dir / "poster.png").is_file() else None,
            "pdf": str(final_dir / "poster.pdf") if (final_dir / "poster.pdf").is_file() else None,
            "pptx": pptx_info["pptx"] if pptx_info else None,
        },
        "render_status": report.get("status"),
        "pptx_details": pptx_info,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Stage assets, build contact sheets, and initialize workspace.")
    p_init.add_argument("--asset-dir", type=Path, required=True, help="SemParser asset dir or plain folder of figures")
    p_init.add_argument("--work-dir", type=Path, required=True)
    p_init.add_argument("--canvas-width", type=int, default=2544)
    p_init.add_argument("--canvas-height", type=int, default=1696)
    p_init.add_argument("--title", default=None, help="Optional explicit poster title override")
    p_init.add_argument("--custom-assets", action="store_true", help="Force plain custom-folder staging mode")
    p_init.set_defaults(func=cmd_init)

    p_chk = sub.add_parser("check", help="Render poster in Chromium, measure DOM, run quality_gate, and save checkpoint.")
    p_chk.add_argument("--work-dir", type=Path, required=True)
    p_chk.add_argument("--label", default=None, help="Checkpoint label (auto-numbered if omitted)")
    p_chk.add_argument("--canvas-width", type=int, default=None)
    p_chk.add_argument("--canvas-height", type=int, default=None)
    p_chk.add_argument("--relax-contract", action="store_true", help="Render & measure with safety checks only (allows custom overrides)")
    p_chk.set_defaults(func=cmd_check)

    p_fit = sub.add_parser("fit", help="Run deterministic small-overflow fit on a checkpoint.")
    p_fit.add_argument("--work-dir", type=Path, required=True)
    p_fit.add_argument("--checkpoint", default="best", help="best, current, or explicit checkpoint label")
    p_fit.set_defaults(func=cmd_fit)

    p_exp = sub.add_parser("export", help="Export final poster to PNG, PDF, and editable PPTX.")
    p_exp.add_argument("--work-dir", type=Path, required=True)
    p_exp.add_argument("--checkpoint", default="best", help="best, current, or explicit checkpoint label")
    p_exp.add_argument("--formats", default="png,pdf,pptx", help="Comma-separated formats: png,pdf,pptx")
    p_exp.add_argument("--out", type=Path, default=None)
    p_exp.add_argument("--safety-only", action="store_true", help="Validate source safety only (skip strict design_state contract)")
    p_exp.set_defaults(func=cmd_export)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
