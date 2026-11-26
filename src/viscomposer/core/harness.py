#!/usr/bin/env python3
"""Run the bounded persistent-session paper-to-poster designer harness."""

from __future__ import annotations

import argparse
import json
import io
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.viscomposer.core.codex_designer import (  # noqa: E402
    DesignerSessionError,
    PersistentCodexDesigner,
)
from src.viscomposer.core.claude_designer import (  # noqa: E402
    PersistentClaudeDesigner,
)
from src.semparser.llm.claude_cli import (  # noqa: E402
    is_claude_model,
)
from src.viscomposer.core.host_edits import (  # noqa: E402
    HostEditError,
    apply_initial_response,
    apply_revision_response,
    effective_response_schemas,
    parse_structured_message,
)
from src.viscomposer.core.host_prompts import (  # noqa: E402
    initial_host_prompt,
    regression_revision_host_prompt,
    revision_host_prompt,
)
from src.viscomposer.core.geometry_fit import (  # noqa: E402
    apply_small_overflow_fit,
)
from src.viscomposer.core.harness_budget import (  # noqa: E402
    add_budget_arguments,
    add_model_effort_arguments,
    resolve_cli_namespace,
    resolve_namespace_harness_budget,
    resolve_namespace_model_effort,
)
from src.viscomposer.core.render_html import (  # noqa: E402
    PosterContractError,
    render_bundle,
    validate_bundle,
    validate_source_safety,
)
from src.viscomposer.core.workbench import (  # noqa: E402
    SOURCE_FILES,
    bundle_from_workspace,
    load_json,
    render_checkpoint,
    source_diff,
    source_hashes,
    stage_inputs,
    write_json,
)


def usage_summary(turns: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    """Aggregate duration, tokens, and API-equivalent cost."""

    token_keys = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
    )
    totals = {
        key: sum(int(item.get(key, 0)) for item in turns)
        for key in token_keys
    }
    totals["estimated_api_cost_usd"] = round(
        sum(float(item.get("estimated_api_cost_usd", 0)) for item in turns), 8
    )
    return {
        "schema_version": 1,
        "persistent_session": True,
        "model_turns": len(turns),
        "total_wall_seconds": round(wall_seconds, 3),
        "turns": turns,
        "totals": totals,
    }


def _figure_contact_sheet(
    *,
    figure_paths: list[Path],
    catalog: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Create one bounded visual input instead of attaching every full-size Figure."""

    tile_width, image_height, label_height = 700, 430, 54
    columns = 2
    rows = max(1, math.ceil(len(figure_paths) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    figures = [item for item in catalog if item.get("kind") == "figure"]
    for index, path in enumerate(figure_paths):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (image_height + label_height)
        with Image.open(path) as source:
            tile = ImageOps.contain(
                source.convert("RGB"),
                (tile_width - 24, image_height - 20),
            )
        canvas.paste(
            tile,
            (
                x + (tile_width - tile.width) // 2,
                y + label_height + (image_height - tile.height) // 2,
            ),
        )
        label = str(figures[index].get("label") or f"Figure {index + 1}")
        draw.text((x + 16, y + 18), label, fill="#102633")
        draw.rectangle(
            (x + 4, y + 4, x + tile_width - 4, y + image_height + label_height - 4),
            outline="#c7d4d9",
            width=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)
    return output


def _appendix_contact_sheets(
    *,
    figure_paths: list[Path],
    catalog: list[dict[str, Any]],
    output_dir: Path,
    page_size: int = 8,
) -> list[Path]:
    """Paginate the secondary appendix pool so its candidates stay inspectable."""

    if page_size < 1:
        raise ValueError("page_size must be positive")
    appendix_items = [
        item for item in catalog if item.get("kind") == "appendix_figure"
    ]
    sheets: list[Path] = []
    for start in range(0, len(figure_paths), page_size):
        page_number = len(sheets) + 1
        page_paths = figure_paths[start : start + page_size]
        page_items = [
            {**item, "kind": "figure"}
            for item in appendix_items[start : start + page_size]
        ]
        output = output_dir / f"appendix_figure_contact_sheet_{page_number:02d}.jpg"
        sheets.append(
            _figure_contact_sheet(
                figure_paths=page_paths,
                catalog=page_items,
                output=output,
            )
        )
    return sheets


def _brand_contact_sheet(
    *,
    catalog: list[dict[str, Any]],
    workspace: Path,
    output: Path,
) -> Path | None:
    """Preview every permitted brand asset before the first design decision."""

    brands = [item for item in catalog if item.get("kind") == "brand"]
    if not brands:
        return None
    tile_width, image_height, label_height = 460, 190, 62
    columns = min(3, len(brands))
    rows = math.ceil(len(brands) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(brands):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (image_height + label_height)
        path = workspace / str(item["relative_path"])
        try:
            if path.suffix.lower() == ".svg":
                import cairosvg

                data = cairosvg.svg2png(
                    url=str(path),
                    output_width=900,
                    output_height=360,
                )
                source = Image.open(io.BytesIO(data))
            else:
                source = Image.open(path)
            with source:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                white.alpha_composite(rgba)
                tile = ImageOps.contain(
                    white.convert("RGB"),
                    (tile_width - 28, image_height - 20),
                )
            canvas.paste(
                tile,
                (
                    x + (tile_width - tile.width) // 2,
                    y + label_height + (image_height - tile.height) // 2,
                ),
            )
        except (ImportError, OSError, ValueError):
            draw.text((x + 18, y + 105), "Preview unavailable", fill="#8a2d2d")
        label = f"{str(item.get('brand_kind')).upper()} · {item.get('label')}"
        draw.text((x + 16, y + 20), label, fill="#102633")
        draw.rectangle(
            (x + 4, y + 4, x + tile_width - 4, y + image_height + label_height - 4),
            outline="#c7d4d9",
            width=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)
    return output



def _checkpoint_contact_sheet(
    checkpoints: list[dict[str, Any]],
    output: Path,
) -> None:
    """Create a labelled horizontal comparison of rendered checkpoints."""

    rendered = [
        (item["name"], item["dir"] / "poster.png")
        for item in checkpoints
        if (item["dir"] / "poster.png").is_file()
    ]
    if not rendered:
        return
    tile_width, tile_height, label_height = 720, 540, 52
    canvas = Image.new(
        "RGB",
        (tile_width * len(rendered), tile_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(rendered):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
        x = index * tile_width + (tile_width - tile.width) // 2
        canvas.paste(tile, (x, label_height + (tile_height - tile.height) // 2))
        draw.text((index * tile_width + 18, 16), label, fill="#102633")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def _evidence_detail_sheet(
    *, report: dict[str, Any], poster: Path, output: Path
) -> Path | None:
    """Create zoomed node crops for scientific-image admission decisions."""

    metrics = report.get("scientificImageMetrics") or []
    rects = report.get("nodeRects") or {}
    node_ids = list(dict.fromkeys(
        str(item.get("node_id"))
        for item in metrics
        if item.get("node_id") and item.get("node_id") in rects
    ))
    if not node_ids or not poster.is_file():
        return None
    tile_width, tile_height, label_height = 800, 430, 58
    columns = 2
    rows = math.ceil(len(node_ids) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    with Image.open(poster) as source:
        source = source.convert("RGB")
        for index, node_id in enumerate(node_ids):
            rect = rects[node_id]
            left = max(0, math.floor(float(rect["x"])))
            top = max(0, math.floor(float(rect["y"])))
            right = min(source.width, math.ceil(float(rect["x"]) + float(rect["w"])))
            bottom = min(source.height, math.ceil(float(rect["y"]) + float(rect["h"])))
            if right <= left or bottom <= top:
                continue
            crop = source.crop((left, top, right, bottom))
            crop.thumbnail((tile_width - 28, tile_height - label_height - 18))
            x0 = (index % columns) * tile_width
            y0 = (index // columns) * tile_height
            node_metrics = [item for item in metrics if item.get("node_id") == node_id]
            sizes = ", ".join(
                f"{float((item.get('effective') or {}).get('w') or 0):.0f}x"
                f"{float((item.get('effective') or {}).get('h') or 0):.0f}px"
                for item in node_metrics
            )
            draw.text((x0 + 16, y0 + 15), f"{node_id} | effective {sizes}", fill="#102633")
            x = x0 + (tile_width - crop.width) // 2
            y = y0 + label_height + (tile_height - label_height - crop.height) // 2
            sheet.paste(crop, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output


def _gate_score(checkpoint: dict[str, Any]) -> tuple[float, ...]:
    """Rank saved checkpoints without letting a later revision win by recency."""

    gate = checkpoint["gate"]
    geometry = gate.get("geometry") or {}
    fonts = gate.get("font_stats") or {}
    territories = geometry.get("territories") or []
    max_void = max(
        (float(item.get("largest_vertical_void_ratio") or 0) for item in territories),
        default=0,
    )
    outside_or_missing = len(geometry.get("outside") or []) + len(
        geometry.get("missing_images") or []
    )
    hard_overflow = geometry.get("hard_overflow") or []
    small_scientific_images = geometry.get("small_scientific_images") or []
    flex_margin_spacers = geometry.get("flex_margin_spacers") or []
    title_only_sidecars = geometry.get("title_only_sidecars") or []
    bottom_gaps = [
        max(0.0, float(item.get("painted_bottom_gap_ratio") or 0))
        for item in territories
    ]
    max_bottom_gap = max(bottom_gaps, default=0.0)
    dom_bottom_gaps = [
        max(0.0, float(item.get("bottom_gap_ratio") or 0))
        for item in territories
    ]
    bottom_gap_spread = (
        max(dom_bottom_gaps) - min(dom_bottom_gaps)
        if len(dom_bottom_gaps) >= 2 else 0.0
    )
    title_penalty = max(0.0, 64.0 - float(fonts.get("title_px") or 0))
    median_penalty = max(0.0, 19.0 - float(fonts.get("median_px") or 0))
    p10_penalty = max(0.0, 16.5 - float(fonts.get("p10_px") or 0))
    return (
        1.0 if gate.get("accepted") else 0.0,
        -float(len(gate.get("semantic_issues") or [])),
        -float(outside_or_missing),
        -float(geometry.get("overflow_excess_px") or 0),
        -float(len(hard_overflow)),
        -float(len(geometry.get("empty_shells") or [])),
        -float(len(small_scientific_images)),
        -float(len(flex_margin_spacers)),
        -float(len(title_only_sidecars)),
        -max_bottom_gap,
        -bottom_gap_spread,
        -max_void,
        -title_penalty,
        -median_penalty,
        -p10_penalty,
        -float(len(gate.get("issues") or [])),
    )


def _checkpoint_record(
    *,
    name: str,
    directory: Path,
    report: dict[str, Any],
    gate: dict[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    return {
        "name": name,
        "dir": directory,
        "report": report,
        "gate": gate,
        "source_hashes": source_hashes(workspace),
    }


def _manifest_checkpoints(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "directory": str(item["dir"]),
            "poster": (
                str(item["dir"] / "poster.png")
                if (item["dir"] / "poster.png").is_file()
                else None
            ),
            "accepted": item["gate"].get("accepted", False),
            "issues": item["gate"].get("issues", []),
            "source_hashes": item["source_hashes"],
        }
        for item in checkpoints
    ]


def _copy_final(
    *,
    selected: dict[str, Any],
    output_dir: Path,
    allowed_assets: set[str],
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> dict[str, Any]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        shutil.copy2(selected["dir"] / name, final_dir / name)
    shutil.copytree(selected["dir"] / "assets", final_dir / "assets")
    bundle, _ = bundle_from_workspace(final_dir)
    validate_bundle(
        bundle,
        allowed_assets,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    final_report = render_bundle(
        bundle,
        final_dir,
        export_pdf=True,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    shutil.copy2(selected["dir"] / "quality_gate.json", final_dir / "quality_gate.json")
    for name in (
        "poster.html",
        "style.css",
        "poster.png",
        "poster.pdf",
        "design_state.json",
        "critique.json",
        "validation.json",
        "quality_gate.json",
    ):
        shutil.copy2(final_dir / name, output_dir / name)
    return final_report


def _export_last_checkpoint_fallback(
    *,
    checkpoints: list[dict[str, Any]],
    output_dir: Path,
    allowed_assets: set[str],
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> dict[str, Any]:
    """Render the last round when every checkpoint was rejected before rendering.

    Only the safety part of the source contract is enforced; plan/DOM
    correspondence errors do not block this export. Returns an audit record
    whose ``poster`` is None when no safe renderable source exists.
    """

    audit: dict[str, Any] = {
        "applied": False,
        "reason": "no checkpoint passed the source contract",
        "source_contract": "safety_only",
        "checkpoint": None,
        "poster": None,
    }
    candidates = [
        item
        for item in checkpoints
        if all((item["dir"] / name).is_file() for name in SOURCE_FILES)
    ]
    if not candidates:
        audit["error"] = "no checkpoint has complete source files"
        return audit
    last = candidates[-1]
    audit["checkpoint"] = last["name"]
    final_dir = output_dir / "final"
    try:
        final_dir.mkdir(parents=True, exist_ok=False)
        for name in SOURCE_FILES:
            shutil.copy2(last["dir"] / name, final_dir / name)
        if (last["dir"] / "assets").is_dir():
            shutil.copytree(last["dir"] / "assets", final_dir / "assets")
        bundle, _ = bundle_from_workspace(final_dir)
        validate_source_safety(
            bundle,
            allowed_assets,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        report = render_bundle(
            bundle,
            final_dir,
            export_pdf=True,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
    except Exception as exc:  # includes browser errors; never mask the rejected run
        audit["error"] = f"{type(exc).__name__}: {exc}"
        return audit
    for name in ("quality_gate.json", "validation.json"):
        if (last["dir"] / name).is_file():
            shutil.copy2(last["dir"] / name, final_dir / name)
    for name in (
        "poster.html",
        "style.css",
        "poster.png",
        "poster.pdf",
        "design_state.json",
        "critique.json",
        "validation.json",
        "quality_gate.json",
    ):
        if (final_dir / name).is_file():
            shutil.copy2(final_dir / name, output_dir / name)
    audit.update(
        {
            "applied": True,
            "render_status": report.get("status"),
            "poster": str(output_dir / "poster.png"),
        }
    )
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one clean persistent designer run."""

    started = time.monotonic()
    budget = resolve_namespace_harness_budget(args)
    model_effort = resolve_namespace_model_effort(args)
    cost_limit_usd = getattr(args, "cost_limit_usd", None)
    turn_timeout_seconds = getattr(args, "timeout", None)
    turn_reserve_usd = getattr(args, "turn_reserve_usd", 1.5)
    revision_policy = getattr(args, "revision_policy", "stable")
    asset_dir = args.asset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    brand_arg = getattr(args, "brand_dir", None)
    brand_dir = brand_arg.expanduser().resolve() if brand_arg else None
    appendix_arg = getattr(args, "appendix_asset_dir", None)
    appendix_asset_dir = (
        appendix_arg.expanduser().resolve() if appendix_arg else None
    )
    if appendix_asset_dir is not None and not (
        appendix_asset_dir / "assets" / "meta" / "appendix_figures.json"
    ).is_file():
        raise FileNotFoundError(
            f"not an appendix asset overlay: {appendix_asset_dir}"
        )
    if not (asset_dir / "assets" / "meta" / "metadata.json").is_file():
        raise FileNotFoundError(f"not an asset-extraction result: {asset_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; persistent runs are clean-only"
        )
    if cost_limit_usd is not None and cost_limit_usd <= 0:
        raise ValueError("--cost-limit-usd must be positive when enabled")
    if turn_timeout_seconds is not None and turn_timeout_seconds <= 0:
        raise ValueError("--timeout must be positive when enabled")
    if turn_reserve_usd <= 0:
        raise ValueError("--turn-reserve-usd must be positive")
    if args.canvas_width < 1 or args.canvas_height < 1:
        raise ValueError("--canvas-width and --canvas-height must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = output_dir / "workspace"
    workspace.mkdir()
    checkpoints_root = output_dir / "checkpoints"
    checkpoints_root.mkdir()
    traces_root = output_dir / "traces"
    traces_root.mkdir()

    context, figure_paths, appendix_figure_paths, allowed_assets = stage_inputs(
        asset_dir=asset_dir,
        workspace=workspace,
        brand_dir=brand_dir,
        appendix_asset_dir=appendix_asset_dir,
    )
    brand_requirements = context.get("brand_requirements") or {}
    require_branding = brand_requirements if any(brand_requirements.values()) else False
    metadata = context["metadata"]
    paper_title = str(
        metadata.get("title")
        or metadata.get("paper_title")
        or metadata.get("name")
        or asset_dir.name
    )
    contact_sheet = _figure_contact_sheet(
        figure_paths=figure_paths,
        catalog=context["asset_catalog"],
        output=output_dir / "model_inputs" / "figure_contact_sheet.jpg",
    )
    appendix_sheets = _appendix_contact_sheets(
        figure_paths=appendix_figure_paths,
        catalog=context["asset_catalog"],
        output_dir=output_dir / "model_inputs",
    )
    brand_sheet = _brand_contact_sheet(
        catalog=context["asset_catalog"],
        workspace=workspace,
        output=output_dir / "model_inputs" / "brand_contact_sheet.jpg",
    )
    visual_inputs = [contact_sheet, *appendix_sheets] + (
        [brand_sheet] if brand_sheet else []
    )
    context["visual_input_order"] = [
        {
            "role": "primary_body_figures",
            "path": str(contact_sheet),
            "asset_count": len(figure_paths),
        },
        *[
            {
                "role": "secondary_appendix_figures",
                "path": str(path),
                "page": index,
            }
            for index, path in enumerate(appendix_sheets, start=1)
        ],
        *(
            [{"role": "brand_assets", "path": str(brand_sheet)}]
            if brand_sheet
            else []
        ),
    ]
    write_json(workspace / "grounded_context.json", context)
    initial_schema, revision_schema = effective_response_schemas(
        output_dir,
        max_visual_checkpoints=budget.max_revisions,
    )
    write_json(
        output_dir / "input_provenance.json",
        {
            "asset_dir": str(asset_dir),
            "input_mode": "asset_bundle",
            "appendix_asset_dir": (
                str(appendix_asset_dir) if appendix_asset_dir else None
            ),
            "paper_title": paper_title,
            "paper_figure_count": len(figure_paths),
            "appendix_figure_count": len(appendix_figure_paths),
            "model_visual_inputs": [str(path) for path in visual_inputs],
            "brand_dir": str(brand_dir) if brand_dir else None,
            "brand_requirements": brand_requirements,
            "brand_policy": context.get("brand_policy") or {},
            "external_aesthetic_reference": False,
            "prior_session_reused": False,
            "prior_poster_reused": False,
            "filesystem_protocol": "host_applied_exact_edits",
            "direct_agent_write": False,
            "direct_write_blocker": "kernel does not permit nested bwrap namespaces",
            "canvas": {
                "width_px": args.canvas_width,
                "height_px": args.canvas_height,
                "aspect_ratio": args.canvas_width / args.canvas_height,
            },
        },
    )

    designer_class = (
        PersistentClaudeDesigner if is_claude_model(args.model)
        else PersistentCodexDesigner
    )
    designer = designer_class(
        workspace=workspace,
        trace_root=traces_root,
        model=args.model,
        effort=model_effort,
        timeout=turn_timeout_seconds,
    )
    turn_records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    stop_reason = "checkpoint_limit"

    initial = designer.start(
        name="01_initial_design",
        prompt=initial_host_prompt(
            context,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
        ),
        images=visual_inputs,
        output_schema=initial_schema,
    )
    turn_records.append({"name": initial.name, **initial.usage})
    initial_response = parse_structured_message(initial.message)
    write_json(traces_root / initial.name / "structured_response.json", initial_response)
    apply_initial_response(workspace, initial_response)
    first_dir = checkpoints_root / "01_initial"
    report, gate = render_checkpoint(
        workspace=workspace,
        checkpoint_dir=first_dir,
        allowed_assets=allowed_assets,
        paper_title=paper_title,
        require_branding=require_branding,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
    )
    checkpoints.append(
        _checkpoint_record(
            name="01_initial",
            directory=first_dir,
            report=report,
            gate=gate,
            workspace=workspace,
        )
    )

    for feedback_index in range(1, budget.max_revisions + 1):
        if checkpoints[-1]["gate"].get("accepted"):
            stop_reason = "quality_gate_pass"
            break
        spent = sum(
            float(item.get("estimated_api_cost_usd", 0)) for item in turn_records
        )
        if (
            cost_limit_usd is not None
            and spent + turn_reserve_usd > cost_limit_usd
        ):
            stop_reason = "cost_guard"
            break
        previous = checkpoints[-1]
        screenshot = previous["dir"] / "poster.png"
        evidence_sheet = _evidence_detail_sheet(
            report=previous["report"],
            poster=screenshot,
            output=previous["dir"] / "evidence_detail_sheet.jpg",
        )
        regression_reference = None
        if (
            revision_policy == "regression-aware"
            and feedback_index == budget.max_revisions
            and len(checkpoints) >= 2
        ):
            regression_reference = checkpoints[-2]
        turn_name = f"{feedback_index + 1:02d}_visual_revision"
        turn = designer.resume(
            name=turn_name,
            prompt=(
                regression_revision_host_prompt(
                    checkpoint=feedback_index,
                    gate=previous["gate"],
                    prior_gate=regression_reference["gate"],
                    final_checkpoint=True,
                    canvas_width=args.canvas_width,
                    canvas_height=args.canvas_height,
                )
                if regression_reference is not None
                else revision_host_prompt(
                    checkpoint=feedback_index,
                    gate=previous["gate"],
                    final_checkpoint=feedback_index == budget.max_revisions,
                    canvas_width=args.canvas_width,
                    canvas_height=args.canvas_height,
                )
            ),
            images=(
                ([screenshot] if screenshot.is_file() else [])
                + ([evidence_sheet] if evidence_sheet is not None else [])
                + (
                    [regression_reference["dir"] / "poster.png"]
                    if regression_reference is not None
                    else []
                )
            ),
            output_schema=revision_schema,
        )
        turn_records.append({"name": turn.name, **turn.usage})
        response = parse_structured_message(turn.message)
        write_json(traces_root / turn.name / "structured_response.json", response)
        edit_error: str | None = None
        try:
            edit_audit = apply_revision_response(workspace, response)
        except HostEditError as exc:
            edit_error = str(exc)
            edit_audit = []
        write_json(traces_root / turn.name / "edit_audit.json", edit_audit)

        checkpoint_dir = checkpoints_root / f"{feedback_index + 1:02d}_revision"
        report, gate = render_checkpoint(
            workspace=workspace,
            checkpoint_dir=checkpoint_dir,
            allowed_assets=allowed_assets,
            paper_title=paper_title,
            require_branding=require_branding,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
        )
        if edit_error:
            gate["accepted"] = False
            gate["issues"].insert(0, f"host edit rejected: {edit_error}")
            write_json(checkpoint_dir / "quality_gate.json", gate)
            write_json(workspace / "quality_gate.json", gate)
        source_diff(
            previous["dir"],
            checkpoint_dir,
            checkpoint_dir / "source_diff.patch",
        )
        checkpoints.append(
            _checkpoint_record(
                name=checkpoint_dir.name,
                directory=checkpoint_dir,
                report=report,
                gate=gate,
                workspace=workspace,
            )
        )
        if gate.get("accepted"):
            stop_reason = "quality_gate_pass"
            break

    prefit_eligible = [
        item
        for item in checkpoints
        if (item["dir"] / "poster.png").is_file()
        and all((item["dir"] / name).is_file() for name in SOURCE_FILES)
    ]
    latest = (
        max(
            enumerate(prefit_eligible),
            key=lambda pair: (_gate_score(pair[1]), pair[0]),
        )[1]
        if prefit_eligible
        else None
    )
    if latest is not None and latest is not checkpoints[-1]:
        for name in SOURCE_FILES:
            shutil.copy2(latest["dir"] / name, workspace / name)
    latest_geometry = (latest["gate"].get("geometry") or {}) if latest else {}
    fit_audit = {
        "applied": False,
        "reason": (
            "quality preconditions not met" if latest else "no rendered checkpoint"
        ),
    }
    if (
        latest is not None
        and not latest["gate"].get("semantic_issues")
        and not latest_geometry.get("outside")
        and not latest_geometry.get("missing_images")
        and not latest_geometry.get("empty_shells")
    ):
        fit_audit = apply_small_overflow_fit(workspace, latest["report"])
    write_json(output_dir / "deterministic_fit.json", fit_audit)
    if fit_audit.get("applied"):
        checkpoint_dir = (
            checkpoints_root / f"{len(checkpoints) + 1:02d}_deterministic_fit"
        )
        report, gate = render_checkpoint(
            workspace=workspace,
            checkpoint_dir=checkpoint_dir,
            allowed_assets=allowed_assets,
            paper_title=paper_title,
            require_branding=require_branding,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
        )
        source_diff(
            latest["dir"],
            checkpoint_dir,
            checkpoint_dir / "source_diff.patch",
        )
        write_json(checkpoint_dir / "deterministic_fit.json", fit_audit)
        checkpoints.append(
            _checkpoint_record(
                name=checkpoint_dir.name,
                directory=checkpoint_dir,
                report=report,
                gate=gate,
                workspace=workspace,
            )
        )

    eligible = [
        item
        for item in checkpoints
        if (item["dir"] / "poster.png").is_file()
        and all((item["dir"] / name).is_file() for name in SOURCE_FILES)
    ]
    _checkpoint_contact_sheet(checkpoints, output_dir / "checkpoint_comparison.jpg")
    wall_seconds = time.monotonic() - started
    usage = usage_summary(turn_records, wall_seconds)
    write_json(output_dir / "llm_usage.json", usage)
    write_json(
        output_dir / "checkpoint_manifest.json",
        _manifest_checkpoints(checkpoints),
    )
    thread_ids = {item.get("thread_id") for item in turn_records}
    common_manifest = {
        "schema_version": 1,
        "architecture": "persistent_host_applied_designer_v1",
        "backend": getattr(designer, "backend", "codex exec"),
        "asset_dir": str(asset_dir),
        "input_mode": "asset_bundle",
        "output_dir": str(output_dir),
        "model": args.model,
        "model_effort": model_effort,
        "reasoning_effort": model_effort,
        "harness_level": budget.level,
        "max_revisions": budget.max_revisions,
        "max_visual_checkpoints": budget.max_revisions,
        "max_model_turns": budget.max_model_calls,
        "total_model_call_limit": budget.max_model_calls,
        "thread_id": designer.thread_id,
        "same_session_verified": len(thread_ids) == 1 and designer.thread_id in thread_ids,
        "model_turns": len(turn_records),
        "actual_revisions": max(0, len(turn_records) - 1),
        "visual_feedback_checkpoints": max(0, len(turn_records) - 1),
        "early_exit_on_quality_gate": True,
        "stop_reason": stop_reason,
        "conditioning": {
            "external_aesthetic_reference": False,
            "canvas": {
                "width_px": args.canvas_width,
                "height_px": args.canvas_height,
                "aspect_ratio": args.canvas_width / args.canvas_height,
            },
            "paper_figure_count": len(figure_paths),
            "appendix_figure_count": len(appendix_figure_paths),
            "brand_assets_only": bool(require_branding),
            "brand_requirements": brand_requirements,
            "revision_images_are_prior_stage_only": True,
            "revision_policy": revision_policy,
        },
        "filesystem_protocol": "host_applied_exact_edits",
        "direct_agent_write": False,
        "deterministic_fit_applied": bool(fit_audit.get("applied")),
        "cost_limit_usd": cost_limit_usd,
        "turn_timeout_seconds": turn_timeout_seconds,
        "runtime_guards": {
            "cost_limit_usd": cost_limit_usd,
            "turn_timeout_seconds": turn_timeout_seconds,
        },
        "api_equivalent_cost_usd": usage["totals"]["estimated_api_cost_usd"],
        "total_wall_seconds": usage["total_wall_seconds"],
    }
    if not eligible:
        fallback = _export_last_checkpoint_fallback(
            checkpoints=checkpoints,
            output_dir=output_dir,
            allowed_assets=allowed_assets,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
        )
        write_json(output_dir / "fallback_export.json", fallback)
        fallback_gate = (
            load_json(output_dir / "quality_gate.json")
            if fallback["applied"] and (output_dir / "quality_gate.json").is_file()
            else None
        )
        manifest = {
            **common_manifest,
            "status": "rejected",
            "selected_checkpoint": fallback["checkpoint"] if fallback["applied"] else None,
            "final_validation": fallback.get("render_status"),
            "final_gate": fallback_gate,
            "poster": fallback["poster"],
            "checkpoint_comparison": None,
            "fallback_export": fallback,
        }
        write_json(output_dir / "run_manifest.json", manifest)
        return manifest

    selected = max(
        enumerate(eligible),
        key=lambda pair: (_gate_score(pair[1]), pair[0]),
    )[1]
    final_report = _copy_final(
        selected=selected,
        output_dir=output_dir,
        allowed_assets=allowed_assets,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
    )
    selected_gate = load_json(selected["dir"] / "quality_gate.json")
    manifest = {
        **common_manifest,
        "status": "success" if selected_gate["accepted"] else "rejected",
        "selected_checkpoint": selected["name"],
        "final_validation": final_report.get("status"),
        "final_gate": selected_gate,
        "poster": str(output_dir / "poster.png"),
        "checkpoint_comparison": str(output_dir / "checkpoint_comparison.jpg"),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--brand-dir", type=Path)
    parser.add_argument("--appendix-asset-dir", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    add_model_effort_arguments(parser)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout per model turn in seconds; disabled when omitted.",
    )
    parser.add_argument("--canvas-width", type=int, default=2400)
    parser.add_argument("--canvas-height", type=int, default=1800)
    add_budget_arguments(parser)
    parser.add_argument(
        "--revision-policy",
        choices=("stable", "regression-aware"),
        default="stable",
    )
    parser.add_argument(
        "--cost-limit-usd",
        type=float,
        default=None,
        help="Optional estimated-cost guard before revisions; disabled when omitted.",
    )
    parser.add_argument(
        "--turn-reserve-usd",
        type=float,
        default=1.5,
        help="Estimated reserve for the next turn; used only with --cost-limit-usd.",
    )
    args = resolve_cli_namespace(parser, parser.parse_args())
    try:
        manifest = run(args)
    except (
        DesignerSessionError,
        FileNotFoundError,
        FileExistsError,
        HostEditError,
        PosterContractError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
