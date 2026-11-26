"""Standalone w/o Iterative Semantic Verification (Initial SV + Ordinary Geometric Verification) implementation.

Cleanly isolates Iterative Semantic Verification while keeping BOTH Initial SV (Round-0 design_state planning)
and Ordinary Geometric Verification (GV):
  1. Keeps Round-0 Initial SV & Visual-First (VF) planning:
     - 4-file structured declaration (poster.html, style.css, design_state.json, critique.json)
       establishing macro narrative stages, visual territories, and initial claim-evidence pairing.
  2. Keeps 100% of Ordinary Geometric Verification (GV) in quality_gate, controller_state,
     checkpoint ranking, early stop, and deterministic small-overflow padding fit:
     - Overflow, outside-canvas, missing images, empty shells, flex margin spacers,
       title-only sidecars, text orientation, typography floors, territory vertical void (<= 0.075),
       painted bottom gap (<= 0.035), shared bottom baseline spread (<= 0.025), figure size floors,
       verbatim H1 title, and brand/QR geometry.
     - IMPORTANT: Preserves Round-0 DOM attributes ([data-territory-id], [data-stage-id], [data-node-id])
       and design_plan directly without running wo_sv.ensure_geometry_hooks (which previously injected
       spurious box-* node IDs onto wrapper cards and caused false-positive emptyShells and checkpoint rollbacks).
  3. Completely removes Iterative Semantic Verification in revision turns (Rounds 1..K):
     - No semantic contract checks, no orphan-evidence check, no claim-evidence spatial distance
       (delta <= 0.42) check, and no stage progression check in quality_gate or controller_state.
     - Revision turns edit poster.html and style.css only (no design_state/critique updates) using
       VASP's native geometric/whitespace/typography repair craft, allowing the agent to condense
       verbose figure walkthrough prose into concise high-level bullets to resolve overflow,
       fund >=20px typography, and align column bottom baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.viscomposer import compose, schemas
from src.ablation import wo_sv
from src.viscomposer.arms import ARMS, ASSETS, BASIC, SEM, VF, Arm
from src.viscomposer.prompt_blocks import (
    AGENT_OPENING,
    INITIAL,
    PI_OPENING,
    PRINCIPLES,
    PROTOCOL,
    REVISION,
)
from src.viscomposer.core import host_prompts as HP
from src.viscomposer.core.prompts import canvas_prompt_text
from src.viscomposer.core.render_html import (
    PosterContractError,
    validate_source_safety,
)
from src.viscomposer.core.workbench import (
    _normalized_title_text,
    _visible_h1_titles,
    load_json,
)

FULL_ARM = ARMS["full"]
# Arm with SEM + VF + BASIC + ASSETS for native geometric & layout repair prompt blocks
GEOM_REVISION_ARM = Arm("geom_rev", frozenset({BASIC, ASSETS, VF, SEM}))

INITIAL_SOURCE_FILES = ("poster.html", "style.css", "design_state.json", "critique.json")
REVISION_SOURCE_FILES = ("poster.html", "style.css")


def response_schemas(max_revisions: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    initial_schema, _ = schemas.response_schemas(FULL_ARM, max(max_revisions, 1))
    _, revision_schema = wo_sv.response_schemas(max(max_revisions, 1))
    return initial_schema, revision_schema


def initial_images(context: dict[str, Any]) -> list[Path]:
    return compose.initial_images(context, FULL_ARM)


def revision_images(render: Path, detail_sheet: Path | None) -> list[Path]:
    return wo_sv.revision_images(render, detail_sheet)


INITIAL_SV_SCOPE_NOTE = (
    "INITIAL SEMANTIC PLAN (ROUND 0 ONLY — NO ITERATIVE SEMANTIC VERIFICATION):\n"
    "On this first turn, declare the 4-file `design_state` (`stages`, `territories`, `nodes`, `edges`, "
    "`reading_order`) to organize the poster's macro sections and place each selected Figure into its "
    "corresponding section. Because semantic verification is performed only for initial macro planning "
    "and is disabled during subsequent revisions, keep section body text concise and high-level "
    "(2–3 short summary bullets per section): state the main takeaway claims for major figures, "
    "and for secondary or multi-panel figures/tables rely on a short identifying `<figcaption>` "
    "rather than writing exhaustive panel-by-panel or row-by-row prose walkthroughs that would "
    "crowd the column budget."
)


def build_initial_prompt(context: dict[str, Any], *, canvas_width: int, canvas_height: int) -> str:
    text = PROTOCOL.compose(FULL_ARM)
    if not text.startswith(AGENT_OPENING):
        raise ValueError("composed protocol lost its opening sentence")
    protocol = PI_OPENING + text[len(AGENT_OPENING):]
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(FULL_ARM), canvas_width, canvas_height),
        HP.canvas_host_contract(canvas_width, canvas_height),
        protocol,
        INITIAL.compose(FULL_ARM),
        INITIAL_SV_SCOPE_NOTE,
        "GROUNDED PAPER PACKAGE\n"
        + json.dumps(
            compose.package(context, FULL_ARM, source="ours"),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ]
    return "\n\n".join(part for part in parts if part)


WO_ITER_SV_REVISION_PROTOCOL = (
    "This harness gives you no filesystem, shell, or web tools. Deliver each turn only by calling "
    "the one submit tool offered for that turn; its arguments are the structured response.\n"
    "On revision turns, return exact `old_text` -> `new_text` replacements in `file_edits` "
    "(at most 16 edits targeting `poster.html` or `style.css`) and `summary`; `design_state` and "
    "`critique` are not updated or checked during revisions.\n"
    "HTML must keep one `.poster` root element, the existing `[data-territory-id]` attributes on "
    "the story territories, `[data-node-id]` on content nodes, and `data-node-id=\"branding\"` "
    "on the header brand/QR cluster so the browser geometry engine can measure territory voids, "
    "bottom baselines, and box overflows."
)

WO_ITER_SV_REVISION_NOTE = (
    "GEOMETRIC & WHITESPACE REVISION TURN (model round {round} — ITERATIVE SEMANTIC VERIFICATION OFF).\n"
    "Iterative semantic verification (`design_state` edges, claim–evidence proximity, and prose-to-figure "
    "walkthrough checks) is disabled in revision turns. Your sole objective in this turn is to satisfy "
    "all geometric, whitespace, figure-scale, and typography predicates reported in CONTROLLER STATE:\n"
    "1. Eliminate all `hard_overflow`, `outside`, `empty_shells`, `flex_margin_spacers`, and `title_only_sidecars`.\n"
    "2. Eliminate internal vertical voids (`largest_vertical_void_ratio <= 0.075`) and align all scientific "
    "territories to one shared bottom baseline (`painted_bottom_gap_ratio <= 0.035`, `bottom_gap_spread <= 0.025`) "
    "strictly via geometric and visual adjustments (enlarging Figures, expanding figure/card containers, or "
    "rebalancing grid/flex proportions). Do NOT add new body prose or expand figure explanations to fill voids.\n"
    "3. Ensure every scientific Figure meets the legibility size floor (`>= 306x111 px`, largest `>= 3.5%` canvas area) "
    "and all text meets the print floor (`title >= 64px`, `chapter h2 >= 30px`, `median >= 19px`, `p10 >= 16.5px`, "
    "`below_17_count <= 4`).\n"
    "4. Fund larger body font sizes (`20–22px`) and resolve `hard_overflow` by condensing text: because iterative "
    "semantic verification is off, whenever a territory overflows or has `median < 19px` / `below_17_count > 0`, "
    "aggressively condense verbose panel-by-panel or table-walkthrough sentences in `poster.html` into concise, "
    "high-level 1-line summary bullets (relying on `<figcaption>` to identify each Figure/Table) so the larger "
    "font sizes and enlarged Figures fit cleanly with zero overflow."
)


def build_revision_prompt(
    *,
    checkpoint: int,
    gate: dict[str, Any],
    final_checkpoint: bool,
    canvas_width: int,
    canvas_height: int,
    current_html: str | None = None,
    current_css: str | None = None,
) -> str:
    round_no = checkpoint + 1
    # Use VASP's native REVISION repair craft, stripping SEM-only instructions that ask for prose expansion or design_state
    native_rev = REVISION.compose(GEOM_REVISION_ARM)
    native_rev = (
        native_rev.replace(
            "Return the full\nupdated design_state because it is small, and a critique for this checkpoint describing\nroot causes, integrated decisions, and remaining risks. ",
            "",
        )
        .replace(
            "Return the full updated design_state because it is small. ",
            "",
        )
        .replace(
            ", or fill the side column with substantive explanation",
            "",
        )
        .replace(
            "Narrative correctness\noutranks a numeric density target.",
            "Geometric fit, readable typography (>=20px body), and zero vertical void are the primary targets.",
        )
    )
    source_block = ""
    if current_html is not None and current_css is not None:
        source_block = (
            "CURRENT WORKSPACE SOURCE (apply exact `old_text` -> `new_text` replacements against these files)\n"
            f"=== poster.html ===\n{current_html}\n\n"
            f"=== style.css ===\n{current_css}"
        )
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(GEOM_REVISION_ARM), canvas_width, canvas_height),
        HP.canvas_host_contract(canvas_width, canvas_height),
        WO_ITER_SV_REVISION_PROTOCOL,
        WO_ITER_SV_REVISION_NOTE.format(round=round_no),
        native_rev,
        source_block,
        "CONTROLLER STATE\n" + json.dumps(wo_sv.controller_state(checkpoint, gate), ensure_ascii=False, indent=2),
    ]
    return "\n\n".join(part for part in parts if part)


def bundle_from_workspace(workspace: Path, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read poster.html and style.css directly WITHOUT calling wo_sv.ensure_geometry_hooks.

    Round 0 already declared [data-territory-id] and [data-node-id] in poster.html and design_state.json.
    Calling ensure_geometry_hooks previously injected extra box-* node IDs onto wrapper cards/figures,
    which caused false-positive emptyShells in render_html.py and rolled checkpoints back to 01_initial.
    """
    html_path = workspace / "poster.html"
    css_path = workspace / "style.css"
    if not html_path.is_file() or not css_path.is_file():
        raise PosterContractError("workspace must contain poster.html and style.css")
    raw_html = html_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    state = initial_state
    if not state and (workspace / "design_state.json").is_file():
        try:
            state = load_json(workspace / "design_state.json")
        except Exception:
            state = {}
    state = state if isinstance(state, dict) else {}
    return {
        "html": raw_html,
        "css": css,
        "design_plan": {
            "concept": str(state.get("concept") or ""),
            "reading_order": list(state.get("reading_order") or []),
            "territories": list(state.get("territories") or []),
            "nodes": list(state.get("nodes") or []),
            "edges": [],
        },
    }


def validate_bundle(
    bundle: dict[str, Any],
    allowed_assets: set[str],
    *,
    canvas_width: int,
    canvas_height: int,
) -> None:
    """Validate HTML/CSS safety and asset paths without enforcing iterative design_state DOM equality."""
    validate_source_safety(bundle, allowed_assets, canvas_width=canvas_width, canvas_height=canvas_height)


def contract_error(
    workspace: Path,
    allowed_assets: set[str],
    *,
    initial_state: dict[str, Any] | None = None,
    canvas_width: int,
    canvas_height: int,
) -> tuple[str | None, str]:
    html = ""
    try:
        if (workspace / "poster.html").is_file():
            html = (workspace / "poster.html").read_text(encoding="utf-8")
        bundle = bundle_from_workspace(workspace, initial_state=initial_state)
        validate_bundle(bundle, allowed_assets, canvas_width=canvas_width, canvas_height=canvas_height)
        return None, bundle["html"]
    except (PosterContractError, KeyError, TypeError, ValueError) as exc:
        return f"source contract error: {exc}", html


def _identity_territories(initial_state: dict[str, Any] | None) -> set[str]:
    base = {"masthead", "header", "branding", "identity"}
    if not initial_state or not isinstance(initial_state, dict):
        return base
    identity_nodes = {
        str(item.get("id"))
        for item in (initial_state.get("nodes") or [])
        if isinstance(item, dict) and item.get("semantic_role") == "identity"
    }
    for item in initial_state.get("territories") or []:
        if not isinstance(item, dict):
            continue
        nids = set(map(str, item.get("node_ids") or []))
        if nids and nids <= identity_nodes and item.get("id"):
            base.add(str(item["id"]))
    return base


def quality_gate(
    *,
    html: str,
    report: dict[str, Any],
    paper_title: str,
    require_branding: bool | dict[str, Any],
    initial_state: dict[str, Any] | None = None,
    contract_err: str | None = None,
) -> dict[str, Any]:
    """Evaluate ONLY geometric, typographic, whitespace, and figure-scale predicates (identical to GV in VASP)."""
    issues: list[str] = []
    warnings: list[str] = []
    if contract_err:
        issues.append(contract_err)

    expected_title = _normalized_title_text(paper_title)
    visible_titles = _visible_h1_titles(html)
    if len(visible_titles) != 1:
        issues.append("HTML must contain exactly one visible H1 paper title")
    elif visible_titles[0] != expected_title:
        issues.append("visible H1 must reproduce the extracted paper title verbatim")

    if report.get("status") != "pass":
        issues.append("browser geometry status is not pass")
    if report.get("hardOverflow"):
        issues.append(f"{len(report['hardOverflow'])} hard overflow item(s) remain")
    if report.get("outside"):
        issues.append(f"{len(report['outside'])} element(s) leave the canvas")
    if report.get("missingImages"):
        issues.append(f"{len(report['missingImages'])} local image(s) are missing")
    if report.get("emptyShells"):
        issues.append(f"{len(report['emptyShells'])} large empty content shell(s) remain")
    if report.get("flexMarginSpacers"):
        issues.append(f"{len(report['flexMarginSpacers'])} large flex margin spacer(s) split scientific content")
    if report.get("titleOnlySidecars"):
        issues.append(f"{len(report['titleOnlySidecars'])} title-only sidecar(s) leave a locally empty grid column")

    text_orientation_issues = report.get("textOrientationIssues") or []
    if text_orientation_issues:
        examples = ", ".join(repr(str(item.get("text") or "")[:36]) for item in text_orientation_issues[:4])
        issues.append(
            f"{len(text_orientation_issues)} informative text element(s) are rotated, skewed, or vertically written: {examples}"
        )

    fonts = report.get("fontStats") or {}
    title_px = float(fonts.get("title_px") or 0)
    median_px = float(fonts.get("median_px") or 0)
    p10_px = float(fonts.get("p10_px") or 0)
    if title_px < 64:
        issues.append(f"title is only {title_px:.1f}px")
    if median_px < 19:
        issues.append(f"median visible text is only {median_px:.1f}px")
    if p10_px < 16.5:
        issues.append(f"10th-percentile visible text is only {p10_px:.1f}px")
    if int(fonts.get("below_17_count") or 0) > 4:
        issues.append("too many text leaves fall below the 17px print floor")

    chapters = report.get("chapterMetrics") or []
    if chapters and min(float(item.get("h2_font_px") or 0) for item in chapters) < 30:
        issues.append("at least one chapter heading is below 30px")

    skip_tids = _identity_territories(initial_state)
    scientific_territories: list[dict[str, Any]] = []
    for item in report.get("territoryMetrics") or []:
        tid = str(item.get("id") or "")
        if tid in skip_tids:
            continue
        scientific_territories.append(item)
        if float(item.get("largest_vertical_void_ratio") or 0) > 0.075:
            issues.append(f"territory {tid} has a {float(item['largest_vertical_void_ratio']):.3f} vertical void")
        if float(item.get("painted_height_ratio") or 1) < 0.47:
            warnings.append(f"territory {tid} has low painted coverage ({float(item['painted_height_ratio']):.3f})")
        if float(item.get("painted_bottom_gap_ratio") or 0) > 0.035:
            issues.append(f"territory {tid} leaves a {float(item['painted_bottom_gap_ratio']):.3f} painted bottom void")

    bottom_gaps = [max(0.0, float(item.get("bottom_gap_ratio") or 0)) for item in scientific_territories]
    if len(bottom_gaps) >= 2 and max(bottom_gaps) - min(bottom_gaps) > 0.025:
        issues.append(
            f"scientific territories do not share one bottom baseline (painted-gap spread {max(bottom_gaps) - min(bottom_gaps):.3f})"
        )

    canvas = report.get("canvas") or [2544, 1696]
    canvas_width = max(1.0, float(canvas[0]))
    canvas_height = max(1.0, float(canvas[1]))
    raw_images = report.get("scientificImageMetrics") or []
    scientific_images = [
        img for img in raw_images
        if not any(k in str(img.get("src") or "").lower() for k in ("logo", "qr_", "brand"))
    ]
    small_scientific_images: list[dict[str, Any]] = []
    min_figure_width = canvas_width * 0.12
    min_figure_height = canvas_height * 0.065
    max_area_ratio = 0.0
    for item in scientific_images:
        effective = item.get("effective") or {}
        width = float(effective.get("w") or 0)
        height = float(effective.get("h") or 0)
        area_ratio = (width * height) / (canvas_width * canvas_height)
        if area_ratio > max_area_ratio:
            max_area_ratio = area_ratio
        if width + 0.5 < min_figure_width or height + 0.5 < min_figure_height:
            small_scientific_images.append(item)
            issues.append(
                f"figure {item.get('src')} is thumbnail-sized ({width:.0f}x{height:.0f}px effective); enlarge it or omit it"
            )
    if scientific_images and max_area_ratio < 0.035:
        issues.append(f"largest scientific figure occupies only {max_area_ratio:.3f} of the canvas (require >= 0.035)")

    brand = report.get("brandMetrics") or {}
    brand_padding_intrusions = brand.get("padding_intrusions") or []
    if require_branding:
        wo_sv._brand_issues(issues, brand, brand_padding_intrusions, require_branding)

    overflow_excess_px = round(
        sum(
            max(0.0, float(item["scroll"][0]) - float(item["client"][0]))
            + max(0.0, float(item["scroll"][1]) - float(item["client"][1]))
            for item in (report.get("hardOverflow") or [])
            if isinstance(item.get("scroll"), list) and isinstance(item.get("client"), list)
        ),
        1,
    )

    geometry = {
        "status": report.get("status"),
        "hard_overflow": report.get("hardOverflow") or [],
        "outside": report.get("outside") or [],
        "missing_images": report.get("missingImages") or [],
        "empty_shells": report.get("emptyShells") or [],
        "scientific_images": scientific_images,
        "small_scientific_images": small_scientific_images,
        "flex_margin_spacers": report.get("flexMarginSpacers") or [],
        "title_only_sidecars": report.get("titleOnlySidecars") or [],
        "text_orientation_issues": text_orientation_issues,
        "brand_padding_intrusions": brand_padding_intrusions,
        "overflow_excess_px": overflow_excess_px,
        "territories": [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "largest_vertical_void_ratio",
                    "painted_height_ratio",
                    "large_vertical_gaps",
                    "painted_bottom_gap_ratio",
                    "bottom_gap_ratio",
                )
            }
            for item in scientific_territories
        ],
        "branding": brand,
    }

    return {
        "schema_version": 1,
        "accepted": len(issues) == 0,
        "issues": list(dict.fromkeys(issues)),
        "semantic_issues": [],
        "semantic_warnings": list(dict.fromkeys(warnings)),
        "font_stats": fonts,
        "far_correspondence": [],
        "stage_centers": [],
        "geometry": geometry,
    }


gate_score = wo_sv.gate_score
select_checkpoint = wo_sv.select_checkpoint
should_stop = wo_sv.should_stop
fit_allowed = wo_sv.fit_allowed
