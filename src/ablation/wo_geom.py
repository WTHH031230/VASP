"""Standalone w/o Whitespace/Geometry Verification (Semantic-Verification-Only) implementation.

Cleanly decouples Whitespace & Ordinary Geometric Verification from Semantic Verification (SV):
  1. Keeps 100% of Semantic Verification (SV):
     - Full 4-file design state declaration (poster.html, style.css, design_state.json, critique.json).
     - Grounded claim-evidence graph planning (stages, territories, nodes, typed claim-evidence edges,
       reading_order, primary visual anchor).
     - All semantic & evidence predicates in quality_gate and CONTROLLER STATE during revisions:
       schema & narrative stage order, reading_order monotonicity, HTML node/stage/territory binding,
       no orphan evidence, primary visual anchor (>= 3.5% canvas area), figure legibility floor
       (>= 12% W x 6.5% H), claim-evidence spatial proximity (delta <= 0.42), left-to-right stage
       progression (tau <= 80px), verbatim H1 title, and brand/QR contract.
  2. Keeps 100% of Visual-First (VF) asset admission & SemParser assets.
  3. Completely removes Whitespace & Ordinary Geometric Verification:
     - Removes territory vertical void (<= 0.075), painted bottom gap (<= 0.035), shared bottom
       baseline spread (<= 0.025), low painted coverage warnings, empty shells, flex margin spacers,
       title-only sidecars, hard overflow / pixel-budget checks, and typography floor checks from
       quality_gate, CONTROLLER STATE, and checkpoint ranking.
     - Removes the corresponding whitespace, bottom-baseline, internal-void, sidecar, pixel-budget,
       and overflow prompt blocks (matching the v4 boundary for removed geometric instructions).
     - Disables deterministic geometry padding fit (apply_small_overflow_fit).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.viscomposer import compose, gate, schemas
from src.viscomposer.arms import ARMS, ASSETS, BASIC, SEM, VF, Arm
from src.viscomposer.prompt_blocks import (
    AGENT_OPENING,
    PI_OPENING,
    PRINCIPLES,
    INITIAL,
    PROTOCOL,
    REVISION,
)
from src.viscomposer.core.host_prompts import canvas_host_contract
from src.viscomposer.core.prompts import canvas_prompt_text
from src.viscomposer.core.workbench import EVIDENCE_KINDS, ROLE_RANK

# Base arm configurations:
# - FULL_ARM is used for semantic schema, contract, and SV predicate evaluation.
# - NO_GEOM_PROMPT_ARM (without SEM, with base_declaration=True) is used with PRINCIPLES,
#   INITIAL, and REVISION because SEM owns exclusively the whitespace, bottom-baseline,
#   internal-void, sidecar, pixel-budget, and overflow prompt blocks while keeping the full
#   4-file design_state declaration and claim-evidence graph in the base.
FULL_ARM = ARMS["full"]
NO_GEOM_PROMPT_ARM = Arm("no_sem", frozenset({BASIC, ASSETS, VF}), base_declaration=True)

SOURCE_FILES = ("poster.html", "style.css", "design_state.json", "critique.json")


def response_schemas(max_revisions: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    return schemas.response_schemas(FULL_ARM, max(max_revisions, 1))


def _protocol_text(harness: str = "pi") -> str:
    text = PROTOCOL.compose(FULL_ARM)
    if harness == "codex":
        return text
    if not text.startswith(AGENT_OPENING):
        raise ValueError("composed protocol lost its opening sentence")
    return PI_OPENING + text[len(AGENT_OPENING):]


SV_INITIAL_NOTE = (
    "Focus on semantic claim-evidence organization: pair every admitted Figure, Table, Equation, "
    "or Algorithm with body prose in the same section that explicitly explains its visual content "
    "and connects it to the paper's claims. Size each section naturally for its semantic content; "
    "do not stretch cards, insert filler blocks, or force columns to share an equal bottom baseline "
    "merely to fill vertical canvas space."
)

SV_REVISION_NOTE = (
    "SEMANTIC VERIFICATION TURN (model round {round}). Inspect the rendered poster, the figure "
    "detail sheet, and the semantic controller state below. Prioritize repairing any violated "
    "semantic or claim-evidence predicates (claim-evidence adjacency delta <= 0.42, orphan evidence, "
    "left-to-right stage order, figure legibility floor, primary visual anchor, and substantive "
    "prose grounding of every displayed figure/table). Do not stretch containers or add filler "
    "spacing to eliminate bottom or vertical whitespace; leave natural column heights as they fall."
)


def initial_prompt(
    context: dict[str, Any],
    *,
    canvas_width: int,
    canvas_height: int,
    source: str = "ours",
    harness: str = "pi",
) -> str:
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(NO_GEOM_PROMPT_ARM), canvas_width, canvas_height),
        canvas_host_contract(canvas_width, canvas_height),
        _protocol_text(harness),
        INITIAL.compose(NO_GEOM_PROMPT_ARM),
        SV_INITIAL_NOTE,
        "GROUNDED PAPER PACKAGE\n"
        + json.dumps(
            compose.package(context, FULL_ARM, source=source),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ]
    return "\n\n".join(part for part in parts if part)


def controller_state_sv_only(checkpoint: int, gate_result: dict[str, Any]) -> dict[str, Any]:
    """Controller state containing ONLY Semantic Verification & VF/Asset predicates (no whitespace/geometry)."""
    geom = gate_result.get("geometry") or {}
    return {
        "checkpoint": checkpoint,
        "accepted": gate_result.get("accepted", False),
        "deterministic_issues": gate_result.get("issues", []),
        "semantic_issues": gate_result.get("semantic_issues", []),
        "semantic_warnings": gate_result.get("semantic_warnings", []),
        "anchor": gate_result.get("anchor"),
        "far_correspondence": gate_result.get("far_correspondence", []),
        "stage_centers": gate_result.get("stage_centers", []),
        "geometry": {
            "scientific_images": geom.get("scientific_images", []),
            "small_scientific_images": geom.get("small_scientific_images", []),
            "branding": geom.get("branding", {}),
        },
    }


def revision_prompt(
    *,
    checkpoint: int,
    gate_result: dict[str, Any],
    final_checkpoint: bool,
    canvas_width: int,
    canvas_height: int,
    harness: str = "pi",
) -> str:
    round_no = checkpoint + 1
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(NO_GEOM_PROMPT_ARM), canvas_width, canvas_height),
        canvas_host_contract(canvas_width, canvas_height),
        _protocol_text(harness),
        SV_REVISION_NOTE.format(round=round_no),
        REVISION.compose(NO_GEOM_PROMPT_ARM),
        "CONTROLLER STATE\n"
        + json.dumps(controller_state_sv_only(checkpoint, gate_result), ensure_ascii=False, indent=2),
    ]
    return "\n\n".join(part for part in parts if part)


def quality_gate_sv_only(
    *,
    state: dict[str, Any],
    html: str,
    report: dict[str, Any],
    paper_title: str,
    require_branding: bool | dict[str, Any],
    contract_error: str | None = None,
) -> dict[str, Any]:
    """Evaluate ONLY Semantic Verification (SV), Visual-First (VF), and Brand predicates.
    Ordinary geometric & whitespace predicates (void ratio, bottom gap, baseline spread,
    overflow, empty shells, sidecars, font floors) are excluded from issues.
    """
    issues, warnings = gate.semantic_contract_issues(state, html, paper_title, FULL_ARM)
    semantic_issues = list(issues)
    if contract_error:
        issues.insert(0, contract_error)

    node_map = {item["id"]: item for item in state.get("nodes") or []}
    node_rects = report.get("nodeRects") or {}
    canvas = report.get("canvas") or [2544, 1696]
    canvas_width = max(1.0, float(canvas[0]))
    canvas_height = max(1.0, float(canvas[1]))

    # 1. VF figure legibility floor & primary anchor & claim-evidence proximity (delta <= 0.42)
    figure_node_ids = {
        item["id"] for item in state.get("nodes") or [] if item.get("kind") == "figure"
    }
    scientific_images = [
        item for item in report.get("scientificImageMetrics") or []
        if item.get("node_id") in figure_node_ids
    ]
    small_scientific_images: list[dict[str, Any]] = []
    min_figure_width = canvas_width * 0.12
    min_figure_height = canvas_height * 0.065
    for item in scientific_images:
        effective = item.get("effective") or {}
        width = float(effective.get("w") or 0)
        height = float(effective.get("h") or 0)
        if width + 0.5 < min_figure_width or height + 0.5 < min_figure_height:
            small_scientific_images.append(item)
            issues.append(
                f"figure {item.get('node_id')} is thumbnail-sized "
                f"({width:.0f}x{height:.0f}px effective); enlarge it or omit it"
            )

    anchor = (state.get("visual_hierarchy") or {}).get("primary_anchor")
    anchor_rect = node_rects.get(anchor) if anchor else None
    if anchor and not anchor_rect:
        issues.append(f"primary anchor {anchor} has no rendered rectangle")
    elif anchor_rect:
        area_ratio = (
            float(anchor_rect["w"]) * float(anchor_rect["h"]) / (canvas_width * canvas_height)
        )
        if area_ratio < 0.035:
            issues.append(f"primary anchor {anchor} occupies only {area_ratio:.3f} of the canvas")

    far_correspondence: list[dict[str, Any]] = []
    for edge in report.get("edgeDistances") or []:
        source = node_map.get(edge.get("from"))
        target = node_map.get(edge.get("to"))
        if not source or not target:
            continue
        if not ({source["kind"], target["kind"]} & EVIDENCE_KINDS):
            continue
        distance = float(edge.get("normalized_distance") or 0)
        if distance > 0.42:
            far_correspondence.append(
                {"from": edge.get("from"), "to": edge.get("to"), "distance": round(distance, 3)}
            )
    if far_correspondence:
        msg = f"{len(far_correspondence)} claim–evidence edge(s) are not visually adjacent"
        issues.append(msg)
        semantic_issues.append(msg)

    # 2. SEM left-to-right macro narrative stage progression (tau = 80px)
    centers: list[tuple[str, float]] = []
    stage_metrics: dict[str, dict[str, Any]] = {}
    for stage in state.get("stages") or []:
        rects = [node_rects.get(node_id) for node_id in stage["node_ids"]]
        rects = [rect for rect in rects if rect]
        if not rects:
            continue
        left = min(float(rect["x"]) for rect in rects)
        top = min(float(rect["y"]) for rect in rects)
        right = max(float(rect["x"]) + float(rect["w"]) for rect in rects)
        bottom = max(float(rect["y"]) + float(rect["h"]) for rect in rects)
        stage_metrics[stage["id"]] = {
            "id": stage["id"],
            "rect": {"x": left, "y": top, "w": right - left, "h": bottom - top},
        }
    ordered_stages = sorted(
        (
            item for item in state.get("stages") or []
            if item.get("role") in ROLE_RANK and item.get("role") != "identity"
        ),
        key=lambda item: ROLE_RANK[item["role"]],
    )
    for stage in ordered_stages:
        metric = stage_metrics.get(stage["id"])
        if metric:
            rect = metric["rect"]
            centers.append((stage["role"], float(rect["x"]) + float(rect["w"]) / 2))
    for previous, current in zip(centers, centers[1:]):
        if current[1] + 80 < previous[1]:
            msg = f"macro story moves left from {previous[0]} to {current[0]}"
            issues.append(msg)
            semantic_issues.append(msg)
    if len(centers) >= 4 and centers[-1][1] - centers[0][1] < canvas_width * 0.4:
        warnings.append("semantic stages use less than 40% of canvas width progression")

    # 3. Brand & QR contract
    brand = report.get("brandMetrics") or {}
    brand_padding_intrusions = brand.get("padding_intrusions") or []
    gate._brand_issues(issues, brand, brand_padding_intrusions, require_branding)

    return {
        "schema_version": 1,
        "accepted": not issues,
        "issues": list(dict.fromkeys(issues)),
        "semantic_issues": list(dict.fromkeys(semantic_issues)),
        "semantic_warnings": list(dict.fromkeys(warnings)),
        "anchor": anchor,
        "far_correspondence": far_correspondence,
        "font_stats": {},
        "stage_centers": [{"role": role, "x": round(x, 2)} for role, x in centers],
        "geometry": {
            "status": None,
            "hard_overflow": [],
            "outside": [],
            "missing_images": [],
            "empty_shells": [],
            "scientific_images": scientific_images,
            "small_scientific_images": small_scientific_images,
            "flex_margin_spacers": [],
            "title_only_sidecars": [],
            "text_orientation_issues": [],
            "brand_padding_intrusions": brand_padding_intrusions,
            "overflow_excess_px": 0.0,
            "territories": [],
            "branding": brand,
        },
    }


def gate_score_sv_only(gate_result: dict[str, Any]) -> tuple[float, ...]:
    """Rank checkpoints strictly by Semantic & VF predicates (ignoring whitespace/geometry)."""
    geometry = gate_result.get("geometry") or {}
    return (
        1.0 if gate_result.get("accepted") else 0.0,
        -float(len(gate_result.get("semantic_issues") or [])),
        -float(len(gate_result.get("far_correspondence") or [])),
        -float(len(geometry.get("small_scientific_images") or [])),
        -float(len(gate_result.get("issues") or [])),
    )


def select_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in checkpoints if item.get("eligible")]
    if not eligible:
        return None
    return max(enumerate(eligible), key=lambda pair: (gate_score_sv_only(pair[1]["gate"]), pair[0]))[1]
