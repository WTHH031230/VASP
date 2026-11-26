"""Model-facing inputs per arm: prompts, package keys, images, controller state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.viscomposer.arms import ASSETS, BASIC, SEM, VF, Arm
from src.viscomposer.prompt_blocks import (
    INITIAL,
    PRINCIPLES,
    REVISION,
    TEMPLATE_INITIAL,
    TEMPLATE_REVISION,
    final_note,
    protocol,
)
from src.viscomposer.text_first import initial_instructions, revision_instructions
from src.viscomposer.core.host_prompts import canvas_host_contract
from src.viscomposer.core.prompts import canvas_prompt_text

BASE = "base"

PACKAGE_OWNERS = {
    "metadata": BASE,
    "asset_catalog": BASE,
    "appendix_policy": VF,
    "brand_requirements": ASSETS,
    "brand_policy": ASSETS,
    "structured_evidence": ASSETS,
    "numbered_paper_lines": ASSETS,
    "paper_text": BASE,
    "visual_input_order": VF,
}

IMAGE_ROLE_OWNERS = {
    "primary_body_figures": VF,
    "secondary_appendix_figures": VF,
    "brand_assets": ASSETS,
}

GEOMETRY_OWNERS = {
    "status": SEM, "hard_overflow": SEM, "outside": SEM, "missing_images": SEM,
    "empty_shells": SEM, "scientific_images": VF, "small_scientific_images": VF,
    "flex_margin_spacers": SEM, "title_only_sidecars": SEM,
    "text_orientation_issues": BASIC, "brand_padding_intrusions": ASSETS,
    "overflow_excess_px": SEM, "territories": SEM, "branding": ASSETS,
}


def _owned(arm: Arm, owner: str) -> bool:
    return owner == BASE or arm.has(owner)


def package(context: dict[str, Any], arm: Arm, *, source: str = "ours") -> dict[str, Any]:
    """GROUNDED PAPER PACKAGE for the arm. The full arm returns the context unchanged."""
    if not arm.has(ASSETS) and source == "ours":
        raise ValueError("an arm without assets needs the ResearchStudio package (source='researchstudio')")
    if arm.is_full:
        return context
    unknown = set(context) - set(PACKAGE_OWNERS)
    if unknown:
        raise ValueError(f"package keys without an owner: {sorted(unknown)}")
    return {key: value for key, value in context.items() if _owned(arm, PACKAGE_OWNERS[key])}


def initial_images(context: dict[str, Any], arm: Arm) -> list[Path]:
    """Contact sheets attached to the first turn, in visual_input_order."""
    images = []
    for item in context.get("visual_input_order") or []:
        owner = IMAGE_ROLE_OWNERS.get(item.get("role"))
        if owner is None:
            raise ValueError(f"visual input role without an owner: {item.get('role')!r}")
        if arm.has(owner):
            images.append(Path(item["path"]))
    return images


def revision_images(arm: Arm, render: Path, detail_sheet: Path | None) -> list[Path]:
    images = [render] if render.is_file() else []
    if arm.has(VF) and detail_sheet is not None:
        images.append(detail_sheet)
    return images


def initial_prompt(context: dict[str, Any], arm: Arm, *, canvas_width: int,
                   canvas_height: int, source: str = "ours", harness: str = "pi") -> str:
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(arm), canvas_width, canvas_height),
        canvas_host_contract(canvas_width, canvas_height),
        protocol(arm, harness),
        INITIAL.compose(arm),
        TEMPLATE_INITIAL if arm.replacement == "template" else "",
        initial_instructions(arm),
        "GROUNDED PAPER PACKAGE\n" + json.dumps(
            package(context, arm, source=source),
            ensure_ascii=False, separators=(",", ":")),
    ]
    return "\n\n".join(part for part in parts if part)


def controller_state(arm: Arm, checkpoint: int, gate: dict[str, Any]) -> dict[str, Any]:
    """CONTROLLER STATE with only the plugged-in modules' keys, in the original order."""
    state: dict[str, Any] = {
        "checkpoint": checkpoint,
        "accepted": gate.get("accepted", False),
        "deterministic_issues": gate.get("issues", []),
    }
    if arm.has(SEM):
        state["semantic_warnings"] = gate.get("semantic_warnings", [])
        state["font_stats"] = gate.get("font_stats", {})
    if arm.has(VF):
        state["far_correspondence"] = gate.get("far_correspondence", [])
    if arm.has(SEM):
        state["stage_centers"] = gate.get("stage_centers", [])
    geometry = gate.get("geometry", {})
    if not arm.is_full:
        unknown = set(geometry) - set(GEOMETRY_OWNERS)
        if unknown:
            raise ValueError(f"geometry keys without an owner: {sorted(unknown)}")
        geometry = {k: v for k, v in geometry.items() if arm.has(GEOMETRY_OWNERS[k])}
    state["geometry"] = geometry
    return state


def revision_prompt(arm: Arm, *, checkpoint: int, gate: dict[str, Any], final_checkpoint: bool,
                    canvas_width: int, canvas_height: int, harness: str = "pi") -> str:
    if not arm.has(BASIC):
        raise ValueError("an arm without the basic harness has no revision turns")
    parts = [
        canvas_prompt_text(PRINCIPLES.compose(arm), canvas_width, canvas_height),
        canvas_host_contract(canvas_width, canvas_height),
        protocol(arm, harness),
        final_note(arm, checkpoint, final_checkpoint),
        REVISION.compose(arm),
        TEMPLATE_REVISION if arm.replacement == "template" else "",
        revision_instructions(arm),
        "" if not arm.has(SEM) else
        "CONTROLLER STATE\n" + json.dumps(controller_state(arm, checkpoint, gate),
                                          ensure_ascii=False, indent=2),
    ]
    return "\n\n".join(part for part in parts if part)
