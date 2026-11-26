"""Text-First (Flexible Layout) composition module for the VF ablation (Reviewer qPYK).

Decouples visual prioritization from layout flexibility:
  - Removes Visual-First (VF) figure contact sheets, evidence detail sheets, primary
    anchor requirements, and figure minimum-size gates.
  - Preserves full layout flexibility (2-5 territories with flexible width_weight
    proportions and multi-column / column-spanning CSS Grid or Flex layouts), avoiding
    any rigid equal-width 4-column template.
  - Instructs a strictly prose-prioritized design and revision workflow: write
    comprehensive grounded section prose first, place figures only as compact inline
    supporting illustrations in remaining space, resolve vertical voids by expanding
    grounded text rather than enlarging figures, and resolve overflow by compacting
    supporting figures before cutting prose.
"""

from __future__ import annotations

from src.viscomposer.arms import TF, Arm

TEXT_FIRST_INITIAL = (
    "TEXT-FIRST COMPOSITION (flexible layout, prose priority). Build the poster around a\n"
    "comprehensive, self-contained written scientific narrative while retaining full layout\n"
    "flexibility across the canvas:\n"
    "1. Flexible multi-column and spanning layout: do not lock the poster into a rigid\n"
    "   equal-width four-column template. Freely choose 2 to 5 visual territories\n"
    "   (design_state.territories) with content-driven width_weight proportions, and use\n"
    "   flexible CSS Grid or Flex layouts where sections, tables, equations, or wide blocks\n"
    "   may span multiple columns or sub-tracks as needed.\n"
    "2. Prose-first narrative budgeting: first plan and write thorough, self-contained\n"
    "   section prose in reading order (problem/framing → overall method → mechanisms/details\n"
    "   → results/takeaway), presenting the background, formal method and equations,\n"
    "   experimental setup, and quantitative/qualitative findings in detail (aim for roughly\n"
    "   650–850 words of substantive body prose across the poster while keeping the rendered\n"
    "   body median at least 19 px and informative text at least 17 px). Allocate territory\n"
    "   widths and vertical budgets primarily to house this complete written exposition.\n"
    "3. Figures as compact inline accompaniment: treat scientific Figures strictly as\n"
    "   secondary inline supporting illustrations placed inside the section that discusses\n"
    "   them. Size each admitted Figure conservatively to fit the remaining space after\n"
    "   funding the section's prose (keep individual Figures compact—typically around\n"
    "   220–320 px in height or under 5–6% of the canvas area—rather than creating a large\n"
    "   hero centerpiece). Never compress, truncate, or omit grounded prose in order to\n"
    "   enlarge a Figure."
)

TEXT_FIRST_REVISION = (
    "Maintain the text-first (prose-prioritized) composition and flexible layout across\n"
    "revisions: keep content-driven territory proportions and multi-column spanning freedom,\n"
    "and keep written scientific exposition primary over figure size.\n"
    "- When any territory reports a vertical void (largest_vertical_void_ratio > 0.075), low\n"
    "  painted coverage, or a bottom baseline gap (painted_bottom_gap_ratio > 0.035 or\n"
    "  bottom_gap_spread > 0.025), resolve it by expanding grounded section prose with\n"
    "  additional method/result details from the paper, refining text reflow, or rebalancing\n"
    "  territory width_weight—never enlarge scientific Figures or add Figures to fill\n"
    "  whitespace.\n"
    "- When hard overflow (hard_overflow) occurs, shrink, crop, or compact the supporting\n"
    "  Figures and container spacing first so that the written scientific prose remains\n"
    "  complete and legible at the required type floor."
)


def is_text_first(arm: Arm) -> bool:
    """True when the arm uses the Text-First (Flexible Layout) module."""
    return arm.has(TF) or arm.replacement == "text_first"


def initial_instructions(arm: Arm) -> str:
    return TEXT_FIRST_INITIAL if is_text_first(arm) else ""


def revision_instructions(arm: Arm) -> str:
    return TEXT_FIRST_REVISION if is_text_first(arm) else ""
