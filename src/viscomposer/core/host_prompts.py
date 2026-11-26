"""Safe structured-edit prompts when the host cannot provide a nested sandbox."""

from __future__ import annotations

import json
from typing import Any

from src.viscomposer.core.prompts import DESIGN_PRINCIPLES, canvas_prompt_text


HOST_PROTOCOL = """
This environment cannot expose filesystem tools safely. Do not call tools, inspect the
repository, run commands, or attempt file writes. Act as the persistent designer through
the structured response contract. On the first turn, return the complete four-file
design state. On visual turns, return exact old_text → new_text edits for poster.html
and style.css plus the complete small design_state and critique objects. The host applies
only those exact edits inside the isolated result directory.

HTML must contain one '.poster' root, link style.css, and use only permitted local
'assets/...' paths. It must not contain scripts, event handlers, iframes, objects,
embeds, or external URLs. Every design_state node appears exactly once as data-node-id;
every stage appears exactly once as data-stage-id; every territory appears as
data-territory-id. The brand cluster, not the whole masthead, is data-node-id="branding".
The HTML DOM order and design_state.reading_order must agree. Branding is an identity
node and may have its own visual territory, but identity is never a scientific story
stage and must not appear in design_state.stages.
""".strip()


def canvas_host_contract(canvas_width: int, canvas_height: int) -> str:
    """State the exact editable CSS canvas required by the host renderer."""

    if canvas_width < 1 or canvas_height < 1:
        raise ValueError("canvas dimensions must be positive")
    return (
        "CANVAS CONTRACT\n"
        f"style_css must define the .poster canvas at exactly "
        f"{int(canvas_width)}px × {int(canvas_height)}px. "
        "All fixed layout arithmetic must reconcile to this box."
    )


def initial_host_prompt(
    context: dict[str, Any],
    *,
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> str:
    """Request the initial files through a strict structured response."""

    return "\n\n".join(
        [
            canvas_prompt_text(DESIGN_PRINCIPLES, canvas_width, canvas_height),
            canvas_host_contract(canvas_width, canvas_height),
            HOST_PROTOCOL,
            """
Follow GROUNDED PAPER PACKAGE.visual_input_order exactly. The primary body-Figure sheet
comes first. Zero or more separately paginated appendix-Figure sheets follow as a
secondary evidence pool, and the brand sheet comes last when brand assets exist. Inspect
all supplied sheets together with the captions and paper text below. Create a
finished first draft, not a wireframe. First decide the grounded
claim–evidence graph and semantic stages. Then jointly choose text amount, evidence size,
territories, typography, and geometry. Use stable generic IDs. Use only permitted relative Figure paths in poster_html.
For every Figure you admit, identify the unique point it makes and fund enough physical
area for that point to be read. Appendix Figures remain secondary: admit one only when no
body Figure serves the same narrative job or when it supplies uniquely decisive evidence,
and record that specific reason in design_state.decision_log.
Any scientific Figure that cannot be funded at a legible physical size must be omitted
rather than used as a thumbnail.
Do not place a short heading alone in a narrow grid column beside a tall Figure, Table,
Equation, or Algorithm. Unless that side column also contains substantive explanatory
content, make the heading span the full content width above its evidence. A decorative
side label does not justify a locally empty column.
When brand_requirements has nonzero minima, use actual asset_catalog entries whose
brand_kind is logo or qr inside data-node-id="branding". For every brand image, copy the
catalog label verbatim into data-brand-label and copy brand_kind into data-brand-kind.
Wrap each QR and its visible label in an element marked data-brand-item. The visible QR
label must be exactly its catalog label (Paper, Code, or Project), with no reinterpretation
such as "Project + code". Its alt text must also contain that exact label. Treat
the label as ordinary horizontal reading text and normally place it centered below the
QR; never rotate it or use vertical writing to save width.
Every Logo, QR, and QR row must remain within the brand cluster's inner content box after
padding and borders are subtracted. Include the QR row's min-content width in fixed-grid
arithmetic; touching the card's outer edge while consuming its padding is a failure. Treat
brand_policy exclusions as unavailable; never recreate an excluded mark as text. Never
replace a supplied mark with styled text.

Before returning, do static layout arithmetic at every fixed grid level and leave a
small overflow margin. design_state.paper_title and design_state.poster_title must both
copy GROUNDED PAPER PACKAGE.metadata.title verbatim. The single visible H1 must contain
that same complete title; markup-only spans and line breaks are allowed, but textual
rewriting is not. A separate subtitle may add context but must not replace or continue
the title.

Return only the required structured object. poster_html and style_css become the exact
editable source for later checkpoints. critique.checkpoint must be 0 and should state the
assumptions and visual risks that need verification after rendering.
""".strip(),
            "GROUNDED PAPER PACKAGE\n"
            + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        ]
    )


def revision_host_prompt(
    *,
    checkpoint: int,
    gate: dict[str, Any],
    final_checkpoint: bool,
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> str:
    """Request exact incremental edits after a same-session visual inspection."""

    final_note = (
        f"CONVERGENCE TURN. This is model round {checkpoint + 1} and the final "
        "permitted visual checkpoint. Work from the "
        "current evidence selection and visual structure. Do not add scientific Figures "
        "or start a redesign; verify legibility and make funded local repairs, including "
        "removing any detected internal spacer between related prose and evidence."
        if final_checkpoint
        else f"STRUCTURAL TURN (model round {checkpoint + 1}). Resolve evidence "
        "admission, figure scale, text quantity, "
        "claim-evidence adjacency, and territory allocation together. Every thumbnail-"
        "sized Figure must be enlarged to a useful size or removed in this turn."
    )
    state = {
        "checkpoint": checkpoint,
        "accepted": gate.get("accepted", False),
        "deterministic_issues": gate.get("issues", []),
        "semantic_warnings": gate.get("semantic_warnings", []),
        "font_stats": gate.get("font_stats", {}),
        "far_correspondence": gate.get("far_correspondence", []),
        "stage_centers": gate.get("stage_centers", []),
        "geometry": gate.get("geometry", {}),
    }
    return "\n\n".join(
        [
            canvas_prompt_text(DESIGN_PRINCIPLES, canvas_width, canvas_height),
            canvas_host_contract(canvas_width, canvas_height),
            HOST_PROTOCOL,
            final_note,
            """
The first attached image is the current render from this same session. The second, when
present, is a labelled detail sheet of scientific Figure nodes at their measured
effective sizes. Inspect both and use the controller state below as supporting evidence.
Diagnose root causes before
editing: reading order, visual anchor, title fidelity, physical type hierarchy, unused
space, claim/evidence adjacency, captions, and branding/QR balance. Narrative correctness
outranks a numeric density target.

Before proposing edits, make a pixel budget from the reported client and scroll sizes.
Every expansion must name the space that funds it; masthead growth reduces story height.
Use safety margins rather than setting a CSS value exactly at a reported threshold.
Never rewrite or abbreviate the paper title to solve fit. Repair title overflow through
masthead width, masthead height, natural wrapping, and funded local geometry while
preserving the exact metadata title and the 64 px title floor. Do not trade one overflow
for another or remove an image shell in a way that creates a larger territorial void. If
content cannot fit at the required type floor, remove the lowest-priority supporting
item.

Treat hard overflow as an absolute blocker. A revision is not an improvement if either
the hard-overflow count or aggregate excess grows, even when it fixes whitespace or
typography. Preserve a currently full, aligned story bottom while repairing local
overflow; do not enlarge content merely to satisfy a density preference that already
passes.

Do not replace a bottom void with similarly large internal gaps or `space-between`
distribution; resize or reflow meaningful content, or change the territory allocation.
Treat a reported flex-margin spacer as a structural defect: remove the spring and keep
the affected Figure adjacent to the prose it explains while preserving the shared
bottom baseline.
Treat every reported title-only sidecar as a structural defect. Make the heading span
above its evidence, or fill the side column with substantive explanation; never leave a
short heading alone beside a tall Table, Figure, Equation, or Algorithm. Do not create a
new title-only sidecar while repairing another issue.
Do not use a blanket selector to raise all story text. Typography changes must target
the measured low leaves and must be funded for the additional wrapping they create.
Inspect table columns across every body row, especially after rowspans, and correct any
role whose alignment changes between rows. Verify that continuous body prose uses the
shared automatic-hyphenation/justification rule without applying it to short display
copy, captions, labels, metrics, or lists.

Return at most 16 exact, minimal old_text → new_text operations against the poster_html
and style_css you produced previously. Each old_text must be a unique verbatim substring;
prioritize integrated root-cause edits over many selector-level patches. Do not return
whole-file replacements. Preserve grounded content and stable node IDs. Return the full
updated design_state because it is small, and a critique for this checkpoint describing
root causes, integrated decisions, and remaining risks. The host will reject ambiguous
or stale edits instead of guessing.
""".strip(),
            "CONTROLLER STATE\n" + json.dumps(state, ensure_ascii=False, indent=2),
        ]
    )


def _gate_metric_snapshot(gate: dict[str, Any]) -> dict[str, float | int]:
    """Return compact lower-is-better metrics for revision comparison."""

    geometry = gate.get("geometry") or {}
    fonts = gate.get("font_stats") or {}
    territories = geometry.get("territories") or []
    return {
        "semantic_issue_count": len(gate.get("semantic_issues") or []),
        "outside_or_missing_count": len(geometry.get("outside") or [])
        + len(geometry.get("missing_images") or []),
        "hard_overflow_count": len(geometry.get("hard_overflow") or []),
        "overflow_excess_px": float(geometry.get("overflow_excess_px") or 0),
        "empty_shell_count": len(geometry.get("empty_shells") or []),
        "small_scientific_image_count": len(
            geometry.get("small_scientific_images") or []
        ),
        "below_17px_text_count": int(fonts.get("below_17_count") or 0),
        "max_vertical_void_ratio": max(
            (float(item.get("largest_vertical_void_ratio") or 0) for item in territories),
            default=0.0,
        ),
        "max_painted_bottom_gap_ratio": max(
            (float(item.get("painted_bottom_gap_ratio") or 0) for item in territories),
            default=0.0,
        ),
    }


def regression_revision_host_prompt(
    *,
    checkpoint: int,
    gate: dict[str, Any],
    prior_gate: dict[str, Any],
    final_checkpoint: bool,
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> str:
    """Add a prior-render regression guard without changing the stable prompt."""

    prompt = revision_host_prompt(
        checkpoint=checkpoint,
        gate=gate,
        final_checkpoint=final_checkpoint,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    prior = _gate_metric_snapshot(prior_gate)
    current = _gate_metric_snapshot(gate)
    audit = {
        "interpretation": "All metrics are lower-is-better.",
        "prior": prior,
        "current": current,
        "worsened": {
            key: {"prior": prior[key], "current": current[key]}
            for key in prior
            if float(current[key]) > float(prior[key]) + 1e-6
        },
        "improved": {
            key: {"prior": prior[key], "current": current[key]}
            for key in prior
            if float(current[key]) < float(prior[key]) - 1e-6
        },
    }
    guard = """
REGRESSION GUARD. The third attached image is the immediately preceding render from this
same session, before the current edits. It is diagnostic evidence, not an aesthetic
template. Compare it with the current render and the regression audit below. Keep the
current checkpoint's genuine gains, but restore any locally superior allocation from the
prior render when a measured metric or visibly coherent region regressed. Edit the
current source only. Do not initiate a redesign merely because the reference differs.

REGRESSION AUDIT
""".strip() + "\n" + json.dumps(audit, ensure_ascii=False, indent=2)
    marker = "CONTROLLER STATE\n"
    if marker not in prompt:
        raise ValueError("stable revision prompt lacks controller-state marker")
    return prompt.replace(marker, guard + "\n\n" + marker, 1)
