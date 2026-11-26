"""Module ownership of every sentence in the VisComposer designer prompts."""


from __future__ import annotations

import json

from src.viscomposer.arms import ASSETS, BASIC, SEM, VF, Arm
from src.viscomposer.segments import ANY, Block, listing, partition
from src.viscomposer.core import host_prompts as HP
from src.viscomposer.core.prompts import DESIGN_PRINCIPLES, canvas_prompt_text

_W, _H = 2544, 1696
_ROUND = 98  # sentinel checkpoint used to cut the final notes out of the live text


def _between(text: str, prefix: str, suffix: str) -> str:
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("live prompt no longer has the expected surrounding parts")
    return text[len(prefix): len(text) - len(suffix)]


def _live_texts() -> tuple[str, str, str, str]:
    head = "\n\n".join([
        canvas_prompt_text(DESIGN_PRINCIPLES, _W, _H),
        HP.canvas_host_contract(_W, _H),
        HP.HOST_PROTOCOL,
    ]) + "\n\n"
    context: dict = {}
    initial = _between(
        HP.initial_host_prompt(context, canvas_width=_W, canvas_height=_H),
        head,
        "\n\nGROUNDED PAPER PACKAGE\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
    )
    gate: dict = {}
    state = {"checkpoint": _ROUND, "accepted": False, "deterministic_issues": [],
             "semantic_warnings": [], "font_stats": {}, "far_correspondence": [],
             "stage_centers": [], "geometry": {}}
    tail = "\n\nCONTROLLER STATE\n" + json.dumps(state, ensure_ascii=False, indent=2)
    notes = {}
    for final in (False, True):
        body = _between(
            HP.revision_host_prompt(checkpoint=_ROUND, gate=gate, final_checkpoint=final,
                                    canvas_width=_W, canvas_height=_H),
            head, tail,
        )
        note, instructions = body.split("\n\n", 1)
        notes[final] = note
    marker = f"model round {_ROUND + 1}"
    for note in notes.values():
        if note.count(marker) != 1:
            raise ValueError("final note no longer names the model round once")
    return (initial, instructions,
            notes[False].replace(marker, "model round {round}"),
            notes[True].replace(marker, "model round {round}"))


INITIAL_TEXT, REVISION_TEXT, STRUCTURAL_NOTE, CONVERGENCE_NOTE = _live_texts()


def _intro(arm: Arm) -> str:
    steps = ["form a semantic plan"]
    if arm.has(VF):
        steps.append("place the important evidence")
    if arm.has(BASIC):
        steps.append("render, inspect the actual result, diagnose root causes, and edit "
                     "the same source incrementally")
        return ("You are the persistent lead information designer for a research poster. "
                "Work like a human designer in one continuing source workspace: "
                + ", ".join(steps)
                + ". Do not replace the design merely because a checkpoint arrives.\n\n")
    return "You are the lead information designer for a research poster: " + listing(steps) + ".\n\n"


def _derive(arm: Arm) -> str:
    return "Derive each paper's visual identity from the supplied scientific Figures. "


def _avoid(arm: Arm) -> str:
    return "Avoid dashboard-card walls, dark full-width banners, and gratuitous gradients.\n"


_DIAGNOSE = (
    ("reading order", SEM), ("visual anchor", VF), ("title fidelity", None),
    ("physical type hierarchy", None), ("unused space", None),
    ("claim/evidence adjacency", VF), ("captions", None), ("branding/QR balance", ASSETS),
)


def _diagnose(arm: Arm) -> str:
    items = [name for name, owner in _DIAGNOSE if owner is None or arm.has(owner)]
    return "Diagnose root causes before editing: " + listing(items) + ". "


def _inspect(arm: Arm) -> str:
    what = "both" if arm.has(VF) else "it"
    if arm.has(SEM):
        return f"Inspect {what} and use the controller state below as supporting evidence.\n"
    return f"Inspect {what}.\n"


PRINCIPLES = partition("DESIGN_PRINCIPLES", DESIGN_PRINCIPLES, [
    Block("You are the persistent lead information designer", (BASIC, VF), render=_intro),
    Block("The poster is a {canvas_width}×{canvas_height} landscape canvas."),
    Block(" Its macro story must progress left to right:"),
    Block("Design claim–evidence units, not separate text and image columns.", (VF,)),
    Block("Use a coherent global system rather than local patches:"),
    Block("Derive each paper's", (ASSETS,), render=_derive),
    Block("Asset-catalog palette_profile.accent_hexes"),
    Block("Do not copy every sampled color"),
    Block("Preserve\none poster's chosen palette"),
    Block("Aim for physical\nposter readability"),
    Block("If four story territories force small type"),
    Block("The paper title is immutable identity copy."),
    Block("Treat scientific typography as part of that system."),
    Block("Treat both axes as explicit budgets.", (SEM,)),
    Block("A very wide Figure should not be placed", (VF,)),
    Block("The asset catalog separates primary body Figures", (VF,)),
    Block("Every admitted scientific Figure must perform", (VF,)),
    Block("Use whitespace to separate ideas", (SEM,)),
    Block("Prefer enlarging\nuseful evidence", (VF,)),
    Block("Avoid dashboard-card walls,", (VF,), render=_avoid),
    Block("Treat the bottom of the scientific story", (SEM,)),
    Block("Use larger evidence,\ncontent reflow", (VF,)),
    Block("No internal vertical void", (SEM,)),
    Block("Size important non-QR logos", (ASSETS,)),
    Block("Use only the supplied local paper and brand assets."),
    Block("Brand catalog entries explicitly", (ASSETS,)),
    Block("All informative\ntext must follow", (BASIC,)),
    Block("Place Paper, Code, and\nProject labels", (ASSETS,)),
    Block("Tables, equations, and algorithms may be\nfaithfully", (ASSETS,)),
    Block("Do not invent\nscientific facts"),
])

INITIAL = partition("initial instructions", INITIAL_TEXT, [
    Block("Follow GROUNDED PAPER PACKAGE.visual_input_order exactly.", (VF,)),
    Block("Inspect\nall supplied sheets", (VF, ASSETS), ANY,
          render=lambda arm: "Inspect the captions and paper text below. "),
    Block("Create a\nfinished first draft, not a wireframe."),
    Block("First decide the grounded\nclaim–evidence graph", (VF,),
          render=lambda arm: ("First decide the semantic stages. Then jointly choose text amount, "
                              "territories, typography, and geometry. ")),
    Block("Use stable generic IDs."),
    Block("Use only permitted relative Figure paths in poster_html."),
    Block("For every Figure you admit,", (VF,)),
    Block("Appendix Figures remain secondary:", (VF,), render=lambda arm: ""),
    Block("Any scientific Figure that cannot be funded", (VF,)),
    Block("Do not place a short heading alone", (SEM,)),
    Block("When brand_requirements has nonzero minima,", (ASSETS,)),
    Block("Before returning, do static layout arithmetic", (SEM,)),
    Block("design_state.paper_title and design_state.poster_title"),
    Block("The single visible H1 must contain"),
    Block("Return only the required structured object."),
    Block("critique.checkpoint must be 0"),
])

REVISION = partition("revision instructions", REVISION_TEXT, [
    Block("The first attached image is the current render from this same session."),
    Block("The second, when\npresent, is a labelled detail sheet", (VF,)),
    Block("Inspect both and use the controller state", (VF, SEM), render=_inspect),
    Block("Diagnose root causes before\nediting:", (VF, SEM, ASSETS), render=_diagnose),
    Block("Narrative correctness\noutranks", (SEM,)),
    Block("Before proposing edits, make a pixel budget", (SEM,)),
    Block("Never rewrite or abbreviate the paper title"),
    Block("Do not trade one overflow", (SEM,)),
    Block("Treat hard overflow as an absolute blocker.", (SEM,)),
    Block("Do not replace a bottom void", (SEM,)),
    Block("Treat every reported title-only sidecar", (SEM,)),
    Block("Do not use a blanket selector", (SEM,)),
    Block("Inspect table columns across every body row"),
    Block("Return at most 16 exact, minimal"),
    Block("Preserve grounded content and stable node IDs."),
    Block("Return the full\nupdated design_state"),
    Block("The host will reject ambiguous"),
])


PROTOCOL = partition("HOST_PROTOCOL", HP.HOST_PROTOCOL, [
    Block("This environment cannot expose filesystem tools safely."),
    Block("Act as the persistent designer through", (BASIC,),
          render=lambda arm: "Respond through the structured response contract. "),
    Block("On the first turn, return the complete four-file"),
    Block("On visual turns, return exact old_text", (BASIC,)),
    Block("The host applies\nonly those exact edits", (BASIC,)),
    Block("HTML must contain one '.poster' root"),
    Block("Every design_state node appears exactly once as data-node-id;"),
    Block("The brand cluster, not the whole masthead,", (ASSETS,)),
    Block("The HTML DOM order and design_state.reading_order must agree."),
    Block("Branding is an identity\nnode"),
])


def final_note(arm: Arm, checkpoint: int, final_checkpoint: bool) -> str:
    """STRUCTURAL/CONVERGENCE round roles (SEM); visual-first clauses (VF)."""
    round_no = checkpoint + 1
    if arm.has(SEM, VF):
        template = CONVERGENCE_NOTE if final_checkpoint else STRUCTURAL_NOTE
        return template.format(round=round_no)
    if arm.has(SEM):
        if final_checkpoint:
            return CONVERGENCE_NOTE.format(round=round_no)
        return (f"STRUCTURAL TURN (model round {round_no}). Resolve text quantity and "
                "territory allocation together.")
    if arm.has(VF) and not final_checkpoint:
        return ("Resolve evidence admission, figure scale, and claim-evidence adjacency "
                "together. Every thumbnail-sized Figure must be enlarged to a useful size "
                "or removed in this turn.")
    return ""


TEMPLATE_INITIAL = (
    "LAYOUT TEMPLATE (fixed). Below a full-width title band, use exactly four content columns\n"
    "of equal width separated by equal gutters. First write the poster's sections in reading\n"
    "order and assign them to the columns from left to right, stacking sections top to bottom\n"
    "within each column. Place every Figure, Table, Equation, or Algorithm inside the section\n"
    "that discusses it and scale it to fit that column's content width. Content never spans\n"
    "columns and column widths never change."
)
TEMPLATE_REVISION = (
    "Keep the fixed layout template: four equal-width content columns below the title band,\n"
    "sections stacked within columns, and every Figure scaled to its own column's width.\n"
    "Do not change column widths or let content span columns."
)


AGENT_OPENING = (
    "This environment cannot expose filesystem tools safely. Do not call tools, inspect the\n"
    "repository, run commands, or attempt file writes. "
)
CODEX_OPENING = AGENT_OPENING  # backward-compatible alias
PI_OPENING = (
    "This harness gives you no filesystem, shell, or web tools. Deliver each turn only by\n"
    "calling the one submit tool offered for that turn; its arguments are the structured response. "
)
if not HP.HOST_PROTOCOL.startswith(AGENT_OPENING):
    raise ValueError("HOST_PROTOCOL no longer opens with the agent sentence")


def protocol(arm: Arm, harness: str) -> str:
    """HOST_PROTOCOL for the arm; swaps the opening sentence for pi vs generic agent CLIs."""
    text = PROTOCOL.compose(arm)
    if harness in ("agent", "codex"):
        return text
    if harness != "pi":
        raise ValueError(f"unknown harness: {harness}")
    if not text.startswith(AGENT_OPENING):
        raise ValueError("composed protocol lost its opening sentence")
    return PI_OPENING + text[len(AGENT_OPENING):]

