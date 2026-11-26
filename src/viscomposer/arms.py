"""The four removable modules and the ablation arms built from them.

Every arm keeps the base (task, canvas, first-turn files, HTML safety, export) and
drops exactly one module. The full arm plugs every module in and must reproduce the
existing system (src/viscomposer/core) byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass

BASIC = "basic"      # revision loop: render feedback, persistent session, exact edits
ASSETS = "assets"    # semantics-directed asset package (ours; ResearchStudio when off)
VF = "vf"            # visual-first composition
SEM = "sem"          # declared design state and the feedback that reads it
TF = "tf"            # text-first composition (flexible layout + prose priority in place of VF)
MODULES = (BASIC, ASSETS, VF, SEM)
ALL_MODULES = (*MODULES, TF)


@dataclass(frozen=True)
class Arm:
    name: str
    active: frozenset[str]
    replacement: str | None = None   # a prior system's practice in place of the removed module
    # The design-state declaration stays in the base; SEM controls checks and feedback.
    base_declaration: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.active) - set(ALL_MODULES)
        if unknown:
            raise ValueError(f"unknown modules: {sorted(unknown)}")

    def has(self, *modules: str) -> bool:
        """True when every named module is plugged in."""
        return all(item in self.active for item in modules)

    def any(self, *modules: str) -> bool:
        """True when at least one named module is plugged in."""
        return any(item in self.active for item in modules)

    @property
    def is_full(self) -> bool:
        return self.active == frozenset(MODULES) and self.replacement is None

    @property
    def declares_state(self) -> bool:
        """design_state exists when either module that reads it is plugged in."""
        return self.any(VF, SEM) or self.base_declaration

    @property
    def declares_plan(self) -> bool:
        """The SEM part of the declaration (stages, territories, roles, critique) is present."""
        return self.has(SEM) or self.base_declaration


def _without(module: str | None) -> frozenset[str]:
    return frozenset(item for item in MODULES if item != module)


ARMS: dict[str, Arm] = {
    "full": Arm("full", _without(None)),
    "no_assets": Arm("no_assets", _without(ASSETS)),
    "no_vf": Arm("no_vf", _without(VF)),
    "no_sem": Arm("no_sem", _without(SEM)),
    "no_basic": Arm("no_basic", _without(BASIC)),
    # visual-first replaced by the layout-first template
    "no_vf_template": Arm("no_vf_template", _without(VF), replacement="template"),
    # visual-first replaced by text-first composition with flexible layout
    "no_vf_text_first": Arm("no_vf_text_first", _without(VF) | {TF}, replacement="text_first"),
    "text_first": Arm("text_first", _without(VF) | {TF}, replacement="text_first"),
    # additive chain: the loop alone, then with our assets
    "basic_only": Arm("basic_only", frozenset({BASIC})),
    "basic_assets": Arm("basic_assets", frozenset({BASIC, ASSETS})),
}
FULL = ARMS["full"]


def arm_from_modules(modules: set[str] | frozenset[str], name: str | None = None) -> Arm:
    """Build an arbitrary module combination (used by tests and offline analysis)."""
    active = frozenset(modules)
    return Arm(name or "+".join(item for item in MODULES if item in active) or "base", active)
