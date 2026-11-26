"""Loop control per arm: checkpoint ranking, early stop, deterministic-fit precondition.

gate_score ports harness._gate_score; each of its 16 keys is kept only when the module
that produces it is plugged in (spec §4.3). Ties go to the later checkpoint, as in the
harness (max over (score, index)).
"""

from __future__ import annotations

from typing import Any

from engine.arms import SEM, VF, Arm

GENERIC = "generic"
STATE = "state"
KEY_OWNERS = (
    GENERIC,  # 1 accepted
    STATE,    # 2 semantic issues (VF and SEM plan checks)
    SEM,      # 3 outside or missing images
    SEM,      # 4 overflow excess px
    SEM,      # 5 hard overflow count
    SEM,      # 6 empty shells
    VF,       # 7 small scientific images
    SEM,      # 8 flex margin spacers
    SEM,      # 9 title-only sidecars
    SEM,      # 10 max painted bottom gap
    SEM,      # 11 bottom gap spread
    SEM,      # 12 max vertical void
    SEM,      # 13 title px penalty
    SEM,      # 14 median px penalty
    SEM,      # 15 p10 px penalty
    GENERIC,  # 16 issue count
)


def _full_score(gate: dict[str, Any]) -> tuple[float, ...]:
    geometry = gate.get("geometry") or {}
    fonts = gate.get("font_stats") or {}
    territories = geometry.get("territories") or []
    max_void = max(
        (float(item.get("largest_vertical_void_ratio") or 0) for item in territories), default=0,
    )
    outside_or_missing = len(geometry.get("outside") or []) + len(geometry.get("missing_images") or [])
    bottom_gaps = [max(0.0, float(item.get("painted_bottom_gap_ratio") or 0)) for item in territories]
    dom_bottom_gaps = [max(0.0, float(item.get("bottom_gap_ratio") or 0)) for item in territories]
    bottom_gap_spread = (
        max(dom_bottom_gaps) - min(dom_bottom_gaps) if len(dom_bottom_gaps) >= 2 else 0.0
    )
    return (
        1.0 if gate.get("accepted") else 0.0,
        -float(len(gate.get("semantic_issues") or [])),
        -float(outside_or_missing),
        -float(geometry.get("overflow_excess_px") or 0),
        -float(len(geometry.get("hard_overflow") or [])),
        -float(len(geometry.get("empty_shells") or [])),
        -float(len(geometry.get("small_scientific_images") or [])),
        -float(len(geometry.get("flex_margin_spacers") or [])),
        -float(len(geometry.get("title_only_sidecars") or [])),
        -max(bottom_gaps, default=0.0),
        -bottom_gap_spread,
        -max_void,
        -max(0.0, 64.0 - float(fonts.get("title_px") or 0)),
        -max(0.0, 19.0 - float(fonts.get("median_px") or 0)),
        -max(0.0, 16.5 - float(fonts.get("p10_px") or 0)),
        -float(len(gate.get("issues") or [])),
    )


def _key_active(arm: Arm, owner: str) -> bool:
    if owner == GENERIC:
        return True
    if owner == STATE:
        return arm.declares_state
    return arm.has(owner)


def gate_score(gate: dict[str, Any], arm: Arm) -> tuple[float, ...]:
    score = _full_score(gate)
    return tuple(value for value, owner in zip(score, KEY_OWNERS) if _key_active(arm, owner))


def select(checkpoints: list[dict[str, Any]], arm: Arm) -> dict[str, Any] | None:
    """Best eligible checkpoint; each item needs 'gate' and an 'eligible' flag.

    Ranking by predicates belongs to SEM, so without SEM the last eligible checkpoint is returned.
    """
    eligible = [item for item in checkpoints if item.get("eligible")]
    if not eligible:
        return None
    if not arm.has(SEM):
        return eligible[-1]
    return max(enumerate(eligible), key=lambda pair: (gate_score(pair[1]["gate"], arm), pair[0]))[1]


def should_stop(gate: dict[str, Any], arm: Arm) -> bool:
    """Early stop on a passing gate belongs to semantic feedback (spec §4.4)."""
    return arm.has(SEM) and bool(gate.get("accepted"))


def fit_allowed(gate: dict[str, Any], arm: Arm) -> bool:
    """Deterministic fit belongs to semantic feedback; same precondition as the harness."""
    geometry = gate.get("geometry") or {}
    return (
        arm.has(SEM)
        and not gate.get("semantic_issues")
        and not geometry.get("outside")
        and not geometry.get("missing_images")
        and not geometry.get("empty_shells")
    )
