"""Shared model-effort and harness-revision budget resolution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


HARNESS_LEVEL_REVISIONS = {
    "low": 1,
    "medium": 2,
    "high": 4,
    "xhigh": 8,
    "max": 16,
}
DEFAULT_HARNESS_LEVEL = "medium"
MIN_REVISIONS = 1
MAX_REVISIONS = 16


@dataclass(frozen=True)
class HarnessBudget:
    """Resolved revision budget for one independent designer session."""

    level: str
    max_revisions: int

    @property
    def max_model_calls(self) -> int:
        """Include the initial design call in the total call ceiling."""

        return self.max_revisions + 1


def resolve_harness_budget(
    harness_level: str | None,
    max_revisions: int | None,
) -> HarnessBudget:
    """Resolve named and legacy numeric budgets without silent precedence."""

    if harness_level not in (None, "custom", *HARNESS_LEVEL_REVISIONS):
        choices = ", ".join(HARNESS_LEVEL_REVISIONS)
        raise ValueError(f"unknown harness level {harness_level!r}; choose from {choices}")
    if max_revisions is not None and not MIN_REVISIONS <= max_revisions <= MAX_REVISIONS:
        raise ValueError(
            f"--max-visual-checkpoints must be between {MIN_REVISIONS} and "
            f"{MAX_REVISIONS} ({MIN_REVISIONS + 1} to {MAX_REVISIONS + 1} total "
            "model calls including the initial design)"
        )

    if harness_level == "custom":
        if max_revisions is None:
            raise ValueError("resolved custom harness level requires an explicit revision count")
        inferred = next(
            (
                name
                for name, revisions in HARNESS_LEVEL_REVISIONS.items()
                if revisions == max_revisions
            ),
            "custom",
        )
        return HarnessBudget(level=inferred, max_revisions=max_revisions)
    if harness_level is None and max_revisions is None:
        harness_level = DEFAULT_HARNESS_LEVEL
        max_revisions = HARNESS_LEVEL_REVISIONS[harness_level]
    elif harness_level is not None and max_revisions is None:
        max_revisions = HARNESS_LEVEL_REVISIONS[harness_level]
    elif harness_level is None:
        assert max_revisions is not None
        harness_level = next(
            (
                name
                for name, revisions in HARNESS_LEVEL_REVISIONS.items()
                if revisions == max_revisions
            ),
            "custom",
        )
    else:
        expected = HARNESS_LEVEL_REVISIONS[harness_level]
        if max_revisions != expected:
            raise ValueError(
                f"--harness-level {harness_level} requires "
                f"--max-visual-checkpoints {expected}, got {max_revisions}"
            )

    assert harness_level is not None and max_revisions is not None
    return HarnessBudget(level=harness_level, max_revisions=max_revisions)


def resolve_namespace_harness_budget(args: argparse.Namespace) -> HarnessBudget:
    """Resolve a CLI or legacy programmatic Namespace and store canonical fields."""

    budget = resolve_harness_budget(
        getattr(args, "harness_level", None),
        getattr(args, "max_visual_checkpoints", None),
    )
    args.harness_level = budget.level
    args.max_visual_checkpoints = budget.max_revisions
    return budget


def resolve_namespace_model_effort(args: argparse.Namespace) -> str:
    """Prefer the explicit axis name while accepting legacy ``effort`` callers."""

    model_effort = getattr(args, "model_effort", None)
    legacy_effort = getattr(args, "legacy_effort", None)
    programmatic_effort = getattr(args, "effort", None)
    supplied = [
        str(value)
        for value in (model_effort, legacy_effort, programmatic_effort)
        if value is not None
    ]
    if len(set(supplied)) > 1:
        raise ValueError(
            "--model-effort and legacy --effort values disagree: "
            + ", ".join(supplied)
        )
    resolved = supplied[0] if supplied else "medium"
    args.model_effort = resolved
    # Preserve consumers that inspect ``args.effort`` after parsing.
    args.effort = resolved
    return resolved


def add_budget_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared named and numeric harness budget CLI."""

    parser.add_argument(
        "--harness-level",
        choices=tuple(HARNESS_LEVEL_REVISIONS),
        default=None,
        help=(
            "Harness revision budget: low=1, medium=2 (default), high=4, "
            "xhigh=8, max=16 revisions after the initial design."
        ),
    )
    parser.add_argument(
        "--max-visual-checkpoints",
        type=int,
        default=None,
        help=(
            "Legacy explicit revision ceiling after the initial design (1-16). "
            "If combined with --harness-level, the values must agree."
        ),
    )


def add_model_effort_arguments(parser: argparse.ArgumentParser) -> None:
    """Add an explicit model-effort axis plus its backwards-compatible alias."""

    parser.add_argument(
        "--model-effort",
        default=None,
        help="Designer model reasoning effort (default: medium).",
    )
    parser.add_argument(
        "--effort",
        dest="legacy_effort",
        default=None,
        help="Deprecated alias for --model-effort.",
    )


def resolve_cli_namespace(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve shared axes and report inconsistent CLI values through argparse."""

    try:
        resolve_namespace_model_effort(args)
        resolve_namespace_harness_budget(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args
