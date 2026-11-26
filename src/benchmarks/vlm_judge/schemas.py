"""JSON Schemas passed to ``codex exec --output-schema``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _aspect_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["score", "reason"],
    }


def single_aspect_schema() -> dict[str, Any]:
    """Schema for one independent Paper2Poster judge dimension."""

    return _aspect_schema()


def combined_schema(aspects: Iterable[str]) -> dict[str, Any]:
    """Schema for a combined call over the requested aspect names."""

    selected = tuple(aspects)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {aspect: _aspect_schema() for aspect in selected},
        "required": list(selected),
    }
