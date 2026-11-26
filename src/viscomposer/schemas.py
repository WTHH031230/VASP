"""Structured-response schemas per arm.

Starts from the harness's own effective schemas (host_edits.effective_response_schemas)
and removes the fields of unplugged modules (spec §4.1). The full arm is the harness
output unchanged.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from src.viscomposer.arms import Arm
from src.viscomposer.gate import NODE_FIELDS, STATE_FIELDS, declares
from src.viscomposer.core.host_edits import effective_response_schemas


def harness_schemas(max_visual_checkpoints: int) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        initial_path, revision_path = effective_response_schemas(
            Path(tmp), max_visual_checkpoints=max_visual_checkpoints)
        return (json.loads(initial_path.read_text(encoding="utf-8")),
                json.loads(revision_path.read_text(encoding="utf-8")))


def _prune_object(schema: dict[str, Any], keep: set[str]) -> None:
    schema["properties"] = {k: v for k, v in schema["properties"].items() if k in keep}
    schema["required"] = [k for k in schema.get("required", []) if k in keep]


def _prune_response(schema: dict[str, Any], arm: Arm) -> dict[str, Any]:
    if arm.is_full:
        return schema
    schema = copy.deepcopy(schema)
    keep = set(schema["properties"])
    if not arm.declares_state:
        keep.discard("design_state")
    if not arm.declares_plan:
        keep.discard("critique")
        schema.get("$defs", {}).pop("critique", None)
    _prune_object(schema, keep)
    if "design_state" in schema["properties"]:
        state = schema["properties"]["design_state"]
        _prune_object(state, {k for k, owner in STATE_FIELDS.items() if declares(arm, owner)})
        nodes = state["properties"]["nodes"]["items"]
        _prune_object(nodes, {k for k, owner in NODE_FIELDS.items() if declares(arm, owner)})
    return schema


def response_schemas(arm: Arm, max_visual_checkpoints: int) -> tuple[dict[str, Any], dict[str, Any]]:
    initial, revision = harness_schemas(max_visual_checkpoints)
    return _prune_response(initial, arm), _prune_response(revision, arm)


def tool_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Self-contained parameter schema for a pi submit tool: local $refs inlined."""
    defs = schema.get("$defs", {})

    def inline(value: Any, seen: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/"):]
                if name in seen:
                    raise ValueError(f"recursive schema reference: {name}")
                merged = {**defs[name], **{k: v for k, v in value.items() if k != "$ref"}}
                return inline(merged, seen + (name,))
            # Only the two schema-level keys go; "title" may be a property name.
            return {k: inline(v, seen) for k, v in value.items() if k not in ("$defs", "$schema")}
        if isinstance(value, list):
            return [inline(item, seen) for item in value]
        return value

    return inline(schema)
