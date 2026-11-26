"""Strict host-side application of structured persistent-designer responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.viscomposer.core.harness_budget import MAX_REVISIONS, MIN_REVISIONS
from src.viscomposer.core.workbench import load_json, write_json


HERE = Path(__file__).resolve().parent
SCHEMAS = HERE.parents[2] / "configs" / "schemas" / "viscomposer"


class HostEditError(ValueError):
    """Raised when a model edit is ambiguous, stale, or out of scope."""


def effective_response_schemas(
    output_dir: Path,
    *,
    max_visual_checkpoints: int | None = None,
) -> tuple[Path, Path]:
    """Inline local refs and expand critique checkpoints only for long harnesses."""

    if max_visual_checkpoints is not None and not (
        MIN_REVISIONS <= max_visual_checkpoints <= MAX_REVISIONS
    ):
        raise ValueError(
            f"max_visual_checkpoints must be {MIN_REVISIONS}-{MAX_REVISIONS}"
        )
    state = load_json(SCHEMAS / "design_state.schema.json")
    state.pop("$schema", None)
    state.pop("title", None)

    def rewrite_refs(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite_refs(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite_refs(item) for item in value]
        if value == "#/$defs/id":
            return "#/$defs/state_id"
        if value == "#/$defs/semantic_role":
            return "#/$defs/state_semantic_role"
        if value == "#/$defs/story_role":
            return "#/$defs/state_story_role"
        return value

    state_defs = state.pop("$defs")
    state = rewrite_refs(state)
    initial = load_json(SCHEMAS / "initial_designer_response.schema.json")
    checkpoint_schema = initial["$defs"]["critique"]["properties"]["checkpoint"]
    if (
        max_visual_checkpoints is not None
        and max_visual_checkpoints > int(checkpoint_schema["maximum"])
    ):
        checkpoint_schema["maximum"] = max_visual_checkpoints
    initial["$defs"]["state_id"] = state_defs["id"]
    initial["$defs"]["state_semantic_role"] = state_defs["semantic_role"]
    initial["$defs"]["state_story_role"] = state_defs["story_role"]
    initial["properties"]["design_state"] = state
    revision = load_json(SCHEMAS / "revision_designer_response.schema.json")
    revision["$defs"] = {
        "state_id": state_defs["id"],
        "state_semantic_role": state_defs["semantic_role"],
        "state_story_role": state_defs["story_role"],
        "critique": initial["$defs"]["critique"],
    }
    revision["properties"]["design_state"] = state
    revision["properties"]["critique"] = {"$ref": "#/$defs/critique"}
    initial_path = output_dir / "effective_initial_response.schema.json"
    revision_path = output_dir / "effective_revision_response.schema.json"
    write_json(initial_path, initial)
    write_json(revision_path, revision)
    return initial_path, revision_path


def parse_structured_message(message: str) -> dict[str, Any]:
    """Parse one Codex structured response and reject non-object payloads."""

    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise HostEditError(f"designer response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HostEditError("designer response must be one JSON object")
    return value


def apply_initial_response(workspace: Path, response: dict[str, Any]) -> None:
    """Materialize the first complete design response in the isolated workspace."""

    required = {"poster_html", "style_css", "design_state", "critique", "summary"}
    if set(response) != required:
        raise HostEditError(
            f"initial designer response fields disagree with contract: {sorted(response)}"
        )
    (workspace / "poster.html").write_text(
        str(response["poster_html"]), encoding="utf-8"
    )
    (workspace / "style.css").write_text(
        str(response["style_css"]), encoding="utf-8"
    )
    write_json(workspace / "design_state.json", response["design_state"])
    write_json(workspace / "critique.json", response["critique"])


def apply_revision_response(
    workspace: Path,
    response: dict[str, Any],
) -> list[dict[str, str]]:
    """Atomically apply unique exact replacements to the two source files."""

    required = {"file_edits", "design_state", "critique", "summary"}
    if set(response) != required:
        raise HostEditError(
            f"revision designer response fields disagree with contract: {sorted(response)}"
        )
    buffers = {
        name: (workspace / name).read_text(encoding="utf-8")
        for name in ("poster.html", "style.css")
    }
    audit: list[dict[str, str]] = []
    for index, edit in enumerate(response["file_edits"]):
        name = str(edit.get("file", ""))
        if name not in buffers:
            raise HostEditError(f"edit {index} targets forbidden file: {name}")
        old = str(edit.get("old_text", ""))
        new = str(edit.get("new_text", ""))
        count = buffers[name].count(old)
        if count != 1:
            raise HostEditError(
                f"edit {index} old_text occurs {count} times in {name}; expected exactly 1"
            )
        buffers[name] = buffers[name].replace(old, new, 1)
        audit.append(
            {
                "file": name,
                "reason": str(edit.get("reason", "")),
                "old_sha256": hashlib.sha256(old.encode("utf-8")).hexdigest(),
                "new_sha256": hashlib.sha256(new.encode("utf-8")).hexdigest(),
            }
        )
    for name, content in buffers.items():
        (workspace / name).write_text(content, encoding="utf-8")
    write_json(workspace / "design_state.json", response["design_state"])
    write_json(workspace / "critique.json", response["critique"])
    return audit
