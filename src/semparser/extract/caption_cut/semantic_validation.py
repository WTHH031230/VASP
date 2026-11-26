"""Validate semantic ranges and legacy fields; coverage is diagnostic only."""

import re

STRUCTURED_FIELDS = {
    "tables": {"num": "number_id", "caption": "string", "headers": "list",
               "rows": "list", "footnotes": "list", "emphasis": "object",
               "source_line_ids": "list", "status": "status"},
    "equations": {"label": "number_id", "latex": "nullable_string", "context": "string",
                  "source_line_ids": "list", "status": "status"},
    "algorithms": {"num": "number_id", "title": "string", "inputs": "list", "outputs": "list",
                   "steps": "list", "source_line_ids": "list", "status": "status"},
}


METADATA_FIELDS = {"title": "string", "authors": "list", "institutes": "list",
                   "author_index_map": "object", "venue": "string", "paper_url": "string",
                   "code_url": "string", "arxiv_id": "string", "project_url": "string"}


class ContractError(ValueError):
    """Machine-consumption failure, not a crop-quality or coverage failure."""


def check_type(value, kind, path):
    valid = {
        "string": lambda: isinstance(value, str),
        "list": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
        "number_id": lambda: type(value) is int or (isinstance(value, str) and bool(value.strip())),
        "nullable_string": lambda: value is None or isinstance(value, str),
        "status": lambda: isinstance(value, str) and value in ("ok", "partial", "unresolved"),
    }[kind]()
    if not valid:
        raise ContractError(f"{path}: expected {kind}; received {value!r}")


def check_fields(obj, fields, path):
    check_type(obj, "object", path)
    for key, kind in fields.items():
        if key not in obj:
            raise ContractError(f"{path}.{key}: missing required field ({kind})")
        check_type(obj[key], kind, f"{path}.{key}")


def validate_semantic(obj, n_lines, spec):
    check_fields(obj, {"segments": "list", "tables": "list", "equations": "list",
                      "algorithms": "list", "metadata": "object", "page_outer_boundaries": "list"}, "$response")
    if not obj["segments"]:
        raise ContractError("segments: empty; return the complete annotation")
    coverage = [0] * n_lines
    for index, segment in enumerate(obj["segments"]):
        path = f"segments[{index}]"
        check_type(segment, "object", path)
        role = segment.get("role")
        if not isinstance(role, str) or role not in spec["roles"]:
            raise ContractError(f"{path}.role={role!r}: not in {spec['roles']}. Headings use role=heading and level=section/subsection/runin.")
        start, end = segment.get("start"), segment.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < n_lines:
            raise ContractError(f"{path}: invalid source range {start!r}..{end!r}, expected 0 <= start <= end < {n_lines}")
        for line_id in range(start, end + 1):
            coverage[line_id] += 1
        if role == "heading" and segment.get("level") not in ("section", "subsection", "runin"):
            raise ContractError(f"{path}.level: heading requires section/subsection/runin")
        if role in ("figure_caption", "table_title"):
            check_fields(segment, {"num": "number_id", "family": "string"}, path)
        if role == "subfig_caption":
            if "parent" not in segment:
                raise ContractError(f"{path}.parent: required; use null only when truly unknown")
            if segment["parent"] is not None:
                check_type(segment["parent"], "number_id", f"{path}.parent")
            check_fields(segment, {"family": "string"}, path)
    for key, fields in spec["structured_fields"].items():
        for index, item in enumerate(obj[key]):
            check_fields(item, fields, f"{key}[{index}]")
    check_fields(obj["metadata"], spec["metadata_fields"], "metadata")
    for index, page in enumerate(obj["page_outer_boundaries"]):
        path = f"page_outer_boundaries[{index}]"
        check_fields(page, {"top": "object", "bottom": "object", "left": "object", "right": "object"}, path)
        if type(page.get("page")) is not int:
            raise ContractError(f"{path}.page: required integer")
        for side in ("top", "bottom", "left", "right"):
            entry = page[side]
            check_fields(entry, {"confidence": "string", "reason": "string"}, f"{path}.{side}")
            if entry["confidence"] not in ("high", "medium", "low"):
                raise ContractError(f"{path}.{side}.confidence: high/medium/low only")
            if "candidate_id" not in entry or (entry["candidate_id"] is not None and not isinstance(entry["candidate_id"], str)):
                raise ContractError(f"{path}.{side}.candidate_id: required string or null")
    # Semantic coverage is a quality outcome, not a regeneration trigger.
    return {"source_lines": n_lines,
            "missing_line_ids": [i for i, count in enumerate(coverage) if count == 0],
            "overlapping_line_ids": [i for i, count in enumerate(coverage) if count > 1]}


def build_spec(base_task):
    roles_text = base_task.split("Roles:\n", 1)[1].split("\nCritical distinctions", 1)[0]
    roles = re.findall(r'^- "([a-z_]+)":', roles_text, re.MULTILINE)
    assert roles and len(roles) == len(set(roles)) and "heading" in roles
    return {"roles": roles, "structured_fields": STRUCTURED_FIELDS,
            "metadata_fields": METADATA_FIELDS}
