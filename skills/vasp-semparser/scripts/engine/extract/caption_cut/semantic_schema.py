"""Strict semantic JSON schema and lossless adapters to the existing bundle format."""

import copy
import json

from jsonschema import Draft202012Validator


class FormatError(ValueError):
    """Invalid machine-consumption format; eligible for a bounded format retry."""


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def array_schema(items):
    return {"type": "array", "items": items}


def build_schema(n_lines, roles):
    text = {"type": "string"}
    number_id = {"anyOf": [{"type": "integer"}, {"type": "string", "pattern": r"\S"}]}
    line_id = {"type": "integer", "minimum": 0, "maximum": n_lines - 1}
    family = {"type": "string", "pattern": r"\S"}
    status = {"type": "string", "enum": ["ok", "partial", "unresolved"]}
    special = {"heading", "figure_caption", "table_title", "subfig_caption"}

    def segment(names, extra=None):
        return object_schema({"start": line_id, "end": line_id,
                              "role": {"type": "string", "enum": names}, **(extra or {})})

    entry = object_schema({"key": text, "value": {"$ref": "#/$defs/json_value"}})
    emphasis = array_schema({"$ref": "#/$defs/dictionary_entry"})
    metadata = object_schema({
        "title": text, "authors": array_schema(text),
        "author_index_map": array_schema(object_schema({
            "author": text, "institute_indices": array_schema({"type": "integer", "minimum": 1})})),
        "institutes": array_schema(text), "venue": text, "paper_url": text,
        "code_url": text, "arxiv_id": text, "project_url": text,
    })
    side = object_schema({"candidate_id": {"type": ["string", "null"]},
                          "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                          "reason": text})
    schema = object_schema({
        "segments": {"type": "array", "minItems": 1, "items": {"anyOf": [
            segment([role for role in roles if role not in special]),
            segment(["heading"], {"level": {"type": "string", "enum": ["section", "subsection", "runin"]}}),
            segment(["figure_caption", "table_title"], {"num": number_id, "family": family}),
            segment(["subfig_caption"], {"parent": {"anyOf": [*number_id["anyOf"], {"type": "null"}]}, "family": family}),
        ]}},
        "tables": array_schema(object_schema({
            "num": number_id, "caption": text, "headers": array_schema(text),
            "rows": array_schema(array_schema(text)), "footnotes": array_schema(text),
            "emphasis": emphasis, "source_line_ids": array_schema(line_id),
            "status": status, "family": family,
        })),
        "equations": array_schema(object_schema({
            "label": number_id, "latex": {"type": ["string", "null"]}, "context": text,
            "source_line_ids": array_schema(line_id), "status": status, "family": family,
        })),
        "algorithms": array_schema(object_schema({
            "num": number_id, "title": text, "inputs": array_schema(text),
            "outputs": array_schema(text), "steps": array_schema(text),
            "source_line_ids": array_schema(line_id), "status": status, "family": family,
        })),
        "metadata": metadata,
        "page_outer_boundaries": array_schema(object_schema({
            "page": {"type": "integer", "minimum": 1},
            "top": side, "bottom": side, "left": side, "right": side,
        })),
    })
    schema["$defs"] = {
        "dictionary_entry": entry,
        "json_value": {"anyOf": [
            {"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
            array_schema({"$ref": "#/$defs/json_value"}),
            object_schema({"entries": array_schema({"$ref": "#/$defs/dictionary_entry"})}),
        ]},
    }
    Draft202012Validator.check_schema(schema)
    return schema


def encode_value(value):
    if isinstance(value, dict):
        return {"entries": [{"key": k, "value": encode_value(v)} for k, v in value.items()]}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    return value


def decode_entries(entries, path):
    result = {}
    for index, entry in enumerate(entries):
        key = entry["key"]
        if key in result:
            raise FormatError(f"{path}[{index}]: duplicate dictionary key {key!r}")
        result[key] = decode_value(entry["value"], f"{path}[{index}].value")
    return result


def decode_value(value, path):
    if isinstance(value, dict):
        return decode_entries(value["entries"], path + ".entries")
    if isinstance(value, list):
        return [decode_value(v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def to_wire(legacy):
    result = copy.deepcopy(legacy)
    result["metadata"]["author_index_map"] = [
        {"author": author, "institute_indices": indices}
        for author, indices in legacy["metadata"]["author_index_map"].items()]
    for item in result["tables"]:
        item["emphasis"] = encode_value(item["emphasis"])["entries"]
    return result


def to_legacy(wire):
    result = copy.deepcopy(wire)
    authors = {}
    for index, item in enumerate(wire["metadata"]["author_index_map"]):
        if item["author"] in authors:
            raise FormatError(f"metadata.author_index_map[{index}]: duplicate author {item['author']!r}")
        authors[item["author"]] = item["institute_indices"]
    result["metadata"]["author_index_map"] = authors
    for index, item in enumerate(result["tables"]):
        item["emphasis"] = decode_entries(item["emphasis"], f"tables[{index}].emphasis")
    return result


def parse_response(raw, schema):
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise FormatError(f"duplicate JSON object key {key!r}")
            obj[key] = value
        return obj

    def reject_constant(value):
        raise FormatError(f"non-JSON numeric constant {value}")

    try:
        obj = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        excerpt = raw[max(0, exc.pos - 60):exc.pos + 100]
        raise FormatError(f"JSON syntax at line {exc.lineno}, column {exc.colno}: {exc.msg}; near {excerpt!r}") from exc
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(obj))
    if errors:
        error = errors[0]
        path = "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in error.absolute_path)
        message = error.message[:1200]
        raise FormatError(f"{path}: {message}")
    return obj
