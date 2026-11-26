"""Open caption names, closed processing roles; no geometry lives here."""

import re

DEFAULTS = {"figure_caption": "Figure", "subfig_caption": "Figure",
            "table_title": "Table", "equation_caption": "Equation",
            "algorithm": "Algorithm", "figure": "Figure", "table": "Table",
            "equation": "Equation", "tables": "Table", "equations": "Equation",
            "algorithms": "Algorithm"}
ALIASES = {"fig": "Figure", "figure": "Figure", "tab": "Table",
           "table": "Table", "equation": "Equation", "eq": "Equation",
           "algorithm": "Algorithm", "alg": "Algorithm"}


def family_name(value=None, role="figure_caption"):
    if value is None or value == "":
        return DEFAULTS.get(role, "Figure")
    if not isinstance(value, str):
        raise ValueError("family must be a string")
    value = " ".join(value.split()).rstrip(".")
    if not value or len(value) > 80 or not all(
            c.isalnum() or c in " _-" for c in value):
        raise ValueError(f"invalid family name: {value!r}")
    return ALIASES.get(value.casefold(), value)


def family_maps(segments, line_ids):
    valid = set(line_ids)
    result = {}
    for segment in segments:
        family = family_name(segment.get("family"), segment.get("role"))
        for line_id in range(segment.get("start", 0), segment.get("end", -1) + 1):
            if line_id in valid:
                result[line_id] = family
    return result


def split_label(label):
    match = re.fullmatch(r"(.+)\s+(\S+)", str(label).strip())
    if not match:
        return None
    try:
        return family_name(match[1]), match[2]
    except ValueError:
        return None


def caption_identity(label):
    match = re.search(r"(Figure|Fig|Table|Tab)\.?\s*((?:[A-Z]\.?)?\d+(?:\.\d+)*)",
                      str(label), re.I)
    if match:
        return (family_name(match[1]).casefold(), match[2])
    parts = split_label(label)
    return (parts[0].casefold(), parts[1]) if parts else None


def namespaced_num(number, family, default="Table"):
    family = family_name(family, default.lower())
    return number if family == default else f"{family.casefold()}:{number}"


def normalize_families(segments, structured):
    names, capabilities = {}, {}
    for segment in segments:
        role = segment.get("role")
        if role not in DEFAULTS:
            continue
        family = family_name(segment.get("family"), role)
        family = names.setdefault(family.casefold(), family)
        segment["family"] = family
        if role in {"figure_caption", "table_title"}:
            previous = capabilities.setdefault(family.casefold(), role)
            if previous != role:
                raise ValueError(f"one family cannot have two processing roles: {family}")
    for key in ("tables", "equations", "algorithms"):
        for element in structured.get(key, []):
            family = family_name(element.get("family"), key)
            element["family"] = names.setdefault(family.casefold(), family)
    return segments, structured
