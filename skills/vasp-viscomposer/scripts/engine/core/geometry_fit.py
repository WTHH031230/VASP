"""Deterministically compact small local overflows without shrinking typography."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MARKER = "deterministic-small-overflow-fit"
MAX_NODE_EXCESS_PX = 48.0
MAX_TOTAL_EXCESS_PX = 64.0
MIN_CONTAINER_PADDING_PX = 12.0
MIN_CHILD_MARGIN_PX = 7.0
MIN_CHILD_PADDING_PX = 10.0
MIN_DESCENDANT_MARGIN_PX = 3.0


def _px(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def build_small_overflow_fit(report: dict[str, Any]) -> dict[str, Any]:
    """Build conservative CSS for bounded vertical container overflows.

    The repair only reduces existing local padding and bounded child/descendant spacing.
    It never changes font sizes, line heights, content, evidence dimensions, or global
    tracks.
    """

    hard = report.get("hardOverflow") or []
    if not hard:
        return {"applied": False, "reason": "no hard overflow", "css": "", "nodes": []}
    if report.get("outside") or report.get("missingImages"):
        return {
            "applied": False,
            "reason": "outside or missing assets require a structural repair",
            "css": "",
            "nodes": [],
        }

    by_node: dict[str, dict[str, Any]] = {}
    total_excess = 0.0
    for item in hard:
        client = item.get("client") or [0, 0]
        scroll = item.get("scroll") or [0, 0]
        horizontal = _px(scroll[0]) - _px(client[0])
        vertical = _px(scroll[1]) - _px(client[1])
        node_id = str(item.get("node_id") or "")
        if (
            not node_id
            or item.get("tag") not in {"ARTICLE", "DIV", "SECTION", "FIGURE"}
            or horizontal > 4
            or vertical <= 4
            or vertical > MAX_NODE_EXCESS_PX
            or not item.get("box")
            or not item.get("children")
        ):
            return {
                "applied": False,
                "reason": "overflow is not a bounded local vertical container case",
                "css": "",
                "nodes": [],
            }
        total_excess += vertical
        current = by_node.get(node_id)
        if current is None or vertical > current["vertical_excess_px"]:
            by_node[node_id] = {"item": item, "vertical_excess_px": vertical}
    if total_excess > MAX_TOTAL_EXCESS_PX:
        return {
            "applied": False,
            "reason": f"total overflow {total_excess:.1f}px exceeds safe compaction limit",
            "css": "",
            "nodes": [],
        }

    rules: list[str] = []
    audits: list[dict[str, Any]] = []
    for node_id, entry in sorted(by_node.items()):
        item = entry["item"]
        selector = f'[data-node-id="{node_id}"]'
        box = item["box"]
        declarations: list[str] = []
        estimated = 0.0
        for key, css_name in (
            ("padding_top", "padding-top"),
            ("padding_bottom", "padding-bottom"),
        ):
            old = _px(box.get(key))
            new = min(old, MIN_CONTAINER_PADDING_PX)
            if new + 0.01 < old:
                declarations.append(f"{css_name}:{new:g}px!important")
                estimated += old - new
        if declarations:
            rules.append(f"{selector}{{{';'.join(declarations)}}}")

        child_changes = 0
        descendant_changes = 0
        for child in item["children"]:
            child_declarations: list[str] = []
            for key, css_name, floor in (
                ("margin_top", "margin-top", MIN_CHILD_MARGIN_PX),
                ("margin_bottom", "margin-bottom", MIN_CHILD_MARGIN_PX),
                ("padding_top", "padding-top", MIN_CHILD_PADDING_PX),
                ("padding_bottom", "padding-bottom", MIN_CHILD_PADDING_PX),
            ):
                old = _px(child.get(key))
                new = min(old, floor)
                if new + 0.01 < old:
                    child_declarations.append(f"{css_name}:{new:g}px!important")
                    estimated += old - new
            if child_declarations:
                index = int(child["index"])
                rules.append(
                    f"{selector}>:nth-child({index})"
                    f"{{{';'.join(child_declarations)}}}"
                )
                child_changes += 1
        required = entry["vertical_excess_px"] + 2
        if estimated < required:
            for descendant in item.get("descendants") or []:
                path = str(descendant.get("selector") or "")
                if not path:
                    continue
                descendant_declarations: list[str] = []
                reclaimed = 0.0
                for key, css_name in (
                    ("margin_top", "margin-top"),
                    ("margin_bottom", "margin-bottom"),
                ):
                    old = _px(descendant.get(key))
                    new = min(old, MIN_DESCENDANT_MARGIN_PX)
                    if new + 0.01 < old:
                        descendant_declarations.append(f"{css_name}:{new:g}px!important")
                        reclaimed += old - new
                if descendant_declarations:
                    rules.append(
                        f"{selector}{path}"
                        f"{{{';'.join(descendant_declarations)}}}"
                    )
                    estimated += reclaimed
                    descendant_changes += 1
                if estimated >= required:
                    break
        if estimated < required:
            return {
                "applied": False,
                "reason": f"safe spacing budget is insufficient for node {node_id}",
                "css": "",
                "nodes": [],
            }
        audits.append(
            {
                "node_id": node_id,
                "vertical_excess_px": round(entry["vertical_excess_px"], 3),
                "estimated_reduction_px": round(estimated, 3),
                "direct_children_adjusted": child_changes,
                "descendants_adjusted": descendant_changes,
            }
        )

    css = (
        f"\n/* {MARKER}:start */\n"
        + "\n".join(rules)
        + f"\n/* {MARKER}:end */\n"
    )
    return {"applied": True, "reason": "bounded local compaction", "css": css, "nodes": audits}


def apply_small_overflow_fit(workspace: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Append one auditable deterministic-fit block to the maintained stylesheet."""

    style_path = workspace / "style.css"
    style = style_path.read_text(encoding="utf-8")
    if MARKER in style:
        return {"applied": False, "reason": "fit marker already present", "css": "", "nodes": []}
    audit = build_small_overflow_fit(report)
    if audit["applied"]:
        style_path.write_text(style.rstrip() + audit["css"], encoding="utf-8")
    return audit
