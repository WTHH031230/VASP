"""Quality gate and source contract, split by module.

Ports src/pipeline/paper2poster/workbench.py (quality_gate, semantic_contract_issues,
_schema_issues, design_plan_from_state, bundle_from_workspace) and the correspondence
part of render_html.validate_bundle, keeping every message and its order. Each check
runs only when its owning module is plugged in (spec §3.4). The full arm must return
exactly the stored quality_gate.json of the main batch.

Ownership key: STATE = either VF or SEM (the design_state exists).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.viscomposer.arms import ASSETS, BASIC, SEM, VF, Arm
from src.viscomposer.core.render_html import (
    PosterContractError,
    _node_ids_in_html,
    validate_source_safety,
)
from src.viscomposer.core.workbench import (
    EVIDENCE_KINDS,
    ROLE_RANK,
    _normalized_title_text,
    _visible_h1_titles,
    load_json,
)

STATE = "state"

STATE_FIELDS = {
    "schema_version": STATE, "paper_title": SEM, "poster_title": SEM, "concept": SEM,
    "stages": SEM, "territories": SEM, "nodes": STATE, "edges": VF,
    "reading_order": SEM, "visual_hierarchy": VF, "type_scale": SEM, "decision_log": SEM,
}
NODE_FIELDS = {"id": STATE, "kind": STATE, "semantic_role": SEM, "stage_id": SEM,
               "summary": SEM, "priority": SEM}


def owns(arm: Arm, owner: str) -> bool:
    return arm.declares_state if owner == STATE else arm.has(owner)


def source_files(arm: Arm) -> tuple[str, ...]:
    files = ["poster.html", "style.css"]
    if arm.declares_state:
        files.append("design_state.json")
    if arm.declares_plan:
        files.append("critique.json")
    return tuple(files)


def declares(arm: Arm, owner: str) -> bool:
    """Whether a design_state field is declared (checks still follow `owns`)."""
    return arm.declares_plan if owner == SEM else owns(arm, owner)


def schema_issues(state: dict[str, Any], arm: Arm) -> list[str]:
    issues: list[str] = []
    required = {field for field, owner in STATE_FIELDS.items() if owns(arm, owner)}
    missing = sorted(required - set(state))
    if missing:
        return [f"design_state is missing required fields: {missing}"]
    if state.get("schema_version") != 1:
        issues.append("design_state schema_version must equal 1")
    for key, minimum, maximum in (
        ("stages", 4, 6), ("territories", 2, 5), ("nodes", 6, 36),
        ("edges", 3, 80), ("reading_order", 5, 36),
    ):
        if not owns(arm, STATE_FIELDS[key]):
            continue
        value = state.get(key)
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            issues.append(f"design_state {key} must contain {minimum}-{maximum} items")
    if issues:
        return issues
    identifier = re.compile(r"^[a-z][a-z0-9_-]{1,48}$")
    semantic_roles = set(ROLE_RANK)
    node_kinds = {
        "claim", "figure", "table", "equation", "algorithm", "transition",
        "context", "conclusion", "branding",
    }
    node_fields = {field for field, owner in NODE_FIELDS.items() if owns(arm, owner)}
    for collection, fields in (
        ("stages", {"id", "role", "title", "purpose", "node_ids"}),
        ("territories", {"id", "title", "purpose", "width_weight", "node_ids"}),
        ("nodes", node_fields),
        ("edges", {"from", "to", "relation"}),
    ):
        if not owns(arm, STATE_FIELDS[collection]):
            continue
        for index, item in enumerate(state[collection]):
            if not isinstance(item, dict):
                issues.append(f"design_state {collection}[{index}] must be an object")
                continue
            absent = sorted(fields - set(item))
            if absent:
                issues.append(f"design_state {collection}[{index}] misses fields: {absent}")
    if issues:
        return issues
    for collection in ("stages", "territories", "nodes"):
        if not owns(arm, STATE_FIELDS[collection]):
            continue
        for item in state[collection]:
            if not identifier.fullmatch(str(item["id"])):
                issues.append(f"invalid {collection} ID: {item['id']}")
    if arm.has(SEM):
        for stage in state["stages"]:
            if stage["role"] not in semantic_roles - {"identity"}:
                issues.append(f"invalid stage role: {stage['role']}")
    for node in state["nodes"]:
        if node["kind"] not in node_kinds:
            issues.append(f"invalid node kind: {node['kind']}")
        if arm.has(SEM) and node["semantic_role"] not in semantic_roles:
            issues.append(f"invalid node semantic_role: {node['semantic_role']}")
    if arm.has(VF):
        hierarchy = state.get("visual_hierarchy")
        if not isinstance(hierarchy, dict) or not {
            "primary_anchor", "secondary_anchors", "rationale"
        } <= set(hierarchy):
            issues.append("visual_hierarchy is incomplete")
    if arm.has(SEM):
        type_scale = state.get("type_scale")
        if not isinstance(type_scale, dict) or not {
            "title_px", "chapter_px", "group_px", "body_px", "caption_px"
        } <= set(type_scale):
            issues.append("type_scale is incomplete")
    return issues


def semantic_contract_issues(state: dict[str, Any], html: str, paper_title: str,
                             arm: Arm) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    if not arm.declares_state:
        return [], warnings
    issues = schema_issues(state, arm)
    if issues:
        return issues, warnings

    nodes = state["nodes"]
    node_map = {item["id"]: item for item in nodes}
    node_ids = set(node_map)
    if len(node_ids) != len(nodes):
        issues.append("design_state node IDs are not unique")

    if arm.has(SEM):
        stage_map = {item["id"]: item for item in state["stages"]}
        stage_roles = [item["role"] for item in state["stages"]]
        for required in ("framing", "overview", "mechanism", "result"):
            if required not in stage_roles:
                issues.append(f"story is missing required semantic role: {required}")
        ranks = [ROLE_RANK[role] for role in stage_roles]
        if ranks != sorted(ranks):
            issues.append("story stages do not follow framing → overview → mechanism → result")

        staged_nodes: list[str] = []
        for stage in state["stages"]:
            for node_id in stage["node_ids"]:
                staged_nodes.append(node_id)
                node = node_map.get(node_id)
                if node is None:
                    issues.append(f"stage {stage['id']} references unknown node {node_id}")
                elif node["stage_id"] != stage["id"]:
                    issues.append(f"node {node_id} disagrees with its enclosing stage")
                elif node["semantic_role"] != stage["role"]:
                    issues.append(
                        f"node {node_id} is {node['semantic_role']} inside {stage['role']} stage"
                    )
        expected_staged = {item["id"] for item in nodes if item["semantic_role"] != "identity"}
        if set(staged_nodes) != expected_staged or len(staged_nodes) != len(set(staged_nodes)):
            issues.append("non-identity nodes must belong to exactly one semantic stage")

        territory_nodes = [
            node_id for territory in state["territories"] for node_id in territory["node_ids"]
        ]
        if set(territory_nodes) != node_ids or len(territory_nodes) != len(set(territory_nodes)):
            issues.append("every node must belong to exactly one visual territory")

        reading_order = state["reading_order"]
        if set(reading_order) != node_ids or len(reading_order) != len(node_ids):
            issues.append("reading_order must contain every node exactly once")
        story_ranks = [
            ROLE_RANK[node_map[node_id]["semantic_role"]]
            for node_id in reading_order
            if node_id in node_map and node_map[node_id]["semantic_role"] != "identity"
        ]
        if story_ranks != sorted(story_ranks):
            issues.append("reading_order moves backward across semantic phases")

    html_nodes = re.findall(r"data-node-id\s*=\s*['\"]([^'\"]+)['\"]", html)
    if set(html_nodes) != node_ids or len(html_nodes) != len(set(html_nodes)):
        issues.append("HTML must contain every design_state node exactly once")
    if arm.has(SEM):
        html_stages = re.findall(r"data-stage-id\s*=\s*['\"]([^'\"]+)['\"]", html)
        if set(html_stages) != set(stage_map) or len(html_stages) != len(set(html_stages)):
            issues.append("HTML must contain every story stage exactly once")
        territory_ids = {item["id"] for item in state["territories"]}
        html_territories = re.findall(r"data-territory-id\s*=\s*['\"]([^'\"]+)['\"]", html)
        if set(html_territories) != territory_ids:
            issues.append("HTML visual territories disagree with design_state")

    if arm.has(VF):
        connected: set[str] = set()
        for edge in state["edges"]:
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                issues.append(f"edge references unknown nodes: {edge['from']} → {edge['to']}")
            connected.update((edge["from"], edge["to"]))
        orphan_evidence = sorted(
            item["id"] for item in nodes
            if item["kind"] in EVIDENCE_KINDS and item["id"] not in connected
        )
        if orphan_evidence:
            issues.append(f"evidence nodes have no semantic correspondence: {orphan_evidence}")

        anchor = state["visual_hierarchy"]["primary_anchor"]
        if anchor not in node_map:
            issues.append(f"primary visual anchor is unknown: {anchor}")
        elif node_map[anchor]["kind"] not in EVIDENCE_KINDS:
            issues.append("primary visual anchor must be a Figure, Table, Equation, or Algorithm")

    if arm.has(SEM):
        expected_title = _normalized_title_text(paper_title)
        if _normalized_title_text(state["paper_title"]) != expected_title:
            issues.append("design_state paper_title must exactly match extracted metadata title")
        if _normalized_title_text(state["poster_title"]) != expected_title:
            issues.append("design_state poster_title must exactly match extracted metadata title")
        visible_titles = _visible_h1_titles(html)
        if len(visible_titles) != 1:
            issues.append("HTML must contain exactly one visible H1 paper title")
        elif visible_titles[0] != expected_title:
            issues.append("visible H1 must reproduce the extracted paper title verbatim")
    return issues, warnings


def quality_gate(*, state: dict[str, Any], html: str, report: dict[str, Any], paper_title: str,
                 require_branding: bool | dict[str, Any], arm: Arm,
                 contract_error: str | None = None) -> dict[str, Any]:
    issues, warnings = semantic_contract_issues(state, html, paper_title, arm)
    semantic_issues = list(issues)
    if contract_error:
        issues.insert(0, contract_error)
    if arm.has(SEM):
        if report.get("status") != "pass":
            issues.append("browser geometry status is not pass")
        if report.get("hardOverflow"):
            issues.append(f"{len(report['hardOverflow'])} hard overflow item(s) remain")
        if report.get("outside"):
            issues.append(f"{len(report['outside'])} element(s) leave the canvas")
        if report.get("missingImages"):
            issues.append(f"{len(report['missingImages'])} local image(s) are missing")
        if report.get("emptyShells"):
            issues.append(f"{len(report['emptyShells'])} large empty content shell(s) remain")
        if report.get("flexMarginSpacers"):
            issues.append(
                f"{len(report['flexMarginSpacers'])} large flex margin spacer(s) split "
                "scientific content"
            )
        if report.get("titleOnlySidecars"):
            issues.append(
                f"{len(report['titleOnlySidecars'])} title-only sidecar(s) leave a "
                "locally empty grid column"
            )
    text_orientation_issues = (report.get("textOrientationIssues") or []) if arm.has(BASIC) else []
    if text_orientation_issues:
        examples = ", ".join(
            repr(str(item.get("text") or "")[:36]) for item in text_orientation_issues[:4]
        )
        issues.append(
            f"{len(text_orientation_issues)} informative text element(s) are "
            f"rotated, skewed, or vertically written: {examples}"
        )

    fonts = (report.get("fontStats") or {}) if arm.has(SEM) else {}
    identity_territories: set[str] = set()
    scientific_territories: list[dict[str, Any]] = []
    if arm.has(SEM):
        title_px = float(fonts.get("title_px") or 0)
        median_px = float(fonts.get("median_px") or 0)
        p10_px = float(fonts.get("p10_px") or 0)
        if title_px < 64:
            issues.append(f"title is only {title_px:.1f}px")
        if median_px < 19:
            issues.append(f"median visible text is only {median_px:.1f}px")
        if p10_px < 16.5:
            issues.append(f"10th-percentile visible text is only {p10_px:.1f}px")
        if int(fonts.get("below_17_count") or 0) > 4:
            issues.append("too many text leaves fall below the 17px print floor")

        chapters = report.get("chapterMetrics") or []
        if len(chapters) < 3:
            warnings.append("fewer than three shared chapter markers were detected")
        if chapters and min(float(item.get("h2_font_px") or 0) for item in chapters) < 30:
            issues.append("at least one chapter heading is below 30px")
        consistency = report.get("chapterConsistency") or {}
        if len(chapters) >= 2 and not consistency.get("consistent"):
            issues.append("chapter components do not share one computed type/spacing signature")

        if report.get("darkWideSurfaces"):
            warnings.append("a dark wide surface may dominate the narrative")
        identity_nodes = {
            item["id"] for item in state.get("nodes") or []
            if item.get("semantic_role") == "identity"
        }
        identity_territories = {
            item["id"] for item in state.get("territories") or []
            if set(item.get("node_ids") or []) and set(item["node_ids"]) <= identity_nodes
        }
        for item in report.get("territoryMetrics") or []:
            if item.get("id") in identity_territories:
                continue
            scientific_territories.append(item)
            if float(item.get("largest_vertical_void_ratio") or 0) > 0.075:
                issues.append(
                    f"territory {item.get('id')} has a "
                    f"{float(item['largest_vertical_void_ratio']):.3f} vertical void"
                )
            if float(item.get("painted_height_ratio") or 1) < 0.47:
                warnings.append(
                    f"territory {item.get('id')} has low painted coverage "
                    f"({float(item['painted_height_ratio']):.3f})"
                )
            if float(item.get("painted_bottom_gap_ratio") or 0) > 0.035:
                issues.append(
                    f"territory {item.get('id')} leaves a "
                    f"{float(item['painted_bottom_gap_ratio']):.3f} painted bottom void"
                )
        bottom_gaps = [
            max(0.0, float(item.get("bottom_gap_ratio") or 0)) for item in scientific_territories
        ]
        if len(bottom_gaps) >= 2 and max(bottom_gaps) - min(bottom_gaps) > 0.025:
            issues.append(
                "scientific territories do not share one bottom baseline "
                f"(painted-gap spread {max(bottom_gaps) - min(bottom_gaps):.3f})"
            )

    node_map = {item["id"]: item for item in state.get("nodes") or []}
    node_rects = report.get("nodeRects") or {}
    canvas = report.get("canvas") or [2400, 1800]
    canvas_width = max(1.0, float(canvas[0]))
    canvas_height = max(1.0, float(canvas[1]))
    scientific_images: list[dict[str, Any]] = []
    small_scientific_images: list[dict[str, Any]] = []
    anchor = None
    far_correspondence: list[dict[str, Any]] = []
    if arm.has(VF):
        figure_node_ids = {
            item["id"] for item in state.get("nodes") or [] if item.get("kind") == "figure"
        }
        scientific_images = [
            item for item in report.get("scientificImageMetrics") or []
            if item.get("node_id") in figure_node_ids
        ]
        min_figure_width = canvas_width * 0.12
        min_figure_height = canvas_height * 0.065
        for item in scientific_images:
            effective = item.get("effective") or {}
            width = float(effective.get("w") or 0)
            height = float(effective.get("h") or 0)
            if width + 0.5 < min_figure_width or height + 0.5 < min_figure_height:
                small_scientific_images.append(item)
                issues.append(
                    f"figure {item.get('node_id')} is thumbnail-sized "
                    f"({width:.0f}x{height:.0f}px effective); enlarge it or omit it"
                )
        anchor = (state.get("visual_hierarchy") or {}).get("primary_anchor")
        anchor_rect = node_rects.get(anchor) if anchor else None
        if anchor and not anchor_rect:
            issues.append(f"primary anchor {anchor} has no rendered rectangle")
        elif anchor_rect:
            area_ratio = (
                float(anchor_rect["w"]) * float(anchor_rect["h"]) / (canvas_width * canvas_height)
            )
            if area_ratio < 0.035:
                issues.append(f"primary anchor {anchor} occupies only {area_ratio:.3f} of the canvas")
        for edge in report.get("edgeDistances") or []:
            source = node_map.get(edge.get("from"))
            target = node_map.get(edge.get("to"))
            if not source or not target:
                continue
            if not ({source["kind"], target["kind"]} & EVIDENCE_KINDS):
                continue
            distance = float(edge.get("normalized_distance") or 0)
            if distance > 0.42:
                far_correspondence.append(
                    {"from": edge.get("from"), "to": edge.get("to"), "distance": round(distance, 3)}
                )
        if far_correspondence:
            issues.append(
                f"{len(far_correspondence)} claim–evidence edge(s) are not visually adjacent"
            )

    centers: list[tuple[str, float]] = []
    if arm.has(SEM):
        stage_metrics: dict[str, dict[str, Any]] = {}
        for stage in state.get("stages") or []:
            rects = [node_rects.get(node_id) for node_id in stage["node_ids"]]
            rects = [rect for rect in rects if rect]
            if not rects:
                continue
            left = min(float(rect["x"]) for rect in rects)
            top = min(float(rect["y"]) for rect in rects)
            right = max(float(rect["x"]) + float(rect["w"]) for rect in rects)
            bottom = max(float(rect["y"]) + float(rect["h"]) for rect in rects)
            stage_metrics[stage["id"]] = {
                "id": stage["id"],
                "rect": {"x": left, "y": top, "w": right - left, "h": bottom - top},
            }
        ordered_stages = sorted(
            (
                item for item in state.get("stages") or []
                if item.get("role") in ROLE_RANK and item.get("role") != "identity"
            ),
            key=lambda item: ROLE_RANK[item["role"]],
        )
        for stage in ordered_stages:
            metric = stage_metrics.get(stage["id"])
            if metric:
                rect = metric["rect"]
                centers.append((stage["role"], float(rect["x"]) + float(rect["w"]) / 2))
        for previous, current in zip(centers, centers[1:]):
            if current[1] + 80 < previous[1]:
                issues.append(f"macro story moves left from {previous[0]} to {current[0]}")
        if len(centers) >= 4 and centers[-1][1] - centers[0][1] < canvas_width * 0.4:
            warnings.append("semantic stages use less than 40% of canvas width progression")

    brand = (report.get("brandMetrics") or {}) if arm.has(ASSETS) else {}
    brand_padding_intrusions = brand.get("padding_intrusions") or []
    if arm.has(ASSETS):
        _brand_issues(issues, brand, brand_padding_intrusions, require_branding)

    geometry = {
        "status": report.get("status"),
        "hard_overflow": report.get("hardOverflow") or [],
        "outside": report.get("outside") or [],
        "missing_images": report.get("missingImages") or [],
        "empty_shells": report.get("emptyShells") or [],
        "scientific_images": scientific_images,
        "small_scientific_images": small_scientific_images,
        "flex_margin_spacers": report.get("flexMarginSpacers") or [],
        "title_only_sidecars": report.get("titleOnlySidecars") or [],
        "text_orientation_issues": text_orientation_issues,
        "brand_padding_intrusions": brand_padding_intrusions,
        "overflow_excess_px": round(
            sum(
                max(0, float((item.get("scroll") or (0, 0))[0])
                    - float((item.get("client") or (0, 0))[0]))
                + max(0, float((item.get("scroll") or (0, 0))[1])
                      - float((item.get("client") or (0, 0))[1]))
                for item in report.get("hardOverflow") or []
            ),
            3,
        ),
        "territories": [
            {key: item.get(key) for key in (
                "id", "largest_vertical_void_ratio", "painted_height_ratio",
                "large_vertical_gaps", "painted_bottom_gap_ratio", "bottom_gap_ratio",
            )}
            for item in report.get("territoryMetrics") or []
            if item.get("id") not in identity_territories
        ],
        "branding": report.get("brandMetrics") or {},
    }
    if not arm.has(SEM):
        # Measured values stay out of the arm's gate unless their module is plugged in.
        for key in ("status", "hard_overflow", "outside", "missing_images", "empty_shells",
                    "flex_margin_spacers", "title_only_sidecars", "overflow_excess_px",
                    "territories"):
            geometry[key] = None if key == "status" else (0.0 if key == "overflow_excess_px" else [])
    if not arm.has(ASSETS):
        geometry["branding"] = {}
    return {
        "schema_version": 1,
        "accepted": not issues,
        "issues": list(dict.fromkeys(issues)),
        "semantic_issues": list(dict.fromkeys(semantic_issues)),
        "semantic_warnings": list(dict.fromkeys(warnings)),
        "anchor": anchor,
        "far_correspondence": far_correspondence,
        "font_stats": fonts,
        "stage_centers": [{"role": role, "x": round(x, 2)} for role, x in centers],
        "geometry": geometry,
    }


def _brand_issues(issues: list[str], brand: dict[str, Any], brand_padding_intrusions: list,
                  require_branding: bool | dict[str, Any]) -> None:
    if brand_padding_intrusions:
        maximum = max(float(item.get("max_excess") or 0) for item in brand_padding_intrusions)
        issues.append(
            f"{len(brand_padding_intrusions)} brand mark(s) intrude into the card's "
            f"content padding by up to {maximum:.1f}px"
        )
    if not require_branding:
        return
    if isinstance(require_branding, dict):
        min_non_qr = int(require_branding.get("min_non_qr") or 0)
        min_qr = int(require_branding.get("min_qr") or 0)
        raw_max_non_qr = require_branding.get("max_non_qr")
        max_non_qr = int(raw_max_non_qr) if raw_max_non_qr is not None else None
    else:
        min_non_qr, min_qr = 2, 0
        max_non_qr = None
    actual_non_qr = int(brand.get("non_qr_count") or 0)
    actual_qr = int(brand.get("qr_count") or 0)
    if not brand.get("present"):
        issues.append("required brand cluster is missing")
    if actual_non_qr < min_non_qr:
        issues.append(f"brand cluster uses {actual_non_qr}/{min_non_qr} required non-QR marks")
    if max_non_qr is not None and actual_non_qr > max_non_qr:
        issues.append(f"brand cluster uses {actual_non_qr}/{max_non_qr} permitted non-QR marks")
    if actual_qr < min_qr:
        issues.append(f"brand cluster uses {actual_qr}/{min_qr} required QR marks")
    if min_non_qr and float(brand.get("non_qr_min_width") or 0) + 0.5 < 150:
        issues.append("a non-QR brand mark renders below 150px")
    if actual_non_qr and actual_qr and (
        float(brand.get("qr_max_width") or 0) > float(brand.get("non_qr_max_width") or 0)
    ):
        issues.append("QR code is larger than the dominant logo")
    low_contrast = [
        item for item in brand.get("images") or []
        if not item.get("is_qr") and item.get("contrast_ratio") is not None
        and float(item["contrast_ratio"]) < 1.5
    ]
    if low_contrast:
        issues.append(
            "low-contrast brand mark(s): "
            + ", ".join(str(item.get("src") or item.get("alt")) for item in low_contrast)
        )
    if isinstance(require_branding, dict) and "qr_assets" in require_branding:
        expected_qr = {
            str(item.get("relative_path") or ""): str(item.get("label") or "")
            for item in (require_branding.get("qr_assets") or [])
            if isinstance(item, dict) and item.get("relative_path")
        }
        for item in brand.get("images") or []:
            if not item.get("is_qr"):
                continue
            source = str(item.get("src") or "")
            expected_label = expected_qr.get(source)
            if not expected_label:
                issues.append(f"QR asset is not in the grounded brand contract: {source}")
                continue
            alt = " ".join(str(item.get("alt") or "").split())
            visible = " ".join(str(item.get("visible_label") or "").split())
            declared = " ".join(str(item.get("data_brand_label") or "").split())
            if declared != expected_label:
                issues.append(f"QR {source} data-brand-label must be exactly {expected_label!r}")
            if item.get("data_brand_kind") != "qr":
                issues.append(f"QR {source} data-brand-kind must be exactly 'qr'")
            if visible != expected_label:
                issues.append(f"QR {source} visible label must be exactly {expected_label!r}")
            if not re.search(rf"\b{re.escape(expected_label)}\b", alt, re.I):
                issues.append(f"QR {source} alt text must identify it as {expected_label!r}")
    if isinstance(require_branding, dict) and "logo_assets" in require_branding:
        expected_logos = {
            str(item.get("relative_path") or ""): str(item.get("label") or "")
            for item in (require_branding.get("logo_assets") or [])
            if isinstance(item, dict) and item.get("relative_path")
        }
        for item in brand.get("images") or []:
            if item.get("is_qr"):
                continue
            source = str(item.get("src") or "")
            expected_label = expected_logos.get(source)
            if not expected_label:
                issues.append(f"Logo asset is not in the grounded brand contract: {source}")
                continue
            declared = " ".join(str(item.get("data_brand_label") or "").split())
            if item.get("data_brand_kind") != "logo":
                issues.append(f"Logo {source} data-brand-kind must be exactly 'logo'")
            if declared != expected_label:
                issues.append(f"Logo {source} data-brand-label must be exactly {expected_label!r}")
            alt = " ".join(str(item.get("alt") or "").split())
            if expected_label.casefold() not in alt.casefold():
                issues.append(f"Logo {source} alt text must identify {expected_label!r}")


# --- source contract -------------------------------------------------------------------

def design_plan(state: dict[str, Any], arm: Arm) -> dict[str, Any]:
    """Renderer plan; raises KeyError on the first missing key like design_plan_from_state."""
    plan: dict[str, Any] = {}
    for key in ("concept", "reading_order", "territories", "nodes", "edges"):
        if owns(arm, STATE_FIELDS[key]):
            plan[key] = state[key]
    return plan


def bundle_from_workspace(workspace: Path, arm: Arm) -> tuple[dict[str, Any], dict[str, Any]]:
    files = source_files(arm)
    missing = [name for name in files if not (workspace / name).is_file()]
    if missing:
        raise PosterContractError(f"designer did not create required files: {missing}")
    state = load_json(workspace / "design_state.json") if arm.declares_state else {}
    critique = load_json(workspace / "critique.json") if arm.declares_plan else None
    bundle = {
        "html": (workspace / "poster.html").read_text(encoding="utf-8"),
        "css": (workspace / "style.css").read_text(encoding="utf-8"),
        "design_plan": design_plan(state, arm),
        "rationale": state["concept"] if arm.has(SEM) else "",
    }
    return bundle, {"design_state": state, "critique": critique}


def validate_bundle(bundle: dict[str, Any], allowed_assets: set[str], arm: Arm, *,
                    canvas_width: int, canvas_height: int) -> None:
    """Safety (base) then the plan/DOM correspondence owned by VF and SEM."""
    validate_source_safety(bundle, allowed_assets, canvas_width=canvas_width,
                           canvas_height=canvas_height)
    if not arm.declares_state:
        return
    html = str(bundle.get("html", ""))
    plan = bundle.get("design_plan") or {}
    nodes = plan.get("nodes") or []
    node_ids = [str(item.get("id", "")) for item in nodes]
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise PosterContractError("design_plan node IDs must be non-empty and unique")
    known = set(node_ids)
    missing_html = sorted(known - _node_ids_in_html(html))
    if missing_html:
        raise PosterContractError(f"plan nodes missing from HTML: {missing_html}")
    if arm.has(SEM):
        territories = plan.get("territories") or []
        territory_ids = [str(item.get("id", "")) for item in territories]
        if not territory_ids or len(territory_ids) != len(set(territory_ids)):
            raise PosterContractError("territory IDs must be non-empty and unique")
        html_territories = set(re.findall(r"data-territory-id\s*=\s*['\"]([^'\"]+)['\"]", html))
        missing_territories = sorted(set(territory_ids) - html_territories)
        if missing_territories:
            raise PosterContractError(f"plan territories missing from HTML: {missing_territories}")
        for territory in territories:
            unknown_nodes = sorted(set(map(str, territory.get("node_ids") or [])) - known)
            if unknown_nodes:
                raise PosterContractError(
                    f"territory {territory.get('id')} references unknown nodes: {unknown_nodes}"
                )
        reading_order = [str(value) for value in plan.get("reading_order") or []]
        unknown_order = sorted(set(reading_order) - known)
        if unknown_order:
            raise PosterContractError(f"reading_order references unknown nodes: {unknown_order}")
    if arm.has(VF):
        linked: set[str] = set()
        for edge in plan.get("edges") or []:
            source = str(edge.get("from", ""))
            target = str(edge.get("to", ""))
            if source not in known or target not in known:
                raise PosterContractError(f"edge references unknown nodes: {source} -> {target}")
            linked.update((source, target))
        orphans = sorted(
            str(item["id"]) for item in nodes
            if item.get("kind") in EVIDENCE_KINDS and str(item["id"]) not in linked
        )
        if orphans:
            raise PosterContractError(f"orphan evidence nodes: {orphans}")


def contract_error(workspace: Path, allowed_assets: set[str], arm: Arm, *, canvas_width: int,
                   canvas_height: int) -> tuple[str | None, dict[str, Any], str]:
    """Pre-render check as render_checkpoint does it: (error, state, html)."""
    state: dict[str, Any] = {}
    html = ""
    try:
        bundle, maintained = bundle_from_workspace(workspace, arm)
        state = maintained["design_state"]
        html = bundle["html"]
        validate_bundle(bundle, allowed_assets, arm, canvas_width=canvas_width,
                        canvas_height=canvas_height)
        return None, state, html
    except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if not state and arm.declares_state and (workspace / "design_state.json").is_file():
            try:
                state = load_json(workspace / "design_state.json")
            except (OSError, json.JSONDecodeError):
                state = {}
        if not html and (workspace / "poster.html").is_file():
            html = (workspace / "poster.html").read_text(encoding="utf-8")
        return f"source contract error: {exc}", state, html
