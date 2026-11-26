"""Workspace contracts, deterministic gates, and checkpoint rendering."""

from __future__ import annotations

import difflib
import hashlib
from html import unescape
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from engine.core.render_html import (
    PosterContractError,
    render_bundle,
    validate_bundle,
)
from engine.core.visual_assets import (
    image_foreground_profile,
    image_palette_profile,
)


HERE = Path(__file__).resolve().parent
STATE_SCHEMA = HERE.parents[2] / "schemas" / "design_state.schema.json"
SOURCE_FILES = ("poster.html", "style.css", "design_state.json", "critique.json")
ROLE_RANK = {
    "identity": -1,
    "framing": 0,
    "overview": 1,
    "mechanism": 2,
    "result": 3,
    "takeaway": 4,
}
EVIDENCE_KINDS = {"figure", "table", "equation", "algorithm"}
QR_CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
QR_PAPER_HOST_PARTS = (
    "arxiv", "doi.org", "openreview", "acm.org", "ieee", "springer",
    "aclanthology", "semanticscholar", "mlr.press", "neurips.cc",
    "proceedings",
)


def canonical_qr_label(url: str, fallback: str = "Project") -> tuple[str, str]:
    """Return a destination label without confusing project sites with repositories."""

    try:
        host = (urlsplit(str(url or "")).hostname or "").lower()
    except ValueError:
        host = ""
    if any(host == domain or host.endswith(f".{domain}") for domain in QR_CODE_HOSTS):
        return "code", "Code"
    if any(part in host for part in QR_PAPER_HOST_PARTS):
        return "paper", "Paper"
    if host:
        return "project", "Project"
    label = str(fallback or "Project").strip().title()
    return label.lower(), label


def _bundled_logo_policy(
    metadata: dict[str, Any],
    available_labels: list[str],
) -> dict[str, Any]:
    """Avoid presenting one selectively fetched logo as the paper's full identity."""

    institutes = {
        " ".join(str(value).split()).casefold()
        for value in (metadata.get("institutes") or [])
        if str(value).strip()
    }
    matched = {
        " ".join(str(value).split()).casefold()
        for value in available_labels
        if str(value).strip()
    }
    coverage = len(matched) / max(1, len(institutes)) if institutes else 1.0
    omit = len(institutes) >= 3 and coverage < 0.25
    return {
        "status": "omit_incomplete" if omit else "representative",
        "institution_count": len(institutes),
        "matched_institution_count": len(matched),
        "coverage": round(coverage, 4),
        "reason": (
            "available logos cover less than 25% of a multi-institution paper"
            if omit else "available logo coverage is sufficient for selective display"
        ),
    }



def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_stem(label: str) -> str:
    value = "".join(char.lower() if char.isalnum() else "_" for char in label)
    return "_".join(part for part in value.split("_") if part) or "asset"


def stage_inputs(
    *,
    asset_dir: Path,
    workspace: Path,
    brand_dir: Path | None,
    appendix_asset_dir: Path | None = None,
) -> tuple[dict[str, Any], list[Path], list[Path], set[str]]:
    """Copy primary and secondary evidence into an isolated designer workspace."""

    meta_dir = asset_dir / "assets" / "meta"
    metadata = load_json(meta_dir / "metadata.json")
    captions = load_json(meta_dir / "captions.json")
    structured = load_json(meta_dir / "structured_elements.json")
    source_lines = load_json(meta_dir / "source_lines.json")
    figures = load_json(meta_dir / "figures.json")
    appendix_source = appendix_asset_dir or asset_dir
    appendix_meta_dir = appendix_source / "assets" / "meta"
    appendix_meta = appendix_meta_dir / "appendix_figures.json"
    appendix_figures = load_json(appendix_meta) if appendix_meta.is_file() else []
    appendix_caption_path = appendix_meta_dir / "captions.json"
    appendix_captions = (
        load_json(appendix_caption_path)
        if appendix_caption_path.is_file()
        else captions
    )

    figure_root = workspace / "assets" / "figures"
    appendix_figure_root = workspace / "assets" / "appendix_figures"
    brand_root = workspace / "assets" / "brand"
    figure_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    image_inputs: list[Path] = []
    appendix_image_inputs: list[Path] = []
    allowed_assets: set[str] = set()
    for index, figure in enumerate(figures, start=1):
        source = asset_dir / "assets" / str(figure["file"])
        if not source.is_file():
            raise FileNotFoundError(f"missing Figure asset: {source}")
        suffix = source.suffix.lower() or ".png"
        name = f"{index:02d}_{_safe_stem(str(figure['label']))}{suffix}"
        relative = f"assets/figures/{name}"
        shutil.copy2(source, figure_root / name)
        staged = workspace / relative
        image_inputs.append(staged)
        allowed_assets.add(relative)
        catalog.append(
            {
                "attachment_index": index,
                "kind": "figure",
                "scope": "body",
                "priority": "primary",
                "label": str(figure["label"]),
                "page": figure.get("page"),
                "width": int(figure["width"]),
                "height": int(figure["height"]),
                "aspect_ratio": round(
                    int(figure["width"]) / max(1, int(figure["height"])), 3
                ),
                "relative_path": relative,
                "caption": captions.get(figure["label"], ""),
                "palette_profile": image_palette_profile(source),
            }
        )
    for index, figure in enumerate(appendix_figures, start=1):
        source = appendix_source / "assets" / str(figure["file"])
        if not source.is_file():
            raise FileNotFoundError(f"missing appendix Figure asset: {source}")
        suffix = source.suffix.lower() or ".png"
        name = f"{index:02d}_{_safe_stem(str(figure['label']))}{suffix}"
        relative = f"assets/appendix_figures/{name}"
        appendix_figure_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, appendix_figure_root / name)
        staged = workspace / relative
        appendix_image_inputs.append(staged)
        allowed_assets.add(relative)
        catalog.append(
            {
                "attachment_index": index,
                "kind": "appendix_figure",
                "scope": "appendix",
                "priority": "secondary",
                "label": str(figure["label"]),
                "page": figure.get("page"),
                "width": int(figure["width"]),
                "height": int(figure["height"]),
                "aspect_ratio": round(
                    int(figure["width"]) / max(1, int(figure["height"])), 3
                ),
                "relative_path": relative,
                "caption": appendix_captions.get(figure["label"], ""),
                "palette_profile": image_palette_profile(source),
            }
        )
    # Neutral Logo/QR assets now travel with the paper bundle. ``brand_dir`` remains
    # an explicit override/addition for old runs and hand-curated marks.
    logo_labels = {
        str(filename): str(label)
        for label, filename in (metadata.get("logos") or {}).items()
        if filename and filename != "missing"
    }
    qr_entries = {
        Path(str(item.get("path") or "")).name: item
        for item in (metadata.get("qr") or [])
        if isinstance(item, dict) and item.get("path")
    }
    bundled_logo_sources: list[
        tuple[Path, str, str, str | None, str]
    ] = []
    logos_dir = asset_dir / "assets" / "logos"
    if logos_dir.is_dir():
        bundled_logo_sources.extend(
            (
                source,
                "logo",
                logo_labels.get(source.name, source.stem),
                None,
                "bundle",
            )
            for source in sorted(logos_dir.iterdir()) if source.is_file()
        )
    logo_policy = _bundled_logo_policy(
        metadata,
        [item[2] for item in bundled_logo_sources],
    )
    brand_exclusions: list[dict[str, str]] = []
    if logo_policy["status"] == "omit_incomplete":
        brand_exclusions.extend(
            {
                "relative_source": str(source.relative_to(asset_dir)),
                "label": label,
                "reason": str(logo_policy["reason"]),
            }
            for source, _, label, _, _ in bundled_logo_sources
        )
        bundled_logo_sources = []

    brand_sources = list(bundled_logo_sources)
    qr_dir = asset_dir / "assets" / "qr"
    if qr_dir.is_dir():
        for source in sorted(qr_dir.iterdir()):
            if not source.is_file():
                continue
            item = qr_entries.get(source.name, {})
            qr_kind, qr_label = canonical_qr_label(
                str(item.get("url") or ""),
                str(item.get("label") or source.stem),
            )
            if isinstance(item, dict):
                item["kind"] = qr_kind
                item["label"] = qr_label
            brand_sources.append(
                (source, "qr", qr_label, item.get("url"), "bundle")
            )
    if brand_dir is not None:
        if not brand_dir.is_dir():
            raise FileNotFoundError(f"brand directory does not exist: {brand_dir}")
        for source in sorted(path for path in brand_dir.iterdir() if path.is_file()):
            kind = "qr" if "qr" in source.stem.lower() else "logo"
            brand_sources.append((source, kind, source.stem, None, "override"))

    used_brand_names: set[str] = set()
    logo_count = 0
    qr_count = 0
    if brand_sources:
        brand_root.mkdir(parents=True, exist_ok=True)
    logo_assets: list[dict[str, str]] = []
    qr_assets: list[dict[str, str]] = []
    for source, brand_kind, label, target_url, origin in brand_sources:
        base_name = f"{brand_kind}_{source.name}"
        name = base_name
        serial = 2
        while name in used_brand_names:
            name = f"{brand_kind}_{serial}_{source.name}"
            serial += 1
        used_brand_names.add(name)
        relative = f"assets/brand/{name}"
        shutil.copy2(source, workspace / relative)
        allowed_assets.add(relative)
        logo_count += brand_kind == "logo"
        qr_count += brand_kind == "qr"
        catalog.append(
            {
                "kind": "brand",
                "brand_kind": brand_kind,
                "label": label,
                "target_url": target_url,
                "origin": origin,
                "relative_path": relative,
                "visual_profile": image_foreground_profile(source),
                "palette_profile": (
                    image_palette_profile(source)
                    if brand_kind == "logo"
                    else {"profile_available": False, "accent_hexes": []}
                ),
            }
        )
        contract_entry = {"relative_path": relative, "label": label}
        if brand_kind == "qr":
            qr_assets.append(contract_entry)
        else:
            logo_assets.append(contract_entry)

    brand_requirements = {
        "available_non_qr": logo_count,
        "available_qr": qr_count,
        "min_non_qr": min(2, logo_count),
        "min_qr": min(1, qr_count),
        "max_non_qr": logo_count,
        "logo_assets": logo_assets,
        "qr_assets": qr_assets,
    }

    context = {
        "metadata": metadata,
        "asset_catalog": catalog,
        "appendix_policy": {
            "available_count": len(appendix_image_inputs),
            "pool_role": "secondary_evidence",
            "admission_rule": (
                "Use only when no body Figure serves the same narrative job or when "
                "the appendix Figure provides uniquely decisive evidence."
            ),
            "palette_rule": (
                "An appendix Figure may influence the poster palette only if admitted."
            ),
        },
        "brand_requirements": brand_requirements,
        "brand_policy": {
            "bundled_logos": logo_policy,
            "excluded_assets": brand_exclusions,
            "qr_labels_are_canonical": True,
        },
        "structured_evidence": {
            "tables": structured.get("tables", []),
            "equations": structured.get("equations", []),
            "algorithms": structured.get("algorithms", []),
        },
        "numbered_paper_lines": [
            {"id": item["id"], "text": item["text"]}
            for item in source_lines
            if str(item.get("text", "")).strip()
        ],
    }
    write_json(workspace / "asset_catalog.json", catalog)
    write_json(workspace / "grounded_context.json", context)
    shutil.copy2(STATE_SCHEMA, workspace / "design_state.schema.json")
    return context, image_inputs, appendix_image_inputs, allowed_assets


def design_plan_from_state(state: dict[str, Any]) -> dict[str, Any]:
    """Map persistent state to the renderer's semantic-plan contract."""

    return {
        "concept": state["concept"],
        "reading_order": state["reading_order"],
        "territories": state["territories"],
        "nodes": state["nodes"],
        "edges": state["edges"],
    }


def bundle_from_workspace(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the model-maintained files as a renderable bundle."""

    missing = [name for name in SOURCE_FILES if not (workspace / name).is_file()]
    if missing:
        raise PosterContractError(f"designer did not create required files: {missing}")
    state = load_json(workspace / "design_state.json")
    critique = load_json(workspace / "critique.json")
    bundle = {
        "html": (workspace / "poster.html").read_text(encoding="utf-8"),
        "css": (workspace / "style.css").read_text(encoding="utf-8"),
        "design_plan": design_plan_from_state(state),
        "rationale": state["concept"],
    }
    return bundle, {"design_state": state, "critique": critique}


def _schema_issues(state: dict[str, Any]) -> list[str]:
    """Validate required schema structure without adding a runtime dependency."""

    issues: list[str] = []
    required = {
        "schema_version", "paper_title", "poster_title", "concept", "stages",
        "territories", "nodes", "edges", "reading_order", "visual_hierarchy",
        "type_scale", "decision_log",
    }
    missing = sorted(required - set(state))
    if missing:
        return [f"design_state is missing required fields: {missing}"]
    if state.get("schema_version") != 1:
        issues.append("design_state schema_version must equal 1")
    for key, minimum, maximum in (
        ("stages", 4, 6), ("territories", 2, 5), ("nodes", 6, 36),
        ("edges", 3, 80), ("reading_order", 5, 36),
    ):
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
    for collection, fields in (
        ("stages", {"id", "role", "title", "purpose", "node_ids"}),
        ("territories", {"id", "title", "purpose", "width_weight", "node_ids"}),
        ("nodes", {"id", "kind", "semantic_role", "stage_id", "summary", "priority"}),
        ("edges", {"from", "to", "relation"}),
    ):
        for index, item in enumerate(state[collection]):
            if not isinstance(item, dict):
                issues.append(f"design_state {collection}[{index}] must be an object")
                continue
            absent = sorted(fields - set(item))
            if absent:
                issues.append(
                    f"design_state {collection}[{index}] misses fields: {absent}"
                )
    if issues:
        return issues
    for collection in ("stages", "territories", "nodes"):
        for item in state[collection]:
            if not identifier.fullmatch(str(item["id"])):
                issues.append(f"invalid {collection} ID: {item['id']}")
    for stage in state["stages"]:
        if stage["role"] not in semantic_roles - {"identity"}:
            issues.append(f"invalid stage role: {stage['role']}")
    for node in state["nodes"]:
        if node["kind"] not in node_kinds:
            issues.append(f"invalid node kind: {node['kind']}")
        if node["semantic_role"] not in semantic_roles:
            issues.append(f"invalid node semantic_role: {node['semantic_role']}")
    hierarchy = state.get("visual_hierarchy")
    if not isinstance(hierarchy, dict) or not {
        "primary_anchor", "secondary_anchors", "rationale"
    } <= set(hierarchy):
        issues.append("visual_hierarchy is incomplete")
    type_scale = state.get("type_scale")
    if not isinstance(type_scale, dict) or not {
        "title_px", "chapter_px", "group_px", "body_px", "caption_px"
    } <= set(type_scale):
        issues.append("type_scale is incomplete")
    return issues


def _normalized_title_text(value: str) -> str:
    """Normalize markup-only wrapping while preserving title wording and case."""

    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return " ".join(unescape(value).split())


def _visible_h1_titles(markup: str) -> list[str]:
    """Return normalized wording from every visible paper-title H1."""

    return [
        _normalized_title_text(fragment)
        for fragment in re.findall(r"<h1\b[^>]*>(.*?)</h1>", markup, flags=re.I | re.S)
    ]


def semantic_contract_issues(
    state: dict[str, Any],
    html: str,
    paper_title: str,
) -> tuple[list[str], list[str]]:
    """Validate generic narrative and claim–evidence invariants."""

    issues = _schema_issues(state)
    warnings: list[str] = []
    if issues:
        return issues, warnings

    nodes = state["nodes"]
    node_map = {item["id"]: item for item in nodes}
    node_ids = set(node_map)
    if len(node_ids) != len(nodes):
        issues.append("design_state node IDs are not unique")

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
    expected_staged = {
        item["id"] for item in nodes if item["semantic_role"] != "identity"
    }
    if set(staged_nodes) != expected_staged or len(staged_nodes) != len(set(staged_nodes)):
        issues.append("non-identity nodes must belong to exactly one semantic stage")

    territory_nodes = [
        node_id
        for territory in state["territories"]
        for node_id in territory["node_ids"]
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
    html_stages = re.findall(r"data-stage-id\s*=\s*['\"]([^'\"]+)['\"]", html)
    if set(html_stages) != set(stage_map) or len(html_stages) != len(set(html_stages)):
        issues.append("HTML must contain every story stage exactly once")
    territory_ids = {item["id"] for item in state["territories"]}
    html_territories = re.findall(
        r"data-territory-id\s*=\s*['\"]([^'\"]+)['\"]", html
    )
    if set(html_territories) != territory_ids:
        issues.append("HTML visual territories disagree with design_state")

    connected: set[str] = set()
    for edge in state["edges"]:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            issues.append(f"edge references unknown nodes: {edge['from']} → {edge['to']}")
        connected.update((edge["from"], edge["to"]))
    orphan_evidence = sorted(
        item["id"]
        for item in nodes
        if item["kind"] in EVIDENCE_KINDS and item["id"] not in connected
    )
    if orphan_evidence:
        issues.append(f"evidence nodes have no semantic correspondence: {orphan_evidence}")

    anchor = state["visual_hierarchy"]["primary_anchor"]
    if anchor not in node_map:
        issues.append(f"primary visual anchor is unknown: {anchor}")
    elif node_map[anchor]["kind"] not in EVIDENCE_KINDS:
        issues.append("primary visual anchor must be a Figure, Table, Equation, or Algorithm")

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


def quality_gate(
    *,
    state: dict[str, Any],
    html: str,
    report: dict[str, Any],
    paper_title: str,
    require_branding: bool | dict[str, Any],
    contract_error: str | None = None,
) -> dict[str, Any]:
    """Combine hard production checks with generic semantic invariants."""

    issues, warnings = semantic_contract_issues(state, html, paper_title)
    semantic_issues = list(issues)
    if contract_error:
        issues.insert(0, contract_error)
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
    text_orientation_issues = report.get("textOrientationIssues") or []
    if text_orientation_issues:
        examples = ", ".join(
            repr(str(item.get("text") or "")[:36])
            for item in text_orientation_issues[:4]
        )
        issues.append(
            f"{len(text_orientation_issues)} informative text element(s) are "
            f"rotated, skewed, or vertically written: {examples}"
        )

    fonts = report.get("fontStats") or {}
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
        item["id"]
        for item in state.get("nodes") or []
        if item.get("semantic_role") == "identity"
    }
    identity_territories = {
        item["id"]
        for item in state.get("territories") or []
        if set(item.get("node_ids") or []) and set(item["node_ids"]) <= identity_nodes
    }
    scientific_territories: list[dict[str, Any]] = []
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
        max(0.0, float(item.get("bottom_gap_ratio") or 0))
        for item in scientific_territories
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
    figure_node_ids = {
        item["id"]
        for item in state.get("nodes") or []
        if item.get("kind") == "figure"
    }
    scientific_images = [
        item
        for item in report.get("scientificImageMetrics") or []
        if item.get("node_id") in figure_node_ids
    ]
    small_scientific_images: list[dict[str, Any]] = []
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
            float(anchor_rect["w"]) * float(anchor_rect["h"])
            / (canvas_width * canvas_height)
        )
        if area_ratio < 0.035:
            issues.append(
                f"primary anchor {anchor} occupies only {area_ratio:.3f} of the canvas"
            )

    far_correspondence: list[dict[str, Any]] = []
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
                {
                    "from": edge.get("from"),
                    "to": edge.get("to"),
                    "distance": round(distance, 3),
                }
            )
    if far_correspondence:
        issues.append(
            f"{len(far_correspondence)} claim–evidence edge(s) are not visually adjacent"
        )

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
            item
            for item in state.get("stages") or []
            if item.get("role") in ROLE_RANK and item.get("role") != "identity"
        ),
        key=lambda item: ROLE_RANK[item["role"]],
    )
    centers = []
    for stage in ordered_stages:
        metric = stage_metrics.get(stage["id"])
        if metric:
            rect = metric["rect"]
            centers.append((stage["role"], float(rect["x"]) + float(rect["w"]) / 2))
    for previous, current in zip(centers, centers[1:]):
        if current[1] + 80 < previous[1]:
            issues.append(
                f"macro story moves left from {previous[0]} to {current[0]}"
            )
    if (
        len(centers) >= 4
        and centers[-1][1] - centers[0][1] < canvas_width * 0.4
    ):
        warnings.append("semantic stages use less than 40% of canvas width progression")

    brand = report.get("brandMetrics") or {}
    brand_padding_intrusions = brand.get("padding_intrusions") or []
    if brand_padding_intrusions:
        maximum = max(
            float(item.get("max_excess") or 0)
            for item in brand_padding_intrusions
        )
        issues.append(
            f"{len(brand_padding_intrusions)} brand mark(s) intrude into the card's "
            f"content padding by up to {maximum:.1f}px"
        )
    if require_branding:
        if isinstance(require_branding, dict):
            min_non_qr = int(require_branding.get("min_non_qr") or 0)
            min_qr = int(require_branding.get("min_qr") or 0)
            raw_max_non_qr = require_branding.get("max_non_qr")
            max_non_qr = (
                int(raw_max_non_qr) if raw_max_non_qr is not None else None
            )
        else:
            min_non_qr, min_qr = 2, 0
            max_non_qr = None
        actual_non_qr = int(brand.get("non_qr_count") or 0)
        actual_qr = int(brand.get("qr_count") or 0)
        if not brand.get("present"):
            issues.append("required brand cluster is missing")
        if actual_non_qr < min_non_qr:
            issues.append(
                f"brand cluster uses {actual_non_qr}/{min_non_qr} required non-QR marks"
            )
        if max_non_qr is not None and actual_non_qr > max_non_qr:
            issues.append(
                f"brand cluster uses {actual_non_qr}/{max_non_qr} permitted non-QR marks"
            )
        if actual_qr < min_qr:
            issues.append(f"brand cluster uses {actual_qr}/{min_qr} required QR marks")
        if min_non_qr and float(brand.get("non_qr_min_width") or 0) + 0.5 < 150:
            issues.append("a non-QR brand mark renders below 150px")
        if actual_non_qr and actual_qr and (
            float(brand.get("qr_max_width") or 0)
            > float(brand.get("non_qr_max_width") or 0)
        ):
            issues.append("QR code is larger than the dominant logo")
        low_contrast = [
            item
            for item in brand.get("images") or []
            if not item.get("is_qr")
            and item.get("contrast_ratio") is not None
            and float(item["contrast_ratio"]) < 1.5
        ]
        if low_contrast:
            issues.append(
                "low-contrast brand mark(s): "
                + ", ".join(
                    str(item.get("src") or item.get("alt")) for item in low_contrast
                )
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
                    issues.append(
                        f"QR asset is not in the grounded brand contract: {source}"
                    )
                    continue
                alt = " ".join(str(item.get("alt") or "").split())
                visible = " ".join(str(item.get("visible_label") or "").split())
                declared = " ".join(
                    str(item.get("data_brand_label") or "").split()
                )
                if declared != expected_label:
                    issues.append(
                        f"QR {source} data-brand-label must be exactly "
                        f"{expected_label!r}"
                    )
                if item.get("data_brand_kind") != "qr":
                    issues.append(
                        f"QR {source} data-brand-kind must be exactly 'qr'"
                    )
                if visible != expected_label:
                    issues.append(
                        f"QR {source} visible label must be exactly {expected_label!r}"
                    )
                if not re.search(rf"\b{re.escape(expected_label)}\b", alt, re.I):
                    issues.append(
                        f"QR {source} alt text must identify it as {expected_label!r}"
                    )
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
                    issues.append(
                        f"Logo asset is not in the grounded brand contract: {source}"
                    )
                    continue
                declared = " ".join(
                    str(item.get("data_brand_label") or "").split()
                )
                if item.get("data_brand_kind") != "logo":
                    issues.append(
                        f"Logo {source} data-brand-kind must be exactly 'logo'"
                    )
                if declared != expected_label:
                    issues.append(
                        f"Logo {source} data-brand-label must be exactly "
                        f"{expected_label!r}"
                    )
                alt = " ".join(str(item.get("alt") or "").split())
                if expected_label.casefold() not in alt.casefold():
                    issues.append(
                        f"Logo {source} alt text must identify {expected_label!r}"
                    )

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
        "geometry": {
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
                    max(
                        0,
                        float((item.get("scroll") or (0, 0))[0])
                        - float((item.get("client") or (0, 0))[0]),
                    )
                    + max(
                        0,
                        float((item.get("scroll") or (0, 0))[1])
                        - float((item.get("client") or (0, 0))[1]),
                    )
                    for item in report.get("hardOverflow") or []
                ),
                3,
            ),
            "territories": [
                {
                    key: item.get(key)
                    for key in (
                        "id", "largest_vertical_void_ratio",
                        "painted_height_ratio", "large_vertical_gaps",
                        "painted_bottom_gap_ratio", "bottom_gap_ratio",
                    )
                }
                for item in report.get("territoryMetrics") or []
                if item.get("id") not in identity_territories
            ],
            "branding": report.get("brandMetrics") or {},
        },
    }


def source_hashes(workspace: Path) -> dict[str, str]:
    """Hash maintained design files for incremental-edit evidence."""

    result: dict[str, str] = {}
    for name in SOURCE_FILES:
        path = workspace / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def source_diff(previous: Path, current: Path, output: Path) -> None:
    """Save a unified source diff between two rendered checkpoints."""

    lines: list[str] = []
    for name in SOURCE_FILES:
        old = previous / name
        new = current / name
        old_lines = old.read_text(encoding="utf-8").splitlines(True) if old.is_file() else []
        new_lines = new.read_text(encoding="utf-8").splitlines(True) if new.is_file() else []
        lines.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"{previous.name}/{name}",
                tofile=f"{current.name}/{name}",
            )
        )
    output.write_text("".join(lines), encoding="utf-8")


def render_checkpoint(
    *,
    workspace: Path,
    checkpoint_dir: Path,
    allowed_assets: set[str],
    paper_title: str,
    require_branding: bool | dict[str, Any],
    canvas_width: int = 2400,
    canvas_height: int = 1800,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Snapshot current sources, render them, and publish feedback to the workspace."""

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        source = workspace / name
        if source.is_file():
            shutil.copy2(source, checkpoint_dir / name)
    if (workspace / "assets").is_dir():
        shutil.copytree(workspace / "assets", checkpoint_dir / "assets")

    report: dict[str, Any] = {"status": "contract_error"}
    contract_error: str | None = None
    state: dict[str, Any] = {}
    html = ""
    try:
        bundle, maintained = bundle_from_workspace(workspace)
        state = maintained["design_state"]
        html = bundle["html"]
        validate_bundle(
            bundle,
            allowed_assets,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        report = render_bundle(
            bundle,
            checkpoint_dir,
            export_pdf=False,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
    except (PosterContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        contract_error = f"source contract error: {exc}"
        if not state and (workspace / "design_state.json").is_file():
            try:
                state = load_json(workspace / "design_state.json")
            except (OSError, json.JSONDecodeError):
                state = {}
        if not html and (workspace / "poster.html").is_file():
            html = (workspace / "poster.html").read_text(encoding="utf-8")
        write_json(checkpoint_dir / "validation.json", report)

    if state:
        gate = quality_gate(
            state=state,
            html=html,
            report=report,
            paper_title=paper_title,
            require_branding=require_branding,
            contract_error=contract_error,
        )
    else:
        gate = {
            "schema_version": 1,
            "accepted": False,
            "issues": [contract_error or "design_state.json is unreadable"],
            "semantic_warnings": [],
        }
    write_json(checkpoint_dir / "quality_gate.json", gate)
    write_json(checkpoint_dir / "source_hashes.json", source_hashes(workspace))
    write_json(workspace / "browser_diagnostics.json", report)
    write_json(workspace / "quality_gate.json", gate)
    return report, gate
