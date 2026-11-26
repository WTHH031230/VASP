"""Standalone w/o Semantic Verification (Geometry-Only Verification) implementation.

Cleanly decouples Semantic Verification (SV) from Ordinary Geometric Verification:
  1. Keeps 100% of Visual-First (VF) asset admission & sizing (contact sheets, detail
     sheet, figure size floor >= 12% W x 6.5% H, largest figure >= 3.5% canvas area)
     and SemParser assets.
  2. Keeps 100% of Ordinary Geometric Verification in the refinement loop (overflow,
     out-of-canvas, empty shells, territory vertical void <= 0.075, painted bottom gap
     <= 0.035, bottom baseline spread <= 0.025, title-only sidecars, flex margin spacers,
     typography floors, checkpoint ranking by geometry, early stop, and deterministic
     small-overflow padding fit).
  3. Completely removes Semantic Verification (SV):
     - No design_state.json or critique.json declaration (HTML + CSS + summary only).
     - No claim-evidence graph planning, no orphan-evidence check, no claim-evidence
       spatial distance (delta <= 0.42) check, no stage role / left-to-right stage check.
     - Initial composition decouples text and figure tracks (high-level plain-text section
       summaries without figure walkthroughs or inline keyword highlighting; figures placed
       by geometric column fit with brief identifying labels).
     - Revision turns perform strictly geometry-only repairs (CSS sizing, spacing, and
       reflow) without rewriting prose to ground or explain figures.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from src.viscomposer.core import host_prompts as HP
from src.viscomposer.core.prompts import canvas_prompt_text
from src.viscomposer.core.render_html import (
    PosterContractError,
    validate_source_safety,
)
from src.viscomposer.core.workbench import _normalized_title_text, _visible_h1_titles

SOURCE_FILES = ("poster.html", "style.css")


# --- 1. Response Schemas (HTML + CSS only; no design_state or critique) ----------------

def response_schemas(max_revisions: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    initial = {
        "type": "object",
        "additionalProperties": False,
        "required": ["poster_html", "style_css", "summary"],
        "properties": {
            "poster_html": {"type": "string", "minLength": 200},
            "style_css": {"type": "string", "minLength": 80},
            "summary": {"type": "string", "minLength": 20},
        },
    }
    revision = {
        "type": "object",
        "additionalProperties": False,
        "required": ["file_edits", "summary"],
        "properties": {
            "file_edits": {
                "type": "array",
                "minItems": 0,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["file", "old_text", "new_text", "reason"],
                    "properties": {
                        "file": {"type": "string", "enum": ["poster.html", "style.css"]},
                        "old_text": {"type": "string", "minLength": 1},
                        "new_text": {"type": "string"},
                        "reason": {"type": "string", "minLength": 3},
                    },
                },
            },
            "summary": {"type": "string", "minLength": 20},
        },
    }
    return initial, revision


# --- 2. Pre-render DOM Geometry Hook Injector (Pure Stdlib) ----------------------------

_VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


class _DOMNode:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]], parent: _DOMNode | None = None) -> None:
        self.tag = tag.lower()
        self.attrs: dict[str, str | None] = {k.lower(): v for k, v in attrs}
        self.parent = parent
        self.children: list[_DOMNode | str] = []

    @property
    def classes(self) -> list[str]:
        return (self.attrs.get("class") or "").lower().split()

    def element_children(self) -> list[_DOMNode]:
        return [c for c in self.children if isinstance(c, _DOMNode)]

    def iter_elements(self) -> list[_DOMNode]:
        out: list[_DOMNode] = []
        stack = [self]
        while stack:
            curr = stack.pop(0)
            out.append(curr)
            stack.extend(curr.element_children())
        return out

    def has_ancestor_attr(self, key: str, val: str | None = None) -> bool:
        p = self.parent
        while p is not None:
            if key in p.attrs and (val is None or p.attrs[key] == val):
                return True
            p = p.parent
        return False

    def to_html(self) -> str:
        if self.tag == "#root":
            return "".join(c.to_html() if isinstance(c, _DOMNode) else c for c in self.children)
        attr_parts = []
        for k, v in self.attrs.items():
            if v is None:
                attr_parts.append(k)
            else:
                escaped = str(v).replace("&", "&amp;").replace('"', "&quot;")
                attr_parts.append(f'{k}="{escaped}"')
        attr_str = (" " + " ".join(attr_parts)) if attr_parts else ""
        if self.tag in _VOID_ELEMENTS:
            return f"<{self.tag}{attr_str}>"
        inner = "".join(c.to_html() if isinstance(c, _DOMNode) else c for c in self.children)
        return f"<{self.tag}{attr_str}>{inner}</{self.tag}>"


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = _DOMNode("#root", [])
        self.curr = self.root

    def handle_decl(self, decl: str) -> None:
        self.curr.children.append(f"<!{decl}>")

    def handle_comment(self, data: str) -> None:
        self.curr.children.append(f"<!--{data}-->")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _DOMNode(tag, attrs, self.curr)
        self.curr.children.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _DOMNode(tag, attrs, self.curr)
        self.curr.children.append(node)
        if tag.lower() not in _VOID_ELEMENTS:
            self.curr = node

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in _VOID_ELEMENTS:
            return
        p = self.curr
        while p is not self.root and p.tag != t:
            p = p.parent or self.root
        if p is not self.root:
            self.curr = p.parent or self.root

    def handle_data(self, data: str) -> None:
        self.curr.children.append(data)

    def handle_entityref(self, name: str) -> None:
        self.curr.children.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.curr.children.append(f"&#{name};")


def ensure_geometry_hooks(html: str) -> str:
    """Ensure render_bundle can measure territoryMetrics and emptyShells without design_state."""
    try:
        parser = _TreeBuilder()
        parser.feed(html)
        root = parser.root
    except Exception:
        return html

    all_els = root.iter_elements()
    poster = next((el for el in all_els if "poster" in el.classes), None)
    if poster is None:
        poster = next((el for el in all_els if el.tag == "body"), root)

    poster_els = poster.iter_elements()

    # 1. Ensure branding hook if a qr/logo cluster exists in header/masthead
    if not any(el.attrs.get("data-node-id") == "branding" for el in poster_els):
        for el in poster_els:
            if any(c in el.classes for c in ("branding", "brand-cluster", "brand-block", "masthead-brand", "header-right", "qr-cluster", "brand-zone")):
                el.attrs["data-node-id"] = "branding"
                break

    # 2. Ensure [data-territory-id] on main content columns/territories
    territories = [el for el in poster_els if "data-territory-id" in el.attrs]
    if not territories:
        candidates: list[_DOMNode] = []
        for cls_key in ("territory", "column", "col", "poster-col", "grid-col"):
            found = [
                el for el in poster_els
                if cls_key in el.classes
                and el.tag in ("div", "section", "article", "aside")
                and not el.has_ancestor_attr("data-node-id", "branding")
                and not any(h in el.classes for h in ("header", "masthead", "footer"))
            ]
            if 2 <= len(found) <= 6:
                candidates = found
                break

        if not candidates:
            # Check main / poster-body / content-grid children
            for wrapper in poster_els:
                if wrapper.tag == "main" or any(c in wrapper.classes for c in ("poster-body", "content-grid", "columns", "territories", "main-grid", "body-grid")):
                    children = [
                        c for c in wrapper.element_children()
                        if c.tag in ("div", "section", "article", "aside")
                        and not any(h in c.classes for h in ("header", "masthead", "footer", "banner"))
                    ]
                    if 2 <= len(children) <= 6:
                        candidates = children
                        break

        if not candidates:
            direct = [
                c for c in poster.element_children()
                if c.tag in ("div", "section", "main", "article")
                and c.tag not in ("header", "footer")
                and not any(h in c.classes for h in ("header", "masthead", "banner", "footer", "title-bar"))
            ]
            if len(direct) == 1:
                inner = [
                    c for c in direct[0].element_children()
                    if c.tag in ("div", "section", "article", "aside")
                    and not any(h in c.classes for h in ("header", "masthead", "footer"))
                ]
                if 2 <= len(inner) <= 6:
                    candidates = inner
            elif 2 <= len(direct) <= 6:
                candidates = direct

        for idx, col in enumerate(candidates, 1):
            col.attrs["data-territory-id"] = f"col-{idx}"
        territories = candidates

    # 3. Ensure [data-node-id] on content blocks inside territories so owned rects are non-empty
    node_counter = 1
    for terr in territories:
        terr_els = terr.iter_elements()
        owned = [el for el in terr_els if "data-node-id" in el.attrs and el is not terr]
        if not owned:
            children = [
                c for c in terr.element_children()
                if c.tag in ("section", "article", "div", "figure", "table")
            ]
            if not children:
                children = [terr]
            for child in children:
                if "data-node-id" not in child.attrs:
                    child.attrs["data-node-id"] = f"box-{node_counter}"
                    node_counter += 1

    for el in poster_els:
        if (el.tag == "figure" or any(c in el.classes for c in ("card", "panel", "section-card", "block"))) and "data-node-id" not in el.attrs:
            if not el.has_ancestor_attr("data-node-id", "branding"):
                el.attrs["data-node-id"] = f"box-{node_counter}"
                node_counter += 1

    return root.to_html()


# --- 3. Source Validation & Bundle Construction ---------------------------------------

def bundle_from_workspace(workspace: Path) -> dict[str, Any]:
    html_path = workspace / "poster.html"
    css_path = workspace / "style.css"
    if not html_path.is_file() or not css_path.is_file():
        raise PosterContractError("workspace must contain poster.html and style.css")
    raw_html = html_path.read_text(encoding="utf-8")
    tagged_html = ensure_geometry_hooks(raw_html)
    css = css_path.read_text(encoding="utf-8")
    return {
        "html": tagged_html,
        "css": css,
        "design_plan": {
            "concept": "",
            "reading_order": [],
            "territories": [],
            "nodes": [],
            "edges": [],
        },
    }


def validate_bundle(bundle: dict[str, Any], allowed_assets: set[str], *,
                    canvas_width: int, canvas_height: int) -> None:
    validate_source_safety(bundle, allowed_assets, canvas_width=canvas_width, canvas_height=canvas_height)


def contract_error(workspace: Path, allowed_assets: set[str], *,
                   canvas_width: int, canvas_height: int) -> tuple[str | None, str]:
    html = ""
    try:
        if (workspace / "poster.html").is_file():
            html = (workspace / "poster.html").read_text(encoding="utf-8")
        bundle = bundle_from_workspace(workspace)
        validate_bundle(bundle, allowed_assets, canvas_width=canvas_width, canvas_height=canvas_height)
        return None, bundle["html"]
    except (PosterContractError, KeyError, TypeError, ValueError) as exc:
        return f"source contract error: {exc}", html


# --- 4. Pure Geometry Quality Gate & Loop Control -------------------------------------

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

def quality_gate(*, html: str, report: dict[str, Any], paper_title: str,
                 require_branding: bool | dict[str, Any],
                 contract_err: str | None = None) -> dict[str, Any]:
    """Evaluate ONLY geometric, typographic, whitespace, and figure-scale predicates."""
    issues: list[str] = []
    warnings: list[str] = []
    if contract_err:
        issues.append(contract_err)

    # Title verbatim check (basic identity preservation)
    expected_title = _normalized_title_text(paper_title)
    visible_titles = _visible_h1_titles(html)
    if len(visible_titles) != 1:
        issues.append("HTML must contain exactly one visible H1 paper title")
    elif visible_titles[0] != expected_title:
        issues.append("visible H1 must reproduce the extracted paper title verbatim")

    # Browser box & overflow geometry
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
        issues.append(f"{len(report['flexMarginSpacers'])} large flex margin spacer(s) split scientific content")
    if report.get("titleOnlySidecars"):
        issues.append(f"{len(report['titleOnlySidecars'])} title-only sidecar(s) leave a locally empty grid column")

    text_orientation_issues = report.get("textOrientationIssues") or []
    if text_orientation_issues:
        examples = ", ".join(repr(str(item.get("text") or "")[:36]) for item in text_orientation_issues[:4])
        issues.append(f"{len(text_orientation_issues)} informative text element(s) are rotated, skewed, or vertically written: {examples}")

    # Typography floors
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
    if chapters and min(float(item.get("h2_font_px") or 0) for item in chapters) < 30:
        issues.append("at least one chapter heading is below 30px")

    # Territory whitespace & baseline alignment
    scientific_territories: list[dict[str, Any]] = []
    for item in report.get("territoryMetrics") or []:
        tid = str(item.get("id") or "")
        if tid in ("masthead", "header", "branding", "identity"):
            continue
        scientific_territories.append(item)
        if float(item.get("largest_vertical_void_ratio") or 0) > 0.075:
            issues.append(f"territory {tid} has a {float(item['largest_vertical_void_ratio']):.3f} vertical void")
        if float(item.get("painted_height_ratio") or 1) < 0.47:
            warnings.append(f"territory {tid} has low painted coverage ({float(item['painted_height_ratio']):.3f})")
        if float(item.get("painted_bottom_gap_ratio") or 0) > 0.035:
            issues.append(f"territory {tid} leaves a {float(item['painted_bottom_gap_ratio']):.3f} painted bottom void")

    bottom_gaps = [max(0.0, float(item.get("bottom_gap_ratio") or 0)) for item in scientific_territories]
    if len(bottom_gaps) >= 2 and max(bottom_gaps) - min(bottom_gaps) > 0.025:
        issues.append(f"scientific territories do not share one bottom baseline (painted-gap spread {max(bottom_gaps) - min(bottom_gaps):.3f})")

    # Visual-First figure size floors (kept active because VF is ON in w/o SV)
    canvas = report.get("canvas") or [2544, 1696]
    canvas_width = max(1.0, float(canvas[0]))
    canvas_height = max(1.0, float(canvas[1]))
    raw_images = report.get("scientificImageMetrics") or []
    scientific_images = [
        img for img in raw_images
        if not any(k in str(img.get("src") or "").lower() for k in ("logo", "qr_", "brand"))
    ]
    small_scientific_images: list[dict[str, Any]] = []
    min_figure_width = canvas_width * 0.12
    min_figure_height = canvas_height * 0.065
    max_area_ratio = 0.0
    for item in scientific_images:
        effective = item.get("effective") or {}
        width = float(effective.get("w") or 0)
        height = float(effective.get("h") or 0)
        area_ratio = (width * height) / (canvas_width * canvas_height)
        if area_ratio > max_area_ratio:
            max_area_ratio = area_ratio
        if width + 0.5 < min_figure_width or height + 0.5 < min_figure_height:
            small_scientific_images.append(item)
            issues.append(
                f"figure {item.get('src')} is thumbnail-sized ({width:.0f}x{height:.0f}px effective); enlarge it or omit it"
            )
    if scientific_images and max_area_ratio < 0.035:
        issues.append(f"largest scientific figure occupies only {max_area_ratio:.3f} of the canvas (require >= 0.035)")

    # Brand geometry
    brand = report.get("brandMetrics") or {}
    brand_padding_intrusions = brand.get("padding_intrusions") or []
    if require_branding:
        _brand_issues(issues, brand, brand_padding_intrusions, require_branding)

    overflow_excess_px = round(
        sum(
            max(0.0, float(item["scroll"][0]) - float(item["client"][0]))
            + max(0.0, float(item["scroll"][1]) - float(item["client"][1]))
            for item in (report.get("hardOverflow") or [])
            if isinstance(item.get("scroll"), list) and isinstance(item.get("client"), list)
        ),
        1,
    )

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
        "overflow_excess_px": overflow_excess_px,
        "territories": scientific_territories,
        "branding": brand,
    }

    return {
        "schema_version": 1,
        "accepted": len(issues) == 0,
        "issues": issues,
        "semantic_issues": [],
        "semantic_warnings": warnings,
        "font_stats": fonts,
        "far_correspondence": [],
        "stage_centers": [],
        "geometry": geometry,
    }


def gate_score(gate: dict[str, Any]) -> tuple[float, ...]:
    """Rank checkpoints purely by geometric and typographic quality."""
    geometry = gate.get("geometry") or {}
    fonts = gate.get("font_stats") or {}
    territories = geometry.get("territories") or []
    max_void = max(
        (float(item.get("largest_vertical_void_ratio") or 0) for item in territories), default=0.0
    )
    outside_or_missing = len(geometry.get("outside") or []) + len(geometry.get("missing_images") or [])
    bottom_gaps = [max(0.0, float(item.get("painted_bottom_gap_ratio") or 0)) for item in territories]
    dom_bottom_gaps = [max(0.0, float(item.get("bottom_gap_ratio") or 0)) for item in territories]
    bottom_gap_spread = (
        max(dom_bottom_gaps) - min(dom_bottom_gaps) if len(dom_bottom_gaps) >= 2 else 0.0
    )
    return (
        1.0 if gate.get("accepted") else 0.0,
        -float(outside_or_missing),
        -float(geometry.get("overflow_excess_px") or 0),
        -float(len(geometry.get("hard_overflow") or [])),
        -float(len(geometry.get("empty_shells") or [])),
        -float(len(geometry.get("small_scientific_images") or [])),
        -float(len(geometry.get("flex_margin_spacers") or [])),
        -float(len(geometry.get("title_only_sidecars") or [])),
        -max(bottom_gaps, default=0.0),
        -bottom_gap_spread,
        -max_void,
        -max(0.0, 64.0 - float(fonts.get("title_px") or 0)),
        -max(0.0, 19.0 - float(fonts.get("median_px") or 0)),
        -max(0.0, 16.5 - float(fonts.get("p10_px") or 0)),
        -float(len(gate.get("issues") or [])),
    )


def select_checkpoint(checkpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in checkpoints if item.get("eligible")]
    if not eligible:
        return None
    return max(enumerate(eligible), key=lambda pair: (gate_score(pair[1]["gate"]), pair[0]))[1]


def should_stop(gate: dict[str, Any]) -> bool:
    return bool(gate.get("accepted"))


def fit_allowed(gate: dict[str, Any]) -> bool:
    geometry = gate.get("geometry") or {}
    return (
        not geometry.get("outside")
        and not geometry.get("missing_images")
        and not geometry.get("empty_shells")
    )


# --- 5. Prompts (Visual-First + Geometry-Only; NO Semantic Claim-Evidence Coupling) ----

WO_SV_PRINCIPLES = """You are the persistent lead layout designer for a research poster. Work in one continuing source workspace: place the selected visual figures, lay out the poster sections, render, inspect the geometric measurements, and edit the source incrementally.

The poster is a {canvas_width}×{canvas_height} landscape canvas. Use a coherent global visual system: one type scale, one spacing scale, and a restrained color palette derived from the supplied scientific Figures. Asset-catalog palette_profile.accent_hexes are deterministic candidates sampled from those assets: select a small compatible subset and adapt their lightness for accessible background/text/accent roles. Do not copy every sampled color. Preserve one poster's chosen palette across revisions unless color contrast is broken. Avoid dashboard-card walls, dark full-width banners, and gratuitous gradients.

Aim for physical poster readability: title >= 64px, chapter headings >= 30px, group headings >= 24px, body copy targeting 20–24px (median >= 19px, 10th percentile >= 16.5px, and at most four visible text leaves below 17px), captions >= 16px. The paper title is immutable identity copy. Treat both axes as explicit pixel budgets. Every admitted scientific Figure must be displayed at a legible scale (at least 12% of canvas width and 6.5% of canvas height, with the largest Figure occupying at least 3.5% of the canvas area); never keep a Figure as a tiny decorative thumbnail. Give every displayed Figure a brief 3–6 word identifying caption label (`<figcaption>`) immediately below the image.

Use whitespace to separate columns cleanly, never as accidental leftover holes. Treat the bottom of the main columns as one shared baseline (painted bottom void <= 3.5% and spread across columns <= 2.5%). No internal vertical void inside any column may exceed 7.5% of the column height. All informative text must follow normal horizontal left-to-right reading (no rotated or vertical text). Size important non-QR logos to be clearly recognizable and keep QR codes compact in the header branding cluster (`data-node-id="branding"`). Use only the supplied local paper and brand assets. Do not invent scientific facts."""


WO_SV_PROTOCOL = """This harness gives you no filesystem, shell, or web tools. Deliver each turn only by calling the one submit tool offered for that turn; its arguments are the structured response.
On the first turn, return `poster_html`, `style_css`, and `summary`.
On revision turns, return exact `old_text` -> `new_text` replacements in `file_edits` (at most 16 edits) and `summary`.
HTML must contain one `.poster` root element. Mark each main layout column with `data-territory-id="col-1"` (etc.), each section/figure card with `data-node-id="box-1"` (etc.), and the header logo/QR cluster with `data-node-id="branding"` so the browser geometry engine can measure column voids and box overflows."""


WO_SV_INITIAL_INSTRUCTIONS = """Follow GROUNDED PAPER PACKAGE.visual_input_order to inspect the supplied figure contact sheets and brand assets. Create a finished first draft on the {canvas_width}×{canvas_height} canvas.

DECOUPLED CONTENT & LAYOUT WORKFLOW (NO CLAIM–EVIDENCE SEMANTIC VERIFICATION):
1. Independent Text & Visual Tracks:
   - Do NOT construct a claim–evidence graph or bind text paragraphs to specific figures.
   - Organize the text into 4 to 5 high-level overview sections (e.g., Background, Overview, Approach, Discussion/Summary). In each section, write only 2 to 3 brief, high-level summary bullets or short paragraphs capturing general paper themes from the abstract/introduction.
   - Keep all section prose strictly high-level and self-contained: do NOT walk through, explain, or reference the internal sub-modules, architecture blocks, curves, or specific numerical table/chart comparisons shown inside the displayed Figures.
   - Render body bullets in clean, uniform plain text without inline `<strong>` or `<b>` keyword highlighting.
2. Geometric Figure Placement:
   - Select 3 to 5 clear Figures from the catalog and size them generously so each Figure meets the minimum size floor (>= 306×111 px) and the primary Figure occupies >= 3.5% of the canvas.
   - Place Figures into whichever column track best balances column heights and fills vertical space—even if a Figure sits in a separate visual column or in a section whose text discusses a different high-level aspect of the paper.
   - Attach only a short, generic 3–6 word title label in `<figcaption>` beneath each Figure (e.g., "Figure 1: System Overview" or "Figure 2: Evaluation"), without a takeaway sentence.
3. Static Layout Arithmetic:
   - Before returning, check that column heights align at the bottom baseline without vertical voids (> 7.5%) or container overflow, and that the single visible `<h1>` reproduces `GROUNDED PAPER PACKAGE.metadata.title` verbatim."""


WO_SV_REVISION_INSTRUCTIONS = """GEOMETRY-ONLY REVISION TURN (model round {round}).
The first attached image is the current render from this same session; the second, when present, is a labelled detail sheet of the rendered Figures. Inspect them together with the CONTROLLER STATE below.

STRICT GEOMETRY-ONLY REPAIR POLICY:
1. Repair ONLY geometric, typographic, and whitespace defects reported in CONTROLLER STATE:
   - Eliminate any `hard_overflow`, `outside` elements, or `empty_shells`.
   - Eliminate internal vertical voids (`largest_vertical_void_ratio > 0.075`) and bottom baseline gaps (`painted_bottom_gap_ratio > 0.035` or `bottom_gap_spread > 0.025`) by adjusting CSS flex/grid track sizing, figure display dimensions, container heights, padding, or margins.
   - Fix any font-size floor violations (`title_px >= 64`, `median_px >= 19`, `p10_px >= 16.5`, `below_17_count <= 4`, chapter `h2 >= 30`) or undersized figures (`small_scientific_images`).
2. Do NOT perform semantic or content revisions:
   - Do NOT rewrite, expand, or edit section body text to explain or reference the Figures.
   - Do NOT add inline `<strong>` or bold keyword emphasis to body text.
   - Do NOT move Figures between sections for narrative/semantic alignment; adjust only geometric sizing, spacing, and CSS layout properties so all geometric checks pass.
Return at most 16 exact, minimal `old_text` -> `new_text` replacements in `file_edits`."""


def initial_images(context: dict[str, Any]) -> list[Path]:
    return [Path(item["path"]) for item in (context.get("visual_input_order") or []) if item.get("path")]


def revision_images(render: Path, detail_sheet: Path | None) -> list[Path]:
    imgs = [render] if render.is_file() else []
    if detail_sheet is not None and detail_sheet.is_file():
        imgs.append(detail_sheet)
    return imgs


def controller_state(checkpoint: int, gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "accepted": gate.get("accepted", False),
        "deterministic_issues": gate.get("issues", []),
        "font_stats": gate.get("font_stats", {}),
        "geometry": gate.get("geometry", {}),
    }


def build_initial_prompt(context: dict[str, Any], *, canvas_width: int, canvas_height: int) -> str:
    parts = [
        canvas_prompt_text(WO_SV_PRINCIPLES, canvas_width, canvas_height),
        HP.canvas_host_contract(canvas_width, canvas_height),
        WO_SV_PROTOCOL,
        WO_SV_INITIAL_INSTRUCTIONS.format(canvas_width=canvas_width, canvas_height=canvas_height),
        "GROUNDED PAPER PACKAGE\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":")),
    ]
    return "\n\n".join(parts)


def build_revision_prompt(*, checkpoint: int, gate: dict[str, Any], final_checkpoint: bool,
                          canvas_width: int, canvas_height: int) -> str:
    round_no = checkpoint + 1
    parts = [
        canvas_prompt_text(WO_SV_PRINCIPLES, canvas_width, canvas_height),
        HP.canvas_host_contract(canvas_width, canvas_height),
        WO_SV_PROTOCOL,
        WO_SV_REVISION_INSTRUCTIONS.format(round=round_no),
        "CONTROLLER STATE\n" + json.dumps(controller_state(checkpoint, gate), ensure_ascii=False, indent=2),
    ]
    return "\n\n".join(parts)
