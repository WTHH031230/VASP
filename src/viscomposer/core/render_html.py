"""Validate and render model-authored poster HTML in an offline browser."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from src.viscomposer.core.visual_assets import (
    contrast_ratio,
    image_foreground_profile,
)


DEFAULT_CANVAS_WIDTH = 2400
DEFAULT_CANVAS_HEIGHT = 1800
PIXELS_PER_PRINT_INCH = 50
PRINT_SCALE = 1.92


class PosterContractError(ValueError):
    """Raised when generated source violates the harness contract."""


def _node_ids_in_html(html: str) -> set[str]:
    return set(re.findall(r"data-node-id\s*=\s*['\"]([^'\"]+)['\"]", html))


def validate_source_safety(
    bundle: dict[str, Any],
    allowed_assets: set[str],
    *,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> None:
    """Fail closed on unsafe HTML/CSS, unknown assets, or a wrong canvas."""

    html = str(bundle.get("html", ""))
    css = str(bundle.get("css", ""))
    lowered = (html + "\n" + css).lower()
    forbidden = (
        "<script", "<iframe", "<object", "<embed", "javascript:",
        "@import", "url(http://", "url(https://", "onload=", "onclick=",
    )
    found = [token for token in forbidden if token in lowered]
    if found:
        raise PosterContractError(f"forbidden generated source token(s): {found}")
    if "class=\"poster\"" not in html and "class='poster'" not in html:
        raise PosterContractError("HTML must contain one .poster root")
    if "style.css" not in html:
        raise PosterContractError("HTML must link style.css")
    width_token = f"{int(canvas_width)}px"
    height_token = f"{int(canvas_height)}px"
    if ".poster" not in css or width_token not in css or height_token not in css:
        raise PosterContractError(
            "CSS must define a "
            f"{int(canvas_width)}x{int(canvas_height)} .poster canvas"
        )

    local_refs = set(
        re.findall(r"(?:src|href)\s*=\s*['\"](assets/[^'\"]+)['\"]", html)
    )
    unknown_assets = sorted(local_refs - allowed_assets)
    if unknown_assets:
        raise PosterContractError(f"unknown staged asset references: {unknown_assets}")
    external_refs = re.findall(r"(?:src|href)\s*=\s*['\"]((?:https?:|//)[^'\"]+)", html)
    if external_refs:
        raise PosterContractError(f"external resources are not allowed: {external_refs}")


def validate_bundle(
    bundle: dict[str, Any],
    allowed_assets: set[str],
    *,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> None:
    """Fail closed on unsafe HTML/CSS and inconsistent correspondence graphs."""

    validate_source_safety(
        bundle,
        allowed_assets,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
    )
    html = str(bundle.get("html", ""))
    plan = bundle.get("design_plan") or {}
    nodes = plan.get("nodes") or []
    node_ids = [str(item.get("id", "")) for item in nodes]
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise PosterContractError("design_plan node IDs must be non-empty and unique")
    known = set(node_ids)
    html_nodes = _node_ids_in_html(html)
    missing_html = sorted(known - html_nodes)
    if missing_html:
        raise PosterContractError(f"plan nodes missing from HTML: {missing_html}")

    territories = plan.get("territories") or []
    territory_ids = [str(item.get("id", "")) for item in territories]
    if not territory_ids or len(territory_ids) != len(set(territory_ids)):
        raise PosterContractError("territory IDs must be non-empty and unique")
    html_territories = set(
        re.findall(r"data-territory-id\s*=\s*['\"]([^'\"]+)['\"]", html)
    )
    missing_territories = sorted(set(territory_ids) - html_territories)
    if missing_territories:
        raise PosterContractError(
            f"plan territories missing from HTML: {missing_territories}"
        )
    for territory in territories:
        unknown_nodes = sorted(set(map(str, territory.get("node_ids") or [])) - known)
        if unknown_nodes:
            raise PosterContractError(
                f"territory {territory.get('id')} references unknown nodes: {unknown_nodes}"
            )

    editorial = bundle.get("editorial_plan")
    if editorial is not None:
        groups = editorial.get("major_groups") or []
        group_ids = [str(item.get("id", "")) for item in groups]
        if not 5 <= len(group_ids) <= 9 or len(group_ids) != len(set(group_ids)):
            raise PosterContractError(
                "editorial plan must contain 5-9 uniquely named visual groups"
            )
        anchors = [
            str(editorial.get("primary_anchor", "")),
            *map(str, editorial.get("secondary_anchors") or []),
        ]
        unknown_anchors = sorted(set(anchors) - set(group_ids))
        if unknown_anchors:
            raise PosterContractError(
                f"editorial anchors reference unknown visual groups: {unknown_anchors}"
            )
        html_groups = set(
            re.findall(r"data-visual-group\s*=\s*['\"]([^'\"]+)['\"]", html)
        )
        unknown_html_groups = sorted(html_groups - set(group_ids))
        if unknown_html_groups:
            raise PosterContractError(
                "HTML references unknown visual groups: "
                f"{unknown_html_groups}"
            )
        for group in groups:
            unknown_nodes = sorted(set(map(str, group.get("node_ids") or [])) - known)
            if unknown_nodes:
                raise PosterContractError(
                    f"visual group {group.get('id')} references unknown nodes: "
                    f"{unknown_nodes}"
                )

    reading_order = [str(value) for value in plan.get("reading_order") or []]
    unknown_order = sorted(set(reading_order) - known)
    if unknown_order:
        raise PosterContractError(f"reading_order references unknown nodes: {unknown_order}")
    edges = plan.get("edges") or []
    linked: set[str] = set()
    for edge in edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source not in known or target not in known:
            raise PosterContractError(f"edge references unknown nodes: {source} -> {target}")
        linked.update((source, target))
    evidence_kinds = {"figure", "table", "equation", "algorithm"}
    orphans = sorted(
        str(item["id"])
        for item in nodes
        if item.get("kind") in evidence_kinds and str(item["id"]) not in linked
    )
    if orphans:
        raise PosterContractError(f"orphan evidence nodes: {orphans}")


def write_bundle_sources(bundle: dict[str, Any], target_dir: Path) -> None:
    """Persist editable generated source and its semantic plan."""

    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "poster.html").write_text(str(bundle["html"]), encoding="utf-8")
    (target_dir / "style.css").write_text(str(bundle["css"]), encoding="utf-8")
    (target_dir / "design_plan.json").write_text(
        json.dumps(bundle["design_plan"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_bundle(
    bundle: dict[str, Any],
    target_dir: Path,
    *,
    export_pdf: bool = True,
    canvas_width: int = DEFAULT_CANVAS_WIDTH,
    canvas_height: int = DEFAULT_CANVAS_HEIGHT,
) -> dict[str, Any]:
    """Render PNG/PDF and return measured geometry/correspondence diagnostics."""

    if canvas_width < 1 or canvas_height < 1:
        raise ValueError("canvas dimensions must be positive")

    write_bundle_sources(bundle, target_dir)
    html_path = target_dir / "poster.html"
    png_path = target_dir / "poster.png"
    pdf_path = target_dir / "poster.pdf"
    edges = bundle["design_plan"]["edges"]
    major_groups = (bundle.get("editorial_plan") or {}).get("major_groups") or []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        context = browser.new_context(
            viewport={"width": int(canvas_width), "height": int(canvas_height)},
            device_scale_factor=1,
            java_script_enabled=False,
        )
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.url.startswith(("http://", "https://"))
            else route.continue_(),
        )
        page.goto(html_path.as_uri(), wait_until="load")
        page.screenshot(path=str(png_path), full_page=False)
        validation = page.evaluate(
            """
            (payload) => {
              const edges = payload.edges;
              const majorGroups = payload.majorGroups;
              const posterEl = document.querySelector('.poster');
              if (!posterEl) return {fatal: 'missing .poster'};
              const poster = posterEl.getBoundingClientRect();
              const all = [...posterEl.querySelectorAll('*')];
              const number = (value) => Number.parseFloat(value) || 0;
              const relativeChildSelector = (root, descendant) => {
                const parts = [];
                let current = descendant;
                while (current && current !== root) {
                  const parent = current.parentElement;
                  if (!parent) return null;
                  const index = [...parent.children].indexOf(current) + 1;
                  parts.unshift(`>:nth-child(${index})`);
                  current = parent;
                }
                return current === root ? parts.join('') : null;
              };
              const overflow = all.filter((el) =>
                el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1
              ).slice(0, 80).map((el) => {
                const style = getComputedStyle(el);
                return {
                  tag: el.tagName, cls: String(el.className || ''),
                  node_id: el.closest('[data-node-id]')?.dataset.nodeId || null,
                  client: [el.clientWidth, el.clientHeight],
                  scroll: [el.scrollWidth, el.scrollHeight],
                  box: {
                    padding_top: number(style.paddingTop),
                    padding_bottom: number(style.paddingBottom)
                  },
                  children: [...el.children].slice(0, 30).map((child, index) => {
                    const childStyle = getComputedStyle(child);
                    return {
                      index: index + 1,
                      tag: child.tagName,
                      cls: String(child.className || ''),
                      margin_top: number(childStyle.marginTop),
                      margin_bottom: number(childStyle.marginBottom),
                      padding_top: number(childStyle.paddingTop),
                      padding_bottom: number(childStyle.paddingBottom)
                    };
                  }),
                  descendants: [...el.querySelectorAll('*')].slice(0, 80).map((child) => {
                    const childStyle = getComputedStyle(child);
                    return {
                      selector: relativeChildSelector(el, child),
                      tag: child.tagName,
                      cls: String(child.className || ''),
                      margin_top: number(childStyle.marginTop),
                      margin_bottom: number(childStyle.marginBottom)
                    };
                  })
                };
              });
              const outside = all.filter((el) => {
                const r = el.getBoundingClientRect();
                return r.left < poster.left - 1 || r.top < poster.top - 1 ||
                       r.right > poster.right + 1 || r.bottom > poster.bottom + 1;
              }).slice(0, 80).map((el) => ({
                tag: el.tagName, cls: String(el.className || ''),
                node_id: el.closest('[data-node-id]')?.dataset.nodeId || null
              }));
              const missingImages = [...document.images]
                .filter((img) => !img.complete || img.naturalWidth === 0)
                .map((img) => img.getAttribute('src'));
              const textLeaves = all.filter((el) =>
                !el.children.length && el.textContent.trim() &&
                el.getBoundingClientRect().width > 0 &&
                el.getBoundingClientRect().height > 0
              );
              const textRecords = textLeaves.map((el) => ({
                text: el.textContent.trim().slice(0, 90),
                font_px: parseFloat(getComputedStyle(el).fontSize),
                node_id: el.closest('[data-node-id]')?.dataset.nodeId || null
              }));
              const textOrientationIssues = all.filter((el) => {
                const text = String(el.innerText || '').trim().replace(/\\s+/g, ' ');
                const r = el.getBoundingClientRect();
                if (!text || r.width <= 0 || r.height <= 0) return false;
                const style = getComputedStyle(el);
                const writingMode = style.writingMode || 'horizontal-tb';
                let rotatedOrSkewed = false;
                if (style.transform && style.transform !== 'none') {
                  try {
                    const matrix = new DOMMatrixReadOnly(style.transform);
                    const angle = Math.atan2(matrix.b, matrix.a) * 180 / Math.PI;
                    rotatedOrSkewed = Math.abs(angle) > 0.5 ||
                      Math.abs(matrix.c + matrix.b) > 0.01;
                  } catch (_) {
                    rotatedOrSkewed = true;
                  }
                }
                return writingMode !== 'horizontal-tb' || rotatedOrSkewed;
              }).slice(0, 40).map((el) => {
                const style = getComputedStyle(el);
                return {
                  tag: el.tagName,
                  cls: String(el.className || ''),
                  node_id: el.closest('[data-node-id]')?.dataset.nodeId || null,
                  text: String(el.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 90),
                  writing_mode: style.writingMode || '',
                  transform: style.transform || 'none'
                };
              });
              const smallText = textRecords.filter((item) =>
                item.font_px < 17 && item.text.length > 10
              ).slice(0, 60);
              const fontValues = textRecords.map((item) => item.font_px)
                .filter(Number.isFinite).sort((a, b) => a - b);
              const percentile = (fraction) => fontValues.length
                ? fontValues[Math.min(fontValues.length - 1, Math.floor((fontValues.length - 1) * fraction))]
                : null;
              const fontStats = {
                leaf_count: fontValues.length,
                min_px: percentile(0),
                p10_px: percentile(0.1),
                median_px: percentile(0.5),
                title_px: posterEl.querySelector('h1')
                  ? parseFloat(getComputedStyle(posterEl.querySelector('h1')).fontSize)
                  : null,
                below_17_count: fontValues.filter((value) => value < 17).length,
                below_20_count: fontValues.filter((value) => value < 20).length,
                below_22_count: fontValues.filter((value) => value < 22).length
              };
              const visualContainers = {
                bordered_count: 0, rounded_count: 0, shadowed_count: 0,
                filled_surface_count: 0
              };
              const darkWideSurfaces = [];
              const colorLuminance = (value) => {
                const match = value.match(/rgba?\\(([^)]+)\\)/);
                if (!match) return null;
                const parts = match[1].split(',').map(Number);
                if (parts.length > 3 && parts[3] < 0.05) return null;
                return (0.2126*parts[0] + 0.7152*parts[1] + 0.0722*parts[2]) / 255;
              };
              all.forEach((el) => {
                const r = el.getBoundingClientRect();
                if (r.width < 24 || r.height < 18) return;
                const style = getComputedStyle(el);
                const border = ['Top', 'Right', 'Bottom', 'Left']
                  .reduce((sum, side) => sum + parseFloat(style['border' + side + 'Width'] || 0), 0);
                if (border > 0.5) visualContainers.bordered_count += 1;
                if (parseFloat(style.borderRadius || 0) > 4) visualContainers.rounded_count += 1;
                if (style.boxShadow && style.boxShadow !== 'none') visualContainers.shadowed_count += 1;
                if (!['transparent', 'rgba(0, 0, 0, 0)'].includes(style.backgroundColor)) {
                  visualContainers.filled_surface_count += 1;
                }
                const luminance = colorLuminance(style.backgroundColor);
                if (
                  luminance !== null && luminance < 0.32 &&
                  r.width / poster.width > 0.65 && r.height > 40
                ) {
                  darkWideSurfaces.push({
                    tag: el.tagName,
                    cls: String(el.className || ''),
                    width_ratio: r.width / poster.width,
                    height: r.height,
                    luminance
                  });
                }
              });
              const branding = posterEl.querySelector('[data-node-id="branding"]');
              const brandImages = branding ? [...branding.querySelectorAll('img')].map((img) => {
                const r = img.getBoundingClientRect();
                const src = img.getAttribute('src') || '';
                const isQr = /(^|\\/)qr[_-]/i.test(src) ||
                  Boolean(img.closest(
                    '[data-brand-kind="qr"], .qr-wrap, .qr-item, .qr-unit'
                  )) ||
                  /\\bqr\\b/i.test(img.alt || '');
                const item = img.closest(
                  '[data-brand-item], .qr-wrap, .qr-item, .qr-unit'
                ) || img.parentElement;
                return {
                  alt: img.alt || '',
                  cls: String(img.className || ''),
                  src,
                  width: r.width,
                  height: r.height,
                  rect: {
                    x: r.x-poster.x, y: r.y-poster.y,
                    right: r.right-poster.x, bottom: r.bottom-poster.y
                  },
                  data_brand_kind: img.dataset.brandKind || '',
                  data_brand_label: img.dataset.brandLabel || '',
                  visible_label: isQr && item
                    ? String(item.innerText || '').trim().replace(/\\s+/g, ' ')
                    : '',
                  background_luminance: colorLuminance(
                    getComputedStyle(img.parentElement || img).backgroundColor
                  ),
                  is_qr: isQr
                };
              }) : [];
              const nonQrBrandImages = brandImages.filter((item) => !item.is_qr);
              const qrBrandImages = brandImages.filter((item) => item.is_qr);
              const brandingRect = branding ? branding.getBoundingClientRect() : null;
              const brandingStyle = branding ? getComputedStyle(branding) : null;
              const brandContentRect = brandingRect && brandingStyle ? {
                left: brandingRect.left-poster.left + number(brandingStyle.borderLeftWidth) + number(brandingStyle.paddingLeft),
                top: brandingRect.top-poster.top + number(brandingStyle.borderTopWidth) + number(brandingStyle.paddingTop),
                right: brandingRect.right-poster.left - number(brandingStyle.borderRightWidth) - number(brandingStyle.paddingRight),
                bottom: brandingRect.bottom-poster.top - number(brandingStyle.borderBottomWidth) - number(brandingStyle.paddingBottom)
              } : null;
              const brandPaddingIntrusions = brandContentRect ? brandImages.map((item) => {
                const excess = {
                  left: Math.max(0, brandContentRect.left-item.rect.x),
                  top: Math.max(0, brandContentRect.top-item.rect.y),
                  right: Math.max(0, item.rect.right-brandContentRect.right),
                  bottom: Math.max(0, item.rect.bottom-brandContentRect.bottom)
                };
                return {...item, excess, max_excess: Math.max(...Object.values(excess))};
              }).filter((item) => item.max_excess > 1) : [];
              const brandMetrics = {
                present: Boolean(branding),
                rect: brandingRect ? {
                  x: brandingRect.x-poster.x, y: brandingRect.y-poster.y,
                  w: brandingRect.width, h: brandingRect.height
                } : null,
                width_ratio: brandingRect ? brandingRect.width/poster.width : 0,
                content_rect: brandContentRect,
                padding_intrusions: brandPaddingIntrusions,
                images: brandImages,
                non_qr_count: nonQrBrandImages.length,
                non_qr_min_width: nonQrBrandImages.length
                  ? Math.min(...nonQrBrandImages.map((item) => item.width)) : 0,
                non_qr_max_width: nonQrBrandImages.length
                  ? Math.max(...nonQrBrandImages.map((item) => item.width)) : 0,
                qr_count: qrBrandImages.length,
                qr_max_width: qrBrandImages.length
                  ? Math.max(...qrBrandImages.map((item) => item.width)) : 0
              };
              const scientificImageMetrics = [...posterEl.querySelectorAll('img')]
                .filter((img) => !img.closest('[data-node-id="branding"]'))
                .map((img) => {
                  const r = img.getBoundingClientRect();
                  const style = getComputedStyle(img);
                  const naturalWidth = img.naturalWidth || 0;
                  const naturalHeight = img.naturalHeight || 0;
                  let effectiveWidth = r.width;
                  let effectiveHeight = r.height;
                  if (
                    style.objectFit === 'contain' && naturalWidth > 0 &&
                    naturalHeight > 0 && r.width > 0 && r.height > 0
                  ) {
                    const scale = Math.min(r.width/naturalWidth, r.height/naturalHeight);
                    effectiveWidth = naturalWidth*scale;
                    effectiveHeight = naturalHeight*scale;
                  }
                  return {
                    node_id: img.closest('[data-node-id]')?.dataset.nodeId || null,
                    src: img.getAttribute('src'), alt: img.alt || '',
                    object_fit: style.objectFit,
                    rect: {x:r.x-poster.x, y:r.y-poster.y, w:r.width, h:r.height},
                    natural: {w:naturalWidth, h:naturalHeight},
                    effective: {w:effectiveWidth, h:effectiveHeight}
                  };
                });
              const chapterMetrics = [...posterEl.querySelectorAll('.chapter')].map((chapter) => {
                const territory = chapter.closest('[data-territory-id]');
                const h2 = chapter.querySelector('h2');
                const marker = chapter.querySelector(':scope > span');
                const style = getComputedStyle(chapter);
                const headingStyle = h2 ? getComputedStyle(h2) : style;
                const markerRect = marker ? marker.getBoundingClientRect() : null;
                const r = chapter.getBoundingClientRect();
                return {
                  territory_id: territory ? territory.dataset.territoryId : null,
                  rect: {x:r.x-poster.x, y:r.y-poster.y, w:r.width, h:r.height},
                  client_height: chapter.clientHeight,
                  scroll_height: chapter.scrollHeight,
                  gap_px: parseFloat(style.columnGap || style.gap || '0') || 0,
                  h2_font_px: parseFloat(headingStyle.fontSize) || 0,
                  h2_line_height_px: parseFloat(headingStyle.lineHeight) || 0,
                  marker_width: markerRect ? markerRect.width : 0,
                  marker_height: markerRect ? markerRect.height : 0
                };
              });
              const chapterRange = (key) => {
                const values = chapterMetrics.map((item) => Number(item[key]) || 0);
                return values.length ? Math.max(...values)-Math.min(...values) : 0;
              };
              const chapterConsistent = chapterMetrics.length >= 2 &&
                chapterRange('gap_px') <= 1 &&
                chapterRange('h2_font_px') <= 0.5 &&
                chapterRange('h2_line_height_px') <= 1 &&
                chapterRange('marker_height') <= 1;
              const chapterConsistency = {
                count: chapterMetrics.length,
                signature_count: chapterConsistent ? 1 : 2,
                consistent: chapterConsistent
              };
              const verticalPaintMetrics = (container) => {
                const cr = container.getBoundingClientRect();
                const intervals = [];
                const addRect = (r) => {
                  const start = Math.max(cr.top, r.top);
                  const end = Math.min(cr.bottom, r.bottom);
                  if (r.width > 0 && end > start) intervals.push([start, end]);
                };
                [container, ...container.querySelectorAll('*')].forEach((el) => {
                  const style = getComputedStyle(el);
                  if (
                    style.display === 'none' || style.visibility === 'hidden' ||
                    Number(style.opacity) === 0
                  ) return;
                  if (el.matches('img,svg,canvas,video,table')) {
                    addRect(el.getBoundingClientRect());
                  }
                  [...el.childNodes]
                    .filter((node) =>
                      node.nodeType === Node.TEXT_NODE && node.textContent.trim()
                    )
                    .forEach((node) => {
                      const range = document.createRange();
                      range.selectNodeContents(node);
                      [...range.getClientRects()].forEach(addRect);
                    });
                });
                intervals.sort((a, b) => a[0] - b[0]);
                const merged = [];
                intervals.forEach(([start, end]) => {
                  const previous = merged[merged.length - 1];
                  if (previous && start <= previous[1] + 6) {
                    previous[1] = Math.max(previous[1], end);
                  } else {
                    merged.push([start, end]);
                  }
                });
                const gaps = [];
                let cursor = cr.top;
                merged.forEach(([start, end]) => {
                  if (start > cursor) gaps.push({start: cursor, end: start});
                  cursor = Math.max(cursor, end);
                });
                if (cursor < cr.bottom) gaps.push({start: cursor, end: cr.bottom});
                const largest = gaps.reduce((best, gap) =>
                  gap.end-gap.start > best.end-best.start ? gap : best,
                  {start: cr.top, end: cr.top}
                );
                const painted = merged.reduce(
                  (sum, interval) => sum + interval[1]-interval[0], 0
                );
                return {
                  height: cr.height,
                  painted_height_ratio: painted/Math.max(1, cr.height),
                  largest_blank_ratio: (largest.end-largest.start)/Math.max(1, cr.height),
                  largest_blank: {
                    y: largest.start-poster.y,
                    h: largest.end-largest.start
                  }
                };
              };
              const rects = {};
              [...posterEl.querySelectorAll('[data-node-id]')].forEach((el) => {
                const id = el.dataset.nodeId;
                const r = el.getBoundingClientRect();
                if (!(id in rects)) rects[id] = {
                  x: r.x - poster.x, y: r.y - poster.y, w: r.width, h: r.height
                };
              });
              const nodePaintMetrics = [...posterEl.querySelectorAll('[data-node-id]')]
                .map((el) => ({
                  id: el.dataset.nodeId,
                  ...verticalPaintMetrics(el)
                }));
              const emptyShells = nodePaintMetrics.filter((item) =>
                item.height >= 180 && item.largest_blank_ratio > 0.28
              );
              const horizontalOverlap = (a, b) => Math.max(
                0, Math.min(a.right, b.right) - Math.max(a.left, b.left)
              );
              const evidenceLike = (el) =>
                el.matches('table,figure,picture,img,svg,canvas') ||
                Boolean(el.querySelector('table,figure,picture,img,svg,canvas')) ||
                /(?:equation|formula|algorithm|table|figure)/i.test(
                  String(el.className || '')
                );
              const titleOnlySidecars = [];
              all.filter((container) => {
                const style = getComputedStyle(container);
                const r = container.getBoundingClientRect();
                return style.display.includes('grid') && r.height >= 150 && r.width >= 360;
              }).forEach((container) => {
                const cr = container.getBoundingClientRect();
                const children = [...container.children].filter((child) => {
                  const style = getComputedStyle(child);
                  const r = child.getBoundingClientRect();
                  return style.display !== 'none' && style.visibility !== 'hidden' &&
                    Number(style.opacity) !== 0 && r.width > 0 && r.height > 0;
                });
                children.filter((child) => child.matches('h2,h3,h4')).forEach((heading) => {
                  const hr = heading.getBoundingClientRect();
                  const companions = children.filter((child) => {
                    if (child === heading || !evidenceLike(child)) return false;
                    const r = child.getBoundingClientRect();
                    const horizontallySeparate =
                      r.left >= hr.right - 8 || r.right <= hr.left + 8;
                    const topAligned = Math.abs(r.top - hr.top) <= Math.max(18, r.height*0.08);
                    return horizontallySeparate && topAligned && r.height >= 150;
                  });
                  if (!companions.length) return;
                  const companion = companions.sort((a, b) => {
                    const ar = a.getBoundingClientRect();
                    const br = b.getBoundingClientRect();
                    return br.width*br.height - ar.width*ar.height;
                  })[0];
                  const er = companion.getBoundingClientRect();
                  const blankTop = hr.bottom;
                  const blankBottom = er.bottom;
                  const blankHeight = Math.max(0, blankBottom-blankTop);
                  const filler = children.some((child) => {
                    if (child === heading || child === companion) return false;
                    const r = child.getBoundingClientRect();
                    return horizontalOverlap(r, hr) >= hr.width*0.35 &&
                      r.top < blankBottom-8 && r.bottom > blankTop+8;
                  });
                  const sideWidthRatio = hr.width/Math.max(1, cr.width);
                  const rowVoidRatio = blankHeight/Math.max(1, er.height);
                  const containerVoidAreaRatio =
                    hr.width*blankHeight/Math.max(1, cr.width*cr.height);
                  if (
                    !filler && sideWidthRatio <= 0.42 &&
                    blankHeight >= 64 && rowVoidRatio >= 0.32 &&
                    containerVoidAreaRatio >= 0.04
                  ) {
                    const owner = container.closest('[data-node-id]');
                    titleOnlySidecars.push({
                      node_id: owner ? owner.dataset.nodeId : null,
                      container_tag: container.tagName,
                      container_class: String(container.className || ''),
                      heading: heading.textContent.trim().replace(/\\s+/g, ' ').slice(0, 120),
                      heading_tag: heading.tagName,
                      companion_tag: companion.tagName,
                      side_width_ratio: sideWidthRatio,
                      row_void_ratio: rowVoidRatio,
                      blank_height_px: blankHeight,
                      container_void_area_ratio: containerVoidAreaRatio
                    });
                  }
                });
              });
              const visualGroupMetrics = majorGroups.map((group) => {
                const groupRects = (group.node_ids || [])
                  .map((id) => rects[id])
                  .filter(Boolean);
                if (!groupRects.length) return {
                  id: group.id, rect: null, node_ids: group.node_ids || []
                };
                const left = Math.min(...groupRects.map((r) => r.x));
                const top = Math.min(...groupRects.map((r) => r.y));
                const right = Math.max(...groupRects.map((r) => r.x+r.w));
                const bottom = Math.max(...groupRects.map((r) => r.y+r.h));
                return {
                  id: group.id,
                  rect: {x:left, y:top, w:right-left, h:bottom-top},
                  node_ids: group.node_ids || []
                };
              });
              const territoryMetrics = [...posterEl.querySelectorAll('[data-territory-id]')]
                .map((territory) => {
                  const tr = territory.getBoundingClientRect();
                  const parent = territory.parentElement;
                  const siblings = parent
                    ? [...parent.children].filter((child) => child.hasAttribute('data-territory-id'))
                    : [];
                  const parentRect = parent ? parent.getBoundingClientRect() : tr;
                  const footer = posterEl.querySelector('footer');
                  const footerRect = footer ? footer.getBoundingClientRect() : null;
                  const frameTop = tr.top;
                  let frameBottom = siblings.length >= 2
                    ? Math.max(tr.bottom, parentRect.bottom)
                    : tr.bottom;
                  if (
                    footerRect && footerRect.top > frameTop && footerRect.top < frameBottom
                  ) frameBottom = footerRect.top;
                  const frameHeight = Math.max(1, frameBottom-frameTop);
                  const owned = [...territory.querySelectorAll('[data-node-id]')]
                    .filter((node) => node.closest('[data-territory-id]') === territory)
                    .map((node) => node.getBoundingClientRect())
                    .filter((r) => r.width > 0 && r.height > 0);
                  if (!owned.length) return {
                    id: territory.dataset.territoryId,
                    node_count: 0,
                    rect: {x: tr.x-poster.x, y: frameTop-poster.y, w: tr.width, h: frameHeight},
                    dom_rect: {x: tr.x-poster.x, y: tr.y-poster.y, w: tr.width, h: tr.height}
                  };
                  const top = Math.min(...owned.map((r) => r.top));
                  const bottom = Math.max(...owned.map((r) => r.bottom));
                  const left = Math.min(...owned.map((r) => r.left));
                  const right = Math.max(...owned.map((r) => r.right));
                  const directMajorGroups = [...territory.children].filter((child) => {
                    const r = child.getBoundingClientRect();
                    return r.width * r.height > tr.width * frameHeight * 0.018;
                  }).length;
                  const paintedIntervals = [];
                  const addPaintedRect = (r) => {
                    const start = Math.max(frameTop, r.top);
                    const end = Math.min(frameBottom, r.bottom);
                    if (r.width > 0 && end > start) paintedIntervals.push([start, end]);
                  };
                  [...territory.querySelectorAll('*')].forEach((el) => {
                    const style = getComputedStyle(el);
                    if (
                      style.display === 'none' || style.visibility === 'hidden' ||
                      Number(style.opacity) === 0
                    ) return;
                    if (el.matches('img,svg,canvas,video')) {
                      addPaintedRect(el.getBoundingClientRect());
                    }
                    [...el.childNodes]
                      .filter((node) =>
                        node.nodeType === Node.TEXT_NODE && node.textContent.trim()
                      )
                      .forEach((node) => {
                        const range = document.createRange();
                        range.selectNodeContents(node);
                        [...range.getClientRects()].forEach(addPaintedRect);
                      });
                  });
                  paintedIntervals.sort((a, b) => a[0] - b[0]);
                  const mergedIntervals = [];
                  paintedIntervals.forEach(([start, end]) => {
                    const previous = mergedIntervals[mergedIntervals.length - 1];
                    if (previous && start <= previous[1] + 6) {
                      previous[1] = Math.max(previous[1], end);
                    } else {
                      mergedIntervals.push([start, end]);
                    }
                  });
                  const internalGaps = mergedIntervals.slice(1).map((interval, index) => ({
                    start: mergedIntervals[index][1],
                    end: interval[0],
                    height: Math.max(0, interval[0] - mergedIntervals[index][1])
                  })).filter((gap) => gap.height > 0);
                  const largestInternalGap = internalGaps.reduce(
                    (largest, gap) => gap.height > largest.height ? gap : largest,
                    {start: top, end: top, height: 0}
                  );
                  const paintedHeight = mergedIntervals.reduce(
                    (sum, interval) => sum + interval[1] - interval[0], 0
                  );
                  const paintedTop = mergedIntervals.length
                    ? mergedIntervals[0][0] : tr.top;
                  const paintedBottom = mergedIntervals.length
                    ? mergedIntervals[mergedIntervals.length - 1][1] : tr.top;
                  const edgeGaps = [
                    {start: frameTop, end: paintedTop, height: paintedTop - frameTop},
                    ...internalGaps,
                    {start: paintedBottom, end: frameBottom, height: frameBottom - paintedBottom}
                  ];
                  const largestVerticalVoid = edgeGaps.reduce(
                    (largest, gap) => gap.height > largest.height ? gap : largest,
                    {start: frameTop, end: frameTop, height: 0}
                  );
                  const gapAllowance = 36;
                  const excessGapHeight = edgeGaps.reduce(
                    (sum, gap) => sum + Math.max(0, gap.height-gapAllowance), 0
                  );
                  const largeVerticalGaps = edgeGaps
                    .filter((gap) => gap.height > frameHeight*0.05)
                    .sort((a, b) => b.height-a.height)
                    .map((gap) => ({
                      y: gap.start-poster.y,
                      h: gap.height,
                      ratio: gap.height/frameHeight
                    }));
                  return {
                    id: territory.dataset.territoryId,
                    node_count: owned.length,
                    major_group_count: directMajorGroups,
                    rect: {x: tr.x-poster.x, y: frameTop-poster.y, w: tr.width, h: frameHeight},
                    dom_rect: {x: tr.x-poster.x, y: tr.y-poster.y, w: tr.width, h: tr.height},
                    content_bounds: {
                      x: left-poster.x, y: top-poster.y, w: right-left, h: bottom-top
                    },
                    top_gap_ratio: (top-frameTop)/frameHeight,
                    bottom_gap_ratio: (frameBottom-bottom)/frameHeight,
                    occupied_height_ratio: (bottom-top)/frameHeight,
                    painted_height_ratio: paintedHeight/frameHeight,
                    painted_content_extent_ratio:
                      (paintedBottom-paintedTop)/frameHeight,
                    painted_top_gap_ratio: (paintedTop-frameTop)/frameHeight,
                    painted_bottom_gap_ratio:
                      (frameBottom-paintedBottom)/frameHeight,
                    largest_internal_gap_px: largestInternalGap.height,
                    largest_internal_gap_ratio:
                      largestInternalGap.height/frameHeight,
                    largest_internal_gap: {
                      y: largestInternalGap.start-poster.y,
                      h: largestInternalGap.height
                    },
                    largest_vertical_void_px: largestVerticalVoid.height,
                    largest_vertical_void_ratio:
                      largestVerticalVoid.height/frameHeight,
                    largest_vertical_void: {
                      y: largestVerticalVoid.start-poster.y,
                      h: largestVerticalVoid.height
                    },
                    excess_vertical_gap_px: excessGapHeight,
                    excess_vertical_gap_ratio:
                      excessGapHeight/frameHeight,
                    large_vertical_gap_count: largeVerticalGaps.length,
                    large_vertical_gaps: largeVerticalGaps
                  };
                });
              const flexMarginSpacers = all.filter((el) => {
                const territory = el.closest('[data-territory-id]');
                const parent = el.parentElement;
                if (!territory || !parent) return false;
                const parentDisplay = getComputedStyle(parent).display;
                if (!parentDisplay.includes('flex')) return false;
                const territoryRect = territory.getBoundingClientRect();
                const style = getComputedStyle(el);
                return number(style.marginTop) > territoryRect.height*0.045 ||
                  number(style.marginBottom) > territoryRect.height*0.045;
              }).map((el) => {
                const territory = el.closest('[data-territory-id]');
                const territoryRect = territory.getBoundingClientRect();
                const style = getComputedStyle(el);
                return {
                  node_id: el.closest('[data-node-id]')?.dataset.nodeId || null,
                  territory_id: territory.dataset.territoryId,
                  tag: el.tagName, cls: String(el.className || ''),
                  margin_top_px: number(style.marginTop),
                  margin_bottom_px: number(style.marginBottom),
                  max_margin_ratio: Math.max(
                    number(style.marginTop), number(style.marginBottom)
                  )/Math.max(1, territoryRect.height)
                };
              });
              const bottomGaps = territoryMetrics
                .map((item) => item.bottom_gap_ratio)
                .filter(Number.isFinite);
              const visualBalance = {
                territory_count: territoryMetrics.length,
                bottom_gap_spread: bottomGaps.length
                  ? Math.max(...bottomGaps) - Math.min(...bottomGaps) : null
              };
              const diagonal = Math.hypot(poster.width, poster.height);
              const edgeDistances = edges.map((edge) => {
                const a = rects[edge.from], b = rects[edge.to];
                if (!a || !b) return {...edge, missing_dom_node:true};
                const ax=a.x+a.w/2, ay=a.y+a.h/2, bx=b.x+b.w/2, by=b.y+b.h/2;
                return {...edge, normalized_distance:Math.hypot(ax-bx, ay-by)/diagonal};
              });
              return {
                canvas: [poster.width, poster.height], overflow, outside, missingImages,
                smallText, fontStats, visualContainers, darkWideSurfaces,
                textOrientationIssues,
                visualGroupMetrics, territoryMetrics, visualBalance,
                brandMetrics, chapterMetrics, chapterConsistency,
                scientificImageMetrics, nodePaintMetrics, emptyShells,
                flexMarginSpacers, titleOnlySidecars,
                nodeRects: rects, edgeDistances,
                longEdges: edgeDistances.filter((edge) => edge.normalized_distance > 0.45)
              };
            }
            """,
            {"edges": edges, "majorGroups": major_groups},
        )
        if export_pdf:
            page.emulate_media(media="screen")
            page.pdf(
                path=str(pdf_path),
                width=f"{canvas_width / PIXELS_PER_PRINT_INCH:g}in",
                height=f"{canvas_height / PIXELS_PER_PRINT_INCH:g}in",
                scale=PRINT_SCALE,
                print_background=True,
            )
        context.close()
        browser.close()

    for item in (validation.get("brandMetrics") or {}).get("images") or []:
        source = target_dir / str(item.get("src") or "")
        profile = image_foreground_profile(source)
        item.update(profile)
        foreground = profile.get("foreground_luminance")
        background = item.get("background_luminance")
        if foreground is not None and background is not None:
            item["contrast_ratio"] = round(
                contrast_ratio(float(foreground), float(background)),
                3,
            )

    inline_or_table_tags = {"I", "SPAN", "STRONG", "TR", "TBODY", "THEAD"}
    text_block_tags = {"H1", "H2", "H3", "H4", "H5", "H6", "P", "LI"}
    validation["hardOverflow"] = [
        item for item in validation["overflow"]
        if item["tag"] not in inline_or_table_tags
        and not (
            item["tag"] in text_block_tags
            and item["scroll"][0] - item["client"][0] <= 4
            and item["scroll"][1] - item["client"][1] <= 8
        )
        and (
            item["scroll"][0] - item["client"][0] > 4
            or item["scroll"][1] - item["client"][1] > 4
        )
    ]
    validation["status"] = "pass" if not (
        validation.get("fatal")
        or validation["hardOverflow"]
        or validation["outside"]
        or validation["missingImages"]
    ) else "warning"
    (target_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return validation
