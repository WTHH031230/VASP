"""Self-contained DOM-to-editable-PPTX poster exporter using Playwright + python-pptx."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

MAX_SLIDE_INCHES = 56.0  # PowerPoint hard limit


def _parse_rgba(css: str) -> tuple[int, int, int, float] | None:
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)", str(css or "").strip())
    if not m:
        return None
    return int(round(float(m.group(1)))), int(round(float(m.group(2)))), int(round(float(m.group(3)))), (
        float(m.group(4)) if m.group(4) is not None else 1.0
    )


def _align_enum(css: str) -> PP_ALIGN:
    val = str(css or "").lower()
    if val == "center":
        return PP_ALIGN.CENTER
    if val in ("right", "end"):
        return PP_ALIGN.RIGHT
    if val == "justify":
        return PP_ALIGN.JUSTIFY
    return PP_ALIGN.LEFT


DOM_EXTRACT_JS = r"""
() => {
  const root = document.querySelector(".poster") || document.body;
  const rr = root.getBoundingClientRect();
  const rootStyle = getComputedStyle(root);
  const shapes = [];
  const images = [];
  const texts = [];

  const isVisible = (el, st) => {
    if (!st || st.display === "none" || st.visibility === "hidden" || parseFloat(st.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 0.5 && r.height > 0.5;
  };

  for (const el of root.querySelectorAll("*")) {
    const st = getComputedStyle(el);
    if (!isVisible(el, st)) continue;
    const r = el.getBoundingClientRect();
    const x = r.left - rr.left;
    const y = r.top - rr.top;
    const w = r.width;
    const h = r.height;
    const tag = el.tagName.toLowerCase();

    if (tag === "img") {
      const nw = el.naturalWidth || w;
      const nh = el.naturalHeight || h;
      let ex = x, ey = y, ew = w, eh = h;
      const fit = (st.objectFit || "fill").toLowerCase();
      if (fit === "contain" && nw > 0 && nh > 0) {
        const s = Math.min(w / nw, h / nh);
        ew = nw * s; eh = nh * s;
        ex = x + (w - ew) / 2; ey = y + (h - eh) / 2;
      }
      images.push({
        x: ex, y: ey, w: ew, h: eh,
        src: el.currentSrc || el.getAttribute("src") || "",
      });
      continue;
    }

    const bg = st.backgroundColor;
    const bw = Math.max(parseFloat(st.borderTopWidth)||0, parseFloat(st.borderRightWidth)||0,
                        parseFloat(st.borderBottomWidth)||0, parseFloat(st.borderLeftWidth)||0);
    const hasBg = bg && bg !== "rgba(0, 0, 0, 0)" && bg !== "transparent";
    if (hasBg || bw >= 0.5) {
      shapes.push({
        x, y, w, h,
        bg: hasBg ? bg : null,
        borderColor: bw >= 0.5 ? st.borderTopColor : null,
        borderWidth: bw,
        radius: parseFloat(st.borderTopLeftRadius) || 0,
      });
    }

    // Collect direct text nodes on leaf/text-bearing elements
    const directText = Array.from(el.childNodes)
      .filter(n => n.nodeType === Node.TEXT_NODE)
      .map(n => n.textContent)
      .join("")
      .replace(/\s+/g, " ")
      .trim();
    if (directText) {
      const hasBlockChild = Array.from(el.children).some(c => {
        const cs = getComputedStyle(c).display;
        return cs === "block" || cs === "flex" || cs === "grid" || cs === "table";
      });
      const textToUse = hasBlockChild ? directText : (el.innerText || directText).trim();
      if (textToUse && !el.dataset._pptxCaptured) {
        if (!hasBlockChild) {
          for (const d of el.querySelectorAll("*")) d.dataset._pptxCaptured = "1";
        }
        const weight = parseInt(st.fontWeight, 10) || (st.fontWeight === "bold" ? 700 : 400);
        texts.push({
          x, y, w, h,
          text: textToUse,
          fontSize: parseFloat(st.fontSize) || 18,
          bold: weight >= 600,
          italic: st.fontStyle === "italic",
          color: st.color,
          align: st.textAlign,
          fontFamily: (st.fontFamily || "Arial").split(",")[0].replace(/["']/g, "").trim(),
        });
      }
    }
  }
  return {
    width: rr.width,
    height: rr.height,
    background: rootStyle.backgroundColor,
    shapes,
    images,
    texts,
  };
}
"""


def export_poster_to_pptx(
    html_path: Path,
    out_pptx: Path,
    *,
    canvas_width: int = 2544,
    canvas_height: int = 1696,
    ppi: float = 50.0,
) -> dict[str, Any]:
    """Render poster HTML in Chromium and reconstruct an editable PowerPoint slide."""
    html_path = html_path.resolve()
    out_pptx = out_pptx.resolve()
    out_pptx.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": canvas_width, "height": canvas_height, "deviceScaleFactor": 1})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")
        data = page.evaluate(DOM_EXTRACT_JS)
        # Screenshot SVG images cleanly via Playwright if needed
        svg_pngs: dict[int, bytes] = {}
        img_locs = page.locator(".poster img")
        for idx, item in enumerate(data["images"]):
            src = str(item.get("src") or "")
            if src.lower().endswith(".svg"):
                try:
                    svg_pngs[idx] = img_locs.nth(idx).screenshot(type="png")
                except Exception:
                    pass
        browser.close()

    w_px = float(data["width"] or canvas_width)
    h_px = float(data["height"] or canvas_height)
    eff_ppi = max(ppi, w_px / MAX_SLIDE_INCHES, h_px / MAX_SLIDE_INCHES)
    slide_w = w_px / eff_ppi
    slide_h = h_px / eff_ppi
    px_to_in = lambda px: Inches(float(px) / eff_ppi)
    px_to_pt = lambda px: Pt(max(6.0, float(px) * 72.0 / eff_ppi))

    prs = Presentation()
    prs.slide_width = Inches(slide_w)
    prs.slide_height = Inches(slide_h)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    bg = _parse_rgba(data.get("background", ""))
    if bg and bg[3] > 0:
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(bg[0], bg[1], bg[2])

    shape_count = 0
    for sh in data["shapes"]:
        bg_c = _parse_rgba(sh.get("bg") or "")
        bd_c = _parse_rgba(sh.get("borderColor") or "")
        if (not bg_c or bg_c[3] <= 0.01) and (not bd_c or bd_c[3] <= 0.01):
            continue
        kind = MSO_SHAPE.ROUNDED_RECTANGLE if float(sh.get("radius") or 0) >= 6 else MSO_SHAPE.RECTANGLE
        box = slide.shapes.add_shape(kind, px_to_in(sh["x"]), px_to_in(sh["y"]), px_to_in(sh["w"]), px_to_in(sh["h"]))
        if bg_c and bg_c[3] > 0.01:
            box.fill.solid()
            box.fill.fore_color.rgb = RGBColor(bg_c[0], bg_c[1], bg_c[2])
        else:
            box.fill.background()
        if bd_c and bd_c[3] > 0.01 and float(sh.get("borderWidth") or 0) >= 0.5:
            box.line.color.rgb = RGBColor(bd_c[0], bd_c[1], bd_c[2])
            box.line.width = px_to_pt(sh["borderWidth"])
        else:
            box.line.fill.background()
        shape_count += 1

    img_count = 0
    for idx, im in enumerate(data["images"]):
        if idx in svg_pngs:
            slide.shapes.add_picture(io.BytesIO(svg_pngs[idx]), px_to_in(im["x"]), px_to_in(im["y"]), px_to_in(im["w"]), px_to_in(im["h"]))
            img_count += 1
            continue
        src = str(im.get("src") or "")
        local_path = None
        if src.startswith("file://"):
            local_path = Path(unquote(urlsplit(src).path))
        elif src:
            local_path = (html_path.parent / src).resolve()
        if local_path and local_path.is_file() and local_path.suffix.lower() != ".svg":
            slide.shapes.add_picture(str(local_path), px_to_in(im["x"]), px_to_in(im["y"]), px_to_in(im["w"]), px_to_in(im["h"]))
            img_count += 1

    text_count = 0
    for t in data["texts"]:
        tb = slide.shapes.add_textbox(px_to_in(t["x"]), px_to_in(t["y"]), px_to_in(max(t["w"], 8)), px_to_in(max(t["h"], 8)))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        lines = str(t["text"]).splitlines() or [""]
        tc = _parse_rgba(t.get("color") or "")
        for li, line in enumerate(lines):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.alignment = _align_enum(t.get("align", "left"))
            run = p.add_run()
            run.text = line
            run.font.size = px_to_pt(t["fontSize"])
            run.font.bold = bool(t.get("bold"))
            run.font.italic = bool(t.get("italic"))
            if t.get("fontFamily"):
                run.font.name = t["fontFamily"]
            if tc:
                run.font.color.rgb = RGBColor(tc[0], tc[1], tc[2])
        text_count += 1

    prs.save(str(out_pptx))
    return {
        "pptx": str(out_pptx),
        "slide_inches": [round(slide_w, 2), round(slide_h, 2)],
        "effective_ppi": round(eff_ppi, 2),
        "elements": {"shapes": shape_count, "images": img_count, "texts": text_count},
    }
