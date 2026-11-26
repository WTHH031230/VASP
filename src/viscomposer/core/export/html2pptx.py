#!/usr/bin/env python3
"""Export finalized poster HTML as an editable PowerPoint slide.

The adapter keeps ResearchStudio's DOM-to-PowerPoint conversion isolated from
the designer loop.  It fixes file-URL handling for result directories that
contain spaces, preserves the harness print scale, and reconstructs supported
page backgrounds as native PowerPoint fill and shape objects.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit

from pdf2image import convert_from_path
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn


REPO_ROOT = Path(__file__).resolve().parents[4]
VENDOR_CONVERTER = (
    REPO_ROOT
    / "third_party"
    / "ResearchStudio"
    / "ResearchStudio-Reel"
    / "skills"
    / "paper2poster"
    / "html2pptx"
    / "scripts"
    / "html_to_pptx.py"
)
DEFAULT_PRINT_PIXELS_PER_INCH = 50.0


def _load_vendor_converter() -> ModuleType:
    """Load the pinned ResearchStudio converter without modifying sys.path."""

    if not VENDOR_CONVERTER.is_file():
        raise FileNotFoundError(f"missing ResearchStudio converter: {VENDOR_CONVERTER}")
    from src.viscomposer.core.export.vendor_compat import load_patched_module
    return load_patched_module(VENDOR_CONVERTER)


def _browser_context(playwright, width: int, height: int):
    browser = playwright.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage", "--no-sandbox"],
    )
    context = browser.new_context(
        viewport={"width": width, "height": height},
        device_scale_factor=1,
        # The page remains offline, but Playwright needs script execution for
        # runtime CSS injection when isolating the background layer.
        java_script_enabled=True,
    )
    return browser, context


def detect_canvas_pixels(html_path: Path) -> tuple[int, int]:
    """Read the rendered ``.poster`` dimensions instead of assuming A0."""

    with sync_playwright() as playwright:
        browser, context = _browser_context(playwright, 1280, 960)
        page = context.new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.url.startswith(("http://", "https://"))
            else route.continue_(),
        )
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        box = page.locator(".poster").bounding_box()
        context.close()
        browser.close()
    if not box:
        raise ValueError(f"{html_path} has no visible .poster root")
    width, height = round(box["width"]), round(box["height"])
    if width < 1 or height < 1:
        raise ValueError(f"invalid poster canvas: {width}x{height}")
    return width, height


def _normalise_image_sources(dom: dict) -> int:
    """Decode Chromium file URLs before the vendor's ``Path.exists`` check."""

    changed = 0
    for element in dom.get("elements") or []:
        source = element.get("src")
        if not isinstance(source, str) or not source.startswith("file://"):
            continue
        path = unquote(urlsplit(source).path)
        normalised = f"file://{path}"
        if normalised != source:
            element["src"] = normalised
            changed += 1
    return changed


def _missing_image_sources(dom: dict) -> list[str]:
    """Return local image paths that the converter would silently skip."""

    missing: list[str] = []
    for element in dom.get("elements") or []:
        source = element.get("src")
        if not isinstance(source, str) or not source.startswith("file://"):
            continue
        path = Path(source[7:])
        if not path.is_file():
            missing.append(str(path))
    return sorted(set(missing))


_RGBA_RE = re.compile(
    r"rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,"
    r"\s*(\d+(?:\.\d+)?)(?:\s*,\s*(\d+(?:\.\d+)?))?\s*\)"
)
_GRID_RE = re.compile(
    r"linear-gradient\(\s*(90deg|0deg)\s*,\s*(rgba?\([^)]+\))"
    r"\s+([\d.]+)px\s*,\s*rgba?\([^)]+\)\s+[\d.]+px\s*\)"
)


def _parse_rgba(value: str) -> tuple[int, int, int, float]:
    match = _RGBA_RE.search(value or "")
    if not match:
        raise ValueError(f"unsupported CSS color for native background: {value!r}")
    red, green, blue = (round(float(match.group(i))) for i in range(1, 4))
    alpha = float(match.group(4) or 1.0)
    return red, green, blue, alpha


def apply_native_background(pptx_path: Path, dom: dict) -> dict:
    """Reconstruct the poster root background without a slide-size bitmap."""

    root = next(
        (
            element
            for element in dom.get("elements") or []
            if "poster" in str(element.get("tag_class") or "").split()
        ),
        None,
    )
    if root is None:
        raise ValueError("DOM has no .poster root for native background export")

    background = _parse_rgba(str(root.get("bg_color") or ""))
    presentation = Presentation(pptx_path)
    slide = presentation.slides[0]
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(*background[:3])

    image = str(root.get("bg_image") or "none")
    grid_match = _GRID_RE.search(image)
    if image not in {"", "none"} and grid_match is None:
        raise ValueError(
            "unsupported non-native poster background; use a solid fill or "
            "a two-stop 0/90-degree repeating grid"
        )

    grid_shapes = 0
    pattern = None
    if grid_match is not None:
        size_match = re.search(
            r"([\d.]+)px\s+([\d.]+)px",
            str(root.get("bg_size") or ""),
        )
        if size_match is None:
            raise ValueError("native grid background is missing a pixel background-size")
        size_x, size_y = map(float, size_match.groups())
        angle = grid_match.group(1)
        foreground = _parse_rgba(grid_match.group(2))
        stripe_px = float(grid_match.group(3))
        blended = tuple(
            round(foreground[i] * foreground[3] + background[i] * (1 - foreground[3]))
            for i in range(3)
        )
        canvas_width = float(root["w"])
        canvas_height = float(root["h"])
        period_px = size_x if angle == "90deg" else size_y
        span_px = canvas_width if angle == "90deg" else canvas_height
        shape_tree = slide.shapes._spTree
        position = 0.0
        while position < span_px:
            if angle == "90deg":
                left = round(position / canvas_width * presentation.slide_width)
                width = max(
                    1,
                    round(stripe_px / canvas_width * presentation.slide_width),
                )
                stripe = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE,
                    left,
                    0,
                    width,
                    presentation.slide_height,
                )
            else:
                top = round(position / canvas_height * presentation.slide_height)
                height = max(
                    1,
                    round(stripe_px / canvas_height * presentation.slide_height),
                )
                stripe = slide.shapes.add_shape(
                    MSO_SHAPE.RECTANGLE,
                    0,
                    top,
                    presentation.slide_width,
                    height,
                )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = RGBColor(*blended)
            stripe.line.fill.background()
            stripe.shadow.inherit = False
            style = stripe._element.find(qn("p:style"))
            if style is not None:
                # LibreOffice otherwise applies the theme effectRef as a shadow.
                stripe._element.remove(style)
            shape_tree.remove(stripe._element)
            shape_tree.insert(2, stripe._element)
            grid_shapes += 1
            position += period_px
        pattern = f"{angle} grid every {period_px:g}px"

    presentation.save(pptx_path)
    return {
        "mode": "native",
        "solid_rgb": list(background[:3]),
        "pattern": pattern,
        "pattern_shapes": grid_shapes,
    }


def render_pptx(pptx_path: Path, png_path: Path, *, dpi: int = 96) -> None:
    """Render through an isolated, correctly encoded LibreOffice profile."""

    pdf_path = pptx_path.with_suffix(".pdf")
    with tempfile.TemporaryDirectory(prefix="p2p_soffice_") as temp_dir:
        profile = Path(temp_dir) / "profile"
        command = [
            "soffice",
            "--headless",
            f"-env:UserInstallation={profile.as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_path.parent),
            str(pptx_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    if result.returncode != 0 or not pdf_path.is_file():
        detail = (result.stderr or result.stdout or "unknown soffice failure").strip()
        raise RuntimeError(f"PPTX render failed: {detail}")
    pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1)
    if not pages:
        raise RuntimeError(f"PPTX render produced no pages: {pdf_path}")
    pages[0].save(png_path)


def export_html_to_pptx(
    html_path: Path,
    output_dir: Path,
    *,
    name: str = "poster",
    print_pixels_per_inch: float = DEFAULT_PRINT_PIXELS_PER_INCH,
    render_preview: bool = True,
) -> dict:
    """Build an editable PPTX and return a machine-readable export report."""

    started = time.monotonic()
    html_path = html_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if print_pixels_per_inch <= 0:
        raise ValueError("print_pixels_per_inch must be positive")

    canvas_width, canvas_height = detect_canvas_pixels(html_path)
    slide_width = canvas_width / print_pixels_per_inch
    slide_height = canvas_height / print_pixels_per_inch
    if max(slide_width, slide_height) > 56:
        scale = 56 / max(slide_width, slide_height)
        slide_width *= scale
        slide_height *= scale

    converter = _load_vendor_converter()
    dom = converter.extract_dom(
        html_path,
        viewport_w=canvas_width,
        viewport_h=canvas_height,
    )
    normalised_sources = _normalise_image_sources(dom)
    missing_sources = _missing_image_sources(dom)
    if missing_sources:
        preview = "\n  ".join(missing_sources[:8])
        raise FileNotFoundError(
            "poster HTML is not self-contained; missing local image assets "
            f"(use the selected final/poster.html):\n  {preview}"
        )
    dom_path = output_dir / f"{name}_dom.json"
    dom_path.write_text(json.dumps(dom, ensure_ascii=False), encoding="utf-8")

    pptx_path = output_dir / f"{name}.pptx"
    counts = converter.build_pptx(
        dom,
        pptx_path,
        width_inch=slide_width,
        height_inch=slide_height,
        corrections={},
        match_wrap=True,
    )
    background = apply_native_background(pptx_path, dom)

    preview_path = output_dir / f"{name}_pptx.png"
    render_error = None
    if render_preview:
        try:
            render_pptx(pptx_path, preview_path)
        except Exception as error:  # Keep the editable artifact on preview failure.
            render_error = str(error)

    report = {
        "schema_version": 1,
        "source_html": str(html_path),
        "pptx": str(pptx_path),
        "preview_png": str(preview_path) if preview_path.is_file() else None,
        "canvas_px": [canvas_width, canvas_height],
        "slide_inches": [round(slide_width, 4), round(slide_height, 4)],
        "print_pixels_per_inch": print_pixels_per_inch,
        "normalised_image_sources": normalised_sources,
        "elements": counts,
        "background": background,
        "render_error": render_error,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / f"{name}_export.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, help="final poster.html")
    parser.add_argument("output_dir", type=Path, help="PPTX artifact directory")
    parser.add_argument("--name", default="poster")
    parser.add_argument(
        "--print-pixels-per-inch",
        type=float,
        default=DEFAULT_PRINT_PIXELS_PER_INCH,
    )
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    report = export_html_to_pptx(
        args.html,
        args.output_dir,
        name=args.name,
        print_pixels_per_inch=args.print_pixels_per_inch,
        render_preview=not args.no_preview,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["render_error"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

