"""Small deterministic visual profiles for raster poster assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def _linear(channel: float) -> float:
    value = channel / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def image_foreground_profile(path: Path) -> dict[str, Any]:
    """Describe the alpha-weighted foreground of a raster image.

    Vector assets and unreadable files return ``profile_available=False`` rather than
    failing poster generation.
    """

    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
            image.thumbnail((256, 256))
            pixels = list(image.getdata())
    except (OSError, ValueError):
        return {"profile_available": False}
    weights = [alpha / 255.0 for _, _, _, alpha in pixels if alpha >= 8]
    visible = [pixel for pixel in pixels if pixel[3] >= 8]
    total_weight = sum(weights)
    if not visible or total_weight <= 0:
        return {"profile_available": False}
    mean = [
        sum(pixel[channel] * (pixel[3] / 255.0) for pixel in visible) / total_weight
        for channel in range(3)
    ]
    luminance = (
        0.2126 * _linear(mean[0])
        + 0.7152 * _linear(mean[1])
        + 0.0722 * _linear(mean[2])
    )
    chroma = (max(mean) - min(mean)) / 255.0
    return {
        "profile_available": True,
        "foreground_rgb": [round(value, 2) for value in mean],
        "foreground_luminance": round(luminance, 4),
        "foreground_chroma": round(chroma, 4),
        "visible_pixel_ratio": round(len(visible) / max(1, len(pixels)), 4),
        "near_white_foreground": luminance > 0.92 and chroma < 0.08,
    }


def image_palette_profile(path: Path, *, max_colors: int = 4) -> dict[str, Any]:
    """Return restrained chromatic palette candidates from one raster asset.

    Near-white, near-black, and low-chroma colors are deliberately excluded: they are
    useful layout neutrals but should not make every paper converge on the same visual
    identity. Vector and unreadable assets fail closed so they remain available as
    marks without inventing a raster-derived palette.
    """

    if max_colors < 1:
        raise ValueError("max_colors must be positive")
    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
            image.thumbnail((256, 256))
            visible = Image.new("RGB", image.size, "white")
            visible.paste(image.convert("RGB"), mask=image.getchannel("A"))
            quantized = visible.quantize(colors=24, method=Image.Quantize.MEDIANCUT)
            palette = quantized.getpalette() or []
            counts = quantized.getcolors(maxcolors=24) or []
    except (OSError, ValueError):
        return {"profile_available": False, "accent_hexes": []}

    candidates: list[tuple[int, float, tuple[int, int, int]]] = []
    for count, index in counts:
        offset = int(index) * 3
        if offset + 2 >= len(palette):
            continue
        rgb = tuple(int(value) for value in palette[offset : offset + 3])
        high, low = max(rgb), min(rgb)
        chroma = (high - low) / 255.0
        luminance = (
            0.2126 * _linear(rgb[0])
            + 0.7152 * _linear(rgb[1])
            + 0.0722 * _linear(rgb[2])
        )
        if chroma < 0.16 or luminance < 0.045 or luminance > 0.92:
            continue
        candidates.append((int(count), chroma, rgb))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[tuple[int, int, int]] = []
    for _, _, rgb in candidates:
        if any(sum(abs(a - b) for a, b in zip(rgb, prior)) < 70 for prior in selected):
            continue
        selected.append(rgb)
        if len(selected) >= max_colors:
            break
    return {
        "profile_available": True,
        "accent_hexes": ["#%02x%02x%02x" % rgb for rgb in selected],
    }


def contrast_ratio(first_luminance: float, second_luminance: float) -> float:
    """Return the WCAG luminance contrast ratio for two normalized values."""

    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)
