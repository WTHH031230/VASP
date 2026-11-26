"""Diagnostic contact-sheet and zoomed evidence-detail renderers for VisComposer."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def figure_contact_sheet(
    *,
    figure_paths: list[Path],
    catalog: list[dict[str, Any]],
    output: Path,
) -> Path | None:
    """Create one bounded visual input showing all body Figures with labels."""
    if not figure_paths:
        return None
    tile_width, image_height, label_height = 700, 430, 54
    columns = 2
    rows = max(1, math.ceil(len(figure_paths) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    figures = [item for item in catalog if item.get("kind") == "figure"]
    for index, path in enumerate(figure_paths):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (image_height + label_height)
        with Image.open(path) as source:
            tile = ImageOps.contain(
                source.convert("RGB"),
                (tile_width - 24, image_height - 20),
            )
        canvas.paste(
            tile,
            (
                x + (tile_width - tile.width) // 2,
                y + label_height + (image_height - tile.height) // 2,
            ),
        )
        label = str(figures[index].get("label") if index < len(figures) else path.stem)
        draw.text((x + 16, y + 18), label, fill="#102633")
        draw.rectangle(
            (x + 4, y + 4, x + tile_width - 4, y + image_height + label_height - 4),
            outline="#c7d4d9",
            width=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)
    return output


def appendix_contact_sheets(
    *,
    figure_paths: list[Path],
    catalog: list[dict[str, Any]],
    output_dir: Path,
    page_size: int = 8,
) -> list[Path]:
    """Paginate the secondary appendix pool so its candidates stay inspectable."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    appendix_items = [item for item in catalog if item.get("kind") == "appendix_figure"]
    sheets: list[Path] = []
    for start in range(0, len(figure_paths), page_size):
        page_number = len(sheets) + 1
        page_paths = figure_paths[start : start + page_size]
        page_items = [{**item, "kind": "figure"} for item in appendix_items[start : start + page_size]]
        output = output_dir / f"appendix_figure_contact_sheet_{page_number:02d}.jpg"
        built = figure_contact_sheet(figure_paths=page_paths, catalog=page_items, output=output)
        if built is not None:
            sheets.append(built)
    return sheets


def brand_contact_sheet(
    *,
    catalog: list[dict[str, Any]],
    workspace: Path,
    output: Path,
) -> Path | None:
    """Preview every permitted brand asset (logos and QR codes)."""
    brands = [item for item in catalog if item.get("kind") == "brand"]
    if not brands:
        return None
    tile_width, image_height, label_height = 460, 190, 62
    columns = min(3, len(brands))
    rows = math.ceil(len(brands) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(brands):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (image_height + label_height)
        path = workspace / str(item["relative_path"])
        try:
            if path.suffix.lower() == ".svg":
                import cairosvg

                data = cairosvg.svg2png(url=str(path), output_width=900, output_height=360)
                source = Image.open(io.BytesIO(data))
            else:
                source = Image.open(path)
            with source:
                rgba = source.convert("RGBA")
                white = Image.new("RGBA", rgba.size, "white")
                white.alpha_composite(rgba)
                tile = ImageOps.contain(white.convert("RGB"), (tile_width - 28, image_height - 20))
            canvas.paste(
                tile,
                (
                    x + (tile_width - tile.width) // 2,
                    y + label_height + (image_height - tile.height) // 2,
                ),
            )
        except (ImportError, OSError, ValueError):
            draw.text((x + 18, y + 105), "Preview unavailable", fill="#8a2d2d")
        label = f"{str(item.get('brand_kind')).upper()} · {item.get('label')}"
        draw.text((x + 16, y + 20), label, fill="#102633")
        draw.rectangle(
            (x + 4, y + 4, x + tile_width - 4, y + image_height + label_height - 4),
            outline="#c7d4d9",
            width=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90, optimize=True)
    return output


def checkpoint_contact_sheet(checkpoints: list[dict[str, Any]], output: Path) -> Path | None:
    """Create a labelled horizontal comparison of rendered checkpoints."""
    rendered = [
        (item["name"], Path(item["dir"]) / "poster.png")
        for item in checkpoints
        if (Path(item["dir"]) / "poster.png").is_file()
    ]
    if not rendered:
        return None
    tile_width, tile_height, label_height = 720, 540, 52
    canvas = Image.new("RGB", (tile_width * len(rendered), tile_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(rendered):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
        x = index * tile_width + (tile_width - tile.width) // 2
        canvas.paste(tile, (x, label_height + (tile_height - tile.height) // 2))
        draw.text((index * tile_width + 18, 16), label, fill="#102633")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return output


def evidence_detail_sheet(
    *, report: dict[str, Any], poster: Path, output: Path
) -> Path | None:
    """Create zoomed node crops for scientific-image legibility & scale inspection."""
    metrics = report.get("scientificImageMetrics") or []
    rects = report.get("nodeRects") or {}
    node_ids = list(
        dict.fromkeys(
            str(item.get("node_id"))
            for item in metrics
            if item.get("node_id") and item.get("node_id") in rects
        )
    )
    if not node_ids or not poster.is_file():
        return None
    tile_width, tile_height, label_height = 800, 430, 58
    columns = 2
    rows = math.ceil(len(node_ids) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    with Image.open(poster) as source:
        source = source.convert("RGB")
        for index, node_id in enumerate(node_ids):
            rect = rects[node_id]
            left = max(0, math.floor(float(rect["x"])))
            top = max(0, math.floor(float(rect["y"])))
            right = min(source.width, math.ceil(float(rect["x"]) + float(rect["w"])))
            bottom = min(source.height, math.ceil(float(rect["y"]) + float(rect["h"])))
            if right <= left or bottom <= top:
                continue
            crop = source.crop((left, top, right, bottom))
            crop.thumbnail((tile_width - 28, tile_height - label_height - 18))
            x0 = (index % columns) * tile_width
            y0 = (index // columns) * tile_height
            node_metrics = [item for item in metrics if item.get("node_id") == node_id]
            sizes = ", ".join(
                f"{float((item.get('effective') or {}).get('w') or 0):.0f}x"
                f"{float((item.get('effective') or {}).get('h') or 0):.0f}px"
                for item in node_metrics
            )
            draw.text((x0 + 16, y0 + 15), f"{node_id} | effective {sizes}", fill="#102633")
            x = x0 + (tile_width - crop.width) // 2
            y = y0 + label_height + (tile_height - label_height - crop.height) // 2
            sheet.paste(crop, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output
