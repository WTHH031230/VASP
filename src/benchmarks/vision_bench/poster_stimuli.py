"""Prepare question-specific poster stimuli without modifying source images."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageColor, ImageOps


def prepare_question_poster(
    poster: Path,
    question: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any] | None]:
    """Return the source poster or a derived stimulus required by a question."""
    transform = question.get("input_transform")
    if not transform:
        return poster, None
    if not isinstance(transform, Mapping):
        raise ValueError(f"Invalid input_transform for {question.get('id')}")
    transform_type = str(transform.get("type", ""))
    if transform_type != "centered_distance_view":
        raise ValueError(
            f"Unsupported input transform {transform_type!r} "
            f"for {question.get('id')}"
        )

    scale = float(transform.get("linear_scale", 0.3))
    if not 0.0 < scale < 1.0:
        raise ValueError("distance-view linear_scale must be between 0 and 1")
    background = str(transform.get("canvas_background", "#7F7F7F"))
    background_rgb = ImageColor.getrgb(background)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{poster.stem}_{question['id']}_distance_view.png"
    with Image.open(poster) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = source.size
        reduced_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        reduced = source.resize(reduced_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), background_rgb)
        offset = (
            (width - reduced_size[0]) // 2,
            (height - reduced_size[1]) // 2,
        )
        canvas.paste(reduced, offset)
        canvas.save(output_path, format="PNG", optimize=True)

    metadata = {
        "type": transform_type,
        "linear_scale": scale,
        "source_poster": str(poster.resolve()),
        "stimulus_path": str(output_path.resolve()),
        "source_size": [width, height],
        "stimulus_canvas_size": [width, height],
        "embedded_poster_size": list(reduced_size),
        "embedded_poster_offset": list(offset),
        "canvas_background": background,
        "resampling": "LANCZOS",
    }
    return output_path, metadata
