"""ResearchStudio asset package for the arm without our assets (DEC-018; spec §2).

Source: ResearchStudio paper2assets, PDF route, as used for the Table 2 crop benchmark
(results/exp/researchstudio_pdf_agent/<paper>/assets). Replacements follow the block
ablation plan (notes/ablation_block_plan_postsubmission_20260924.md, ASSET-OFF):

- crops and captions: ResearchStudio's figures.json / captions.json; no missing crop is
  back-filled, label and page come from ResearchStudio, no scope or priority field
- text: `paper_text` = ResearchStudio text.txt verbatim (pdftotext -layout, unnumbered,
  references and appendix included), form feeds turned into newlines only
- no structured table / equation / algorithm records, no appendix pool (count 0)
- no brand assets: the PDF route produces none, and brand belongs to the assets module
- palette_profile and the body contact sheet are recomputed from these crops with the
  harness's own functions; metadata stays ours (base)
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from src.viscomposer.core.harness import _figure_contact_sheet
from src.viscomposer.core.visual_assets import image_palette_profile
from src.viscomposer.core.workbench import load_json, write_json

REPO = Path(__file__).resolve().parents[2]
RS_ROOT = Path(os.environ.get("VASP_RS_ROOT", str(REPO / "results" / "exp" / "researchstudio_pdf_agent")))


def _safe_stem(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower() or "figure"


def stage_researchstudio(*, paper: str, metadata: dict[str, Any], workspace: Path,
                         output_dir: Path, rs_root: Path | None = None) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    """Stage ResearchStudio assets into the workspace; returns (context, allowed, provenance)."""
    root = rs_root.resolve() if rs_root is not None else RS_ROOT
    source = root / paper if (root / paper / "assets" / "meta").is_dir() else root
    meta = source / "assets" / "meta"
    figures = load_json(meta / "figures.json")
    captions = load_json(meta / "captions.json")
    text = (meta / "text.txt").read_text(encoding="utf-8").replace("\f", "\n")

    figure_root = workspace / "assets" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    staged: list[Path] = []
    allowed: set[str] = set()
    hashes: dict[str, str] = {}
    for index, figure in enumerate(figures, start=1):
        path = source / str(figure["file"])
        if not path.is_file():
            raise FileNotFoundError(f"missing ResearchStudio crop: {path}")
        label = str(figure.get("caption_label") or f"Figure {index}")
        name = f"{index:02d}_{_safe_stem(label)}{path.suffix.lower() or '.png'}"
        relative = f"assets/figures/{name}"
        shutil.copy2(path, figure_root / name)
        staged.append(workspace / relative)
        allowed.add(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        width, height = int(figure["width"]), int(figure["height"])
        catalog.append({
            "attachment_index": index,
            "kind": "figure",
            "label": label,
            "page": figure.get("page"),
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(1, height), 3),
            "relative_path": relative,
            "caption": captions.get(label, str(figure.get("caption") or "")),
            "palette_profile": image_palette_profile(path),
        })
    sheet = _figure_contact_sheet(figure_paths=staged, catalog=catalog,
                                  output=output_dir / "model_inputs" / "figure_contact_sheet.jpg")
    context = {
        "metadata": metadata,
        "asset_catalog": catalog,
        "appendix_policy": {
            "available_count": 0,
            "pool_role": "secondary_evidence",
            "admission_rule": (
                "Use only when no body Figure serves the same narrative job or when "
                "the appendix Figure provides uniquely decisive evidence."
            ),
            "palette_rule": "An appendix Figure may influence the poster palette only if admitted.",
        },
        "paper_text": text,
        "visual_input_order": [
            {"role": "primary_body_figures", "path": str(sheet), "asset_count": len(staged)},
        ],
    }
    write_json(workspace / "grounded_context.json", context)
    provenance = {
        "source": str(source),
        "figure_count": len(staged),
        "figure_sha256": hashes,
        "text_sha256": hashlib.sha256((meta / "text.txt").read_bytes()).hexdigest(),
        "text_chars": len(text),
    }
    return context, allowed, provenance
