#!/usr/bin/env python3
"""Directly stage precomputed MinerU outputs (100% as-is) into PaperX root_folder layout."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

EXP_DIR = Path(os.environ.get("VASP_OUT_DIR", "./outputs/paperx"))
CANONICAL_MANIFEST = Path(os.environ.get("VASP_DATA_DIR", "../VASP-data")) / "paper_list_100.json"
PRELABEL_ROOT = Path(os.environ.get("MINERU_PRELABEL_DIR", "./mineru_prelabel"))
OUT_ROOT = EXP_DIR / "mineru_outputs"


def main() -> None:
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    canonical_sorted = sorted(manifest["papers"])
    prelabel_dirs = sorted(d for d in PRELABEL_ROOT.iterdir() if d.is_dir())

    if len(canonical_sorted) != 100 or len(prelabel_dirs) != 100:
        raise RuntimeError(f"Expected 100 canonical papers and 100 prelabel dirs, got {len(canonical_sorted)} and {len(prelabel_dirs)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    missing_img_refs = 0

    for idx, (canon_name, pre_dir) in enumerate(zip(canonical_sorted, prelabel_dirs), 1):
        prefix = pre_dir.name.split("_", 1)[0]
        if prefix != f"{idx:03d}":
            raise RuntimeError(f"Index mismatch at {idx}: {pre_dir.name}")

        src_auto = pre_dir / "mineru_raw" / "paper" / "auto"
        src_md = src_auto / "paper.md"
        src_pdf = src_auto / "paper_origin.pdf"
        src_imgs = src_auto / "images"
        if not (src_md.is_file() and src_pdf.is_file() and src_imgs.is_dir()):
            raise RuntimeError(f"Incomplete MinerU auto dir: {src_auto}")

        dst_auto = OUT_ROOT / canon_name / "auto"
        dst_imgs = dst_auto / "images"
        dst_imgs.mkdir(parents=True, exist_ok=True)

        dst_md = dst_auto / f"{canon_name}.md"
        dst_pdf = dst_auto / f"{canon_name}_origin.pdf"

        # 1. Direct byte-for-byte reuse of MinerU paper.md
        shutil.copy2(src_md, dst_md)

        # 2. Symlink origin PDF
        if dst_pdf.exists() or dst_pdf.is_symlink():
            dst_pdf.unlink()
        dst_pdf.symlink_to(src_pdf)

        # 3. Symlink each MinerU cropped image into writable dst_imgs dir
        img_count = 0
        for img_file in sorted(src_imgs.iterdir()):
            if not img_file.is_file():
                continue
            link_path = dst_imgs / img_file.name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(img_file)
            img_count += 1

        # Verify all markdown image references exist in dst_imgs
        md_text = dst_md.read_text(encoding="utf-8")
        refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md_text)
        for ref in refs:
            ref_clean = ref.strip()
            if not (dst_auto / ref_clean).is_file():
                missing_img_refs += 1

        records.append({
            "index": idx,
            "canonical_name": canon_name,
            "prelabel_dir": pre_dir.name,
            "source_auto": str(src_auto),
            "staged_auto": str(dst_auto),
            "md_bytes": dst_md.stat().st_size,
            "total_cropped_images": img_count,
            "md_image_refs": len(refs),
        })

    (EXP_DIR / "paper_index_map.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Staged {len(records)}/100 papers into {OUT_ROOT}")
    print(f"Broken markdown image refs across all 100 papers: {missing_img_refs}")
    print(f"Total md_image_refs across 100 papers: {sum(r['md_image_refs'] for r in records)}")


if __name__ == "__main__":
    main()
