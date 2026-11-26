#!/usr/bin/env python3
"""Standalone CLI exposing VASP SemParser's full deterministic engine to any agent.

Subcommands:
  prepare      Extract numbered PDF lines, outer-boundary candidates, dynamic JSON schema, and text.
  apply-roles  Validate an agent-authored semantic JSON and publish segments2 / structured / metadata.
  crop         Run full caption-anchored wall-bounded region growing (caption_cut.py) + contact sheet.
  render-page  Render any PDF page with an optional PDF-point coordinate ruler grid for inspection.
  recrop       Losslessly re-crop any exact PDF-point rectangle at high DPI and update manifest.json.
  package      Run captions, metadata cleanup/enrichment, logos, QR codes, and assemble Asset Package.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPTS_DIR = Path(__file__).resolve().parent
ENGINE_DIR = SCRIPTS_DIR / "engine"
EXTRACT_DIR = ENGINE_DIR / "extract"
CC_DIR = EXTRACT_DIR / "caption_cut"
for _p in (CC_DIR, EXTRACT_DIR, ENGINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import textgap  # noqa: E402
import segment_roles  # noqa: E402
import semantic_prompt  # noqa: E402
import semantic_schema  # noqa: E402
import semantic_validation  # noqa: E402
from family_interface import normalize_families  # noqa: E402
from outer_rule_candidates import (  # noqa: E402
    extract_outer_boundary_candidates,
    normalize_page_boundaries,
    prompt_candidates,
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int = 16) -> ImageFont.ImageFont:
    for cand in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _run_step(cmd: list[str], log_path: Path, *, ok_rcs: tuple[int, ...] = (0,)) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    combined = (proc.stdout or "") + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode not in ok_rcs:
        tail = combined.strip()[-1000:]
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc.returncode, proc.stdout


def _build_crop_contact_sheet(figures_dir: Path, out_path: Path) -> Path | None:
    manifest_path = figures_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    entries = _read_json(manifest_path)
    items = [e for e in entries if isinstance(e, dict) and e.get("file") and (figures_dir / e["file"]).is_file()]
    if not items:
        return None
    tile_w, img_h, label_h = 720, 440, 68
    cols = 2
    rows = max(1, math.ceil(len(items) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * (img_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    for idx, entry in enumerate(items):
        r, c = divmod(idx, cols)
        x, y = c * tile_w, r * (img_h + label_h)
        img_path = figures_dir / entry["file"]
        with Image.open(img_path) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (tile_w - 24, img_h - 16))
        sheet.paste(thumb, (x + (tile_w - thumb.width) // 2, y + label_h + (img_h - thumb.height) // 2))
        label = str(entry.get("label") or img_path.stem)
        page = entry.get("page", "?")
        bbox = entry.get("bbox") or []
        bbox_str = ",".join(f"{float(v):.0f}" for v in bbox[:4]) if len(bbox) >= 4 else "?"
        flags = entry.get("flags") or []
        flag_str = f" FLAGS:{','.join(flags)}" if flags else " OK"
        header = f"{label} (p.{page}, {entry.get('scope','body')}) [{bbox_str}]"
        sub = f"{entry.get('file')}{flag_str}"
        border_col = "#d9534f" if flags else "#c5d1d8"
        draw.text((x + 12, y + 10), header[:85], fill="#102633", font=font)
        draw.text((x + 12, y + 34), sub[:85], fill=("#b52b27" if flags else "#4a6070"), font=font)
        draw.rectangle((x + 3, y + 3, x + tile_w - 3, y + img_h + label_h - 3), outline=border_col, width=3 if flags else 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90, optimize=True)
    return out_path


def _normalize_agent_payload(raw_text: str) -> str:
    """Accept clean JSON (or fenced JSON) and tolerate plain-dict author_index_map / table emphasis."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(obj, dict):
        return text
    obj = copy.deepcopy(obj)
    obj.setdefault("tables", [])
    obj.setdefault("equations", [])
    obj.setdefault("algorithms", [])
    obj.setdefault("page_outer_boundaries", [])
    meta = obj.get("metadata")
    if isinstance(meta, dict):
        for k in ("title", "venue", "paper_url", "code_url", "arxiv_id", "project_url"):
            meta.setdefault(k, "")
        for k in ("authors", "institutes"):
            meta.setdefault(k, [])
        aim = meta.get("author_index_map")
        if aim is None:
            meta["author_index_map"] = []
        elif isinstance(aim, dict):
            meta["author_index_map"] = [
                {"author": str(author), "institute_indices": list(indices)}
                for author, indices in aim.items()
            ]
    for tbl in obj.get("tables") or []:
        if isinstance(tbl, dict):
            tbl.setdefault("footnotes", [])
            tbl.setdefault("family", "Table")
            tbl.setdefault("status", "ok")
            emp = tbl.get("emphasis")
            if emp is None:
                tbl["emphasis"] = []
            elif isinstance(emp, dict):
                tbl["emphasis"] = semantic_schema.encode_value(emp)["entries"]
    for eq in obj.get("equations") or []:
        if isinstance(eq, dict):
            eq.setdefault("family", "Equation")
            eq.setdefault("status", "ok")
    for alg in obj.get("algorithms") or []:
        if isinstance(alg, dict):
            alg.setdefault("family", "Algorithm")
            alg.setdefault("status", "ok")
    for seg in obj.get("segments") or []:
        if isinstance(seg, dict):
            r = seg.get("role")
            if r == "figure_caption" and "family" not in seg:
                seg["family"] = "Figure"
            elif r == "table_title" and "family" not in seg:
                seg["family"] = "Table"
            elif r == "subfig_caption" and "family" not in seg:
                seg["family"] = "Figure"
    return json.dumps(obj, ensure_ascii=False)


def cmd_prepare(args: argparse.Namespace) -> int:
    pdf_src = args.pdf.resolve()
    if not pdf_src.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_src}")
    work = args.work_dir.resolve()
    input_dir = work / "input"
    sem_dir = work / "semantic"
    text_dir = work / "text"
    logs_dir = work / "logs"
    for d in (input_dir, sem_dir, text_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    pdf_dst = input_dir / "paper.pdf"
    if pdf_dst.resolve() != pdf_src:
        shutil.copy2(pdf_src, pdf_dst)

    # Extract full plain text
    _run_step(
        [sys.executable, str(ENGINE_DIR / "text/extract_text.py"), str(pdf_dst), str(text_dir)],
        logs_dir / "extract_text.log",
    )

    doc = fitz.open(pdf_dst)
    content_lines = textgap.content_document_lines(doc, include_appendix=args.include_appendix)
    page_indices = [pi for pi, rows in enumerate(content_lines) if rows]
    lines: list[tuple[int, int, fitz.Rect, str]] = []
    line_sections: dict[int, str] = {}
    for pi in page_indices:
        for r, t, section in content_lines[pi]:
            g = len(lines)
            lines.append((g, pi, r, t))
            line_sections[g] = section

    source_doc = [
        {
            "id": g,
            "page": pi + 1,
            "bbox": [r.x0, r.y0, r.x1, r.y1],
            "text": t,
            **({"section": line_sections[g]} if args.include_appendix else {}),
        }
        for g, pi, r, t in lines
    ]
    outer_candidates = extract_outer_boundary_candidates(doc, source_doc)
    outer_candidates_doc = {
        "schema_version": "dp-p2p.outer-boundary-candidates.v1",
        "candidates": outer_candidates,
    }
    clean = lambda s: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    numbered_parts: list[str] = []
    for pi in page_indices:
        numbered_parts.append(f"\n=== PDF PAGE {pi + 1} ===")
        numbered_parts.extend(f"[{g}] {clean(t)}" for g, p, _r, t in lines if p == pi)
    numbered = "\n".join(numbered_parts).lstrip("\n")

    content_pages = [pi + 1 for pi in page_indices]
    prompt_cands = prompt_candidates(outer_candidates)
    boundary_prompt = (
        segment_roles.OUTER_BOUNDARY_PROMPT
        .replace("__CONTENT_PAGES__", json.dumps(content_pages, ensure_ascii=False))
        .replace("__OUTER_BOUNDARY_CANDIDATES__", json.dumps(prompt_cands, ensure_ascii=False, separators=(",", ":")))
    )

    base_task = segment_roles.PROMPT.split("\nLines:\n__LINES__", 1)[0]
    spec = semantic_validation.build_spec(base_task)
    schema = semantic_schema.build_schema(len(source_doc), spec["roles"])
    full_prompt = semantic_prompt.build_prompt(base_task, numbered, boundary_prompt)

    _write_json(sem_dir / "source_lines.json", source_doc)
    _write_json(sem_dir / "outer_boundary_candidates.json", outer_candidates_doc)
    _write_json(sem_dir / "boundary_candidates_compact.json", {"content_pages": content_pages, "candidates": prompt_cands})
    _write_json(sem_dir / "response.schema.json", schema)
    _write_json(sem_dir / "prepare_meta.json", {
        "pdf": str(pdf_dst),
        "total_pdf_pages": len(doc),
        "content_pages": content_pages,
        "line_count": len(source_doc),
        "include_appendix": bool(args.include_appendix),
    })
    (sem_dir / "numbered_lines.txt").write_text(numbered + "\n", encoding="utf-8")
    (sem_dir / "semantic_task.txt").write_text(full_prompt + "\n", encoding="utf-8")

    summary = {
        "status": "prepared",
        "work_dir": str(work),
        "pdf": str(pdf_dst),
        "total_pdf_pages": len(doc),
        "content_pages": content_pages,
        "line_count": len(source_doc),
        "files": {
            "numbered_lines": str(sem_dir / "numbered_lines.txt"),
            "boundary_candidates": str(sem_dir / "boundary_candidates_compact.json"),
            "response_schema": str(sem_dir / "response.schema.json"),
            "semantic_task": str(sem_dir / "semantic_task.txt"),
            "text": str(text_dir / "text.txt"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_apply_roles(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    sem_dir = work / "semantic"
    source_doc = _read_json(sem_dir / "source_lines.json")
    outer_doc = _read_json(sem_dir / "outer_boundary_candidates.json")
    candidates = outer_doc["candidates"]
    meta = _read_json(sem_dir / "prepare_meta.json")
    pages = meta["content_pages"]
    schema = _read_json(sem_dir / "response.schema.json")
    base_task = segment_roles.PROMPT.split("\nLines:\n__LINES__", 1)[0]
    spec = semantic_validation.build_spec(base_task)

    raw_text = args.response.resolve().read_text(encoding="utf-8")
    normalized_raw = _normalize_agent_payload(raw_text)
    try:
        wire = semantic_schema.parse_response(normalized_raw, schema)
        obj = semantic_schema.to_legacy(wire)
        quality = semantic_validation.validate_semantic(obj, len(source_doc), spec)
        structured = segment_roles.normalize_structured(obj, len(source_doc))
        segments, corrections = segment_roles.enforce_explicit_figure_caption_invariants(obj["segments"], source_doc)
        segments, structured = normalize_families(segments, structured)
        page_bounds = normalize_page_boundaries(obj["page_outer_boundaries"], candidates, pages)
    except (semantic_schema.FormatError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "validation_error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    outputs = {
        "semantic_wire.json": wire,
        "semantic_legacy.json": obj,
        "segments2.json": segments,
        "structured_elements.json": structured,
        "metadata_candidate.json": obj["metadata"],
        "page_outer_boundaries.json": page_bounds,
        "semantic_corrections.json": corrections,
        "semantic_quality.json": quality,
    }
    for fn, val in outputs.items():
        _write_json(sem_dir / fn, val)

    role_counts: Counter[str] = Counter()
    for seg in segments:
        r = str(seg.get("role", "?"))
        role_counts[r] += int(seg.get("end", 0)) - int(seg.get("start", 0)) + 1

    viz_dir = None
    if args.viz:
        viz_dir = sem_dir / "roles_viz"
        viz_dir.mkdir(exist_ok=True)
        doc = fitz.open(work / "input/paper.pdf")
        z = 2.0
        font = _font(14)
        by_line: dict[int, tuple[str, Any]] = {}
        for s in segments:
            for g in range(int(s.get("start", 0)), int(s.get("end", -1)) + 1):
                by_line[g] = (str(s.get("role", "?")), s.get("num", s.get("parent")))
        per_page: dict[int, list[dict[str, Any]]] = {}
        for item in source_doc:
            per_page.setdefault(int(item["page"]) - 1, []).append(item)
        rc = {
            "figure_caption": (0, 90, 255),
            "subfig_caption": (0, 180, 220),
            "table_title": (150, 0, 200),
            "equation_caption": (0, 170, 90),
            "figure_content": (255, 170, 0),
        }
        for pi, items in sorted(per_page.items()):
            pix = doc[pi].get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            dr = ImageDraw.Draw(ov)
            for it in items:
                x0, y0, x1, y1 = it["bbox"]
                if by_line.get(it["id"], ("", None))[0] == "body":
                    dr.rectangle([x0 * z, y0 * z, x1 * z, y1 * z], fill=(120, 120, 120, 45))
            img = Image.alpha_composite(img, ov).convert("RGB")
            d2 = ImageDraw.Draw(img)
            for it in items:
                ro, nm = by_line.get(it["id"], ("", None))
                if ro in rc:
                    x0, y0, x1, y1 = it["bbox"]
                    cc = rc[ro]
                    d2.rectangle([x0 * z, y0 * z, x1 * z, y1 * z], outline=cc, width=3)
                    tag = ro.split("_")[0] + (f"{nm}" if nm is not None else "")
                    d2.text((x0 * z, max(0, y0 * z - 15)), tag, fill=cc, font=font)
            img.save(viz_dir / f"page{pi + 1}.png")

    fig_nums = sorted({str(s["num"]) for s in segments if s.get("role") == "figure_caption" and s.get("num") is not None})
    summary = {
        "status": "accepted",
        "segments": len(segments),
        "role_line_counts": dict(role_counts),
        "figure_caption_nums": fig_nums,
        "tables": len(structured.get("tables") or []),
        "equations": len(structured.get("equations") or []),
        "algorithms": len(structured.get("algorithms") or []),
        "auto_corrections": len(corrections),
        "quality": quality,
        "segments_path": str(sem_dir / "segments2.json"),
        "roles_viz_dir": str(viz_dir) if viz_dir else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_crop(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    sem_dir = work / "semantic"
    fig_dir = work / "figures"
    logs_dir = work / "logs"
    seg_path = sem_dir / "segments2.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Missing {seg_path}; run apply-roles first.")
    meta = _read_json(sem_dir / "prepare_meta.json")
    include_appendix = args.include_appendix or bool(meta.get("include_appendix"))

    fig_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(CC_DIR / "caption_cut.py"),
        str(work / "input"),
        str(fig_dir),
        "--seg",
        str(seg_path),
    ]
    if (sem_dir / "source_lines.json").is_file():
        cmd += ["--source-lines", str(sem_dir / "source_lines.json")]
    if (sem_dir / "page_outer_boundaries.json").is_file():
        cmd += ["--outer-boundaries", str(sem_dir / "page_outer_boundaries.json")]
    if (sem_dir / "structured_elements.json").is_file():
        cmd += ["--structured-elements", str(sem_dir / "structured_elements.json")]
    if include_appendix:
        cmd.append("--include-appendix")
    _run_step(cmd, logs_dir / "caption_cut.log")
    # Sync semantic sidecars into figures_dir so build_package.py discovers them automatically
    for sidecar in ("segments2.json", "source_lines.json", "structured_elements.json", "page_outer_boundaries.json"):
        src_sc = sem_dir / sidecar
        if src_sc.is_file():
            shutil.copy2(src_sc, fig_dir / sidecar)

    contact = _build_crop_contact_sheet(fig_dir, fig_dir / "crop_contact_sheet.jpg")
    manifest = _read_json(fig_dir / "manifest.json") if (fig_dir / "manifest.json").is_file() else []
    report = _read_json(fig_dir / "report.json") if (fig_dir / "report.json").is_file() else {"flags": []}

    summary = {
        "status": "cropped",
        "figures_dir": str(fig_dir),
        "crop_count": len(manifest),
        "crops": [
            {
                "label": m.get("label"),
                "page": m.get("page"),
                "scope": m.get("scope"),
                "kind": m.get("kind"),
                "bbox": m.get("bbox"),
                "render_wh": m.get("render_wh"),
                "file": str(fig_dir / m["file"]) if m.get("file") else None,
                "flags": m.get("flags", []),
            }
            for m in manifest
        ],
        "invariant_flags": report.get("flags", []),
        "contact_sheet": str(contact) if contact else None,
        "overlays_dir": str(fig_dir / "overlay"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_render_page(args: argparse.Namespace) -> int:
    doc = fitz.open(args.pdf.resolve())
    pi = args.page - 1
    if not (0 <= pi < len(doc)):
        raise ValueError(f"Page {args.page} out of range (1..{len(doc)})")
    page = doc[pi]
    scale = args.dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if args.grid and args.grid > 0:
        draw = ImageDraw.Draw(img)
        font = _font(12)
        step = float(args.grid)
        w_pt, h_pt = page.rect.width, page.rect.height
        x = 0.0
        while x <= w_pt:
            px = x * scale
            draw.line([(px, 0), (px, pix.height)], fill=(0, 140, 220), width=1)
            draw.text((px + 2, 2), f"x={int(x)}", fill=(0, 90, 180), font=font)
            x += step
        y = 0.0
        while y <= h_pt:
            py = y * scale
            draw.line([(0, py), (pix.width, py)], fill=(0, 140, 220), width=1)
            draw.text((2, py + 2), f"y={int(y)}", fill=(0, 90, 180), font=font)
            y += step
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(json.dumps({
        "status": "rendered",
        "page": args.page,
        "page_rect_pt": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1],
        "dpi": args.dpi,
        "grid_pt": args.grid,
        "output": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_recrop(args: argparse.Namespace) -> int:
    coords = [float(v.strip()) for v in args.bbox.split(",")]
    if len(coords) != 4:
        raise ValueError("--bbox must be four comma-separated PDF point coordinates: x0,y0,x1,y1")
    x0, y0, x1, y1 = coords
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox (need x1>x0 and y1>y0): {coords}")
    doc = fitz.open(args.pdf.resolve())
    pi = args.page - 1
    if not (0 <= pi < len(doc)):
        raise ValueError(f"Page {args.page} out of range (1..{len(doc)})")
    page = doc[pi]
    clip = fitz.Rect(x0, y0, x1, y1) & page.rect
    if clip.is_empty:
        raise ValueError(f"Clipped bbox is empty on page {args.page}: {coords}")
    scale = args.dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out))

    manifest_updated = False
    if args.work_dir and args.label:
        fig_dir = args.work_dir.resolve() / "figures"
        manifest_path = fig_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            rel_file = str(out.relative_to(fig_dir)) if out.is_relative_to(fig_dir) else str(out)
            matched = False
            for entry in manifest:
                if isinstance(entry, dict) and str(entry.get("label")) == args.label:
                    entry["page"] = args.page
                    entry["bbox"] = [round(clip.x0, 2), round(clip.y0, 2), round(clip.x1, 2), round(clip.y1, 2)]
                    entry["render_wh"] = [pix.width, pix.height]
                    entry["file"] = rel_file
                    entry["flags"] = []
                    entry["quality"] = "ok"
                    entry["manual_recrop"] = True
                    matched = True
                    break
            if not matched:
                manifest.append({
                    "label": args.label,
                    "page": args.page,
                    "scope": args.scope,
                    "kind": args.kind,
                    "bbox": [round(clip.x0, 2), round(clip.y0, 2), round(clip.x1, 2), round(clip.y1, 2)],
                    "render_wh": [pix.width, pix.height],
                    "degen": False,
                    "file": rel_file,
                    "flags": [],
                    "quality": "ok",
                    "manual_recrop": True,
                })
            _write_json(manifest_path, manifest)
            _build_crop_contact_sheet(fig_dir, fig_dir / "crop_contact_sheet.jpg")
            manifest_updated = True

    print(json.dumps({
        "status": "recropped",
        "page": args.page,
        "bbox_pt": [round(clip.x0, 2), round(clip.y0, 2), round(clip.x1, 2), round(clip.y1, 2)],
        "render_wh": [pix.width, pix.height],
        "output": str(out),
        "manifest_updated": manifest_updated,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    work = args.work_dir.resolve()
    out_dir = args.out.resolve()
    pdf_path = work / "input/paper.pdf"
    sem_dir = work / "semantic"
    fig_dir = work / "figures"
    text_txt = work / "text/text.txt"
    logs_dir = work / "logs"

    meta_prep = _read_json(sem_dir / "prepare_meta.json") if (sem_dir / "prepare_meta.json").is_file() else {}
    include_appendix = args.include_appendix or bool(meta_prep.get("include_appendix"))

    # 1. Captions
    cap_dir = work / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    cap_cmd = [sys.executable, str(ENGINE_DIR / "captions/extract_captions.py"), str(pdf_path), str(cap_dir)]
    if (sem_dir / "segments2.json").is_file():
        cap_cmd += ["--seg", str(sem_dir / "segments2.json")]
    if include_appendix:
        cap_cmd.append("--include-appendix")
    _run_step(cap_cmd, logs_dir / "extract_captions.log")

    # 2. Metadata
    meta_dir = work / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_json = meta_dir / "metadata.json"
    cand_json = sem_dir / "metadata_candidate.json"
    if cand_json.is_file():
        _run_step(
            [sys.executable, str(ENGINE_DIR / "metadata/extract_metadata.py"), str(pdf_path), str(cand_json), str(meta_dir)],
            logs_dir / "extract_metadata.log",
        )

    # 3. Enrich metadata (best-effort)
    if meta_json.is_file() and not args.skip_enrich:
        try:
            _run_step(
                [sys.executable, str(ENGINE_DIR / "metadata/enrich_metadata.py"), str(meta_json)],
                logs_dir / "enrich_metadata.log",
                ok_rcs=(0, 1),
            )
        except Exception:
            pass

    # 4. Logos (best-effort)
    logos_dir = work / "logos"
    if meta_json.is_file() and not args.skip_logos:
        logos_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_step(
                [sys.executable, str(ENGINE_DIR / "logos/fetch_logos.py"), str(meta_json), str(logos_dir)],
                logs_dir / "fetch_logos.log",
                ok_rcs=(0, 1),
            )
        except Exception:
            pass

    # 5. QR codes (best-effort)
    qr_dir = work / "qr"
    if meta_json.is_file() and not args.skip_qr:
        qr_dir.mkdir(parents=True, exist_ok=True)
        try:
            _run_step(
                [sys.executable, str(ENGINE_DIR / "qr/make_qr.py"), str(meta_json), str(qr_dir)],
                logs_dir / "make_qr.log",
                ok_rcs=(0, 1),
            )
        except Exception:
            pass

    # 6. Build standardized VASP Asset Package
    bcmd = [sys.executable, str(ENGINE_DIR / "bundle/build_package.py"), "--out", str(out_dir)]
    if fig_dir.is_dir():
        bcmd += ["--figures-dir", str(fig_dir)]
    if text_txt.is_file():
        bcmd += ["--text", str(text_txt)]
    if (cap_dir / "captions.json").is_file():
        bcmd += ["--captions", str(cap_dir / "captions.json")]
    if meta_json.is_file():
        bcmd += ["--metadata", str(meta_json)]
    if logos_dir.is_dir():
        bcmd += ["--logos-dir", str(logos_dir)]
    if qr_dir.is_dir():
        bcmd += ["--qr-dir", str(qr_dir)]
    if pdf_path.is_file():
        bcmd += ["--pdf", str(pdf_path)]
    _run_step(bcmd, logs_dir / "build_package.log", ok_rcs=(0, 1))

    manifest = _read_json(out_dir / "manifest.json") if (out_dir / "manifest.json").is_file() else {}
    print(json.dumps({
        "status": "packaged",
        "asset_dir": str(out_dir),
        "manifest": str(out_dir / "manifest.json"),
        "package_complete": manifest.get("complete"),
        "counts": manifest.get("counts"),
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Extract numbered PDF lines, boundary candidates, dynamic schema, and text.")
    p_prep.add_argument("--pdf", type=Path, required=True)
    p_prep.add_argument("--work-dir", type=Path, required=True)
    p_prep.add_argument("--include-appendix", action="store_true")
    p_prep.set_defaults(func=cmd_prepare)

    p_apply = sub.add_parser("apply-roles", help="Validate agent-authored semantic roles JSON and publish artifacts.")
    p_apply.add_argument("--work-dir", type=Path, required=True)
    p_apply.add_argument("--response", type=Path, required=True)
    p_apply.add_argument("--viz", action="store_true", help="Render role debug overlays under semantic/roles_viz/")
    p_apply.set_defaults(func=cmd_apply_roles)

    p_crop = sub.add_parser("crop", help="Run full caption-anchored wall-bounded cropping + build contact sheet.")
    p_crop.add_argument("--work-dir", type=Path, required=True)
    p_crop.add_argument("--include-appendix", action="store_true")
    p_crop.set_defaults(func=cmd_crop)

    p_rp = sub.add_parser("render-page", help="Render a PDF page with optional PDF-point coordinate grid.")
    p_rp.add_argument("--pdf", type=Path, required=True)
    p_rp.add_argument("--page", type=int, required=True, help="1-based PDF page number")
    p_rp.add_argument("--out", type=Path, required=True)
    p_rp.add_argument("--dpi", type=int, default=150)
    p_rp.add_argument("--grid", type=float, default=50.0, help="Grid spacing in PDF points (0 disables)")
    p_rp.set_defaults(func=cmd_render_page)

    p_rc = sub.add_parser("recrop", help="Losslessly re-crop an exact PDF-point rectangle and optionally update manifest.")
    p_rc.add_argument("--pdf", type=Path, required=True)
    p_rc.add_argument("--page", type=int, required=True, help="1-based PDF page number")
    p_rc.add_argument("--bbox", required=True, help="x0,y0,x1,y1 in PDF points")
    p_rc.add_argument("--out", type=Path, required=True)
    p_rc.add_argument("--dpi", type=int, default=300)
    p_rc.add_argument("--work-dir", type=Path, default=None, help="Optional work dir to update figures/manifest.json")
    p_rc.add_argument("--label", default=None, help="Figure label in manifest.json to update or add (e.g. 'Figure 2')")
    p_rc.add_argument("--scope", choices=("body", "appendix"), default="body")
    p_rc.add_argument("--kind", choices=("figure", "table"), default="figure")
    p_rc.set_defaults(func=cmd_recrop)

    p_pkg = sub.add_parser("package", help="Assemble the standardized VASP Asset Package.")
    p_pkg.add_argument("--work-dir", type=Path, required=True)
    p_pkg.add_argument("--out", type=Path, required=True)
    p_pkg.add_argument("--include-appendix", action="store_true")
    p_pkg.add_argument("--skip-enrich", action="store_true")
    p_pkg.add_argument("--skip-logos", action="store_true")
    p_pkg.add_argument("--skip-qr", action="store_true")
    p_pkg.set_defaults(func=cmd_package)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
