---
name: vasp-semparser
description: >
  Semantics-directed scientific PDF asset parser based on VASP's Semantic-Geometric Decoupling engine.
  Use when extracting clean, intact figures, tables, captions, structured equations/algorithms, metadata,
  logos, and QR codes from an academic paper or technical PDF—either as Stage 1 of VASP poster generation
  or as a standalone high-precision PDF asset extraction tool.
---

# VASP SemParser (Experimental Agent Skill)

Extract intact, lossless visual and structured assets from scientific PDFs using **Semantic-Geometric Decoupling**: you (the agent) perform semantic role classification and visual quality verification, while the bundled deterministic engine performs exact PDF vector/text-bounded region growing (`caption_cut.py`), invariant checking, and Asset Package assembly.

> **Agent Autonomy:** Treat the bundled scripts, schemas, role taxonomy, and defaults as a high-efficiency foundation rather than rigid constraints. If you are unsatisfied with any extracted result, crop, role assignment, script behavior, or default setting—or if the document or user's specific requirements call for something different—you are free to adjust, override, or modify anything directly based on your own judgment.

## Prerequisites

All core engine code is self-contained under `scripts/engine/`. Install Python dependencies if missing:

```bash
pip install -r <skill_dir>/scripts/requirements.txt
```

## End-to-End Workflow

### 1. Prepare Numbered PDF Lines & Geometry Candidates
Run the deterministic PDF pre-parser (add `--include-appendix` if appendix figures are needed):

```bash
python <skill_dir>/scripts/semparser_cli.py prepare \
  --pdf <paper.pdf> \
  --work-dir <work_dir> [--include-appendix]
```

This creates:
- `<work_dir>/semantic/numbered_lines.txt` — every content line `[id] text` grouped by PDF page
- `<work_dir>/semantic/boundary_candidates_compact.json` — detected outer rules/margins per page
- `<work_dir>/semantic/response.schema.json` — exact dynamic JSON Schema for this PDF
- `<work_dir>/text/text.txt` — reading-order plain text

### 2. Perform Semantic Role Assignment & Structured Extraction
Read `<work_dir>/semantic/numbered_lines.txt`, `<work_dir>/semantic/boundary_candidates_compact.json`, and `references/role_assignment_guide.md`.

Write a single JSON file `<work_dir>/semantic/roles_response.json` assigning every line ID `0 .. N-1` to its semantic role (`figure_caption`, `subfig_caption`, `figure_content`, `body`, `heading`, `table_title`, `algorithm`, `equation_caption`, `title_author`, `page_header`, `page_footer`, `footnote`, `page_number`, `other`) along with structured `tables`, `equations`, `algorithms`, `metadata`, and `page_outer_boundaries`.

Validate and publish the semantic artifacts (pass `--viz` to render role overlays for debugging):

```bash
python <skill_dir>/scripts/semparser_cli.py apply-roles \
  --work-dir <work_dir> \
  --response <work_dir>/semantic/roles_response.json [--viz]
```

If validation reports a schema or range issue, fix `<work_dir>/semantic/roles_response.json` and re-run `apply-roles`.

### 3. Run Deterministic Caption-Anchored Region-Growing Cropping
Run the full bundled VASP geometric cropper (`caption_cut.py` with outer-rule, structured-rule, table-band, rendered-text-wall-edge refinement, and R2 invariant checks):

```bash
python <skill_dir>/scripts/semparser_cli.py crop \
  --work-dir <work_dir> [--include-appendix]
```

This outputs:
- `<work_dir>/figures/figures/*.png` — lossless high-DPI cropped figures
- `<work_dir>/figures/crop_contact_sheet.jpg` — labeled visual overview of every crop with PDF bbox and R2 flags
- `<work_dir>/figures/overlay/pageN.png` — full-page red-box crop overlays
- `<work_dir>/figures/report.json` — R2 invariant flags (`I2_overlap`, `I3_swallows_caption`, `I4_raster_uncovered`)

### 4. Visually Inspect & Surgically Repair Edge Cases
**Always visually inspect** `<work_dir>/figures/crop_contact_sheet.jpg` (and individual crops or `<work_dir>/figures/overlay/pageN.png` for any flagged/suspicious figures).

If any crop clips a legend/axis/panel, absorbs neighboring prose, or misses a non-standard visual:
- **Option A (Semantic fix)**: update the misclassified line roles in `<work_dir>/semantic/roles_response.json` (e.g. changing a diagram label from `body` to `figure_content`, or fixing a caption range) and re-run `apply-roles` + `crop`.
- **Option B (Direct lossless PDF recrop)**: inspect the page with a PDF-point coordinate grid and recrop directly at 300 DPI:
  ```bash
  python <skill_dir>/scripts/semparser_cli.py render-page \
    --pdf <work_dir>/input/paper.pdf --page <N> --out <work_dir>/pageN_grid.png --grid 50

  python <skill_dir>/scripts/semparser_cli.py recrop \
    --pdf <work_dir>/input/paper.pdf --page <N> --bbox x0,y0,x1,y1 \
    --out <work_dir>/figures/figures/page2_figure_Figure2.png \
    --work-dir <work_dir> --label "Figure 2"
  ```
- **Option C (Custom adaptation)**: modify any bundled script or produce custom extracted assets directly if the PDF uses an unusual layout (slides, multi-page spreads, scanned pages, non-academic reports).

### 5. Assemble the Standardized VASP Asset Package
Package crops, captions, enriched metadata, logos, QR codes, text, and structured tables/equations/algorithms (see `references/asset_package_spec.md`):

```bash
python <skill_dir>/scripts/semparser_cli.py package \
  --work-dir <work_dir> \
  --out <asset_dir> [--include-appendix] [--skip-enrich] [--skip-logos] [--skip-qr]
```

Pass `<asset_dir>` directly to `vasp-viscomposer` (or hand the extracted assets directly to the user).
