---
name: vasp-viscomposer
description: >
  Vision-centric academic poster composer backed by VASP's full deterministic Playwright rendering,
  DOM geometry measurement, semantic claim-evidence verification harness, checkpoint selection, and
  PNG/PDF/editable-PPTX export. Use when designing, refining, or exporting a scientific poster from a
  VASP Asset Package (produced by vasp-semparser) or from custom user figures and paper notes.
---

# VASP VisComposer (Experimental Agent Skill)

Compose and iteratively refine scientific posters using **Vision-Centric Anchoring** and **Semantic-Geometric Decoupling**: you (the agent) act directly as the persistent lead information designer, while the bundled deterministic harness (`scripts/harness_cli.py` + `scripts/engine/`) handles asset staging, visual contact sheets, Chromium rendering, exact DOM geometric/semantic gate verification, zoomed figure diagnostics, checkpoint selection, and PNG/PDF/editable-PPTX export.

> **Agent Autonomy:** Treat the Visual-First workflow, bundled harness checks, schemas, and default layout/style settings as a high-efficiency foundation rather than rigid constraints. If you are unsatisfied with any part of the output, check, script behavior, or default rule—or if the user's specific requirements call for a different canvas size, style, content structure, or trade-off—you are free to modify, adapt, or override anything directly based on your own judgment (including using `--relax-contract` / `--safety-only` or editing the bundled scripts when appropriate).

## Prerequisites

All core harness modules are self-contained under `scripts/engine/`. Ensure dependencies and Chromium are installed:

```bash
pip install -r <skill_dir>/scripts/requirements.txt
playwright install chromium
```

## End-to-End Closed-Loop Workflow

### 1. Stage Assets & Build Visual Contact Sheets
Initialize a working directory from either a `vasp-semparser` Asset Package or any folder of user figures:

```bash
python <skill_dir>/scripts/harness_cli.py init \
  --asset-dir <asset_dir> \
  --work-dir <work_dir> \
  [--canvas-width 2544 --canvas-height 1696] [--title "Custom Title"]
```

This populates `<work_dir>/workspace/` (`assets/`, `grounded_context.json`, `design_state.schema.json`) and generates visual contact sheets under `<work_dir>/model_inputs/`.

### 2. Inspect Visual Evidence First (Visual-First Planning)
**Before writing HTML/CSS**, visually open the generated contact sheets in `visual_input_order`:
1. `<work_dir>/model_inputs/figure_contact_sheet.jpg` (primary body figures)
2. Any `<work_dir>/model_inputs/appendix_figure_contact_sheet_*.jpg` (secondary reserve pool)
3. `<work_dir>/model_inputs/brand_contact_sheet.jpg` (logos & QR codes, if present)

Read `<work_dir>/workspace/grounded_context.json` plus `references/visual_first_guide.md` and `references/design_state_guide.md`.

### 3. Author Initial Poster & Semantic Declaration
Write three files directly inside `<work_dir>/workspace/`:
- `poster.html` — complete poster markup with `.poster` root and `data-stage-id` / `data-territory-id` / `data-node-id` bindings
- `style.css` — coherent palette (derived from figure palette profiles), typography scale, and explicit canvas dimensions
- `design_state.json` — declared stages, territories, claim-evidence nodes/edges, reading order, and decision log matching `design_state.schema.json`

### 4. Render, Measure & Visually Diagnose (Closed-Loop Refinement)
Run the full deterministic harness on your workspace:

```bash
python <skill_dir>/scripts/harness_cli.py check \
  --work-dir <work_dir> [--label 01_initial] [--relax-contract]
```

Then **visually inspect both**:
- `<work_dir>/checkpoints/<label>/poster.png` (full poster render)
- `<work_dir>/checkpoints/<label>/evidence_detail_sheet.jpg` (zoomed crops of every placed scientific figure with measured effective pixel dimensions)

Combine your visual inspection with the printed `controller_state` and `regression_audit` (see `references/gate_diagnostics_guide.md`):
- Edit `<work_dir>/workspace/poster.html`, `style.css`, and `design_state.json` incrementally to resolve root causes (overflows, figure legibility/scale, claim-evidence adjacency, visual hierarchy, whitespace balance, branding contrast).
- Re-run `harness_cli.py check --work-dir <work_dir>` after each structural or polish iteration until the poster passes gates and looks crisp to your own eye.
- Optional: if only tiny residual overflow remains, run `python <skill_dir>/scripts/harness_cli.py fit --work-dir <work_dir>` to apply deterministic CSS overflow fitting.

### 5. Select & Export Final Deliverables (PNG, PDF, Editable PPTX)
Export the highest-ranked checkpoint (`best`), your current workspace (`current`), or any named checkpoint (`--checkpoint <label>`):

```bash
python <skill_dir>/scripts/harness_cli.py export \
  --work-dir <work_dir> \
  --checkpoint best \
  --formats png,pdf,pptx [--out <final_dir>] [--safety-only]
```

This writes print-ready `poster.pdf`, crisp `poster.png`, editable `poster.pptx`, and the clean HTML/CSS/asset bundle into `<work_dir>/final/` (or `--out`).
