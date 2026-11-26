# HTML Contract & `design_state.json` Guide

The deterministic harness (`harness_cli.py check`) verifies both the rendered DOM and the semantic claim-evidence graph declared in `workspace/design_state.json` (exact JSON Schema in `schemas/design_state.schema.json`).

## HTML & CSS Contract (`workspace/poster.html` + `workspace/style.css`)
- Single `.poster` root element sized to `<canvas_width>px x <canvas_height>px` in `style.css`.
- Link `<link rel="stylesheet" href="style.css">` and reference only local `assets/...` images.
- No `<script>`, inline `on*` event handlers, `<iframe>`, `<object>`, `<embed>`, or external HTTP URLs.
- **Semantic bindings**:
  - Every stage in `design_state.stages` appears once as `data-stage-id="<stage_id>"`.
  - Every territory in `design_state.territories` appears as `data-territory-id="<territory_id>"`.
  - Every node in `design_state.nodes` appears once as `data-node-id="<node_id>"`.
  - The brand cluster (logos/QRs, NOT the entire masthead) uses `data-node-id="branding"`. Identity/branding is never listed in `design_state.stages`.
  - Brand images carry `data-brand-kind="logo"|"qr"` and `data-brand-label="<verbatim catalog label>"`; wrap each QR + label in `data-brand-item`.
- HTML DOM order and `design_state.reading_order` must match.

## `design_state.json` Top-Level Structure
Consult `schemas/design_state.schema.json` (also copied into `workspace/design_state.schema.json`) for every field. Key fields:
- `paper_title` & `poster_title`: paper title (matches visible `H1`).
- `concept`: 1-2 sentence poster design concept.
- `palette`: chosen colors + source asset notes.
- `stages`: ordered narrative stages (e.g. `framing`, `method`, `details`, `results`).
- `territories`: visual layout regions mapped to stages.
- `nodes`: every semantic unit (`id`, `kind`, `stage`, `territory`, `title`, `assets`, `primary_anchor`, etc.).
- `edges`: explicit claim-evidence links (`from`, `to`, `relation`) so the harness can measure `far_correspondence`.
- `reading_order`: ordered list of `node` IDs matching DOM order.
- `decision_log`: rationale for palette selection, figure admission/omission (especially any appendix figure), and major structural revisions.
