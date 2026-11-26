# VASP Visual-First Composition & Typography Guide

These principles encode VASP's **Vision-Centric Anchoring** and **Semantic-Geometric Decoupling** methodology.

## 1. Visual-First Planning Order
1. **Inspect visual evidence first** (`model_inputs/figure_contact_sheet.jpg`, then any `appendix_figure_contact_sheet_*.jpg` and `brand_contact_sheet.jpg`) alongside `workspace/grounded_context.json`.
2. **Macro story progresses left to right**: Problem / Framing -> Overall Method -> Mechanisms / Details -> Results / Takeaway. Vertical stacking inside one visual group is fine, but later narrative stages must not move back to the left of earlier stages.
3. **Design claim-evidence units, not separate text and image columns**: every Figure, Table, Equation, or Algorithm sits next to the claim it explains, supports, or derives.
4. **One unmistakable primary anchor**: give the centerpiece figure the largest, cleanest physical footprint; keep supporting groups visually subordinate.
5. **Appendix figures are a secondary reserve pool**: admit an appendix figure only when no body figure serves that narrative job or when it supplies uniquely decisive evidence (record the reason in `design_state.decision_log`).

## 2. Physical Size, Layout & Palette System
- **Canvas**: default `2544x1696` landscape (adaptable if the user asks for another aspect ratio/size). Define `.poster` in `style.css` at exact `<W>px x <H>px`.
- **Palette from scientific figures**: derive a restrained, coherent palette from `asset_catalog[*].palette_profile.accent_hexes` (sampled deterministically from the paper's own figures/logos). Do not let QR black/white pixels drive the palette, avoid gratuitous gradients, dark full-width banners, or dashboard-card walls.
- **Physical readability scale** (for `2544x1696`):
  - Title (`H1`): `68-84 px` (floor `>= 64 px`), verbatim paper title unless user requests a custom title
  - Chapter headings: `32-42 px`
  - Group headings: `24-32 px`
  - Body text: `20-24 px` (rendered median `>= 19 px`, avoid `< 17 px` informative text)
  - Captions: `17-20 px`
- **Figure scale floor**: every admitted scientific figure must earn its space—contain-aware effective footprint should reach roughly `>= 12%` of canvas width and `>= 6.5%` of canvas height. Never keep a tiny decorative thumbnail; enlarge it or omit it and reorganize surrounding prose.
- **Shared bottom baseline & whitespace discipline**:
  - Bottom of scientific territories should align within roughly `~2-3%` of canvas height.
  - Avoid internal vertical voids exceeding `~7.5%` of a territory, avoid `margin-top: auto` / `space-between` springs that separate a figure from its prose, and never leave a short heading alone in a narrow column beside a tall figure/table ("title-only sidecar").
- **Typography & tables**:
  - Use automatic hyphenation + justification for continuous body prose; keep titles, short claims, captions, labels, metrics, and lists left-aligned.
  - In re-rendered HTML tables, align method/text columns left and numeric columns right with `font-variant-numeric: tabular-nums`.
- **Branding & QR**:
  - Use actual local asset files (`assets/...`) inside `data-node-id="branding"`.
  - Keep non-QR logos `>= 160 px` with visible background contrast (use a dark local plate for transparent white marks).
  - Keep QR labels verbatim (`Paper`, `Code`, `Project`), horizontal, placed cleanly below their QR inside `data-brand-item`.
