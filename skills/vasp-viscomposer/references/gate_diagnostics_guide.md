# Reading `harness_cli.py check` Diagnostics & Fixing Root Causes

Each `check` call renders `poster.png` in headless Chromium, generates `evidence_detail_sheet.jpg` (zoomed crops of every placed scientific figure at measured effective pixel sizes), and outputs `controller_state`:

| Field | What It Measures | Recommended Fix |
|---|---|---|
| `deterministic_issues` | Blocking contract, overflow, overlap, figure-scale, title, or branding violations | Resolve root cause with explicit pixel budgeting before local tweaks |
| `geometry.hard_overflow` / `overflow_excess_px` | Elements overflowing their container or canvas | Fund growth by shrinking/reflowing or dropping lowest-priority text—never trade one overflow for another |
| `geometry.small_scientific_images` | Admitted figures below the ~12% W / 6.5% H contain-aware effective floor | Enlarge the figure by reallocating territory space, or omit it and reorganize prose |
| `geometry.empty_shells` / `territories[*].largest_vertical_void_ratio` / `painted_bottom_gap_ratio` | Empty gaps or uneven column bottoms | Enlarge key evidence, reflow content, or rebalance territory columns (do not insert filler cards or flex springs) |
| `geometry.flex_margin_spacers` | Auto-margin / `space-between` springs pushing figures away from claims | Remove the spring so claim and evidence stay visually adjacent |
| `geometry.title_only_sidecars` | Short heading stranded in a narrow side column beside tall evidence | Span the heading across full width above the evidence, or add substantive explanation |
| `far_correspondence` | Claim and linked evidence placed too far apart spatially | Co-locate the claim and its supporting figure/table in the same visual unit |
| `stage_centers` | Narrative stage center x-coordinates moving backwards right-to-left | Restore left-to-right macro progression |
| `font_stats` | Rendered font distribution (`median`, `below_17_count`, title size) | Raise low leaf nodes specifically and fund the extra line wrap; never shrink globally |
| `regression_audit` | Metrics that `worsened` vs `improved` compared to the previous checkpoint | Keep genuine gains; undo edits that caused regressions |

If only tiny residual overflow remains after all semantic/visual checks pass, run `harness_cli.py fit --work-dir <work_dir>` to apply VASP's deterministic small-overflow CSS fit.
