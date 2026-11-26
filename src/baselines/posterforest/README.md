# PosterForest Codex Adapter

The adapter executes the upstream PosterForest program: Poster Tree generation
and refinement, binary layout, bullet generation,
visual modification planning, deterministic PPTX generation, and rendering.
Only CAMEL model objects are replaced by isolated `codex exec --ephemeral`
calls. Prompt files are linked into a per-run work directory so third-party
source remains unchanged. The adapter also corrects an upstream step-06
use-before-assignment by initializing the same natural-size values before their
first read, and maps the upstream agent wait deadline to the configured Codex
transport timeout. Neither compatibility correction changes layout formulas,
thresholds, prompts, or retry decisions. PosterForest's own Docling conversion and caption-
linked asset extraction run before all eight official steps. No shared asset
bundle, parser patch, layout patch, retry patch, or timeout patch is used.

The upstream freeze pins `transformers==4.48.0`, while its README separately
pins `docling-parse==4.5.0` but omits the top-level `docling` version. Use the
last compatible pre-Heron stack: `docling==2.49.0`,
`docling-core==2.48.4`, `docling-parse==4.5.0`, and
`docling-ibm-models==3.9.1`. Docling 2.50+ defaults to RT-DETR-v2, which the
upstream-pinned Transformers release cannot load.

```bash
python \
  -m src.baseline.posterforest.run dataset/<paper>/paper.pdf \
  results/posterforest/gpt5.6-sol-strict/<paper>
```

`adapter_run.json` records status, wall time, per-call tokens, and API-equivalent
cost. Final normalized outputs are `poster.png` and `poster.pptx`.
