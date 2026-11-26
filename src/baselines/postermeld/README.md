# PosterMELD Codex Adapter

This adapter retains PosterMELD's ABT narrative, visual classification,
template selection, capacity planning, block layout, styling, quality gate, and
editable PPTX renderer. Its `LangGraphAgent` calls are routed through isolated
`codex exec --ephemeral` calls. Direct HTTP visual reviewers use the same Codex
backend. PosterMELD's unchanged MinerU cloud-v4 client talks to a local v4
compatibility bridge backed by MinerU 3.4.5 and the local
MinerU2.5-Pro-2605-1.2B model. Its official Marker fallback remains available;
no repository asset bundle or parser stub is injected.

The local service uses MinerU 3.4.5 with
`OpenDataLab/MinerU2.5-Pro-2605-1.2B`; weights and configuration live under
`<MINERU_WEIGHTS_DIR>`. PosterMELD itself runs
in the `postermeld` conda environment, installed from the
upstream requirements including its Marker fork.

Generated teaser/background services are disabled because they are separate
image-generation APIs rather than the requested Codex text/vision backend.

```bash
bash scripts/baseline/start_local_mineru.sh
python \
  -m src.baseline.postermeld.run dataset/<paper>/paper.pdf \
  results/postermeld/gpt5.6-sol-strict/<paper>
```

The run publishes the upstream PPTX/PNG, timing log, Codex traces, and
`adapter_run.json` with wall time, tokens, and API-equivalent cost. It follows
the upstream benchmark semantics: a valid non-draft PNG is a completed
generation, while `quality_gate_accepted` and `quality_gate_failures` report the
stricter internal quality decision separately. A quality-gate rejection never
suppresses the poster or its external benchmark evaluation.
