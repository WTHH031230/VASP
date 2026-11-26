# PosterGen Codex Baseline

This adapter runs `third_party/PosterGen` without modifying the upstream checkout.
All text and vision model calls are replaced by authenticated `codex exec --ephemeral`
invocations. Marker extraction, three-column planning, layout balancing, styling, and
PowerPoint rendering remain upstream.

## Environment

Use Python 3.11 with PosterGen's Marker, PyTorch, python-pptx, Pillow, PyYAML,
LibreOffice, and Poppler dependencies. The existing Paper2Poster environment can be
cloned and supplemented with:

```bash
python -m pip install -r src/baseline/postergen/requirements.txt
```

No provider API key is needed; `codex login status` must succeed.

## One Paper

```bash
python src/baseline/postergen/run.py \
  --paper dataset/<paper>/paper.pdf \
  --output-dir results/postergen/gpt5.6-sol/<paper> \
  --codex-model gpt-5.6-sol --reasoning-effort medium
```

Dataset papers do not contain branding assets, so the batch defaults to no conference
logo and derives its palette from a selected paper figure. Pass `--logo` and
`--aff-logo` when those assets are available.

## Resumable Batch

```bash
python src/baseline/postergen/run_batch.py \
  --dataset-root dataset \
  --output-root results/postergen/gpt5.6-sol \
  --limit 100 --jobs 12 \
  --codex-model gpt-5.6-sol --reasoning-effort medium
```

Each result contains editable PPTX, PNG preview, PosterGen intermediate JSON,
`timing_cost_log.json`, `adapter_run.json`, and preserved `codex_traces/`.
`batch_status.json` aggregates per-paper time, tokens, and standard API-equivalent
cost. Marker extraction is protected by a cross-process GPU lock, while later
Codex stages run at the requested paper-level concurrency. Successful calls are
also appended under a cross-project lock to
`~/.codex/usage-ledger/ephemeral.jsonl`; override it with `--usage-ledger` or disable
it with `--no-global-ledger`. The ledger contains usage metadata, never prompts.
