# research-studio-real Codex Baseline

This adapter runs ResearchStudio's complete `paper2assets -> paper2poster`
workflow with one authenticated, ephemeral `codex exec` agent per paper. The
upstream skills, scripts, templates, and html2pptx converter remain read-only in
`third_party/ResearchStudio`; the adapter makes disposable per-run skill copies
and runs every agent against a private staging bundle. Only the trusted runner
promotes files into the final result directory. The host kernel disables nested
user namespaces, so Codex's inner sandbox is bypassed after this isolation step.

## One paper

Use the Python 3.11 `p2p` environment with PyMuPDF, Pillow, Playwright,
python-pptx, LibreOffice, and Poppler available:

```bash
conda run -n vasp \
  python src/baseline/research-studio-real/run.py \
  --paper dataset/<paper>/paper.pdf \
  --output-dir results/research-studio-real/gpt5.6-sol/<paper> \
  --codex-model gpt-5.6-sol \
  --reasoning-effort medium
```

Use `--check` for a free path/dependency preflight. Partial bundles can be
continued with `--resume`; `--force` explicitly removes an old result first.

## Resumable 100-paper batch

```bash
conda run -n vasp \
  python src/baseline/research-studio-real/run_batch.py \
  --dataset-root dataset \
  --output-root results/research-studio-real/gpt5.6-sol \
  --limit 100 --jobs 24 \
  --codex-model gpt-5.6-sol --reasoning-effort medium
```

Each paper contains the editable PPTX, HTML/PDF/PNG poster, ResearchStudio
asset package, `adapter_run.json`, an append-only `codex_attempts.jsonl`, and
Codex traces. The atomic `batch_status.json` records per-paper time/tokens/cost,
mean/median/p95 generation time, mean delivered-poster cost, and total spend
including retries. Cost is an API-equivalent estimate using the documented
gpt-5.6-sol token rates; authenticated Codex subscription usage may not be
billed as API usage.

Every completed Codex call is also appended, under a cross-process file lock,
to the same global ledger as the Paper2Poster and PosterGen baselines:
`~/.codex/usage-ledger/ephemeral.jsonl`. Its records use the shared per-call
schema and `application: "research-studio-real"`; prompts are never written there.

The batch is safe to interrupt and rerun: complete posters are skipped and
partial bundles resume from ResearchStudio's cache checks.
