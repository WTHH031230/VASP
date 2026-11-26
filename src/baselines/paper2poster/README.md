# Paper2Poster Codex Baseline

This adapter runs the upstream checkout in `third_party/Paper2Poster` while
replacing its CAMEL/OpenAI model calls with the authenticated `codex exec`
command. The Paper2Poster algorithms, prompt templates, 80-poster layout set,
and rendering code remain upstream; this directory contains only the
integration layer. No provider API key or `python-dotenv` package is required.

## Invocation

Use the repository's `p2p` environment, which must contain Paper2Poster's
Python dependencies, LibreOffice, and Poppler. Confirm that `codex login status`
succeeds, then run:

```bash
proxy
conda run -n p2p python -m pip install \
  -r src/baseline/paper2poster/requirements.txt
codex login status
```

The dependency file follows the upstream environment because PDF parsing uses
Docling/Marker. It can be large; the adapter does not install it implicitly.

Generate one poster with:

```bash
conda run -n p2p python -m src.baseline.paper2poster.run \
  --paper dataset/<paper-name>/paper.pdf \
  --output-dir results/paper2poster-codex/<paper-name> \
  --codex-model <model-id> \
  --reasoning-effort medium \
  --max-workers 2
```

Omit `--codex-model` and `--reasoning-effort` to use Codex's built-in defaults.
Use `--check` to validate paths without invoking a model. For a cheaper smoke
run, add `--no-commenter`; this disables the visual overflow correction loop.

## Backend Contract

Each upstream `ChatAgent.step()` launches:

```text
codex exec --json --sandbox read-only --ephemeral \
  --ignore-user-config --ignore-rules -
```

Prompts are passed over stdin, avoiding command-line length limits. CAMEL image
messages are materialized as PNG files and attached with `--image`. The adapter
parses the final `agent_message` and token usage from the JSONL event stream,
then exposes them in the response shape expected by Paper2Poster. Multi-turn
actor corrections retain their conversation history even though individual
Codex processes are ephemeral. A minimal in-process CAMEL shim prevents the
upstream provider factory from importing or calling an SDK backend. Personal
Codex configuration, MCP servers, and execution rules are not loaded; saved CLI
authentication is still reused.

## Outputs and Reproducibility

The requested output directory contains the editable PPTX, rendered previews,
`adapter_run.json`, and Paper2Poster intermediates. By default,
`codex_traces/` also preserves prompts, JSONL events, stderr, and VLM images.
These traces may contain the full paper text; use `--no-traces` for sensitive
papers. Existing non-empty output directories are rejected unless `--force`
is explicitly supplied.

Each model step is a separate Codex invocation, so a full visual-feedback run
can be slow and consume substantial quota. Keep `--max-workers` conservative.

## Resumable Batch

Run a deterministic, serial 100-paper batch with per-paper logs and an atomic
`batch_status.json`:

```bash
conda run -n paper2poster \
  python -m src.baseline.paper2poster.run_batch \
  --dataset-root dataset \
  --output-root results/paper2poster/gpt5.6-sol \
  --limit 100 \
  --codex-model gpt-5.6-sol \
  --reasoning-effort medium
```

Completed output directories are detected by manifest, PPTX, and preview image
and skipped on a resumed run. Failed papers are logged and do not stop later
papers.

Successful ephemeral calls are also appended to the cross-project ledger at
`~/.codex/usage-ledger/ephemeral.jsonl`. Existing project traces can be added
idempotently without invoking a model:

```bash
python -m src.baselines.usage_ledger \
  --results-root results/paper2poster/gpt5.6-sol \
  --application paper2poster
```
