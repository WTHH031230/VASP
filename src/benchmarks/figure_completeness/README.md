# Figure completeness

Independent VLM evaluation of saved, already associated/grouped Figure bboxes.
No extraction, ownership matching, bbox adjustment or quality-driven retries.

## Protocol

- **1**: all meaningful Figure content visible in the reference is preserved.
- **0**: any content is absent/cut off, or the entire Figure was missed.
- Whole-Figure misses come from saved matching and require no model call.
- Every GT Figure is in the denominator. Execution errors are unscored, not 0;
  final rates are withheld until every entry in a subset is resolved.
- Blank margins, external captions and extra surrounding text do not affect
  this completeness-only score. Content already absent from GT is out of scope.

Existing native-Figure-only GT-assisted ownership/grouping is inherited,
including outer-union boxes for multi-component groups. This evaluates grouped
bbox completeness, not native PNG fidelity, raw subfigure completeness, or
unconstrained detection/attribution accuracy.

Default inputs: the 2026-09-08 single-pass ours batch referenced by the assets
mainline marker (462 matched body Figures), and the P2P DocLayout grouped-AP
evaluation with the 2026-09-03 GT (455 matches plus 7 whole-Figure misses).
Saved GT documents must be identical. Total: 924 binary entries, 917 VLM judgments.
Promotion's generic backend-introduction change was live-smoke-tested on one
paper, not rerun on all 100 papers; the result batch is identified explicitly.

One call receives exactly two images: GT first, candidate second. Both are
rendered from the same original PDF at 300 DPI, with fixed bboxes and no added
margin or resizing. Method names and IoU are not supplied to the model. The
existing tool-free, ephemeral Codex backend stages neutral filenames and uses
an isolated temporary directory; no previous conversation is resumed.

Defaults: `gpt-5.6-sol`, `medium`, 8 jobs, 900-second timeout, at most 2 total
attempts per match across resumes. Only timeout/no output/interruption/invalid
format retries. Tool/configuration failures stop the item. A valid zero is
accepted exactly like a one. Output is only `{"complete": 0}` or `{"complete": 1}`.

The [official model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
supports image input, medium effort and structured output. The one-field schema
follows [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas).

## Commands

Use the p2p Python environment with PyMuPDF and the existing VLM backend.
Code is Python 3.11-compatible; the installed environment can be newer.

```bash
bash scripts/benchmark/run_figure_completeness.sh --help
bash scripts/benchmark/run_figure_completeness.sh pilot --prepare-only
bash scripts/benchmark/run_figure_completeness.sh pilot
# Inspect review_pilot.html before continuing.
bash scripts/benchmark/run_figure_completeness.sh full
bash scripts/benchmark/run_figure_completeness.sh full --report-only
```

Set `FIGURE_COMPLETENESS_PYTHON` to the p2p executable if necessary. The script
also accepts `FIGURE_COMPLETENESS_MODEL`, `_EFFORT`, `_JOBS`, `_TIMEOUT`.
Use `--output-root` for a new experiment. Changed PDFs, GT, boxes, model/effort,
prompt/schema, DPI, timeout or attempt budget cannot reuse the same root.
Jobs may change without changing judgments. Additional methods can use repeatable
`--method NAME=/absolute/path/to/saved/grouped-ap/method` options.

Pilot: 20 common GTs selected with seed 20260908 before scoring, requiring 40 calls.
The full phase reuses accepted pilot judgments. The common-matched report is
secondary; the primary full score always includes whole-Figure misses.

## Outputs

- `experiment.json`: frozen inputs/configuration, PDF and source-code hashes.
- `pilot_selection.json`, `prompt.txt`, `response.schema.json`: fixed protocol.
- `inputs/<gt-key>/`: PNGs, dimensions, original geometry and image hashes.
- `items/<method>/<gt-key>/result.json`: decision or explicit execution error.
- `items/.../attempts/`: raw prompts/schemas/events/errors, usage and attempt states.
- `summary.json`, `per_figure.csv`, `RESULTS.md`: all-GT and pilot/common reports.
- `review_pilot.html`, `review_full.html`: native-resolution inspection galleries.

The output root is locked against concurrent runners. Accepted results and
successful durable raw responses are reused without model calls. Token totals
include known usage across all attempts; unknown usage is explicit. The backend
retains its historical dollar-estimate profile; these are not current-price
estimates or invoices and are not used as experimental scores.

## Offline verification

```bash
python -B src/benchmark/figure_completeness/tests/test_figure_completeness.py
python -B src/benchmark/vlm_judge/tests/test_vlm_judge.py
```

After a phase finishes, audit frozen source/PDF/image hashes, caller settings,
raw model responses, automatic whole-miss zeros, and the lifetime attempt cap:

```bash
python -B -m src.benchmark.figure_completeness.audit \
  results/exp/figure-completeness-ours-p2p-20260908 --phase pilot
python -B -m src.benchmark.figure_completeness.audit \
  results/exp/figure-completeness-ours-p2p-20260908 --phase full
```

Optional `--render-check` locates PDF renderer warnings; `--poppler-check`
then produces independent Poppler crops for those cases. These checks make no
model calls and do not replace submitted images or change judgments. Poppler
comparison requires `pdftoppm` on `PATH`. Visual spot-checks are quality assurance,
not an independent human annotation study.
