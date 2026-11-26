# VisionBench

Judgement-based visual evaluation for academic posters. The model judges only what is directly visible in the poster image. It does not read the paper, does not assess whether the research is correct, and does not return a 1–5 aesthetic score.

The benchmark has 3 groups and 12 atomic questions. Each question for a poster runs in a fresh model session, with no shared context. The model returns a structured verdict and evidence. Pass rates are aggregated by local code.

The English question bank in `rubrics/*.json` is the source of truth for question text, pass and fail conditions, and applicability.

## Layout

```text
vision_bench/
├── README.md
├── eval_poster.py          # one poster, one session per question
├── evaluator.py            # prompts, validation, aggregation
├── poster_stimuli.py       # DIST01 distance-view stimulus
├── rubrics/
├── tests/
└── requirements.txt
```

This package is the evaluation procedure only. Poster images and previous run results are not included.

## Questions

| Group | Questions |
|---|---|
| Figure presentation quality | `AE04`, `AE13`, `IS01` |
| Visual guidance and emphasis | `ENH01`, `DIST01`, `COL01`, `EN05`, `EN07` |
| Layout and spatial organization | `AL01`, `AL05`, `DEN01`, `WS01` |

`gate` and `discriminative` are used only when aggregating results. They are not written into the model prompt.

Every question uses the original poster except `DIST01`. `DIST01` receives a generated distance view: the poster is scaled to 30% of its width and height, centered on a `#7F7F7F` canvas of the original size, and that derived image is the only input for the question.

## Verdicts and aggregation

Each result is `PASS`, `FAIL`, `UNCERTAIN`, or `NOT_APPLICABLE`, plus evidence, a location, and a confidence in `[0, 1]`.

Only `PASS` and `FAIL` enter the pass rate:

```text
pass_rate = PASS / (PASS + FAIL)
weighted_pass_rate = weight of PASS items / weight of PASS and FAIL items
```

`UNCERTAIN` and `NOT_APPLICABLE` stay out of the denominator. The poster summary is `aggregate.visual_evaluation`. A report is `completed` only when all 12 answers are present. Weights and the critical flag live on each question in the rubric files.

Each report records `rubric_sha256`. `--resume` refuses to continue when that hash, `--model`, or `--reasoning-effort` differs from the existing report.

## Requirements

- Python 3.10 or newer
- Pillow: `python -m pip install -r requirements.txt`
- A logged-in Codex CLI to run the model

Prompt construction, validation, and aggregation do not require Codex. `--dry-run` writes prompts and the `DIST01` image without calling the model.

## Evaluate one poster

From this directory:

```bash
python eval_poster.py \
  --poster path/to/poster.png \
  --output-dir path/to/output \
  --max-workers 6
```

`--model` and `--reasoning-effort` are optional. When they are omitted, Codex uses its own configuration. The default run covers all 12 questions in 12 independent sessions. `--max-workers` only sets concurrency.

The Codex working directory is an empty temporary directory, so the session cannot read the rubric files in this package. The poster is passed with `--image`.

Run selected questions:

```bash
python eval_poster.py \
  --poster path/to/poster.png \
  --output-dir path/to/output \
  --question-ids AE13 EN07
```

If `<poster_id>.json` already contains answers, the command leaves it unchanged. Pass `--resume` to keep every stored answer and run only requested questions that are still missing. A dry-run file with no answers is replaced by the next real run.

Each poster produces `<poster_id>.json` and a `<poster_id>_runs/` directory with the per-question prompt, command, and model output.

## Reproducibility

- Record the model, reasoning effort, and `rubric_sha256`.
- Keep one session per question. Do not place multiple questions in one context.
- Do not let the model read the paper, the rubric files, or other outside material.
- `DIST01` must use the generated 30% centered distance view.

## Tests

```bash
python -m unittest discover -s tests -v
```

