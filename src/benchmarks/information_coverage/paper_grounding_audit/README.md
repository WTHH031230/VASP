# Paper grounding for official-poster information units

This package checks whether information units taken from author-made academic posters are supported by the corresponding original papers. The audit reads the poster-derived units and the extracted paper text. It does not look at the poster image, and it does not score how well a generated poster covers the units.

The released units cover 100 posters, with 7 units each (700 units). `reference_units/reference_information_units_before.json` is the poster-only extraction. `reference_units/reference_information_units_after.json` is that same set after the paper-grounded edits from the completed audit: 57 statements narrowed to what the paper supports, 8 contradicted statements corrected, and 5 unsupported statements replaced with a nearby paper-supported claim.

A fresh run audits the before file. It does not rewrite the after file.

## Layout

```text
information_coverage_bench/
├── README.md
├── prepare_paper_texts.py       # PDF to page-marked text
├── run_audit.py                 # first pass, one session per paper
├── adjudicate_flags.py          # second pass on units that are not fully supported
├── verify_final_evidence.py     # page and quote check, no model call
├── prompts/
│   ├── audit_prompt.md
│   ├── audit_output_schema.json
│   ├── adjudication_prompt.md
│   └── adjudication_output_schema.json
└── reference_units/
    ├── reference_information_units_before.json
    └── reference_information_units_after.json
```

Paper PDFs, extracted paper text, model logs, and previous audit outputs are not included. The scripts write those under `paper_grounding_audit/` when you run them.

## Requirements

- Python 3.10 or newer
- Ghostscript (`gs`) with the `txtwrite` device
- A logged-in Codex CLI (`codex` on `PATH`)

The model and reasoning effort are fixed in the scripts: `gpt-5.6-sol`, `medium`. Each paper uses its own ephemeral, read-only Codex session. A session may make up to three attempts, and each attempt stops after 1200 seconds.

`prepare_paper_texts.py` and `verify_final_evidence.py` do not call the model.

## Papers

Set `VASP_PDF_ROOT` (defaults to `<repo_root>/dataset`) to the directory containing source paper PDFs named:

```text
{id}_*.pdf
```

IDs must be exactly 0 through 99, with no gaps and no duplicates. The numeric prefix is the poster ID. `prepare_paper_texts.py` records each PDF's SHA-256 in `paper_grounding_audit/paper_manifest.json`.

## Run the audit

From this directory:

```bash
python prepare_paper_texts.py
python run_audit.py --workers 5
python adjudicate_flags.py --workers 5
python verify_final_evidence.py
```

`prepare_paper_texts.py` extracts one text file per PDF page, drops overlong vector and font-encoding lines, and joins the pages with `===== PDF PAGE N =====` markers. Those markers are the page numbers used in evidence. Pass `--force` to extract again.

`run_audit.py` sends each paper's seven before-file units to a new session. The session must read the extracted text for that paper and judge the statement as written. An existing valid `paper_grounding_audit/results/{id}.json` is reused. Pass `--force` to replace it, `--ids 0 4 12` to select posters, or `--summarize-only` to rebuild the first-pass summary from saved results.

`adjudicate_flags.py` opens a new session only for units whose first-pass verdict is not `SUPPORTED`. It judges the original statement, not the first-pass replacement. Saved second-pass results are reused unless you pass `--force`. `--aggregate-only` rebuilds the final tables from saved results.

`verify_final_evidence.py` checks the second-pass evidence for units that remain not fully supported. It checks that each cited page file exists, then compares the quote with that page after Unicode normalization. It writes `paper_grounding_audit/evidence_verification.json`.

## Verdicts

First pass, one verdict and one issue type per unit:

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Every central part of the statement is supported by this paper. |
| `SUPPORTED_WITH_NARROWING` | The core is present, but the statement is broader, stronger, more specific, or more compound than the paper supports. |
| `NOT_FOUND` | The paper does not provide adequate evidence. A related-work citation is not enough. |
| `CONTRADICTED` | The paper materially disagrees in number, direction, scope, condition, dataset, or conclusion. |
| `UNREADABLE` | Text extraction is too incomplete to judge. |

Issue types are `NONE`, `OVERSTATEMENT`, `EXTRA_DETAIL`, `NUMERIC_MISMATCH`, `CLAIM_MISMATCH`, `POSTER_ONLY`, `VERSION_MISMATCH`, `COMPOUND_PARTIAL_SUPPORT`, and `UNVERIFIABLE`.

The session also reports paper identity as `MATCH`, `TITLE_ALIAS_OR_RENAMED`, `MISMATCH`, or `UNCERTAIN`, using the extracted first page rather than the file name.

Second pass, for each flagged unit:

| Decision | Meaning |
|---|---|
| `CONFIRM` | The first-pass verdict and its material reasoning hold. |
| `REVISE` | There is a real grounding problem, but the verdict, issue type, evidence, or replacement should change. |
| `DISMISS` | The original statement is fully supported. The final verdict must be `SUPPORTED`. |

Reasoning and recommended statements are written in Chinese. Evidence quotes stay in the paper's language. `recommended_statement` is empty when the final verdict is `SUPPORTED`.

## Outputs

A full run creates `paper_grounding_audit/`:

| Path | Contents |
|---|---|
| `paper_manifest.json` | PDF path, SHA-256, page count, and text path |
| `paper_pages/{id}/page-NNN.txt` | One extracted page per file |
| `paper_text/{id}.txt` | Page-marked full text |
| `results/{id}.json` | First-pass verdicts for seven units |
| `logs/{id}/` | Prompt, command, and model output for each attempt |
| `summary.json`, `issues.csv`, `REPORT.md` | First-pass aggregate |
| `adjudication/results/{id}.json` | Second pass for papers that had a flag |
| `adjudication/logs/{id}/` | Second-pass model calls |
| `adjudication_summary.json`, `FINAL_ISSUES.csv`, `FINAL_REPORT.md`, `CRITICAL_ISSUES.md` | Final aggregate |
| `evidence_verification.json` | Mechanical page and quote check |

The Codex working directory is the repository root, so the session can read the text path stored in the manifest. The session is sandboxed read-only.

## Released unit files

Each poster has `poster_id`, `title`, and `information_units`. Each unit has:

- `unit_id`, such as `0-U1` through `0-U7`
- `information_type`
- `topic`
- `local_context`
- `statement`, the claim that is audited
- `coverage_requirement`, what a generated poster must communicate
- `official_evidence`, where the claim appears on the official poster

The before file was extracted from official poster images only. The after file keeps the same 700 unit IDs. Its edited statements and coverage requirements are the paper-grounded versions.
