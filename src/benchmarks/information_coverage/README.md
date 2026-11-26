# Full-100 Key-Information Coverage Benchmark

This directory contains the Key-Information Coverage benchmark scripts for the 100-paper evaluation set. The benchmark evaluates whether a candidate poster covers the core scientific information units communicated by the author-made reference poster (independent of visual styling or layout quality).

## Key Files

- `benchmark_common.py`: Shared manifest discovery, JSON I/O, and Codex runner utilities.
- `extraction_prompt.md` / `evaluation_prompt.md`: Prompt templates for reference information-unit extraction and candidate poster coverage evaluation.
- `run_reference_extraction.py`: Extracts 7 structured reference information units per official poster.
- `run_evaluations.py`: Evaluates candidate posters against the reference information units.
- `compare_full100.py`: Aggregates coverage metrics across posters and information types (`comparison_summary.md` / `.json`, `per_poster_comparison.csv`, `per_type_metrics.csv`).
- `validate_full100.py`: Checks completeness and schema consistency across extracted units and evaluation outputs.
- `run_all.py`: End-to-end orchestration wrapper.
- `paper_grounding_audit/`: Two-pass paper-grounding audit pipeline that verifies every extracted information unit against the corresponding original paper PDF.

## Quick Verification

From this directory:

```bash
python validate_full100.py --expect-baseline
```

## Running the Benchmark

Model stages invoke fresh `codex exec --ephemeral` sessions and reuse existing valid outputs unless `--force` is passed.

```bash
# Run the full pipeline (reusing existing valid results)
python run_all.py --workers 5

# Re-extract reference information units for specific poster IDs
python run_reference_extraction.py --ids 65 --force --workers 1

# Re-evaluate candidate poster variants for specific poster IDs
python run_evaluations.py --variants official rs whh --ids 65 --force --workers 1

# Rebuild summary statistics and validate outputs
python compare_full100.py
python validate_full100.py
```

For multi-backend single-poster evaluation (supporting `codex`, `claude`, `agy`, `pi-api`, and `pi-chatgpt`), use the unified CLI:

```bash
bash scripts/eval_benchmark.sh information_coverage \
  --poster path/to/poster.png \
  --reference-units path/to/reference_information_units_after.json \
  --paper-id "<paper_id_or_title>" \
  --out path/to/result.json
```

## Paper-Grounding Audit

See [paper_grounding_audit/README.md](paper_grounding_audit/README.md) for the two-pass audit workflow that produces the paper-grounded reference units (`reference_information_units_after.json`) from the initial poster-only extraction (`reference_information_units_before.json`).
