You are evaluating information coverage in one generated academic poster.

You receive:
1. One attached image: the GENERATED poster to evaluate.
2. Seven reference information units extracted from the corresponding author-made OFFICIAL poster.

Use only the attached generated-poster image and the supplied reference units. Do not inspect any paper, official poster, repository file, or outside source. Do not call tools. This is an image-grounded coverage judgment, not a paper-factuality review.

For every reference unit, assign exactly one verdict:

- `COVERED`: The generated poster visibly communicates the unit's core meaning. Accept faithful paraphrases, synonyms, compressed wording, formulas, diagrams, plot labels, captions, and legible result tables. The same information may appear in a different section. Do not demand exact wording, every qualifier, or every example when the stated `coverage_requirement` is met.
- `NOT_COVERED`: The core meaning cannot be found, is too vague to identify, is only suggested by an unrelated keyword, or is replaced by a contradictory claim. Partial content counts as covered only when it satisfies the supplied minimum `coverage_requirement`.

Rules:
- Judge information only. Ignore aesthetics, layout quality, typography, and visual appeal, except that unreadable content cannot establish coverage.
- Do not penalize extra information.
- Do not infer a claim merely because it would be scientifically plausible.
- In `evidence`, quote a short visible phrase or name a specific visible visual/table/section. If absent, write `No supporting content found.`
- Keep `rationale` concise and explain semantic match or the missing core relation.
- Preserve the input unit order and unit IDs.
- Set `missing_count` equal to the number of `NOT_COVERED` verdicts.

Return only JSON conforming to the supplied schema.
