You are auditing whether information units extracted from an author-made academic poster are grounded in the supplied original paper PDF.

You have one extracted paper text file. It contains explicit markers of the form `===== PDF PAGE N =====`. You also receive exactly seven poster-derived information units. Use only this paper text; do not inspect the poster, generated posters, repository materials, or external sources.

Your job is not to ask whether the poster said the claim. Your job is to determine whether the claim, as currently written, is supported by this exact paper version.

First verify paper identity from the title and first-page content:

- `MATCH`: the PDF is the paper expected for this benchmark ID.
- `TITLE_ALIAS_OR_RENAMED`: the scientific work matches, but the benchmark/file title is an older, shortened, or alternate title.
- `MISMATCH`: the PDF is a different scientific work.
- `UNCERTAIN`: identity cannot be established from the extracted text.

For each unit, assign exactly one verdict:

- `SUPPORTED`: every central part of the statement is explicitly supported by the paper or is a direct, unavoidable consequence of text/equations/results in the paper. Minor paraphrase is allowed.
- `SUPPORTED_WITH_NARROWING`: the central idea is present, but the current statement is broader, stronger, more specific, or more compound than the paper supports. Use this whenever one material clause of a compound statement is unsupported.
- `NOT_FOUND`: the paper does not provide adequate evidence for the statement. Do not treat a cited related-work claim as evidence that this paper established the claim.
- `CONTRADICTED`: the paper provides evidence that materially disagrees with the statement, including a different number, direction, scope, condition, dataset, or conclusion.
- `UNREADABLE`: text extraction is too incomplete to make a reliable judgment. Use sparingly.

Choose one issue type:

- `NONE`
- `OVERSTATEMENT`
- `EXTRA_DETAIL`
- `NUMERIC_MISMATCH`
- `CLAIM_MISMATCH`
- `POSTER_ONLY`
- `VERSION_MISMATCH`
- `COMPOUND_PARTIAL_SUPPORT`
- `UNVERIFIABLE`

Audit rules:

1. Judge semantic content, not word overlap.
2. Check exact numbers, dataset names, comparison directions, guarantees, assumptions, and qualifiers whenever they materially affect the claim.
3. For theoretical claims, distinguish assumptions, theorems, interpretations, and empirical observations.
4. For empirical claims, verify the named comparison and direction. Do not upgrade “competitive” to “best” or “some settings” to a universal result.
5. Background/motivation can be supported by the paper's own framing. A claim attributed to this work requires this work's own result, not only a bibliography citation.
6. Provide one to three short, exact evidence excerpts with PDF page number and section/figure/table context. Each quote must be short enough to locate easily.
7. If the verdict is `NOT_FOUND`, evidence should normally be empty. If `CONTRADICTED`, quote the conflicting paper evidence.
8. `recommended_statement` must be empty for `SUPPORTED`. For `SUPPORTED_WITH_NARROWING` or `CONTRADICTED`, provide a concise replacement fully grounded in the paper. For `NOT_FOUND`, provide a grounded salvage only if a clearly related claim exists; otherwise leave it empty.
9. Write `reasoning`, `overall_assessment`, and `recommended_statement` in Chinese. Preserve paper terminology and evidence quotes in their original language.
10. Return the seven units in the supplied order and output only JSON conforming to the schema.

