# SemParser Role Assignment & Structured Extraction Guide

This reference documents the exact VASP SemParser semantic contract expected by `semparser_cli.py apply-roles`.

## 1. Closed Line Role Taxonomy

Assign every numbered line ID (`0 .. N-1`) from `<work>/semantic/numbered_lines.txt` to exactly one contiguous range in `segments`:

| Role | Meaning | Required Extra Fields |
|---|---|---|
| `title_author` | Paper title + author/affiliation block on page 1 (never `page_header`) | — |
| `page_header` | Repeated page-level chrome at top (running head, conference banner, preprint notice) | — |
| `page_footer` | Repeated page-level chrome at bottom (copyright/publisher footer; bare numbers are `page_number`) | — |
| `heading` | Section, subsection, or run-in heading | `"level": "section" \| "subsection" \| "runin"` |
| `body` | Main paragraph prose (including paragraphs that merely discuss a figure) | — |
| `figure_caption` | Numbered figure caption (`Figure N:` / `Fig. N:` / `Scheme N:`) + ALL continuation lines | `"num": int \| str`, `"family": "Figure" \| <prefix>` |
| `subfig_caption` | Sub-figure caption (`(a) ...`, `(b) ...`) + continuation lines | `"parent": int \| str \| null`, `"family": <parent family>` |
| `table_title` | Numbered table caption/title (`Table N: ...`) + continuation lines | `"num": int \| str`, `"family": "Table" \| <prefix>` |
| `equation_caption` | Label/number of a standalone display equation, e.g. `(3)` | — |
| `algorithm` | Pseudocode/algorithm float lines INCLUDING its `Algorithm N` title (acts as a cropping wall) | — |
| `figure_content` | Text rendered INSIDE a figure or table graphic (axis ticks/labels, legend entries, diagram nodes, plot panel titles, table cells) | — |
| `footnote` | Footnotes / acknowledgments | — |
| `references` | Bibliography / reference list | — |
| `page_number` | Bare page number | — |
| `other` | Anything else | — |

### Critical Distinctions
- **Captions are both anchors and walls**: every line of a figure caption must be covered under `figure_caption` with the same `num` and `family`. If lines are interleaved, emit multiple ranges with the same `num` and `family`.
- **In-figure text must be `figure_content`**: axis ticks, legends, panel titles, node labels, and tiny symbols inside diagrams are `figure_content`, NOT `body` or `heading` (otherwise the deterministic wall-grower will stop early and slice the figure!).
- **Open Family Interface**: ordinary figures use `"family": "Figure"`. If the document uses another evidenced visual caption prefix (e.g. `Scheme 1`, `Chart 2`, `Plate 1`), set `"role": "figure_caption"` and `"family": "Scheme"` (do not invent families absent from the document).

## 2. Structured Elements (`tables`, `equations`, `algorithms`)

Copy strings and numbers faithfully from the supplied lines—never invent cells, values, or steps. Use `status: "unresolved"` when extracted text cannot recover structure faithfully.

- `tables`: `num`, `caption`, `headers` (`list[str]`), `rows` (`list[list[str]]`), `footnotes` (`list[str]`), `emphasis` (either plain JSON object e.g. `{"bold_cells": [[0, 1]]}` or wire array `[{"key": ..., "value": ...}]`), `source_line_ids`, `status` (`"ok"|"partial"|"unresolved"`), `family` (`"Table"`).
- `equations`: `label`, `latex` (`str | null`), `context`, `source_line_ids`, `status`, `family` (`"Equation"`).
- `algorithms`: `num`, `title`, `inputs`, `outputs`, `steps`, `source_line_ids`, `status`, `family` (`"Algorithm"`).

## 3. Metadata (`metadata`)

- `title`: full printed title (fix line-break hyphenation only).
- `authors`: display names in byline order.
- `institutes`: deduplicated affiliations in first-appearance order.
- `author_index_map`: either plain object `{"Author Name": [1, 2]}` or wire list `[{"author": "Author Name", "institute_indices": [1, 2]}]` (1-indexed into `institutes`).
- `venue`: real conference/journal short name + year only when evidenced (never `"arXiv"` or `"Preprint"`).
- `paper_url`, `code_url`, `project_url`, `arxiv_id`: only when explicitly evidenced in the supplied lines; otherwise `""`.

## 4. Page Outer Boundaries (`page_outer_boundaries`)

Inspect `<work>/semantic/boundary_candidates_compact.json`. If a page has an explicit decorative/running outer frame or page-header/footer rule candidate that separates page chrome from content, select its `candidate_id` with `confidence` (`"high"|"medium"|"low"`) and `reason`; otherwise leave `page_outer_boundaries: []` (or set sides to `candidate_id: null`).
