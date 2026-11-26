"""Single-pass paper adaptation, complete example and fixed output contract."""

import json

from semantic_schema import to_wire

GUIDANCE = """
SINGLE-PASS PAPER ADAPTATION:
In this same analysis, recognize this paper's evidenced caption conventions,
wrapped/interleaved captions, panel labels, page chrome, and body/appendix
numbering. Apply those observations directly to all supplied pages. Return only
the final extraction; do not generate a separate prompt, profile, or explanation.
Keep the original closed roles and source-line IDs. A section heading is
role=heading with level=section; section is not a role. Recognize every supplied
element, including elements outside the examples you first notice.

STRICT TRANSPORT REPRESENTATION:
The complete output example below uses the exact attached response schema.
author_index_map is an array of {author, institute_indices} records, one per
author; the host mechanically converts it to the existing author-name map.
Table emphasis is an array of {key, value} records. An empty map is []. Values
can be JSON scalars or arrays. A nested dictionary is represented as
{entries: [{key, value}, ...]} recursively. For example, the emphasis map
{"bold_cells": [[0, 1]]} is [{"key": "bold_cells", "value": [[0, 1]]}.
Preserve evidenced information; do not invent emphasis. Keys/authors in a map
must be unique. All strings, including LaTeX, must be valid JSON string values.
Each source range must satisfy 0 <= start <= end < the supplied line count.
"""

FAMILY_CONTRACT = """
OPEN FAMILY INTERFACE (fixed downstream contract):
- Existing processing roles and their semantics are unchanged. A numbered
  raster/vector visual with a caption prefix other than Figure (e.g. Scheme,
  Chart, Plate) can use role=figure_caption with family equal to its evidenced
  caption prefix. Ordinary Figure/Fig captions use family=Figure. Do not invent
  a family absent from the paper. Continue ordinary numbering in num, not family.
- Add family to numbered caption segments and subfig_caption segments; a
  subfig_caption uses its parent's family. All segments of one caption have the
  same family and num. Numbered tabular families reuse table_title and the tables
  structured array; pseudocode families reuse algorithm and algorithms;
  equations reuse equation_caption and equations. Also preserve family on
  structured entries. A family has exactly one processing capability.
- family is a name, NOT a processing role or geometry rule. Never output new
  roles, code, bounding boxes, crop directions, thresholds, or geometry settings.
  Figure-like captions remain both anchors and walls. Internal diagram/plot
  text stays figure_content; do not impose strict ownership of every internal
  text token. Tables remain structured data and internal geometry diagnostics,
  not public Figure images. Family+number identifies an element within its
  body/appendix scope; Scheme 1 must not become Figure 1.
"""

def prompt_example(base_task):
    _, sample_text = base_task.split("Output ONE JSON object, no fences:\n", 1)
    sample = json.loads(sample_text)
    for seg in sample["segments"]:
        if seg["role"] == "figure_caption":
            seg["family"] = "Figure"
    for item in sample["tables"]:
        item["family"] = "Table"
    sample["page_outer_boundaries"] = []
    return to_wire(sample)

def build_prompt(base_task, numbered, boundaries):
    instructions, _ = base_task.split("Output ONE JSON object, no fences:\n", 1)
    instructions = instructions.replace(
        "headers and rows are arrays of strings.",
        "headers is an array of strings; rows is an array of arrays of strings.")
    instructions = instructions.replace(
        "author_index_map: map every author name to 1-indexed positions in institutes,",
        "author_index_map: list {author, institute_indices} records mapping each author to 1-indexed positions in institutes,")
    # Preserve the original instruction -> full example -> paper -> boundary order.
    return (instructions + GUIDANCE + FAMILY_CONTRACT +
            "\nOutput ONE JSON object, no fences (illustrative values only):\n" +
            json.dumps(prompt_example(base_task), ensure_ascii=False, indent=2) +
            "\nLines:\n" + numbered + boundaries)
