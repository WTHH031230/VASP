"""Sample a method-independent GT list for human review; no model calls or scores."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data import digest, file_sha256, read_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT = ROOT / "results/exp/figure-completeness-ours-p2p-20260908/experiment.json"
GT_FIELDS = ("gt_key", "paper_index", "paper_title", "pdf", "figure_id", "page", "gt_bbox")
CSV_FIELDS = ("sample_id", "gt_key", "paper_index", "paper_title", "figure_id", "page",
              "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "reference_image", "pdf_path")


def select_figures(figures: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Uniform GT sampling without replacement, independent of methods and scores."""

    population = [{key: row[key] for key in GT_FIELDS} for row in figures]
    population.sort(key=lambda row: row["gt_key"])
    if len({row["gt_key"] for row in population}) != len(population):
        raise ValueError("duplicate GT keys in sampling population")
    if not 1 <= size <= len(population):
        raise ValueError(f"sample size must be between 1 and {len(population)}")
    return random.Random(seed).sample(population, size)


def make_package(experiment_path: Path, output_root: Path, size: int, seed: int) -> dict[str, Any]:
    experiment_path = experiment_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing files/annotations: {output_root}")
    experiment = read_json(experiment_path)
    if digest(experiment["identity"]) != experiment["fingerprint"]:
        raise ValueError("experiment fingerprint does not match its frozen identity")
    catalog = experiment["identity"]["catalog"]
    population = sorted(
        ({key: row[key] for key in GT_FIELDS} for row in catalog["figures"]),
        key=lambda row: row["gt_key"],
    )
    selected = select_figures(catalog["figures"], size, seed)
    records = []
    sources = []
    for index, figure in enumerate(selected, start=1):
        metadata = read_json(experiment_path.parent / "inputs" / figure["gt_key"] / "images.json")
        reference = metadata["images"]["reference"]
        if metadata["identity"]["page"] != figure["page"] or reference["bbox_normalized"] != figure["gt_bbox"]:
            raise ValueError(f"GT reference geometry mismatch: {figure['gt_key']}")
        source = (experiment_path.parent / reference["path"]).resolve()
        if not source.is_relative_to(experiment_path.parent):
            raise ValueError("GT image must be inside the source experiment")
        if file_sha256(source) != reference["sha256"]:
            raise ValueError(f"GT reference image changed: {source}")
        if not Path(figure["pdf"]).is_file():
            raise ValueError(f"source PDF missing: {figure['pdf']}")
        sample_id = f"H{index:03d}"
        records.append({
            "sample_id": sample_id, **figure,
            "reference_image": f"references/{sample_id}.png",
            "reference_sha256": reference["sha256"],
        })
        sources.append(source)
    selection = {
        "schema": "dp-p2p.human-figure-sample.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "sample_size": size,
        "population_size": len(population),
        "population_paper_count": len({row["paper_index"] for row in population}),
        "selected_paper_count": len({row["paper_index"] for row in selected}),
        "sampling_unit": "body GT Figure, not paper or method result",
        "algorithm": "random.Random(seed).sample(population sorted by gt_key, sample_size); retain draw order",
        "python_version": platform.python_version(),
        "population_gt_keys": [row["gt_key"] for row in population],
        "population_sha256": digest(population),
        "source_experiment": str(experiment_path),
        "source_experiment_sha256": file_sha256(experiment_path),
        "source_gt_sha256": catalog["gt_sha256"],
        "selected_gt_keys": [row["gt_key"] for row in selected],
        "selection_sha256": digest({"seed": seed, "population": population, "selected": selected}),
        "filters": [],
        "model_calls": 0,
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "references").mkdir()
    for record, source in zip(records, sources):
        destination = output_root / record["reference_image"]
        shutil.copyfile(source, destination)
        if file_sha256(destination) != record["reference_sha256"]:
            raise ValueError(f"copied GT image mismatch: {destination}")
    with (output_root / "sample_list.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **{key: record[key] for key in CSV_FIELDS if key in record},
                **dict(zip(("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"), record["gt_bbox"])),
                "pdf_path": record["pdf"],
            })
    (output_root / "sample_gt_keys.txt").write_text(
        "".join(row["gt_key"] + "\n" for row in records), encoding="utf-8",
    )
    (output_root / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    readme = f"""# Human Completeness Evaluation: Fixed GT Sample Manifest

- seed: `{seed}`; randomly sampled {size} figures without replacement from all {len(population)} main-text GT figures.
- Covers {selection['selected_paper_count']} papers; the sampling unit is the figure, not the paper.
- No filtering by method, match status, IoU, VLM score, or visual difficulty; no re-seeding.
- IDs such as `H001` are fixed by random draw order. All methods must be evaluated on the exact same GT subset without replacing difficult or missed items.

## Files

- `sample_list.csv`: UTF-8 BOM CSV (Excel-friendly) with sample ID, GT key, paper, figure number, PDF page, bbox, GT crop path, and original PDF path.
- `sample_gt_keys.txt`: GT keys only, one per line, for batch processing.
- `selection.json`: Full sampling metadata, population manifest, algorithm, source, and image SHA-256 hashes.
- `references/`: Byte-for-byte copies of the {size} existing GT reference crops (unboxed, unscaled, not re-cropped).

PDF page indices are 1-based and may differ from printed page numbers; `paper_index` is 0-based.
Bounding boxes use the evaluator's 1000x1000 independently normalized coordinates; `reference_image` is relative to this directory.
`pdf_path` in the CSV records the original source path; the archive does not bundle full PDFs.

## Dual-Annotator Verification Protocol

This bundle fixes only the sampled population and GT references; it is not yet a blinded candidate evaluation packet.
For each method, prepare the corresponding matched/aggregated crops for these GT figures while preserving original bboxes and match assignments; missed figures must not be omitted.
During formal adjudication, present the GT reference alongside one anonymized candidate at a time, hiding method name, VLM score, and IoU.
Two annotators score independently and reconcile disagreements; when scoring 0, record the missing element and its location.
Each annotator sheet should contain `sample_id, candidate_id, complete, missing_note`; the method mapping is held separately by the organizer.
Rubric: complete = 1; any actual content missing = 0; whole figure missed = 0; margin differences or tight framing without clipping are not penalized.
Evaluation is anchored to visible content in the GT reference; any flawed GT sample must be flagged and handled uniformly rather than silently replaced.

Existing VLM scores are neither imported into nor modified by this bundle. Report human-subset results separately.

## Reproduction

Run from the repository root (the output directory must not already exist to avoid overwriting manual records):

```bash
python -B -m src.benchmarks.figure_completeness.sample_human \\
  --experiment {experiment_path} \\
  --seed {seed} --size {size} --output-root /tmp/figure-human-sample-reproduction
```
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    archive = output_root / f"gt_sample_{size}_seed{seed}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, path.relative_to(output_root))
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selection = make_package(args.experiment, args.output_root, args.size, args.seed)
    print(json.dumps({key: selection[key] for key in (
        "seed", "population_size", "sample_size", "selected_paper_count", "selection_sha256", "model_calls",
    )}, ensure_ascii=False, indent=2))
    print(f"output_root: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
