# GT-conditioned Figure crop protocol

This protocol separates crop geometry from semantic output precision for the
100-paper assets benchmark. The available ground truth contains Figure boxes
only; it does not contain exhaustive Table or other non-Figure ignore boxes.

## Candidate input

Each adapter exposes the visual crops and native page bboxes retained by the
corresponding poster method. When a parser exposes Figure and Table channels,
both are included in the candidate pool. Method-provided semantic categories
are recorded for audit but never used to include, exclude, or score a crop.

Page indices are zero based in adapter manifests. Every bbox is additionally
normalized to a top-left 1000 x 1000 page coordinate system.

## GT-conditioned association and grouping

For a prediction and GT Figure on the same page, an association is eligible
when either box contains the center of the other. Each prediction is assigned
to at most one eligible GT by, in order:

1. intersection divided by the smaller box area;
2. IoU;
3. absolute intersection area;
4. GT annotation order as a deterministic tie-break.

All components assigned to one GT Figure are replaced by their outer-union
bbox. A GT with no assigned component receives IoU zero. An in-scope candidate
with no Figure association is retained in the audit output but does not affect
crop-quality metrics, because the Figure-only GT cannot tell whether it is a
valid Table or a meaningless crop.

## Primary crop-quality table

For each GT Figure g, let q_g be the IoU of its grouped prediction and GT bbox,
or zero when the Figure is missing. With N GT Figures:

- **mIoU** = sum(q_g) / N;
- **R@50** = count(q_g >= 0.50) / N;
- **R@75** = count(q_g >= 0.75) / N;
- **mAR@50:95** = the mean of R@t for t in {0.50, 0.55, ..., 0.95}.

These are GT-conditioned crop metrics, not COCO average precision. They measure
recoverable crop geometry under oracle GT ownership and must not be described
as end-to-end Figure classification or attribution performance.

## Candidate-audit table

The separate audit reports the number of in-scope visual candidates, components
associated with Figure GT, unassociated candidates, covered/missing GT Figures,
multi-component GT groups, and components per covered GT. An unassociated
candidate is deliberately not called a false positive.

## Command and outputs

```bash
python scripts/benchmark/run_gt_conditioned_figure_crop_bench.py \
  --prediction-root results/exp/native_bbox_all_visual \
  --output-root results/exp/gt_conditioned_figure_crop_body_all_visual \
  --scope body
```

The root output contains `crop_quality_table.md`, `candidate_audit_table.md`,
the corresponding CSV/JSON files, and a combined comparison. Each method also
contains `summary.json`, `per_gt_figure.csv`, `per_component_assignment.csv`,
and overlay-compatible page-bbox JSON files.
