# Figure Crop Benchmark

## Current primary protocol

The current fair crop-only comparison is the GT-conditioned protocol documented
in [`GT_CONDITIONED_CROP_PROTOCOL.md`](GT_CONDITIONED_CROP_PROTOCOL.md). It
produces two separate reports:

- crop quality over every GT Figure (`mIoU`, `R@50`, `R@75`, `mAR@50:95`), with
  missing GT scored as zero and unassociated visual candidates excluded;
- a non-scoring candidate audit that reports proposal volume, Figure association,
  and fragmentation without treating valid but unannotated Tables as false positives.

Method-provided Figure/Table labels are ignored. This protocol is intentionally
not COCO AP because the local benchmark has Figure-only GT and no exhaustive
non-Figure ignore annotations.

This package evaluates full-paper Figure assets against the derived
`results/figure-caption-pairs/` ground truth. GT boxes are normalized to an
independent 1000 x 1000 page canvas; active paper2assets boxes are read in PDF
points and normalized using the corresponding source page dimensions.

Run the 100-paper body-Figure evaluation without any model calls:

```bash
python scripts/benchmark/run_figure_crop_bench.py \\
  results/asset_extraction/gpt5.6-sol \\
  results/eval/figure_crop/asset_extraction-gpt5.6-sol \\
  --method asset_extraction-gpt5.6-sol
```

The primary outputs are `summary.json`, `summary.md`, `per_paper.csv`, and
`per_figure.csv`. Spatial detection uses same-page maximum-IoU matching. A
usable crop defaults to at least 95% GT coverage, at most 10% contamination,
and an existing image file.

Label-only detection is reported separately for image-only baselines whose
metadata lacks a source page bbox. Such baselines do not receive geometric crop
scores until an adapter can recover their source page and box.
