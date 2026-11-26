#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  cat << "USAGE"
Usage: bash scripts/run_ablation.sh <variant> [args...]

Component & Composition-Priority Ablations (forwarded to src.viscomposer.run --arm <arm>):
  no_assets           w/o SemParser (ResearchStudio assets)
  no_vf_template      w/o anchored composition (fixed layout template)
  no_vf               w/o anchored composition (bare)
  text_first          w/o anchored composition (text-first on identical flexible grid)
  no_sem              w/o semantic verification (w/o iter. SV & GV)
  no_basic            w/o refinement loop (single-pass draft)

Verification-Decoupling Ablations (src/ablation/):
  wo_iter_sv          w/o iterative SV (init. SV + GV)
  wo_sv               w/o all SV (GV only)
  wo_geom             w/o GV (init. + iter. SV)

Visual-Theme Sensitivity (src/ablation/theme_sensitivity/):
  theme_sensitivity   Apply CSS themes T0-T8 onto frozen layout HTML
USAGE
  exit 0
fi

VARIANT="$1"
shift
cd "$ROOT"

case "$VARIANT" in
  no_assets|no_vf|no_vf_template|no_vf_text_first|text_first|no_sem|no_basic|basic_only|basic_assets)
    exec "$PY" -m src.viscomposer.run --arm "$VARIANT" "$@"
    ;;
  wo_geom|wo_sv|wo_iter_sv)
    exec "$PY" -m "src.ablation.run_${VARIANT}" "$@"
    ;;
  theme_sensitivity)
    exec "$PY" -m src.ablation.theme_sensitivity.apply_themes "$@"
    ;;
  *)
    echo "Unknown ablation variant: $VARIANT. Run with --help for all variants." >&2
    exit 1
    ;;
esac
