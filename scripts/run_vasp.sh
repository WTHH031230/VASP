#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"

PDF=""
OUT=""
BACKEND="pi-api"
MODEL="gpt-5.6-sol"
EFFORT="medium"
PROVIDER="openai-compat"
ARM="full"
MAX_REVISIONS="2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pdf) PDF="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --effort) EFFORT="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --max-revisions) MAX_REVISIONS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --pdf <paper.pdf> --out <output_dir> [--backend pi-api|pi-chatgpt|codex|claude|agy] [--model gpt-5.6-sol] [--effort medium] [--provider openai-compat] [--arm full] [--max-revisions 2]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PDF" || -z "$OUT" ]]; then
  echo "Error: --pdf and --out are required. Run with --help for usage." >&2
  exit 1
fi

PDF_ABS="$(cd "$(dirname "$PDF")" && pwd)/$(basename "$PDF")"
PAPER_NAME="$(basename "$PDF_ABS" .pdf)"
if [[ "$PAPER_NAME" == "paper" ]]; then
  PAPER_NAME="$(basename "$(dirname "$PDF_ABS")")"
fi
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

STAGE_IN="$OUT_ABS/_input_paper"
STAGE1_OUT="$OUT_ABS/stage1_assets"
STAGE2_OUT="$OUT_ABS/stage2_poster"
mkdir -p "$STAGE_IN/$PAPER_NAME"
ln -sf "$PDF_ABS" "$STAGE_IN/$PAPER_NAME/paper.pdf"

cd "$ROOT"

echo "[VASP Stage 1/2] Running SemParser (backend=$BACKEND, model=$MODEL)..."
"$PY" -m src.semparser.run "$STAGE_IN" "$STAGE1_OUT" \
  --include-appendix \
  --backend "$BACKEND" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --effort "$EFFORT"

ASSETS_DIR="$STAGE1_OUT/$PAPER_NAME"
if [[ ! -d "$ASSETS_DIR" ]]; then
  ASSETS_DIR="$STAGE1_OUT/assets/$PAPER_NAME"
fi

echo "[VASP Stage 2/2] Running VisComposer (designer=$BACKEND, arm=$ARM, model=$MODEL)..."
"$PY" -m src.viscomposer.run \
  --arm "$ARM" \
  --asset-dir "$ASSETS_DIR" \
  --output-dir "$STAGE2_OUT" \
  --designer "$BACKEND" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --effort "$EFFORT" \
  --max-revisions "$MAX_REVISIONS"

echo "[VASP Done] Outputs saved under $OUT_ABS"
