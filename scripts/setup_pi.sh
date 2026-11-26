#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/setup_pi.sh"
  echo "Installs the latest Pi CLI (@earendil-works/pi-coding-agent) and google-auth-library under src/viscomposer/pi/."
  exit 0
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/src/viscomposer/pi"
npm install
