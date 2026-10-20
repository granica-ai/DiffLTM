#!/usr/bin/env bash
# Example: CFG equality evaluation on ACSLabour, one training seed.
set -euo pipefail
cd "$(dirname "$0")/.."

root=${DIFFLTM_OUTPUTS:-outputs}
device=${DIFFLTM_DEVICE:-cuda}
run="$root/acslabour/conditional_learnt_ce/500"

# Acquire ACSLabour first: bash scripts/download_acslabour.sh
uv run python train.py experiment=conditional dataset=acslabour model=learnt_ce \
  seed=500 "device=$device" "output_dir=$run"

uv run python evaluate.py "run_dir=$run" "device=$device" \
  task=conditional method=cfg slices=slices/acslabour/equality/slices.csv \
  "output_dir=$run/evaluations/equality_cfg"
