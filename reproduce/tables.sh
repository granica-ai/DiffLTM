#!/usr/bin/env bash
# Example: unconditional DiffLTM evaluation on ACSLabour, one training seed.
set -euo pipefail
cd "$(dirname "$0")/.."

root=${DIFFLTM_OUTPUTS:-outputs}
device=${DIFFLTM_DEVICE:-cuda}
run="$root/acslabour/unconditional_learnt_ce/500"

# Acquire ACSLabour first: bash scripts/download_acslabour.sh
uv run python train.py experiment=unconditional dataset=acslabour model=learnt_ce \
  seed=500 "device=$device" "output_dir=$run"

uv run python evaluate.py "run_dir=$run" "device=$device" \
  task=unconditional "output_dir=$run/evaluations/unconditional"
