#!/usr/bin/env bash
# Example: Arizona wage paths and subgroup CDFs, one training seed.
set -euo pipefail
cd "$(dirname "$0")/.."

root=${DIFFLTM_OUTPUTS:-outputs}
device=${DIFFLTM_DEVICE:-cuda}
run="$root/cps_wage_az/cps_learnt_ce/500"
evaluation="$run/evaluations/cps"

uv run python train.py experiment=cps dataset=cps_wage_az model=learnt_ce \
  model.backbone=mlp training.epochs=2000 \
  seed=500 "device=$device" "output_dir=$run"

uv run python evaluate.py "run_dir=$run" "device=$device" \
  task=economics family=cps method=cfg "output_dir=$evaluation"

uv run python scripts/analyze_econ.py cps --results "$evaluation" \
  --output "$root/economics/cps_wage_az" --cache "${DIFFLTM_DATA:-data}"
