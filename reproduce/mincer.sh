#!/usr/bin/env bash
# Example: Mincer CFG and earnings-profile SMC evaluation on ACSLabour, one seed.
set -euo pipefail
cd "$(dirname "$0")/.."

root=${DIFFLTM_OUTPUTS:-outputs}
device=${DIFFLTM_DEVICE:-cuda}
slices=slices/acslabour/equality/slices.csv
seed=500
conditional="$root/acslabour/conditional_learnt_ce/$seed"
unconditional="$root/acslabour/unconditional_learnt_ce/$seed"

evaluate_one() {
  local family=$1 method=$2 run=$3
  local out="$run/evaluations/${family}_$method"
  uv run python evaluate.py "run_dir=$run" "device=$device" \
    task=economics "family=$family" "method=$method" "slices=$slices" \
    "output_dir=$out"
}

# Acquire ACSLabour first: bash scripts/download_acslabour.sh
uv run python train.py experiment=conditional dataset=acslabour model=learnt_ce \
  "seed=$seed" "device=$device" "output_dir=$conditional"
uv run python train.py experiment=unconditional dataset=acslabour model=learnt_ce \
  "seed=$seed" "device=$device" "output_dir=$unconditional"

evaluate_one mincer cfg "$conditional"
evaluate_one profiles smc "$unconditional"

uv run python scripts/analyze_econ.py mincer \
  --results "$conditional/evaluations/mincer_cfg" --output "$root/economics/mincer" \
  --cache "${DIFFLTM_DATA:-data}"
uv run python scripts/analyze_econ.py profiles \
  --results "$unconditional/evaluations/profiles_smc" --output "$root/economics/profiles" \
  --cache "${DIFFLTM_DATA:-data}"
