# Reproduction examples

These editable scripts demonstrate training, evaluation and postprocessing for one dataset, model and random seed, running the configured number of training epochs. The [protocol](../paper/protocol.md) describes the settings; the [experiment list](../paper/experiments.csv) records the settings and result paths for each table and figure.

## Data

| Dataset | Source | Preparation |
|---|---|---|
| ACSLabour | U.S. Census survey records for individual people | `bash scripts/download_acslabour.sh` (52 ZIPs, about 2 GB) |
| ACSPubCov, ACSMobility | U.S. Census files downloaded by the Folktables library | Downloaded on first use |
| CPS | Current Population Survey earnings records | Downloaded on first use |
| Synthetic reparameterization | Reference-data simulation settings in `diffltm/data/dgp.json` | Generated locally |
| Adult | OpenML dataset 45068, specified in `diffltm/data/openml_registry.py` | Downloaded on first use |

Raw files, processed tables and reusable train/validation/test splits are stored under `data/`. Set `DIFFLTM_DATA` to use another location. Set `DIFFLTM_ACSLABOUR_RAW` if the ACSLabour ZIPs are stored elsewhere.

## Run an example

```bash
bash reproduce/tables.sh       # ACSLabour, unconditional learnt_ce, seed 500
bash reproduce/conditional.sh  # ACSLabour, CFG equality conditions, seed 500
bash reproduce/mincer.sh       # ACSLabour, Mincer CFG and earnings-profile SMC, seed 500
bash reproduce/CPS_minwage.sh  # CPS Arizona, wage time series and subgroup distributions, seed 500
```

The scripts default to CUDA and save under `outputs/`. Set `DIFFLTM_DEVICE=cpu` or `DIFFLTM_OUTPUTS=/path/to/outputs` as needed. Edit the command-line configuration arguments to change the dataset, model or seed. For a small CPU example, use the [quick start](../README.md#quick-start).

Training reuses a completed run only when its configuration matches. Failed or incomplete runs need a fresh training directory. Evaluation requires a new output directory and never overwrites existing results. To evaluate a completed checkpoint again, run its `evaluate.py` command with a fresh `output_dir`; pass that directory to postprocessing. To redo postprocessing alone, run the final command in `CPS_minwage.sh`.

Conditional examples use the supplied slices. Change `slices=.../equality/slices.csv` to `range/slices.csv` or `linear/slices.csv` for other tasks. Train CFG models with `experiment=conditional`; rejection, SMC and Harpoon use `experiment=unconditional`. Harpoon supports `onehot_repr` and `learnt_ce`. Non-linear `arbit_num` conditions support rejection, SMC and `ideal` only.

## Plot saved results

Plotting reads completed evaluations and writes PDF, PNG, plotted values and a record of the input files. For the single-seed Arizona example, create a CSV listing the figures to plot, their result files and the training seed:

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ.get("DIFFLTM_OUTPUTS", "outputs"))
inventory = pd.read_csv("paper/experiments.csv")
inventory = inventory.loc[inventory.output.isin(["figure3", "figure9"])].copy()
inventory["seeds"] = "[500]"
inventory["results"] = str(root / "economics/cps_wage_az/cdfs.csv")
inventory.to_csv(root / "cps_figures.csv", index=False)
PY
uv run python scripts/plot_results.py \
  --manifest "${DIFFLTM_OUTPUTS:-outputs}/cps_figures.csv" \
  --output "${DIFFLTM_OUTPUTS:-outputs}/figures"
```

For other experiments, select the relevant rows from `paper/experiments.csv` and set their `results`, `seeds`, `sampler` and `selection` to match the saved evaluations.

The supplied `paper/experiments.csv` expects seed 500 for CPS and seeds 500, 501 and 502 for other experiments. Once the corresponding results are available, plot selected outputs directly:

```bash
uv run python scripts/plot_results.py --manifest paper/experiments.csv --only figure3 figure9
```

Use the CSV columns `figsize`, `xlim` and `ylim` to adjust figure size and axis limits.
