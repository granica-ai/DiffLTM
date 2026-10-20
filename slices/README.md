# Evaluation slices

A slice is the subset of rows satisfying a condition, such as a particular education level. Use `slices/<dataset>/<task>/slices.csv` with `evaluate.py task=conditional`. All methods and training seeds use the same supplied conditions.

Each dataset/task directory contains:

- `slices.csv`: conditions, rarity groups and the number of matching rows in the reference split. Rarity groups indicate how frequently each condition occurs.
- `run_counts.csv`: the number of matching train/validation/test rows for seeds 500, 501 and 502.
- `manifest.json`: dataset and split identifiers, construction settings and checksums used to detect changed files.

To verify the supplied conditions against locally built data, or generate missing sets of conditions:

```bash
uv run python scripts/make_slices.py acslabour
```

The command uses data stored under `data/` (or `DIFFLTM_DATA`) and checks existing complete files before reusing them. It reports an error if the data or condition definitions do not match. This keeps evaluations tied to the same observations and train/validation/test assignments.

See the [protocol](../paper/protocol.md#conditional-evaluation-figure-1-figures-8–18) for task definitions, condition construction and scoring, and its [econometrics experiments section](../paper/protocol.md#econometrics-experiments) for Mincer analysis.
