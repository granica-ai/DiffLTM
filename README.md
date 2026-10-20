# DiffLTM

Code for **DiffLTM: Continuous Diffusions Large Tabular Models for Conditional Generation**.

## Overview

Conditional generation is a key capability to be supported by Large Tabular Models, enabling a number of applications. Accurate conditional generation has to contend with two key challenges: (i) Data comprise a mixture of numerical and categorical entries, the latter often with high cardinality and heavily unbalanced proportions; (ii) Conditional generation becomes increasingly difficult for rare sub-populations. The best models in this class use a hybrid of continuous and discrete diffusions. In this paper, we design a continuous diffusion model with strong generation quality. An important element is the use of cross-entropy loss for categoricals, for which we provide a first theoretical justification.
We introduce metrics that focus on rare subpopulations. Further, we generalize conditional generation techniques to enable new types of conditioning. Finally, we show that our model (trained on labor data) can capture classical insights from econometrics.

This repository contains:

- **Model:** DiffLTM with one-hot and learned categorical representations, and MSE or cross-entropy objectives.
- **Samplers:** unconditional generation, classifier-free guidance, rejection, SMC and Harpoon, supporting equality, range and linear conditions.
- **Evaluation:** fidelity, including rare-subpopulation metrics through [ltm-eval](https://github.com/granica-ai/ltm-eval), machine-learning efficiency (predictive performance when training on generated data) and distance to closest training record (DCR).
- **Experimental protocol:** dataset builders, shared evaluation slices, training configurations and plotting tools for reproducing the DiffLTM experiments and econometric analyses.

## Quick start

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
```

Train a small conditional model on [Adult](https://www.openml.org/d/45068), which is downloaded from OpenML on first use. This takes a few minutes on CPU:

```bash
uv run python train.py experiment=smoke dataset=adult output_dir=outputs/adult \
  training.epochs=50 training.batch_size=1024
```

Evaluate the generated data's fidelity, predictive usefulness and distance to training records:

```bash
uv run python evaluate.py run_dir=outputs/adult task=unconditional
```

Draw conditional samples with classifier-free guidance:

```bash
uv run python evaluate.py run_dir=outputs/adult task=sample method=cfg n=1000 \
  'condition={equality:{sex:"Female"},ranges:{age:[30,40]}}'
```

Each evaluation writes `scores.csv`, its samples and a `manifest.json` to a new directory under `<run_dir>/evaluations/`.

**Settings.** Both entry points are configured with [Hydra](https://hydra.cc). Print the resolved defaults with `uv run python train.py --cfg job`.

| Setting | Options |
|---|---|
| `experiment=` | `unconditional`, `conditional` (CFG, with equality, range and linear conditions), `cps`, `reparam`, `smoke` |
| `model=` | `learnt_ce` (paper default), `onehot_repr`, `onehot_mse`, `onehot_ce` |
| `dataset=` | `adult`, `acslabour`, `acspubcov`, `acsmobility`, `cps_wage_az`, `cps_wage_pa`, `sc_reparam_{level,log,quadratic}` |
| `task=` | `unconditional`, `conditional` (with `slices=<csv>`), `economics`, `sample` |
| `method=` | `cfg`, `rejection`, `smc`, `harpoon`, `ideal`, `unconditional` |

CFG requires a checkpoint trained with conditioning enabled (`experiment=conditional`, `smoke`, `cps` or `reparam`). Rejection, SMC and Harpoon use an unconditional checkpoint.

## Paper reproduction

The [reproduction guide](reproduce/README.md) covers data acquisition and provides editable examples for training, evaluation and plotting.
The [protocol](paper/protocol.md) specifies the experimental settings, and the [experiment list](paper/experiments.csv) lists the configurations and saved results needed for each table and figure.
Shared evaluation slices are included in [`slices/`](slices/); the scripts in [`reproduce/`](reproduce/) demonstrate one dataset, model and random seed at a time.

## Repository structure

```text
train.py, evaluate.py   Hydra entry points for training and evaluation
configs/                dataset, model and experiment configs
diffltm/
  data/                 dataset builders, reusable splits and preprocessing
  models/               DiffLTM model and denoisers
  generation/           conditions, Euler/CFG, rejection, SMC and Harpoon samplers
  economics/            CPS synthetic control, Mincer and reparameterization analyses
  trainer.py            training loop, EMA and checkpoint selection
  evaluation.py         generation and scoring for all tasks
  metrics.py            ltm-eval fidelity adapter, MLE and DCR
  plotting.py           figures from saved results
scripts/                data download, evaluation conditions, econometrics analysis and plotting
reproduce/              example training, evaluation and postprocessing commands
slices/                 fixed conditions for evaluating subpopulations
paper/                  protocol and experiment list
tests/                  test suite (uv run pytest -q)
```

**Documentation.**

- [paper/protocol.md](paper/protocol.md): datasets, model and training, evaluation and econometrics experiments.
- [reproduce/README.md](reproduce/README.md): running and adapting the examples.
- [slices/README.md](slices/README.md): how the evaluation slices were built.

## Citation

Please cite the DiffLTM manuscript when using this software:

```bibtex
@misc{diffltm,
  title     = {{DiffLTM}: Continuous Diffusions Large Tabular Models for Conditional Generation},
  author    = {Alankrita Bhatt and Mukur Gupta and Andrea Montanari and Pulkit Tandon},
  year      = {2026}
}
```

## License

MIT; see [LICENSE](LICENSE). Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
