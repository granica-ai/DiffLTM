# Protocol

This document describes the datasets, training, evaluation and econometrics experiments. The [experiment list](experiments.csv) records the datasets, settings and result files needed for each table and figure. See [reproduce/README.md](../reproduce/README.md) for commands. Experiments use training seeds 500, 501 and 502, except CPS, which uses seed 500 only. The example scripts demonstrate one training seed.

## Datasets

ACS is the U.S. Census Bureau's American Community Survey. CPS is the Current Population Survey; its earnings records are distributed by the National Bureau of Economic Research (NBER). Row counts below are after filtering and removing observations reserved for the econometrics comparisons, but before the train/validation/test split.

| Dataset | Data source and included observations | Rows available for splitting | Prediction target |
|---|---|---|---|
| ACSLabour | ACS 2018 five-year release; employed wage earners aged 16 or older in all 50 states, DC and Puerto Rico, with positive wages and working hours | 6,892,317 | Log annual wage income (`log_wage`, regression) |
| ACSPubCov | Annual ACS files for 2014–2018 combined into one table; all 50 states and Puerto Rico; people younger than 65 with income at most $30,000 | 5,916,565 | Public health-insurance coverage (`PUBCOV`, classification) |
| ACSMobility | Annual ACS files for 2014–2018 combined into one table; all 50 states and Puerto Rico; people aged 19–34 with recorded commute times | 2,041,325 | Mobility status (`MIG`, classification) |
| CPS (`cps_wage_az`, `cps_wage_pa`) | Earnings records for 2010–2019; hourly-paid private-sector workers aged 16–64. The two datasets differ in which state's 2017–2019 observations are reserved for comparison | AZ: 486,686; PA: 485,042 | Log hourly wage, adjusted to 2019 dollars (`LOG_REAL_HOURWAGE`, regression) |
| Synthetic reparameterization (`sc_reparam_level`, `_log`, `_quadratic`) | Reference observations simulated independently of the trained models for 11 states over 11 time periods, with 1,200 rows per state and period; generated using `diffltm/data/dgp.json` and seed 42 | 141,600 | The same outcome expressed as `Y`, `log Y` or `Y(M−Y)` (regression) |
| Adult (`adult`) | OpenML dataset 45068, specified in `diffltm/data/openml_registry.py`; used for the quick-start example | See source dataset | Whether income exceeds $50,000 (classification) |

The synthetic reparameterization datasets are simulated reference data for the outcome-scale experiment. They are used to train and evaluate models; throughout this protocol, rows produced by a trained model are called **generated rows**.

PubCov has 19 features and Mobility 20. Mobility excludes the employment-status-of-parents column `ESP`, which has no recorded values for the included people. ACS category codes are kept as recorded. A missing categorical value is its own category, `__NA__`.

**Observations reserved for econometrics comparisons.** These rows are excluded before the train/validation/test split. They are used to compare model-generated outcomes with observed outcomes:

- CPS: Arizona (`cps_wage_az`) or Pennsylvania (`cps_wage_pa`) for 2017–2019.
- Synthetic reparameterization: state `S=6` in periods `T=8,9,10`, when the simulated intervention takes effect. This removes 3,600 rows from the full 145,200-row dataset.

For the quadratic outcome `Y(M−Y)`, `M` is the largest `Y` in the simulated reference data after these exclusions. It is computed once and shared by all training runs.

**Splits.** Each run assigns 70% of rows to training, 10% to validation and 20% to testing, using the run seed. For classification, the split preserves the target class proportions.

**Preprocessing**, fitted on the training split only:

- **Numerical columns:** a transform based on 1,000 training quantiles maps values to an approximately normal distribution. The synthetic reparameterization datasets (`sc_reparam_*`) instead use standardisation: subtract the training mean and divide by the training standard deviation. Missing values are filled with the training mean. Integer-valued columns (ages, hours, commute time) are rounded when decoding.
- **Categorical columns:** coded with a reserved "unknown" category, which is never generated. CPS `YEAR` is categorical.

## Model and training

**Diffusion.** DiffLTM is a continuous diffusion model over a vector containing the transformed numerical values and one representation vector per categorical column. It uses the noise schedule `σ(t) = (σ_min^{1/ρ} + t(σ_max^{1/ρ} − σ_min^{1/ρ}))^ρ` with `(σ_min, σ_max, ρ) = (0.002, 80, 7)`, without preconditioning or loss weighting. Sampling uses 100 Euler steps.

**Categorical representations:**

- **One-hot:** a vector with one entry per category, set to one for the observed category and zero elsewhere.
- **Learned embeddings**, normalised to norm `√d`, with `d = 16` (4 for synthetic reparameterization).

**Objectives.** For categorical columns, the denoiser predicts either the representation itself (MSE) or logits. Logits are trained with MSE on their probability-weighted mean representation, or with cross-entropy:

- **Numerical loss:** the mean squared error.
- **Categorical MSE:** averaged within each column over its coordinates, then across columns.
- **Cross-entropy:** averaged across columns.
- **Total loss:** the sum of the numerical and categorical losses.

**Decoding.**

- Logit models decode each category as the argmax of the last step's logits.
- Representation models decode each category as the nearest code.

**Denoising networks.** MLP means multilayer perceptron; AdaLN adds adaptive layer normalisation to incorporate the diffusion time.

| Network | Used for | How diffusion time enters the network |
|---|---|---|
| MLP, width 1024, depth 5 | ACS and CPS | time added at the input |
| AdaLN MLP | Synthetic reparameterization | time-modulated hidden blocks |

**Conditional models (classifier-free guidance).** The denoiser receives condition values and masks identifying the active constraints:

- **Equality:** for a numerical or categorical column, replace its noisy state with a learned token and supply the target value or categorical representation separately.
- **Range:** supply lower and upper bounds in transformed numerical coordinates.
- **Linear:** for `L ≤ wᵀz ≤ U` in standardised raw units, supply `w`, `L`, `U` and `[s, s−L, U−s]`. Here `s = wᵀz` is recomputed from the current noisy state using an interpolated inverse numerical transform.

Training conditions contain the corresponding clean training row. For each row, draw a probability uniformly from `[0,1]`, then independently select columns with that probability. Selected categorical columns use equality conditions; selected numerical columns use equality or range conditions with equal probability. Range widths are log-uniform between 2% and 100% of the transformed training span, with bounds clipped to that span.

With probability 0.5, also add a linear constraint over a randomly chosen subset of numerical columns, using a normalised Gaussian direction. Its width is log-uniform between 2% and 100% of the span implied by the standardised training minima and maxima. Columns involved in this constraint do not also receive equality or range conditions. Finally, drop the whole condition with probability 0.15.

A guidance weight of 1 gives the conditional prediction. These mixed training conditions differ from the fixed, separate equality, range and linear tasks used for evaluation below.

**Training.** The experiment configurations below use AdamW (learning rate 3e-4, weight decay 1e-4), cosine decay, gradient clipping at 1.0, bf16 mixed precision on GPU, and an exponential moving average (decay 0.997) of the denoiser weights.

| Experiment | Network | Training epochs | Batch size | Validation every |
|---|---|---|---|---|
| ACS, unconditional and conditional | MLP 1024 × 5 | 2,000 | 8,192 | 10 epochs |
| CPS | MLP 1024 × 5 | 2,000 | 8,192 | 10 epochs |
| Synthetic reparameterization | AdaLN 256 × 1, embedding dim 4 | 1,000 | 4,096 | 20 epochs |


**Checkpoint selection.**

- Every run trains for its full budget, with no early stopping.
- The evaluated checkpoint has the lowest validation loss, computed with the averaged denoiser.
- A run that hits a non-finite training loss fails and is never evaluated.

**Periodic evaluation.** Let `N_train`, `N_val` and `N_test` denote the numbers of training, validation and test rows. During training, the model periodically generates `N_train` unconditional rows. Fidelity and machine-learning efficiency (MLE) are scored on validation data in a background process. These diagnostics never use test data and do not influence checkpoint selection.

## Evaluation and metrics

### Samplers

A sampler produces rows from a trained model. For conditional generation, adherence is the fraction of generated rows that satisfy the condition.

| Sampler | Model | Description |
|---|---|---|
| CFG | conditional | Guidance weight 1. Equality-conditioned columns are set to their target values after sampling; adherence is reported before and after. |
| Rejection | unconditional | Filters a shared stream of unconditional samples, using at most the first 10 million candidates per run. If too few accepted rows are available, the condition’s scores are reported as missing. |
| SMC | unconditional | Sequential Monte Carlo with ancestral transitions, four replicates of at least 8,192 particles, and resampling when the effective sample size (ESS), divided by the particle count, is below 0.95. The final step keeps only rows that satisfy the condition exactly. |
| Harpoon | unconditional `onehot_repr` or `learnt_ce` | Euler sampling with a guidance step of size η = 0.2. The step follows the gradient of an L1 equality loss plus separate L2 norms of lower and upper interval violations. Output is not adjusted; adherence is reported as generated. |
| Ideal | none | Real training rows matching the condition, as a reference. |

We tune Harpoon's guidance step size η on ACSLabour and find that 0.2 performs best. We use this value for all other datasets as well.

For learned-embedding/CE checkpoints, Harpoon applies its equality penalty to probability-weighted normalized embeddings. Categorical output uses the last integration prediction's logits, so the final correction does not directly change the returned categorical values.

**Range-bound snapping.** The `snap_ranges` behavior is implemented by `transform_num_ranges`: it adjusts transformed lower and upper bounds to the first and last model coordinates that decode inside the original interval. This accounts for quantile-transform ties and integer rounding, which a direct transform of the endpoints can miss. CFG additionally clips these bounds to the transformed training span. Adherence is always checked against the original bounds in raw units.

For unconditional Euler, CFG and Harpoon sampling, a row whose final numerical/representation state or categorical logits contain NaN or infinity is discarded and redrawn with the same settings. Generation fails if discarded rows, including failed redraws, exceed `max(10, ceil(0.001 × n))`, where `n` is the number of rows to generate (use `0.1 × n` for Harpoon). Euler and CFG also allow at most three redraw rounds per batch; Harpoon has no separate round limit. CFG sets equality-conditioned columns to their target values as described above. Rejection instead filters candidates, and SMC excludes terminal condition violations through its weights and resampling; SMC does not use this row-redraw policy.

### Unconditional evaluation (Tables 1–2)

Each run generates `N_train` rows once:

- **Fidelity** compares `N_test` generated rows with the full test split. It is computed with [ltm-eval](https://github.com/granica-ai/ltm-eval) 0.1.0 (`clean="safe"`) and reports:
  - **Shape:** per-column similarity, one minus the Kolmogorov–Smirnov (KS) distance for numerical columns, and one minus the total variation (TV) distance for categorical ones.
  - **Trend:** pairwise similarity, using correlation for numerical pairs and contingency tables otherwise.
  - **Conditional Shape, all groups and rare groups:** per-column distributions within groups defined by each conditioning column, with 10 bins, a minimum group size of 10, and rare meaning frequency below 5%.
- **Machine-learning efficiency (MLE):** the ratio of predictive performance when training on generated rows to performance when training on real rows, evaluated on the same real test split (details below).
- **Distance to closest record (DCR):** median distance from each generated row to its nearest training row, divided by the same median for test rows. Each table is capped at 10,000 rows, with scaling fitted on training data.

**MLE computation (Appendix B.4).** Reuse the `N_train` generated rows from this run. In each of ten rounds:

1. Sample `N_test` rows without replacement from the real training split and, separately, `N_test` rows from the generated pool. Rows may appear again in other rounds.
2. Fit a fresh XGBoost model on each subset (100 trees, depth 10, learning rate 0.1, seed 42). The target is excluded from the features; feature encoders and scaling are fitted once on the full real training split.
3. Evaluate both models on the full real test split, using ROC-AUC for classification or R² for regression. These are the train-real/test-real (TRTR) and train-generated/test-real (TSTR) scores.

Average the ten TSTR scores and the ten TRTR scores separately, then compute `MLE = mean(TSTR) / mean(TRTR)`. The ten rounds vary the training subsets, not the XGBoost seed. This gives one MLE value per generative-model training seed; the table reports the mean and standard deviation across those seeds.

Undefined scores are reported as missing, not as zero.

### Conditional evaluation (Figures 1, 8 and 10–20)

**Conditions** come from the supplied files in `slices/<dataset>/<task>/`. Each condition defines a subset of rows, called a slice. The same file is used by all methods and seeds. There are three tasks, each with 50 conditions per dataset, ten in each of five groups defined by how often the condition occurs: >3%, 1–3%, 0.3–1%, 0.1–0.3% and ≤0.1% of training rows.

| Task | Construction |
|---|---|
| Equality | One to four categorical columns, with values copied from a random training row. |
| Range | One to four numerical columns. Each interval covers a random quantile range of width 0.2%–80%. |
| Linear | A random unit direction over one to four numerical columns, in standardised units, with an interval around a training row's projection. |

The number of constrained columns is chosen uniformly from one to four, limited by the available columns. Numerical-range widths are log-uniform quantile fractions in `[.002,.8]`; linear widths are log-uniform fractions in `[.02,1]` of the possible range of the linear combination, computed from the minimum and maximum of each standardised numerical column. Linear standardisation uses training means and population standard deviations (`ddof=0`). For integer columns, range lower bounds round down and upper bounds round up.

A condition must:

- select a set of reference-training rows different from every other condition;
- exclude at least one training row;
- match at least 100 test rows in every run;
- select at least two distinct values in each range-constrained numerical column.

Rarity is measured on the training portion of a fixed reference split (seed 0), so rarity groups are identical for all runs. ACSLabour also has a small `arbit_num` task of non-linear conditions on numerical columns, used with rejection, SMC and the ideal training-data reference.

**Sample counts.**

- CFG, SMC and Harpoon generate `min(n_train^C, 100,000)` rows, where `n^C` counts rows matching condition C.
- Rejection retains the first `min(n_test^C, 100,000)` accepted rows.
- The ideal reference uses `min(n_train^C, n_test^C, 100,000)` real training rows.
- Fidelity compares `min(n_generated, n_test^C, 100,000)` generated rows with the same number of real test rows matching the condition, using all modelled columns. Generated rows are not filtered for adherence.

Rejection uses `n_test^C` because fidelity scoring needs at most that many accepted rows, as in Appendix C.2. Generating up to `n_train^C` accepted rows would add rejection-sampling work without increasing the number of rows scored. The other samplers retain their training-count generation budgets; scoring still uses matched counts. If rejection exhausts its candidate budget, its scores are missing rather than computed on an incomplete sample.

**Aggregation.** For each method and metric:

1. Average each condition's score over the available seeds.
2. Within each rarity group, report the mean and standard error across conditions, with each condition weighted equally.

Undefined scores are omitted at each stage, and counts of valid seeds and conditions are recorded. Unconditional tables average one score per training seed. Outcome-scale and earnings-profile plots average one estimate per seed at each time period or age group. Their standard error is the sample standard deviation across seeds divided by the square root of the number of seeds. CPS uses only seed 500, so its plots show that run’s estimates without error bands across seeds; one seed is insufficient to estimate this uncertainty.

## Econometrics experiments

**CPS minimum wage (Figures 2, 3, 9).**

- **Training:** one conditional model for the Arizona holdout dataset and another for the Pennsylvania holdout dataset, both at seed 500 only.
- **Generation:** 100,000 rows per year from 2010 to 2019, conditioning on the evaluated state and year.
- **Wage time series:** the 10th percentile of log hourly wage adjusted to 2019 dollars, converted back to each year's dollar value for plotting.
- **Synthetic control:** estimate the evaluated state's wage time series as a weighted combination of other states' log-wage time series. These comparison states are called donors. Fit non-negative weights that sum to one using observed data from 2010–2016 and standardised predictors. Exclude the evaluated state from the donor set.
- **Subgroup distributions:** generate 100,000 rows per Arizona subgroup/year in 2016 and 2019, regardless of how many observed or test rows match the condition. Condition on `STATE,YEAR,EDUC` for high school/bachelor's and `STATE,YEAR,OCC_MAJOR` for healthcare/cleaning/sales. Including the state/year samples, Arizona therefore generates 2,000,000 rows across 20 conditions; Pennsylvania generates 1,000,000 across ten state/year conditions.
- **Cumulative distribution functions (CDFs):** show the fraction of hourly wages at or below each dollar value, using all observed rows and all 100,000 generated rows separately. Wages are expressed in each year's dollars. The CPS figures show the empirical CDFs for seed 500 only. The curves retain their exact steps; they are not smoothed.
- **Comparison statistics:** KS distance and the share earning below $11/hour use equal-sized observed and generated samples: `min(N_real, N_generated, 100000)` rows each. Subsampling uses the saved scoring seed. In `cdfs.csv`, `n` records the full count used for a curve, while `n_scored` records the count used for these comparisons.

**Synthetic reparameterization: changing the outcome scale (Figure 4).**

- **Training:** separate models on `Y`, `log Y` and `Y(M−Y)`.
- **Generation:** 100,000 rows for the simulated intervention state (`S=6`) at each time period (`T=0,…,10`).
- **Synthetic control:** fit separately for each outcome scale, using the reference data after the intervention observations have been excluded. Fit the weights using periods before the intervention; other simulated states supply the comparison time series.
- **Known reference:** calculate the quadratic outcome's true median from the distribution used to simulate the data.
- **Values outside the possible range:** keep generated values outside the range allowed by the simulation and report their proportion.
- **Reporting:** calculate the generated median for each time period and training seed, then report the mean and standard error across seeds.

**Mincer equation and earnings profiles (Figures 5–6).** These use ACSLabour models.

The one-seed example in [`reproduce/mincer.sh`](../reproduce/mincer.sh) evaluates Mincer with CFG and earnings profiles with SMC.

- **Mincer:** fits the regression of log wage on schooling, experience and experience² identically on real test rows and generated rows. For each condition, generate up to 100,000 rows satisfying the regression's data requirements. Fit every compared method on the same number of rows. Error is the squared coefficient error, including the intercept.
- **Mincer slices and diagnostics:** use the shared ACSLabour equality conditions, excluding those involving schooling (`SCHL`), age (`AGEP`) or `log_wage`. Average errors across seeds per slice, then report the mean and standard error across slices in each rarity group. An ideal reference can be sampled from real training rows using `method=ideal`. General-population unconditional generation and the `ENG=4` (no English) condition are reported separately and excluded from the rarity plot.
- **Earnings profiles:** median wages within age groups from 100,000 generated rows per education level for China-born female private-sector workers. Figure 6 has separate subplots for observed data, CFG and SMC, each showing high school, bachelor's and master's curves in five-year age groups from 20 to 50, with mean and standard error across training seeds.
