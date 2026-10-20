"""Publication CPS synthetic control in native log-wage space."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TARGET = "LOG_REAL_HOURWAGE"

YEARS = list(range(2010, 2020))

TREATMENT_YEAR = 2017

PRE_YEARS = [year for year in YEARS if year < TREATMENT_YEAR]

POST_YEARS = [year for year in YEARS if year >= TREATMENT_YEAR]

TREATED_STATES = ["AZ", "PA"]

CPI = {
    2010: 320.4,
    2011: 330.5,
    2012: 337.5,
    2013: 342.5,
    2014: 348.3,
    2015: 348.9,
    2016: 353.4,
    2017: 361.0,
    2018: 369.8,
    2019: 376.5,
}

FEDERAL_FLOOR_STATES = [
    "AL",
    "GA",
    "ID",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "MS",
    "NH",
    "NC",
    "ND",
    "OK",
    "PA",
    "SC",
    "TN",
    "TX",
    "UT",
    "VA",
    "WI",
    "WY",
]


def donors_for(treated: str) -> list[str]:
    """Return federal-floor donors, excluding the placebo from its own pool."""
    if treated not in TREATED_STATES:
        raise ValueError(f"treated state must be one of {TREATED_STATES}")
    return [state for state in FEDERAL_FLOOR_STATES if state != treated]


def nominal(log_real_wage: float, year: int) -> float:
    """Convert a constant-2019-dollar log wage to nominal dollars."""
    return float(np.exp(log_real_wage) * CPI[year] / CPI[2019])


EDUC_HS_MAX = 39  # grade92 <= 39: HS graduate/GED or less

EDUC_BA_MIN = 43  # grade92 >= 43: bachelor's or more

OCC_FOOD = [13]  # docc00 major groups

OCC_SALES = [16]

OCC_CLEAN_CARE = [14, 15]

MIN_DONOR_WEIGHT = 1e-4


def outcome_panel(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """Collapse person rows to one wage quantile per state-year.

    Rows are states and columns are the ten years 2010-2019. In ``fit_sc``,
    the resulting shape is 22 x 10 for Arizona (AZ plus 21 donors) and
    21 x 10 for Pennsylvania (PA plus 20 donors).
    """
    return df.groupby(["STATE", "YEAR"])[TARGET].quantile(q).unstack().sort_index()


def predictor_matrix(df: pd.DataFrame, y: pd.DataFrame, q: float) -> pd.DataFrame:
    """14 predictors x states: 7 pooled covariates and 7 lagged outcomes."""
    pre = df[df.YEAR < TREATMENT_YEAR]
    g = pre.groupby("STATE")
    label = f"p{int(round(q * 100))}"

    preds = pd.DataFrame(
        {
            "share_age_16_24": g["AGE"].apply(lambda s: s.between(16, 24).mean()),
            "share_educ_le_hs": g["EDUC"].apply(lambda s: (s <= EDUC_HS_MAX).mean()),
            "share_educ_ge_ba": g["EDUC"].apply(lambda s: (s >= EDUC_BA_MIN).mean()),
            "share_occ_food": g["OCC_MAJOR"].apply(lambda s: s.isin(OCC_FOOD).mean()),
            "share_occ_sales": g["OCC_MAJOR"].apply(lambda s: s.isin(OCC_SALES).mean()),
            "share_occ_clean_care": g["OCC_MAJOR"].apply(lambda s: s.isin(OCC_CLEAN_CARE).mean()),
            # One value per state: the mean log real wage across all 2010-2016 rows.
            "mean_log_real_wage": g[TARGET].mean(),
        }
    )
    for yr in PRE_YEARS:
        preds[f"{label}_{yr}"] = y[yr]
    return preds.T


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Scale predictors before fitting synthetic-control weights.

    Only the 14 predictors are standardized; the final outcome path remains in
    its original log-wage units.
    """
    scale = X.std(axis=1, ddof=0)
    if not np.isfinite(X.to_numpy()).all() or (scale <= 0).any():
        raise ValueError("Non-finite or constant synthetic-control predictors")
    return X.sub(X.mean(axis=1), axis=0).div(scale, axis=0)


def prune_weights(w: np.ndarray, min_weight: float = MIN_DONOR_WEIGHT) -> np.ndarray:
    """Zero weights <= ``min_weight`` and renormalize the rest proportionally."""
    kept = w > min_weight
    if not kept.any():
        raise ValueError(f"no donor weight exceeds {min_weight}")
    pruned = np.where(kept, w, 0.0)
    return pruned / pruned.sum()


def solve_weights(
    x_treated: np.ndarray, X_donors: np.ndarray, w0: np.ndarray
) -> tuple[np.ndarray, float]:
    """Minimize ||x - Xw||², then prune and renormalize negligible weights."""

    def loss(w: np.ndarray) -> float:
        r = x_treated - X_donors @ w
        return float(r @ r)

    def grad(w: np.ndarray) -> np.ndarray:
        return -2.0 * X_donors.T @ (x_treated - X_donors @ w)

    res = minimize(
        loss,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(w0),
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError(f"Synthetic-control optimizer failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    w = prune_weights(w / w.sum())
    return w, loss(w)


def multistart(
    x_treated: np.ndarray, X_donors: np.ndarray, n_starts: int, seed: int
) -> tuple[np.ndarray, float, list]:
    """Uniform start plus ``n_starts`` Dirichlet starts.

    The objective is convex but X'X has rank <= 14 against ~21 donors, so the
    argmin can be a face rather than a point. Spread across starts measures
    that degeneracy; it is not sampling uncertainty.
    """
    rng = np.random.default_rng(seed)
    n = X_donors.shape[1]
    starts = [np.full(n, 1.0 / n)]
    starts += [rng.dirichlet(np.ones(n)) for _ in range(n_starts)]
    solutions = [solve_weights(x_treated, X_donors, w0) for w0 in starts]
    losses = np.array([s[1] for s in solutions])
    best = int(np.argmin(losses))
    return solutions[best][0], float(losses[best]), solutions


def fit_sc(df: pd.DataFrame, treated: str, q: float, *, n_starts: int = 20, seed: int = 42) -> dict:
    """Fit one synthetic control and return every number the figures need."""
    donors = donors_for(treated)
    states = [treated] + donors
    sub = df[df.STATE.isin(states)]

    y = outcome_panel(sub, q)
    X = predictor_matrix(sub, y, q)
    Xz = standardize(X)

    x_t = Xz[treated].to_numpy(dtype=float)
    X_don = Xz[donors].to_numpy(dtype=float)
    w, loss, solutions = multistart(x_t, X_don, n_starts, seed)

    y_sc = {t: float(y.loc[donors, t].to_numpy() @ w) for t in YEARS}
    y_obs = {t: float(y.loc[treated, t]) for t in YEARS}
    gap = {t: y_obs[t] - y_sc[t] for t in YEARS}

    def rmse(ts: list[int]) -> float:
        return float(np.sqrt(np.mean([gap[t] ** 2 for t in ts])))

    x_sc = X_don @ w
    losses = np.array([s[1] for s in solutions])
    tied = np.array([s[0] for s in solutions])[losses <= losses.min() + 1e-8]
    sc_spread = {
        str(t): float(
            (y.loc[donors, t].to_numpy() @ tied.T).max()
            - (y.loc[donors, t].to_numpy() @ tied.T).min()
        )
        for t in POST_YEARS
    }

    return {
        "treated": treated,
        "quantile": q,
        "n_donors": len(donors),
        "weights": {s: float(wt) for s, wt in zip(donors, w)},
        "objective": loss,
        "balance": {
            name: {"treated": float(x_t[i]), "synthetic": float(x_sc[i])}
            for i, name in enumerate(X.index)
        },
        "y_treated_log": {str(t): y_obs[t] for t in YEARS},
        "y_synthetic_log": {str(t): y_sc[t] for t in YEARS},
        "y_treated_nominal": {str(t): float(nominal(y_obs[t], t)) for t in YEARS},
        "y_synthetic_nominal": {str(t): float(nominal(y_sc[t], t)) for t in YEARS},
        "gap_log": {str(t): gap[t] for t in YEARS},
        "rmse": {
            "pre": rmse(PRE_YEARS),
            "post": rmse(POST_YEARS),
        },
        "diagnostics": {
            "starts_at_optimum": int(len(tied)),
            "starts_total": int(len(solutions)),
            "objective_spread": float(losses.max() - losses.min()),
            "sc_path_spread_across_optima": sc_spread,
            "donor_hull": {
                str(t): {
                    "treated": y_obs[t],
                    "lo": float(y.loc[donors, t].min()),
                    "hi": float(y.loc[donors, t].max()),
                    "inside": bool(y.loc[donors, t].min() <= y_obs[t] <= y.loc[donors, t].max()),
                }
                for t in YEARS
            },
        },
    }
