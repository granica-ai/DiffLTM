"""Analytic counterfactuals and simplex SC; no inverse of the folded target."""

from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import kstest

from ..data.synthetic import DEFAULT_CONFIG as DGP


def validate_generated(frame, period, dgp, n):
    """Check every decoded row; never truncate category codes or filter failures."""
    if len(frame) != n:
        raise ValueError(f"Generated {len(frame)} rows; expected {n}")
    for column, expected in [("S", dgp.treated_unit), ("T", period)]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, expected).all():
            raise ValueError(f"Generated rows violate {column}={expected}")
    for column in [c for c in ("U", "Y", "LOG_Y", "QUAD_Y") if c in frame]:
        if not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"Nonfinite generated {column}")


def q50(values):
    return float(np.quantile(values, 0.5, method="inverted_cdf"))


def level_values(frame):
    values = frame.Y.to_numpy(float) if "Y" in frame else np.exp(frame.LOG_Y.to_numpy(float))
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite level outcomes")
    return values


def score_cell(frame, period, dgp=None):
    dgp = DGP if dgp is None else dgp
    y = level_values(frame)
    mu = dgp.mu_s[dgp.treated_unit - 1] + dgp.mu_t[period]
    sigma = dgp.sigma_s[dgp.treated_unit - 1] + dgp.sigma_t[period]
    noise_cdf = "uniform" if dgp.noise == "uniform" else "norm"
    y_args = (
        (mu - np.sqrt(3) * sigma, 2 * np.sqrt(3) * sigma) if dgp.noise == "uniform" else (mu, sigma)
    )
    u_args = (-np.sqrt(3), 2 * np.sqrt(3)) if dgp.noise == "uniform" else (0.0, 1.0)
    err = q50(y) - mu
    result = dict(
        period=period,
        held=period in dgp.post_periods,
        n=len(y),
        median=q50(y),
        oracle=mu,
        median_err=err,
        abs_z=abs(err) / sigma,
        y_sd_ratio=float(np.std(y, ddof=1) / sigma),
        y_ks=float(kstest(y, noise_cdf, args=y_args).statistic),
    )
    if "U" in frame:
        u = frame.U.to_numpy(float)
        if not np.isfinite(u).all():
            raise ValueError("nonfinite U")
        result.update(
            u_sd=float(np.std(u, ddof=1)),
            u_median=q50(u),
            u_ks=float(kstest(u, noise_cdf, args=u_args).statistic),
            joint_dgp_rmse=float(np.sqrt(np.mean((y - mu - sigma * u) ** 2))),
        )
    return result


def fit_sc(frame, log=False, *, value_col=None, dgp=None):
    """Fit simplex weights using only treated pre-period and donor medians."""
    dgp = DGP if dgp is None else dgp
    if log and value_col is not None:
        raise ValueError("Choose either log conversion or a native value column")
    frame = frame.copy()
    for column, allowed in [("S", range(1, len(dgp.mu_s) + 1)), ("T", range(len(dgp.mu_t)))]:
        codes = pd.to_numeric(frame[column], errors="coerce")
        if not codes.isin(allowed).all():
            raise ValueError(f"Invalid SC {column} labels")
        frame[column] = codes.astype(int)
    # Remove treated post outcomes before any log or aggregation.
    frame = frame.loc[~((frame.S == dgp.treated_unit) & frame["T"].isin(dgp.post_periods))].copy()
    y = frame[value_col].to_numpy(float) if value_col is not None else level_values(frame)
    if log and (y <= 0).any():
        raise ValueError("SC log fit needs positive retained outcomes")
    if not np.isfinite(y).all():
        raise ValueError("SC needs finite retained outcomes")
    frame["value"] = np.log(y) if log else y
    medians = frame.groupby(["T", "S"]).value.agg(q50).unstack().sort_index()
    expected_states = set(range(1, len(dgp.mu_s) + 1))
    if set(medians.columns) != expected_states or set(medians.index) != set(range(len(dgp.mu_t))):
        raise ValueError("missing donor or treated-pre cells")
    donors = [s for s in medians.columns if s != dgp.treated_unit]
    pre = [t for t in range(len(dgp.mu_t)) if t not in dgp.post_periods]
    x = medians.loc[pre, donors].to_numpy()
    target = medians.loc[pre, dgp.treated_unit].to_numpy()
    donor_paths = medians.loc[range(len(dgp.mu_t)), donors].to_numpy()
    if not all(np.isfinite(v).all() for v in (x, target, donor_paths)):
        raise ValueError("missing donor or treated-pre cells")
    # An analytic gradient and common objective scale avoid SLSQP false
    # failures from finite-difference noise near the simplex boundary.
    scale = max(float(np.max(np.abs(target))), 1.0)
    x_fit, y_fit = x / scale, target / scale
    result = minimize(
        lambda w: np.mean((x_fit @ w - y_fit) ** 2),
        np.full(len(donors), 1 / len(donors)),
        method="SLSQP",
        jac=lambda w: 2 * x_fit.T @ (x_fit @ w - y_fit) / len(y_fit),
        bounds=[(0.0, 1.0)] * len(donors),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-7 or min(result.x) < -1e-7:
        raise RuntimeError(f"invalid SC optimization: {result.message}")
    path = donor_paths @ result.x
    return (np.exp(path) if log else path), dict(zip(map(str, donors), map(float, result.x)))


def summarize(rows):
    frame = pd.DataFrame(rows)
    return {
        label: {
            "median_rmse": float(np.sqrt(np.mean(group.median_err**2))),
            "mean_abs_z": float(group.abs_z.mean()),
        }
        for label, group in (("held", frame.loc[frame.held]), ("seen", frame.loc[~frame.held]))
    }


def transform(y, maximum):
    y = np.asarray(y, dtype=float)
    return y * (maximum - y)


def quadratic_cdf(z, low, high, maximum):
    """Exact pushforward CDF for Y uniform on [low, high]."""
    z = np.asarray(z, dtype=float)
    vertex = maximum / 2
    radius = np.sqrt(np.maximum(vertex**2 - z, 0))
    overlap = np.maximum(np.minimum(high, vertex + radius) - np.maximum(low, vertex - radius), 0)
    return np.where(z >= vertex**2, 1.0, np.clip(1 - overlap / (high - low), 0.0, 1.0))


def oracle(period, dgp, maximum):
    if dgp.noise != "uniform":
        raise ValueError("Quadratic oracle currently requires uniform noise")
    mu = dgp.mu_s[dgp.treated_unit - 1] + dgp.mu_t[period]
    sigma = dgp.sigma_s[dgp.treated_unit - 1] + dgp.sigma_t[period]
    low, high = mu - np.sqrt(3) * sigma, mu + np.sqrt(3) * sigma
    ends = transform([low, high], maximum)
    zmin = float(min(ends))
    zmax = maximum**2 / 4 if low <= maximum / 2 <= high else float(max(ends))
    median = brentq(lambda z: float(quadratic_cdf(z, low, high, maximum)) - 0.5, zmin, zmax)
    sd = np.sqrt((maximum - 2 * mu) ** 2 * sigma**2 + 0.8 * sigma**4)
    return dict(median=median, sd=float(sd), low=low, high=high, mu=mu, sigma=sigma)


def score_quadratic(frame, period, dgp, *, maximum):
    z = frame.QUAD_Y.to_numpy(float)
    if len(z) < 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite transformed outcomes")
    truth = oracle(period, dgp, maximum)
    median = q50(z)
    error = median - truth["median"]
    result = dict(
        period=period,
        held=period in dgp.post_periods,
        n=len(z),
        median=median,
        oracle=truth["median"],
        median_err=error,
        abs_z=abs(error) / truth["sd"],
        z_sd_ratio=float(z.std(ddof=1) / truth["sd"]),
        z_ks=float(
            kstest(
                z, partial(quadratic_cdf, low=truth["low"], high=truth["high"], maximum=maximum)
            ).statistic
        ),
        above_global_support_fraction=float(np.mean(z > maximum**2 / 4)),
        metric_units="Z",
    )
    cell_ends = transform([truth["low"], truth["high"]], maximum)
    cell_low = float(min(cell_ends))
    cell_high = (
        maximum**2 / 4 if truth["low"] <= maximum / 2 <= truth["high"] else float(max(cell_ends))
    )
    result["outside_cell_support_fraction"] = float(np.mean((z < cell_low) | (z > cell_high)))
    if "U" in frame:
        u = frame.U.to_numpy(float)
        if not np.isfinite(u).all():
            raise ValueError("Nonfinite U")
        result.update(
            u_sd=float(u.std(ddof=1)),
            u_median=q50(u),
            u_ks=float(kstest(u, "uniform", args=(-np.sqrt(3), 2 * np.sqrt(3))).statistic),
            joint_dgp_rmse=float(
                np.sqrt(np.mean((z - transform(truth["mu"] + truth["sigma"] * u, maximum)) ** 2))
            ),
        )
    return result
