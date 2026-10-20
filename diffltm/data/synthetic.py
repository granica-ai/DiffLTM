"""Pinned location-scale panel, independent of run and split seeds."""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DGPConfig:
    mu_s: tuple[float, ...]
    sigma_s: tuple[float, ...]
    mu_t: tuple[float, ...]
    sigma_t: tuple[float, ...]
    treated_unit: int
    post_periods: tuple[int, ...]
    treated_post_mu: tuple[float, ...]
    treated_post_sigma: tuple[float, ...]
    rows_per_cell: int = 1_200
    noise: str = "uniform"
    seed: int = 42


def validate(config: DGPConfig) -> None:
    mu_s = np.asarray(config.mu_s, dtype=float)
    sigma_s = np.asarray(config.sigma_s, dtype=float)
    mu_t = np.asarray(config.mu_t, dtype=float)
    sigma_t = np.asarray(config.sigma_t, dtype=float)
    post_mu = np.asarray(config.treated_post_mu, dtype=float)
    post_sigma = np.asarray(config.treated_post_sigma, dtype=float)

    if any(x.ndim != 1 for x in (mu_s, sigma_s, mu_t, sigma_t, post_mu, post_sigma)):
        raise ValueError("DGP vectors must be one-dimensional")
    if len(mu_s) < 2 or len(mu_t) < 2:
        raise ValueError("Need at least two units and periods")
    if (
        not config.post_periods
        or set(config.post_periods) != set(range(min(config.post_periods), len(mu_t)))
        or min(config.post_periods) < 1
    ):
        raise ValueError("post_periods must be a nonempty trailing block with a pre-period")
    if not isinstance(config.rows_per_cell, (int, np.integer)):
        raise ValueError("rows_per_cell must be an integer")
    if len(mu_s) != len(sigma_s) or len(mu_t) != len(sigma_t):
        raise ValueError("mu and sigma vectors must have matching lengths")
    if not all(np.isfinite(x).all() for x in (mu_s, sigma_s, mu_t, sigma_t, post_mu, post_sigma)):
        raise ValueError("mu and sigma vectors must be finite")
    if config.treated_unit not in range(1, len(mu_s) + 1):
        raise ValueError("treated_unit must be a one-based state ID")
    if len(set(config.post_periods)) != len(config.post_periods):
        raise ValueError("post_periods must not contain duplicates")
    if not set(config.post_periods) <= set(range(len(mu_t))):
        raise ValueError("post_periods must index the time vectors")
    if len(post_mu) != len(config.post_periods) or len(post_sigma) != len(post_mu):
        raise ValueError("post mu/sigma vectors must match post_periods")
    if config.rows_per_cell <= 0:
        raise ValueError("rows_per_cell must be positive")
    if config.noise not in {"uniform", "gaussian"}:
        raise ValueError("noise must be 'uniform' or 'gaussian'")

    cell_sigma = sigma_s[:, None] + sigma_t
    if (cell_sigma <= 0).any() or (post_sigma <= 0).any():
        raise ValueError("every cell scale must be positive")

    if config.noise == "uniform":
        cell_mu = mu_s[:, None] + mu_t
        if (cell_mu - np.sqrt(3.0) * cell_sigma <= 0).any():
            raise ValueError("uniform support includes nonpositive untreated Y")


def sample_u(
    rng: np.random.Generator,
    n: int,
    distribution: str,
) -> np.ndarray:
    if distribution == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    if distribution == "gaussian":
        return rng.normal(size=n)
    raise ValueError(f"unknown noise distribution: {distribution}")


def generate_panel(config: DGPConfig | None = None) -> pd.DataFrame:
    """Generate the complete panel, including held-out treated observations."""
    config = DEFAULT_CONFIG if config is None else config
    validate(config)
    rng = np.random.default_rng(config.seed)
    post_index = {period: i for i, period in enumerate(config.post_periods)}
    frames = []

    for state, (mu_s, sigma_s) in enumerate(
        zip(config.mu_s, config.sigma_s),
        start=1,
    ):
        for period, (mu_t, sigma_t) in enumerate(
            zip(config.mu_t, config.sigma_t),
        ):
            u = sample_u(rng, config.rows_per_cell, config.noise)
            y0 = mu_s + mu_t + (sigma_s + sigma_t) * u
            treated_post = state == config.treated_unit and period in post_index
            if treated_post:
                i = post_index[period]
                y = config.treated_post_mu[i] + config.treated_post_sigma[i] * u
            else:
                y = y0
            frames.append(
                pd.DataFrame(
                    {
                        "S": state,
                        "T": period,
                        "U": u,
                        "Y0": y0,
                        "Y": y,
                        "TREATED_POST": treated_post,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def load_dgp(path):
    """Load a supplied DGP, checking the derived counts in saved manifests."""
    payload = json.loads(path.read_text())
    names = {field.name for field in fields(DGPConfig)}
    derived = {"n_units", "n_periods", "n_rows_full", "n_rows_retained"}
    if set(payload) - names - derived:
        raise ValueError(f"Unknown DGP fields: {set(payload) - names - derived}")
    dgp = DGPConfig(**{key: value for key, value in payload.items() if key in names})
    validate(dgp)
    n_units, n_periods = len(dgp.mu_s), len(dgp.mu_t)
    expected = dict(
        n_units=n_units,
        n_periods=n_periods,
        n_rows_full=n_units * n_periods * dgp.rows_per_cell,
        n_rows_retained=(n_units * n_periods - len(dgp.post_periods)) * dgp.rows_per_cell,
    )
    for key, value in expected.items():
        if key in payload and payload[key] != value:
            raise ValueError(f"DGP {key}: supplied {payload[key]}, calculated {value}")
    return dgp


DEFAULT_CONFIG = load_dgp(Path(__file__).with_name("dgp.json"))
