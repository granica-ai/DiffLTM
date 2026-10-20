"""Fixed economic requests, matched Mincer fits and native-space summaries."""

import numpy as np
import pandas as pd

from ..artifacts import identity, request_seed
from ..data.core import category_values
from ..generation.conditioning import Condition
from ..metrics import subset

CPS_SUBGROUPS = {
    "high_school": ("EDUC", "39"),
    "bachelors": ("EDUC", "43"),
    "healthcare": ("OCC_MAJOR", "10"),
    "cleaning": ("OCC_MAJOR", "14"),
    "sales": ("OCC_MAJOR", "16"),
}

MINCER_GENERAL = "general_population"
MINCER_ENG4 = "eng_no_english"
MINCER_DIAGNOSTICS = {MINCER_GENERAL, MINCER_ENG4}
MINCER_BLOCKED_COLUMNS = {"SCHL", "AGEP", "log_wage"}


def observed_cell(table, condition):
    """Select the source panel with the same categorical predicate as the model."""
    reference = table.reference
    columns = list(condition.equality)
    view = reference[columns].copy()
    for col in columns:
        view[col] = category_values(view[col])
    return reference.loc[condition.mask(view)]


SCHOOL_YEARS = {
    1: 0,
    2: 0,
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 12,
    17: 12,
    18: 13,
    19: 14,
    20: 14,
    21: 16,
    22: 18,
    23: 19,
    24: 20,
}


def eligible_mincer(frame):
    schooling = pd.to_numeric(frame.SCHL, errors="coerce").map(SCHOOL_YEARS)
    experience = pd.to_numeric(frame.AGEP, errors="coerce") - schooling - 6
    valid = (
        np.isfinite(pd.to_numeric(frame.log_wage, errors="coerce"))
        & np.isfinite(schooling)
        & np.isfinite(experience)
        & (experience >= 0)
    )
    result = frame.loc[valid].copy()
    result["schooling"] = schooling[valid]
    result["experience"] = experience[valid]
    return result


def ols(frame):
    frame = eligible_mincer(frame)
    x = np.column_stack(
        [np.ones(len(frame)), frame.schooling, frame.experience, frame.experience**2]
    )
    y = frame.log_wage.to_numpy(float)
    if len(y) < 4:
        raise ValueError("Insufficient Mincer rows")
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 4:
        raise ValueError("Rank-deficient Mincer fit")
    return beta


def matched_mincer(real_test, generated, seed):
    real = eligible_mincer(real_test)
    real = subset(real, min(100000, len(real)), seed)
    usable = {name: eligible_mincer(frame) for name, frame in generated.items()}
    n = min([len(real)] + [len(f) for f in usable.values()])
    chosen = subset(real, n, seed)
    metadata = dict(
        real_eligible=len(eligible_mincer(real_test)),
        real_capped=len(real),
        fitted=n,
        real_rows_identity=identity(chosen.index.tolist()),
        real_row_ids=chosen.index.tolist(),
        scoring_seed=seed,
        generated_eligible={k: len(v) for k, v in usable.items()},
    )
    if n < 4:
        return [], dict(status="unsupported", **metadata)
    try:
        truth = ols(chosen)
        rows = []
        for method, frame in usable.items():
            beta = ols(subset(frame, n, request_seed(seed, method, "mincer-fit")))
            rows.append(
                dict(
                    method=method,
                    coefficients=beta.tolist(),
                    real_coefficients=truth.tolist(),
                    coefficient_error=float(np.square(beta - truth).sum()),
                    fitted=n,
                    status="complete",
                    omission_reason=None,
                )
            )
        return rows, dict(status="complete", **metadata)
    except ValueError as exc:
        return [], dict(status="unsupported", reason=str(exc), **metadata)


def requests(table, family, test, slices=None, method=None):
    name = table.manifest["name"]
    if family == "cps":
        if name not in {"cps_wage_az", "cps_wage_pa"}:
            raise ValueError("CPS dataset required")
        state = table.manifest["held_state"]
        result = [
            (str(year), Condition(equality={"STATE": state, "YEAR": str(year)}), 100000)
            for year in range(2010, 2020)
        ]
        if state == "AZ":
            for label, (column, code) in CPS_SUBGROUPS.items():
                for year in [2016, 2019]:
                    condition = Condition(
                        equality={"STATE": state, "YEAR": str(year), column: code}
                    )
                    result.append(
                        (
                            f"{label}_{year}",
                            condition,
                            100000,
                        )
                    )
        return result
    if family == "reparam":
        if not name.startswith("sc_reparam_"):
            raise ValueError("Synthetic panel required")
        return [(str(t), Condition(equality={"S": "6", "T": str(t)}), 100000) for t in range(11)]
    if family == "profiles":
        if name != "acslabour":
            raise ValueError("ACSLabour required")
        return [
            (
                label,
                Condition(equality={"POBP": "207", "COW": "1", "SEX": "2", "SCHL": code}),
                100000,
            )
            for label, code in [("High school", "16"), ("Bachelor's", "21"), ("Master's", "22")]
        ]
    if family == "mincer":
        if name != "acslabour":
            raise ValueError("ACSLabour required")
        if method == "unconditional":
            return [
                (
                    MINCER_GENERAL,
                    Condition(),
                    min(100000, len(eligible_mincer(test))),
                )
            ]
        if slices is None:
            raise ValueError("ACSLabour Mincer battery required")
        result = []
        for row in slices.itertuples():
            c = Condition.loads(row.predicate)
            columns = set(c.equality) | set(c.ranges) | set(c.weights)
            if columns & MINCER_BLOCKED_COLUMNS:
                continue
            if (
                c.ranges
                or c.weights
                or c.predicate
                or not set(c.equality) <= set(table.schema.categorical)
            ):
                raise ValueError("Invalid Mincer predicate")
            n = min(100000, len(eligible_mincer(test.loc[c.mask(test)])))
            result.append((row.slice_id, c, n))
        condition = Condition(equality={"ENG": "4"})
        result.append(
            (
                MINCER_ENG4,
                condition,
                min(100000, len(eligible_mincer(test.loc[condition.mask(test)]))),
            )
        )
        return result
    raise ValueError(family)


def profiles(frame):
    rows = []
    for lo in range(20, 60 + 1, 5):
        cell = frame.loc[(frame.AGEP >= lo) & (frame.AGEP < lo + 5), "log_wage"]
        rows.append(
            dict(
                age_mid=lo + 2.5,
                median=float(cell.median()) if len(cell) else float("nan"),
                count=len(cell),
                units="log_2018_annual_wage",
            )
        )
    return rows


def synthetic_summary(frame, period, table):
    from ..data.synthetic import DEFAULT_CONFIG
    from .synthetic_metrics import score_cell, score_quadratic

    if table.manifest["name"].endswith("quadratic"):
        return score_quadratic(frame, period, DEFAULT_CONFIG, maximum=table.manifest["M"])
    if table.manifest["name"].endswith("log"):
        values = frame.LOG_Y.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite native log outcome")
        median = float(np.quantile(values, 0.5, method="inverted_cdf"))
        truth = float(np.log(DEFAULT_CONFIG.mu_s[5] + DEFAULT_CONFIG.mu_t[period]))
        return dict(
            period=period,
            held=period in DEFAULT_CONFIG.post_periods,
            n=len(frame),
            median=median,
            oracle=truth,
            median_err=median - truth,
            metric_units="LOG_Y",
        )
    return score_cell(frame, period, DEFAULT_CONFIG)


def empirical_cdfs(real, synthetic):
    records = []
    for label, frame in [("real", real), ("generated", synthetic)]:
        values = np.sort(frame.to_numpy(float).reshape(-1))
        n = len(values)
        records.append(
            pd.DataFrame(
                dict(
                    value=values,
                    cdf=np.searchsorted(values, values, side="right") / n,
                    method=label,
                    n=n,
                )
            )
        )
    return pd.concat(records, ignore_index=True)
