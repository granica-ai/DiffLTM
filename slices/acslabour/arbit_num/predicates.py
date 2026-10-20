"""Numerical predicates used by the ACSLabour arbitrary-condition battery."""

import numpy as np


def ellipse(row):
    value = ((row["AGEP"] - 43) / 10) ** 2 + ((row["log_wage"] - 10.6) / 0.5) ** 2
    return value <= 0.64


def annulus(row):
    value = ((row["AGEP"] - 43) / 10) ** 2 + ((row["log_wage"] - 10.6) / 0.5) ** 2
    return (value >= 0.25) & (value <= 0.64)


def opposite_hours(row):
    return (row["AGEP"] - 43) * (row["WKHP"] - 40) < -200


def hourly_proxy(row):
    log_hours = row["WKHP"].log() if hasattr(row["WKHP"], "log") else np.log(row["WKHP"])
    value = row["log_wage"] - log_hours - np.log(52)
    return (value >= np.log(60)) & (value <= np.log(90))


def age_hours_linear(row):
    return row["AGEP"] + row["WKHP"] < 50


def wage_age_linear(row):
    return row["log_wage"] - 0.03 * row["AGEP"] < 7.25


PREDICATES = [
    ellipse,
    annulus,
    opposite_hours,
    hourly_proxy,
    age_hours_linear,
    wage_age_linear,
]
