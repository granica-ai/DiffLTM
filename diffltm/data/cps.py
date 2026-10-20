"""NBER MORG acquisition with the publication universe and fixed mappings."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from ..artifacts import atomic_write, file_identity
from .core import Schema, make_table

STATE_FIPS_TO_POSTAL = {}

ALL_STATE_FIPS: dict[int, str] = {
    1: "AL",
    2: "AK",
    4: "AZ",
    5: "AR",
    6: "CA",
    8: "CO",
    9: "CT",
    10: "DE",
    11: "DC",
    12: "FL",
    13: "GA",
    15: "HI",
    16: "ID",
    17: "IL",
    18: "IN",
    19: "IA",
    20: "KS",
    21: "KY",
    22: "LA",
    23: "ME",
    24: "MD",
    25: "MA",
    26: "MI",
    27: "MN",
    28: "MS",
    29: "MO",
    30: "MT",
    31: "NE",
    32: "NV",
    33: "NH",
    34: "NJ",
    35: "NM",
    36: "NY",
    37: "NC",
    38: "ND",
    39: "OH",
    40: "OK",
    41: "OR",
    42: "PA",
    44: "RI",
    45: "SC",
    46: "SD",
    47: "TN",
    48: "TX",
    49: "UT",
    50: "VT",
    51: "VA",
    53: "WA",
    54: "WV",
    55: "WI",
    56: "WY",
}

FEDERAL_FLOOR_FIPS: list[int] = [
    1,
    13,
    16,
    18,
    19,
    20,
    21,
    22,
    28,
    33,
    37,
    38,
    40,
    42,
    45,
    47,
    48,
    49,
    51,
    55,
    56,
]

CPI_U_RS: dict[int, float] = {
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

DOLLAR_YEAR = 2019

READ_COLS: list[str] = [
    "stfips",
    "year",
    "minsamp",
    "age",
    "sex",
    "race",
    "grade92",
    "class94",
    "lfsr94",
    "paidhre",
    "I25c",
    "earnhre",
    "earnwt",
    "docc00",
]

AGE_MIN, AGE_MAX = 16, 64

MISH_KEEP = {4, 8}  # outgoing rotation groups (already all of MORG)

LFSR_KEEP = {1, 2}  # 1=Employed-At Work, 2=Employed-Absent

CLASS_KEEP = {4, 5}  # 4=Private For Profit, 5=Private Nonprofit

PAIDHRE_YES = 1  # paid by the hour

ALLOC_NONE = 0  # I25c == 0 -> no earnings components allocated

EARNHRE_MIN_CENTS = 200

EARNHRE_TOPCODE_CENTS = 9999

TARGET_COL = "LOG_REAL_HOURWAGE"

RENAME_COLS = {
    "age": "AGE",
    "sex": "SEX",
    "race": "RACE",
    "grade92": "EDUC",
    "docc00": "OCC_MAJOR",
    "year": "YEAR",
    "earnwt": "EARNWT",
}


def _passes_filters(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["stfips"].isin(STATE_FIPS_TO_POSTAL)
        & df["age"].between(AGE_MIN, AGE_MAX)
        & df["minsamp"].isin(MISH_KEEP)
        & df["lfsr94"].isin(LFSR_KEEP)
        & df["class94"].isin(CLASS_KEEP)
        & (df["paidhre"] == PAIDHRE_YES)
        & (df["I25c"] == ALLOC_NONE)
        & df["earnhre"].notna()
        & df["earnhre"].between(EARNHRE_MIN_CENTS, EARNHRE_TOPCODE_CENTS - 1)
    )
    return df.loc[m]


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename MORG columns to the modelling schema and build the target.

    ``LOG_REAL_HOURWAGE`` is the natural log of the constant-2019-dollar hourly
    wage. ``earnhre`` is in cents, so the nominal dollar wage is ``earnhre/100``
    and the deflator is ``CPI[2019] / CPI[year]``. The transform is invertible:
    the nominal wage comes back as
    ``exp(LOG_REAL_HOURWAGE) * CPI[year] / CPI[2019]``, which is what the
    wage-heaping figures need.
    """
    df = df.copy()

    factor = df["year"].map(lambda y: CPI_U_RS[DOLLAR_YEAR] / CPI_U_RS[int(y)])
    nominal = df["earnhre"].astype(np.float64) / 100.0
    df[TARGET_COL] = np.log(nominal * factor.astype(np.float64))

    df["STATE"] = df["stfips"].astype(int).map(STATE_FIPS_TO_POSTAL)
    df = df.rename(columns=RENAME_COLS)
    df["YEAR"] = df["YEAR"].astype(int)
    df["AGE"] = df["AGE"].astype(np.float64)
    for col in ("SEX", "RACE", "EDUC", "OCC_MAJOR"):
        df[col] = df[col].astype(int)
    return df


STATE_FIPS_TO_POSTAL = ALL_STATE_FIPS


def build(name, cache):
    state = {"cps_wage_az": "AZ", "cps_wage_pa": "PA"}[name]
    frames = []
    raw_identity = {}
    for year in range(2010, 2020):
        raw_path = Path(cache) / "raw" / "cps" / f"morg{year % 100:02d}.dta"
        if not raw_path.exists():
            atomic_write(
                raw_path,
                lambda p: urlretrieve(
                    f"https://data.nber.org/morg/annual/morg{year % 100:02d}.dta", p
                ),
            )
        raw_identity[str(year)] = file_identity(raw_path)
        raw = pd.read_stata(raw_path, columns=READ_COLS, convert_categoricals=False)
        if not raw.year.eq(year).all():
            raise ValueError("CPS source year mismatch")
        raw.index = [f"{year}:{i}" for i in range(len(raw))]
        frames.append(derive_columns(_passes_filters(raw)))
    panel = pd.concat(frames)
    hold = panel.STATE.eq(state) & panel.YEAR.between(2017, 2019)
    schema = Schema(
        ["AGE", "LOG_REAL_HOURWAGE"],
        ["STATE", "YEAR", "SEX", "EDUC", "RACE", "OCC_MAJOR"],
        "LOG_REAL_HOURWAGE",
        "regression",
        ["AGE"],
    )
    return make_table(
        panel.loc[~hold],
        schema,
        name,
        dict(
            downloader="NBER MORG",
            years=list(range(2010, 2020)),
            cpi=CPI_U_RS,
            raw_identity=raw_identity,
        ),
        panel,
        exclusions=f"STATE={state},YEAR=2017..2019",
        held_state=state,
    )
