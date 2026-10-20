"""ACS raw product cache and the three explicitly versioned release tables."""

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write, file_identity
from .core import Schema, make_table

STATES = "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY PR".split()
LABOUR_CAT = "ST PUMA SEX RAC1P HISP SCHL POBP CIT ENG OCCP INDP COW MAR HICOV WKW".split()
EXPECTED = {"acslabour": 6892317, "acspubcov": 5916565, "acsmobility": 2041325}


def task_frame(raw, name):
    if name == "acslabour":
        raw = raw.rename(columns={"STATE": "ST"})
        frame = raw.loc[
            (raw.AGEP >= 16) & raw.ESR.isin([1, 2]) & (raw.WAGP > 0) & (raw.WKHP > 0)
        ].copy()
        frame["log_wage"] = np.log(frame.WAGP.astype(float) * frame.ADJINC.astype(float) / 1e6)
        schema = Schema(
            ["AGEP", "WKHP", "log_wage"], LABOUR_CAT, "log_wage", "regression", ["AGEP", "WKHP"]
        )
    else:
        from folktables import ACSMobility, ACSPublicCoverage

        if name == "acspubcov":
            task, target, num, integer = ACSPublicCoverage, "PUBCOV", ["AGEP", "PINCP"], ["AGEP"]
            frame = raw.loc[(raw.AGEP < 65) & (raw.PINCP <= 30000)].copy()
        elif name == "acsmobility":
            task, target, num, integer = (
                ACSMobility,
                "MIG",
                ["AGEP", "WKHP", "JWMNP", "PINCP"],
                ["AGEP", "WKHP", "JWMNP"],
            )
            frame = raw.loc[(raw.AGEP > 18) & (raw.AGEP < 35) & raw.JWMNP.notna()].copy()
        else:
            raise KeyError(name)
        features = [c for c in task.features if not (name == "acsmobility" and c == "ESP")]
        frame[target] = (frame[target] == 1).astype(int)
        schema = Schema(
            num, [c for c in features if c not in num] + [target], target, "binclass", integer
        )
    if not np.isfinite(frame[schema.numerical].to_numpy(float)).all():
        raise ValueError("ACS release numerics must be complete after task eligibility")
    return frame[schema.columns], schema


# Person-file jurisdiction codes, independent of Folktables' annual-task registry.
LABOUR_FIPS = {
    state: int(code)
    for state, code in (
        item.split(":")
        for item in """
        AK:02 AL:01 AR:05 AZ:04 CA:06 CO:08 CT:09 DC:11 DE:10 FL:12 GA:13
        HI:15 IA:19 ID:16 IL:17 IN:18 KS:20 KY:21 LA:22 MA:25 MD:24 ME:23
        MI:26 MN:27 MO:29 MS:28 MT:30 NC:37 ND:38 NE:31 NH:33 NJ:34 NM:35
        NV:32 NY:36 OH:39 OK:40 OR:41 PA:42 PR:72 RI:44 SC:45 SD:46 TN:47
        TX:48 UT:49 VA:51 VT:50 WA:53 WI:55 WV:54 WY:56
        """.split()
    )
}
LABOUR_READ_COLS = ["SERIALNO", "SPORDER", "ADJINC", "AGEP", "ESR", "WKHP", "WAGP"] + LABOUR_CAT


def labour_archives(cache, raw_dir=None):
    """Locate all 52 existing ZIPs; never download or modify the source dump."""
    root = Path(
        raw_dir
        or os.environ.get("DIFFLTM_ACSLABOUR_RAW")
        or Path(cache) / "raw" / "acslabour" / "zips"
    ).expanduser()
    archives = {state: root / f"csv_p{state.lower()}.zip" for state in sorted(LABOUR_FIPS)}
    missing = [p.name for p in archives.values() if not p.is_file()]
    if missing:
        raise NotImplementedError(
            "Automatic ACSLabour acquisition is not implemented. "
            f"ACSLabour requires the existing 2018 five-year person ZIPs in {root}; "
            f"missing: {', '.join(missing)}. Set DIFFLTM_ACSLABOUR_RAW to the complete "
            "raw dump, or link it at <cache>/raw/acslabour/zips. No download was attempted."
        )
    return archives


def read_labour_archive(path, state, chunksize=400000):
    """Stream the original builder's columns and filters, without extracting CSVs."""
    frames, raw_rows = [], 0
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No person CSV in {path}")
        for member in members:
            with archive.open(member) as handle:
                for chunk in pd.read_csv(
                    handle,
                    usecols=LABOUR_READ_COLS,
                    dtype={"SERIALNO": str},
                    chunksize=chunksize,
                    low_memory=False,
                ):
                    if not chunk.ST.eq(LABOUR_FIPS[state]).all():
                        raise ValueError(f"Wrong or missing jurisdiction code in {path}:{member}")
                    # Source positions count excluded rows too, and span all CSV members.
                    chunk.index = [
                        f"5-Year:2018:{state}:{i}" for i in range(raw_rows, raw_rows + len(chunk))
                    ]
                    raw_rows += len(chunk)
                    frame, schema = task_frame(chunk, "acslabour")
                    frames.append(frame)
    if not frames:
        raise ValueError(f"Empty person CSV in {path}")
    frame = pd.concat(frames)
    return (
        frame,
        schema,
        dict(
            archive=Path(path).name,
            sha256=file_identity(path),
            members=members,
            raw_rows=raw_rows,
            eligible_rows=len(frame),
            state_fips=LABOUR_FIPS[state],
        ),
    )


def build_labour(cache):
    archives = labour_archives(cache)  # Fail before partial work if the dump is incomplete.
    frames, sources = [], {}
    for state, path in archives.items():
        frame, schema, sources[state] = read_labour_archive(path, state)
        frames.append(frame)
    frame = pd.concat(frames)
    if set(frame.ST.unique()) != set(LABOUR_FIPS.values()):
        raise ValueError("ACSLabour must retain all 52 jurisdictions, including DC and PR")
    if sources["DC"]["raw_rows"] != 32887 or sources["DC"]["eligible_rows"] != 17443:
        raise ValueError(
            "ACSLabour DC count differs from the audited 32,887 raw / 17,443 eligible rows"
        )
    if len(frame) != EXPECTED["acslabour"]:
        raise ValueError(f"ACSLabour expected {EXPECTED['acslabour']} rows, acquired {len(frame)}")
    return make_table(
        frame,
        schema,
        "acslabour",
        dict(
            reader="census-person-zip/local-v1",
            horizon="5-Year",
            years=[2018],
            states=list(archives),
            archives=sources,
        ),
    )


def build(name, cache):
    if name == "acslabour":
        return build_labour(cache)
    if name not in {"acspubcov", "acsmobility"}:
        raise KeyError(name)
    from folktables import ACSDataSource

    horizon = "1-Year"
    years = list(range(2014, 2019))
    states = sorted(STATES)
    frames = []
    raw_identity = {}
    for year in years:
        source = ACSDataSource(
            survey_year=str(year),
            horizon=horizon,
            survey="person",
            root_dir=str(Path(cache) / "acs_downloads"),
        )
        for state in states:
            path = Path(cache) / "raw" / "acs" / horizon / str(year) / f"{state}.parquet"
            if path.exists():
                raw = pd.read_parquet(path)
            else:
                raw = source.get_data(states=[state], download=True)
                atomic_write(path, lambda p: raw.to_parquet(p, index=False))
            raw_identity[f"{horizon}/{year}/{state}"] = file_identity(path)
            raw.index = [f"{horizon}:{year}:{state}:{i}" for i in range(len(raw))]
            frame, schema = task_frame(raw, name)
            frames.append(frame)
    frame = pd.concat(frames)
    if len(frame) != EXPECTED[name]:
        raise ValueError(
            f"{name}: expected audited {EXPECTED[name]} rows, acquired {len(frame)}; investigate source drift"
        )
    return make_table(
        frame,
        schema,
        name,
        dict(
            downloader="folktables==0.0.12",
            horizon=horizon,
            years=years,
            states=states,
            raw_identity=raw_identity,
        ),
    )
