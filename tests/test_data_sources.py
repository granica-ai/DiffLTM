from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diffltm.data.acs import task_frame
from diffltm.data.core import make_table
from diffltm.data.preprocessing import Preprocessor


def test_acs_filters_preserve_zero_missing_and_schema():
    from folktables import ACSMobility, ACSPublicCoverage

    cols = (
        set(ACSMobility.features)
        | set(ACSPublicCoverage.features)
        | {"MIG", "PUBCOV", "ST", "PUMA"}
    )
    raw = pd.DataFrame({c: np.ones(4) for c in cols})
    raw["AGEP"] = [20, 20, 20, 65]
    raw["JWMNP"] = [10, np.nan, 0, 15]
    raw["PINCP"] = [100, 100, 30000, 100]
    raw["GCL"] = [np.nan, 0, 0, 0]
    mobility, schema = task_frame(raw, "acsmobility")
    assert mobility.index.tolist() == [0, 2]
    assert len(schema.columns) == 21 and "ESP" not in schema.columns and "ST" not in schema.columns
    table = make_table(mobility, schema, "acsmobility", "fixture")
    assert table.frame.GCL.tolist() == ["__NA__", "0"]
    pubcov, schema = task_frame(raw, "acspubcov")
    assert (
        len(pubcov) == 3
        and len(schema.columns) == 20
        and "ESP" in schema.columns
        and "MIL" in schema.columns
    )
    assert "ACS_YEAR" not in schema.columns and schema.target == "PUBCOV"


@pytest.mark.parametrize(
    "task,target", ["regression:amount".split(":"), "multiclass:label".split(":")]
)
def test_openml_target_and_roles(monkeypatch, task, target):
    from diffltm.data.dataset_spec import OpenMLDatasetSpec
    from diffltm.data.openml import OPENML_REGISTRY, build

    data = pd.DataFrame({"x": [1.0, np.nan, 3.0], "kind": ["a", None, "b"], "id": [1, 2, 3]})
    y = pd.Series([1.0, 2.0, 3.0]) if task == "regression" else pd.Series(["a", "b", "c"])
    monkeypatch.setitem(
        OPENML_REGISTRY,
        "fixture",
        OpenMLDatasetSpec(
            name="fixture", openml_id=123, target=target, task=task, n_classes=0, drop_cols=["id"]
        ),
    )
    monkeypatch.setattr(
        "diffltm.data.openml.fetch_openml", lambda **kwargs: SimpleNamespace(data=data, target=y)
    )
    table = build("fixture", "unused")
    assert set(table.frame) == {"x", "kind", target}
    assert table.schema.target == target
    assert table.frame.kind.iloc[1] == "__NA__"
    p = Preprocessor(table.schema, "standardize").fit(table.frame)
    assert np.isfinite(p.transform(table.frame)).all()


def test_labour_zip_reader_is_offline_and_preserves_filters_and_ids(tmp_path, monkeypatch):
    import builtins
    import zipfile

    from diffltm.data.acs import LABOUR_READ_COLS, read_labour_archive

    raw = pd.DataFrame({c: np.ones(8) for c in LABOUR_READ_COLS})
    raw["ST"] = 11
    raw["AGEP"] = [30, 15, 30, 30, 30, 30, 30, 30]
    raw["ESR"] = [1, 1, 3, 2, 1, 1, 2, 1]
    raw["WAGP"] = [100, 100, 100, 0, 100, 100, 200, 100]
    raw["WKHP"] = [40, 40, 40, 40, 0, np.nan, 20, 0]
    raw["ADJINC"] = 1200000
    raw["ENG"] = [np.nan, 0, 0, 0, 0, 0, 0, 0]
    path = tmp_path / "csv_pdc.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("part1.csv", raw.iloc[:4].to_csv(index=False))
        z.writestr("part2.csv", raw.iloc[4:].to_csv(index=False))
        z.writestr("README.txt", "not a person table")
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name.startswith("folktables"):
            raise AssertionError("ACSLabour must not use Folktables")
        return original_import(name, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("ACSLabour must not download")

    monkeypatch.setattr(builtins, "__import__", import_guard)
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    frame, schema, meta = read_labour_archive(path, "DC", chunksize=1)
    other, _, other_meta = read_labour_archive(path, "DC", chunksize=100)
    pd.testing.assert_frame_equal(frame, other)
    assert meta == other_meta and meta["raw_rows"] == 8 and meta["eligible_rows"] == 2
    assert frame.index.tolist() == ["5-Year:2018:DC:0", "5-Year:2018:DC:6"]
    np.testing.assert_allclose(frame.log_wage, np.log([120, 240]))
    table = make_table(frame, schema, "acslabour", meta)
    assert table.frame.ENG.tolist() == ["__NA__", "0"]
    assert len(schema.columns) == 18 and schema.target == "log_wage"
    assert "SERIALNO" not in schema.columns and "SPORDER" not in schema.columns
    with pytest.raises(ValueError, match="jurisdiction"):
        read_labour_archive(path, "CA")


def test_labour_requires_complete_existing_dump(tmp_path, monkeypatch):
    from diffltm.data.acs import LABOUR_FIPS, STATES, labour_archives

    assert set(LABOUR_FIPS) == set(STATES) | {"DC"}
    assert len(LABOUR_FIPS) == 52 and LABOUR_FIPS["DC"] == 11 and LABOUR_FIPS["PR"] == 72
    assert "DC" not in STATES and len(STATES) == 51
    for state in LABOUR_FIPS:
        if state != "DC":
            (tmp_path / f"csv_p{state.lower()}.zip").touch()
    monkeypatch.setenv("DIFFLTM_ACSLABOUR_RAW", str(tmp_path))
    with pytest.raises(NotImplementedError, match="csv_pdc.zip"):
        labour_archives(tmp_path / "cache")
    (tmp_path / "csv_pdc.zip").touch()
    paths = labour_archives(tmp_path / "cache")
    assert len(paths) == 52 and paths["DC"] == tmp_path / "csv_pdc.zip"
    assert not (tmp_path / "cache").exists()
