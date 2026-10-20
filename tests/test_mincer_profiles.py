"""Mincer requests, saved-only fitting and manuscript Figures 5–6 aggregation."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from diffltm.artifacts import file_identity, identity, write_json
from diffltm.data.core import Schema
from diffltm.economics.experiments import (
    MINCER_ENG4,
    MINCER_GENERAL,
    eligible_mincer,
    matched_mincer,
    ols,
    requests,
)
from diffltm.evaluation import Evaluation
from diffltm.generation.conditioning import (
    Condition,
)
from diffltm.plotting import plot_coefficient_error, plot_profiles
from scripts import analyze_econ


@pytest.fixture
def mincer_case(tmp_path, monkeypatch):
    rng = np.random.default_rng(81)
    frame = pd.DataFrame(
        dict(
            SCHL=rng.choice(["16", "21", "22"], 360),
            AGEP=rng.integers(30, 65, 360),
            log_wage=rng.normal(10, 1, 360),
            SEX=np.tile(["1", "2"], 180),
            ENG=np.tile(["1", "1", "4", "4"], 90),
        )
    )
    table = SimpleNamespace(
        frame=frame,
        schema=Schema(["AGEP", "log_wage"], ["SCHL", "SEX", "ENG"], "log_wage", "regression"),
        manifest=dict(name="acslabour", identity="fixture-data", version="v1"),
    )
    splits = dict(
        train=np.arange(240), validation=np.array([], dtype=int), test=np.arange(240, 360)
    )
    path = tmp_path / "slices/slices.csv"
    path.parent.mkdir()
    pd.DataFrame(
        [
            dict(slice_id="sex", predicate=Condition(equality={"SEX": "2"}).dumps()),
            dict(slice_id="school", predicate=Condition(equality={"SCHL": "16"}).dumps()),
        ]
    ).assign(bucket=">3%", reference_train_fraction=0.5).to_csv(path, index=False)
    slice_manifest = dict(csv_sha256=file_identity(path))
    write_json(path.with_name("manifest.json"), slice_manifest)
    ev = object.__new__(Evaluation)
    ev.table, ev.seed = table, 500
    ev.frames = {k: frame.iloc[v] for k, v in splits.items()}
    ev.checkpoint = dict(split_id="fixture-split")
    ev.run_dir = tmp_path / "run"
    ev.run_dir.mkdir()
    ev.identity = dict(
        dataset="acslabour",
        dataset_identity="fixture-data",
        training_seed=500,
        split_id="fixture-split",
        training_runs=[dict(run_dir=str(ev.run_dir))],
    )
    ev.slices = lambda *args: (pd.read_csv(path), slice_manifest)
    monkeypatch.setattr(analyze_econ, "load_table", lambda *args: table)
    monkeypatch.setattr(analyze_econ, "load_run_splits", lambda *args: splits)
    monkeypatch.setattr(analyze_econ, "verify_result_training", lambda *args: None)
    return SimpleNamespace(ev=ev, table=table, path=path, root=tmp_path)


def test_mincer_shared_battery_and_diagnostics(mincer_case):
    case = mincer_case
    slices = pd.read_csv(case.path)
    cells = requests(case.table, "mincer", case.ev.frames["test"], slices, "cfg")
    assert [label for label, _, _ in cells] == ["sex", MINCER_ENG4]
    assert cells[-1][1].equality == {"ENG": "4"}
    assert cells[-1][2] == 60
    # The unconditional diagnostic is independent of any slice file.
    case.ev.slices = lambda *args: pytest.fail("Unconditional must not read a battery")
    calls = []

    def generate(method, condition, n):
        calls.append((method, condition, n))
        return case.ev.frames["test"], dict(sample_key="general", generation_seed=1)

    case.ev.generate = generate
    result = case.ev.economic("mincer", "unconditional")
    assert calls == [("unconditional", None, 120)]
    assert result[0]["cell"] == MINCER_GENERAL and result[0]["cell_type"] == "diagnostic"


@pytest.mark.parametrize("column", ["SCHL", "AGEP", "log_wage"])
def test_mincer_excludes_regression_columns(mincer_case, column):
    slices = pd.DataFrame(
        [dict(slice_id="blocked", predicate=Condition(equality={column: "16"}).dumps())]
    )
    cells = requests(mincer_case.table, "mincer", mincer_case.table.frame, slices, "cfg")
    assert [label for label, _, _ in cells] == [MINCER_ENG4]


@pytest.mark.parametrize("legacy", [False, True])
def test_mincer_saved_analysis_and_ideal_reference(mincer_case, legacy):
    case, ev = mincer_case, mincer_case.ev
    paths = []
    calls = []

    def generate(method, condition, n):
        calls.append(method)
        train = ev.frames["train"]
        if condition is not None:
            train = train.loc[condition.mask(train)]
        frame = train.sample(n=n, random_state=19).copy()
        frame.log_wage += {"cfg": 0.2, "smc": 0.5, "unconditional": 0.1}[method]
        key = identity([method, condition.dumps() if condition else None, n])
        root = (ev.run_dir if legacy else ev.output) / "samples" / key
        root.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / "samples.parquet", index=False)
        meta = dict(
            status="complete",
            sample_key=key,
            generation_seed=19,
            sha256=file_identity(root / "samples.parquet"),
        )
        write_json(root / "manifest.json", meta)
        return frame, meta

    ev.generate = generate
    for method in ["cfg", "smc", "ideal", "unconditional"]:
        ev.output = case.root / method
        cells = ev.economic("mincer", method, case.path)
        if method == "ideal":
            assert all(c["sample_key"] is None for c in cells)
        if legacy:
            for cell in cells:
                cell.pop("cell_type")
                cell.pop("slice_csv_sha256", None)
        write_json(
            ev.output / "manifest.json",
            dict(
                **ev.identity,
                status="complete",
                cells=cells,
                sample_layout="run" if legacy else "evaluation",
                config=dict(
                    task="economics",
                    family="mincer",
                    run_dir=str(ev.run_dir),
                    slices=str(case.path),
                ),
            ),
        )
        paths.append(ev.output)
    assert "ideal" not in calls
    result = analyze_econ.analyze("mincer", paths, case.root / "summary")
    assert result.status.eq("complete").all() and result.coefficient_error.notna().all()
    assert set(result.loc[result.cell_type.eq("rarity"), "cell"]) == {"sex"}
    assert set(result.loc[result.cell_type.eq("diagnostic"), "cell"]) == {
        MINCER_GENERAL,
        MINCER_ENG4,
    }
    assert result.loc[result.cell_type.eq("diagnostic"), "bucket"].isna().all()
    for _, group in result.groupby("cell"):
        assert group.fitted.nunique() == 1
        for real in group.real_coefficients:
            np.testing.assert_array_equal(real, group.real_coefficients.iloc[0])
    # Reconstruct the ideal draw from the training split, never from test data.
    ideal_meta = json.loads((case.root / "ideal/manifest.json").read_text())["cells"][0]
    ideal_train = eligible_mincer(ev.frames["train"].loc[ev.frames["train"].SEX.eq("2")])
    expected = ideal_train.sample(
        n=ideal_meta["returned"], random_state=ideal_meta["generation_seed"]
    )
    assert set(expected.index).isdisjoint(ev.frames["test"].index)
    actual = result.loc[result.method.eq("ideal") & result.cell.eq("sex"), "coefficients"].iloc[0]
    np.testing.assert_allclose(actual, ols(expected))
    repeated = analyze_econ.analyze("mincer", paths, case.root / "repeat")
    pd.testing.assert_frame_equal(result, repeated)
    if not legacy:
        battery = pd.read_csv(case.path).assign(bucket="(1%,3%]")
        battery.to_csv(case.path, index=False)
        write_json(case.path.with_name("manifest.json"), dict(csv_sha256=file_identity(case.path)))
        with pytest.raises(ValueError, match="slice checksum"):
            analyze_econ.analyze("mincer", paths, case.root / "changed")


def test_mincer_plot_averages_seeds_then_slices_and_excludes_diagnostics(tmp_path):
    rows = []
    for seed, values in [(500, [0.0, 10.0]), (501, [2.0, np.nan])]:
        for cell, value in zip(["a", "b"], values):
            rows.append(dict(seed=seed, cell=cell, coefficient_error=value, cell_type="rarity"))
        rows.append(
            dict(seed=seed, cell=MINCER_ENG4, coefficient_error=999, cell_type="diagnostic")
        )
    frame = pd.DataFrame(rows).assign(
        method="cfg",
        dataset="acslabour",
        dataset_identity="data",
        units="coefficient_squared_error",
        bucket=">3%",
        slice_csv_sha256="shared",
        omission_reason=None,
    )
    frame["status"] = np.where(frame.coefficient_error.isna(), "undefined", "complete")
    output = tmp_path / "mincer"
    fig, _ = plot_coefficient_error(frame, dataset="acslabour", output=output)
    values = pd.read_csv(output.with_suffix(".csv")).iloc[0]
    assert values["mean"] == pytest.approx(5.5)  # mean of slice means 1 and 10
    assert values["se"] == pytest.approx(4.5)
    assert values.valid_conditions == 2
    plt.close(fig)


def test_profiles_keep_separate_education_curves_and_seed_uncertainty(tmp_path):
    rows = [
        dict(
            dataset="acslabour",
            dataset_identity="data",
            units="log_2018_annual_wage",
            method=method,
            cell=cell,
            seed=seed,
            age_mid=age,
            median=10 + level + (seed - 500) + age / 100,
        )
        for method in ["real", "cfg", "smc", "rejection"]
        for level, cell in enumerate(["High school", "Bachelor's", "Master's"])
        for seed in [500, 501, 502]
        for age in [22.5, 47.5, 52.5]
    ]
    frame = pd.DataFrame(rows)
    output = tmp_path / "profiles"
    fig, axes = plot_profiles(frame, dataset="acslabour", output=output)
    values = pd.read_csv(output.with_suffix(".csv"))
    assert [ax.get_title() for ax in axes] == ["Real", "DiffLTM-CFG", "DiffLTM-SMC"]
    assert all(len(ax.containers) == 3 and ax.get_xlim() == (20, 50) for ax in axes)
    assert set(values.method) == {"real", "cfg", "smc"} and values.age_mid.max() == 47.5
    np.testing.assert_allclose(values.se, 1 / np.sqrt(3))
    assert values.n_runs.eq(3).all()
    plt.close(fig)
    with pytest.raises(ValueError, match="Real, CFG and SMC"):
        plot_profiles(frame.loc[frame.method.ne("smc")], dataset="acslabour")


def test_mincer_all_coefficients_and_equal_fit_sizes():
    rng = np.random.default_rng(4)
    real = pd.DataFrame(
        {
            "SCHL": rng.choice([16, 21, 22], 300),
            "AGEP": rng.integers(28, 65, 300),
            "log_wage": rng.normal(size=300),
        }
    )
    real.index = real.index.astype(str)
    generated = {"a": real.iloc[:180].copy(), "b": real.iloc[:200].copy()}
    generated["a"]["log_wage"] += 2
    rows, meta = matched_mincer(real, generated, 99)
    assert {r["fitted"] for r in rows} == {180}
    for row in rows:
        assert row["coefficient_error"] == pytest.approx(
            np.square(np.array(row["coefficients"]) - row["real_coefficients"]).sum()
        )
    assert meta["real_rows_identity"]
