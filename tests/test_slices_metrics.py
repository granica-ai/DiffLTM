import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diffltm.artifacts import (
    write_json,
)
from diffltm.data.core import Schema
from diffltm.data.preprocessing import Preprocessor, fit_numeric_fill, imputed_frame
from diffltm.evaluation import Evaluation
from diffltm.generation.conditioning import Condition
from diffltm.metrics import dcr, fidelity
from diffltm.plotting import aggregate_conditions, aggregate_runs, check_identity
from scripts import make_slices as slices


def test_bucket_boundaries():
    assert [slices.bucket(f) for f in [1, 0.03, 0.01, 0.003, 0.001, 0]] == [0, 1, 2, 3, 4, None]


def test_changed_arbitrary_predicate_does_not_replace_frozen_battery(tmp_path, monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace

    source = Path(__file__).parents[1] / "slices/acslabour/arbit_num"
    battery = tmp_path / "acslabour/arbit_num"
    shutil.copytree(source, battery)
    before = {p.name: p.read_bytes() for p in battery.iterdir() if p.is_file()}
    manifest = json.loads((battery / "manifest.json").read_text())
    table = SimpleNamespace(manifest=manifest["dataset"], frame=None)
    assert slices.frozen_battery(table, "arbit_num", tmp_path) == battery / "slices.csv"
    monkeypatch.setattr(
        slices, "arbitrary_conditions", lambda: [Condition(predicate="changed.py:fn:new-hash")]
    )
    with pytest.raises(ValueError, match="Frozen battery predicate mismatch"):
        slices.frozen_battery(table, "arbit_num", tmp_path)
    assert before == {p.name: p.read_bytes() for p in battery.iterdir() if p.is_file()}


def test_proposals_types_precision_and_anchor(small_table):
    train = small_table.frame
    schema = small_table.schema
    stats = (
        train[schema.numerical].mean(),
        train[schema.numerical].std(ddof=0),
        train[schema.numerical].min(),
        train[schema.numerical].max(),
    )
    rng = np.random.default_rng(19)
    for task in ["equality", "range", "linear"]:
        for _ in range(20):
            c = slices.propose(train, schema, task, rng, stats)
            restored = Condition.loads(c.dumps())
            assert np.array_equal(c.mask(train), restored.mask(train))
            assert not c.equality or set(c.equality) <= set(schema.categorical)
            if task == "linear":
                assert all(v == float(f"{v:.10g}") for v in c.weights.values())
                assert all(v / 1e-4 == pytest.approx(round(v / 1e-4)) for v in c.bounds)


def test_shortfall_is_explicit_and_test_gate_not_relaxed(tmp_path, small_table, monkeypatch):
    monkeypatch.setattr(slices, "PROPOSAL_BUDGET", 30)
    with pytest.raises(slices.IncompleteBatteryError):
        slices.make_battery(small_table, "equality", tmp_path / "slices", tmp_path)
    import json

    meta = json.loads((tmp_path / "slices/fixture/equality/manifest.json").read_text())
    assert meta["status"] == "incomplete" and meta["accepted"] == 0
    assert meta["constants"]["min_test"] == 100


def test_fidelity_sparse_and_constant_are_not_zero_filled(small_table):
    same = fidelity(small_table.frame, small_table.frame, small_table.schema, 42)
    assert same["shape"] == pytest.approx(1.0) and same["trend"] == pytest.approx(1.0)
    assert np.isnan(same["cond_shape_rare"])
    frame = small_table.frame.copy()
    frame["x"] = 1
    assert np.isfinite(fidelity(frame, frame, small_table.schema, 42)["shape"])


def test_dcr_identity_and_duplicate_training(small_table):
    train = small_table.frame.iloc[:70]
    hold = small_table.frame.iloc[70:]
    assert dcr(train, hold, hold, small_table.schema, 0) == pytest.approx(1.0)
    assert dcr(train, hold, train, small_table.schema, 0) < 1e-5


def test_run_aggregation_not_pseudoreplicated():
    frame = pd.DataFrame(
        {
            "method": ["a"] * 12,
            "bucket": ["b"] * 12,
            "training_seed": [500] * 10 + [501, 502],
            "value": [0.0] * 10 + [1.0, 2.0],
        }
    )
    result = aggregate_runs(frame, "value", ["method", "bucket"]).iloc[0]
    assert result["mean"] == 1 and result.n_runs == 3
    assert result.se == pytest.approx(1 / np.sqrt(3))
    with pytest.raises(ValueError):
        check_identity(pd.DataFrame({"dataset": ["acspubcov"]}), "acsmobility")


def test_publication_metric_reference(tmp_path):
    import json

    from diffltm.metrics import mle

    root = Path(__file__).parent / "fixtures"
    frame = pd.read_csv(root / "metric_reference.csv")
    expected = json.loads((root / "metric_reference.json").read_text())
    schema = Schema(["x", "y"], ["cat"], "y", "regression")
    train, test, syn = frame.iloc[:120], frame.iloc[120:180], frame.iloc[180:]
    value, rounds = mle(train, test, syn, schema, 33)
    np.testing.assert_allclose(rounds["trtr"], expected["trtr"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(rounds["tstr"], expected["tstr"], atol=1e-8, rtol=1e-8)
    assert value == pytest.approx(expected["ratio"], abs=1e-8)
    assert dcr(train, test, syn, schema, 33) == pytest.approx(expected["dcr"], abs=1e-6)


def test_membership_eligibility_and_shared_imputation():
    schema = Schema(["x"], ["cat"], "x", "regression")
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "cat": ["__NA__"] * 4})
    fill = fit_numeric_fill(train, schema)
    view = imputed_frame(train, schema, fill)
    assert fill[0] == pytest.approx(7 / 3)
    assert Condition(ranges={"x": (2, 3)}).mask(view).tolist() == [False, True, True, False]
    holdout = pd.DataFrame({"x": [100.0, np.nan], "cat": ["a", "__NA__"]})
    assert imputed_frame(holdout, schema, fill).x.iloc[1] == pytest.approx(7 / 3)
    assert view.cat.eq("__NA__").all()
    first, _ = slices.membership_key(Condition(ranges={"x": (1, 2)}), view)
    same, _ = slices.membership_key(Condition(ranges={"x": (0.9, 2.1)}), view)
    assert first == same and first is not None
    assert slices.membership_key(Condition(ranges={"x": (0, 10)}), view)[0] is None
    assert slices.membership_key(Condition(ranges={"x": (1.9, 2.1)}), view)[0] is None


def test_condition_aggregation_equal_weights_counts_and_undefined():
    records = []
    values = {"a": [0.0, 1.0, np.nan], "b": [10.0, np.nan, np.nan], "c": [np.nan] * 3}
    for name, scores in values.items():
        for seed, value in zip([500, 501, 502], scores):
            records.append(
                dict(
                    method="cfg",
                    bucket=">3%",
                    slice=name,
                    training_seed=seed,
                    value=value,
                    status="complete" if np.isfinite(value) else "undefined",
                    omission_reason=None if np.isfinite(value) else "not applicable",
                )
            )
    conditions, summary = aggregate_conditions(pd.DataFrame(records))
    assert conditions.valid_seeds.tolist() == [2, 1, 0]
    row = summary.iloc[0]
    assert row["mean"] == 5.25 and row.se == pytest.approx(4.75)
    assert row.valid_conditions == 2 and row.total_conditions == 3
    assert conditions.total_seeds.eq(3).all()
    for names, mean in [(["a"], 0.5), (["c"], np.nan)]:
        _, single = aggregate_conditions(pd.DataFrame(records).query("slice in @names"))
        assert np.isnan(single.se.iloc[0])
        assert (
            np.isnan(single["mean"].iloc[0]) if np.isnan(mean) else single["mean"].iloc[0] == mean
        )
    broken = pd.DataFrame(records)
    broken.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate_conditions(broken)


def test_shared_battery_and_evaluator_use_run_imputation(tmp_path, monkeypatch, model_config):
    from diffltm.artifacts import file_identity
    from diffltm.data.preprocessing import imputation_identity

    sizes = [50000, 2000, 500, 200, 80]
    categories = np.concatenate([np.repeat(str(i), n) for i, n in enumerate(sizes)])
    categories = np.concatenate([categories, np.repeat("rest", 100000 - len(categories))])
    values = np.random.default_rng(1).normal(size=len(categories))
    values[::7] = np.nan
    from diffltm.data.core import make_table

    table = make_table(
        pd.DataFrame({"x": values, "a": categories}),
        Schema(["x"], ["a"], "x", "regression"),
        "fixture",
        "fixture",
    )
    table.frame.to_parquet(tmp_path / "table.parquet")
    table.manifest["files"] = {"table.parquet": file_identity(tmp_path / "table.parquet")}
    candidates = iter([Condition(equality={"a": str(i)}) for i in range(5)])
    monkeypatch.setattr(slices, "QUOTA", 1)
    monkeypatch.setattr(slices, "MIN_TEST", 1)
    monkeypatch.setattr(slices, "propose", lambda *args: next(candidates))
    path = slices.make_battery(table, "equality", tmp_path / "slices", tmp_path / "data")
    assert path == tmp_path / "slices/fixture/equality/slices.csv"
    manifest = json.loads(path.with_name("manifest.json").read_text())
    battery = pd.read_csv(path)
    audit = pd.read_csv(path.with_name("run_counts.csv"))
    assert len(battery) == 5 and len(audit) == 15
    assert manifest["slice_seed"] == 0
    assert manifest["proposed_depths"] == {"1": 5}
    assert audit.imputation_id.nunique() == 3
    assert slices.frozen_battery(table, "equality", tmp_path / "slices") == path
    # Test the evaluator's production >=100 gate on one common slice.
    battery = battery.iloc[:1]
    battery.to_csv(path, index=False)
    audit = audit.loc[audit.slice_id.isin(battery.slice_id)]
    audit.to_csv(path.with_name("run_counts.csv"), index=False)
    manifest.update(
        csv_sha256=file_identity(path),
        run_counts_sha256=file_identity(path.with_name("run_counts.csv")),
    )
    write_json(path.with_name("manifest.json"), manifest)
    for seed in [500, 501]:
        frames, state = slices.membership_view(table, seed, tmp_path / "data")
        ev = object.__new__(Evaluation)
        ev.table, ev.frames, ev.seed = table, frames, seed
        fill = np.array(state["fill"])
        assert imputation_identity(table.schema, fill) == state["imputation_id"]
        ev.model = SimpleNamespace(preprocessor=SimpleNamespace(fill=fill))
        ev.checkpoint = {"split_id": state["split_id"]}
        ev.split_digest = state["split_digest"]
        restored, _ = ev.slices(path, "cfg")
        assert restored.iloc[0].bucket == battery.iloc[0].bucket
        assert (
            restored.iloc[0].train_count
            == audit.loc[audit.training_seed.eq(seed), "train_count"].iloc[0]
        )
    # Numerical membership uses each run's training fill, including on held-out rows.
    condition = Condition(ranges={"x": (-0.005, 0.005)})
    for seed in [500, 501]:
        frames, state = slices.membership_view(table, seed, tmp_path / "data")
        assert np.isfinite(frames["test"].x).all()
        p = Preprocessor(table.schema, "standardize", seed).fit(frames["train"])
        assert np.allclose(p.fill, state["fill"])
        # Reapplying the fitted imputation does not change membership.
        from diffltm.data.preprocessing import imputed_frame

        assert np.array_equal(
            condition.mask(frames["test"]),
            condition.mask(imputed_frame(frames["test"], table.schema, state["fill"])),
        )
