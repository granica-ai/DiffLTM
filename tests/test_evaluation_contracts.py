from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from diffltm.data.core import Schema
from diffltm.evaluation import Evaluation
from diffltm.generation.conditioning import Condition


def session(tmp_path, train, test):
    ev = object.__new__(Evaluation)
    ev.frames = {"train": train, "test": test}
    ev.seed = 500
    ev.table = SimpleNamespace(
        schema=Schema(["x"], ["a"], "x", "regression"), manifest={"identity": "fixture"}
    )
    ev.checkpoint = {"split_id": "fixture-split"}
    ev.identity = {
        "dataset": "fixture",
        "dataset_identity": "fixture",
        "split_id": "fixture-split",
        "training_seed": 500,
    }
    ev.control_root = tmp_path
    return ev


def test_final_unconditional_has_no_100k_cap(tmp_path, monkeypatch):
    train = pd.DataFrame({"x": np.arange(150001, dtype=float), "a": "x"})
    test = train.iloc[:100001].copy()
    ev = session(tmp_path, train, test)
    calls = {}

    def generate(method, condition, n):
        calls["generation"] = n
        return train.copy(), dict(generation_seed=12, sample_key="one-pool")

    def fidelity(real, generated, schema, seed):
        calls["fidelity"] = (len(real), len(generated))
        return {"shape": 1.0}

    def mle(real, holdout, generated, *args, **kwargs):
        calls["mle"] = (len(real), len(holdout), len(generated))
        return 1.0, {}

    monkeypatch.setattr(ev, "generate", generate)
    monkeypatch.setattr("diffltm.scoring.fidelity", fidelity)
    monkeypatch.setattr("diffltm.scoring.mle", mle)
    monkeypatch.setattr("diffltm.scoring.dcr", lambda *args: 1.0)
    rows, _ = ev.unconditional()
    assert calls == {
        "generation": 150001,
        "fidelity": (100001, 100001),
        "mle": (150001, 100001, 150001),
    }
    assert all(row["n_scored"] == (10000 if row["metric"] == "dcr" else 100001) for row in rows)


@pytest.mark.parametrize(
    "method,expected",
    [("cfg", 1000), ("smc", 1000), ("harpoon", 1000), ("rejection", 200), ("ideal", 200)],
)
def test_conditional_requests_and_no_adherence_filter(tmp_path, monkeypatch, method, expected):
    train = pd.DataFrame({"x": np.arange(1000, dtype=float), "a": "x"})
    test = train.iloc[:200].copy()
    ev = session(tmp_path, train, test)
    predicate = Condition(equality={"a": "x"})
    battery = pd.DataFrame(
        [
            dict(
                slice_id="s",
                predicate=predicate.dumps(),
                task="equality",
                train_count=1000,
                test_count=200,
                bucket=">3%",
                train_fraction=1.0,
                reference_train_fraction=0.5,
            )
        ]
    )
    monkeypatch.setattr(
        ev, "slices", lambda *args: (battery, {"csv_sha256": "shared", "task": "equality"})
    )
    calls = {}

    def generate(method, condition, n):
        calls["requested"] = n
        frame = pd.DataFrame({"x": np.zeros(n), "a": "not-x"})
        return frame, dict(
            generation_seed=12, diagnostics={"pre_pin_adherence": 0.0, "post_pin_adherence": 0.0}
        )

    def fidelity(real, generated, schema, seed, cap):
        calls["scored_input"] = len(generated)
        if method != "ideal":
            assert generated.a.eq("not-x").all()
        return {"shape": 0.5}

    monkeypatch.setattr(ev, "generate", generate)
    monkeypatch.setattr("diffltm.evaluation.fidelity", fidelity)
    rows, _ = ev.conditional("unused", method)
    assert rows[0]["requested"] == expected and rows[0]["n_scored"] == 200
    assert calls["scored_input"] == expected


def test_cps_fixed_budget_generates_only_named_state():
    from diffltm.economics.experiments import requests

    for name, state in [("cps_wage_az", "AZ"), ("cps_wage_pa", "PA")]:
        reference = pd.DataFrame(
            {
                "STATE": [state] * 4,
                "YEAR": [2016, 2016, 2019, 2019],
                "EDUC": [39, 43, 39, 43],
                "OCC_MAJOR": [10, 14, 16, 10],
            }
        )
        table = SimpleNamespace(manifest={"name": name, "held_state": state}, reference=reference)
        result = requests(table, "cps", None)
        assert len(result) == (20 if state == "AZ" else 10)
        assert all(n == 100000 and len(c.equality) == 2 for _, c, n in result[:10])
        assert all(len(c.equality) == 3 and n == 100000 for _, c, n in result[10:])
        assert {c.equality["STATE"] for _, c, _ in result} == {state}


def test_undefined_scores_do_not_enter_averages(tmp_path):
    ev = session(tmp_path, pd.DataFrame(), pd.DataFrame())
    rows = ev.score_rows({"dcr": float("inf"), "mle": float("nan")})
    assert all(r["status"] == "undefined" and np.isnan(r["value"]) for r in rows)


def test_rejection_shortfall_omits_metrics_and_continues(tmp_path, monkeypatch):
    train = pd.DataFrame({"x": np.arange(20, dtype=float), "a": "x"})
    ev = session(tmp_path, train, train.iloc[:10].copy())
    battery = pd.DataFrame(
        [
            dict(
                slice_id=s,
                predicate=Condition(equality={"a": "x"}).dumps(),
                task="equality",
                train_count=20,
                test_count=10,
                bucket=">3%",
                train_fraction=1.0,
                reference_train_fraction=0.5,
            )
            for s in ["short", "full"]
        ]
    )
    monkeypatch.setattr(ev, "slices", lambda *args: (battery, {"csv_sha256": "shared"}))
    draws = iter([3, 10])

    def generate(method, condition, n):
        returned = next(draws)
        return train.iloc[:returned], dict(
            generation_seed=12,
            diagnostics=dict(
                budget_exhausted=returned < n, candidate_budget=10000000, pool_extent=10000000
            ),
        )

    scored = []

    def fidelity(real, generated, *args, **kwargs):
        scored.append(len(generated))
        return {"shape": 0.5}

    monkeypatch.setattr(ev, "generate", generate)
    monkeypatch.setattr("diffltm.evaluation.fidelity", fidelity)
    rows, _ = ev.conditional("unused", "rejection")
    assert scored == [10]
    missing = [r for r in rows if r["slice"] == "short"]
    assert len(missing) == 4
    assert all(
        r["status"] == "undefined"
        and np.isnan(r["value"])
        and r["n_scored"] == 0
        and r["returned"] == 3
        and r["budget_exhausted"]
        and "budget exhausted" in r["omission_reason"]
        for r in missing
    )
    assert rows[-1]["value"] == 0.5 and not rows[-1]["budget_exhausted"]
