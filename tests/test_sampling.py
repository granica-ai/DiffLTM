import numpy as np
import pandas as pd
import pytest
import torch

from diffltm.generation.conditioning import Condition
from diffltm.generation.rejection import RejectionPool
from diffltm.generation.sampling import euler
from diffltm.generation.smc import allocation, resample, sample
from diffltm.models.model import DiffLTM


def test_smc_positive_support_and_allocation():
    per, n = allocation(100000)
    assert (per, n) == (25000, 25000)
    assert allocation(100) == (25, 8200)
    rng = torch.Generator().manual_seed(0)
    idx = resample(torch.tensor([-torch.inf, 0.0, -1000.0, -torch.inf]), 100, rng)
    assert idx.eq(1).all()
    with pytest.raises(RuntimeError):
        resample(torch.full((5,), -torch.inf), 10, rng)


def pool_diagnostics(n):
    return dict(
        recovery=dict(
            attempted_rows=n + 1,
            discarded_rows=1,
            replacement_rows=1,
            retry_rounds=1,
            max_retry_rounds_used=1,
        )
    )


def test_rejection_shares_rejected_rows_within_evaluation(tmp_path):
    calls = []

    def sample(n, seed):
        calls.append(seed)
        return pd.DataFrame({"a": ["one", "two"] * (n // 2), "x": np.arange(n)}), pool_diagnostics(
            n
        )

    pool = RejectionPool(sample, 10, chunk_size=4, directory=tmp_path)
    a, meta = pool.get(Condition(equality={"a": "one"}), 3)
    assert len(a) == 3 and len(calls) == 2 and meta["pool_extent"] == 8
    assert meta["recovery"]["attempted_rows"] == 10
    assert meta["recovery"]["discarded_rows"] == meta["recovery"]["replacement_rows"] == 2
    b, _ = pool.get(Condition(equality={"a": "two"}), 4)
    assert len(b) == 4 and len(calls) == 2
    assert len(set(calls)) == 2


def test_rejection_budget_reuses_pool_without_additional_draws(tmp_path, monkeypatch):
    import diffltm.generation.rejection as sampling

    calls = []

    def sample(n, seed):
        calls.append(n)
        return pd.DataFrame({"a": ["one"] * n}), pool_diagnostics(n)

    monkeypatch.setattr(sampling, "REJECTION_MAX_CANDIDATES", 10)
    pool = RejectionPool(sample, 10, chunk_size=4, directory=tmp_path)
    frame, meta = pool.get(Condition(equality={"a": "absent"}), 1)
    assert frame.empty and meta["budget_exhausted"] and meta["pool_extent"] == 10
    assert calls == [4, 4, 2]
    frame, meta = pool.get(Condition(equality={"a": "one"}), 10)
    assert len(frame) == 10 and not meta["budget_exhausted"] and calls == [4, 4, 2]


def test_smc_exact_terminal_and_no_empty_support(preprocessor, model_config, monkeypatch):
    model = DiffLTM(preprocessor, model_config)

    def prediction(x, t, condition=None):
        logits = torch.zeros(len(x), sum(model.cards))
        logits[:, preprocessor.maps[0]["two"]] = 100
        logits[:, model.cards[0] + 1] = 100
        clean = torch.zeros_like(x)
        return clean, logits

    monkeypatch.setattr(model, "predict", prediction)
    c = Condition(
        equality={"a": "two"}, ranges={"x": (-10, 10)}, weights={"x": 1, "y": 1}, bounds=(-20, 20)
    )
    frame, meta = sample(model, 12, 17, c, steps=3, floor=20)
    assert c.mask(frame).all() and meta["pre_pin_adherence"] == 1 and len(meta["replicates"]) == 4
    with pytest.raises(RuntimeError, match="support"):
        sample(model, 12, 17, Condition(equality={"a": "one"}), steps=3, floor=20)


@pytest.mark.parametrize(
    "condition",
    [
        Condition(equality={"a": "two"}),
        Condition(ranges={"x": (-1.0, 1.0)}),
        Condition(weights={"x": 1.0, "y": -2.0}, bounds=(-1.0, 2.0)),
    ],
)
def test_cfg_all_condition_families_score_raw_adherence(preprocessor, model_config, condition):

    model = DiffLTM(preprocessor, model_config)
    raw, meta = euler(model, 30, 12, condition, pin=False, steps=4)
    assert len(raw) == 30 and meta["pre_pin_adherence"] == float(condition.mask(raw).mean())
    pinned, pinned_meta = euler(model, 30, 12, condition, pin=True, steps=4)
    assert meta["pre_pin_adherence"] == pinned_meta["pre_pin_adherence"]
    if condition.ranges or condition.weights:
        pd.testing.assert_frame_equal(raw, pinned)  # No unannounced range or slab projection.
