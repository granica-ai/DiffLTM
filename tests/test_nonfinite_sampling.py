"""Regression coverage for the agreed whole-row redraw policy."""

import json

import pandas as pd
import pytest
import torch
from helpers import assert_rng_equal, config, rng_state

from diffltm import periodic, trainer
from diffltm.artifacts import completed_training
from diffltm.evaluation import Evaluation
from diffltm.generation import sampling
from diffltm.generation.conditioning import Condition, channels
from diffltm.models.model import DiffLTM, sigma

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")


@torch.no_grad()
def original_euler(model, n, seed, condition, guidance, pin, batch_size, steps):
    """Frozen pre-recovery integration/decoding for exact compatibility checks."""
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    encoded = channels(model, condition, 1) if condition is not None else None
    frames = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x = torch.randn(b, model.dim, device=device, generator=rng) * 80
        cond = tuple(v.expand(b, -1) for v in encoded) if encoded is not None else None
        times = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            predicted, logits = model.predict(x, times[i].expand(b), cond)
            if cond is not None and guidance != 1:
                null, null_logits = model.predict(x, times[i].expand(b))
                predicted = null + guidance * (predicted - null)
                if logits is not None:
                    logits = null_logits + guidance * (logits - null_logits)
                    predicted = torch.cat(
                        [predicted[:, : model.n_num], model.logits_to_representation(logits)], 1
                    )
            x = predicted + (sigma(times[i + 1]) / sigma(times[i])) * (x - predicted)
        frames.append(model.decode(x, logits))
    frame = pd.concat(frames, ignore_index=True)
    return condition.pin(frame) if pin and condition is not None else frame


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize(
    "objective,component,guidance",
    [
        ("representation_mse", "state", 1),
        ("logits_mse", "state", 2),
        ("logits_ce", "logits", 1),
    ],
)
@pytest.mark.parametrize("conditional", [False, True])
def test_preserves_valid_rows_normal_draws_settings_and_rng(
    preprocessor, model_config, monkeypatch, device, objective, component, guidance, conditional
):
    model = DiffLTM(
        preprocessor, dict(model_config, representation="onehot", objective=objective)
    ).to(device)
    condition = (
        Condition(equality={"a": "two"}, ranges={"x": (-0.01, 0.01)}) if conditional else None
    )
    args = (model, 19, 123, condition, guidance, True, 8, 3)
    expected = original_euler(*args)
    before = rng_state()
    actual, meta = sampling.euler(*args)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert meta["recovery"]["discarded_rows"] == 0
    assert_rng_equal(before)
    original = model.predict
    call = 0
    # Corrupt two separate positions at the final conditional prediction of batch 1.
    fail_at = 5 if conditional and guidance != 1 else 3

    def predict(x, t, cond=None):
        nonlocal call
        call += 1
        prediction, logits = original(x, t, cond)
        if call == fail_at:
            target = prediction if component == "state" else logits
            target[1, 0] = float("nan")
            target[5, 0] = float("inf")
        return prediction, logits

    monkeypatch.setattr(model, "predict", predict)
    recovered, diagnostics = sampling.euler(*args)
    assert_rng_equal(before)
    keep = [i for i in range(19) if i not in (1, 5)]
    pd.testing.assert_frame_equal(recovered.iloc[keep], expected.iloc[keep], check_exact=True)
    assert len(recovered) == 19
    recovery = diagnostics["recovery"]
    assert recovery["attempted_rows"] == 21
    assert recovery["discarded_rows"] == recovery["replacement_rows"] == 2
    assert recovery["retry_rounds"] == recovery["max_retry_rounds_used"] == 1
    if condition is not None:
        assert recovered.a.eq("two").all()
        assert diagnostics["post_pin_adherence"] == float(condition.mask(recovered).mean())
        assert not condition.mask(recovered).all()  # Finite nonadherent rows were retained.
    call = 0
    repeated, repeat_meta = sampling.euler(*args)
    pd.testing.assert_frame_equal(recovered, repeated, check_exact=True)
    assert repeat_meta == diagnostics


class ToyModel(torch.nn.Module):
    dim = 1

    def __init__(self, bad_by_call):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.bad_by_call = bad_by_call
        self.calls = 0

    def predict(self, x, t, condition=None):
        count = self.bad_by_call.get(self.calls, 0)
        self.calls += 1
        result = x.clone()
        result[:count] = float("nan")
        return result, None

    def decode(self, state, logits):
        assert torch.isfinite(state).all()
        return pd.DataFrame({"x": state[:, 0].cpu().numpy()})


def test_retry_exhaustion_and_repeated_failures_are_counted():
    model = ToyModel({0: 1, 1: 1, 2: 1, 3: 1})
    with pytest.raises(sampling.NonfiniteGenerationError, match="retry rounds exhausted") as exc:
        sampling.euler(model, 4, 1, steps=1)
    recovery = exc.value.diagnostics
    assert model.calls == 4
    assert recovery["retry_rounds"] == 3
    assert recovery["attempted_rows"] == 7 and recovery["discarded_rows"] == 4
    assert recovery["replacement_rows"] == 1


def test_retry_population_shrinks_and_succeeds_on_last_round():
    model = ToyModel({0: 3, 1: 2, 2: 1})
    frame, diagnostics = sampling.euler(model, 8, 1, steps=1)
    recovery = diagnostics["recovery"]
    assert len(frame) == 8 and model.calls == 4
    assert recovery["attempted_rows"] == 14 and recovery["discarded_rows"] == 6
    assert recovery["replacement_rows"] == 3 and recovery["retry_rounds"] == 3
    # The first recovered position must not be redrawn in subsequent rounds.
    rng = torch.Generator().manual_seed(recovery["retry_seed"])
    first_retry = torch.randn(3, 1, generator=rng) * 80
    assert frame.x.iloc[2] == first_retry[2, 0].item()


@pytest.mark.parametrize("n,bad,budget", [(11, 10, 10), (10001, 11, 11)])
def test_discard_budget_boundary(n, bad, budget):
    frame, diagnostics = sampling.euler(ToyModel({0: bad}), n, 1, batch_size=n, steps=1)
    assert len(frame) == n and diagnostics["recovery"]["discard_budget"] == budget
    with pytest.raises(sampling.NonfiniteGenerationError, match="discard budget exceeded") as exc:
        sampling.euler(ToyModel({0: bad + 1}), n, 1, batch_size=n, steps=1)
    assert exc.value.diagnostics["retry_rounds"] == 0


def test_discard_budget_accumulates_across_batches_and_retries():
    for failures in ({0: 6, 2: 6}, {0: 6, 1: 5}):
        with pytest.raises(
            sampling.NonfiniteGenerationError, match="discard budget exceeded"
        ) as exc:
            sampling.euler(ToyModel(failures), 12, 1, batch_size=6, steps=1)
        assert exc.value.diagnostics["discarded_rows"] == sum(failures.values())
        assert exc.value.diagnostics["retry_rounds"] == 1


def test_nonfinite_parameters_and_decoder_stay_fatal(preprocessor, model_config):
    model = DiffLTM(preprocessor, model_config)
    with pytest.raises(ValueError, match="Nonfinite generation endpoint"):
        model.decode(torch.full((1, model.dim), float("nan")))
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite model parameter"):
        sampling.euler(model, 1, 1)


def test_evaluation_policy_and_failed_diagnostics(
    tmp_path, small_table, preprocessor, model_config, monkeypatch
):
    ev = object.__new__(Evaluation)
    ev.model = DiffLTM(preprocessor, model_config)
    ev.table = small_table
    ev.seed, ev.guidance, ev.pin = 500, 1, True
    ev.checkpoint_id, ev.identity = "checkpoint", {}
    ev.run_dir, ev.sample_root = tmp_path, tmp_path / "samples"
    first, a = ev.generate("unconditional", None, 4)
    repeated, b = ev.generate("unconditional", None, 4)
    pd.testing.assert_frame_equal(first, repeated)
    assert a == b and a["request"]["recovery_policy"] == sampling.RECOVERY_POLICY
    monkeypatch.setitem(sampling.RECOVERY_POLICY, "max_retry_rounds", 2)
    _, c = ev.generate("unconditional", None, 4)
    assert c["sample_key"] != a["sample_key"]

    def fail(*args):
        raise sampling.NonfiniteGenerationError("retry rounds exhausted", {"discarded_rows": 4})

    monkeypatch.setattr("diffltm.evaluation.euler", fail)
    with pytest.raises(sampling.NonfiniteGenerationError):
        ev.generate("unconditional", None, 5)
    failed = [json.loads(p.read_text()) for p in ev.sample_root.glob("*/manifest.json")]
    failed = [m for m in failed if m["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["diagnostics"]["recovery"]["discarded_rows"] == 4


@pytest.mark.parametrize("exhaust", [False, True])
def test_periodic_recovery_metadata_and_completion_gate(
    tmp_path, small_table, model_config, monkeypatch, exhaust
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    original = periodic.sample_snapshot

    def sample_snapshot(model, ema, n, seed):
        predict = model.predict
        calls = 0

        def prediction(*args):
            nonlocal calls
            calls += 1
            state, logits = predict(*args)
            if calls == 100 or (exhaust and calls % 100 == 0):
                state[0, 0] = float("nan")
            return state, logits

        before = rng_state()
        with monkeypatch.context() as patch:
            patch.setattr(model, "predict", prediction)
            try:
                return original(model, ema, n, seed)
            finally:
                assert_rng_equal(before)

    monkeypatch.setattr(trainer, "sample_snapshot", sample_snapshot)
    cfg = config(tmp_path, model_config)
    cfg["training"]["periodic_every"] = 1
    if exhaust:
        with pytest.raises(sampling.NonfiniteGenerationError):
            trainer.train(cfg)
        with pytest.raises(ValueError, match="completed training"):
            completed_training(tmp_path / "run")
    else:
        trainer.train(cfg)
        completed_training(tmp_path / "run")
    result = json.loads((tmp_path / "run/periodic/epoch-1.json").read_text())
    assert result["status"] == ("failed" if exhaust else "complete")
    recovery = result["generation"]["recovery"]
    assert recovery["discarded_rows"] == (4 if exhaust else 1)
    assert recovery["retry_rounds"] == (3 if exhaust else 1)


def test_harpoon_budget_only_changes_stopping_limit():
    from diffltm.generation.harpoon import HARPOON_RECOVERY_POLICY

    def run(bad, policy=None):
        model = ToyModel({0: bad})

        def draw(n, rng):
            return model.predict(torch.randn(n, 1, generator=rng), None)

        return sampling._sample_rows(model, 1000, 73, draw, recovery_policy=policy)

    old, a = run(1)
    new, b = run(1, HARPOON_RECOVERY_POLICY)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    assert a["recovery"]["retry_seed"] == b["recovery"]["retry_seed"]
    assert a["recovery"]["discard_budget"] == 10 and b["recovery"]["discard_budget"] == 100
    assert b["recovery"]["policy"] == HARPOON_RECOVERY_POLICY
    with pytest.raises(sampling.NonfiniteGenerationError, match="discard budget exceeded"):
        run(16)
    frame, meta = run(100, HARPOON_RECOVERY_POLICY)
    assert len(frame) == 1000 and meta["recovery"]["discarded_rows"] == 100
    with pytest.raises(sampling.NonfiniteGenerationError, match="discard budget exceeded"):
        run(101, HARPOON_RECOVERY_POLICY)


def test_harpoon_removes_round_cap_but_still_stops_at_discard_budget():
    from diffltm.generation.harpoon import HARPOON_RECOVERY_POLICY

    def run(failures, policy):
        model = ToyModel(failures)
        return sampling._sample_rows(
            model,
            100,
            73,
            lambda n, rng: model.predict(torch.randn(n, 1, generator=rng), None),
            recovery_policy=policy,
        )

    failures = {i: 1 for i in range(4)}
    with pytest.raises(sampling.NonfiniteGenerationError, match="retry rounds exhausted"):
        run(failures, dict(HARPOON_RECOVERY_POLICY, max_retry_rounds=3))
    frame, meta = run(failures, HARPOON_RECOVERY_POLICY)
    assert len(frame) == 100
    assert meta["recovery"]["max_retry_rounds_used"] == 4
    assert meta["recovery"]["discarded_rows"] == 4
    with pytest.raises(sampling.NonfiniteGenerationError, match="discard budget exceeded") as exc:
        run({i: 1 for i in range(100)}, HARPOON_RECOVERY_POLICY)
    assert exc.value.diagnostics["discarded_rows"] == 11
    assert exc.value.diagnostics["attempted_rows"] == 110


def test_harpoon_removing_round_cap_preserves_successful_multibatch_draws():
    from diffltm.generation.harpoon import HARPOON_RECOVERY_POLICY

    def run(policy):
        model = ToyModel({0: 2, 1: 1, 3: 1})
        return sampling._sample_rows(
            model,
            16,
            73,
            lambda n, rng: model.predict(torch.randn(n, 1, generator=rng), None),
            batch_size=8,
            recovery_policy=policy,
        )

    old, a = run(dict(HARPOON_RECOVERY_POLICY, max_retry_rounds=3))
    new, b = run(HARPOON_RECOVERY_POLICY)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    assert a["recovery"].pop("policy")["max_retry_rounds"] == 3
    assert b["recovery"].pop("policy")["max_retry_rounds"] is None
    assert a == b
