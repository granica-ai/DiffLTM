"""Appendix D.4 contracts and hand-calculated fixtures from the retained implementation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import assert_rng_equal, config, rng_state

from diffltm import evaluation, trainer
from diffltm.data.preprocessing import Preprocessor
from diffltm.generation import harpoon, sampling
from diffltm.generation.conditioning import Condition, channels, transform_num_ranges
from diffltm.models.model import DiffLTM

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")


@pytest.mark.parametrize(
    "constraints",
    [
        {},
        {"equality": {"a": "two"}},
        {"ranges": {"x": (0, 1)}},
        {"weights": {"x": 1}, "bounds": (0, 1)},
    ],
)
def test_harpoon_rejects_arbitrary_predicates_before_loading_them(constraints):
    condition = Condition(predicate="untrusted.py:predicate:unused", **constraints)
    with pytest.raises(ValueError, match="Harpoon does not support arbitrary"):
        harpoon.prepare(None, condition, 0.2)


@pytest.mark.parametrize("batch", [1, 7])
def test_separate_range_norms_and_full_equality_gradient(batch):
    x = torch.zeros(batch, 9, requires_grad=True)
    bounds = torch.tensor(
        [
            [3.0, float("inf")],
            [4.0, float("inf")],
            [-float("inf"), -5.0],
            [-float("inf"), -12.0],
            [float("nan"), float("nan")],
            [-float("inf"), float("inf")],
        ]
    )
    target, mask = torch.zeros(1, 9), torch.zeros(1, 9)
    target[:, 6:] = torch.tensor([1.0, -2.0, 3.0])
    mask[:, 6:] = 1
    loss = harpoon.guidance_loss(x, target, mask, bounds)
    torch.testing.assert_close(loss, torch.full((batch,), 18.0 + 6.0))
    (gradient,) = torch.autograd.grad(loss.sum(), x)
    expected = torch.tensor([-0.6, -0.8, 5 / 13, 12 / 13, 0.0, 0.0, -1.0, 1.0, -1.0])
    torch.testing.assert_close(gradient, expected.expand(batch, -1))


@pytest.mark.parametrize("values", [[0.0, 1.0, 2.0], [1.0, 1.5, 2.0]])
def test_satisfied_boundaries_and_unbounded_endpoints_have_zero_gradient(values):
    x = torch.tensor([values], requires_grad=True)
    bounds = torch.tensor([[0.0, 1.0], [1.0, 2.0], [float("nan"), float("nan")]])
    for b in (
        bounds,
        torch.full((3, 2), float("nan")),
        torch.tensor([[-float("inf"), float("inf")]] * 3),
    ):
        loss = harpoon.guidance_loss(x, torch.zeros_like(x), torch.zeros_like(x), b)
        (gradient,) = torch.autograd.grad(loss.sum(), x)
        assert torch.equal(loss, torch.zeros_like(loss))
        assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("eta", [0.0, 0.2])
def test_step_matches_retained_cross_block_jacobian_fixture(batch, eta, monkeypatch):
    def predict(x, t):
        num, cat = x[:, :3], x[:, 3:]
        return torch.cat(
            [
                2 * num + cat.sum(1, keepdim=True),
                3 * cat + num.sum(1, keepdim=True) + cat.new_tensor([-1.0, 1.0, 2.0]),
            ],
            1,
        ), None

    monkeypatch.setattr(harpoon, "sigma", lambda t: t)
    target = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    bounds = torch.tensor([[3.0, float("inf")], [4.0, float("inf")], [float("nan"), float("nan")]])
    result, logits = harpoon._step(
        SimpleNamespace(predict=predict),
        torch.zeros(batch, 6),
        torch.tensor(2.0),
        torch.tensor(1.0),
        target,
        mask,
        bounds,
        eta,
    )
    expected = torch.tensor([0.0, 0.0, 0.0, -0.5, 0.5, 1.0]) - eta * torch.tensor(
        [-0.2, -0.6, -1.0, -5.4, 0.6, 0.6]
    )
    torch.testing.assert_close(result, expected.expand(batch, -1))
    assert not result.requires_grad


def test_condition_conversion_full_onehot_and_unclamped_snapped_bounds(small_table, harpoon_config):
    p = Preprocessor(small_table.schema).fit(small_table.frame)
    model = DiffLTM(p, harpoon_config)
    condition = Condition(
        equality={"a": "two", "y": 0.1}, ranges={"x": (p.raw_min[0] - 1, p.raw_max[0] + 1)}
    )
    target, mask, bounds, linear = harpoon.prepare(model, condition, 0.2)
    assert linear is None
    start, width = model.n_num, model.widths[0]
    assert mask[0, start : start + width].eq(1).all()
    assert target[0, start : start + width].sum() == 1
    assert target[0, start + p.maps[0]["two"]] == 1
    assert torch.isneginf(bounds[0, 0]) and torch.isposinf(bounds[0, 1])
    assert np.allclose(bounds[0].numpy(), transform_num_ranges(p, condition.ranges)["x"])
    with pytest.raises(ValueError, match="range channels"):
        channels(model, condition, 1)
    raw = p.mean.copy()
    raw[1] = 0.1
    assert target[0, 1].item() == pytest.approx(p.transformer.transform(raw[None])[0, 1])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_zero_eta_euler_equivalence_and_parameter_gradient_rng_preservation(
    preprocessor, harpoon_config, device
):
    model = DiffLTM(preprocessor, harpoon_config).to(device)
    for i, parameter in enumerate(model.parameters()):
        parameter.grad = torch.ones_like(parameter) if i % 2 else None
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    gradients = {
        name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()
    }
    condition = Condition(equality={"a": "two"}, ranges={"x": (-0.01, 0.01)})
    before = rng_state()
    original, _ = sampling.euler(model, 19, 33, batch_size=8, steps=4)
    zero, meta = harpoon.sample(model, 19, 33, condition, eta=0, batch_size=8, steps=4)
    pd.testing.assert_frame_equal(original, zero, check_exact=True)
    assert meta["equality_pinning"] is False
    assert meta["adherence"] == float(condition.mask(zero).mean())
    assert not condition.mask(zero).all()
    guided, meta = harpoon.sample(model, 19, 33, condition, batch_size=8, steps=4)
    repeated, repeated_meta = harpoon.sample(model, 19, 33, condition, batch_size=8, steps=4)
    pd.testing.assert_frame_equal(guided, repeated, check_exact=True)
    assert meta == repeated_meta
    assert_rng_equal(before)
    for name, p in model.named_parameters():
        assert torch.equal(p, weights[name])
        assert p.grad is None if gradients[name] is None else torch.equal(p.grad, gradients[name])


@pytest.mark.parametrize("eta", [-1, float("nan"), float("inf")])
def test_invalid_eta(preprocessor, harpoon_config, eta):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        harpoon.sample(
            DiffLTM(preprocessor, harpoon_config), 2, 1, Condition(equality={"a": "two"}), eta
        )


@pytest.mark.parametrize("condition", [None, Condition()])
def test_invalid_predicates(preprocessor, harpoon_config, condition):
    with pytest.raises(ValueError, match="equality/range"):
        harpoon.sample(DiffLTM(preprocessor, harpoon_config), 2, 1, condition)


@pytest.mark.parametrize(
    "change",
    [
        {"representation": "learnt"},
        {"objective": "logits_mse"},
        {"objective": "logits_ce"},
        {"conditional": True},
    ],
)
def test_checkpoint_contract(preprocessor, harpoon_config, change):
    model = DiffLTM(preprocessor, dict(harpoon_config, **change))
    with pytest.raises(ValueError, match="unconditional onehot_repr"):
        harpoon.sample(model, 2, 1, Condition(equality={"a": "two"}))


@pytest.mark.parametrize("exhaust", [False, True])
def test_recovery_repeats_guided_trajectory_only_and_preserves_valid_rows(
    preprocessor, harpoon_config, monkeypatch, exhaust
):
    model = DiffLTM(preprocessor, harpoon_config)
    condition = Condition(equality={"a": "two"}, ranges={"x": (-1, 1)})
    args = dict(eta=0.2, batch_size=8, steps=3)
    expected, _ = harpoon.sample(model, 19, 123, condition, **args)
    original = harpoon._step
    calls = []

    def step(model, x, t, tn, target, mask, bounds, eta, linear=None):
        calls.append(len(x))
        assert eta == 0.2 and mask.sum() == model.widths[0]
        assert torch.isfinite(bounds[0]).all()
        result, logits = original(model, x, t, tn, target, mask, bounds, eta, linear)
        if len(calls) == 3 or (exhaust and len(calls) % 3 == 0):
            result[0, 0] = float("nan")
        return result, logits

    monkeypatch.setattr(harpoon, "_step", step)
    before = rng_state()
    if exhaust:
        with pytest.raises(
            sampling.NonfiniteGenerationError, match="discard budget exceeded"
        ) as exc:
            harpoon.sample(model, 19, 123, condition, **args)
        assert exc.value.diagnostics["discarded_rows"] == 11
        assert exc.value.diagnostics["max_retry_rounds_used"] == 10
    else:
        actual, meta = harpoon.sample(model, 19, 123, condition, **args)
        pd.testing.assert_frame_equal(actual.iloc[1:], expected.iloc[1:], check_exact=True)
        assert calls == [8] * 3 + [1] * 3 + [8] * 3 + [3] * 3
        assert meta["recovery"]["attempted_rows"] == 20
        assert meta["recovery"]["discarded_rows"] == 1
    assert_rng_equal(before)


@pytest.mark.parametrize("recipe", ["onehot_repr", "learnt_ce"])
def test_completed_checkpoint_evaluator_smoke_and_request_identity(
    tmp_path, small_table, harpoon_config, monkeypatch, recipe
):
    if recipe == "learnt_ce":
        harpoon_config = dict(harpoon_config, representation="learnt", objective="logits_ce")
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    monkeypatch.setattr(evaluation, "load_table", lambda *args: small_table)
    cfg = config(tmp_path, harpoon_config)
    trainer.train(cfg)
    ev = evaluation.Evaluation(cfg["output_dir"], cache=cfg["cache"])
    condition = Condition(equality={"a": "two"}, ranges={"x": (-1, 1)})
    frame, meta = ev.generate("harpoon", condition, 12)
    request = meta["request"]
    assert len(frame) == 12 and request["harpoon_eta"] == 0.2 and not request["pin"]
    assert request["guidance_policy"] == harpoon.guidance_policy(ev.model)
    assert meta["diagnostics"]["adherence"] == float(condition.mask(frame).mean())
    ev.guidance, ev.pin = 8, False  # Neither setting changes Harpoon.
    again, cached = ev.generate("harpoon", condition, 12)
    pd.testing.assert_frame_equal(frame, again)
    assert cached == meta
    ev.harpoon_eta = 0
    _, other = ev.generate("harpoon", condition, 12)
    assert other["sample_key"] != meta["sample_key"]
    ev.harpoon_eta = 0.2
    original_identity = evaluation.file_identity
    monkeypatch.setattr(
        evaluation,
        "file_identity",
        lambda path: (
            "changed-guidance-code" if Path(path).name == "harpoon.py" else original_identity(path)
        ),
    )
    _, changed = ev.generate("harpoon", condition, 12)
    assert changed["sample_key"] == meta["sample_key"]
    monkeypatch.setattr(evaluation, "file_identity", original_identity)
    for task, predicate in [
        ("equality", Condition(equality={"a": "two"})),
        ("range", Condition(ranges={"x": (-1, 1)})),
        ("linear", Condition(weights={"x": 1, "y": -1}, bounds=(-1, 1))),
    ]:
        train_n = int(predicate.mask(ev.frames["train"]).sum())
        test_n = int(predicate.mask(ev.frames["test"]).sum())
        battery = pd.DataFrame(
            [
                dict(
                    slice_id=task,
                    task=task,
                    predicate=predicate.dumps(),
                    train_count=train_n,
                    test_count=test_n,
                    bucket=">3%",
                    train_fraction=train_n / len(ev.frames["train"]),
                    reference_train_fraction=0.5,
                )
            ]
        )
        monkeypatch.setattr(
            ev, "slices", lambda *args: (battery, {"task": task, "csv_sha256": "fixture-battery"})
        )
        rows, _ = ev.conditional("fixture", "harpoon")
        assert len(rows) == 4 and all(r["requested"] == train_n for r in rows)
        assert all(r["n_scored"] == min(train_n, test_n) for r in rows)
        assert all(r["adherence"] is not None for r in rows)
    monkeypatch.setattr(ev, "slices", lambda *args: (None, {"task": "mincer"}))
    with pytest.raises(ValueError, match="equality/range/linear tasks"):
        ev.conditional("fixture", "harpoon")
    with pytest.raises(ValueError, match="economics"):
        ev.economic("mincer", "harpoon")
