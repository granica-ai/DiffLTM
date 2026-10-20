"""Scientific fixtures for the normalized continuous linear Harpoon extension."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import assert_rng_equal, rng_state

from diffltm.data.core import Schema
from diffltm.data.preprocessing import Preprocessor
from diffltm.generation import harpoon, sampling
from diffltm.generation.conditioning import Condition, raw_to_standard
from diffltm.models.model import DiffLTM

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")


def test_affine_conversion_matches_raw_predicate_and_positive_rescaling(
    preprocessor, harpoon_config
):
    model = DiffLTM(preprocessor, harpoon_config).double()
    condition = Condition(weights={"x": 2, "y": -3}, bounds=(-1, 2))
    target, mask, bounds, linear = harpoon.prepare(model, condition, 0.2)
    scaled = Condition(weights={"x": 14, "y": -21}, bounds=(-7, 14))
    other = harpoon.prepare(model, scaled, 0.2)[3]
    for a, b in zip(linear, other):
        torch.testing.assert_close(a, b)
    # Far outside the fitted grid: the standardized inverse must remain affine.
    state = torch.zeros(4, model.dim)
    state[:, :2] = torch.tensor([[-100.0, 200.0], [100.0, -200.0], [0.0, 0.0], [0.3, 0.2]])
    raw = preprocessor.transformer.inverse_transform(state[:, :2].numpy())
    raw_projection = raw @ np.array([2.0, -3.0])
    norm = np.linalg.norm(np.array([2.0, -3.0]) * preprocessor.scale)
    expected = (np.maximum(-1 - raw_projection, 0) + np.maximum(raw_projection - 2, 0)) / norm
    loss = harpoon.guidance_loss(state, target, mask, bounds, model, linear)
    np.testing.assert_allclose(loss.numpy(), expected, rtol=2e-6, atol=2e-7)
    assert np.array_equal(loss.numpy() == 0, (raw_projection >= -1) & (raw_projection <= 2))
    loss_scaled = harpoon.guidance_loss(state, target, mask, bounds, model, other)
    torch.testing.assert_close(loss, loss_scaled)
    weights, interval = raw_to_standard(condition.weights, condition.bounds, preprocessor)
    z = (raw - preprocessor.mean) / preprocessor.scale
    np.testing.assert_allclose(
        z @ weights, (raw_projection - np.array([2.0, -3.0]) @ preprocessor.mean) / norm
    )
    assert np.array_equal(
        (z @ weights >= interval[0]) & (z @ weights <= interval[1]), loss.numpy() == 0
    )


@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("projection,sign", [(-6.0, -1), (-3.0, 0), (0.0, 0), (3.0, 0), (6.0, 1)])
def test_complete_state_gradient_and_endpoint_subgradient(batch, projection, sign, monkeypatch):
    def predict(x, t):
        return torch.stack([2 * x[:, 0] + x[:, 2], 3 * x[:, 1] + 2 * x[:, 2], 4 * x[:, 2]], 1), None

    model = SimpleNamespace(
        n_num=2, preprocessor=SimpleNamespace(numerical="standardize"), predict=predict
    )
    x = torch.tensor([[projection / 1.2, 0.0, 0.0]], dtype=torch.float64).repeat(batch, 1)
    weights, interval = (
        torch.tensor([0.6, -0.8], dtype=x.dtype),
        torch.tensor([-3.0, 3.0], dtype=x.dtype),
    )
    target = mask = torch.zeros_like(x)
    bounds = torch.full((2, 2), float("nan"), dtype=x.dtype)
    monkeypatch.setattr(harpoon, "sigma", lambda t: t)
    result, logits = harpoon._step(
        model,
        x,
        torch.tensor(2.0),
        torch.tensor(1.0),
        target,
        mask,
        bounds,
        0.2,
        (weights, interval),
    )
    prior = 0.5 * (predict(x, None)[0] + x)
    expected_gradient = sign * torch.tensor([1.2, -2.4, -1.0], dtype=x.dtype)
    torch.testing.assert_close(result, prior - 0.2 * expected_gradient, atol=1e-12, rtol=1e-12)
    assert not result.requires_grad


def test_mixed_range_and_linear_violations_share_norms_not_independent_penalties():
    model = SimpleNamespace(n_num=3, preprocessor=SimpleNamespace(numerical="standardize"))
    x = torch.tensor([[0.0, 0.0, 0.0, 1000.0]], requires_grad=True)
    target = mask = torch.zeros_like(x)
    bounds = torch.tensor(
        [[3.0, float("inf")], [-float("inf"), -12.0], [float("nan"), float("nan")]]
    )
    linear = torch.tensor([0.0, 0.0, 1.0]), torch.tensor([4.0, float("inf")])
    loss = harpoon.guidance_loss(x, target, mask, bounds, model, linear)
    assert loss.item() == 17  # sqrt(3² + 4²) + 12, not 3 + 4 + 12.
    (gradient,) = torch.autograd.grad(loss.sum(), x)
    torch.testing.assert_close(gradient, torch.tensor([[-0.6, 1.0, -0.8, 0.0]]))


def test_quantile_lookup_inverse_finite_differences_and_clamped_tails(small_table, harpoon_config):
    p = Preprocessor(small_table.schema).fit(small_table.frame)
    model = DiffLTM(p, harpoon_config).double()
    grid = model.lookup_grid
    x = ((grid[[6000, 9000, 10000]] + grid[[6001, 9001, 10001]]) / 2).detach().requires_grad_(True)
    actual = model.raw_standardized(x)
    exact = (p.transformer.inverse_transform(x.detach().numpy()) - p.mean) / p.scale
    np.testing.assert_allclose(actual.detach().numpy(), exact, atol=2e-5, rtol=2e-5)
    weights = torch.tensor([0.6, -0.8], dtype=x.dtype)
    (gradient,) = torch.autograd.grad((actual * weights).sum(), x)
    h = (grid[1] - grid[0]) * 0.1
    differences = (
        model.raw_standardized(x.detach() + h) - model.raw_standardized(x.detach() - h)
    ) / (2 * h)
    torch.testing.assert_close(gradient, differences * weights, atol=1e-8, rtol=1e-8)
    tails = torch.stack([grid[0] - 100, grid[-1] + 100]).requires_grad_(True)
    output = model.raw_standardized(tails)
    (gradient,) = torch.autograd.grad(output.sum(), tails)
    assert not gradient.any()
    expected = (np.stack([p.raw_min, p.raw_max]) - p.mean) / p.scale
    np.testing.assert_allclose(output.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_lookup_plateau_has_zero_gradient(small_table, harpoon_config):
    frame = small_table.frame.copy()
    frame["y"] = 7.0
    p = Preprocessor(small_table.schema).fit(frame)
    model = DiffLTM(p, harpoon_config)
    x = torch.zeros(3, 2, requires_grad=True)
    (gradient,) = torch.autograd.grad(model.raw_standardized(x)[:, 1].sum(), x)
    assert torch.isfinite(gradient).all() and not gradient.any()


def test_integer_relaxation_reports_decoded_violation_without_filtering(
    small_table, harpoon_config, monkeypatch
):
    schema = Schema(["x", "y"], ["a", "b"], "y", "regression", ["x"])
    p = Preprocessor(schema, "standardize").fit(small_table.frame)
    model = DiffLTM(p, harpoon_config)
    condition = Condition(weights={"x": 1}, bounds=(0.2, 0.3))
    target, mask, bounds, linear = harpoon.prepare(model, condition, 0.2)
    clean = torch.zeros(1, model.dim)
    raw = p.mean.copy()
    raw[0] = 0.25
    clean[:, :2] = torch.tensor(p.transformer.transform(raw[None]), dtype=clean.dtype)
    assert harpoon.guidance_loss(clean, target, mask, bounds, model, linear).item() == 0
    monkeypatch.setattr(model, "predict", lambda x, t: (x * 0 + clean, None))
    frame, meta = harpoon.sample(model, 12, 42, condition, steps=1)
    assert len(frame) == 12 and frame.x.eq(0).all()
    assert meta["adherence"] == 0 and not condition.mask(frame).any()
    assert meta["recovery"]["discarded_rows"] == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("numerical", ["standardize", "quantile"])
@pytest.mark.parametrize("mixed", [False, True])
def test_linear_generation_zero_eta_recovery_and_gradient_isolation(
    small_table, harpoon_config, monkeypatch, device, numerical, mixed
):
    preprocessor = Preprocessor(small_table.schema, numerical).fit(small_table.frame)
    model = DiffLTM(preprocessor, harpoon_config).to(device)
    condition = Condition(
        equality={"a": "two"} if mixed else {},
        ranges={"x": (-1, 1)} if mixed else {},
        weights={"x": 1, "y": -2},
        bounds=(-0.1, 0.2),
    )
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    saved = [(p.detach().clone(), p.grad.clone()) for p in model.parameters()]
    before = rng_state()
    expected, _ = sampling.euler(model, 19, 44, steps=3, batch_size=8)
    zero, _ = harpoon.sample(model, 19, 44, condition, eta=0, steps=3, batch_size=8)
    pd.testing.assert_frame_equal(zero, expected, check_exact=True)
    baseline, _ = harpoon.sample(model, 19, 44, condition, steps=3, batch_size=8)
    original = harpoon._step
    prepared = harpoon.prepare(model, condition, 0.2)[3]
    calls = []

    def step(model, x, t, tn, target, mask, bounds, eta, linear=None):
        assert eta == 0.2 and linear is not None
        for a, b in zip(linear, prepared):
            assert torch.equal(a, b)
        calls.append(len(x))
        result, logits = original(model, x, t, tn, target, mask, bounds, eta, linear)
        if len(calls) == 3:
            result[0, 0] = float("nan")
        return result, logits

    monkeypatch.setattr(harpoon, "_step", step)
    recovered, meta = harpoon.sample(model, 19, 44, condition, steps=3, batch_size=8)
    pd.testing.assert_frame_equal(recovered.iloc[1:], baseline.iloc[1:], check_exact=True)
    assert calls == [8] * 3 + [1] * 3 + [8] * 3 + [3] * 3
    assert meta["recovery"]["discarded_rows"] == 1
    assert meta["adherence"] == float(condition.mask(recovered).mean())
    assert_rng_equal(before)
    for p, (value, gradient) in zip(model.parameters(), saved):
        assert torch.equal(p, value) and torch.equal(p.grad, gradient)


def test_quantile_guidance_uses_continuous_inverse_for_raw_linear_predicate(
    small_table, harpoon_config
):
    frame = small_table.frame.copy()
    frame["x"] = np.exp(frame.x)
    p = Preprocessor(small_table.schema).fit(frame)
    model = DiffLTM(p, harpoon_config).double()
    state = torch.zeros(1, model.dim, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        state[0, :2] = torch.tensor([-0.513, 0.417])
    raw = p.transformer.inverse_transform(state.detach().numpy()[:, :2])[0]
    a = np.array([2.0, -3.0])
    norm = np.linalg.norm(a * p.scale)
    center = a @ raw
    condition = Condition(
        weights=dict(zip(p.schema.numerical, a)), bounds=(center + 2 * norm, center + 3 * norm)
    )
    target, mask, bounds, linear = harpoon.prepare(model, condition, 0.2)
    loss = harpoon.guidance_loss(state, target, mask, bounds, model, linear)
    assert loss.item() == pytest.approx(2.0, abs=5e-5)
    (gradient,) = torch.autograd.grad(loss.sum(), state)
    for j in range(model.n_num):
        plus, minus = state.detach().clone(), state.detach().clone()
        plus[0, j] += 1e-6
        minus[0, j] -= 1e-6
        difference = (
            harpoon.guidance_loss(plus, target, mask, bounds, model, linear)
            - harpoon.guidance_loss(minus, target, mask, bounds, model, linear)
        ) / 2e-6
        assert gradient[0, j].item() == pytest.approx(difference.item(), abs=1e-7)
    assert not gradient[:, model.n_num :].any()
