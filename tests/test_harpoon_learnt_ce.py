"""Learned-coordinate guidance with native CE decoding and separate reporting."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from diffltm.data.preprocessing import Preprocessor
from diffltm.generation import harpoon, sampling
from diffltm.generation.conditioning import Condition
from diffltm.models.model import DiffLTM, sigma
from diffltm.plotting import aggregate_conditions, plot_rarity


@pytest.fixture
def ce_model(preprocessor, harpoon_config):
    return DiffLTM(
        preprocessor, dict(harpoon_config, representation="learnt", objective="logits_ce")
    )


DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
    ),
]


PREDICATES = [
    Condition(equality={"a": "two"}),
    Condition(ranges={"x": (-0.1, 0.1)}),
    Condition(weights={"x": 1, "y": -2}, bounds=(-0.1, 0.1)),
    Condition(
        equality={"a": "two", "y": 0.1},
        ranges={"x": (-0.1, 0.1)},
        weights={"x": 1, "y": -2},
        bounds=(-0.1, 0.1),
    ),
]


def test_normalized_embedding_targets_and_fixed_full_block_masks(ce_model):
    model = ce_model
    target, mask, _, _ = harpoon.prepare(model, PREDICATES[-1], 0.2)
    start, width = model.n_num, model.widths[0]
    expected = model.tables()[0][model.preprocessor.maps[0]["two"]]
    torch.testing.assert_close(target[0, start : start + width], expected)
    assert expected.norm().item() == pytest.approx(math.sqrt(width))
    assert mask[0, start : start + width].eq(1).all()
    assert mask.sum() == width + 1 and not mask[0, start + width :].any()
    assert not target.requires_grad and not mask.requires_grad


@pytest.mark.parametrize("batch", [1, 5])
def test_ce_gradient_through_softmax_embedding_mixture_and_cross_blocks(
    ce_model, monkeypatch, batch
):
    model = ce_model.double()
    generator = torch.Generator().manual_seed(17)
    matrix = torch.randn(model.dim, sum(model.cards), generator=generator, dtype=torch.float64) / 10

    class Denoiser(torch.nn.Module):
        def forward(self, x, t, condition=None):
            return x[:, : model.n_num] * 0.3, x @ matrix

    monkeypatch.setattr(model, "denoiser", Denoiser())
    target, mask, bounds, linear = harpoon.prepare(model, PREDICATES[0], 0.2)
    state = torch.randn(1, model.dim, generator=generator, dtype=torch.float64).repeat(batch, 1)

    def loss(x):
        return harpoon.guidance_loss(model.predict(x, torch.ones(len(x)))[0], target, mask, bounds)

    h = 1e-6
    expected = torch.empty_like(state)
    for col in range(model.dim):
        delta = torch.zeros_like(state)
        delta[:, col] = h
        expected[:, col] = (loss(state + delta) - loss(state - delta)) / (2 * h)
    assert expected[:, : model.n_num].abs().sum() > 0  # Categorical loss affects numeric state.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    saved = [(p.detach().clone(), p.grad.clone()) for p in model.parameters()]
    t, tn = torch.tensor(0.8), torch.tensor(0.4)
    predicted, logits = model.predict(state, t.expand(batch))
    endpoint, actual_logits = harpoon._step(model, state, t, tn, target, mask, bounds, 0.2, linear)
    prior = predicted + (sigma(tn) / sigma(t)) * (state - predicted)
    torch.testing.assert_close(endpoint, prior - 0.2 * expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(actual_logits, logits)
    assert not endpoint.requires_grad and not actual_logits.requires_grad
    for p, (value, gradient) in zip(model.parameters(), saved):
        assert torch.equal(p, value) and torch.equal(p.grad, gradient)


@pytest.mark.parametrize("condition", PREDICATES)
@pytest.mark.parametrize("numerical", ["standardize", "quantile"])
@pytest.mark.parametrize("device", DEVICES)
def test_zero_eta_native_euler_parity_partial_batch_and_no_terminal_call(
    small_table, harpoon_config, monkeypatch, condition, numerical, device
):
    pre = Preprocessor(small_table.schema, numerical).fit(small_table.frame)
    model = DiffLTM(pre, dict(harpoon_config, representation="learnt", objective="logits_ce"))
    model = model.to(device)
    expected, _ = sampling.euler(model, 19, 42, batch_size=8, steps=3)
    original = model.predict
    calls = []

    def predict(x, t):
        calls.append(len(x))
        return original(x, t)

    monkeypatch.setattr(model, "predict", predict)
    actual, meta = harpoon.sample(model, 19, 42, condition, eta=0, batch_size=8, steps=3)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert calls == [8] * 6 + [3] * 3
    assert not meta["equality_pinning"]
    assert meta["adherence"] == float(condition.mask(actual).mean())
    assert meta["guidance_policy"]["categorical_decoder"] == "last-step-logits-argmax-mask-unknown"


@pytest.mark.parametrize("condition", PREDICATES)
def test_native_logits_not_nearest_embedding_and_last_correction_timing(
    ce_model, monkeypatch, condition
):
    model = ce_model
    # Three allowed categories: argmax chooses the first, nearest mean chooses second.
    angles = torch.tensor([0.0, 0.0, 10.0, 180.0]) * math.pi / 180
    with torch.no_grad():
        model.embeddings[0].weight.zero_()
        model.embeddings[0].weight[:, :2] = torch.stack([angles.cos(), angles.sin()], 1)
    first = torch.tensor([1e-20, 0.40, 0.35, 0.25]).log()
    other = torch.tensor([1e-20, 0.8, 0.2]).log()
    fixed = torch.cat([first, other])[None]
    representation = model.logits_to_representation(fixed)
    state = torch.cat([torch.zeros(1, model.n_num), representation], 1)
    assert model.decode(state).a.iloc[0] != model.decode(state, fixed).a.iloc[0]
    seen = []

    def predict(x, t):
        logits = fixed.expand(len(x), -1) + x[:, :1] * 0.0001 * torch.arange(fixed.shape[1])
        predicted = torch.cat(
            [x[:, : model.n_num] * 0.3, model.logits_to_representation(logits)], 1
        )
        seen.append((x.detach().clone(), predicted.detach().clone(), logits.detach().clone()))
        return predicted, logits

    monkeypatch.setattr(model, "predict", predict)
    actual, meta = harpoon.sample(model, 7, 31, condition, eta=0.2, steps=1)
    assert len(seen) == 1
    incoming, predicted, logits = seen[0]
    unguided = predicted + (sigma(torch.tensor(0.0)) / sigma(torch.tensor(1.0))) * (
        incoming - predicted
    )
    expected_cats = model.decode(unguided, logits)
    pd.testing.assert_frame_equal(actual[["a", "b"]], expected_cats[["a", "b"]])
    assert not np.array_equal(actual[["x", "y"]].to_numpy(), expected_cats[["x", "y"]].to_numpy())
    assert meta["recovery"]["discarded_rows"] == 0
    # Reserved unknown logits never become a returned category.
    logits[:, 0] = 100
    assert model.decode(unguided, logits).a.eq(expected_cats.a).all()


@pytest.mark.parametrize("component", ["endpoint", "logits"])
@pytest.mark.parametrize("condition", PREDICATES)
def test_ce_recovery_retains_native_logits_and_valid_rows(
    ce_model, monkeypatch, condition, component
):
    model = ce_model
    baseline, _ = harpoon.sample(model, 19, 42, condition, batch_size=8, steps=3)
    original = harpoon._step
    calls = []

    def step(*args):
        calls.append(len(args[1]))
        endpoint, logits = original(*args)
        assert logits is not None and args[7] == 0.2
        if len(calls) == 3:
            (endpoint if component == "endpoint" else logits)[0, 0] = float("nan")
        return endpoint, logits

    monkeypatch.setattr(harpoon, "_step", step)
    actual, meta = harpoon.sample(model, 19, 42, condition, batch_size=8, steps=3)
    pd.testing.assert_frame_equal(actual.iloc[1:], baseline.iloc[1:], check_exact=True)
    assert calls == [8] * 3 + [1] * 3 + [8] * 3 + [3] * 3
    assert meta["recovery"]["discarded_rows"] == 1 and len(actual) == 19
    assert meta["recovery"]["policy"] == harpoon.HARPOON_RECOVERY_POLICY


@pytest.mark.parametrize("component", ["endpoint", "logits"])
def test_ce_recovery_exhaustion_is_explicit(ce_model, monkeypatch, component):
    original = harpoon._step

    def step(*args):
        endpoint, logits = original(*args)
        (endpoint if component == "endpoint" else logits)[0, 0] = float("nan")
        return endpoint, logits

    monkeypatch.setattr(harpoon, "_step", step)
    with pytest.raises(sampling.NonfiniteGenerationError, match="discard budget exceeded") as exc:
        harpoon.sample(ce_model, 4, 42, PREDICATES[0], steps=1)
    assert exc.value.diagnostics["discarded_rows"] == 11


def comparison_frame():
    return pd.DataFrame(
        [
            dict(
                method="harpoon",
                model=model,
                bucket=">3%",
                slice=slice_id,
                training_seed=seed,
                value=value,
                status="complete",
                omission_reason=None,
                metric="shape",
                dataset="fixture",
                dataset_identity="same",
                task="equality",
            )
            for model, value in [("onehot_representation_mse", 0.2), ("learnt_logits_ce", 0.8)]
            for seed in [500, 501, 502]
            for slice_id in ["a", "b"]
        ]
    )


def test_both_harpoon_variants_plot_and_aggregate_separately(tmp_path):
    frame = comparison_frame()
    before = frame.copy(deep=True)
    conditions, summary = aggregate_conditions(frame)
    assert summary["mean"].tolist() == pytest.approx([0.8, 0.2])
    assert conditions.total_seeds.eq(3).all() and summary.total_conditions.eq(2).all()
    assert summary.method.eq("harpoon").all()
    fig, ax = plot_rarity(frame, dataset="fixture", output=tmp_path / "comparison")
    assert set(ax.get_legend_handles_labels()[1]) == {
        "Harpoon (OH/Repr/MSE)",
        "Harpoon (Learnt/Logits/CE)",
    }
    pd.testing.assert_frame_equal(frame, before)
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.parametrize("error", ["duplicate", "missing_seed", "missing_slice", "missing_model"])
def test_variant_reporting_retains_completeness_checks(error):
    frame = comparison_frame()
    if error == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif error == "missing_seed":
        frame = frame.loc[~(frame.model.eq("learnt_logits_ce") & frame.training_seed.eq(502))]
    elif error == "missing_slice":
        frame = frame.iloc[1:]
    else:
        frame = frame.drop(columns="model")
    with pytest.raises(ValueError):
        aggregate_conditions(frame)
