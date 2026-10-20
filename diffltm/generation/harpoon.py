"""Appendix D.4 Harpoon adaptation with a normalized, continuous linear extension."""

import math

import torch

from ..models.model import sigma
from .conditioning import equality_channels, raw_to_standard, transform_num_ranges
from .sampling import RECOVERY_POLICY, _sample_rows

HARPOON_RECOVERY_POLICY = dict(
    RECOVERY_POLICY,
    name="harpoon-nonfinite-row-redraw-v4",
    max_discard_fraction=0.10,
    max_retry_rounds=None,
)

GUIDANCE_POLICY = dict(
    name="harpoon-d4-linear-v2",
    equality_loss="l1-sum-full-onehot",
    range_loss="l2-lower-plus-l2-upper",
    linear_coordinates="unit-direction-standardized-raw",
    linear_intervals="concatenate-with-numerical-ranges",
    linear_inverse="quantile-unrounded-lookup;standardize-exact-affine",
    batch_reduction="sum",
    update="euler-minus-eta-denoiser-jacobian-gradient",
    equality_pinning=False,
    projection=False,
)


def guidance_policy(model):
    if model.config["representation"] == "learnt":
        return dict(
            GUIDANCE_POLICY,
            name="harpoon-learnt-ce-v1",
            equality_loss="l1-sum-full-normalized-learnt-embedding",
            categorical_decoder="last-step-logits-argmax-mask-unknown",
        )
    return dict(GUIDANCE_POLICY)


@torch.no_grad()
def prepare(model, condition, eta):
    if condition is not None and condition.predicate:
        raise ValueError("Harpoon does not support arbitrary numerical predicates")
    if (model.config["representation"], model.config["objective"]) not in {
        ("onehot", "representation_mse"),
        ("learnt", "logits_ce"),
    } or model.config["conditional"]:
        raise ValueError("Harpoon requires an unconditional onehot_repr or learnt_ce checkpoint")
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("Harpoon eta must be finite and nonnegative")
    if condition is None or not (condition.equality or condition.ranges or condition.weights):
        raise ValueError("Harpoon requires a nonempty equality/range/linear predicate")
    target, mask = equality_channels(model, condition)
    bounds = target.new_full((model.n_num, 2), float("nan"))
    for col, endpoints in transform_num_ranges(model.preprocessor, condition.ranges).items():
        bounds[model.preprocessor.schema.numerical.index(col)] = bounds.new_tensor(endpoints)
    linear = None
    if condition.weights:
        p = model.preprocessor
        weights, interval = raw_to_standard(condition.weights, condition.bounds, p)
        if p.numerical == "standardize":
            # Fold the exact affine inverse into the direction/interval once,
            # including any fitted-scaler vs raw-statistics rounding differences.
            interval = interval - weights @ ((p.transformer.mean_ - p.mean) / p.scale)
            weights = weights * (p.transformer.scale_ / p.scale)
        linear = target.new_tensor(weights), target.new_tensor(interval)
    return target, mask, bounds, linear


def guidance_loss(predicted, target, mask, bounds, model=None, linear=None):
    """Per-row L1 equalities plus separate unsquared lower/upper L2 violations."""
    loss = ((predicted - target) * mask).abs().sum(1)
    num = predicted[:, : model.n_num] if linear is not None else predicted[:, : len(bounds)]
    if linear is not None:
        weights, interval = linear
        raw = num if model.preprocessor.numerical == "standardize" else model.raw_standardized(num)
        projection = (raw * weights).sum(1, keepdim=True)
        num = torch.cat([num, projection], 1)
        bounds = torch.cat([bounds, interval[None]], 0)
    # Select BEFORE arithmetic: inactive/one-sided bounds must not poison gradients.
    lower, upper = torch.isfinite(bounds[:, 0]), torch.isfinite(bounds[:, 1])
    return (
        loss
        + torch.linalg.vector_norm((bounds[lower, 0] - num[:, lower]).relu(), dim=1)
        + torch.linalg.vector_norm((num[:, upper] - bounds[upper, 1]).relu(), dim=1)
    )


@torch.no_grad()
def _step(model, x, t, t_next, target, mask, bounds, eta, linear=None):
    if eta:
        with torch.enable_grad():
            state = x.detach().requires_grad_(True)
            predicted, logits = model.predict(state, t.expand(len(x)))
            loss = guidance_loss(predicted, target, mask, bounds, model, linear)
            (gradient,) = torch.autograd.grad(loss.sum(), state, allow_unused=True)
        predicted = predicted.detach()
    else:
        predicted, logits = model.predict(x, t.expand(len(x)))
    result = predicted + (sigma(t_next) / sigma(t)) * (x - predicted)
    if eta and gradient is not None:
        result = result - eta * gradient
    return result, logits.detach() if logits is not None else None


@torch.no_grad()
def sample(model, n, seed, condition, eta=0.2, batch_size=4096, steps=100):
    if steps < 1:
        raise ValueError("Positive step count required")
    target, mask, bounds, linear = prepare(model, condition, eta)
    device = next(model.parameters()).device

    def draw(b, rng):
        x = torch.randn(b, model.dim, device=device, generator=rng) * 80
        times = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            x, logits = _step(model, x, times[i], times[i + 1], target, mask, bounds, eta, linear)
        # Native decoding: CE keeps the last pre-correction logits; Repr uses x.
        # No terminal denoising, equality pinning or projection for either recipe.
        return x, logits

    frame, diagnostics = _sample_rows(
        model,
        n,
        seed,
        draw,
        condition,
        batch_size=batch_size,
        recovery_policy=HARPOON_RECOVERY_POLICY,
    )
    diagnostics.update(
        harpoon_eta=eta,
        guidance_policy=guidance_policy(model),
        adherence=diagnostics["post_pin_adherence"],
    )
    return frame, diagnostics
