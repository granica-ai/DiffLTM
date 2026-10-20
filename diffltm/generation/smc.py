"""Twisted SMC with ancestral VE transitions and an exact decoded terminal gate."""

import math

import pandas as pd
import torch

from ..models.model import sigma
from .conditioning import transform_num_ranges


def allocation(n, replicates=4, multiplier=1, floor=8192):
    if n < 1 or replicates < 1 or multiplier < 1:
        raise ValueError("Positive allocation required")
    per = math.ceil(n / replicates)
    effective = max(multiplier, math.ceil(floor / per))
    return per, per * effective


def resample(log_weights, n, generator):
    if (
        log_weights.ndim != 1
        or n < 1
        or torch.isnan(log_weights).any()
        or torch.isposinf(log_weights).any()
    ):
        raise ValueError("Invalid SMC weights/count")
    support = torch.isfinite(log_weights).nonzero().flatten()
    if not len(support):
        raise RuntimeError("SMC has no positive-weight support")
    w = log_weights[support].double().softmax(0)
    support, w = support[w > 0], w[w > 0]
    cumulative = w.cumsum(0)
    cumulative /= cumulative[-1].clone()
    offset = torch.rand(1, device=w.device, dtype=w.dtype, generator=generator)
    positions = (torch.arange(n, device=w.device, dtype=w.dtype) + offset) / n
    return support[
        torch.searchsorted(cumulative, positions, right=True).clamp(max=len(support) - 1)
    ]


def log_interval(lo, hi, mean, sd):
    a, b = (lo - mean) / sd, (hi - mean) / sd
    # Use survival probabilities in the positive tail to avoid cancellation.
    upper = torch.where(a > 0, torch.special.log_ndtr(-a), torch.special.log_ndtr(b))
    lower = torch.where(a > 0, torch.special.log_ndtr(-b), torch.special.log_ndtr(a))
    return upper + torch.log(-torch.expm1((lower - upper).clamp(max=-1e-15)))


def inverse_numeric(model, values):
    """Exact fitted inverse in torch, including quantile endpoint and integer rules."""
    p = model.preprocessor
    if p.numerical == "standardize":
        raw = values.double() * torch.as_tensor(
            p.transformer.scale_, device=values.device
        ) + torch.as_tensor(p.transformer.mean_, device=values.device)
    else:
        u = torch.special.ndtr(values.double())
        refs = torch.as_tensor(p.transformer.references_, device=values.device)
        qs = torch.as_tensor(p.transformer.quantiles_, device=values.device)
        upper = torch.searchsorted(refs, u.contiguous(), right=True).clamp(1, len(refs) - 1)
        lower = upper - 1
        f = ((u - refs[lower]) / (refs[upper] - refs[lower]).clamp_min(1e-300)).clamp(0, 1)
        cols = torch.arange(model.n_num, device=values.device)
        raw = qs[lower, cols] + f * (qs[upper, cols] - qs[lower, cols])
        raw = torch.where(u - 1e-7 < refs[0], qs[0], raw)
        raw = torch.where(u + 1e-7 > refs[-1], qs[-1], raw)
    for c in p.schema.integer:
        j = p.schema.numerical.index(c)
        raw[..., j] = raw[..., j].round()
    return raw


@torch.no_grad()
def sample(model, n, seed, condition, pin=True, steps=100, replicates=4, floor=8192, multiplier=1):
    if model.config["objective"] == "representation_mse":
        raise ValueError("SMC requires a categorical logit head")
    p = model.preprocessor
    condition.validate(p.schema)
    if any(c in p.schema.numerical for c in condition.equality):
        raise ValueError("SMC equality conditions must be categorical")
    if not (condition.equality or condition.ranges or condition.weights or condition.predicate):
        raise ValueError("SMC requires a predicate")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    per, particles = allocation(n, replicates, multiplier, floor)
    frames, diagnostics = [], []
    ranges = transform_num_ranges(p, condition.ranges)
    predicate = condition.predicate_function()
    for replicate in range(replicates):
        x = torch.randn(particles, model.dim, device=device, generator=rng) * 80
        base_noise = torch.randn(
            128, model.n_num, dtype=torch.float64, device=device, generator=rng
        )
        noise = torch.cat([base_noise, -base_noise])
        weights = torch.zeros(particles, device=device, dtype=torch.float64)
        previous = None
        times = torch.linspace(1, 0, steps + 1, device=device)
        min_ess, resamples = float(particles), 0
        for step in range(steps):
            prediction, logits = model.predict(x, times[step].expand(particles))
            s, sn = sigma(times[step]), sigma(times[step + 1])
            potential = torch.zeros_like(weights)
            split_logits = logits.split(model.cards, 1)
            for c, value in condition.equality.items():
                j = p.schema.categorical.index(c)
                if value not in p.maps[j]:
                    raise ValueError(f"Unseen equality {c}={value}")
                potential += split_logits[j].log_softmax(1)[:, p.maps[j][value]].double()
            for c, (lo, hi) in ranges.items():
                j = p.schema.numerical.index(c)
                potential += log_interval(lo, hi, prediction[:, j].double(), s.double())
            if condition.weights:
                w = torch.tensor(
                    [condition.weights.get(c, 0) for c in p.schema.numerical],
                    dtype=torch.float64,
                    device=device,
                )
                hits = torch.zeros_like(weights)
                for first in range(0, 256, 64):
                    draws = (
                        prediction[:, None, : model.n_num].double()
                        + s.double() * noise[None, first : first + 64]
                    )
                    projection = inverse_numeric(model, draws) @ w
                    hits += (
                        (projection >= condition.bounds[0]) & (projection <= condition.bounds[1])
                    ).sum(1)
                potential += ((hits + 0.5) / 257).log()
            if predicate:
                hits = torch.zeros_like(weights)
                for first in range(0, 256, 64):
                    draws = (
                        prediction[:, None, : model.n_num].double()
                        + s.double() * noise[None, first : first + 64]
                    )
                    raw = inverse_numeric(model, draws)
                    values = {c: raw[..., j] for j, c in enumerate(p.schema.numerical)}
                    hits += torch.as_tensor(predicate(values), device=device).sum(1)
                potential += ((hits + 0.5) / 257).log()
            weights += potential if previous is None else potential - previous
            ess = float(1 / weights.softmax(0).square().sum())
            min_ess = min(ess, min_ess)
            if ess < 0.95 * particles and step < steps - 1:
                ancestors = resample(weights, particles, rng)
                x, prediction, potential = x[ancestors], prediction[ancestors], potential[ancestors]
                weights.zero_()
                resamples += 1
            previous = potential
            up = (sn.square() * (1 - sn.square() / s.square())).clamp_min(0).sqrt()
            coefficient = sn.square() / s.square()
            x = (
                prediction
                + coefficient * (x - prediction)
                + up * torch.randn(x.shape, device=device, generator=rng)
            )
        _, logits = model.predict(x, times[-1].expand(particles))
        frame = model.decode(x, logits)
        hit = torch.as_tensor(condition.mask(frame), device=device)
        weights -= previous
        weights[~hit] = -torch.inf
        ancestors = resample(weights, per, rng)
        frames.append(frame.iloc[ancestors.cpu().numpy()])
        diagnostics.append(
            dict(
                replicate=replicate,
                particles=particles,
                terminal_hits=int(hit.sum()),
                min_ess=min_ess,
                resamples=resamples,
            )
        )
    frame = pd.concat(frames, ignore_index=True).iloc[:n].copy()
    pre = float(condition.mask(frame).mean())
    if pin:
        frame = condition.pin(frame)
    return frame, dict(
        requested=n,
        returned=len(frame),
        pre_pin_adherence=pre,
        post_pin_adherence=float(condition.mask(frame).mean()),
        equality_pinning=pin,
        replicates=diagnostics,
        generation_seed=seed,
    )
