"""100-step Euler generation, CFG and bounded nonfinite-row recovery."""

import json
import math

import pandas as pd
import torch

from ..artifacts import request_seed
from ..models.model import sigma
from .conditioning import channels

RECOVERY_POLICY = dict(
    name="nonfinite-row-redraw-v1",
    max_retry_rounds=3,
    max_discard_fraction=0.001,
    min_discard_budget=10,
)


class NonfiniteGenerationError(ValueError):
    def __init__(self, reason, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"{reason}: {json.dumps(diagnostics, sort_keys=True)}")


@torch.no_grad()
def _sample_rows(
    model, n, seed, draw, condition=None, pin=False, batch_size=4096, recovery_policy=None
):
    """Shared independent-row batching, decoding and bounded nonfinite recovery."""
    if n < 1 or batch_size < 1:
        raise ValueError("Positive sample and batch counts required")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model parameter; row recovery is not applicable")
    model.eval()
    policy = RECOVERY_POLICY if recovery_policy is None else recovery_policy
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    # Stopping limits must not change the replacement noise stream.
    retry_seed = request_seed(seed, RECOVERY_POLICY, "nonfinite-retry")
    retry_generator = torch.Generator(device=device).manual_seed(retry_seed)
    recovery = dict(
        policy=dict(policy),
        retry_seed=retry_seed,
        discard_budget=max(
            policy["min_discard_budget"],
            math.ceil(policy["max_discard_fraction"] * n),
        ),
        attempted_rows=0,
        discarded_rows=0,
        replacement_rows=0,
        retry_rounds=0,
        max_retry_rounds_used=0,
    )

    def invalid(x, logits):
        finite = torch.isfinite(x).all(1)
        if logits is not None:
            finite &= torch.isfinite(logits).all(1)
        return (~finite).nonzero(as_tuple=True)[0]

    frames = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x, logits = draw(b, generator)
        recovery["attempted_rows"] += b
        bad = invalid(x, logits)
        rounds = 0
        while len(bad):
            recovery["discarded_rows"] += len(bad)
            if rounds == 0:
                recovery["replacement_rows"] += len(bad)
            if recovery["discarded_rows"] > recovery["discard_budget"]:
                raise NonfiniteGenerationError("Nonfinite discard budget exceeded", recovery)
            if policy["max_retry_rounds"] is not None and rounds == policy["max_retry_rounds"]:
                raise NonfiniteGenerationError("Nonfinite retry rounds exhausted", recovery)
            rounds += 1
            recovery["retry_rounds"] += 1
            recovery["max_retry_rounds_used"] = max(recovery["max_retry_rounds_used"], rounds)
            replacement, replacement_logits = draw(len(bad), retry_generator)
            recovery["attempted_rows"] += len(bad)
            x[bad] = replacement
            if logits is not None:
                logits[bad] = replacement_logits
            bad = bad[invalid(replacement, replacement_logits)]
        # Decode once in the original batch/order, after checking endpoints AND logits.
        frames.append(model.decode(x, logits))
    frame = pd.concat(frames, ignore_index=True)
    pre = float(condition.mask(frame).mean()) if condition is not None else None
    if pin and condition is not None:
        frame = condition.pin(frame)
    return frame, dict(
        requested=n,
        returned=len(frame),
        pre_pin_adherence=pre,
        post_pin_adherence=float(condition.mask(frame).mean()) if condition is not None else None,
        equality_pinning=bool(pin and condition is not None),
        generation_seed=seed,
        recovery=recovery,
    )


@torch.no_grad()
def euler(model, n, seed, condition=None, guidance=1.0, pin=True, batch_size=4096, steps=100):
    if steps < 1:
        raise ValueError("Positive step count required")
    device = next(model.parameters()).device
    encoded = channels(model, condition, 1) if condition is not None else None

    def draw(b, rng):
        x = torch.randn(b, model.dim, device=device, generator=rng) * 80
        cond = tuple(v.expand(b, -1) for v in encoded) if encoded is not None else None
        times = torch.linspace(1, 0, steps + 1, device=device)
        for i in range(steps):
            t = times[i].expand(b)
            predicted, guided_logits = model.predict(x, t, cond)
            if cond is not None and guidance != 1:
                unconditional, null_logits = model.predict(x, t)
                predicted = unconditional + guidance * (predicted - unconditional)
                if guided_logits is not None:
                    guided_logits = null_logits + guidance * (guided_logits - null_logits)
                    cat = model.logits_to_representation(guided_logits)
                    predicted = torch.cat([predicted[:, : model.n_num], cat], 1)
            s, sn = sigma(times[i]), sigma(times[i + 1])
            x = predicted + (sn / s) * (x - predicted)
        # Euler decoding uses the last integration-step logits (no extra t=0 denoise).
        return x, guided_logits

    return _sample_rows(model, n, seed, draw, condition, pin, batch_size)
