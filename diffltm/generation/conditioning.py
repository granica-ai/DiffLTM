"""One raw-space predicate for slicing, sampling, and final adherence."""

import json
import math
import runpy
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_identity


@cache
def _resolve_predicate(reference):
    filename, name, digest = reference.rsplit(":", 2)
    path = Path(__file__).parents[2] / filename
    if file_identity(path) != digest:
        raise ValueError("Numerical predicate source mismatch")
    function = runpy.run_path(path).get(name)
    if not callable(function):
        raise ValueError(f"Unknown numerical predicate {name!r}")
    return function


@dataclass(frozen=True)
class Condition:
    equality: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    bounds: tuple | None = None
    predicate: str | None = None

    def __post_init__(self):
        if bool(self.weights) != (self.bounds is not None):
            raise ValueError("Linear weights and bounds are required together")
        for lo, hi in list(self.ranges.values()) + ([self.bounds] if self.bounds else []):
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("Invalid interval")
        if self.weights and (
            not np.isfinite(list(self.weights.values())).all() or not any(self.weights.values())
        ):
            raise ValueError("Invalid linear direction")

    def validate(self, schema):
        if not set(self.equality) <= set(schema.columns):
            raise ValueError("Unknown equality column")
        if not (set(self.ranges) | set(self.weights)) <= set(schema.numerical):
            raise ValueError("Range/linear constraints require numerical columns")
        self.predicate_function()

    def predicate_function(self):
        return _resolve_predicate(self.predicate) if self.predicate else None

    def mask(self, frame):
        result = np.ones(len(frame), dtype=bool)
        for c, value in self.equality.items():
            result &= frame[c].eq(value).to_numpy()
        for c, (lo, hi) in self.ranges.items():
            result &= frame[c].between(lo, hi).to_numpy()
        if self.weights:
            s = frame[list(self.weights)].to_numpy(float) @ np.array(
                list(self.weights.values()), dtype=float
            )
            result &= (s >= self.bounds[0]) & (s <= self.bounds[1])
        if self.predicate:
            hit = np.asarray(self.predicate_function()(frame), dtype=bool)
            if hit.shape != result.shape:
                raise ValueError("Numerical predicate must return one Boolean per row")
            result &= hit
        return result

    def dumps(self):
        values = asdict(self)
        if values["predicate"] is None:
            del values["predicate"]
        return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def loads(cls, value):
        return cls(**json.loads(value))

    def pin(self, frame):
        frame = frame.copy()
        for c, value in self.equality.items():
            frame[c] = value
        return frame


def raw_to_standard(weights, bounds, preprocessor):
    a = np.array([weights.get(c, 0.0) for c in preprocessor.schema.numerical])
    shift = float(a @ preprocessor.mean)
    w = a * preprocessor.scale
    norm = np.linalg.norm(w)
    if not norm > 0:
        raise ValueError("Null direction")
    return w / norm, (np.asarray(bounds) - shift) / norm


def standard_to_raw(weights, bounds, mean, scale):
    a = np.asarray(weights) / scale
    return a, np.asarray(bounds) + a @ mean


def _decode_column_at(preprocessor, j, values):
    # Generation endpoints are float32; include that resolution in the search.
    coordinates = np.zeros((len(values), len(preprocessor.schema.numerical)), dtype="float32")
    coordinates[:, j] = values
    return preprocessor.inverse_numeric(coordinates)[:, j]


def _bisect_edge(preprocessor, j, target, want_ge, start):
    """Bracket and bisect the transition in the actual monotone decoder."""
    test = (lambda v: v >= target) if want_ge else (lambda v: v > target)
    lo = hi = start
    span = 1.0
    for _ in range(40):
        v_lo, v_hi = _decode_column_at(preprocessor, j, [lo, hi])
        if not test(v_lo) and test(v_hi):
            break
        if test(v_lo):
            lo -= span
        if not test(v_hi):
            hi += span
        span *= 2
    else:
        return -math.inf if test(_decode_column_at(preprocessor, j, [lo])[0]) else math.inf
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid <= lo or mid >= hi:
            break
        if test(_decode_column_at(preprocessor, j, [mid])[0]):
            hi = mid
        else:
            lo = mid
    return hi if want_ge else lo


def transform_num_ranges(preprocessor, ranges):
    """Decoded-support bounds for SMC/Harpoon; CFG clamps to its training span."""
    result = {}
    for col, (lo, hi) in ranges.items():
        j = preprocessor.schema.numerical.index(col)
        raw = np.stack([preprocessor.mean, preprocessor.mean])
        raw[:, j] = [lo, hi]
        starts = preprocessor.transformer.transform(raw)[:, j]
        lower = _bisect_edge(preprocessor, j, lo, True, float(starts[0]))
        upper = _bisect_edge(preprocessor, j, hi, False, float(starts[1]))
        if lower == math.inf or upper == -math.inf or lower > upper:
            raise ValueError(f"No value of {col!r} decodes into [{lo}, {hi}]")
        result[col] = (lower, upper)
    return result


def equality_channels(model, condition, n=1):
    """Encode full equality blocks, independently of learned CFG channels."""
    condition.validate(model.preprocessor.schema)
    p = model.preprocessor
    device = next(model.parameters()).device
    u = torch.zeros(n, model.dim, device=device)
    mask = torch.zeros_like(u)
    m = model.n_num
    baseline = p.mean.copy()
    for c, value in condition.equality.items():
        if c in p.schema.numerical:
            i = p.schema.numerical.index(c)
            v = baseline.copy()
            v[i] = value
            u[:, i] = float(p.transformer.transform(v[None])[0, i])
            mask[:, i] = 1
        else:
            j = p.schema.categorical.index(c)
            if value not in p.maps[j]:
                raise ValueError(f"Unseen condition {c}={value!r}")
            start = m + sum(model.widths[:j])
            width = model.widths[j]
            u[:, start : start + width] = model.tables()[j][p.maps[j][value]]
            mask[:, start : start + width] = 1
    return u, mask


def channels(model, condition, n):
    """Convert raw equalities, ranges and linear bounds to learned CFG channels."""
    if condition.predicate:
        raise ValueError("CFG does not support arbitrary numerical predicates")
    eq, eq_mask = equality_channels(model, condition, n)
    p = model.preprocessor
    device = eq.device
    u = eq.new_zeros(n, model.condition_dim)
    mask = torch.zeros_like(u)
    d, m = model.dim, model.n_num
    u[:, :d], mask[:, :d] = eq, eq_mask
    for c, endpoints in transform_num_ranges(p, condition.ranges).items():
        if not model.config["ranges"]:
            raise ValueError("Checkpoint was not trained with range channels")
        i = p.schema.numerical.index(c)
        lo, hi = np.clip(endpoints, p.model_min[i], p.model_max[i])
        if lo >= hi:
            raise ValueError(f"Range for {c!r} is empty after clamping to the training span")
        for offset, value in [(d, lo), (d + m, hi)]:
            u[:, offset + i] = float(value)
            mask[:, offset + i] = 1
    if condition.weights:
        if not model.config["linear"]:
            raise ValueError("Checkpoint was not trained with linear channels")
        w, b = raw_to_standard(condition.weights, condition.bounds, p)
        offset = d + (2 * m if model.config["ranges"] else 0)
        u[:, offset : offset + m] = torch.as_tensor(w, device=device, dtype=u.dtype)
        u[:, offset + m : offset + m + 2] = torch.as_tensor(b, device=device, dtype=u.dtype)
        mask[:, offset:] = 1
    return u, mask


def training_channels(model, clean):
    b, d = clean.shape
    m = model.n_num
    device = clean.device
    obs = torch.rand(b, m + len(model.widths), device=device) < torch.rand(b, 1, device=device)
    keep = torch.rand(b, 1, device=device) >= model.config["condition_dropout"]
    obs &= keep
    num_mask = obs[:, :m].clone()
    take_range = (
        num_mask & (torch.rand(b, m, device=device) < 0.5)
        if model.config["ranges"]
        else torch.zeros_like(num_mask)
    )
    num_mask &= ~take_range
    parts, masks = [], []
    if model.config["linear"]:
        k = torch.randint(1, m + 1, (b, 1), device=device)
        support = torch.rand(b, m, device=device).argsort(1).argsort(1) < k
        w = torch.randn(b, m, device=device) * support
        norm = w.norm(dim=1, keepdim=True)
        w = torch.where(norm <= 1e-12, support.to(w.dtype), w)
        w = w / w.norm(dim=1, keepdim=True)
        raw = model.raw_standardized(clean[:, :m])
        lo_box = model.raw_lo
        hi_box = model.raw_hi
        ends = torch.stack([w * lo_box, w * hi_box])
        smin, smax = ends.amin(0).sum(1, keepdim=True), ends.amax(0).sum(1, keepdim=True)
        s = (w * raw).sum(1, keepdim=True)
        width = torch.exp(math.log(0.02) * (1 - torch.rand(b, 1, device=device))) * (
            smax - smin
        ).clamp_min(1e-12)
        lo = s - width * torch.rand(b, 1, device=device)
        hi = lo + width
        lo = torch.minimum(torch.maximum(lo, smin), s)
        hi = torch.maximum(torch.minimum(hi, smax), s)
        gate = ((torch.rand(b, 1, device=device) < 0.5) & keep).float()
        w, bounds = w * gate, torch.cat([lo, hi], 1) * gate
        active = (w != 0) & (gate > 0)
        num_mask &= ~active
        take_range &= ~active
        parts = [w, bounds, torch.zeros(b, 3, device=device)]
        masks = [gate.expand(-1, m), gate.expand(-1, 2), gate.expand(-1, 3)]
    cat_mask = torch.repeat_interleave(obs[:, m:], torch.tensor(model.widths, device=device), dim=1)
    eq_mask = torch.cat([num_mask, cat_mask], 1).float()
    range_parts, range_masks = [], []
    if model.config["ranges"]:
        values = clean[:, :m]
        width = torch.exp(math.log(0.02) * (1 - torch.rand_like(values))) * (
            model.model_hi - model.model_lo
        )
        lo = values - width * torch.rand_like(values)
        hi = lo + width
        lo = torch.minimum(torch.maximum(lo, model.model_lo), values)
        hi = torch.maximum(torch.minimum(hi, model.model_hi), values)
        range_parts = [lo * take_range, hi * take_range]
        range_masks = [take_range.float(), take_range.float()]
    return torch.cat([clean * eq_mask] + range_parts + parts, 1), torch.cat(
        [eq_mask] + range_masks + masks, 1
    )
