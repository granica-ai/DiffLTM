import numpy as np
import pandas as pd
import pytest
import torch

from diffltm.data.core import Schema
from diffltm.data.preprocessing import Preprocessor
from diffltm.generation.conditioning import (
    Condition,
    channels,
    raw_to_standard,
    standard_to_raw,
    training_channels,
    transform_num_ranges,
)
from diffltm.models.model import DiffLTM


@pytest.mark.parametrize("numerical", ["standardize", "quantile"])
def test_snapping_matches_integer_decoder_and_continuous_bounds(numerical, model_config):
    frame = pd.DataFrame({"age": np.repeat(np.arange(6), 20), "value": np.linspace(0, 10, 120)})
    schema = Schema(["age", "value"], [], "value", "regression", ["age"])
    p = Preprocessor(schema, numerical).fit(frame)
    endpoints = transform_num_ranges(p, {"age": (1, 3), "value": (2, 8)})
    model = DiffLTM(p, model_config)
    u, _ = channels(model, Condition(ranges={"age": (1, 3)}), 1)
    assert u[0, model.dim] == pytest.approx(endpoints["age"][0], abs=1e-6)
    grid = np.column_stack([np.linspace(-8, 8, 10001)] * 2).astype("float32")
    raw = p.inverse_numeric(grid)
    for j, (col, bounds) in enumerate([("age", (1, 3)), ("value", (2, 8))]):
        lower, upper = endpoints[col]
        np.testing.assert_array_equal(
            (grid[:, j] >= lower) & (grid[:, j] <= upper),
            (raw[:, j] >= bounds[0]) & (raw[:, j] <= bounds[1]),
        )
    with pytest.raises(ValueError, match="No value"):
        transform_num_ranges(p, {"age": (1.2, 1.4)})
    if numerical == "quantile":
        lo, hi = transform_num_ranges(p, {"age": (0, 5)})["age"]
        assert lo == -np.inf and hi == np.inf
        u, _ = channels(model, Condition(ranges={"age": (0, 5)}), 1)
        assert np.isfinite(u.numpy()).all()
        with pytest.raises(ValueError, match="No value"):
            transform_num_ranges(p, {"age": (6, 8)})


def test_linear_conversion_and_dropout(preprocessor, small_table, model_config):
    model = DiffLTM(preprocessor, dict(model_config, condition_dropout=1.0))
    raw = small_table.frame[["x", "y"]].to_numpy()
    weights = {"x": 2.0, "y": -3.0}
    bounds = (-1.0, 4.0)
    w, b = raw_to_standard(weights, bounds, preprocessor)
    a, original = standard_to_raw(w, b, preprocessor.mean, preprocessor.scale)
    assert np.array_equal(
        ((raw @ np.array([2.0, -3.0])) >= -1) & ((raw @ np.array([2.0, -3.0])) <= 4),
        ((raw @ a) >= original[0]) & ((raw @ a) <= original[1]),
    )
    batch = torch.from_numpy(preprocessor.transform(small_table.frame.iloc[:20]))
    u, m = training_channels(model, model.encode(batch))
    assert not u.any() and not m.any()
    u, m = channels(model, Condition(weights=weights, bounds=bounds), 2)
    assert m[:, -1].eq(1).all()
