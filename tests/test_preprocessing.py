import numpy as np
import pytest
import torch

from diffltm.data.core import Schema
from diffltm.data.preprocessing import Preprocessor
from diffltm.generation.smc import inverse_numeric
from diffltm.models.model import DiffLTM


def test_categorical_round_trip_unknown(preprocessor, small_table):
    frame = small_table.frame.iloc[:10]
    encoded = preprocessor.transform(frame)
    decoded = preprocessor.inverse(encoded[:, :2], encoded[:, 2:])
    for c in preprocessor.schema.categorical:
        assert decoded[c].tolist() == frame[c].tolist()
    assert "__NA__" in preprocessor.maps[0] and preprocessor.maps[0]["__NA__"] != 0
    unseen = frame.copy()
    unseen["a"] = "absent"
    assert (preprocessor.transform(unseen)[:, 2] == 0).all()
    with pytest.raises(ValueError):
        preprocessor.inverse(encoded[:, :2], np.zeros_like(encoded[:, 2:]))


def test_inverse_lookup_exact_decode(preprocessor, small_table, model_config):
    model = DiffLTM(preprocessor, model_config)
    values = torch.from_numpy(preprocessor.transform(small_table.frame))[:, :2]
    expected = preprocessor.transformer.inverse_transform(values.numpy())
    np.testing.assert_allclose(
        inverse_numeric(model, values).numpy(), expected, rtol=1e-6, atol=1e-6
    )


def test_exact_quantile_inverse_rounding(small_table, model_config):
    frame = small_table.frame.copy()
    frame["x"] = np.rint(frame.x * 10)
    schema = Schema(["x", "y"], ["a", "b"], "y", "regression", ["x"])
    p = Preprocessor(schema).fit(frame)
    model = DiffLTM(p, model_config)
    z = torch.randn(31, 2) * 8
    expected = p.transformer.inverse_transform(z.numpy().astype(float))
    expected[:, 0] = np.rint(expected[:, 0])
    np.testing.assert_allclose(inverse_numeric(model, z).numpy(), expected, rtol=1e-7, atol=1e-7)
