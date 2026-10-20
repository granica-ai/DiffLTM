import numpy as np
import pandas as pd
import pytest
import torch

from diffltm.data.core import Schema, make_table
from diffltm.data.preprocessing import Preprocessor


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.fixture
def small_table(tmp_path):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=120),
            "y": rng.normal(size=120),
            "a": np.tile(["one", "two", "__NA__"], 40),
            "b": np.tile(["yes", "no"], 60),
        }
    )
    table = make_table(
        frame, Schema(["x", "y"], ["a", "b"], "y", "regression"), "fixture", "fixture-v1"
    )

    from diffltm.artifacts import file_identity
    from diffltm.data.core import dataset_identity

    path = tmp_path / "fixture.parquet"
    table.frame.to_parquet(path)
    table.manifest["files"] = {"table.parquet": file_identity(path)}
    table.manifest["identity"] = dataset_identity(table.manifest)
    return table


@pytest.fixture
def preprocessor(small_table):
    return Preprocessor(small_table.schema, "standardize").fit(small_table.frame.iloc[:80])


@pytest.fixture
def model_config():
    return dict(
        representation="learnt",
        objective="logits_ce",
        embedding_dim=4,
        backbone="mlp",
        width=16,
        depth=1,
        conditional=True,
        ranges=True,
        linear=True,
        condition_dropout=0.15,
    )


@pytest.fixture
def harpoon_config(model_config):
    return dict(
        model_config,
        representation="onehot",
        objective="representation_mse",
        conditional=False,
        ranges=False,
        linear=False,
    )
