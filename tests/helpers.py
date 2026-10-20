"""Small shared helpers for training and RNG checks."""

import random

import numpy as np
import torch


def config(tmp_path, model_config):
    return dict(
        seed=5,
        device="cpu",
        output_dir=str(tmp_path / "run"),
        cache=str(tmp_path / "data"),
        dataset=dict(name="fixture", numerical="standardize"),
        model=model_config,
        training=dict(
            epochs=3,
            batch_size=200,
            learning_rate=0.001,
            weight_decay=0.0001,
            ema_decay=0.9,
            val_every=1,
            grad_clip=1.0,
            amp=False,
            periodic_every=0,
            periodic_mle=False,
            save_every=1,
        ),
    )


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def assert_rng_equal(before):
    after = rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    for a, b in zip(before[3], after[3]):
        assert torch.equal(a, b)
