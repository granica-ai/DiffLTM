"""Evaluation artifacts are fresh; implementation edits do not alter seeds."""

from pathlib import Path

import pandas as pd
import pytest

from diffltm import evaluation
from diffltm.generation.conditioning import Condition
from diffltm.generation.harpoon import sample as harpoon_sample
from diffltm.generation.sampling import RECOVERY_POLICY
from diffltm.models.model import DiffLTM


@pytest.fixture
def session(tmp_path, small_table, preprocessor, harpoon_config, monkeypatch):
    ev = object.__new__(evaluation.Evaluation)
    ev.model = DiffLTM(preprocessor, harpoon_config)
    ev.table, ev.seed = small_table, 500
    ev.guidance, ev.pin, ev.harpoon_eta = 1, True, 0.2
    ev.checkpoint_id, ev.identity = "fixture-checkpoint", {}
    ev.run_dir, ev.output, ev.sample_root = tmp_path, tmp_path, tmp_path / "samples"
    ev._rejection_pool = None
    calls = []

    def sample(model, n, seed, *args):
        calls.append((n, seed))
        frame = pd.DataFrame(dict(x=[0.5] * n, y=[0.5] * n, a=["two"] * n, b=["yes"] * n))
        return frame, dict(
            recovery=dict(
                policy=dict(RECOVERY_POLICY),
                attempted_rows=n,
                discarded_rows=0,
                replacement_rows=0,
                retry_rounds=0,
                max_retry_rounds_used=0,
            )
        )

    monkeypatch.setattr(evaluation, "euler", sample)
    monkeypatch.setattr("diffltm.generation.smc.sample", sample)
    monkeypatch.setattr("diffltm.generation.harpoon.sample", sample)
    return ev, calls


@pytest.mark.parametrize("method", ["unconditional", "cfg", "smc", "harpoon"])
def test_requests_regenerate_without_hashing_code(session, monkeypatch, method):
    ev, calls = session
    condition = None if method == "unconditional" else Condition(equality={"a": "two"})
    frame, before = ev.generate(method, condition, 5)
    original = evaluation.file_identity

    def check_data_only(path):
        assert Path(path).suffix != ".py", (
            "Implementation files must not enter compatibility or seeds"
        )
        return original(path)

    monkeypatch.setattr(evaluation, "file_identity", check_data_only)
    # Existing sample data are output artifacts, not an automatic generation cache.
    (ev.sample_root / before["sample_key"] / "samples.parquet").write_bytes(b"old sample")
    after, manifest = ev.generate(method, condition, 5)
    assert len(calls) == 2
    assert manifest == before
    pd.testing.assert_frame_equal(frame, after)
    assert (
        not {"implementation", "checkpoint", "rejection_implementation"}
        & manifest["request"].keys()
    )


def test_rejection_shares_within_session_and_cleans_up(session):
    ev, calls = session
    ev.generate("rejection", Condition(equality={"a": "two"}), 5)
    pool = ev._rejection_pool
    ev.generate("rejection", Condition(ranges={"x": (0, 1)}), 7)
    assert len(calls) == 1
    root = pool.root
    pool.close()
    assert not root.exists()
    ev._rejection_pool = None
    ev.generate("rejection", Condition(equality={"a": "two"}), 5)
    assert len(calls) == 2
    ev._rejection_pool.close()


@pytest.mark.parametrize("method", ["unconditional", "rejection"])
def test_common_random_numbers_across_runs_but_not_training_seeds(session, method):
    ev, calls = session
    condition = Condition(equality={"a": "two"}) if method == "rejection" else None

    def draw():
        _, manifest = ev.generate(method, condition, 5)
        actual_seed = calls[-1][1]
        if ev._rejection_pool is not None:
            ev._rejection_pool.close()
            ev._rejection_pool = None
        return manifest["generation_seed"], actual_seed

    first = draw()
    ev.model = DiffLTM(
        ev.model.preprocessor,
        dict(ev.model.config, representation="learnt", objective="logits_ce"),
    )
    ev.identity = dict(
        run_id="another-run", model="another-model", dataset_identity="another-table"
    )
    paired = draw()
    assert paired == first
    ev.seed += 1
    independent = draw()
    assert all(a != b for a, b in zip(first, independent))


@pytest.mark.parametrize(
    "representation,objective", [("onehot", "representation_mse"), ("learnt", "logits_ce")]
)
def test_harpoon_limits_preserve_sampling_stream(session, monkeypatch, representation, objective):
    from diffltm.generation.harpoon import HARPOON_RECOVERY_POLICY

    ev, _ = session
    ev.model = DiffLTM(
        ev.model.preprocessor,
        dict(ev.model.config, representation=representation, objective=objective),
    )
    monkeypatch.setattr("diffltm.generation.harpoon.sample", harpoon_sample)
    condition = Condition(equality={"a": "two"})
    before, original = ev.generate("harpoon", condition, 5)
    monkeypatch.setitem(HARPOON_RECOVERY_POLICY, "max_discard_fraction", 0.20)
    after, changed = ev.generate("harpoon", condition, 5)
    assert original["sample_key"] != changed["sample_key"]
    assert original["generation_seed"] == changed["generation_seed"]
    assert (
        original["diagnostics"]["recovery"]["retry_seed"]
        == changed["diagnostics"]["recovery"]["retry_seed"]
    )
    pd.testing.assert_frame_equal(before, after, check_exact=True)
