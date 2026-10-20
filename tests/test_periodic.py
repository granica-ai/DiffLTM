import copy
import json
import multiprocessing
import os
import random
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from threading import Timer

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import assert_rng_equal, config, rng_state

from diffltm import periodic, trainer
from diffltm.artifacts import (
    completed_training,
)
from diffltm.evaluation import Evaluation
from diffltm.models.model import DiffLTM

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")


def forbidden_model(*args, **kwargs):
    raise AssertionError("The CPU scorer must not construct or sample a model")


def blocked_initializer(ready, release, preprocessor, train_path, validation_path, reference_id):
    ready.set()
    if not release.wait(30):
        raise RuntimeError("Test did not release the scoring worker")
    assert not torch.cuda.is_initialized()
    periodic.initialize(preprocessor, train_path, validation_path, reference_id)
    periodic.euler = forbidden_model
    DiffLTM.__init__ = forbidden_model


def failed_initializer(*args):
    raise RuntimeError("Initialization failed")


def crashed_job(job):
    os._exit(23)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
@pytest.mark.parametrize("use_ema", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_sampling_uses_live_device_ema_null_condition_and_restores_rng_and_modes(
    preprocessor, model_config, monkeypatch, device, use_ema, training
):
    model = DiffLTM(preprocessor, model_config).to(device)
    model.train(training)
    model.embeddings.eval()  # Preserve mixed module modes too.
    ema = copy.deepcopy(model.denoiser).eval() if use_ema else None
    if ema is not None:
        with torch.no_grad():
            next(ema.parameters()).add_(0.1)
    live = model.denoiser
    modes = [(m, m.training) for m in model.modules()]
    state = trainer.cpu_state(model)
    original = model.predict
    calls = []

    def predict(x, t, condition=None):
        assert x.device.type == device and condition is None
        assert model.denoiser is (ema if ema is not None else live)
        assert not model.training
        calls.append(len(x))
        random.random()
        np.random.rand()
        torch.rand(1)
        if device == "cuda":
            torch.rand(1, device=device)
        return original(x, t, condition)

    monkeypatch.setattr(model, "predict", predict)
    before = rng_state()
    result, diagnostics = periodic.sample_snapshot(model, ema, 7, 22)
    assert len(result) == 7 and len(calls) == 100
    assert diagnostics["recovery"]["attempted_rows"] == 7
    assert model.denoiser is live
    assert all(m.training == was_training for m, was_training in modes)
    assert ema is None or not ema.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), state[key], atol=0, rtol=0)
    assert_rng_equal(before)

    def fail(*args):
        model.eval()
        torch.rand(1, device=device)
        random.random()
        np.random.rand()
        raise RuntimeError("Sampling failed")

    monkeypatch.setattr(periodic, "euler", fail)
    with pytest.raises(RuntimeError, match="Sampling failed"):
        periodic.sample_snapshot(model, ema, 7, 22)
    assert model.denoiser is live
    assert all(m.training == was_training for m, was_training in modes)
    assert_rng_equal(before)


def test_references_reject_overlapping_training_and_validation_rows(
    tmp_path, preprocessor, small_table
):
    with pytest.raises(ValueError, match="overlap"):
        periodic.PeriodicWorker(tmp_path, preprocessor.schema, small_table.frame, small_table.frame)


def test_queue_never_waits_for_scoring_and_disk_snapshots_are_immutable(
    tmp_path, preprocessor, small_table, monkeypatch
):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    monkeypatch.setattr(periodic, "initialize", partial(blocked_initializer, ready, release))
    train, validation = small_table.frame.iloc[:80].copy(), small_table.frame.iloc[80:].copy()
    pool = train.copy()
    worker = periodic.PeriodicWorker(tmp_path, preprocessor.schema, train, validation)
    references = {p.name: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    watchdog = Timer(30, release.set)
    watchdog.start()
    before = rng_state()
    try:
        worker.submit(1, 22, pool, True)
        assert ready.wait(20)
        for epoch in [2, 3]:
            worker.submit(epoch, 22 + epoch, pool)
        worker.poll()
        assert not release.is_set(), "Submission blocked on outstanding scoring"
        assert len(worker.pending) == 3
        for epoch in [1, 2, 3]:
            assert json.loads((tmp_path / f"epoch-{epoch}.json").read_text())["status"] == "queued"
        assert references == {
            p.name: p.read_bytes() for p in tmp_path.glob("*.parquet") if p.name in references
        }
        train.loc[:, "x"] = 999
        validation.loc[:, "x"] = 999
        pool.loc[:, "x"] = 999
    finally:
        release.set()
        watchdog.cancel()
        worker.close()
    assert not worker.pending
    assert_rng_equal(before)
    for epoch in [1, 2, 3]:
        result = json.loads((tmp_path / f"epoch-{epoch}.json").read_text())
        assert result["status"] == "complete"
        assert result["generated_rows"] == 80 and result["rows"] == 40
        assert (
            result["reference_identity"]
            == json.loads((tmp_path / "references.json").read_text())["identity"]
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / f"epoch-{epoch}.parquet"),
            small_table.frame.iloc[:80].reset_index(drop=True),
        )
    result = json.loads((tmp_path / "epoch-1.json").read_text())
    assert len(result["mle_rounds"]["trtr"]) == len(result["mle_rounds"]["tstr"]) == 10
    assert result["mle"] == pytest.approx(
        np.mean(result["mle_rounds"]["tstr"]) / np.mean(result["mle_rounds"]["trtr"])
    )


@pytest.mark.parametrize("failure", ["pool", "initializer", "process"])
def test_scoring_failure_is_recorded_and_propagated(
    tmp_path, preprocessor, small_table, monkeypatch, failure
):
    if failure == "initializer":
        monkeypatch.setattr(periodic, "initialize", failed_initializer)
    elif failure == "process":
        monkeypatch.setattr(periodic, "execute", crashed_job)
    worker = periodic.PeriodicWorker(
        tmp_path, preprocessor.schema, small_table.frame.iloc[:80], small_table.frame.iloc[80:]
    )
    worker.submit(1, 22, small_table.frame.iloc[:79])
    if failure == "pool":
        # Observe asynchronous failure before shutdown, without blocking poll().
        future = worker.pending[1][1]
        assert isinstance(future.exception(timeout=30), ValueError)
        with pytest.raises(ValueError, match="N_train"):
            worker.poll()
        worker.close()
    else:
        with pytest.raises(BrokenProcessPool):
            worker.close()
    result = json.loads((tmp_path / "epoch-1.json").read_text())
    assert result["status"] == "failed" and "error" in result


@pytest.mark.parametrize(
    "device,ema,conditional",
    [
        ("cpu", 0, False),
        ("cpu", 0.9, True),
        pytest.param("cuda", 0, True, marks=CUDA),
        pytest.param("cuda", 0.9, False, marks=CUDA),
    ],
)
def test_periodic_training_is_exactly_equal_to_training_without_diagnostics(
    tmp_path, small_table, model_config, monkeypatch, device, ema, conditional
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg = config(
        tmp_path,
        dict(model_config, conditional=conditional, ranges=conditional, linear=conditional),
    )
    cfg["device"] = device
    cfg["training"].update(ema_decay=ema, batch_size=31, amp=device == "cuda")
    trainer.train(cfg)
    before = rng_state()
    baseline = {
        name: torch.load(tmp_path / "run" / name, weights_only=False)
        for name in ["selected.pt", "final.pt"]
    }
    cfg["output_dir"] = str(tmp_path / "periodic")
    cfg["training"]["periodic_every"] = 1
    trainer.train(cfg)
    assert_rng_equal(before)
    for name, expected in baseline.items():
        actual = torch.load(tmp_path / "periodic" / name, weights_only=False)
        for key in ["history", "selected_epoch", "selected_loss", "updates"]:
            assert actual[key] == expected[key]
        for key, value in actual["model"].items():
            torch.testing.assert_close(value, expected["model"][key], atol=0, rtol=0)
    assert completed_training(tmp_path / "periodic")["status"] == "complete"
    for epoch in [1, 2, 3]:
        result = json.loads((tmp_path / "periodic/periodic" / f"epoch-{epoch}.json").read_text())
        assert result["status"] == "complete"
        assert result["generated_rows"] == 84 and result["rows"] == 12
        assert result["generation"]["recovery"]["attempted_rows"] == 84
        assert result["generation"]["recovery"]["discarded_rows"] == 0


def test_training_reaches_full_budget_before_waiting_and_only_then_certifies_completion(
    tmp_path, preprocessor, small_table, model_config, monkeypatch
):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    monkeypatch.setattr(periodic, "initialize", partial(blocked_initializer, ready, release))
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    original_close, original_write = periodic.PeriodicWorker.close, trainer.write_json
    phases = []

    def write(path, value):
        if path.name == "status.json":
            phases.append(value["phase"])
        original_write(path, value)

    def close(self):
        assert not release.is_set(), "Training blocked on the CPU scorer before the final wait"
        assert len(self.pending) == 3
        run = tmp_path / "run"
        status = json.loads((run / "status.json").read_text())
        assert status["phase"] == "waiting_for_periodic" and status["epoch"] == 3
        assert status["status"] == "running"
        with pytest.raises(ValueError, match="completed training"):
            completed_training(run)
        release.set()
        original_close(self)

    monkeypatch.setattr(trainer, "write_json", write)
    monkeypatch.setattr(periodic.PeriodicWorker, "close", close)
    cfg = config(tmp_path, model_config)
    cfg["training"]["periodic_every"] = 1
    watchdog = Timer(30, release.set)
    watchdog.start()
    try:
        trainer.train(cfg)
    finally:
        release.set()
        watchdog.cancel()
    assert phases.count("periodic_sampling") == 3
    assert phases.count("waiting_for_periodic") == 1
    assert phases[-1] == "complete"
    completed_training(tmp_path / "run")


def failed_job(job):
    raise ValueError("CPU scoring failed")


@pytest.mark.parametrize("stage", ["sampling", "scoring"])
def test_periodic_failure_prevents_final_training_completion(
    tmp_path, small_table, model_config, monkeypatch, stage
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    if stage == "sampling":

        def fail(*args):
            raise ValueError("GPU sampling failed")

        monkeypatch.setattr(trainer, "sample_snapshot", fail)
    else:
        monkeypatch.setattr(periodic, "execute", failed_job)
        original_submit = periodic.PeriodicWorker.submit

        def submit(self, *args):
            original_submit(self, *args)
            # Make the failure available to the next training-batch poll.
            assert isinstance(self.pending[1][1].exception(timeout=30), ValueError)

        monkeypatch.setattr(periodic.PeriodicWorker, "submit", submit)
    cfg = config(tmp_path, model_config)
    cfg["training"]["periodic_every"] = 1
    with pytest.raises(ValueError, match=f"{stage} failed"):
        trainer.train(cfg)
    status = json.loads((tmp_path / "run/status.json").read_text())
    assert status["status"] == status["phase"] == "failed"
    assert status["updates"] == 1
    with pytest.raises(ValueError, match="completed training"):
        completed_training(tmp_path / "run")
    if stage == "scoring":
        assert (
            json.loads((tmp_path / "run/periodic/epoch-1.json").read_text())["status"] == "failed"
        )


def scoring_initializer(ready, release, *args):
    periodic.initialize(*args)
    original = periodic.score

    def score(*args):
        ready.set()
        if not release.wait(30):
            raise RuntimeError("Test did not release scoring")
        return original(*args)

    periodic.score = score


def test_running_scorer_does_not_block_next_submission(
    tmp_path, small_table, preprocessor, monkeypatch
):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    monkeypatch.setattr(periodic, "initialize", partial(scoring_initializer, ready, release))
    train, validation = small_table.frame.iloc[:80], small_table.frame.iloc[80:]
    worker = periodic.PeriodicWorker(tmp_path, preprocessor.schema, train, validation)
    watchdog = Timer(30, release.set)
    watchdog.start()
    try:
        worker.submit(1, 22, train)
        assert ready.wait(20)
        worker.submit(2, 23, train)
        assert not release.is_set()
        assert json.loads((tmp_path / "epoch-1.json").read_text())["status"] == "running"
        assert json.loads((tmp_path / "epoch-2.json").read_text())["status"] == "queued"
    finally:
        release.set()
        watchdog.cancel()
        worker.close()
    assert all(
        json.loads((tmp_path / f"epoch-{epoch}.json").read_text())["status"] == "complete"
        for epoch in [1, 2]
    )


@pytest.mark.parametrize("failure", ["persist", "submit"])
def test_pool_write_and_submission_failures_are_recorded(
    tmp_path, preprocessor, small_table, monkeypatch, failure
):
    train, validation = small_table.frame.iloc[:80], small_table.frame.iloc[80:]
    worker = periodic.PeriodicWorker(tmp_path, preprocessor.schema, train, validation)

    def fail(*args, **kwargs):
        raise OSError("Queue failed")

    if failure == "persist":
        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    else:
        monkeypatch.setattr(worker.executor, "submit", fail)
    try:
        with pytest.raises(OSError, match="Queue failed"):
            worker.submit(1, 22, train)
        assert not worker.pending
        assert json.loads((tmp_path / "epoch-1.json").read_text())["status"] == "failed"
    finally:
        worker.close()


def test_periodic_uses_train_pool_and_full_validation_without_cap(
    tmp_path, preprocessor, monkeypatch
):
    train = pd.DataFrame({"x": np.arange(150001)})
    validation = pd.DataFrame({"x": np.arange(100001)}, index=np.arange(150001, 250002))
    pool = tmp_path / "pool.parquet"
    train.to_parquet(pool)
    job = periodic.DiagnosticJob(1, 3, pool, True)
    calls = []

    def fidelity(real, generated, *args):
        assert len(real) == len(generated) == len(validation)
        calls.append("fidelity")
        return {"shape": 1.0}

    def mle(real, holdout, generated, *args, **kwargs):
        assert (len(real), len(holdout), len(generated)) == (150001, 100001, 150001)
        assert kwargs["fit_n"] == 100001
        calls.append("mle")
        return 1.0, {}

    monkeypatch.setattr("diffltm.metrics.fidelity", fidelity)
    monkeypatch.setattr("diffltm.metrics.mle", mle)
    periodic.score(job, preprocessor.schema, train, validation)
    assert calls == ["fidelity", "fidelity", "mle"]


def test_periodic_shutdown_failure_never_certifies_training(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)

    class FailingWorker:
        def __init__(self, *args):
            pass

        def poll(self):
            pass

        def submit(self, *args):
            pass

        def close(self):
            raise RuntimeError("periodic execution failed")

    monkeypatch.setattr(trainer, "PeriodicWorker", FailingWorker)
    cfg = config(tmp_path, model_config)
    cfg["training"]["periodic_every"] = 1
    with pytest.raises(RuntimeError, match="periodic execution failed"):
        trainer.train(cfg)
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="completed training"):
        Evaluation(tmp_path / "run")
