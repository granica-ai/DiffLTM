"""Sample on the training device; queue persisted pools for one CPU scorer."""

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import atomic_write, file_identity, identity, json_safe, write_json
from .generation.sampling import euler


def sample_snapshot(model, ema, n, seed):
    device = next(model.parameters()).device
    live = model.denoiser
    modules = list(model.modules()) + (list(ema.modules()) if ema is not None else [])
    modes = [(module, module.training) for module in modules]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            if ema is not None:
                model.denoiser = ema
            return euler(model, n, seed)  # Null condition, including for CFG models.
    finally:
        model.denoiser = live
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


@dataclass(frozen=True)
class DiagnosticJob:
    epoch: int
    seed: int
    pool: Path
    compute_mle: bool = False
    generation: dict | None = None


def initialize(schema, train_path, validation_path, reference_id):
    global _references, _reference_id
    torch.set_num_threads(1)
    _references = (schema, pd.read_parquet(train_path), pd.read_parquet(validation_path))
    _reference_id = reference_id


def score(job, schema, train, validation):
    return score_frames(
        pd.read_parquet(job.pool), schema, train, validation, job.seed, job.compute_mle
    )


def score_frames(synthetic, schema, train, validation, seed, compute_mle=True):
    from .metrics import fidelity, mle, subset

    if len(synthetic) != len(train):
        raise ValueError("Periodic pool must contain N_train rows")
    n = len(validation)
    result = dict(
        rows=n,
        generated_rows=len(synthetic),
        metric_settings=dict(fidelity_backend="ltm-eval==0.1.0", clean="safe"),
        fidelity=fidelity(validation, subset(synthetic, n, seed), schema, seed),
        ideal=fidelity(validation, subset(train, n, seed), schema, seed),
    )
    if compute_mle:
        result["mle"], result["mle_rounds"] = mle(
            train, validation, synthetic, schema, seed, fit_n=n
        )
    return json_safe(result)


def write_status(job, status, **result):
    write_json(
        job.pool.with_suffix(".json"),
        dict(
            status=status,
            epoch=job.epoch,
            generation_seed=job.seed,
            scoring_seed=job.seed,
            pool=job.pool.name,
            generation=job.generation,
            **result,
        ),
    )


def execute(job):
    try:
        write_status(job, "running")
        result = score(job, *_references)
        write_status(job, "complete", reference_identity=_reference_id, **result)
    except BaseException as exc:
        write_status(job, "failed", error=repr(exc))
        raise


class PeriodicWorker:
    def __init__(self, output, schema, train, validation):
        if len(train.index.intersection(validation.index)):
            raise ValueError("Diagnostic train/validation rows overlap")
        self.output = Path(output)
        train_path, validation_path = (
            self.output / "train.parquet",
            self.output / "validation.parquet",
        )
        atomic_write(train_path, lambda path: train.to_parquet(path))
        atomic_write(validation_path, lambda path: validation.to_parquet(path))
        references = dict(
            schema=schema.__dict__,
            train_sha256=file_identity(train_path),
            validation_sha256=file_identity(validation_path),
        )
        reference_id = identity(references)
        write_json(self.output / "references.json", dict(identity=reference_id, **references))
        self.executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(schema, train_path, validation_path, reference_id),
        )
        self.pending = {}

    def poll(self, wait=False):
        error = None
        for epoch, (job, future) in list(self.pending.items()):
            if not wait and not future.done():
                continue
            del self.pending[epoch]
            try:
                future.result()
            except BaseException as exc:
                # Also covers a dead process or a failed worker initializer.
                write_status(job, "failed", error=repr(exc))
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def submit(self, epoch, seed, synthetic, compute_mle=False, generation=None):
        self.poll()
        job = DiagnosticJob(
            epoch, seed, self.output / f"epoch-{epoch}.parquet", compute_mle, generation
        )
        try:
            atomic_write(job.pool, lambda path: synthetic.to_parquet(path, index=False))
            write_status(job, "queued")
            self.pending[epoch] = (job, self.executor.submit(execute, job))
        except BaseException as exc:
            write_status(job, "failed", error=repr(exc))
            raise

    def close(self):
        try:
            self.poll(wait=True)
        finally:
            self.executor.shutdown(wait=True)
