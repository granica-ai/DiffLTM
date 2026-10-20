import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from helpers import config
from omegaconf import OmegaConf

from diffltm import trainer
from diffltm.artifacts import (
    completed_training,
    verify_result_training,
    write_json,
)
from diffltm.evaluation import Evaluation
from diffltm.models.model import DiffLTM


@pytest.fixture
def entrypoint():
    spec = importlib.util.spec_from_file_location(
        "train_cli", Path(__file__).parents[1] / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_run_reused_without_training_or_artifact_changes(
    tmp_path, small_table, model_config, monkeypatch, entrypoint
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg = config(tmp_path, model_config)
    trainer.train(cfg)
    run = Path(cfg["output_dir"])
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    monkeypatch.setattr(entrypoint, "train", lambda cfg: pytest.fail("Training was invoked"))
    cfg["device"] = "cuda"
    entrypoint.main(OmegaConf.create(cfg))
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        trainer.train(cfg)

    for key in ("seed", "dataset", "model", "training"):
        changed = copy.deepcopy(cfg)
        if key == "seed":
            changed[key] += 1
        else:
            changed[key][next(iter(changed[key]))] = "different"
        with pytest.raises(ValueError, match="configuration mismatch"):
            entrypoint.main(OmegaConf.create(changed))

    selected = run / "selected.pt"
    selected.unlink()
    with pytest.raises(ValueError, match="fresh output_dir"):
        entrypoint.main(OmegaConf.create(cfg))


@pytest.mark.parametrize(
    "status", [None, "running", "failed", "nonfinite_training_loss", "complete"]
)
def test_incomplete_runs_rejected_for_reuse_and_final_evaluation(
    tmp_path, model_config, monkeypatch, entrypoint, status
):
    cfg = config(tmp_path, model_config)
    run = Path(cfg["output_dir"])
    write_json(run / "config.json", cfg)
    if status is not None:
        write_json(run / "status.json", dict(status=status, epoch=1))
    monkeypatch.setattr(entrypoint, "train", lambda cfg: pytest.fail("Training was invoked"))
    for call in (lambda: entrypoint.main(OmegaConf.create(cfg)), lambda: completed_training(run)):
        with pytest.raises(ValueError, match="fresh output_dir"):
            call()


def test_acquisition_failure_reserves_directory_and_fresh_directory_succeeds(
    tmp_path, small_table, model_config, monkeypatch, entrypoint
):
    cfg = config(tmp_path, model_config)
    run = Path(cfg["output_dir"])
    (run / "hydra").mkdir(parents=True)

    def fail(*args):
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(trainer, "load_table", fail)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        entrypoint.main(OmegaConf.create(cfg))
    assert json.loads((run / "config.json").read_text()) == cfg
    with pytest.raises(ValueError, match="fresh output_dir"):
        entrypoint.main(OmegaConf.create(cfg))
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        trainer.train(cfg)
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg["output_dir"] = str(tmp_path / "fresh")
    entrypoint.main(OmegaConf.create(cfg))
    assert completed_training(cfg["output_dir"])["status"] == "complete"


@pytest.mark.parametrize("artifact", ["final.pt", "selected.pt", "status.json", "split.npz"])
def test_partial_artifacts_without_config_are_not_overwritten(
    tmp_path, model_config, monkeypatch, entrypoint, artifact
):
    cfg = config(tmp_path, model_config)
    path = Path(cfg["output_dir"]) / artifact
    path.parent.mkdir()
    path.write_bytes(b"partial run")
    monkeypatch.setattr(
        trainer, "load_table", lambda *args: pytest.fail("Data loading was invoked")
    )
    with pytest.raises(FileExistsError, match="fresh output_dir"):
        entrypoint.main(OmegaConf.create(cfg))
    assert path.read_bytes() == b"partial run"


def test_failure_after_selection_refuses_final_evaluation(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    original = DiffLTM.loss
    batches = 0

    def loss(self, batch):
        nonlocal batches
        if self.training:
            batches += 1
            result = original(self, batch)
            return result if batches == 1 else result * float("nan")
        return torch.tensor(0.1)

    monkeypatch.setattr(DiffLTM, "loss", loss)
    cfg = config(tmp_path, model_config)
    with pytest.raises(RuntimeError, match="nonfinite_training_loss"):
        trainer.train(cfg)
    saved = torch.load(tmp_path / "run/selected.pt", weights_only=False)
    assert saved["selected_epoch"] == 1 and saved["epoch"] == 2
    assert (tmp_path / "run/final.pt").exists()
    with pytest.raises(ValueError, match="completed training"):
        Evaluation(tmp_path / "run")


def test_completion_gate_in_progress_early_minimum_and_result_consumers(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    monkeypatch.setattr("diffltm.evaluation.load_table", lambda *args: small_table)
    original = trainer.save_checkpoint
    seen = []

    def save(path, payload):
        original(path, payload)
        if path.name == "selected.pt" and payload["status"] == "running":
            with pytest.raises(ValueError, match="completed training"):
                Evaluation(path.parent)
            seen.append(payload["epoch"])

    monkeypatch.setattr(trainer, "save_checkpoint", save)
    cfg = config(tmp_path, model_config)
    trainer.train(cfg)
    assert seen
    ev = Evaluation(tmp_path / "run")
    manifest = dict(training_runs=ev.identity["training_runs"])
    verify_result_training(manifest)
    status_path = tmp_path / "run/status.json"
    status = json.loads(status_path.read_text())
    status["status"] = "failed"
    write_json(status_path, status)
    verify_result_training(manifest)  # Saved results retain their completion evidence.
    with pytest.raises(ValueError, match="evidence"):
        verify_result_training({"status": "complete"})
