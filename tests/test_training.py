import pytest
import torch
from helpers import config

from diffltm import trainer
from diffltm.models.model import DiffLTM


def test_full_budget_early_minimum_live_embeddings_and_ema(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    original = DiffLTM.loss
    validation = iter([0.1, 1.0, 2.0])
    captured = {}
    live_model = None

    def controlled_loss(self, batch):
        nonlocal live_model
        live_model = self
        if self.training:
            return original(self, batch)
        value = next(validation)
        if value == 0.1:
            captured.update(trainer.cpu_state(self))
        return torch.tensor(value)

    monkeypatch.setattr(DiffLTM, "loss", controlled_loss)
    cfg = config(tmp_path, model_config)
    path = trainer.train(cfg)
    selected = torch.load(path, weights_only=False)
    final = torch.load(path.with_name("final.pt"), weights_only=False)
    live_state = trainer.cpu_state(live_model)
    assert (
        selected["epoch"] == 3
        and selected["selected_epoch"] == 1
        and selected["selected_loss"] == pytest.approx(0.1)
    )
    assert len(selected["history"]) == 3 and selected["updates"] == 3
    assert selected["selection"] == "minimum_ema_validation_loss"
    for k, v in selected["model"].items():
        torch.testing.assert_close(v, captured[k], atol=0, rtol=0)
    assert any(
        not torch.equal(v, final["model"][k])
        for k, v in selected["model"].items()
        if k.startswith("embeddings.")
    )
    assert all(
        torch.equal(final["model"][k], live_state[k])
        for k in final["model"]
        if k.startswith("embeddings.")
    )
    assert any(
        not torch.equal(final["model"][k], live_state[k])
        for k in final["model"]
        if k.startswith("denoiser.")
    )


def test_validation_rng_progresses_and_checkpoints_reload(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg = config(tmp_path, model_config)
    trainer.train(cfg)
    saved = torch.load(tmp_path / "run/final.pt", weights_only=False)
    assert len({r["validation_loss"] for r in saved["history"]}) > 1
    for name in ("selected.pt", "final.pt"):
        model, checkpoint = trainer.load_model(tmp_path / "run" / name)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, checkpoint["model"][key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional"),
        ),
    ],
)
def test_batches_keep_short_batch_and_progressing_rng(
    tmp_path, small_table, model_config, monkeypatch, device
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg = config(tmp_path, model_config)
    cfg["device"] = device
    cfg["training"]["batch_size"] = 31
    observed = {"train": [], "validation": []}
    original = DiffLTM.loss

    def loss(self, batch):
        assert batch.device.type == device
        observed["train" if self.training else "validation"].append(batch.detach().clone())
        return original(self, batch)

    monkeypatch.setattr(DiffLTM, "loss", loss)
    trainer.train(cfg)
    assert [len(b) for b in observed["train"]] == [31, 31, 22] * 3
    assert [len(b) for b in observed["validation"]] == [12] * 3
    assert not torch.equal(observed["train"][0], observed["train"][3])
    assert torch.equal(observed["validation"][0], observed["validation"][1])


def test_fresh_runs_have_identical_tensor_batches_and_results(
    tmp_path, small_table, model_config, monkeypatch
):
    monkeypatch.setattr(trainer, "load_table", lambda *args: small_table)
    cfg = config(tmp_path, model_config)
    cfg["training"]["batch_size"] = 31
    trainer.train(cfg)
    expected = torch.load(tmp_path / "run/final.pt", weights_only=False)
    cfg.update(output_dir=str(tmp_path / "fresh"))
    trainer.train(cfg)
    actual = torch.load(tmp_path / "fresh/final.pt", weights_only=False)
    assert actual["history"] == expected["history"]
    assert actual["selected_epoch"] == expected["selected_epoch"]
    for key, value in actual["model"].items():
        torch.testing.assert_close(value, expected["model"][key], atol=0, rtol=0)
