from pathlib import Path

import pytest
import torch

from diffltm.generation.conditioning import Condition
from diffltm.generation.sampling import euler
from diffltm.models.denoiser import Denoiser
from diffltm.models.model import DiffLTM, categorical_mse


def test_unequal_cardinality_loss_gradient():
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = categorical_mse(pred, target, [2, 3])
    expected = ((1 + 4) / 2 + (9 + 16 + 25) / 3) / 2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[0.5, 1.0, 1.0, 4 / 3, 5 / 3]]))
    assert categorical_mse(pred[:, :4], target[:, :4], [2, 2]) == torch.nn.functional.mse_loss(
        pred[:, :4], target[:, :4]
    )


@pytest.mark.parametrize("representation", ["onehot", "learnt", "tabrep"])
@pytest.mark.parametrize("objective", ["representation_mse", "logits_mse", "logits_ce"])
@pytest.mark.parametrize("backbone", ["mlp", "adaln"])
def test_every_recipe_backward_reload_generate(
    preprocessor, small_table, model_config, representation, objective, backbone
):
    cfg = dict(model_config, representation=representation, objective=objective, backbone=backbone)
    model = DiffLTM(preprocessor, cfg)
    batch = torch.from_numpy(preprocessor.transform(small_table.frame.iloc[:16]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    restored = DiffLTM(preprocessor, cfg)
    restored.load_state_dict(model.state_dict())
    frame, meta = euler(restored, 12, 42, Condition(equality={"a": "two"}), steps=3)
    assert len(frame) == 12 and frame.a.eq("two").all()
    assert set(frame.a) <= set(preprocessor.levels[0])
    assert meta["post_pin_adherence"] == 1
    x = torch.randn(3, model.dim)
    t = torch.rand(3)
    zeros = (torch.zeros(3, model.condition_dim),) * 2
    for a, b in zip(model.predict(x, t), model.predict(x, t, zeros)):
        if a is not None:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_reference_forward_fixture():
    fixtures = torch.load(
        Path(__file__).parent / "fixtures/denoiser_reference.pt", weights_only=True
    )
    for fixture in fixtures:
        model = Denoiser(9, 2, [4, 3], 7, fixture["config"])
        model.load_state_dict(fixture["state"])
        cond = None if fixture["condition"] is None else (fixture["condition"], fixture["mask"])
        result = model(fixture["x"], fixture["t"], cond)
        for a, b in zip(result, fixture["expected"]):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
