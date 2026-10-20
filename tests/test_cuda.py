import pytest
import torch

from diffltm.generation.conditioning import Condition
from diffltm.generation.sampling import euler
from diffltm.models.model import DiffLTM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
@pytest.mark.parametrize("representation", ["onehot", "learnt", "tabrep"])
@pytest.mark.parametrize("objective", ["representation_mse", "logits_mse", "logits_ce"])
@pytest.mark.parametrize("backbone", ["mlp", "adaln"])
def test_cuda_amp_backward_reload(
    preprocessor, small_table, model_config, representation, objective, backbone
):
    cfg = dict(model_config, representation=representation, objective=objective, backbone=backbone)
    model = DiffLTM(preprocessor, cfg).cuda()
    batch = torch.from_numpy(preprocessor.transform(small_table.frame.iloc[:20])).cuda()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    with torch.autocast("cuda", dtype=dtype):
        loss = model.loss(batch)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    restored = DiffLTM(preprocessor, cfg).cuda()
    restored.load_state_dict(model.state_dict())
    frame, _ = euler(restored, 12, 31, Condition(equality={"a": "two"}), steps=3)
    assert len(frame) == 12 and frame.a.eq("two").all()
