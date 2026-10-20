"""Continuous diffusion with three categorical representations and objectives."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..generation.conditioning import training_channels
from .denoiser import Denoiser
from .tabrep import CatConverter


def sigma(t):
    return (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))) ** 7


def categorical_mse(predicted, target, widths):
    if not widths:
        return predicted.sum() * 0
    return torch.stack(
        [
            (a - b).square().mean()
            for a, b in zip(predicted.split(widths, 1), target.split(widths, 1))
        ]
    ).mean()


class OneHotCodes(nn.Module):
    def __init__(self, cardinality):
        super().__init__()
        self.register_buffer("table", torch.eye(cardinality))


class DiffLTM(nn.Module):
    def __init__(self, preprocessor, config):
        super().__init__()
        self.preprocessor, self.config = preprocessor, dict(config)
        p = preprocessor
        if config["representation"] not in {"onehot", "learnt", "tabrep"} or config[
            "objective"
        ] not in {"representation_mse", "logits_mse", "logits_ce"}:
            raise ValueError("Unknown representation/objective")
        if config["backbone"] not in {"mlp", "adaln"}:
            raise ValueError("Unknown backbone")
        if not config["conditional"] and (config["ranges"] or config["linear"]):
            raise ValueError("Condition blocks require conditional model")
        self.n_num = len(p.schema.numerical)
        if config["linear"] and not self.n_num:
            raise ValueError("Linear conditioning requires numerical columns")
        self.cards = p.cardinalities
        self.widths = [
            c
            if config["representation"] == "onehot"
            else 2
            if config["representation"] == "tabrep"
            else config["embedding_dim"]
            for c in self.cards
        ]
        self.dim = self.n_num + sum(self.widths)
        self.embeddings = nn.ModuleList()
        for c, d in zip(self.cards, self.widths):
            if config["representation"] == "learnt":
                self.embeddings.append(nn.Embedding(c, d))
            else:
                converter = (
                    OneHotCodes(c) if config["representation"] == "onehot" else CatConverter(c)
                )
                self.embeddings.append(converter)
        out_cat = (
            sum(self.widths) if config["objective"] == "representation_mse" else sum(self.cards)
        )
        self.denoiser = Denoiser(self.dim, self.n_num, self.widths, out_cat, config)
        self.condition_dim = self.denoiser.condition_dim
        grid, raw = p.inverse_lookup()
        for name, value in [
            ("lookup_grid", grid),
            ("lookup_raw", raw),
            ("model_lo", p.model_min),
            ("model_hi", p.model_max),
            ("raw_lo", (p.raw_min - p.mean) / p.scale),
            ("raw_hi", (p.raw_max - p.mean) / p.scale),
        ]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    def tables(self):
        if self.config["representation"] == "learnt":
            return [
                F.normalize(e.weight, dim=-1) * math.sqrt(d)
                for e, d in zip(self.embeddings, self.widths)
            ]
        return [e.table for e in self.embeddings]

    def encode(self, batch):
        cat = [
            F.embedding(batch[:, self.n_num + j].long(), table)
            for j, table in enumerate(self.tables())
        ]
        return torch.cat([batch[:, : self.n_num]] + cat, 1)

    def raw_standardized(self, values):
        grid, raw = self.lookup_grid, self.lookup_raw
        lo, hi = grid[0], grid[-1]
        positions = (
            (values.clamp(min=lo, max=hi) - lo) / (hi - lo).clamp_min(1e-12) * (len(grid) - 1)
        )
        left = positions.floor().long().clamp(0, len(grid) - 2)
        columns = torch.arange(self.n_num, device=values.device)
        return torch.lerp(
            raw[left, columns], raw[left + 1, columns], (positions - left).clamp(0, 1)
        )

    def predict(self, x, t, condition=None):
        if condition is not None and self.config["linear"]:
            u, mask = condition
            offset = self.dim + (2 * self.n_num if self.config["ranges"] else 0)
            w = u[:, offset : offset + self.n_num]
            lo, hi = u[:, offset + self.n_num : offset + self.n_num + 2].unbind(1)
            s = (self.raw_standardized(x[:, : self.n_num]) * w).sum(1)
            state = torch.stack([s, s - lo, hi - s], 1)
            condition = (torch.cat([u[:, :-3], state], 1), mask)
        num, cat = self.denoiser(x, t, condition)
        logits = None
        if self.config["objective"] != "representation_mse":
            logits = cat
            cat = self.logits_to_representation(logits)
        return torch.cat([num, cat], 1), logits

    def logits_to_representation(self, logits):
        parts = []
        for values, table in zip(logits.split(self.cards, 1), self.tables()):
            if self.config["representation"] == "tabrep":
                with torch.autocast(device_type=values.device.type, enabled=False):
                    parts.append(values.float().softmax(-1) @ table.float())
            else:
                parts.append(values.softmax(-1) @ table)
        return torch.cat(parts, 1) if parts else logits

    def loss(self, batch):
        clean = self.encode(batch)
        t = torch.rand(len(batch), device=batch.device)
        noisy = clean + sigma(t)[:, None] * torch.randn_like(clean)
        cond = training_channels(self, clean) if self.config["conditional"] else None
        prediction, logits = self.predict(noisy, t, cond)
        num_loss = (
            F.mse_loss(prediction[:, : self.n_num], clean[:, : self.n_num])
            if self.n_num
            else prediction.sum() * 0
        )
        if self.config["objective"] == "logits_ce" and self.cards:
            cat_loss = torch.stack(
                [
                    F.cross_entropy(a, batch[:, self.n_num + j].long())
                    for j, a in enumerate(logits.split(self.cards, 1))
                ]
            ).mean()
        else:
            cat_loss = categorical_mse(
                prediction[:, self.n_num :], clean[:, self.n_num :], self.widths
            )
        return num_loss + cat_loss

    def decode(self, state, logits=None):
        if not torch.isfinite(state).all() or (
            logits is not None and not torch.isfinite(logits).all()
        ):
            raise ValueError("Nonfinite generation endpoint")
        if logits is not None:
            cats = []
            for a in logits.split(self.cards, 1):
                a = a.clone()
                a[:, 0] = -torch.inf
                cats.append(a.argmax(1))
        elif self.config["representation"] == "tabrep":
            cats = [
                e.decode(a, mask_unknown=True)
                for e, a in zip(self.embeddings, state[:, self.n_num :].split(self.widths, 1))
            ]
        else:
            cats = []
            for a, table in zip(state[:, self.n_num :].split(self.widths, 1), self.tables()):
                distance = torch.cdist(a.float(), table[1:].float())
                cats.append(distance.argmin(1) + 1)
        cat = torch.stack(cats, 1) if cats else state.new_empty(len(state), 0)
        return self.preprocessor.inverse(
            state[:, : self.n_num].detach().cpu().numpy(), cat.detach().cpu().numpy()
        )
