"""Publication MLPs: additive time or non-residual time-modulated AdaLN."""

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2)),
            persistent=False,
        )

    def forward(self, t):
        angles = t[:, None] * self.freqs
        return torch.cat([angles.cos(), angles.sin()], -1)


class AdaLNBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.linear = nn.Linear(width, width)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, t):
        scale, shift = self.mod(t).chunk(2, -1)
        return torch.nn.functional.silu(self.linear(self.norm(x) * (1 + scale) + shift))


class Denoiser(nn.Module):
    def __init__(self, dim, n_num, widths, out_cat, config):
        super().__init__()
        self.dim, self.n_num, self.config = dim, n_num, config
        width = config["width"]
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.condition_dim = (
            dim + (2 * n_num if config["ranges"] else 0) + (n_num + 5 if config["linear"] else 0)
        )
        size = dim + 2 * self.condition_dim if config["conditional"] else dim
        self.input_proj = nn.Linear(size, width)
        self.obs_token = nn.Parameter(torch.zeros(dim)) if config["conditional"] else None
        self.adaln = config["backbone"] == "adaln"
        if self.adaln:
            self.blocks = nn.ModuleList([AdaLNBlock(width) for _ in range(config["depth"])])
        else:
            self.hidden = nn.Sequential(
                *[
                    layer
                    for _ in range(config["depth"])
                    for layer in (nn.Linear(width, width), nn.SiLU())
                ]
            )
        self.num_head = nn.Linear(width, n_num) if n_num else None
        self.cat_head = nn.Linear(width, out_cat) if out_cat else None
        perm, start = [], 0
        for i, w in enumerate([1] * n_num + widths):
            for offset in [0, dim, dim + self.condition_dim]:
                perm.extend(range(offset + start, offset + start + w))
            extra = []
            if i < n_num and config["ranges"]:
                extra += [dim + i, dim + n_num + i]
            if i < n_num and config["linear"]:
                extra += [dim + (2 * n_num if config["ranges"] else 0) + i]
            for offset in extra:
                perm += [dim + offset, dim + self.condition_dim + offset]
            start += w
        if config["linear"]:
            for offset in range(self.condition_dim - 5, self.condition_dim):
                perm += [dim + offset, dim + self.condition_dim + offset]
        self.register_buffer("permutation", torch.tensor(perm), persistent=False)
        if config["conditional"] and sorted(perm) != list(range(size)):
            raise ValueError("Invalid conditioning layout")

    def forward(self, x, t, condition=None):
        if self.config["conditional"]:
            if condition is None:
                u = x.new_zeros(len(x), self.condition_dim)
                mask = u
            else:
                u, mask = condition
            x = x * (1 - mask[:, : self.dim]) + self.obs_token * mask[:, : self.dim]
            x = torch.cat([x, u, mask], 1)[:, self.permutation]
        elif condition is not None:
            raise ValueError("Unconditional checkpoint has no CFG channels")
        time = self.time_embed(t)
        h = self.input_proj(x)
        if self.adaln:
            for block in self.blocks:
                h = block(h, time)
        else:
            h = self.hidden(h + time)
        return (
            self.num_head(h) if self.num_head is not None else h[:, :0],
            self.cat_head(h) if self.cat_head is not None else h[:, :0],
        )
