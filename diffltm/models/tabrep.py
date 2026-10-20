"""Fixed unit-circle CatConverter, with interleaved [sin, cos] coordinates."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CatConverter(nn.Module):
    """One categorical column; cardinality includes the caller's unknown ID.

    The integer encoder remains the authority for category order. Tables are
    buffers rather than parameters so device movement and checkpoints preserve
    them. No sqrt(d) scaling or normalization is applied.
    """

    def __init__(self, cardinality: int):
        super().__init__()
        if cardinality < 1:
            raise ValueError("TabRep cardinality must be positive")
        self.cardinality = cardinality
        phase = torch.arange(cardinality, dtype=torch.float64) * (2 * math.pi / cardinality)
        self.register_buffer("table", torch.stack((phase.sin(), phase.cos()), -1).float())

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # BaseGenerator's historical compatibility loader resizes buffers. A
        # codebook is geometry, not a resizable training statistic: reject it.
        table = state_dict.get(prefix + "table")
        if table is not None and table.shape != (self.cardinality, 2):
            error_msgs.append(
                f"TabRep cardinality mismatch for {prefix}table: expected "
                f"({self.cardinality}, 2), got {tuple(table.shape)}"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Unlike advanced indexing, embedding rejects negative IDs instead of
        # treating -1 as the last category. Dataset IDs are already integers.
        return F.embedding(ids, self.table)

    def decode(self, values: torch.Tensor, *, mask_unknown: bool) -> torch.Tensor:
        """Nearest allowed phase in O(batch) space/time, independent of K.

        Positive radial rescaling has no effect. The origin and exact angular
        ties select the lowest allowed ID. Nonfinite points are errors, never
        silently cast into categories. Use float64 for the final angular
        decision (also under AMP); the denoiser/codebook remain float32.
        """
        k = self.cardinality
        first = int(mask_unknown)
        if first >= k:
            raise ValueError("TabRep has no allowed categories after masking unknown ID 0")
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("TabRep decode expects (batch, 2) [sin, cos] values")
        if not torch.isfinite(values).all():
            raise ValueError("TabRep cannot decode nonfinite categorical endpoints")

        with torch.autocast(device_type=values.device.type, enabled=False):
            s, c = values.double().unbind(-1)
            u = torch.remainder(torch.atan2(s, c), 2 * math.pi) * (k / (2 * math.pi))
            lower = u.floor().long().remainder(k)
            upper = (lower + 1).remainder(k)
            if mask_unknown:
                # The two neighbors of the missing root are K-1 and 1.
                lower = torch.where(lower == 0, k - 1, lower)
                upper = torch.where(upper == 0, 1, upper)
            dl = (u - lower).abs()
            du = (u - upper).abs()
            dl = torch.minimum(dl, k - dl)
            du = torch.minimum(du, k - du)
            choose_lower = (dl < du) | ((dl == du) & (lower < upper))
            ids = torch.where(choose_lower, lower, upper)
            return torch.where((s == 0) & (c == 0), first, ids)
