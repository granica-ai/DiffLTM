"""Disk-backed candidates shared only within one evaluation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from ..artifacts import atomic_write, request_seed
from .sampling import RECOVERY_POLICY

REJECTION_MAX_CANDIDATES = 10_000_000


class RejectionPool:
    def __init__(self, sampler, seed, chunk_size=10000, directory=None):
        self._temporary = TemporaryDirectory(prefix="rejection-", dir=directory)
        self.root = Path(self._temporary.name)
        self.sampler, self.seed, self.chunk_size = sampler, seed, chunk_size
        self.chunks = []

    def close(self):
        self._temporary.cleanup()

    def get(self, condition, n):
        if n < 1:
            raise ValueError("Positive rejection request required")
        selected, count, extent = [], 0, 0
        i = 0
        while count < n and extent < REJECTION_MAX_CANDIDATES:
            if i == len(self.chunks):
                size = min(self.chunk_size, REJECTION_MAX_CANDIDATES - extent)
                seed = request_seed(self.seed, i, "rejection-chunk")
                path = self.root / f"chunk-{i:08d}.parquet"
                frame, diagnostics = self.sampler(size, seed)
                if len(frame) != size:
                    raise ValueError("Unconditional sampler returned a short chunk")
                atomic_write(path, lambda p: frame.to_parquet(p, index=False))
                self.chunks.append(dict(file=path.name, diagnostics=diagnostics))
            chunk = self.chunks[i]
            path = self.root / chunk["file"]
            frame = pd.read_parquet(path).iloc[: REJECTION_MAX_CANDIDATES - extent]
            hits = frame.loc[condition.mask(frame)]
            selected.append(hits)
            count += len(hits)
            extent += len(frame)
            i += 1
        result = pd.concat(selected, ignore_index=True).iloc[:n].copy()
        recoveries = [c["diagnostics"]["recovery"] for c in self.chunks[:i]]
        recovery = dict(policy=dict(RECOVERY_POLICY), scope="pool_extent", budget_scope="chunk")
        for field in ("attempted_rows", "discarded_rows", "replacement_rows", "retry_rounds"):
            recovery[field] = sum(r[field] for r in recoveries)
        recovery["max_retry_rounds_used"] = max(r["max_retry_rounds_used"] for r in recoveries)
        return result, dict(
            requested=n,
            returned=len(result),
            pool_extent=extent,
            chunks_used=i,
            candidate_budget=REJECTION_MAX_CANDIDATES,
            budget_exhausted=len(result) < n,
            recovery=recovery,
        )
