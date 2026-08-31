"""Small inference-only compatibility layer for PSICHIC's mean scatter calls.

PSICHIC uses only ``scatter(..., dim=0, reduce='mean')``. Modern PyTorch has
the required native operations, avoiding an ABI-specific torch-scatter wheel.
"""

from __future__ import annotations

import torch


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("PSICHIC compatibility scatter supports dim=0 only")
    size = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
    if out is None:
        out = src.new_zeros((size,) + src.shape[1:])
    expanded_index = index.reshape((-1,) + (1,) * (src.ndim - 1)).expand_as(src)
    if reduce in ("sum", "add"):
        return out.scatter_add_(0, expanded_index, src)
    if reduce == "mean":
        out.scatter_add_(0, expanded_index, src)
        counts = src.new_zeros(size)
        counts.scatter_add_(0, index, src.new_ones(index.shape[0]))
        return out / counts.clamp_min(1).reshape((-1,) + (1,) * (src.ndim - 1))
    if reduce in ("min", "max"):
        initial = torch.inf if reduce == "min" else -torch.inf
        result = torch.full_like(out, initial)
        return result.scatter_reduce_(0, expanded_index, src, reduce="amin" if reduce == "min" else "amax", include_self=True)
    raise ValueError(f"unsupported scatter reduction: {reduce}")


__version__ = "compat"
