"""Borrowed view of the reusable native attention outputs."""

from __future__ import annotations

import weakref

import torch


class DecodeResult:
    """Four native slot tensors, valid until the next public run or close."""

    def __init__(self, owner, generation):
        self._owner_ref = weakref.ref(owner)
        self._generation = generation

    def _owner(self):
        owner = self._owner_ref()
        if owner is None or owner._closed or owner._generation != self._generation:
            raise RuntimeError(
                "SeoulKV4 result is no longer live; read back before the next run or close"
            )
        return owner

    @property
    def device_tensors(self) -> tuple:
        """Native layout per KV head: [1, 32, 4, 256]. Do not retain across close."""
        return tuple(self._owner()._require_initialized()["selector"]["outputs"])

    def to_torch(self) -> torch.Tensor:
        """Synchronize and copy the head diagonal into a CPU [1,1,16,256] tensor."""
        owner = self._owner()
        owner.synchronize()
        groups = []
        for item in self.device_tensors:
            rows = owner._tt.to_torch(item)[0]
            groups.append(torch.stack([rows[g, g] for g in range(4)]))
        return torch.cat(groups, dim=0).reshape(1, 1, 16, 256).contiguous()
