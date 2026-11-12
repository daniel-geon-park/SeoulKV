"""Build or load a grouped Stage-I index without importing TTNN."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

import torch

from seoulkv4._codecs.index import (
    QUERY_FIELDS,
    dump_index,
    fingerprint,
    load_index,
    require_group_scale_cell,
)
from seoulkv4._index_build import build_stage1_cell, centroid_count
from seoulkv4._kernels.common.attention_zones import AttentionZones
from seoulkv4.config import _KernelOptions


class SeoulKV4Index:
    """Query-independent CPU index for four ordered KV heads."""

    def __init__(self, cells, metadata):
        if len(cells) != 4:
            raise ValueError("SeoulKV4 requires four ordered index cells")
        self._zones = AttentionZones.from_fixture(
            cells, context=int(cells[0]["context"])
        )
        self._zones.count()
        self._cells = tuple(
            {
                name: value
                for name, value in require_group_scale_cell(cell).items()
                if name not in QUERY_FIELDS
            }
            for cell in cells
        )
        self._nprobe_source = metadata.get("nprobe_source")

    @classmethod
    def load(cls, path: str | Path) -> SeoulKV4Index:
        """Read a saved index or a grouped Stage-I fixture, discarding legacy Q."""
        cells, metadata = load_index(path)
        return cls(cells, metadata)

    def dump(self, path: str | Path) -> None:
        """Save the static index as an NPZ that `load()` can reopen."""
        dump_index(path, self._cells, {"nprobe_source": self._nprobe_source})

    @classmethod
    def build(
        cls,
        k: torch.Tensor,
        v: torch.Tensor | None = None,
        *,
        sink: int = 128,
        local: int = 256,
        nprobe: int | Sequence[int] = 384,
        lam: float = 0.0625,
        seed: int = 0,
        build_device: str | torch.device = "cpu",
    ) -> SeoulKV4Index:
        """Index CPU BF16 post-RoPE K [4,T,256] with the existing Stage-I codec.

        V, when supplied, is checked for matching layout and dtype. Only K
        contributes to the index. PyTorch may cluster on CPU or CUDA; the
        returned index is always CPU-resident and contains no query or V data.
        """
        if not (
            isinstance(k, torch.Tensor)
            and k.device.type == "cpu"
            and k.dtype == torch.bfloat16
            and k.ndim == 3
            and k.shape[0] == 4
            and k.shape[2] == 256
        ):
            raise ValueError("expected CPU BF16 post-RoPE K [4,T,256]")
        if v is not None and not (
            isinstance(v, torch.Tensor)
            and v.device == k.device
            and v.dtype == k.dtype
            and v.shape == k.shape
        ):
            raise ValueError("V must match K's shape, dtype, and CPU device")
        context = k.shape[1]
        if not 6400 <= context <= 524288 or context % 4:
            raise ValueError("unsupported context for the fixed SeoulKV4 profile")
        if type(sink) is not int or type(local) is not int:
            raise ValueError("sink and local must be integer token counts")
        zones = AttentionZones(context, sink, local)
        retrieval_tokens = zones.local_begin - sink
        if retrieval_tokens < _KernelOptions.skv4_stage15_m:
            raise ValueError(
                f"retrieval zone needs at least {_KernelOptions.skv4_stage15_m} keys "
                "for the fixed Stage-1.5 profile"
            )
        nc = centroid_count(retrieval_tokens)
        if nc > 65536:
            raise ValueError(
                "centroid count exceeds the Fine selector's 16-bit positions"
            )
        if type(nprobe) is int:
            probes = (nprobe,) * 4
        elif isinstance(nprobe, Sequence) and not isinstance(nprobe, (str, bytes)):
            probes = tuple(nprobe)
        else:
            raise ValueError("nprobe must be an integer or four per-head integers")
        if len(probes) != 4 or any(
            type(p) is not int or not 16 <= p <= min(512, nc) or p % 16 for p in probes
        ):
            raise ValueError(
                f"each nprobe must be a multiple of 16 in [16, {min(512, nc)}]"
            )
        if (
            isinstance(lam, bool)
            or not isinstance(lam, Real)
            or not (math.isfinite(lam) and lam >= 0)
        ):
            raise ValueError("lam must be finite and nonnegative")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be a nonnegative signed 64-bit integer")
        device = torch.device(build_device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("build_device must be a PyTorch CPU or CUDA device")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "build_device requests CUDA, but PyTorch CUDA is unavailable"
            )
        cells = [
            build_stage1_cell(
                k[head],
                p=p,
                lam=float(lam),
                seed=seed,
                r0=sink,
                r1=zones.local_begin,
                device=device,
            )
            for head, p in enumerate(probes)
        ]
        return cls(cells, {"nprobe_source": "explicit (build)"})

    def _check_keys(self, k):
        for i, cell in enumerate(self._cells):
            if fingerprint(k[i]) != cell["key_hash"]:
                raise ValueError(f"Stage-I index K fingerprint mismatch at cell {i}")

    def _live_cells(self, q):
        """Private per-instance descriptors; the selector receives live device tags."""
        queries = q.reshape(4, 4, 256)
        return [
            dict(
                cell,
                q=queries[i],
                fine_scoring="post-scale-bf16",
                # Existing dynamic Fine builder needs a nonempty shape witness,
                # but never uploads these IDs when external tags are present.
                coarse_ids=torch.arange(int(cell["p"]), dtype=torch.int32),
                head_mask=torch.full((int(cell["p"]),), 15, dtype=torch.uint8),
            )
            for i, cell in enumerate(self._cells)
        ]

    def metadata(self) -> dict:
        return {
            **self._zones.metadata(),
            "summary_codec": "group-scale-32-v1",
            "nprobe": [int(cell["p"]) for cell in self._cells],
            "lambda": [float(cell["lam"]) for cell in self._cells],
            "key_hashes": [cell["key_hash"] for cell in self._cells],
            "nprobe_source": self._nprobe_source,
        }
