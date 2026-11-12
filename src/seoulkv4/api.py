"""Public SeoulKV4 lifecycle; the caller owns the Tenstorrent device."""

from __future__ import annotations

import torch

from seoulkv4 import _runtime
from seoulkv4.config import SeoulKV4Config, _KernelOptions
from seoulkv4.index import SeoulKV4Index
from seoulkv4.result import DecodeResult


class SeoulKV4:
    """Static-cache PoC. Call run with a fresh CPU BF16 post-RoPE query."""

    def __init__(
        self,
        *,
        device,
        k: torch.Tensor,
        v: torch.Tensor,
        index: SeoulKV4Index,
        config: SeoulKV4Config | None = None,
    ):
        self._config = SeoulKV4Config() if config is None else config
        if not isinstance(self._config, SeoulKV4Config) or not isinstance(
            index, SeoulKV4Index
        ):
            raise TypeError("expected SeoulKV4Config and SeoulKV4Index")
        if device is None:
            raise ValueError("supply a caller-owned TTNN MeshDevice")
        context = index._zones.context
        if not 6400 <= context <= 524288 or context % 4:
            raise ValueError(
                "unsupported context for the fixed SeoulKV4 candidate/tag profile"
            )
        for name, value in (("K", k), ("V", v)):
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.bfloat16
                or tuple(value.shape) != (4, context, 256)
            ):
                raise ValueError(f"expected CPU BF16 {name} [4,{context},256]")
        index._check_keys(k)
        self._device = device
        self._index = index
        self._host_kv = (k, v)
        self._args = _KernelOptions(
            context, self._config.stage15_capacity, self._config.debug
        )
        self._state = None
        self._tt = None
        self._generation = 0
        self._closed = False

    @staticmethod
    def _validate_query(q):
        if (
            not isinstance(q, torch.Tensor)
            or q.device.type != "cpu"
            or q.dtype != torch.bfloat16
            or tuple(q.shape) != (1, 1, 16, 256)
        ):
            raise ValueError("expected CPU BF16 Q [1,1,16,256]")

    def _require_open(self):
        if self._closed:
            raise RuntimeError("SeoulKV4 is closed")

    def _require_initialized(self):
        self._require_open()
        if self._state is None:
            raise RuntimeError("run(q) must initialize SeoulKV4 first")
        return self._state

    @property
    def initialized(self) -> bool:
        return self._state is not None

    def _bind_query(self, q):
        # Private benchmark trace bridge: upload stays outside capture/replay.
        self._validate_query(q)
        _runtime.bind_query(self._require_initialized(), q, self._args)

    def _run_bound(self):
        _runtime.enqueue(self._require_initialized())
        self._generation += 1
        return DecodeResult(self, self._generation)

    def run(self, q: torch.Tensor) -> DecodeResult:
        """Build on the first invocation, then bind Q and enqueue without readback."""
        self._require_open()
        self._validate_query(q)
        if self._state is None:
            self._tt = _runtime._ttnn()
            self._state = _runtime.initialize(
                self._tt, self._device, *self._host_kv, q, self._index, self._args
            )
            self._host_kv = None
        self._bind_query(q)
        return self._run_bound()

    def synchronize(self) -> None:
        self._require_open()
        if self._tt is not None:
            self._tt.synchronize_device(self._device)

    def metadata(self) -> dict:
        """Host-only configuration and program metadata; never reads device tensors."""
        self._require_open()
        return {
            **(_runtime.metadata(self._state) if self.initialized else {}),
            "initialized": self.initialized,
            "cache_layout": "flat",
            "index": self._index.metadata(),
        }

    def debug_snapshot(self) -> dict:
        """Read the latest logical final IDs from the explicit debug buffer."""
        state = self._require_initialized()
        if not self._config.debug:
            raise RuntimeError("debug_snapshot requires SeoulKV4Config(debug=True)")
        self.synchronize()
        ids = self._tt.to_torch(state["selector"]["debug_indices"])
        return {"selected_ids": ids.reshape(4, 4, -1)[..., :100].long()}

    def close(self) -> None:
        """Release this instance's references after pending work; never close the device."""
        if self._closed:
            return
        try:
            self.synchronize()
        finally:
            self._closed = True
            self._state = self._host_kv = self._index = None
            self._tt = self._device = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
