# SeoulKV4 proof-of-concept API

SeoulKV4 runs sparse decode attention over a fixed, flat K/V cache on
Tenstorrent Blackhole. The current profile supports one request, one query
token, four KV heads, and 16 query heads of dimension 256. Paged attention,
cache updates, and request scheduling are outside this API.

| Public type | Purpose |
|---|---|
| `SeoulKV4` | Manages the device workspace for one cache and runs the attention pipeline. |
| `SeoulKV4Config` | Configures candidate capacity and optional diagnostics. |
| `SeoulKV4Index` | Builds, saves, and loads the query-independent Stage I index. |
| `DecodeResult` | Exposes the native device outputs and an explicit CPU readback. |

Each `SeoulKV4` instance serves one cache snapshot. Concurrent calls or
interleaved execution of multiple instances on the same device are not
supported.

## Quick start

```python
from seoulkv4 import SeoulKV4, SeoulKV4Config, SeoulKV4Index

# device: a single-board TTNN MeshDevice opened by the caller.
# k and v: CPU BF16 tensors of shape [4, T, 256]. K is already post-RoPE.
# q0 and q1: post-RoPE CPU BF16 tensors of shape [1, 1, 16, 256].
index = SeoulKV4Index.build(k, v)
# To reuse an index, save it with index.dump("stage1_index.npz") and
# reopen it with SeoulKV4Index.load("stage1_index.npz").

with SeoulKV4(device=device, k=k, v=v, index=index,
              config=SeoulKV4Config()) as attention:
    attention.run(q0)                 # Builds the workspace on first use.
    attention.synchronize()
    y = attention.run(q1).to_torch()   # CPU BF16 [1, 1, 16, 256].
```

The first call also returns a result. The example uses that call for
initialization and reads the result of the next query. See [runtime setup](TT_RUNTIME.md)
for the device configuration and TT-Metal requirements.

## `SeoulKV4Index`

```python
class SeoulKV4Index:
    @classmethod
    def build(cls, k, v=None, *, sink=128, local=256, nprobe=384,
              lam=0.0625, seed=0, build_device="cpu") -> "SeoulKV4Index": ...

    @classmethod
    def load(cls, path: str | Path) -> "SeoulKV4Index": ...

    def dump(self, path: str | Path) -> None: ...

    def metadata(self) -> dict: ...
```

- `build()` clusters post-RoPE K in the retrieval range for each KV head.
  The index uses CSR posting lists with two memberships per key and one
  quantization scale per 32 centroids. Pass `build_device="cuda:0"` to cluster
  with PyTorch on CUDA; the returned index is always on the CPU. V is optional
  and is checked only for matching shape, dtype, and device. Building an index
  requires no query or TT device.
- `nprobe` is the number of centroids selected per query head in Stage I Fine.
  Pass one value for all KV groups or four values in KV head order. Each value
  must be a multiple of 16 between 16 and 512 and cannot exceed the centroid
  count. The first execution checks that the resulting key pool fits the
  Stage 1.5 capacity.
- `dump(path)` saves a static NPZ (version 4). It contains the context, head
  order, sink and local ranges, centroid codes and scales, posting lists,
  selection settings, and key fingerprints. The file does not contain the
  original K/V tensors, queries, replay candidates, or device addresses.
- `load(path)` also accepts older grouped Stage I fixtures (version 3).
  It discards their diagnostic query data. Loading either format does not
  rerun clustering.
- `metadata()` returns the index settings and key fingerprints. The constructor
  checks that K matches the index in context and head order. Each execution
  keeps its query-dependent state separate from the index. To change the
  cache or its sink and local ranges, build or load a matching index.

## `SeoulKV4Config`

```python
@dataclass(frozen=True)
class SeoulKV4Config:
    stage15_capacity: int = 13056
    debug: bool = False
```

`stage15_capacity` sets the number of key slots available per KV group in
Stage 1.5. It must be a multiple of 768 between 12,288 and 24,576. If the live
pool exceeds the configured capacity, the runtime reports an error instead
of truncating candidates.

The current profile uses six Stage 1.5 scorer shards, a top-2,048 frontier per
query head with the first 1,600 entries entering the group union, 400 survivors
per query head in Stage 1.6 with a local limit of 160 per shard, and 100 final
retrieval keys per query head. These budgets and the kernel layouts are fixed.
The index supplies `nprobe`, `lam`, and the sink and local ranges.

`debug=True` enables extra buffers for diagnostics. The 128K debug profile
was validated with 1,433,728 bytes of worker L1 per core. The caller sets the
L1 partition and trace region when opening the device.

## `SeoulKV4`

```python
class SeoulKV4:
    def __init__(
        self, *, device, k, v, index: SeoulKV4Index,
        config: SeoulKV4Config | None = None,
    ): ...

    def run(self, q: torch.Tensor) -> "DecodeResult": ...
    def synchronize(self) -> None: ...
    def metadata(self) -> dict: ...
    def debug_snapshot(self) -> dict: ...
    def close(self) -> None: ...

    @property
    def initialized(self) -> bool: ...
```

**Inputs and ownership.** The caller opens and owns the TTNN MeshDevice.
K and V are CPU BF16 tensors of shape `[4, T, 256]`. Each `run(q)` takes a
CPU BF16 tensor of shape `[1, 1, 16, 256]`. Q and K must already have RoPE
applied. Model projections, Q/K normalization, RoPE, and output gating are
the caller's responsibility. Device-resident Q is not supported.

**First call.** The constructor validates and retains the inputs. The first
`run(q)` uses the supplied Q to initialize the workspace and run the
pipeline. This call may compile kernels, allocate buffers, and read back
initialization data. Keep it outside trace capture and latency measurements.
`initialized` indicates whether the workspace is ready.

**Later calls.** Each `run(q)` writes Q into the same GDDR buffer and enqueues
`query_prepare_coarse → fine → stage15 → refinement_attention`. It reuses the
device workspace and returns without reading back the result or synchronizing
the device. Call `synchronize()` to wait for the work to finish.

**Inspection.** `metadata()` returns host-side information such as the index
configuration and program list without a device readback. With `debug=True`,
`debug_snapshot()` synchronizes and reads the latest selected token IDs as
`{"selected_ids": tensor}` with shape `[4, 4, 100]` in KV group, query head,
and retrieval rank order. Keep this call outside timed invocations.

**Cleanup.** `close()` waits for pending work and releases the instance's
resources. It is safe to call more than once, and leaving a `with` block calls
it automatically. Release external traces and borrowed tensor handles before
closing the instance. The device and its TTNN program cache remain under
the caller's control.

## `DecodeResult`

```python
class DecodeResult:
    @property
    def device_tensors(self) -> tuple: ...

    def to_torch(self) -> torch.Tensor: ...
```

`device_tensors` returns four native output tensors, one per KV group. Each
has shape `[1, 32, 4, 256]`. For a query head with group-local index `h`,
its result is at `tensor[0, h, h]`. The library does not assemble these
outputs into a single device tensor.

`to_torch()` waits for execution and reads the head diagonals into a CPU BF16
tensor of shape `[1, 1, 16, 256]`. The CPU tensor can be kept independently.
The device outputs borrow reusable storage and remain valid only until the
next run or close. Read back or copy the output before then if needed.

## Trace scope

SeoulKV4 does not create or manage traces. The first `run(q)` must happen
outside capture, and any externally managed trace must be released before
closing the instance. Direct binding of device-resident Q and model-level
trace integration are outside the public API.

## Source layout

```text
src/seoulkv4/
  __init__.py      # Public exports; TTNN is loaded on the first run.
  api.py           # Workspace lifecycle and execution.
  config.py        # Runtime settings.
  index.py         # Build, save, and load static indexes.
  result.py        # Borrowed device outputs and CPU readback.
  _clustering.py   # PyTorch IVF clustering.
  _runtime.py      # Device program assembly and dispatch.
  _codecs/         # Key encoding and index serialization.
  _kernels/        # TT-Metal kernel source generators.
  _compat/         # TopK-XL compatibility patch.
```

Modules prefixed with an underscore are internal. The Tenstorrent headers
and compiler used for JIT compilation come from the separately installed
[runtime](TT_RUNTIME.md).
