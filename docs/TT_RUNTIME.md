# Blackhole runtime setup

SeoulKV4 is validated against TTNN/TT-Metal 0.77.0 on Blackhole. Set up the
Tenstorrent environment before calling `SeoulKV4.run(q)`. It must provide the
matching SFPI compiler and runtime source tree, including LLK, the TopK-XL
API, and the TTNN sparse-SDPA device kernels. Installing SeoulKV4 does not
install TTNN or set up a device. The index API only needs PyTorch and NumPy.

The 0.77.0 TopK-XL header needs the
[upstream SFPLOAD fix](https://github.com/tenstorrent/tt-metal/commit/73a7285cab19a193e60c4db4b95a6cf369d7c2da)
for the validated configuration. The compatible one-line backport is included
in the source distribution and wheel. If your runtime does not already
contain the fix, apply it to the TT-Metal tree used by the JIT.

For an installed TTNN wheel, use the Python environment that will run
SeoulKV4:

```bash
tt_runtime="$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(Path(find_spec("ttnn").origin).parent)')"
topk_patch="$(python -c 'from importlib.resources import files; print(files("seoulkv4").joinpath("_compat/tt-metal-v0.77-topk-xl-sfpload.patch"))')"

patch --dry-run -d "$tt_runtime" -p1 < "$topk_patch" &&
patch -d "$tt_runtime" -p1 < "$topk_patch"
```

For a TT-Metal source installation, point `tt_runtime` at the TT-Metal source
root used by the JIT instead. If the dry run reports an already applied or
mismatched patch, check the header revision before proceeding. Use a fresh
JIT cache when validating a changed header.

For a runtime in a custom location, set `TT_METAL_RUNTIME_ROOT` to the
directory containing its `tt_metal/` and `ttnn/` source trees before
importing TTNN. A standard TTNN wheel sets this location automatically.

Open a single-board TTNN MeshDevice and pass it to `SeoulKV4`. The 128K
configuration uses 1,355,904 bytes of worker L1 and 16 MiB of trace storage
when tracing. The optional selected-ID diagnostics were validated with
1,433,728 bytes of worker L1. SeoulKV4 does not change the partition or
close the device.
