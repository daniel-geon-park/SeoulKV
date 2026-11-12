# SeoulKV4 native kernels

This directory contains the Blackhole TT-Metal source builders used by the
SeoulKV4 library:

- `common/`: layouts, measured defaults, and native marker helpers;
- `topk/`: reusable TopK and payload-merge primitives;
- `query_prepare/`: one raw GDDR Q upload and device-prepared persistent L1 operands;
- `stage1_coarse/`: Coarse centroid interval scoring and selection;
- `stage1_fine/`: merged Stage-I fine scoring, selection, and replay;
- `stage15/`: Stage-1.5 ingestion, scoring, TopK, union, and handoff;
- `stage16/`: Stage-1.6 scoring, dynamic scheduling, and union chain;
- `stage2/`: final scoring and Stage-II/attention boundary helpers.

Use package imports for the Python builders. Generated C++ source may cross
stage boundaries, but each stage owns its Python modules.

## ABI ownership

Each stage's `abi.py` owns its argument layouts, CB IDs and semaphore IDs.
Host descriptors and C++ emitters import that same specification directly.
Stage1.5 also owns its standalone/fused ingest contracts. The standalone
sharded Stage-II selector semaphore lives in `stage2/abi.py`.

The common modules contain only genuinely shared definitions:

- `common/argument_abi.py`: `Field`, `Layout`, `CompileABI` and packing tools;
- `common/cb_layout.py`, `common/semaphore_layout.py`: enum-to-C++ emitters;
- `common/selection_abi.py`: shared Coarse/Fine selector argument contracts;
- `common/refinement_abi.py`: the shared Stage1.6/II/attention argument, CB
  and semaphore ABI, including intentionally reused slots and the upstream
  sparse-SDPA prefix.

ABI modules depend only on common specifications and the standard library;
they do not import host builders or codegen.

## Query-per-invocation dataflow

The host binding uploads one BF16 `[1,1,16,256]` Q, stored physically as four
2-KiB GDDR pages. QueryPrepare's four owners generate Coarse/Fine/Stage1.5/
refinement operands, FP32 norms and compact original Q in persistent L1.
No expanded query tensor is written to GDDR. The norm uses SFPU FP32 squares,
reduction and sqrt; the rejected tensor-FPU norm witness is not a production
branch. Original BF16 rounding points are preserved before stage-dependent
power-of-two scaling. Attention reads one512-B compact row/head and builds
its padded diagonal CB locally.

The four programs are fused QueryPrepare/Coarse, Fine, Stage1.5, and fused
refinement/attention.

## Sink/local attention

Final attention includes the index's sink and local ranges in addition to
the retrieval top 100. Pass `sink` and `local` when building the index to use
ranges other than the 128/256 defaults.

The clean attention workers stream dense sink/local chunks before waiting
for retrieval IDs. All chunks share one online softmax; only the last chunk
is padded/masked. The 128-key buffer and four-program schedule are unchanged.
The named streaming ABI lives in `stage16/attention_abi.py`, with explicit
reader/helper generators in `codegen/attention.py`.
