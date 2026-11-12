"""Production fused refinement, using the validated tensor/accessor ABI.

The lower-level builders supply tensor lifetimes/runtime accessors, not
dispatched split programs. No rendered C++ is inspected or transformed.
"""

from seoulkv4._kernels.codegen import refinement_fusion as codegen
from seoulkv4._kernels.stage16.fusion_abi import FusionCB as CB
from seoulkv4._kernels.stage16.fusion_abi import FusionSemaphore as SEM


def combine(tt, a, b, *, n16, context):
    local = n16 // 4
    if local > 2048 or n16 % 4:
        raise ValueError("fused refinement requires four <=2048-entry shards")
    if a["worker_groups"] != b["worker_groups"]:
        raise ValueError("refinement topology mismatch")
    ids = max(1, (context - 1).bit_length())
    quarter = (((local + 31) // 32) + 3) // 4
    old = a["descriptor"].kernels
    cores = old[0].core_ranges
    specs = []

    def cb(index, dtype, size, pages=1):
        specs.append(
            dict(id=int(index), format=str(dtype), page_size=size, pages=pages)
        )
        return tt.CBDescriptor(
            total_size=size * pages,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(
                    buffer_index=int(index), data_format=dtype, page_size=size
                )
            ],
        )

    cbs = [
        cb(CB.FRONTIER_INPUT, tt.uint8, 8192),
        cb(CB.DECODED_KEYS, tt.bfloat16, 2048, 16),
        cb(CB.STAGE16_SCORE_STAGING, tt.uint8, ((quarter * 256 + 63) // 64) * 64),
        cb(CB.TOPK_SECOND_INPUT, tt.bfloat16, 2048),
        cb(CB.COPY_WITNESS, tt.bfloat16, 2048, 2),
        cb(CB.UNION_META, tt.uint32, 4096),
        cb(CB.STAGE2_QUERY, tt.bfloat16, 2048, 16),
        cb(CB.STAGE2_FRONTIER, tt.uint32, 4096, 2),
        cb(CB.STAGE16_FRONTIER, tt.float32, 4096, 2),
        cb(CB.LIVE_CONTROL, tt.bfloat16, 16, 2),
        cb(CB.STAGE16_MERGE, tt.uint32, 4096, 4),
        cb(CB.MATRIX_QUERY, tt.bfloat16, 2048, 8),
        cb(CB.DECODE_SCRATCH, tt.bfloat16, 2048, 8),
        cb(CB.PACKED_CANDIDATES, tt.uint32, 4096, 6),
        cb(CB.PACKED_MIDDLE, tt.uint32, 4096, quarter),
        cb(CB.SOURCE_IDS, tt.uint32, ((local * 4 + 190) // 64) * 64),
        cb(CB.SCORE_TILES, tt.bfloat16, 2048, max(quarter, 5)),
        cb(CB.STAGE2_EXCHANGE, tt.uint32, 4096),
        cb(CB.SELECTOR_INPUT, tt.bfloat16, 4096),
        cb(CB.WRITER_SCRATCH, tt.uint8, 16384),
    ]
    kernels = []
    for kind, donor in zip(
        ("reader", "compute", "writer"), (old[0], old[1], b["descriptor"].kernels[2])
    ):
        kernels.append(
            tt.KernelDescriptor(
                kernel_source=getattr(codegen, kind)(local=local, id_bits=ids),
                source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=list(donor.compile_time_args),
                runtime_args=donor.runtime_args,
                config=donor.config,
            )
        )
    descriptor = tt.ProgramDescriptor(
        kernels=kernels,
        cbs=cbs,
        semaphores=[
            tt.SemaphoreDescriptor(id=int(sem), core_ranges=cores, initial_value=0)
            for sem in SEM
        ],
    )
    io, seen = [], set()
    for tensor in (*a["io"], *b["io"]):
        if id(tensor) not in seen:
            io.append(tensor)
            seen.add(id(tensor))

    def run():
        tt.generic_op(io, descriptor)
        return b["outputs"]

    return dict(
        b,
        run=run,
        descriptor=descriptor,
        io=io,
        stage16_dynamic_input_count=True,
        stage16_schedule=a["stage16_schedule"],
        fusion_cb_specs=specs,
        fusion_cb_bytes=sum(s["page_size"] * s["pages"] for s in specs),
    )
