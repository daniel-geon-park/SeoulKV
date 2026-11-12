"""Stable GDDR Q input and persistent L1 device-prepared operands."""

from __future__ import annotations

import struct

import torch

from seoulkv4._kernels.codegen.query_prepare import emit
from seoulkv4._kernels.query_prepare import abi
from seoulkv4._kernels.common.query_metadata import SUM_WORD_OFFSET
from seoulkv4._kernels.common.core_ranges import core_ranges
from seoulkv4._kernels.stage1_coarse.coarse_native_codec import fine_coordinate_order
from seoulkv4._kernels.stage15.stage15_sfpu_native import coordinate_order


def prepack_scale_tiles(scales):
    """Immutable FP32 tile bytes, one contiguous 32-KiB page per GQA group."""
    tiles = torch.zeros(4, 8, 4, 16, 16, dtype=torch.float32)
    # TT tile faces: top-left, top-right, bottom-left, bottom-right.
    rows = scales.float().reshape(4, 8, 2, 16)
    tiles[:, :, :2, :4, :] = rows[:, :, :, None, :]
    return tiles.contiguous().view(torch.uint8).reshape(4, 32768)


def reference(q, scales, lambdas):
    """Independent host operands used only by offline/readback checks."""
    qg = q.reshape(4, 4, 256).bfloat16()
    coarse, fine = [], []
    for group in qg:
        qc = torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)
        qf = torch.zeros_like(qc)
        qc[0, 0, :4] = group[:, coordinate_order()]
        qf[0, 0, :4] = group[:, fine_coordinate_order()]
        coarse.append(qc)
        fine.append(qf)
    from seoulkv4._kernels.stage15.seoulkv4_stage15_matrix import prepare_matrix_query
    from seoulkv4._kernels.stage16.seoulkv4_union_matrix import matrix_query

    q15 = (qg.float() * scales[:, None] / 64).bfloat16()
    qs = (qg.float() * scales[:, None] / 256).bfloat16()
    norms = qg.float().norm(dim=-1) * torch.as_tensor(lambdas)[:, None]
    return dict(
        coarse=coarse,
        fine=fine,
        norm=norms,
        query_sum=qg.float().sum(dim=-1),
        stage15=prepare_matrix_query(q15),
        matrix=matrix_query(qs),
        compact=q.reshape(1, 1, 16, 256).bfloat16(),
    )


def build(
    tt,
    device,
    scales,
    lambdas,
    *,
    profile=False,
    core_points=None,
    consumer_groups=None,
    consumer_ready_semaphore=0,
):
    if tuple(scales.shape) != (4, 256) or len(lambdas) != 4:
        raise ValueError("QueryPrepare supports GQA4x4, dimension256 only")
    if not torch.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("positive finite key scales required")

    def allocate(shape, *, tiled=False, dram=False, dtype=None):
        return tt.allocate_tensor_on_device(
            tt.Shape(shape),
            tt.bfloat16 if dtype is None else dtype,
            tt.TILE_LAYOUT if tiled else tt.ROW_MAJOR_LAYOUT,
            device,
            tt.DRAM_MEMORY_CONFIG if dram else tt.L1_MEMORY_CONFIG,
        )

    # A physical page is a whole four-head group, not one head. The public Q
    # view is unchanged; this storage-only reshape makes the bulk read legal.
    raw = allocate((1, 1, 4, 1024), dram=True)
    scale = tt.from_torch(
        prepack_scale_tiles(scales),
        dtype=tt.uint8,
        layout=tt.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=tt.L1_MEMORY_CONFIG,
    )
    zeros = tt.from_torch(
        torch.zeros(1, abi.SCRATCH_BYTES, dtype=torch.uint8),
        dtype=tt.uint8,
        layout=tt.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=tt.L1_MEMORY_CONFIG,
    )
    coarse = [allocate((1, 1, 32, 256), tiled=True) for _ in range(4)]
    fine = [allocate((1, 1, 32, 256), tiled=True) for _ in range(4)]
    norm = [allocate((1, 64), dtype=tt.uint8) for _ in range(4)]
    stage15 = allocate((1, 4, 32, 256), tiled=True)
    matrix = allocate((1, 4, 64, 256), tiled=True)
    compact = allocate((1, 1, 16, 256))
    grid = device.compute_with_storage_grid_size()
    writer_count = 4 * abi.WRITERS_PER_GROUP
    points = (
        [(x, y) for y in range(grid.y) for x in range(grid.x)][:writer_count]
        if core_points is None
        else list(core_points)
    )
    if len(points) != writer_count or len(set(points)) != writer_count:
        raise ValueError(f"QueryPrepare requires {writer_count} writer cores")
    if any(not (0 <= x < grid.x and 0 <= y < grid.y) for x, y in points):
        raise ValueError("QueryPrepare core outside worker grid")
    consumers = [[] for _ in range(4)] if consumer_groups is None else consumer_groups
    if len(consumers) != 4 or any(len(g) > abi.MAX_COARSE_CONSUMERS for g in consumers):
        raise ValueError("one bounded Coarse consumer group per Q group required")
    consumer_points = [point for group in consumers for point in group]
    if len(set(consumer_points)) != len(consumer_points) or set(points) & set(
        consumer_points
    ):
        raise ValueError("QueryPrepare and Coarse consumers must use disjoint cores")
    owners = points[:4]
    groups = [[owners[g], points[4 + g]] for g in range(4)]
    physical = {
        point: device.worker_core_from_logical_core(tt.CoreCoord(*point))
        for point in points + consumer_points
    }
    launch_groups = {}
    sources = {
        role: emit(
            role=role,
            profile=profile,
            coarse_order=coordinate_order(),
            fine_order=fine_coordinate_order(),
            matrix_order=torch.arange(256).reshape(32, 8).T.flatten(),
        )
        for role in ("reader", "compute", "writer")
    }
    cb = abi.QueryPrepareCB
    all_cores = core_ranges(tt, points)
    owner_cores = core_ranges(tt, owners)
    cbs = [
        tt.CBDescriptor(
            total_size=page * count,
            core_ranges=cores,
            format_descriptors=[
                tt.CBFormatDescriptor(
                    buffer_index=int(index), data_format=dtype, page_size=page
                )
            ],
        )
        # Common CBs come first and have identical core sets: peer copies use
        # the same L1 addresses. Owner-only allocations cannot disturb these.
        for index, dtype, page, count, cores in (
            (cb.RAW_QUERY, tt.uint8, 2048, 1, all_cores),
            (cb.PRODUCT_TILES, tt.float32, 4096, 8, all_cores),
            (cb.OUTPUT_SCRATCH, tt.uint8, abi.SCRATCH_BYTES, 1, all_cores),
            (cb.QUERY_TILES, tt.bfloat16, 2048, 8, owner_cores),
            (cb.SCALE_TILES, tt.float32, 4096, 8, owner_cores),
            (cb.NORM_TILE, tt.float32, 4096, 2, owner_cores),
        )
    ]
    for group, (x, y) in enumerate(owners):
        tensors = dict(
            coarse=coarse[group],
            fine=fine[group],
            norm=norm[group],
            stage15=stage15,
            matrix=matrix,
            compact=compact,
            zeros=zeros,
        )
        input_tensors = dict(raw_query=raw, scales=scale, zeros=zeros)
        peers = [
            coord
            for point in groups[group][1:]
            for coord in (physical[point].x, physical[point].y)
        ]
        owner_fields = dict(
            owner_x=physical[(x, y)].x,
            owner_y=physical[(x, y)].y,
            consumer_count=len(consumers[group]),
            consumers=[
                coord
                for point in consumers[group]
                for coord in (physical[point].x, physical[point].y)
            ]
            + [0] * (2 * (abi.MAX_COARSE_CONSUMERS - len(consumers[group]))),
        )
        configs = (
            (
                "reader",
                abi.READER_RT,
                abi.READER_CT,
                input_tensors,
                dict(group=group),
                {},
                tt.ReaderConfigDescriptor(),
            ),
            (
                "compute",
                None,
                abi.COMPUTE_CT,
                {},
                {},
                dict(
                    lambda_bits=struct.unpack(
                        "<I", struct.pack("<f", float(lambdas[group]))
                    )[0]
                ),
                tt.ComputeConfigDescriptor(
                    math_fidelity=tt.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    dst_full_sync_en=True,
                ),
            ),
            (
                "writer",
                abi.WRITER_RT,
                abi.WRITER_CT,
                tensors,
                dict(group=group, peers=peers, worker=0, **owner_fields),
                dict(consumer_ready_semaphore=consumer_ready_semaphore),
                tt.WriterConfigDescriptor(),
            ),
        )
        launches = [(x, y, entry) for entry in configs]
        for worker, (wx, wy) in enumerate(groups[group][1:], start=1):
            launches.append(
                (
                    wx,
                    wy,
                    (
                        "writer",
                        abi.WRITER_RT,
                        abi.WRITER_CT,
                        tensors,
                        dict(group=group, peers=peers, worker=worker, **owner_fields),
                        dict(consumer_ready_semaphore=consumer_ready_semaphore),
                        tt.WriterConfigDescriptor(),
                    ),
                )
            )
        for x, y, (role, rt, ct, io, fields, compile_fields, config) in launches:
            compile_args = ct.pack(tt, tensors=io, **compile_fields)
            key = (role, tuple(compile_args))
            entry = launch_groups.setdefault(
                key,
                dict(
                    role=role,
                    ct=compile_args,
                    config=config,
                    points=[],
                    rt=tt.RuntimeArgs(),
                ),
            )
            entry["points"].append((x, y))
            entry["rt"][x][y] = (
                []
                if rt is None
                else rt.pack(
                    **{name: t.buffer_address() for name, t in io.items()}, **fields
                )
            )
    # Match Coarse's descriptor grouping: identical source/compile arguments
    # share one kernel across cores. Lambda differences remain distinct groups.
    kernels = [
        tt.KernelDescriptor(
            kernel_source=sources[entry["role"]],
            source_type=tt.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=core_ranges(tt, entry["points"]),
            compile_time_args=entry["ct"],
            runtime_args=entry["rt"],
            config=entry["config"],
        )
        for entry in launch_groups.values()
    ]
    program = tt.ProgramDescriptor(
        kernels=kernels,
        cbs=cbs,
        semaphores=[
            tt.SemaphoreDescriptor(
                id=int(sem),
                core_ranges=all_cores,
                initial_value=0,
            )
            for sem in abi.QueryPrepareSemaphore
        ],
    )
    io = [raw, scale, zeros, *coarse, *fine, *norm, stage15, matrix, compact]
    return dict(
        raw=raw,
        scales=scale,
        zeros=zeros,
        coarse=coarse,
        fine=fine,
        norm=norm,
        stage15=stage15,
        matrix=matrix,
        compact=compact,
        program=program,
        kernel_count=len(kernels),
        core_points=points,
        io=io,
        run=lambda: tt.generic_op(io, program),
    )


def bind(tt, state, q):
    if tuple(q.shape) != (1, 1, 16, 256) or q.dtype != torch.bfloat16:
        raise ValueError("expected CPU BF16 Q [1,1,16,256]")
    if q.device.type != "cpu":
        raise ValueError("the current decodebench input is a host Q tensor")
    host = tt.from_torch(
        q.reshape(1, 1, 4, 1024).contiguous(),
        dtype=tt.bfloat16,
        layout=tt.ROW_MAJOR_LAYOUT,
    )
    tt.copy_host_to_device_tensor(host, state["raw"])


def readback(tt, state):
    metadata = torch.stack(
        [
            tt.to_torch(n).contiguous().view(torch.float32).flatten()
            for n in state["norm"]
        ]
    )
    return dict(
        coarse=[tt.to_torch(x) for x in state["coarse"]],
        fine=[tt.to_torch(x) for x in state["fine"]],
        norm=metadata[:, :4],
        query_sum=metadata[:, SUM_WORD_OFFSET : SUM_WORD_OFFSET + 4],
        stage15=tt.to_torch(state["stage15"]),
        matrix=tt.to_torch(state["matrix"]),
        compact=tt.to_torch(state["compact"]),
    )
