"""Clean-core streaming-attention arguments and on-core gather protocol."""

from seoulkv4._kernels.common.argument_abi import CompileABI, layout


READER_RT = layout("query", "kv", "handoff", "slot", "head", "context", "sink", "local")
READER_CT = CompileABI(layout("d", "handoff_bytes"), ("query", "kv", "handoff"))
WRITER_RT = layout("output", "kv", "slot", "head", "context")
WRITER_CT = CompileABI(layout("d"), ("output", "kv"))
GATHER_REQUEST = layout("index_l1", "split", "is_last", "dst_l1")


def cpp_defines():
    return (
        READER_RT.cpp_defines("SKV4_ATTN_R_RT")
        + READER_CT.cpp_defines("SKV4_ATTN_R_CT")
        + WRITER_RT.cpp_defines("SKV4_ATTN_W_RT")
        + WRITER_CT.cpp_defines("SKV4_ATTN_W_CT")
        + GATHER_REQUEST.cpp_defines("SKV4_ATTN_REQ")
    )
