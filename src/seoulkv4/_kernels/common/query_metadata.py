"""Shared64-byte query metadata page: four norms and four raw FP32 sums."""

PAGE_BYTES = 64
HEADS = 4
SUM_WORD_OFFSET = HEADS


def cpp_defines():
    return f"#define SKV4_QUERY_SUM_WORD_OFFSET {SUM_WORD_OFFSET}\n"
