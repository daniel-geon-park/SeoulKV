"""Lossless shared int2 layout for native-BFP4 Coarse and packed Fine.

Only physical placement changes. Each 32-centroid block remains 2048 bytes;
every source word belongs to one centroid. Fine reads a block once, copies
its sixteen words per selected centroid, and uses the same permutation on its
lower plane and Q. No duplicate persistent upper plane is needed.
"""

from functools import lru_cache

import torch

from seoulkv4._codecs.index import pack2, unpack2
from seoulkv4._kernels.stage15.stage15_sfpu_native import coordinate_order


@lru_cache(maxsize=1)
def layout():
    """Return physical-field -> original-coordinate labels and Fine order."""
    col = torch.arange(32)
    permutation = (
        2 * ((col % 8) // 2) + 16 * (col % 2)
        + (col // 8) % 2 + 8 * (col // 16)
    )
    native_order = (
        32 * torch.arange(8)[:, None] + torch.argsort(permutation)[None, :]
    ).flatten()
    labels = torch.arange(8192).reshape(32, 256)[:, coordinate_order()]
    stored = torch.empty_like(labels)
    for row in range(32):
        for column in range(256):
            tile, tc = divmod(column, 32)
            source_key = (
                16 * (row // 16) + (row // 8) % 2
                + 2 * (tc // 16) + 4 * ((tc % 8) // 2)
            )
            source_col = (
                32 * tile + 2 * (row % 8) + tc % 2
                + 16 * ((tc // 8) % 2)
            )
            stored[source_key, source_col] = labels[row, native_order[column]]
    transpose = torch.arange(256).reshape(16, 16).T.contiguous().flatten()
    source = stored[:, torch.argsort(transpose)].reshape(16, 32, 16)
    fields = torch.empty_like(source)
    for pair in range(4):
        for row in range(4):
            face_row, half = divmod(row, 2)
            for page_half in range(2):
                page = 2 * pair + page_half
                for v in range(8):
                    fields[pair * 4 + row, :, 8 * page_half + v] = source[
                        face_row * 8 + v, :, 2 * page + half
                    ]

    flat = fields.flatten()
    word_labels = fields.reshape(512, 16)
    row_order = None
    for key in range(32):
        words = (word_labels[:, 0] // 256 == key).nonzero().flatten()
        base = (key // 16) * 64 + (key % 16) * 2
        offsets = torch.tensor([0, 1, 32, 33, 128, 129, 160, 161,
                                256, 257, 288, 289, 384, 385, 416, 417])
        assert torch.equal(words, base + offsets)
        assert (word_labels[words] // 256 == key).all()
        order = word_labels[words].flatten() % 256
        if row_order is None:
            row_order = order
        else:
            assert torch.equal(order, row_order)
    assert torch.equal(flat.sort().values, torch.arange(8192))
    return flat, row_order, row_order[coordinate_order()]


def prepack_upper(upper):
    """Pack canonical [centroid,64] int2 rows in 32-centroid native blocks."""
    codes = unpack2(upper.cpu())
    if len(codes) % 32:
        raise ValueError("native Coarse blocks require 32-centroid padding")
    fields = codes.reshape(-1, 8192)[:, layout()[0]]
    return pack2(fields).contiguous().reshape(-1, 2048)


def prepack_fine_lower(lower):
    """Match the words copied from the shared upper block, still 64B/key."""
    return pack2(unpack2(lower.cpu())[:, layout()[1]]).contiguous()


def fine_coordinate_order():
    return layout()[2]
