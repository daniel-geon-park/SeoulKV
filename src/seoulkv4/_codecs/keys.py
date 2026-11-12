"""Query-independent four-plane key encoding."""

import torch


def _pack_lowbits(values, bits):
    """Pack coordinate codes little-endian within each byte."""
    per_byte = 8 // bits
    if values.shape[-1] % per_byte:
        raise ValueError("head dimension is not packable")
    grouped = values.to(torch.int16).reshape(*values.shape[:-1], -1, per_byte)
    shifts = torch.arange(per_byte, dtype=torch.int16) * bits
    return (grouped << shifts).sum(dim=-1).to(torch.uint8).contiguous()


def _stage15_key_codec(k):
    """Quantize host keys independently of the query."""
    kf = k.float()
    scales = kf.abs().amax(dim=1).clamp_min(1e-9)
    step8 = 2 * scales[:, None, :] / 256
    code8 = (
        torch.clamp(torch.floor((kf + scales[:, None, :]) / step8), 0, 255)
        .to(torch.uint8)
        .contiguous()
    )
    return {"code8": code8, "scales": scales.contiguous()}


def encode_planes(codec, n_conc, n_kv_heads):
    code8 = codec["code8"]
    _, context, d = code8.shape
    slots = n_conc * n_kv_heads
    slot_heads = torch.arange(slots) % n_kv_heads
    return {
        "prefix2_token": _pack_lowbits(code8 >> 6, 2)[slot_heads].reshape(
            slots * context, d // 4
        ),
        "residual2": _pack_lowbits((code8 >> 4) & 3, 2)[slot_heads].reshape(
            slots * context, d // 4
        ),
        "residual4": _pack_lowbits(code8 & 15, 4)[slot_heads].reshape(
            slots * context, d // 2
        ),
    }
