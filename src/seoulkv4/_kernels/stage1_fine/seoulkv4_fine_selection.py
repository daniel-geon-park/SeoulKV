"""Untimed readback of the native Fine selector's cluster handoff and scores."""

import torch


def readback(state, decoded):
    tt = state["tt"]
    packet = tt.to_torch(state["output"]).flatten().long()
    if packet[1] != 0:
        raise RuntimeError(f"Fine status={int(packet[1])}")
    scores = (
        tt.to_torch(state["scores"])[:, 0, :4].permute(1, 0, 2).reshape(4, state["n"])
    )
    heads = tt.to_torch(state["heads"]).long()
    if state.get("heads_are_positions", False):
        cell = state["cell"]
        coarse_ids = cell["coarse_ids"].long()
        head_mask = cell["head_mask"].long()
        mapped = torch.full_like(heads, 0xFFFFFFFF)
        for h in range(4):
            valid = (heads[h] >= 0) & (heads[h] < len(coarse_ids))
            pos = heads[h, valid]
            allowed = ((head_mask[pos] >> h) & 1).bool()
            target = valid.nonzero().flatten()[allowed]
            mapped[h, target] = coarse_ids[pos[allowed]]
        heads = mapped
    return dict(
        selected=packet[16 : 16 + int(packet[0])],
        heads=heads,
        scores=scores.float(),
        decoded=decoded,
    )
