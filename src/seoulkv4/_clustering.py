"""Torch spherical IVF construction shared by the index and research bench."""

import torch

EPS = 1e-12
_MAX_SCORE_ELEMENTS = 16 * 2**20  # 64 MiB of FP32 scores


def _score_rows(n_c):
    return max(1, min(8192, _MAX_SCORE_ELEMENTS // n_c))


def _withdraw(capacity, total):
    remaining = (total - capacity.cumsum(0)).clamp_min(0)
    return torch.cat((remaining.new_tensor([total]), remaining[:-1])) - remaining


def _kmeans(Ku, n_c, iters, seed):
    """Spherical Lloyd (cosine == MIPS here: ||k|| cv measured 0.020)."""
    n = Ku.shape[0]
    g = torch.Generator().manual_seed(seed)
    C = Ku[torch.randperm(n, generator=g)[:n_c].to(Ku.device)].clone()
    chunk = _score_rows(n_c)
    for _ in range(iters):
        lab = torch.empty(n, dtype=torch.long, device=Ku.device)
        for i in range(0, n, chunk):
            lab[i : i + chunk] = (Ku[i : i + chunk] @ C.t()).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, lab, Ku)
        cnt = torch.bincount(lab, minlength=n_c).unsqueeze(1).float()
        C = torch.where(cnt > 0, Cn / cnt.clamp_min(1), C)
        C = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return C


def _midrise(X, bits):
    """Max-abs mid-rise (2^bits levels) — the measured-best simple quantizer for these
    tensors (no heavy tail; clipping below max|x| loses at >=3 bits)."""
    a = X.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    nl = 2**bits
    step = 2 * a / nl
    return -a + (torch.clamp(torch.floor((X + a) / step), 0, nl - 1) + 0.5) * step


def _allocate_leaves(counts, n_c):
    """Allocate exactly n_c balanced fine cells in proportion to coarse occupancy."""
    counts_f = counts.float()
    raw = counts_f * (n_c / max(float(counts.sum()), 1.0))
    alloc = torch.floor(raw).long()
    alloc = torch.where(counts > 0, alloc.clamp_min(1), torch.zeros_like(alloc))
    delta = n_c - int(alloc.sum())
    if delta > 0:
        score = raw - alloc.float()
        score[counts == 0] = -torch.inf
        alloc[score.topk(delta).indices] += 1
    elif delta < 0:
        removable = alloc > 1
        score = raw - alloc.float()
        score[~removable] = torch.inf
        if -delta <= int(removable.sum()):
            alloc[(-score).topk(-delta).indices] -= 1
        else:
            # Very imbalanced buckets can require several leaves back from
            # one donor. Never remove the last leaf or touch an empty bucket.
            donors = score.argsort(stable=True)
            alloc[donors] -= _withdraw((alloc[donors] - 1).clamp_min(0), -delta)
    if int(alloc.sum()) != n_c:
        raise RuntimeError(
            f"hierarchical IVF leaf allocation {int(alloc.sum())} != {n_c}"
        )
    return alloc


def _refined_hierarchical_ivf(
    Ku,
    n_c,
    spill,
    seed,
    coarse_clusters=128,
    coarse_iters=4,
    local_iters=10,
    batch_coarse=16,
    coarse_centroids=None,
    coarse_labels=None,
):
    """Coarse k-means followed by batched exact local Lloyd solves.

    Morton leaves were balanced and fast, but eight random projections discarded too much
    of the 256-D geometry.  Here each coarse bucket gets its own fine spherical-k-means
    solve.  Buckets are padded in small batches so every Lloyd iteration is a handful of
    GEMMs rather than hundreds of tiny launches.  Fine assignment compares roughly 1/K0
    as many key/centroid pairs as global Lloyd at the 128K operating point.
    """
    n, d = Ku.shape
    k0 = min(max(2, coarse_clusters), n_c, n)
    if coarse_centroids is None:
        C0 = _kmeans(Ku, k0, coarse_iters, seed)
        coarse = torch.empty(n, dtype=torch.long, device=Ku.device)
        for start in range(0, n, 8192):
            coarse[start : start + 8192] = (Ku[start : start + 8192] @ C0.t()).argmax(1)
    else:
        C0, coarse = coarse_centroids, coarse_labels
        if C0.shape != (k0, d) or coarse.shape != (n,):
            raise ValueError("batched coarse k-means supplied incompatible tensors")

    counts0 = torch.bincount(coarse, minlength=k0)
    leaves = _allocate_leaves(counts0, n_c)
    active = counts0 > 0
    if bool((leaves[active] < spill).any()):
        # A sparse bucket still needs two distinct posting lists. Reassign
        # leaves from larger buckets without changing the codebook size.
        need = (spill - leaves).clamp_min(0) * active
        spare = (leaves - spill).clamp_min(0)
        deficit = int(need.sum())
        if int(spare.sum()) < deficit:
            raise ValueError("not enough fine leaves for distinct spill lists")
        donors = spare.argsort(descending=True, stable=True)
        leaves += need
        leaves[donors] -= _withdraw(spare[donors], deficit)
    leaf_start = leaves.cumsum(0) - leaves
    order = torch.argsort(coarse, stable=True)
    coarse_start = counts0.cumsum(0) - counts0
    C = torch.empty(n_c, d, dtype=Ku.dtype, device=Ku.device)
    labs = torch.empty(n, spill, dtype=torch.long, device=Ku.device)
    batch_coarse = max(1, int(batch_coarse))
    gen = torch.Generator(device=Ku.device).manual_seed(seed + 0x48494552)
    chunked_buckets = 0

    for cstart in range(0, k0, batch_coarse):
        cend = min(cstart + batch_coarse, k0)
        cb = counts0[cstart:cend]
        lb = leaves[cstart:cend]
        if int(cb.sum()) == 0:
            continue
        first = int(coarse_start[cstart])
        last = int(coarse_start[cend - 1] + counts0[cend - 1])
        max_n, max_l = int(cb.max()), int(lb.max())
        if len(cb) * max_n * max_l > _MAX_SCORE_ELEMENTS or max_l > max_n:
            # One large coarse bucket can make every padded score slab large.
            # Solve these buckets separately and bound each score matrix.
            cursor = first
            for ci, (size, nleaf, offset) in enumerate(
                zip(cb.tolist(), lb.tolist(), leaf_start[cstart:cend].tolist())
            ):
                if not size:
                    continue
                ids = order[cursor : cursor + size]
                cursor += size
                X = Ku[ids]
                if size < nleaf:
                    Cf = X[torch.arange(nleaf, device=Ku.device) % size]
                else:
                    Cf = _kmeans(X, nleaf, local_iters, seed + 0x48494552 + cstart + ci)
                C[offset : offset + nleaf] = Cf
                chunk = _score_rows(nleaf)
                for start in range(0, size, chunk):
                    labs[ids[start : start + chunk]] = (
                        X[start : start + chunk] @ Cf.t()
                    ).topk(spill, dim=1).indices + offset
                chunked_buckets += 1
            continue
        ids = order[first:last]
        local_c = coarse[ids] - cstart
        local_start = cb.cumsum(0) - cb
        rank = torch.arange(ids.numel(), device=Ku.device) - torch.repeat_interleave(
            local_start, cb
        )
        rows = torch.arange(max_n, device=Ku.device)
        cols = torch.arange(max_l, device=Ku.device)
        row_valid = rows[None, :] < cb[:, None]
        col_valid = cols[None, :] < lb[:, None]

        X = torch.zeros(cend - cstart, max_n, d, dtype=Ku.dtype, device=Ku.device)
        X[local_c, rank] = Ku[ids]
        # Invalid padded rows sort last. Reuse a real key for any extra
        # leaf in a singleton bucket so both spill lists remain selectable.
        init_key = (
            torch.rand(cend - cstart, max_n, generator=gen, device=Ku.device)
            .masked_fill(~row_valid, torch.inf)
            .topk(max_l, dim=1, largest=False)
            .indices
        )
        init_key = torch.where(
            col_valid & (cols >= cb[:, None]), init_key[:, :1], init_key
        )
        Cf = X.gather(1, init_key[..., None].expand(-1, -1, d)).clone()

        for _ in range(local_iters):
            score = torch.bmm(X, Cf.transpose(1, 2)).masked_fill(
                ~col_valid[:, None, :], -torch.inf
            )
            lab = score.argmax(2)
            Cn = torch.zeros_like(Cf)
            Cn.scatter_add_(
                1, lab[..., None].expand(-1, -1, d), X * row_valid[..., None]
            )
            cnt = torch.zeros(cend - cstart, max_l, dtype=Ku.dtype, device=Ku.device)
            cnt.scatter_add_(1, lab, row_valid.to(Ku.dtype))
            Cf = torch.where(cnt[..., None] > 0, Cn / cnt[..., None].clamp_min(1), Cf)
            Cf = Cf / Cf.norm(dim=2, keepdim=True).clamp_min(EPS)

        # Re-score once against the final updated codebook.  These exact local top-S
        # labels replace Morton's five-leaf window and become the posting lists directly.
        score = torch.bmm(X, Cf.transpose(1, 2)).masked_fill(
            ~col_valid[:, None, :], -torch.inf
        )
        top = score.topk(spill, dim=2).indices
        labs[ids] = top[local_c, rank] + leaf_start[coarse[ids], None]

        owner = torch.repeat_interleave(
            torch.arange(cend - cstart, device=Ku.device), lb
        )
        owner_start = lb.cumsum(0) - lb
        local_leaf = torch.arange(
            int(lb.sum()), device=Ku.device
        ) - torch.repeat_interleave(owner_start, lb)
        global_leaf = leaf_start[cstart:cend][owner] + local_leaf
        C[global_leaf] = Cf[owner, local_leaf]

    flat = labs.reshape(-1)
    keys = torch.arange(n, device=Ku.device).unsqueeze(1).expand_as(labs).reshape(-1)
    list_order = torch.argsort(flat)
    counts = torch.bincount(flat, minlength=n_c)
    off = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=Ku.device), counts.cumsum(0)]
    )
    local_pairs = int((counts0 * leaves).sum()) * (local_iters + 1)
    stats = {
        "algorithm": "hierarchical",
        "coarse_clusters": k0,
        "coarse_score_pairs": n * k0 * coarse_iters,
        "local_score_pairs": local_pairs,
        "legacy_score_pairs": n * n_c * max(1, local_iters + 1),
    }
    if chunked_buckets:
        stats["chunked_buckets"] = chunked_buckets
    return C, keys[list_order].int(), off, counts, labs, stats
