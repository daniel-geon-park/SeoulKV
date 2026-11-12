"""Grouped Stage-I index serialization and shared bit codecs."""

import hashlib
import json
from os import replace as replace_file
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

VERSION = 3
_INDEX_VERSION = 4
GROUP_SCALE_SIZE = 32
GROUP_SCALE_CODEC = "group-scale-32-v1"
QUERY_FIELDS = frozenset({"q", "query_hash", "coarse_ids", "head_mask"})


def fingerprint(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def pack2(codes):
    x = codes.to(torch.int64).reshape(*codes.shape[:-1], -1, 4)
    return (x[..., 0] | x[..., 1] << 2 | x[..., 2] << 4 | x[..., 3] << 6).byte()


def unpack2(data):
    return torch.stack([(data.long() >> (2 * i)) & 3 for i in range(4)], -1).flatten(-2)


def _coarse_metadata(source, upper, scale):
    """Canonical two-bit centroid operands and residual norm.

    ``scale`` is the full recovered row range ``a``.  The coordinate-rounded
    tensor is retained only as a compatibility/control value; production
    scoring uses exact odd BF16 operands and applies ``a/4`` after the dot.
    The residual norm deliberately follows the historical reference and is
    computed against the unrounded two-bit reconstruction before FP16 storage.
    """
    odd = 2 * unpack2(upper).float() - 3
    full = scale.float().unsqueeze(-1) * odd / 4
    return dict(
        coarse2=full.half(),
        coarse_odd=odd.bfloat16(),
        coarse_residual_norm=(source.float() - full).norm(dim=-1).half(),
    )


def ensure_coarse_metadata(cell):
    """Upgrade a version-1 fixture in memory without rewriting its file."""
    required = ("coarse2", "coarse_odd", "coarse_residual_norm")
    if all(name in cell for name in required):
        return cell
    source = cell.get("coarse_centroids", cell["centroids"])
    if "scale" not in cell:
        x = source.float()
        cell = dict(
            cell,
            scale=(x.abs().amax(-1) * (16 / 15)).clamp_min(1e-9),
        )
    return dict(cell, **_coarse_metadata(source, cell["upper"], cell["scale"]))


def require_group_scale_cell(cell, group=GROUP_SCALE_SIZE):
    """Validate the persisted group-scale summary without transforming it."""
    codec = f"group-scale-{group}-v1"
    if cell.get("summary_codec") != codec or cell.get("group_scale_size") != group:
        raise ValueError(
            "Stage-I fixture is not pre-grouped; migrate or regenerate it "
            "before TT prepare()"
        )
    nc = len(cell["centroids"])
    if nc % group:
        raise ValueError("grouped Stage-I fixture has a partial centroid group")
    group_scale = cell.get("group_fine_scale")
    if not isinstance(group_scale, torch.Tensor) or group_scale.shape != (nc // group,):
        raise ValueError("grouped Stage-I fixture has an invalid compact scale table")
    repeated = cell.get("fine_scale")
    if not isinstance(repeated, torch.Tensor) or repeated.shape != (nc,):
        raise ValueError("grouped Stage-I fixture is missing its CPU scale view")
    if not torch.equal(
        repeated.reshape(-1, group), group_scale[:, None].expand(-1, group)
    ):
        raise ValueError("grouped Stage-I fixture scale views disagree")
    return cell


def expand_lists(cell, selected):
    off, vals = cell["offsets"].long(), cell["values"].long()
    lists = [vals[off[int(c)] : off[int(c) + 1]] for c in selected]
    return torch.cat(lists).unique(sorted=True) if lists else vals[:0]


def validate_static_cell(c):
    """Check the index without requiring any fixture-only query fields."""
    nc, d = c["centroids"].shape
    assert d == 256 and 0 < c["p"] <= min(nc, 1536)
    assert torch.isfinite(c["centroids"]).all()
    assert c["upper"].shape == c["lower"].shape == (nc, 64)
    assert c["coarse2"].shape == c["coarse_odd"].shape == (nc, d)
    assert c["coarse_residual_norm"].shape == (nc,)
    assert torch.isfinite(c["coarse_residual_norm"]).all()
    off, vals = c["offsets"].long(), c["values"].long()
    assert len(off) == nc + 1 and off[0] == 0 and off[-1] == len(vals)
    assert (off[1:] >= off[:-1]).all()
    assert ((vals >= 0) & (vals < c["context"])).all()
    # Each retrieval token occurs in exactly spill distinct lists.
    counts = torch.bincount(vals, minlength=c["context"])
    assert (counts[c["r0"] : c["r1"]] == c["spill"]).all()
    assert counts[: c["r0"]].sum() == counts[c["r1"] :].sum() == 0
    rows = torch.repeat_interleave(torch.arange(nc), off.diff())
    entries = (rows * c["context"] + vals).sort().values
    assert (entries[1:] != entries[:-1]).all()


def validate_cell(c):
    c = ensure_coarse_metadata(c)
    validate_static_cell(c)
    nc, d = c["centroids"].shape
    assert c["q"].shape == (4, d) and torch.isfinite(c["q"]).all()
    ids, mask = c["coarse_ids"].long(), c["head_mask"].long()
    assert len(ids) and (ids >= 0).all() and (ids < nc).all()
    assert (ids[1:] > ids[:-1]).all() and ((mask > 0) & (mask < 16)).all()
    assert all(int(((mask >> h) & 1).sum()) >= c["p"] for h in range(4))


def _load_archive(path, *, allow_index):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        version = meta["version"]
        if version not in (1, 2, VERSION, _INDEX_VERSION):
            raise ValueError("unsupported Stage-I fixture version")
        static = version == _INDEX_VERSION
        if static and not allow_index:
            raise ValueError(
                "this Stage-I index has no replay query; use SeoulKV4Index.load()"
            )
        bf16 = set(meta.get("bf16_arrays", ())) if static else set()
        if not bf16.issubset(z.files):
            raise ValueError("Stage-I index is missing a declared BF16 array")
        cells = []
        for i, info in enumerate(meta["cells"]):
            c = dict(info)
            prefix = f"{i}_"
            for name in z.files:
                if name.startswith(prefix):
                    tensor = torch.from_numpy(z[name])
                    if name in bf16:
                        if tensor.dtype != torch.int16:
                            raise ValueError(f"invalid BF16 storage for {name}")
                        tensor = tensor.view(torch.bfloat16)
                    c[name[len(prefix) :]] = tensor
            if static:
                if QUERY_FIELDS.intersection(c):
                    raise ValueError(
                        "static Stage-I index contains replay query fields"
                    )
                validate_static_cell(c)
            else:
                c["q"] = c["q"].bfloat16()
                c = ensure_coarse_metadata(c)
                validate_cell(c)
            if version >= VERSION:
                c = require_group_scale_cell(c)
            cells.append(c)
    return cells, meta


def load_fixture(path):
    """Read a historical fixture containing a diagnostic query."""
    return _load_archive(path, allow_index=False)


def load_index(path):
    """Read a static index or a historical grouped fixture."""
    return _load_archive(path, allow_index=True)


def _archive_arrays(cells, meta, *, version):
    data, rows, bf16 = {}, [], []
    for i, cell in enumerate(cells):
        scalar = {}
        for name, value in cell.items():
            if version == _INDEX_VERSION and name in QUERY_FIELDS:
                continue
            if isinstance(value, torch.Tensor):
                key = f"{i}_{name}"
                if value.dtype == torch.bfloat16:
                    if version == _INDEX_VERSION:
                        # NumPy has no native BF16 dtype. Store its raw words
                        # with a manifest so load can restore the exact dtype.
                        value = value.view(torch.int16)
                        bf16.append(key)
                    else:
                        value = value.float()
                data[key] = value.numpy()
            else:
                scalar[name] = value
        rows.append(scalar)
    metadata = dict(meta, version=version, cells=rows)
    if version == _INDEX_VERSION:
        metadata["bf16_arrays"] = bf16
    data["metadata"] = np.array(json.dumps(metadata))
    return data


def save_fixture(path, cells, meta):
    """Write the legacy grouped fixture, including its replay query."""
    cells = [ensure_coarse_metadata(require_group_scale_cell(c)) for c in cells]
    for cell in cells:
        validate_cell(cell)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_archive_arrays(cells, meta, version=VERSION))


def dump_index(path, cells, meta):
    """Write a query-free NPZ at the exact path, replacing only on success."""
    path = Path(path)
    data = _archive_arrays(cells, meta, version=_INDEX_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with temp:
            np.savez(temp, **data)
        replace_file(temp.name, path)
    finally:
        Path(temp.name).unlink(missing_ok=True)


def encode_group_scale_cell(cell, group=GROUP_SCALE_SIZE):
    """Pack static centroids and remap their CSR rows; no query required."""
    codec = f"group-scale-{group}-v1"
    if group <= 0 or group % 8:
        raise ValueError("group scale size must be a positive multiple of 8")
    source = cell
    original = source.get("coarse_centroids", source["centroids"])
    centroids = original.float()
    nc, d = centroids.shape
    if d != 256 or nc % group:
        raise ValueError(
            f"group-scale cell requires [multiple of {group},256], got {tuple(centroids.shape)}"
        )

    # Stable ordering keeps equal-range clusters deterministic and makes the
    # physical remap reproducible across fixture loads.
    ranges = source.get("scale")
    if ranges is None:
        ranges = (centroids.abs().amax(-1) * (16 / 15)).clamp_min(1e-9)
    permutation = torch.argsort(ranges.float(), stable=True)
    centroids = centroids[permutation]
    blocks = centroids.reshape(-1, group, d)
    group_scale = (blocks.abs().amax(dim=(1, 2)) * (16 / 15)).clamp_min(1e-9)
    scale = group_scale.repeat_interleave(group)
    code = (
        ((centroids + scale[:, None]) / (scale[:, None] / 8))
        .floor()
        .clamp(0, 15)
        .byte()
    )
    odd = (2 * code.float() - 15).bfloat16()
    decoded = (scale[:, None] * odd.float() / 16).half()
    levels = (
        (
            group_scale[:, None, None]
            * (2 * torch.arange(16).float()[None, None] - 15)
            / 16
        )
        .expand(-1, group, -1)
        .reshape(nc, 16)
        .half()
    )
    upper, lower = pack2(code >> 2), pack2(code & 3)

    # Move every posting list with its centroid.  Stable sorting preserves
    # token order inside each original list.
    offsets = source["offsets"].long()
    values = source["values"]
    lengths = offsets[1:] - offsets[:-1]
    old_cluster = torch.repeat_interleave(torch.arange(nc), lengths)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(nc)
    posting_order = torch.argsort(inverse[old_cluster], stable=True)
    new_lengths = lengths[permutation]
    new_offsets = torch.cat((torch.zeros(1, dtype=torch.long), new_lengths.cumsum(0)))

    # Fine keeps one resident scale per physical 32-centroid group.  The
    # repeated per-centroid view remains a CPU-reference convenience only;
    # TT kernels must consume ``group_fine_scale`` directly.
    group_fine_scale = (group_scale / 16).half()
    fine_scale = group_fine_scale.repeat_interleave(group)
    grouped = dict(
        source,
        coarse_centroids=centroids.to(original.dtype),
        centroids=decoded.bfloat16(),
        upper=upper,
        lower=lower,
        scale=scale,
        decoded=decoded.bfloat16(),
        fine_scale=fine_scale,
        group_fine_scale=group_fine_scale,
        levels=levels,
        centroid_codec="fp16-scale-odd-v1",
        offsets=new_offsets.to(source["offsets"].dtype),
        values=values[posting_order],
        max_codec_error=float((centroids - decoded.float()).abs().max()),
        group_scale_size=group,
        summary_codec=codec,
        **_coarse_metadata(centroids, upper, scale),
    )
    return grouped, inverse
