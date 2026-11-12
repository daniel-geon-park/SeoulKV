"""CPU prefill layout for variable-length, spill-duplicated Stage1.5 blocks."""
import numpy as np
import torch


def align64(n):
    return (int(n)+63)//64*64


def posting_mates(cell):
    """Other membership per CSR occurrence, or UINT32_MAX; reject non-spill2 CSR.

    Sorting is prefill-only. Stable ordering preserves the original CSR layout.
    Tokens with zero/one memberships are valid (sink/local and edge fixtures).
    """
    off=cell['offsets'].cpu().numpy().astype(np.int64)
    ids=cell['values'].cpu().numpy().astype(np.int64)
    if len(off)<1 or off[0]!=0 or off[-1]!=len(ids) or (np.diff(off)<0).any():
        raise ValueError('invalid CSR offsets')
    if ((ids<0)|(ids>=cell['context'])).any():
        raise ValueError('invalid CSR token ID')
    cluster=np.repeat(np.arange(len(off)-1,dtype=np.int64),np.diff(off))
    order=np.argsort(ids,kind='stable');ordered=ids[order]
    if len(ids)>2 and ((ordered[:-2]==ordered[1:-1]) & (ordered[1:-1]==ordered[2:])).any():
        raise ValueError('local ownership requires at most two memberships/token')
    pairs=np.flatnonzero(ordered[:-1]==ordered[1:])
    a,b=order[pairs],order[pairs+1]
    if (cluster[a]==cluster[b]).any():
        raise ValueError('duplicate token inside one cluster')
    mates=np.full(len(ids),0xffffffff,dtype=np.uint32)
    mates[a]=cluster[b];mates[b]=cluster[a]
    return torch.from_numpy(mates.astype(np.int64))


def local_encounter_union(cell, selected, workers=24):
    """Offline ownership proof; concatenation exactly preserves encounter order."""
    if workers<1:raise ValueError('positive worker count required')
    selected=selected.long().tolist();nc=len(cell['offsets'])-1
    if any(c<0 or c>=nc for c in selected):raise ValueError('invalid selected cluster')
    rank={}
    for i,c in enumerate(selected):rank.setdefault(c,i)
    mates=posting_mates(cell).tolist();off=cell['offsets'].tolist();ids=cell['values'].tolist()
    parts=[]
    for w in range(workers):
        out=[]
        for i in range(len(selected)*w//workers,len(selected)*(w+1)//workers):
            c=selected[i]
            if rank[c]!=i:continue # repeated cluster selection, not another membership
            out.extend(ids[j] for j in range(off[c],off[c+1])
                       if rank.get(mates[j],len(selected))>i)
        parts.append(torch.tensor(out,dtype=torch.long))
    return parts


def pack_cluster_blocks(cell, token_prefix, page_bytes=16384, *, include_mates=False):
    if token_prefix.device.type != 'cpu' or token_prefix.dtype != torch.uint8:
        raise ValueError('cluster packing requires CPU uint8 token prefixes at prefill')
    prefix=token_prefix.reshape(cell['context'],64).contiguous().numpy()
    off=cell['offsets'].cpu().numpy().astype(np.int64)
    ids=cell['values'].cpu().numpy().astype(np.int64)
    counts=np.diff(off)
    if (counts<0).any() or off[0]!=0 or off[-1]!=len(ids):
        raise ValueError('invalid CSR offsets')
    if ((ids<0)|(ids>=cell['context'])).any():
        raise ValueError('invalid CSR token ID')
    mate=posting_mates(cell).numpy().astype(np.uint32) if include_mates else None
    metadata_bytes=8 if include_mates else 4
    sizes=counts*64+((counts*metadata_bytes+63)//64)*64
    max_block=int(sizes.max(initial=0))
    page_bytes=max(align64(page_bytes),max_block,64)
    desc=np.zeros((len(counts),4),dtype=np.uint32)
    page=offset=0
    for cid,(n,size) in enumerate(zip(counts,sizes)):
        n,size=int(n),int(size)
        if offset+size>page_bytes:
            page+=1;offset=0
        desc[cid]=page,offset,n,size
        offset+=size
    pages=np.zeros((page+1,page_bytes),dtype=np.uint8)
    for cid in range(len(counts)):
        pg,at,n,size=map(int,desc[cid])
        if not n:continue
        values=ids[off[cid]:off[cid+1]]
        meta=(np.column_stack((values,mate[off[cid]:off[cid+1]]))
              if include_mates else values)
        idbytes=meta.astype('<u4').reshape(-1).view(np.uint8)
        pages[pg,at:at+len(idbytes)]=idbytes
        start=at+align64(n*metadata_bytes)
        pages[pg,start:start+n*64]=prefix[values].reshape(-1)
    padded=np.zeros((align64(desc.nbytes)//16,4),dtype=np.uint32)
    padded[:len(desc)]=desc
    stats=dict(clusters=len(counts),memberships=len(ids),max_count=int(counts.max(initial=0)),
               page_bytes=page_bytes,pages=len(pages),max_block_bytes=max_block,
               descriptor_t1_bytes=padded.nbytes,prefix_t2_bytes=len(ids)*64,
               token_id_t2_bytes=len(ids)*4,
               id_padding_t2_bytes=int(sizes.sum())-len(ids)*(64+metadata_bytes),
               page_padding_t2_bytes=pages.nbytes-int(sizes.sum()),
               allocated_t2_bytes=pages.nbytes)
    stats['mate_t2_bytes']=len(ids)*4 if include_mates else 0
    stats['posting_metadata_bytes']=metadata_bytes
    assert stats['allocated_t2_bytes']==sum(stats[k] for k in (
        'prefix_t2_bytes','token_id_t2_bytes','mate_t2_bytes','id_padding_t2_bytes','page_padding_t2_bytes'))
    return dict(descriptors=torch.from_numpy(padded.astype(np.int32)).reshape(-1,16),
                pages=torch.from_numpy(pages),stats=stats)


def encounter_union(cell, selected):
    """Untimed reference order; runtime derives this from block ID headers on device."""
    off=cell['offsets'].tolist();vals=cell['values'].tolist()
    seen=set();out=[]
    for cid in selected.tolist():
        for token in vals[off[cid]:off[cid+1]]:
            if token not in seen:
                seen.add(token);out.append(token)
    return torch.tensor(out,dtype=torch.long)
