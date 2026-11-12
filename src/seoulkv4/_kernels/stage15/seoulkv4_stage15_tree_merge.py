"""Capacity-safe balanced six-shard Stage1.5 frontier merge.

Every shard first produces an exact K2048 frontier. Odd shards send their
frontier to the neighboring even shard, then shard roots 2 and 4 send their
pair result to root 0. Unlike the retired compact-pair topology, this path
supports the complete 13,056-candidate Stage1.5 input domain.
"""

from seoulkv4._kernels.stage15.abi import Stage15Semaphore as SEM
from seoulkv4._kernels.common.semaphore_layout import cpp_defines as semaphore_defines


TREE_COMPUTE = r"""
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/topk_xl.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t K=get_compile_time_arg_val(0);
    constexpr uint32_t W=get_compile_time_arg_val(1);
    static_assert(K==2048 && W==6);
    constexpr uint32_t TILES=2,SURV_VALUES=0,SURV_IDS=2;
    constexpr uint32_t IN_VALUES=4,IN_IDS=6;
    const uint32_t shard=get_arg_val<uint32_t>(2);
    CircularBuffer control(24);
    control.wait_front(1);
    const bool compact_pair=ckernel::read_tile_value(24,0,10)<=6*1024;
    const uint32_t incoming=compact_pair?2:(shard==0?3:1);

    compute_kernel_hw_startup(10,4);
    pack_untilize_dest_init<TILES,TILES>(4);
    CircularBuffer own_score(10),own_id(11),local_ready(26);
    CircularBuffer score_cb(0),id_cb(1),out_score(4),out_id(5);
    tile_regs_acquire();

    if(!compact_pair) local_ready.wait_front(1);
    own_score.wait_front(TILES);
    topk_xl_copy_tile_init(10);
    topk_xl_copy_tile<K>(10,SURV_VALUES,0,K);
    own_id.wait_front(TILES);
    topk_xl_copy_tile_init(11);
    topk_xl_copy_tile<K>(11,SURV_IDS,0,K);
    if(!compact_pair) local_ready.pop_front(1);

    if(shard!=0) {
        topk_xl_init<K,false>();
        topk_xl_rebuild<K,false>(SURV_VALUES,false);
    }
    for(uint32_t edge=0;edge<incoming;++edge) {
        score_cb.wait_front(TILES);
        topk_xl_copy_tile_init(0);
        topk_xl_copy_tile<K>(0,IN_VALUES,0,K);
        score_cb.pop_front(TILES);
        id_cb.wait_front(TILES);
        topk_xl_copy_tile_init(1);
        topk_xl_copy_tile<K>(1,IN_IDS,0,K);
        id_cb.pop_front(TILES);
        topk_xl_init<K,false>();
        topk_xl_merge<K,false>(SURV_VALUES);
        const bool ascending=shard!=0 && edge+1==incoming;
        topk_xl_rebuild<K,false>(SURV_VALUES,ascending);
    }
    tile_regs_commit(); tile_regs_wait();

    out_score.reserve_back(1);
    pack_untilize_dest<TILES,TILES>(4,1,0,SURV_VALUES);
    out_score.push_back(1);
    pack_untilize_dest_init<TILES,TILES>(5);
    out_id.reserve_back(1);
    pack_untilize_dest<TILES,TILES>(5,1,0,SURV_IDS);
    out_id.push_back(1);
    tile_regs_release();
}
"""


READER_TREE = (
    semaphore_defines(SEM)
    + r"""
    const uint32_t tree_shard=worker/G;
    const bool tree_compact_pair=UNIQUE<=6*1024;
    if(tree_compact_pair && tree_shard==0) {
        cb_reserve_back(12,4);
        cb_reserve_back(13,4);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_SEM_TREE_FRONTIERS_READY));
        // Root retained the odd-shard child-row notification (count 1), then
        // receives the two completed pair frontiers (counts 2 and 3).
        noc_semaphore_wait_min(ready,3);
        noc_semaphore_set(ready,0);
        cb_push_back(12,4);
        cb_push_back(13,4);
    } else if(!tree_compact_pair && (tree_shard&1u)==0) {
        const uint32_t receives=tree_shard==0?3:1;
        cb_reserve_back(12,receives*2);
        cb_reserve_back(13,receives*2);
        auto ready=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(SKV4_SEM_TREE_FRONTIERS_READY));
        noc_semaphore_wait_min(ready,receives);
        /*@S15D_TREE_RESET_B*/
        noc_semaphore_set(ready,0);
        /*@S15D_TREE_RESET_E*/
        cb_push_back(12,receives*2);
        cb_push_back(13,receives*2);
    }
"""
)


WRITER_LEVEL0 = (
    semaphore_defines(SEM)
    + r"""
    if(!compact_pair && (shard&1u)) {
        const uint32_t parent=(shard-1)*G+quarter;
        noc_async_write(get_read_ptr(10),
            peer(parent,get_write_ptr(12)),K*2);
        noc_async_write(get_read_ptr(11),
            peer(parent,get_write_ptr(13)),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
    }
"""
)


WRITER_INTERMEDIATE = (
    semaphore_defines(SEM)
    + r"""
    if(compact_pair && (shard==2 || shard==4)) {
        const uint32_t edge=shard==2?0:1;
        const uint32_t parent=quarter;
        noc_async_write(get_read_ptr(10),
            peer(parent,get_write_ptr(12)+edge*K*2),K*2);
        noc_async_write(get_read_ptr(11),
            peer(parent,get_write_ptr(13)+edge*K*4),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
    } else if(!compact_pair && (shard==2 || shard==4)) {
        cb_wait_front(14,1); cb_wait_front(15,1);
        const uint32_t edge=shard==2?1:2;
        const uint32_t parent=quarter;
        noc_async_write(get_read_ptr(14),
            peer(parent,get_write_ptr(12)+edge*K*2),K*2);
        noc_async_write(get_read_ptr(15),
            peer(parent,get_write_ptr(13)+edge*K*4),K*4);
        noc_async_write_barrier();
        notify(SKV4_SEM_TREE_FRONTIERS_READY,parent);
        cb_pop_front(14,1); cb_pop_front(15,1);
    }
"""
)


def merge_tree_reference(rows, k=2048):
    """Exact score-only reference for the 6->3->2->1 topology."""
    if len(rows) != 6:
        raise ValueError("the production topology has six shards")

    def merge(a, b):
        return sorted([*a, *b], reverse=True)[:k]

    level0 = [merge(rows[i], rows[i + 1]) for i in (0, 2, 4)]
    return merge(merge(level0[0], level0[1]), level0[2])


def compact_pair_source_position(p, unique, n=544, workers=24):
    """Map the fast path's six K1024 rows to the fixed source ABI."""
    if unique > 6 * 1024:
        return p
    if not 0 <= p < 6 * 1024:
        return -1
    shard, local = divmod(p, 1024)
    chunks = (unique + 31) // 32
    compact = 0
    for quarter in range(4):
        worker = shard * 4 + quarter
        lo = (chunks * worker // workers) * 32
        hi = min((chunks * (worker + 1) // workers) * 32, unique)
        live = max(0, hi - lo)
        if local < compact + live:
            return worker * n + local - compact
        compact += live
    return -1


def compact_bitmap_interval(unique, worker, workers=24):
    """Return the fast bitmap's encoded start and source-local live length."""
    if not 0 <= unique <= 6 * 1024 or not 0 <= worker < workers:
        raise ValueError("compact bitmap interval is outside the fast domain")
    group = 4
    shard = worker // group
    chunks = (unique + 31) // 32
    source_lo = (chunks * worker // workers) * 32
    source_hi = min((chunks * (worker + 1) // workers) * 32, unique)
    shard_lo = (chunks * (shard * group) // workers) * 32
    return shard * 1024 + source_lo - shard_lo, max(0, source_hi - source_lo)
