"""Decompose the TLX decode latency floor at batch=1, ctx=1024, qlen=1.

Times each component (host sync, scratch alloc, phase-1 partition, phase-2
reduce, split=1 vs default) to attribute the ~95us floor.
"""
import torch
import triton

from amd_pa_decode import (
    pa_decode_tlx, get_num_splits, _next_pow2,
    _pa_decode_partition_kernel, _pa_decode_reduce_kernel,
)
from bench_pa_decode import build_inputs


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3  # -> us


def main():
    device = "cuda"
    head_dim, page_size = 128, 64
    num_kv_heads, group = 8, 8
    num_q_heads = num_kv_heads * group
    sm_scale = 1.0 / (head_dim ** 0.5)
    b, ctx_len, qlen = 1, 1024, 1

    q, kc, vc, ctx, bt = build_inputs(
        b, [ctx_len] * b, num_q_heads, num_kv_heads, head_dim, page_size,
        query_length=qlen, device=device, pool_pages=4 * (ctx_len // page_size) + 16)
    out = torch.empty_like(q)

    num_tokens = q.shape[0]
    num_seqs = num_tokens // qlen
    qgs = num_q_heads // num_kv_heads
    qlen_pow2 = _next_pow2(qlen)
    group_pow2 = max(16 // qlen_pow2, _next_pow2(qgs))
    m_pow2 = qlen_pow2 * group_pow2

    print(f"config: b={b} ctx={ctx_len} qlen={qlen} "
          f"num_cu={torch.cuda.get_device_properties(0).multi_processor_count}")

    # full call (auto splits)
    t_full = bench(lambda: pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=qlen))
    ns_auto = get_num_splits(num_seqs, num_kv_heads, ctx_len, page_size)
    print(f"  full (auto splits={ns_auto}):        {t_full:8.1f} us")

    # full call forcing splits=1
    t_full1 = bench(lambda: pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=qlen, num_splits=1))
    print(f"  full (forced splits=1):        {t_full1:8.1f} us")

    # component: host sync .item()
    t_item = bench(lambda: int(ctx.max().item()))
    print(f"  .item() host sync:             {t_item:8.1f} us")

    # component: scratch allocation for auto splits
    def alloc():
        m = torch.empty((num_seqs, num_kv_heads, ns_auto, m_pow2, head_dim), dtype=torch.float32, device=device)
        l = torch.empty((num_seqs, num_kv_heads, ns_auto, m_pow2), dtype=torch.float32, device=device)
        return m, l
    t_alloc = bench(alloc)
    print(f"  scratch torch.empty (x2):      {t_alloc:8.1f} us")

    # pre-alloc scratch for isolated kernel launches
    def make_scratch(ns):
        mid = torch.empty((num_seqs, num_kv_heads, ns, m_pow2, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((num_seqs, num_kv_heads, ns, m_pow2), dtype=torch.float32, device=device)
        return mid, lse

    def launch_partition(ns):
        mid, lse = make_scratch(ns)
        def run():
            grid = (num_seqs, num_kv_heads, ns)
            _pa_decode_partition_kernel[grid](
                q, kc, vc, bt, ctx, mid, lse, sm_scale, ns,
                q.stride(0), q.stride(1), q.stride(2),
                kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3),
                vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3),
                bt.stride(0), bt.stride(1),
                mid.stride(0), mid.stride(1), mid.stride(2), mid.stride(3), mid.stride(4),
                lse.stride(0), lse.stride(1), lse.stride(2), lse.stride(3),
                HEAD_DIM=head_dim, PAGE_SIZE=page_size,
                QUERY_GROUP_SIZE=qgs, GROUP_POW2=group_pow2,
                QLEN=qlen, QLEN_POW2=qlen_pow2, M_POW2=m_pow2,
                num_warps=4, waves_per_eu=0)
        return run, mid, lse

    for ns in (1, ns_auto):
        run, mid, lse = launch_partition(ns)
        t = bench(run)
        print(f"  phase-1 partition only (splits={ns}): {t:8.1f} us")

    # phase-2 reduce only (using auto-splits scratch)
    _, mid, lse = launch_partition(ns_auto)
    # populate once
    run_p, _, _ = launch_partition(ns_auto)
    run_p()
    splits_pow2 = _next_pow2(ns_auto)
    def run_reduce():
        grid = (num_tokens, num_q_heads)
        _pa_decode_reduce_kernel[grid](
            out, mid, lse, ns_auto,
            out.stride(0), out.stride(1), out.stride(2),
            mid.stride(0), mid.stride(1), mid.stride(2), mid.stride(3), mid.stride(4),
            lse.stride(0), lse.stride(1), lse.stride(2), lse.stride(3),
            HEAD_DIM=head_dim, QUERY_GROUP_SIZE=qgs, GROUP_POW2=group_pow2,
            QLEN=qlen, SPLITS_POW2=splits_pow2)
    t_red = bench(run_reduce)
    print(f"  phase-2 reduce only (splits={ns_auto}):   {t_red:8.1f} us")

    # empty kernel launch overhead (single trivial launch) via a tiny partition splits=1 grid
    print("\n  note: 'full' includes .item()+alloc+2 launches; compare to sums above")


if __name__ == "__main__":
    main()
