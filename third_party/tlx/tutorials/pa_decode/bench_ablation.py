"""Ablate the two launcher changes under warm, controlled conditions.

do_bench(warmup=...) removes the cold-clock / first-compile confound, so any
delta here is the real steady-state effect of:
  (1) the .item() device->host sync  (old launcher did it every call)
  (2) split count: old = min(8, occ, cdiv(ctx,256)) vs new = min(8, occ)

For each config we time 4 variants:
  new         : no sync, splits = occupancy-only        (current code)
  new+sync    : no split change, but add .item() sync    (isolates sync cost)
  oldsplit    : no sync, splits = old (context-capped)    (isolates split effect)
  old         : sync + old splits                         (full previous behavior)
"""
import torch
import triton

from amd_pa_decode import pa_decode_tlx, _next_pow2
from bench_pa_decode import build_inputs


def bench(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=200) * 1e3  # -> us


def occ_splits(num_seqs, num_kv_heads, cap=8):
    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    return max(1, min(cap, (num_cu * 2) // max(1, num_seqs * num_kv_heads)))


def old_splits(num_seqs, num_kv_heads, ctx_len, cap=8, part=256):
    occ = (torch.cuda.get_device_properties(0).multi_processor_count * 2) // max(1, num_seqs * num_kv_heads)
    return max(1, min(cap, occ, triton.cdiv(ctx_len, part)))


def main():
    device = "cuda"
    head_dim, page_size = 128, 64
    num_kv_heads, group = 8, 8
    num_q_heads = num_kv_heads * group
    sm_scale = 1.0 / (head_dim ** 0.5)

    configs = [(1, 1024), (8, 1024), (32, 1024), (128, 1024),
               (1, 8192), (1, 131072), (8, 131072)]

    print(f"{'batch':>6} {'ctx':>8} | {'nsplit_new':>10} {'nsplit_old':>10} | "
          f"{'new_us':>8} {'new+sync':>9} {'oldsplit':>9} {'old_us':>8} | "
          f"{'sync_delta':>10} {'split_delta':>11} {'total':>7}", flush=True)
    for b, ctx_len in configs:
        pool = 4 * ((ctx_len + page_size - 1) // page_size) + 16
        q, kc, vc, ctx, bt = build_inputs(
            b, [ctx_len] * b, num_q_heads, num_kv_heads, head_dim, page_size,
            query_length=1, device=device, pool_pages=pool)
        out = torch.empty_like(q)
        ns_new = occ_splits(b, num_kv_heads)
        ns_old = old_splits(b, num_kv_heads, ctx_len)

        def f_new():
            pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=1, num_splits=ns_new)

        def f_new_sync():
            _ = int(ctx.max().item())
            pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=1, num_splits=ns_new)

        def f_oldsplit():
            pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=1, num_splits=ns_old)

        def f_old():
            _ = int(ctx.max().item())
            pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=1, num_splits=ns_old)

        t_new = bench(f_new)
        t_new_sync = bench(f_new_sync)
        t_oldsplit = bench(f_oldsplit)
        t_old = bench(f_old)

        sync_delta = t_new_sync - t_new          # cost added by the sync
        split_delta = t_oldsplit - t_new         # penalty of old (fewer) splits
        total = t_old - t_new                     # full improvement new vs old
        print(f"{b:>6} {ctx_len:>8} | {ns_new:>10} {ns_old:>10} | "
              f"{t_new:>8.1f} {t_new_sync:>9.1f} {t_oldsplit:>9.1f} {t_old:>8.1f} | "
              f"{sync_delta:>+10.1f} {split_delta:>+11.1f} {total:>+7.1f}", flush=True)


if __name__ == "__main__":
    main()
