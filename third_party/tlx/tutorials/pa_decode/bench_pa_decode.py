"""Correctness + benchmark harness for the TLX paged-attention decode kernel.

Usage (inside the tlxbuild container, from a neutral dir):
    python3 bench_pa_decode.py --check          # correctness on small shapes
    python3 bench_pa_decode.py --bench           # perf sweep
"""

import argparse
import torch

import triton

from amd_pa_decode import pa_decode_tlx, get_num_splits


def build_inputs(num_seqs, ctx_lens, num_q_heads, num_kv_heads, head_dim, page_size,
                 query_length=1, dtype=torch.bfloat16, device="cuda", seed=0,
                 pool_pages=None):
    """Build paged decode inputs. If pool_pages is set, physical pages are drawn
    from a shared pool of that size (bounds memory for large sweeps); the dense
    reference uses the same block_tables so correctness is unaffected."""
    torch.manual_seed(seed)
    assert len(ctx_lens) == num_seqs
    num_tokens = num_seqs * query_length

    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype, device=device) * 0.2

    max_pages = (max(ctx_lens) + page_size - 1) // page_size
    distinct = num_seqs * max_pages
    total_pages = distinct if pool_pages is None else min(distinct, pool_pages)
    key_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device) * 0.2
    value_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device) * 0.2

    block_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32, device=device)
    for s in range(num_seqs):
        npag = (ctx_lens[s] + page_size - 1) // page_size
        for p in range(max_pages):
            phys = (s * max_pages + (p if p < npag else 0)) % total_pages
            block_tables[s, p] = phys
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    return query, key_cache, value_cache, context_lens, block_tables


def ref_decode(query, key_cache, value_cache, context_lens, block_tables, sm_scale,
               num_q_heads, num_kv_heads, query_length):
    """Dense fp32 reference: gather full K/V from page table, causal over qlen."""
    device = query.device
    head_dim = query.shape[-1]
    page_size = key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    num_seqs = query.shape[0] // query_length
    out = torch.empty_like(query, dtype=torch.float32)

    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        npag = (ctx + page_size - 1) // page_size
        phys = block_tables[s, :npag]
        k = key_cache[phys].to(torch.float32)      # [npag, kvh, page, d]
        v = value_cache[phys].to(torch.float32)
        k = k.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        v = v.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        for qpos in range(query_length):
            gt = s * query_length + qpos
            limit = ctx - query_length + qpos       # inclusive last visible key index
            for qh in range(num_q_heads):
                kvh = qh // group
                q = query[gt, qh].to(torch.float32)         # [d]
                scores = (q[None, :] * k[kvh]).sum(-1) * sm_scale  # [ctx]
                scores = scores[: limit + 1]
                p = torch.softmax(scores, dim=0)
                out[gt, qh] = (p[:, None] * v[kvh, : limit + 1]).sum(0)
    return out


def run_check():
    device = "cuda"
    head_dim, page_size = 128, 64
    num_kv_heads, group = 2, 8
    num_q_heads = num_kv_heads * group
    sm_scale = 1.0 / (head_dim ** 0.5)

    cases = [
        dict(num_seqs=2, ctx_lens=[130, 200], query_length=1),
        dict(num_seqs=3, ctx_lens=[64, 256, 500], query_length=1),
        dict(num_seqs=2, ctx_lens=[300, 777], query_length=2),
        dict(num_seqs=2, ctx_lens=[512, 1024], query_length=4),
        dict(num_seqs=4, ctx_lens=[100, 1000, 4096, 8192], query_length=1),
    ]
    all_ok = True
    for c in cases:
        for num_splits in (1, 4):
            q, kc, vc, ctx, bt = build_inputs(
                c["num_seqs"], c["ctx_lens"], num_q_heads, num_kv_heads, head_dim, page_size,
                query_length=c["query_length"], device=device)
            out = torch.empty_like(q)
            pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale,
                          query_length=c["query_length"], num_splits=num_splits)
            ref = ref_decode(q, kc, vc, ctx, bt, sm_scale, num_q_heads, num_kv_heads, c["query_length"])
            err = (out.to(torch.float32) - ref).abs().max().item()
            rel = err / (ref.abs().max().item() + 1e-6)
            ok = rel < 2e-2
            all_ok &= ok
            print(f"[{'OK ' if ok else 'BAD'}] qlen={c['query_length']} splits={num_splits} "
                  f"ctx={c['ctx_lens']} max_abs_err={err:.4e} rel={rel:.4e}")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return all_ok


def _warmup_gpu(device="cuda", secs=3.0):
    """Spin the GPU so clocks are fully boosted before timing (avoids the
    first-config cold-clock penalty on sub-100us cells)."""
    import time
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    t0 = time.time()
    while time.time() - t0 < secs:
        for _ in range(20):
            a = (a @ b) * 1e-4 + 1.0
    torch.cuda.synchronize()


def run_bench(batches, contexts, query_lengths):
    device = "cuda"
    head_dim, page_size = 128, 64
    num_kv_heads, group = 8, 8
    num_q_heads = num_kv_heads * group
    sm_scale = 1.0 / (head_dim ** 0.5)

    _warmup_gpu(device)
    print(f"{'batch':>6} {'ctx':>8} {'qlen':>5} {'splits':>7} {'us':>10} {'GB/s':>9}", flush=True)
    for qlen in query_lengths:
        for ctx_len in contexts:
            for b in batches:
                # shared page pool bounds memory (values irrelevant for timing)
                pool = 4 * ((ctx_len + page_size - 1) // page_size) + 16
                q, kc, vc, ctx, bt = build_inputs(
                    b, [ctx_len] * b, num_q_heads, num_kv_heads, head_dim, page_size,
                    query_length=qlen, device=device, pool_pages=pool)
                out = torch.empty_like(q)

                def run():
                    pa_decode_tlx(out, q, kc, vc, ctx, bt, sm_scale, query_length=qlen)

                try:
                    ms = triton.testing.do_bench(run, warmup=25, rep=100)
                except Exception as e:
                    print(f"{b:>6} {ctx_len:>8} {qlen:>5}  ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
                    continue
                # bytes: KV read = 2 * b * num_kv_heads * ctx * head_dim * 2 bytes (bf16)
                kv_bytes = 2 * b * num_kv_heads * ctx_len * head_dim * 2
                gbs = kv_bytes / (ms * 1e-3) / 1e9
                ns = get_num_splits(b, num_kv_heads, ctx_len, page_size)
                print(f"{b:>6} {ctx_len:>8} {qlen:>5} {ns:>7} {ms*1e3:>10.1f} {gbs:>9.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192, 32768, 131072])
    ap.add_argument("--qlens", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    if not args.check and not args.bench:
        args.check = True
    if args.check:
        run_check()
    if args.bench:
        run_bench(args.batches, args.contexts, args.qlens)
