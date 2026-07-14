"""Standalone runner/benchmark for aiter pa_decode_gluon (no TLX deps).

Builds paged decode inputs, validates aiter output vs a dense fp32 reference,
and times it. Meant to run in a container with a compatible ROCm/upstream triton
(gluon cdna3/cdna4). Requires pa_decode_gluon.py in the same directory.
"""

import argparse
import sys
import types as _types

import torch
import triton


def install_aiter_shim():
    import triton.language as tl

    def _mod(name):
        m = _types.ModuleType(name)
        sys.modules[name] = m
        return m

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    aiter = _mod("aiter")
    aiter.dtypes = _types.SimpleNamespace(
        fp8=torch.float8_e4m3fnuz, bf16=torch.bfloat16,
        fp16=torch.float16, fp32=torch.float32)
    _mod("aiter.ops")
    _mod("aiter.ops.triton")
    _mod("aiter.ops.triton.utils")
    _mod("aiter.ops.triton.utils._triton")
    arch_info = _mod("aiter.ops.triton.utils._triton.arch_info")
    arch_info.get_arch = lambda: arch
    types_mod = _mod("aiter.ops.triton.utils.types")
    types_mod.torch_to_triton_dtype = {
        torch.bfloat16: tl.bfloat16, torch.float16: tl.float16,
        torch.float32: tl.float32, torch.float8_e4m3fnuz: tl.float8e4b8}
    aiter.ops = sys.modules["aiter.ops"]
    aiter.ops.triton = sys.modules["aiter.ops.triton"]
    aiter.ops.triton.utils = sys.modules["aiter.ops.triton.utils"]
    aiter.ops.triton.utils._triton = sys.modules["aiter.ops.triton.utils._triton"]
    aiter.ops.triton.utils._triton.arch_info = arch_info
    aiter.ops.triton.utils.types = types_mod


install_aiter_shim()
import pa_decode_gluon as aiter_pa  # noqa: E402

aiter_pa.CXX_PS_REDUCE_AVAILABLE = False


def _no_flydsl(*a, **k):
    raise ImportError("flydsl disabled")


aiter_pa.launch_pa_decode_ps_reduce_flydsl = _no_flydsl


def build_inputs(num_seqs, ctx_lens, num_q_heads, num_kv_heads, head_dim,
                 page_size, query_length=1, dtype=torch.bfloat16, device="cuda",
                 seed=0, pool_pages=None):
    torch.manual_seed(seed)
    num_tokens = num_seqs * query_length
    query = torch.randn(num_tokens, num_q_heads, head_dim, dtype=dtype,
                        device=device) * 0.2
    max_pages = (max(ctx_lens) + page_size - 1) // page_size
    distinct = num_seqs * max_pages
    total_pages = distinct if pool_pages is None else min(distinct, pool_pages)
    key_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim,
                            dtype=dtype, device=device) * 0.2
    value_cache = torch.randn(total_pages, num_kv_heads, page_size, head_dim,
                              dtype=dtype, device=device) * 0.2
    block_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32,
                               device=device)
    for s in range(num_seqs):
        npag = (ctx_lens[s] + page_size - 1) // page_size
        for p in range(max_pages):
            phys = (s * max_pages + (p if p < npag else 0)) % total_pages
            block_tables[s, p] = phys
    context_lens = torch.tensor(ctx_lens, dtype=torch.int32, device=device)
    return query, key_cache, value_cache, context_lens, block_tables


def ref_decode(query, key_cache, value_cache, context_lens, block_tables,
               sm_scale, num_q_heads, num_kv_heads, query_length):
    head_dim = query.shape[-1]
    page_size = key_cache.shape[2]
    group = num_q_heads // num_kv_heads
    num_seqs = query.shape[0] // query_length
    out = torch.empty_like(query, dtype=torch.float32)
    for s in range(num_seqs):
        ctx = int(context_lens[s].item())
        npag = (ctx + page_size - 1) // page_size
        phys = block_tables[s, :npag]
        k = key_cache[phys].to(torch.float32)
        v = value_cache[phys].to(torch.float32)
        k = k.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        v = v.permute(1, 0, 2, 3).reshape(num_kv_heads, npag * page_size, head_dim)[:, :ctx]
        for qpos in range(query_length):
            gt = s * query_length + qpos
            limit = ctx - query_length + qpos
            for qh in range(num_q_heads):
                kvh = qh // group
                q = query[gt, qh].to(torch.float32)
                scores = (q[None, :] * k[kvh]).sum(-1) * sm_scale
                scores = scores[: limit + 1]
                p = torch.softmax(scores, dim=0)
                out[gt, qh] = (p[:, None] * v[kvh, : limit + 1]).sum(0)
    return out


def to_aiter_layout(key_cache, value_cache):
    B, H, P, D = key_cache.shape
    x = 16 // key_cache.dtype.itemsize
    key = key_cache.reshape(B, H, P, D // x, x).permute(0, 1, 3, 2, 4).contiguous()
    val = value_cache.permute(0, 1, 3, 2).contiguous()
    return key, val


def run_aiter(out, query, key_a, val_a, context_lens, block_tables, sm_scale,
              query_length, num_kv_heads, ps):
    batch = query.shape[0] // query_length
    mcpn = aiter_pa.get_recommended_splits(batch, num_kv_heads)
    aiter_pa.pa_decode_gluon(
        out, query, key_a, val_a, context_lens, block_tables, sm_scale,
        query_length=query_length, max_context_partition_num=mcpn,
        context_partition_size=256, compute_type=torch.bfloat16, ps=ps)
    return mcpn


def resolve_ps(out, query, key_a, val_a, ctx, bt, sm_scale, qlen, num_kv_heads):
    for ps in (True, False):
        try:
            run_aiter(out, query, key_a, val_a, ctx, bt, sm_scale, qlen,
                      num_kv_heads, ps)
            return ps
        except Exception as e:
            if ps is False:
                raise
    return None


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
        dict(num_seqs=4, ctx_lens=[1000, 4096, 8192, 9000], query_length=1),
    ]
    all_ok = True
    for c in cases:
        q, kc, vc, ctx, bt = build_inputs(
            c["num_seqs"], c["ctx_lens"], num_q_heads, num_kv_heads, head_dim,
            page_size, query_length=c["query_length"], device=device)
        ref = ref_decode(q, kc, vc, ctx, bt, sm_scale, num_q_heads,
                         num_kv_heads, c["query_length"])
        key_a, val_a = to_aiter_layout(kc, vc)
        out = torch.empty_like(q)
        try:
            ps = resolve_ps(out, q, key_a, val_a, ctx, bt, sm_scale,
                            c["query_length"], num_kv_heads)
        except Exception as e:
            print(f"[BAD] qlen={c['query_length']} ctx={c['ctx_lens']} "
                  f"EXC {type(e).__name__}: {str(e)[:120]}", flush=True)
            all_ok = False
            continue
        e_ai = (out.float() - ref).abs().max().item()
        r_ai = e_ai / (ref.abs().max().item() + 1e-6)
        ok = r_ai < 2e-2
        all_ok &= ok
        print(f"[{'OK ' if ok else 'BAD'}] qlen={c['query_length']} "
              f"ctx={c['ctx_lens']} ps={ps} rel={r_ai:.2e}", flush=True)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return all_ok


def _warmup_gpu(device="cuda", secs=3.0):
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
    print(f"{'batch':>6} {'ctx':>8} {'qlen':>5} {'ps':>4} {'us':>10} {'GB/s':>9}",
          flush=True)
    for qlen in query_lengths:
        for ctx_len in contexts:
            for b in batches:
                pool = 4 * ((ctx_len + page_size - 1) // page_size) + 16
                q, kc, vc, ctx, bt = build_inputs(
                    b, [ctx_len] * b, num_q_heads, num_kv_heads, head_dim,
                    page_size, query_length=qlen, device=device, pool_pages=pool)
                key_a, val_a = to_aiter_layout(kc, vc)
                out = torch.empty_like(q)
                kv_bytes = 2 * b * num_kv_heads * ctx_len * head_dim * 2
                try:
                    ps = resolve_ps(out, q, key_a, val_a, ctx, bt, sm_scale,
                                    qlen, num_kv_heads)
                    mcpn = aiter_pa.get_recommended_splits(b, num_kv_heads)

                    def run():
                        aiter_pa.pa_decode_gluon(
                            out, q, key_a, val_a, ctx, bt, sm_scale,
                            query_length=qlen, max_context_partition_num=mcpn,
                            context_partition_size=256,
                            compute_type=torch.bfloat16, ps=ps)

                    ms = triton.testing.do_bench(run, warmup=25, rep=100)
                    gbs = kv_bytes / (ms * 1e-3) / 1e9
                    print(f"{b:>6} {ctx_len:>8} {qlen:>5} {str(ps):>4} "
                          f"{ms*1e3:>10.1f} {gbs:>9.1f}", flush=True)
                except Exception as e:
                    print(f"{b:>6} {ctx_len:>8} {qlen:>5}  ERR "
                          f"{type(e).__name__}: {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[1024, 8192, 32768, 131072])
    ap.add_argument("--qlens", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    if not args.check and not args.bench:
        args.check = True
    if args.check:
        run_check()
    if args.bench:
        run_bench(args.batches, args.contexts, args.qlens)
