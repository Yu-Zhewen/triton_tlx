"""Head-to-head: TLX paged-decode kernel vs its aiter source (pa_decode_gluon).

Runs both kernels on identical logical inputs (same GPU), validates each against a
dense fp32 reference, and reports latency + KV bandwidth side by side.

The aiter kernel (aiter/ops/triton/gluon/pa_decode_gluon.py) only needs a tiny
`aiter` shim (dtypes + arch_info + types); its optional csrc/flydsl PS-reduce
backends are forced off so the pure-gluon reduce fallback is used.

Usage (inside tlxbuild container, from this dir, with pa_decode_gluon.py present):
    python3 bench_compare.py --check
    python3 bench_compare.py --bench
"""

import argparse
import os
import sys
import types as _types

import torch
import triton

# Canonical kernel now lives one level up in tutorials/; the input builder and
# dense reference are exported from that same module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amd_pa_decode import pa_decode_tlx, get_num_splits, build_inputs, ref_decode


# --------------------------------------------------------------------------- #
#  Minimal `aiter` shim so pa_decode_gluon.py imports without the full package
# --------------------------------------------------------------------------- #
def install_aiter_shim():
    import triton.language as tl

    def _mod(name):
        m = _types.ModuleType(name)
        sys.modules[name] = m
        return m

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]

    aiter = _mod("aiter")
    dtypes = _types.SimpleNamespace(
        fp8=torch.float8_e4m3fnuz,
        bf16=torch.bfloat16,
        fp16=torch.float16,
        fp32=torch.float32,
    )
    aiter.dtypes = dtypes

    ops = _mod("aiter.ops")
    triton_mod = _mod("aiter.ops.triton")
    utils = _mod("aiter.ops.triton.utils")
    _triton = _mod("aiter.ops.triton.utils._triton")
    arch_info = _mod("aiter.ops.triton.utils._triton.arch_info")
    arch_info.get_arch = lambda: arch
    types_mod = _mod("aiter.ops.triton.utils.types")
    types_mod.torch_to_triton_dtype = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.float8_e4m3fnuz: tl.float8e4b8,
    }

    aiter.ops = ops
    ops.triton = triton_mod
    triton_mod.utils = utils
    utils._triton = _triton
    _triton.arch_info = arch_info
    utils.types = types_mod


install_aiter_shim()
import pa_decode_gluon as aiter_pa  # noqa: E402

# Force the pure-gluon PS-reduce fallback (no csrc / no flydsl needed).
aiter_pa.CXX_PS_REDUCE_AVAILABLE = False


def _no_flydsl(*a, **k):
    raise ImportError("flydsl disabled")


aiter_pa.launch_pa_decode_ps_reduce_flydsl = _no_flydsl


def to_aiter_layout(key_cache, value_cache):
    """My layout -> aiter layout.
    key:   [B,H,P,D]      -> [B, H, D//x, P, x]
    value: [B,H,P,D]      -> [B, H, D, P]   (non-transposed)
    """
    B, H, P, D = key_cache.shape
    x = 16 // key_cache.dtype.itemsize  # 8 for bf16
    key = key_cache.reshape(B, H, P, D // x, x).permute(0, 1, 3, 2, 4).contiguous()
    val = value_cache.permute(0, 1, 3, 2).contiguous()
    return key, val


def run_aiter(out, query, key_cache, value_cache, context_lens, block_tables,
              sm_scale, query_length, num_kv_heads):
    key_a, val_a = to_aiter_layout(key_cache, value_cache)
    batch = query.shape[0] // query_length
    mcpn = aiter_pa.get_recommended_splits(batch, num_kv_heads)
    # default production path is ps=True; fall back to ps=False if unavailable
    for ps in (True, False):
        try:
            aiter_pa.pa_decode_gluon(
                out, query, key_a, val_a, context_lens, block_tables, sm_scale,
                query_length=query_length,
                max_context_partition_num=mcpn,
                context_partition_size=256,
                compute_type=torch.bfloat16,
                ps=ps,
            )
            return ps, mcpn
        except Exception:
            if ps is False:
                raise
    return None, mcpn


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
        ref = ref_decode(q, kc, vc, ctx, bt, sm_scale, num_q_heads, num_kv_heads,
                         c["query_length"])

        out_tlx = torch.empty_like(q)
        pa_decode_tlx(out_tlx, q, kc, vc, ctx, bt, sm_scale,
                      query_length=c["query_length"])
        e_tlx = (out_tlx.float() - ref).abs().max().item()
        r_tlx = e_tlx / (ref.abs().max().item() + 1e-6)

        out_ai = torch.empty_like(q)
        ps, mcpn = run_aiter(out_ai, q, kc, vc, ctx, bt, sm_scale,
                             c["query_length"], num_kv_heads)
        e_ai = (out_ai.float() - ref).abs().max().item()
        r_ai = e_ai / (ref.abs().max().item() + 1e-6)

        ok = r_tlx < 2e-2 and r_ai < 2e-2
        all_ok &= ok
        print(f"[{'OK ' if ok else 'BAD'}] qlen={c['query_length']} "
              f"ctx={c['ctx_lens']} | TLX rel={r_tlx:.2e} | "
              f"aiter(ps={ps},mcpn={mcpn}) rel={r_ai:.2e}", flush=True)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return all_ok


def run_bench(batches, contexts, query_lengths):
    device = "cuda"
    head_dim, page_size = 128, 64
    num_kv_heads, group = 8, 8
    num_q_heads = num_kv_heads * group
    sm_scale = 1.0 / (head_dim ** 0.5)

    hdr = (f"{'batch':>6} {'ctx':>8} {'qlen':>5} | "
           f"{'tlx_us':>9} {'tlx_GB/s':>9} | {'aiter_us':>9} {'ai_GB/s':>9} | "
           f"{'speedup':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for qlen in query_lengths:
        for ctx_len in contexts:
            for b in batches:
                pool = 4 * ((ctx_len + page_size - 1) // page_size) + 16
                q, kc, vc, ctx, bt = build_inputs(
                    b, [ctx_len] * b, num_q_heads, num_kv_heads, head_dim,
                    page_size, query_length=qlen, device=device, pool_pages=pool)
                kv_bytes = 2 * b * num_kv_heads * ctx_len * head_dim * 2

                out_tlx = torch.empty_like(q)

                def run_t():
                    pa_decode_tlx(out_tlx, q, kc, vc, ctx, bt, sm_scale,
                                  query_length=qlen)

                key_a, val_a = to_aiter_layout(kc, vc)
                mcpn = aiter_pa.get_recommended_splits(b, num_kv_heads)
                out_ai = torch.empty_like(q)

                # resolve which ps path works once, then time it
                ps_used, _ = run_aiter(out_ai, q, kc, vc, ctx, bt, sm_scale,
                                       qlen, num_kv_heads)

                def run_a():
                    aiter_pa.pa_decode_gluon(
                        out_ai, q, key_a, val_a, ctx, bt, sm_scale,
                        query_length=qlen, max_context_partition_num=mcpn,
                        context_partition_size=256, compute_type=torch.bfloat16,
                        ps=ps_used)

                try:
                    t_ms = triton.testing.do_bench(run_t, warmup=25, rep=100)
                    tlx_gbs = kv_bytes / (t_ms * 1e-3) / 1e9
                    tlx_s = f"{t_ms*1e3:>9.1f}"
                    tlx_bw = f"{tlx_gbs:>9.1f}"
                except Exception as e:
                    tlx_s, tlx_bw, t_ms = f"ERR", f"{type(e).__name__}"[:9], None

                try:
                    a_ms = triton.testing.do_bench(run_a, warmup=25, rep=100)
                    ai_gbs = kv_bytes / (a_ms * 1e-3) / 1e9
                    ai_s = f"{a_ms*1e3:>9.1f}"
                    ai_bw = f"{ai_gbs:>9.1f}"
                except Exception as e:
                    ai_s, ai_bw, a_ms = f"ERR", f"{type(e).__name__}"[:9], None

                if t_ms and a_ms:
                    spd = f"{a_ms / t_ms:>7.2f}x"
                else:
                    spd = f"{'-':>8}"
                print(f"{b:>6} {ctx_len:>8} {qlen:>5} | {tlx_s} {tlx_bw} | "
                      f"{ai_s} {ai_bw} | {spd}", flush=True)


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
