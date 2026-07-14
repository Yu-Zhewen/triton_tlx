# TLX paged-attention decode kernel (AMD CDNA4 / gfx950)

A two-phase split-K paged-attention **decode** kernel written with Triton + TLX,
ported from the algorithm in ROCm/aiter `pa_decode_gluon`. Targets AMD MI350
(gfx950 / CDNA4). Supports bf16/fp16 KV, GQA, and multi-token prediction
(query_length 1–4) with causal masking across the query positions.

## Files

| File | Description |
|------|-------------|
| `amd_pa_decode.py` | The kernel. Phase 1 (`_pa_decode_partition_kernel`): async double-buffered KV streaming into LDS, MFMA QK^T, base-2 online softmax, P@V, per-split partial output + LSE. Phase 2 (`_pa_decode_reduce_kernel`): LSE merge of split partials. Host entry: `pa_decode_tlx(...)`. |
| `bench_pa_decode.py` | Correctness (`--check`, vs a dense fp32 reference) and perf sweep (`--bench`) harness for the TLX kernel. |
| `aiter_only.py` | Standalone runner/benchmark for the aiter `pa_decode_gluon` reference (needs `pa_decode_gluon.py` present at runtime + a tiny `aiter` shim it installs internally). Used for the head-to-head. |
| `bench_compare.py` | Single-container TLX-vs-aiter comparison (both kernels, same inputs). |
| `bench_floor.py` | Decomposes the batch-1 latency floor into its host/kernel components. |
| `bench_ablation.py` | Warm, controlled ablation of the `.item()` sync removal vs the split-count change. |
| `pa_decode_compare.md` | Full TLX-vs-aiter latency comparison on MI350 (144 configs: batch × context × query_length). |

## Usage

Run inside a container with Triton + TLX (`triton.language.extra.tlx`):

```bash
python3 bench_pa_decode.py --check          # correctness on small shapes
python3 bench_pa_decode.py --bench          # TLX perf sweep
# full intended suite:
python3 bench_pa_decode.py --bench \
    --batches 1 2 4 8 16 32 64 128 256 \
    --contexts 1024 8192 32768 131072 \
    --qlens 1 2 3 4
```

The aiter reference (`aiter_only.py`) is designed to run in a container with an
upstream/ROCm Triton (gluon cdna4 working); the Meta `fb.beta` fork's AMD gluon
`mfma` path is currently broken.

## Results summary

On MI350 (gfx950), warm, same GPU, identical bf16 inputs: TLX beats aiter
`pa_decode_gluon` across nearly all configs — typically **1.6–2.7×** at batch ≥ 8
(memory-bandwidth-bound), sustaining **~7 TB/s (~88% of HBM3E peak)** at long
context vs aiter's ~2.7–2.9 TB/s. See `pa_decode_compare.md` for the full table.
