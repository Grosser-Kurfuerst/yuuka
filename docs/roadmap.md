# Decode optimization roadmap

**Objective.** Close the single-request decode gap against the llama.cpp
reference on gfx1151 for `Qwen3.5-4B` Q4_K_M, and make decode throughput
independent of context length. The proposed acceptance target is **60 tokens/s
or more at a 128-token prompt and 55 tokens/s or more at a 2048-token prompt**
with greedy sampling on one request, which is parity with the reference
measured below.

**Status: closed.** All seven steps have landed and the record is re-recorded
in [benchmarks.md](benchmarks.md) and [kernels.md](kernels.md). The objective is
met: `pp128/tg128` 60.70 t/s (ratio 0.98 to llama.cpp), `pp512/tg128` 60.41, and
`pp2048/tg32` 58.60, and the decode rate is flat in context. Two intermediate
rows were not met and are recorded as measured: the 190 GB/s per-shape bandwidth
(the wide shapes reach 179-230 GB/s and the 1024- and 4096-input projections
stay at 92-158) and the 15.5 ms step (17.1 ms). Step 5's int8 re-evaluation
failed its gate and the FP32 route stays selected; step 6 cut 0.25 ms of its
0.5 ms tail row. What remains is the narrow-projection per-row cost, the
launch-bound norm tail, and multi-request batching, which was out of scope. This file is a plan, not a record: when a step lands, its
measured result goes into [benchmarks.md](benchmarks.md) and the row in
[implementation order](#implementation-order) is updated here.

**Scope.** The dense hybrid `qwen35` HIP path at batch 1. Fused mixed-shape
multi-request decode batching is out of scope; it is the separate 16x aggregate
gap recorded in [benchmarks.md](benchmarks.md#openai-server). Prefill, sampling,
the scheduler, and the C ABI are unchanged by this plan.

## Baseline and attribution

### Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-19 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs (HIP reports 20 WGPs), Wave32, 2900 MHz, 120 GiB shared memory |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| yuuka | commit `7b7aa4c`, Go 1.24.4, `cmake --build build` (Release) |
| llama.cpp reference | `../llama.cpp` build `b1b770d90` (12359), ROCm backend, `-ngl 99 -b 512 -ub 512` |
| Sampling | Greedy, temperature 0 |
| Warmup / repetitions | as stated per table; medians |
| Clock | not pinned; one benchmark process on the GPU at a time |

The device clock is not pinned, so absolute values move a few percent between
sessions. Every claim below was measured in one session, side by side, with the
numbers and repetitions stated. `llama-bench` rejects an `f32` KV cache type, so
llama.cpp rows use its default `f16` KV; KV format is inside noise for decode at
these lengths ([KV cache formats](benchmarks.md#kv-cache-formats)).

### Headline gap

`yuuka bench --backend hip --kv-type f32 --chunk-size 512` against
`llama-bench -ngl 99 -b 512 -ub 512`, same model, same device:

| Test | yuuka | llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| pp128 / tg128 | 38.63 t/s | 61.99 t/s | 0.62 |
| pp128 / tg32 (context 160) | 41.24 t/s | 61.30 t/s | 0.67 |
| pp128 / tg512 (context 640) | 34.37 t/s | 61.94 t/s | 0.55 |
| pp512 / tg128 | 31.37 t/s | 61.84 t/s | 0.51 |
| pp2048 / tg32 | 18.44 t/s | 61.49 t/s | 0.30 |

The reference is flat in generated length and in context (61.3–61.9 t/s across
all five rows). yuuka loses 6.9 t/s between 32 and 512 generated tokens on the
same prompt, which is context, not run length.

### Where one decode step goes

yuuka: `TestGPUProfileDecode` (`engine/gpu/decode_profile_test.go`), one
profiled step, 379 plan operations. Profiling adds a submission gap per
operation, so the instrumented total is 0.7 ms above the unprofiled loop; the
unprofiled `Session.Decode` loop measured 23.40 ms/step at a 64-token context
and 30.15 ms/step at 512 tokens.

| Group | ops | ms (ctx 7) | ms (ctx 512) |
| --- | ---: | ---: | ---: |
| matvec family (`lm_head`, `gemv_residual`, `gemv_pair`, `gemv`) | 217 | 21.02 | 20.98 |
| decode attention (`attention_paged`) | 8 | 0.14 | **8.31** |
| RMSNorm (row and head) | 81 | 0.90 | 0.91 |
| gated delta-net (recurrent + conv/L2) | 48 | 0.81 | 0.65 |
| `ssm_alpha` / `ssm_beta` (32 outputs each) | 48 | 0.43 | 0.43 |
| RoPE, KV write, embedding | 25 | 0.15 | 0.15 |
| **profiled total** | 379 | **23.42** | **31.02** |

llama.cpp: `rocprofv3 --kernel-trace` around `llama-bench -p 0 -n 32 -r 2`,
65 decode steps, grouped by kernel name and grid, per step:

| Kernel | dispatches/step | ms/step | MB/step |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_q` (all dtypes) | 217 | **13.02** | 2,726 |
| `quantize_q8_1` | 217 | 0.29 | 10 |
| `k_get_rows_float_vec` (embedding) | 1 | 0.27 | 1.3 |
| `gated_delta_net_cuda` | 24 | 0.23 | 101 |
| `rms_norm_f32` (four variants) | 65 | 0.37 | – |
| `cpy_scalar`, `bin_bcast`, `concat`, `scale`, `set_rows` | 85 | 0.24 | – |
| activation kernels (`silu`, `sigmoid`, `softplus`, `rope`) | 60 | 0.19 | – |
| `flash_attn_tile` + `flash_attn_combine_results` | 8 + 8 | 0.067 | – |
| `ssm_conv_f32` | 24 | 0.06 | 10 |
| **kernel total** | ~700 | **14.75** | |
| wall clock (`llama-bench` tg) | | **16.13** | |

Two conclusions follow from the two tables.

1. The matvec family is 89.8% of the yuuka step and it is **8.0 ms more
   expensive than the reference for exactly the same 2,726 MB of weights**:
   129.7 GB/s effective against 209 GB/s. This part is context independent.
2. Decode attention costs 8.2 ms more at a 512-token context and **~104x more
   per layer than the reference at the same context**. This part is the whole
   reason the gap widens with context, and it is why the 2048-token row is at
   0.30 instead of 0.51.

Everything else is within 0.4 ms: the delta-net pair, the norms, RoPE, KV
write, the embedding lookup, the logits download (1 MB device-to-host measured
at 0.018 ms), and the sampler (a plain argmax). **Per-token host overhead is not
a factor**: the unprofiled yuuka loop is within 0.5 ms of the sum of its kernel
times, and llama.cpp carries 1.3 ms of the same gap. The older note in
[benchmarks.md](benchmarks.md#decode-output-length) that puts "3 ms per token"
on the host side no longer applies.

### Per-shape weight bandwidth

The same step, per projection. yuuka values are the in-step per-tensor GB/s of
the profiled step; llama.cpp values are the per-dispatch kernel time of the
trace divided by the same matrix bytes. "Blocks" is the grid of the launched
kernel per operation: yuuka launches one block per 32 output channels, the
reference one block per 2 output rows.

| Projection | dtype | K → N | yuuka blocks | yuuka GB/s | ref blocks | ref GB/s | ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `attn_k`, `attn_v` | Q4_K | 2560 → 1024 | 32 | 49 | 512 | 174 | 0.28 |
| `attn_v` | Q6_K | 2560 → 1024 | 32 | 74 | 512 | 186 | 0.40 |
| `attn_output` | Q4_K | 4096 → 2560 | 80 | 96 | 1280 | 185 | 0.52 |
| `ffn_down` | Q4_K | 9216 → 2560 | 80 | 96 | 1280 | 185 | 0.52 |
| `ssm_out` | Q5_K | 4096 → 2560 | 80 | 103 | 1280 | 210 | 0.49 |
| `ffn_down` | Q6_K | 9216 → 2560 | 80 | 138 | 1280 | 192 | 0.72 |
| `attn_gate` | Q4_K | 2560 → 4096 | 128 | 113 | 2048 | 202 | 0.56 |
| `attn_q` | Q4_K | 2560 → 8192 | 256 | 123 | 4096 | 212 | 0.58 |
| `attn_qkv` | Q5_K | 2560 → 8192 | 256 | 132 | 4096 | 219 | 0.60 |
| `ffn_gate`, `ffn_up` (paired) | Q4_K | 2560 → 9216 ×2 | 288 | 141 | 4608 | 219 | 0.64 |
| `lm_head` (tied `token_embd`) | Q6_K | 2560 → 248320 | 7,760 | 186 | 124,160 | 235 | 0.79 |
| `ssm_alpha`, `ssm_beta` | Q8_0 | 2560 → 32 | 1 | 10 | 16 | 19 | 0.53 |

The reference is flat in `N` (174–235 GB/s) and flat in `K` (at `N` = 2560:
`K` = 4096 gives 185 GB/s and `K` = 9216 gives 185 GB/s). yuuka's bandwidth is a
monotone function of the **grid size**: 49 GB/s at 32 blocks, 141 at 288, 186
at 7,760. `K` does
not matter (at `N` = 2560: 96 GB/s for both `K` = 4096 and `K` = 9216).

That the small-`N` loss is not a DRAM effect is confirmed by yuuka's own
isolated sweep: the `2560 → 1024` Q4_K matrix (1.47 MB, entirely inside the
32 MB MALL) reaches only **65 GB/s** in `yuuka_gemv_decode_sweep --threshold`
with the matrix resident, against the reference's 174 GB/s on the same matrix
inside a 2.7 GB rotation. The kernel is limited by concurrency, not by the
memory system.

### Attention scaling

`attention_paged` per layer, yuuka, from the profiled step at three contexts,
against the reference's `flash_attn_tile` + combine from the two traces (the
527-token figure is the last decode step of a `-p 512 -n 16` run):

| Context | yuuka per layer | yuuka 8 layers/token | ref per layer | ref 8 layers/token |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.017 ms | 0.14 ms | – | 0.067 ms |
| 64 | 0.167 ms | 1.34 ms | – | ~0.07 ms |
| 512 | 1.04 ms | 8.31 ms | 0.0105 ms | 0.084 ms |

yuuka's cost is linear in `length` with a slope of about **1.9 µs per position
per layer**; the reference's is flat at about 10 µs per layer. Per generated
token the difference is 8.2 ms at 512 tokens and about 33 ms at 2048 tokens,
which reproduces the measured 18.44 t/s at `pp2048`.

## Root causes

### R-A. Decode attention is serial in position

`kernels/hip/attention/attention.hip:69` launches
`attention_paged_f32_kernel<<<heads, 256>>>`: **16 blocks for a 40-CU device**,
one block per query head, and inside the block the key/value positions are a
strictly sequential loop. Every position costs one dependent global load of a
KV row, a 256-thread tree reduction, a write of the score to global memory, and
two block barriers; the softmax is then computed by `threadIdx.x == 0` alone
over `length` values, and the value accumulation is a second sequential pass of
dependent loads. There is no tiling, no batching of keys, no online softmax, no
split over the key range, and no per-warp reduction. The latency of one DRAM
round trip is exposed once per position per head, which is exactly the measured
1.9 µs slope per layer.

The reference (`ggml/src/ggml-cuda/fattn-tile.cuh:790`,
`fattn-common.cuh:1313`) splits the key range into `parallel_blocks`, keeps an
online softmax (`KQ_max`) in registers, loads K and V in `half2` batches of
`nbatch_K` keys into shared memory, computes both products with WMMA, and
merges the partial results with `flash_attn_combine_results`. Its cost is
independent of context up to the point where the key tiles no longer fit the
grid.

The same serial design exists in the plain `attention_paged_f32` path, the
dynamic-parameter (graph) variant, and the F16 and Q8_0 pool variants, so a fix
must be written once and dispatched by pool format.

### R-B. The decode matvec cannot fill the device or the memory pipeline

`kernels/hip/gemm/lm_head.hip:42` fixes the mapping: 8 warps per block, 4 output
channels per wave, `kStepK = 256`. Consequences, in order of size:

1. **Grid.** One block per 32 output channels. At `N` = 1024 that is 32 blocks
   on 40 CUs; at `N` = 2560 it is 80. A block is 256 threads, so the smallest
   projections run less than one block per CU and the mid-size ones run two,
   which is not enough to cover DRAM latency with only the loads one warp can
   have outstanding. Measured bandwidth tracks this exactly
   ([per-shape table](#per-shape-weight-bandwidth)).
2. **Load width.** A lane covers 4 consecutive `K` values, which is 2 bytes of
   Q4_K weights and 4 bytes of Q6_K; a warp therefore moves 64 bytes per load
   instruction. The reference loads 8–16 bytes per lane (256–512 bytes per warp
   instruction), so it has four to eight times more bytes in flight per issue
   slot for the same number of outstanding requests.
3. **Barriers in `K`.** The activation tile is staged in shared memory once per
   `kStepK`, with a `__syncthreads` before and after, so `K` = 9216 pays 36
   barriers per block and every warp waits on the staging load each step. The
   reference has no barrier inside the `K` loop; the only reduction is one warp
   shuffle tree at the end of the row.
4. **Arithmetic per weight.** The kernel dequantizes to FP32 and issues 8.5–13.8
   VALU per packed weight. The reference pre-quantizes the activation row to
   Q8_1 (`quantize_q8_1`, 217 launches and 0.29 ms per token) and then runs one
   `v_dot4` per four weights, about 0.5 VALU per weight. Neither kernel is ALU
   bound in aggregate on this device, but the decode of four separate 2-byte
   loads plus the FP32 conversion is what keeps the load width at 2 bytes.

The existing int8 route (`kernels/hip/gemv/gemv_i8.hip`, selected by
`backend/hip/dispatch.go:203`) keeps the same block mapping and the same
staging barrier, so it saves the ALU work and nothing else; that is why it
measured 6% faster in step and 5% slower end to end. It is not evidence that an
integer dot product cannot win — it is evidence that the mapping, not the
arithmetic, is the limiter.

### R-C. Small tail

`rmsnorm` costs 12 µs per row of 2560 values (81 launches, 0.9 ms per token)
against the reference's 5.7 µs average; `ssm_alpha` and `ssm_beta` (32 outputs
each) cost 9 and 11 µs per launch because the channel-parallel kernel runs a
single block for them and the per-row fallback is used instead. Together they
are about 0.6 ms per token, roughly the size of the whole post-fix difference
between the two engines, so they belong at the end of the plan rather than at
the front.

## Implementation order

Each row is an independently measurable change with its own gate. Record the
commit, the correctness result and the before/after medians when a row lands.
Steps 1–3 close the context-dependent gap; steps 4–5 close the flat gap; steps
6–7 are the tail and the record.

| Order | Deliverable | Gate to advance | Estimated scope |
| --- | --- | --- | --- |
| 1 | [Reproduce the comparison harness](#step-1-reproduce-the-comparison) | One command prints both engines at 128/512/2048 and the per-shape table | Script and docs |
| | *Landed.* `scripts/decode_compare.sh` plus `scripts/decode_reference_shapes.py` reproduce the headline rows (0.63/0.66/0.55/0.50/0.28 on the 2026-11-19 rerun) and the per-shape table. | | |
| | *Landed (step 2).* Split-key attention: 0.033 ms per layer at a 512-token context in `TestGPUProfileDecode`, `pp2048/tg32` 42.90 t/s, greedy tokens identical, logits within the recorded CPU difference. | | |
| | *Landed (step 3).* F32/F16/Q8_0 and the captured graph take the same kernel: 0.0335/0.0353/0.0318 ms per layer at 512 tokens, replayed tokens identical, and 0.096/0.071/0.061 ms at 2048. The 0.06 ms row at 2048 holds for Q8_0; see the step note for the F32/F16 byte floor. | | |
| | *Landed (step 4).* One wave per output row, sixteen K values per lane, no barrier inside K: matvec 130.3 -> 176.6 GB/s effective, step 23.4 -> 17.3 ms, `pp128/tg128` 38.8 -> 60.16 t/s, `pp2048/tg32` 17.4 -> 58.16. Per-shape 190 GB/s and the 15.5 ms step are not met; see the step note. | | |
| | *Landed (step 5).* The Q8_1 int8 route on the remapped mapping measures 175.3 GB/s in step against 176.6 and 0.67 ms slower per step, so the gate fails and the FP32 route stays selected. The negative result and the per-shape cells are recorded. | | |
| | *Landed (step 6).* Norm kernels reduce with wave shuffles and the beta/alpha projections share one launch: the step drops 17.31 -> 17.10 ms profiled and `pp128/tg128` 60.16 -> 60.70 t/s, short of the 0.5 ms row because the remaining tail is launch bound. | | |
| 2 | [Parallel decode attention over the key range](#step-2-parallelize-decode-attention-over-the-key-range) | 512-token context: ≤ 0.05 ms per layer, top-1 token and logits within the documented tolerance, `pp2048` ≥ 30 t/s | 4–8 files |
| 3 | [Cover every KV format and the graph path](#step-3-cover-every-kv-format-and-the-graph-path) | All three pool formats and the captured graph produce the same tokens, ≤ 0.06 ms per layer from 0 to 2048 tokens | 3–6 files |
| 4 | [Remap the decode matvec](#step-4-remap-the-decode-matvec) | ≥ 190 GB/s for every `N` ≥ 1024, step ≤ 15.5 ms, `pp128` ≥ 50 t/s | 4–8 files |
| 5 | [Activation-quantized integer matvec on the new mapping](#step-5-activation-quantized-integer-matvec-on-the-new-mapping) | ≥ 205 GB/s and ≥ 1.0 ms better step than step 4, tokens unchanged | 4–8 files |
| 6 | [Shrink the norm and narrow-projection tail](#step-6-shrink-the-norm-and-narrow-projection-tail) | −0.5 ms per step, no regression in the CPU comparison gates | 3–5 files |
| 7 | [Re-record the decode record](#step-7-re-record-the-decode-record) | `benchmarks.md` carries the final tables and this file's status is closed out | Docs |

Planning budget. The matvec family at the reference's 209 GB/s is 13.0 ms and
attention at the reference's per-layer cost is 0.1–0.3 ms, so the
post-step-5 step is about **15.5 ms (64 t/s)** at 512 tokens and about 16 ms at
2048 tokens, with the tail from step 6 removed. Steps 4 and 5 overlap: if step 5
is taken directly after step 2, step 4's FP32 remap still lands first, because
it is the numerical reference for the integer route and it isolates dispatch
regressions from arithmetic regressions.

## Step details

### Step 1: reproduce the comparison

Landed in `scripts/decode_compare.sh` and `scripts/decode_reference_shapes.py`.
The script checks the model hash and the `gfx1151` agent, runs the five
protocol cells on both engines, and joins the yuuka per-tensor attribution with
a reference trace by `(dtype, K, N)`. The reference K comes from the
grid of the `quantize_q8_1` launch that precedes each `mul_mat_vec_q`, so no
model-order assumption is needed; the join also asserts that both engines run
the same number of operations per step for each signature.

Add `scripts/decode_compare.sh`: it runs `yuuka bench` and `llama-bench` at
128/512/2048 prompt tokens with the flags of the [protocol](#environment-and-protocol),
prints the token rate and the ratio per cell, and fails when the model hash or
the GPU target does not match. Reuse the trace reduction approach of
`scripts/decode_profile.py` to print the per-shape bandwidth of both engines
from one trace each; keep the raw trace databases under `/tmp` and out of the
repository.

Deliverable: the tables of this document regenerate from two commands.
Acceptance: the script reproduces the baseline rows within the clock drift of
an unpinned session. No dispatch or kernel change.

### Step 2: parallelize decode attention over the key range

Landed as `attention_paged_flash_kernel` plus `attention_flash_combine_kernel`
in `kernels/hip/attention/attention.hip`. One block owns one KV head and one
key tile instead of one query head, so the four query heads that share a KV head
read its K/V rows once; the split over the key range is what removes the
position-dependent cost. The tile is 32 positions up to a 1024-position
capacity and 64 above it, selected by `flash_tile` and mirrored by
`decodeAttentionScratch` in `engine/gpu/model.go`, which allocates the partials
arena. The F16 and Q8_0 pools and the dynamic (graph) variant take the same
kernel, so step 3 became verification and measurement rather than a port. The
serial kernel remains the reference implementation and the fallback for
unsupported head widths and head/KV-head ratios.

The plan below is the original design note and is kept for the rationale.

Replace the serial position loop with a flash-decoding form.

* **Mapping.** Grid `(heads, ntiles)` for the decode shape; each block owns a
  contiguous slice of the key range for one query head. `ntiles` follows the
  context so the grid stays at two to four blocks per CU; a 40-CU device wants
  40–160 blocks and `heads` = 16 gives 1–10 tiles per head.
* **Inner loop.** Batch keys in tiles of at least 64 (the reference uses
  `nbatch_K` of 64–128) into registers or shared memory, compute Q·K in
  wave-wide vector form, keep a running maximum, running normalizer and running
  output vector in registers (online softmax), and accumulate P·V from the same
  tile. One barrier per tile, not per position.
* **Merge.** A `heads × ntiles` reduce pass combines the partials with the
  log-sum-exp rescaling. Keep the partials in the existing scratch arena; the
  sizes are `heads × ntiles × (head_dim + 2)` floats.
* **Numerics.** The online form must be compared against the current kernel,
  not only against the CPU reference: the current kernel is the byte-for-byte
  behaviour the model gates were recorded against. `TestGPUAttentionPathsAgree`
  and `TestGPUMatchesCPU` bound the logit movement; greedy tokens over eight
  steps must match.
* **Fallback.** The serial kernel stays compiled and reachable as the reference
  implementation and for shapes the tiled kernel does not cover (head widths
  other than 64/128/256, `heads` not a multiple of the tile geometry, Q8_0
  pools until step 3).

Files: `kernels/hip/attention/attention.hip` (new kernel and launcher),
`kernels/hip/yuuka_kernels.h`, `backend/hip/hip.go` (op encoding, dispatch
argument), `backend/hip/dispatch.go` (selection rule), `engine/gpu/plan.go`
(`addFullAttention`, `:296`), `tests/hip/paged_test.cpp` (layer-level check),
`engine/gpu/attention_compare_test.go` (model-level check).

Acceptance: at a 512-token context, ≤ 0.05 ms per layer in
`TestGPUProfileDecode`; `pp2048/tg32` ≥ 30 t/s; tokens identical.

### Step 3: cover every KV format and the graph path

Landed with step 2's kernel: `load_key_span` and `load_value_pair` decode the
F16 pool with 8-byte and 4-byte vector spans and the Q8_0 pool per 32-value
group, and the dynamic variant passes the same device-resident position to both
the split kernel and the merge. `TestGPUGraphReplayMatchesAndTimes` now runs
F32, F16, and Q8_0 and requires each format's replayed tokens to equal its plain
stream and all three streams to agree; `test_flash_decode_long` covers the
head_dim 64 and 256 shapes and an F16 pool with graph replay at the ABI level.

Measured per-layer decode attention, `TestGPUProfileDecode` with
`YUUKA_TEST_DECODE_CONTEXT` and `YUUKA_TEST_DECODE_KV`:

| Context | F32 | F16 | Q8_0 |
| --- | ---: | ---: | ---: |
| 512 | 0.0335 | 0.0353 | 0.0318 |
| 2048 | 0.0960 | 0.0711 | 0.0606 |

The 0.06 ms row holds for Q8_0 and misses for F16 and F32. This is the pool
byte traffic, not the kernel: at 2048 tokens one layer reads 16 MB (F32), 8 MB
(F16), and 4 MB (Q8_0) of K and V, so the F32 floor alone is 68 us at the
235 GB/s read ceiling even with a perfect kernel: 34 us for F16 and 17 us for
Q8_0. The profiled figures are 1.4x, 2.1x, and 3.5x those floors; the narrow
formats issue smaller requests, so their streaming efficiency is lower. The
512-token gate is the enforced one.

The F16 and Q8_0 pools and the dynamic-parameter variant need the same
treatment. The Q8_0 pool stages K and V dequantized per tile in shared memory;
the F16 pool loads `half2` directly. Add the dynamic-parameter variant last,
because the graph replay path passes the token and position from device memory
and must not read them on the host.

Acceptance: `TestGPUChunkedPrefillLongContext` and the KV-format token
equivalence test still pass for F32, F16 and Q8_0; `TestGPUGraphReplayMatchesAndTimes`
produces identical tokens; per-layer attention cost ≤ 0.06 ms up to a 2048-token
context in all three formats.

### Step 4: remap the decode matvec

Landed as `gemv_row_kernel` and `gemv_row_pair_kernel` in
`kernels/hip/gemv/gemv_row.hip`, selected inside the streaming dispatch for a
one-row block whose K is a multiple of 512; other row counts and K widths keep
`lm_head_kernel`, and the per-row GEMV stays the fallback. `decode_span16` in
`kernels/hip/quant.cuh` decodes sixteen K values per lane, the activation values
are read once per wave step and reused by both rows of the wave, and no barrier
sits inside K. Residual and gate/up/SwiGLU epilogues are both ported.

Measured: the matvec family moves from 20.9 ms at 130.3 GB/s to 15.44 ms at
176.6 GB/s in step, the decode step from 23.4 to 17.31 ms profiled (22.2 to
16.6 ms unprofiled), and `pp128/tg128` from 38.81 to 60.16 t/s. The 190 GB/s
per-shape row is met by `ffn_gate` and `lm_head` and missed by the narrow and
short-K projections; the step target of 15.5 ms is missed by 1.8 ms. The
rejected variants: 32 K values per lane (131 GB/s, register pressure), four rows
per wave with eight waves per block (156 GB/s, grid), four waves per block
(156 GB/s at four rows per wave), and shared-memory staging of the activation
step (163 GB/s, the barriers cost more than the L1 reads they remove).

The original design note follows.

Change the mapping, not the arithmetic, so the change is numerically
attributable.

* **Mapping.** One warp owns one output row and walks the whole `K` axis, as the
  reference does; optionally two rows per block to share the activation
  loads (`rows_per_block` = 2 for the low-bit types, 1 otherwise). Grid becomes
  `ceil(N / rows_per_block)`, which is 512–124,160 blocks instead of 32–7,760.
* **Load width.** A lane reads 16 contiguous bytes of packed weights per step
  (one 32-value sub-block for Q4_K/Q5_K/Q6_K, one 32-value block for Q8_0) and
  accumulates its own partial dot product; the activation values it needs are
  read from a shared-memory row staged once per block. This removes the 2-byte
  load and the per-step barrier at the same time.
* **Reduction.** One `warp_reduce_sum` per row at the end, and one shared-memory
  cross-warp add when `nwarps` > 1. No barrier inside `K`.
* **Dispatch.** Keep the current kernel as the per-row fallback and keep the
  shape threshold in `SelectDecodeGemvPath` (`backend/hip/dispatch.go:203`) and
  `SelectLMHeadPath` (`:107`); re-measure `lmHeadMinOutputs` (`:130`) afterwards,
  because the crossover moves when the wide kernel gets faster.
* **Fused variants.** `GEMV_RESIDUAL` (residual add in the epilogue) and
  `GEMV_PAIR` (gate/up over one activation row) must be ported in the same
  change, otherwise the plan loses the 96 operations the fusions removed
  ([decode fusion](benchmarks.md#decode-fusion)).

Files: `kernels/hip/gemv/` (new kernel file or a rewritten
`kernels/hip/gemm/lm_head.hip`), `kernels/hip/quant.cuh` (a 16-value span
decoder), `backend/hip/dispatch.go`, `backend/hip/hip.go`,
`tests/hip/gemv_test.cpp`, `build/yuuka_gemv_decode_sweep` (per-shape cells).

Numerics: the reduction order changes, so bit equality with the old kernel is
not expected. Gate on the quantized host reference
(`tests/hip/gemv_test.cpp` tolerance for each dtype), then on the model:
`TestGPUMatchesCPU` within its recorded logit difference, greedy tokens
identical over eight steps.

Acceptance: in-step bandwidth ≥ 190 GB/s for every shape with `N` ≥ 1024 in the
`TestGPUProfileDecode` per-tensor table; the sweep's hot cells within 10% of
`244 GB/s × 0.85`; decode step ≤ 15.5 ms; `pp128/tg128` ≥ 50 t/s.

### Step 5: activation-quantized integer matvec on the new mapping

Landed as `gemv_row_i8_kernel` in `kernels/hip/gemv/gemv_row.hip`, reached
through `Options.Int8Matvec` with `PrequantizedInt8` so the activation row is
quantized once per group by the existing `AddQuantizeQ8_1At` operations. It
keeps the one-row mapping and dots four weight digits against the Q8_1 words
with `v_dot4`, applying the sub-block scale and minimum once per four values.

Result: the gate fails. In step the route measures 175.3 GB/s against 176.6 for
the FP32 route and 15.55 ms of matvec against 15.44; end to end it is 17.23 ms
per step against 16.56 ms over 32 decode steps, with identical tokens and a
0.41 maximum logit difference over eight steps. The int8 form wins on
`ffn_down` Q4_K (173.2 against 158.4 GB/s) and is level elsewhere, but the plan
loses the gate/up pair fusion, and the pair as two launches measures 184.4
against 193.6 GB/s as one. The FP32 kernel of step 4 stays selected and the
int8 route stays opt-in for measurement.

The original design note follows.

With the mapping of step 4 in place, re-evaluate the int8 route that
[benchmarks.md](benchmarks.md#int8-decode-matvec) currently keeps opt-in.

* Pre-quantize the activation row to Q8_1 once per projection group
  (`AddQuantizeQ8_1At` already exists and `decodeActGroup` already shares one
  quantization across `q`/`k`/`v` and across the gate/up pair), then run one
  `v_dot4` per four weights with the per-sub-block scale and minimum applied
  once per sub-block.
* Keep the FP32 kernel from step 4 as the reference and as the route for the
  dtypes whose per-value form is not integer plus scale (F32, F16, Q4_0, Q5_0).
* Do not fuse the quantization into the matvec block: the fused form was
  measured slower than the separate pass
  ([int8 decode matvec](benchmarks.md#int8-decode-matvec)) and the separate
  pass is shared across projections anyway.

Acceptance: the sweep reaches ≥ 205 GB/s on the Q4_K `2560 → 9216` and Q6_K
`9216 → 2560` cells; the step is ≥ 1.0 ms better than step 4 in the same
session; logits stay within the recorded int8 tolerance and greedy tokens match
the step-4 route over eight steps. If the gate fails, keep step 4's kernel
selected and record the negative result.

### Step 6: shrink the norm and narrow-projection tail

Landed:

- `rmsnorm_f32_kernel` reduces with one wave shuffle tree per wave and one
  cross-wave add instead of the eight-barrier tree, and vectorizes the two passes
  over four values when the width is a multiple of four.
- `rmsnorm_heads_f32_kernel` runs one wave per head and needs no barrier.
- `beta` and `alpha` share one `GEMV_PAIR` launch over the same normalized row.

Result: `rmsnorm` 1.08 -> 0.59 ms and `rmsnorm_heads` 0.14 -> 0.10 ms per step,
379 -> 355 operations, step 17.31 -> 17.10 ms, `pp128/tg128` 60.16 -> 60.70 t/s.
The −0.5 ms row is missed by half. The remaining tail is launch bound: 65 norm
launches cost 9 us each against the reference's 5.7 us, and the only remaining
merge (q and k RoPE plus the two head norms) removes eight launches of each and
is worth under 0.1 ms. Fusing norms across layers is a plan-shape change that
this step does not take.

The original design note follows.

* `rmsnorm` (12 µs per 2560-value row, 81 launches per token): one warp per row
  with a warp shuffle reduction, `rows` rows per block, no shared-memory tree.
  Target ≤ 4 µs per launch.
* `ssm_alpha` and `ssm_beta` (32 outputs, 24 launches each, 9 and 11 µs):
  either merge the pair into one launch over the concatenated matrix, or fold
  them into `conv_l2norm`, which already reads the same `attn_qkv` row. Merging
  the launches is the smaller change; it needs a plan-level concatenation of two
  weights, which the prefill path already does for the gate/up pair.
* `rope` and `rmsnorm_heads` are 5–6 µs per launch over 16 launches each; merge
  q and k into one launch per layer while the plan is open.

Acceptance: −0.5 ms per decode step, all kernel tests unchanged, no new
dispatch threshold without a measured table.

### Step 7: re-record the decode record

Landed. [benchmarks.md](benchmarks.md) now carries the re-recorded decode
output-length table, the decode gap attribution with the per-operation and
per-shape tables, the llama.cpp comparison row (0.98), the decode tail section,
and the rejected variants of this plan in decisions. [kernels.md](kernels.md)
sections 4, 5.3, 5.4, 6.5, and 7.2 carry the remapped matvec, the split-key
attention contract, the KV-format reads, and the scores workspace. The plan
gained no operation code: the beta/alpha merge reuses `GEMV_PAIR`, so
[architecture.md](architecture.md) is unchanged. The status line above is
closed. Reproduction is in [benchmarks.md](benchmarks.md#reproduction) and
`scripts/decode_compare.sh`; the raw traces stay under `/tmp`.

## Rejected and deferred

These were considered and are not part of the plan.

* **Widening the channel-parallel mapping instead of remapping.** Eight output
  channels per wave was measured 28% slower (register spills), and a
  `kStepK` of 512 was 7% slower (staging outweighs the barriers)
  ([decode fusion](benchmarks.md#decode-fusion)). The mapping has to change
  shape, not width: the accumulators must leave registers before the row
  covers `K`.
* **Split-`K` with a second reduction pass as the only change.** It multiplies
  the grid but leaves the 2-byte loads, the per-step barrier and the FP32
  dequantization on the critical path; the reference reaches 209 GB/s without
  it.
* **Keeping the serial attention kernel and shrinking its per-position work.**
  Removing the shared-memory reduction would cut the slope, not the design
  defect: the position loop would still expose one dependent DRAM round trip
  per position with 16 blocks resident.
* **HIP Graph replay to remove submission gaps.** Measured inside noise
  (`TestGPUGraphReplayMatchesAndTimes`, ratio 0.999), and the step is already
  within 0.5 ms of its kernel sum.
* **Speculative decoding as a way to hit the target rate.** It is orthogonal and
  prompt dependent; it must not be used to mask the step time. It stays opt-in.
* **Fused mixed-shape multi-request decode batching.** The 16x aggregate gap in
  [serving](benchmarks.md#openai-server) needs a different plan; a single
  request is the current scope.

## Open questions

1. **What is the actual streaming read ceiling on this device?** The record
   claims 244 GB/s; the hardware probe measures 230 GB/s for `hipMemset` and
   106 GB/s for a device-to-device copy, and the reference reaches 235 GB/s on
   the 521 MB `lm_head`. Measure a pure read kernel at 2.7 GB before setting the
   step-5 gate, because a 235 GB/s ceiling makes the matvec target 16.1 ms and
   the 60 t/s row unreachable without weight-read fusion or batching.
2. **Does the 16-byte span decoder need the sub-block scale fetched per lane?**
   The current kernel stages scales through shared memory (`lds/w` 0.16 per
   weight). At 16 bytes per lane a whole 32-value sub-block is inside one lane,
   so the scale and minimum are a single 2-byte read per step; measure whether
   that read should be broadcast from LDS or fetched directly.
3. **What is the best `rows_per_block` after the remap?** The reference uses 2
   for the low-bit types and 1 otherwise. That is a tuning decision for this
   device, not a port: sweep it in `yuuka_gemv_decode_sweep` at `N` = 1024,
   2560 and 9216.
4. **Where does the tile-count crossover sit for the attention kernel?** The
   acceptance table needs `ntiles` as a function of context, and the rule has to
   be a measured table in `SelectDecodeGemvPath`'s style, not a constant.
5. **Does the reference's `flash_attn_tile` advantage survive at 8K context?**
   Not measured here; the longest decode context measured in this session is
   2080 tokens.

## Reproduction

```bash
# Both engines, same model, same device, same session.
MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf
for p in 128 512 2048; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens "$p" --decode-tokens 128 --warmup 2 --reps 3 --kv-type f32
done
for p in 128 512 2048; do
  ../llama.cpp/build/bin/llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 \
    -p "$p" -n 128 -r 3 -o md
done

# yuuka per-operation attribution at one context.
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfileDecode -count=1 -v

# yuuka per-shape kernel ceiling.
./build/yuuka_gemv_decode_sweep
./build/yuuka_gemv_decode_sweep --threshold

# Reference per-kernel attribution of the decode loop.
rocprofv3 --kernel-trace -d /tmp/ref-trace -- \
  ../llama.cpp/build/bin/llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 \
  -p 0 -n 32 -r 2
```

Only one benchmark process may use the GPU at a time; two processes sharing the
device invalidate both results.
