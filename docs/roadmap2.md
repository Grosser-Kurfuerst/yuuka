# Prefill Optimization Roadmap

## 1. Purpose and scope

Phase 3 established a correct chunked prefill path and its performance curves.
The remaining gap to the reference implementations is large: the recorded
512-token prefill rate is about 313 tok/s for `yuuka` HIP versus about
2163 tok/s for llama.cpp HIP on the same GGUF and device. This document turns
the 2026-11-14 prefill attribution into an ordered optimization plan.

Scope:

- `qwen35` single-request prefill on gfx1151.
- The prefill GEMM family, the gated delta-net chunk kernel, and prefill
  attention.
- Dispatch, benchmarking, and correctness gates needed to accept a change.

Non-goals:

- Decode kernel changes. Decode has its own operator family and is not part of
  this plan.
- Converting all quantized weights to FP16 or another expanded format.
- Routing gfx1151 attention through rocWMMA.
- Porting CDNA MFMA tile sizes or W7900 performance assumptions to gfx1151.
- NPU work.

This document owns the optimization order and acceptance gates. Kernel design
constraints remain in [kernels.md](kernels.md); measurements and reproduction
commands remain in [benchmarks.md](benchmarks.md); the benchmark protocol
remains in [testing.md](testing.md). The tables in section 2 are the decision
summary; [benchmarks.md](benchmarks.md) is the authoritative measurement record.
If this plan changes a documented kernel contract, update the owning document
in the same change.

### 1.2 Implementation status (2026-11-15)

| Stage | Status | Evidence |
| --- | --- | --- |
| A1-A3 | Implemented | Recurrent chunk 551 → 67.6 ms |
| A4 | Not needed | A1-A3 met the 150 ms gate |
| B1-B2 | Partially implemented | 64x64/K32 tile for wide rows with the shared span loader; isolated `M=512` GEMM 48-54% faster |
| B3 | Rejected | M-first CTA order regressed the chunk total |
| B4 | Rejected (revisit) | MMA issue-rate test only; 2026-11-16 counters show the kernel is decode/operand bound, not MMA bound |
| B5 gate/up | Implemented | Chunk 994 → 938 ms; 512-token 515.9 → 549.8 tok/s |
| B5 other fusions | Not started | Lower priority; no measured case after B4 |
| C1-C2 | Implemented | Position-7680 attention 2030 → 643 ms |
| C3 | Measured, not default | F16 KV 657.5 ms vs F32 643.1 ms at position 7680 |
| C4 | Not used | Native tile path met the F32 gate; no rocWMMA attention |
| D | Implemented | Path names, per-op profile, no autotuner |

The full 512-token chunk is 581 ms wall at position 0, down from 1553 ms. The
512-token prefill rate is 880.8 tok/s; the 8192 and 32768 rates (412.6 and
209.3 tok/s) were not re-measured after the 2026-11-16 loader change, and the
2048 rate was last measured at 511.0 tok/s. The 450 ms step-4 chunk target and
the 250 tok/s 32768 target are not met; the remaining gap is the GEMM kernel
structure, not the stages implemented here.

## 2. Baseline attribution

### 2.1 Measurement conditions

The attribution run used:

| Field | Value |
| --- | --- |
| Date | 2026-11-14 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host kernel | Linux 7.0.14-28-pve |
| Backend | HIP, F32 KV unless stated, chunk size 512 |
| Sampling | Greedy, temperature 0 |
| Method | 512-row chunks at positions 0, 2048, and 7680 through `Plan.ExecuteProfiled`; a separate chunk-size sweep |
| Warmup | one plan execution before the measured executions |
| Repetitions | one diagnostic execution per position, KV format, and chunk size |

This is an event-instrumented attribution run, not a benchmark under the full
[testing.md](testing.md) protocol. It does not control clocks, power, or
thermal state, and HIP events add overhead around every operation. The
recorded wall time for the same chunk was 1505-1560 ms; the profiled total was
about 1534 ms. Use the shares to choose work, not as release numbers.

### 2.2 Operation attribution

Qwen3.5-4B Q4_K_M, 512-token chunk at position 0:

| Operation | Time | Share | Notes |
| --- | ---: | ---: | --- |
| WMMA GEMM | 909 ms | 59.3% | 248 GEMM submissions |
| Gated delta-net chunk | 551 ms | 36.0% | 24 launches, one per recurrent layer |
| Prefill attention | 48 ms | 3.1% | 8 full-attention layers |
| LM-head GEMV | 12 ms | 0.8% | one row, one submission |
| SwiGLU, add, norms, RoPE, KV write, embedding | about 14 ms | about 1% | elementwise and normalization work |

The same chunk with the Q8_0 model variant measured 723 ms of GEMM and 551 ms
of gated delta-net work. Q8_0 has larger weight rows but a simpler
dequantization path, which indicates that the Q4_K GEMM cost is dominated by
unpacking and scale handling rather than by bytes read.

### 2.3 Attention scaling with position

The same 512-row chunk executed at several absolute positions. Only attention
grows with position; GEMM and gated delta-net time stay essentially constant.

| Chunk start position | F32 KV attention | F16 KV attention |
| ---: | ---: | ---: |
| 0 | 48.7 ms | 45.3 ms |
| 2048 | 421.5 ms | 388.3 ms |
| 7680 | 2030.1 ms | 1475.1 ms |

At position 7680, attention is about 58% of the chunk with F32 KV and about
50% with F16 KV. The F16 pool reduces attention time by about 27% at that
context. At position 0 the difference is within the diagnostic noise.

### 2.4 Chunk-size scaling

One measured execution per chunk size, same model:

| Chunk rows | Time | Throughput |
| ---: | ---: | ---: |
| 128 | 392.5 ms | 326.2 tok/s |
| 512 | 1505.4 ms | 340.1 tok/s |
| 1024 | 3132.6 ms | 326.9 tok/s |
| 2048 | 6812.6 ms | 300.6 tok/s |

Chunk size is not the lever. The default chunk of 512 remains appropriate.

### 2.5 Derived limits

These values are derived from GGUF tensor sizes, model dimensions, and kernel
loop counts. They are not hardware-counter readings.

- GEMM: the transformer matmul weights are about 2.161 GB (FFN 1.371 GB,
  attention 0.609 GB, SSM 0.180 GB). A 32-row M tile reads every weight once
  per group of 32 tokens, so the amortized traffic is about 67.5 MB per token,
  or about 34.6 GB per 512-token chunk. 909 ms therefore corresponds to roughly
  38 GB/s, far below the measured read bandwidth of about 221 GB/s.
- GEMM arithmetic: one token performs about 7.14 GFLOP in the prefill matmuls;
  a 512-token chunk performs about 3.66 TFLOP. 909 ms corresponds to about
  4.0 TFLOP/s, or about 6.8% of the documented 59.4 TFLOP/s FP16 WMMA peak.
- Gated delta-net: each head visits its 128x128 FP32 state four times per token
  row (decay, `S^T k`, update, `S^T q`), about 384 KiB per head-row. For 32
  heads, 512 rows, and 24 layers this is about 155 GB per chunk, or roughly
  280 GB/s effective traffic. The arithmetic is about 51.5 GFLOP per chunk, or
  roughly 94 GFLOP/s.
- Attention: the kernel reads V from global memory once per query row and once
  per key. Eight query rows share one workgroup, and four query heads share one
  KV head, so the same V data can be fetched up to 32 times. V access is also
  on the softmax dependency chain.

The conclusion is that the prefill path is not limited by a single roofline:
GEMM is limited by per-tile unpacking and unpipelined loads, the gated
delta-net kernel is limited by repeated state traffic and low grid occupancy,
and long-context attention is limited by V traffic and exposed memory latency.

## 3. Workstream A: gated delta-net chunk kernel

### 3.1 Problem

`kernels/hip/prefill/prefill.hip::gdn_chunk_f32_kernel` launches one
workgroup per head. Each workgroup loops over every token row and streams the
full 128x128 state matrix from global memory four times per row. It also uses
`block_sum_dynamic`, which performs a block-wide tree reduction with a
`__syncthreads` at every step.

For the validated model this produces:

- 32 workgroups per layer on a 40-CU device, with 128 threads per workgroup.
- About 155 GB of state traffic per 512-token chunk.
- About 26 barriers per token row.

### 3.2 A1: keep the state column in registers

Each thread owns one fixed state column. Keep that column in a local array for
the whole chunk and write it back once:

```text
load column into registers
for each token row:
    update registers
store column back to global
```

The gated delta-net recurrence for a fixed output column is independent of the
other columns once the per-head `q`, `k`, decay, and beta vectors are known.
Keeping the column resident preserves the existing per-column update order;
changing a block reduction order is a separate step that must be covered by the
logit and repeatability tests.

Acceptance:

- Gated delta-net time for 512 rows is at most 150 ms, measured with the same
  protocol as the baseline.
- No VGPR spills in the compiled kernel; record VGPR, LDS, and occupancy with
  `scripts/kernel_resources.sh`.
- Bitwise repeatability across two runs of the same chunk.

Risk: a 128-float column per thread costs about 128 VGPRs before temporaries.
If the compiler spills, fall back to a per-block LDS-resident state tile, or
split the 128 state rows across cooperating threads and combine partial
`S^T k` and `S^T q` sums with shuffles.

### 3.3 A2: increase grid parallelism

The current layer grid has 32 workgroups. Split each head's 128 columns across
two or four workgroups so the layer exposes 64-128 workgroups. The per-column
recurrence is independent, so each block can recompute the shared `q`/`k`
normalization reductions deterministically.

If register pressure still limits occupancy, split the 128 state rows of a
column across four threads, keep 32 values per thread, and reduce the partial
dot products within a wave. Record the resulting occupancy and do not accept a
variant that reduces correctness or determinism.

### 3.4 A3: remove per-step block reductions

Replace `block_sum_dynamic` with warp shuffles plus one shared-memory exchange
per reduction, and reduce the per-row barrier count to the minimum required by
the `q`/`k`/decay/beta handoff. A single barrier per row is the target.

### 3.5 A4: chunked gated delta rule

If A1-A3 still leave the kernel above the acceptance gate, implement the
chunkwise gated delta rule. Within a chunk, express the recurrence relative to
the chunk-start state so the work becomes dense matrix products:

- `KK^T` and `QK^T` for the intra-chunk terms.
- `K @ S` and `Q @ S` projections.
- the lower-triangular `(I + L) delta = U` solve.
- the `K^T @ W` state update.

The reference implementation in llama.cpp reports 1.63x to 2.44x on the
isolated kernel over 512-4096 tokens on CDNA. That result does not transfer
directly to gfx1151; it is a design reference, not a performance claim. This
work is a larger rewrite and must keep the decode recurrence unchanged.

### 3.6 Status (2026-11-15)

A1-A3 are implemented and measured; A4 is not needed for the current gate.
`gdn_chunk_splitreg_f32_kernel` keeps 32 state rows per lane in registers, with
four lanes per column, and exposes `heads * (state / 32)` workgroups. The
per-row cost is two block barriers. Gated RMSNorm runs as a separate row kernel
because the column split would otherwise change the normalization domain.
State sizes other than 128 keep the fused fallback.

Measured under the diagnostic protocol: 551 ms to 67.6 ms for the 24 recurrent
layers of a 512-row chunk, no spills, 93 VGPRs, 16 waves/SIMD, and bitwise
repeatability in `tests/hip/prefill_test.cpp`. The A1 time target (at most
150 ms) is met. The provisional total-chunk target of 1050 ms was not met
(1077 ms) because the GEMM path is unchanged. See
[benchmarks.md](benchmarks.md#gated-delta-net-register-state-2026-11-15).

## 4. Workstream B: prefill GEMM

### 4.1 Problem

`kernels/hip/gemm/wmma_gemm.hip` uses a 32x64 output tile, eight waves, one
16x16 accumulator fragment per wave, a 32-wide K step, and no software
pipelining. Every workgroup decodes its own B tile from the packed weights.
With a 32-row M tile, every weight tile is decoded and read again for each
M tile.

The measured result is about 4.0 TFLOP/s, or about 6.8% of the FP16 WMMA peak,
while consuming only about 38 GB/s of the available read bandwidth. The kernel
is limited by unpacking, synchronization, and exposed load latency, not by
DRAM.

### 4.2 B1: larger M tile with multiple fragments per wave

Increase the output tile so each decoded B tile applies to more rows:

- First step: 64x64 output tile, two 16x16 accumulator fragments per wave.
- Second step: 128x64 or 128x128, four to eight fragments per wave.

This halves or quarters the per-token weight traffic and the number of
dequantization passes. The target is a measured 2x to 3x GEMM improvement on
the 512-row shape when combined with B2.

Acceptance:

- GEMM time for the 512-token chunk is at most 500 ms.
- The GEMM tolerance `2e-2 * (1 + |expected|)` still passes on randomized and
  non-aligned shapes.
- VGPR, LDS, and occupancy are recorded; a regression in occupancy must be
  justified by end-to-end time, not by isolated-kernel time alone.

### 4.3 B2: larger K step and double-buffered LDS

Use a 64- or 128-wide K step by decoding two to four 32-value quantization
segments per iteration. Each segment keeps its own scale/min pair, so the
existing rule that a segment never crosses a scale boundary remains true.

Double-buffer the A and B tiles: prefetch the next K slice while the current
slice feeds WMMA. This removes the load-synchronize-compute serialization.

### 4.4 B3: CTA order for L2 reuse

The current grid maps `x` to output columns and `y` to M tiles, so the GPU
walks all N tiles of one M tile before moving to the next M tile. Change the
order so consecutive CTAs share the same B tile across M tiles. The same B
tile is then likely to stay in L2 while the M tiles that need it execute.

This is a small, independently testable change. Accept it only if it improves
the measured end-to-end chunk time; it must not change numerical results.

### 4.5 B4: int8 tensor-core path

After B1-B3, evaluate true int8 WMMA:

- Unpack Q4_K/Q5_K/Q6_K payloads to int8 and use
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`; Q8_0 uses its int8 payload
  directly.
- Quantize activations per 32-value group to int8 with an FP32 scale.
- Accumulate the integer dot product exactly in int32, then apply `d_w * d_x`
  and the Q4_K/Q5_K min term in the FP32 epilogue.
- Keep an FP16-WMMA fallback for shapes or dtypes where int8 does not win.

The ROCm gfx1151 precision-ladder study measured true int8 at 1.13x to 2.51x
over int8-to-FP16 compute, with the largest gains on K-heavy shapes. Those are
reference ratios, not gfx1151 results for this codebase. This stage is
accepted only with a token-level equality test and a documented tolerance
change if the logits move.

### 4.6 B5: small fusions

These are lower priority and should be evaluated only after B1-B4:

- Fuse the two FFN gate/up GEMMs and apply SwiGLU in the epilogue.
- Fuse the Q/K/V projections for full-attention layers.
- Fuse the beta/alpha projections in recurrent layers.
- Fuse RMSNorm into the activation load or the previous epilogue if it does
  not reduce occupancy.

Every fusion must be judged on end-to-end chunk time and resource usage, not
on launch count alone.

### 4.7 Status (2026-11-15)

B1 and B2 are partially implemented. `gemm_wmma_kernel<DType, 64, 2>` adds a
64x64 tile, two accumulator fragments per wave, a 64-wide K step, and
double-buffered A/B staging. It is selected for `rows >= 64` with `K <= N`.
The original 32x64 kernel remains for tall-K projections and smaller batches
because the wider tile was slower there. B3 (M-first CTA order) was implemented,
measured, and rejected.

Measured result: wide 512-row Q4_K shapes improved 3-7%, the chunk GEMM total
moved from 944 ms to 895 ms, and the wall chunk total from 1077 ms to 1025 ms.
The 500 ms GEMM gate is not met. The isolated wide shapes sat at the same
4.6-4.7 TFLOP/s figure as the sweep's rocBLAS column, which was read as a
device ceiling; the 2026-11-16 attribution shows that figure comes from the fp32
compute path used by both `launch_gemm_blas_f16` and the earlier tile, and that
hipBLAS with fp16 compute reaches 20-33 TFLOP/s on the same shapes. A 64x128
tile, a 128x64 tile with 2x2 fragment waves, a padded LDS stride, and the
M-first CTA order were all measured slower and are not in the kernel. B3's
acceptance rule is therefore applied as written: it was not accepted. The
gate/up part of B5 is implemented in section 4.8; B4 is rejected in section
4.9, and the remaining B5 fusions are unstarted. See
[benchmarks.md](benchmarks.md#prefill-gemm-tiles-2026-11-15).

A 2026-11-16 follow-up replaced the loader: it decodes one contiguous span per
thread, hoists the block header per span, and reads Q4_K/Q5_K spans with
`uint2`/`uint4` loads. Both kernels share it, the 64x64 tile stages a 32-wide K
step (16 KiB, four CTAs per LDS CU), and the element-at-a-time decoder is gone.
Isolated `M = 512` shapes improved 48-54% for Q4_K/Q5_K, 32-49% for Q8_0, 3-23%
for Q6_K, and 14% on the wide F16 shapes, which lifts the wide Q4_K shape from
4.6 to 8.5 TFLOP/s. The 512-token prefill moves from 544 to 881 tok/s and TTFT
from 0.940 s to 0.581 s. The 64-row tile was measured again for tall-K and still
loses at `M = 64`/`M = 128` by 40-90% for F16 and Q6_K, so the `K <= N`
condition stays; a 64-wide K step for the 64x64 tile measured 15-30% slower than
32-wide and is not instantiated. The gate/up fused kernel is unchanged. The
500 ms GEMM gate is still not met. See
[benchmarks.md](benchmarks.md#prefill-gemm-span-decode-2026-11-16).

A same-machine attribution against llama.cpp (`pp512` 2216 tok/s) splits the
remaining 2.5x gap into matmul 476 vs 186 ms, the gated delta-net chunk 67 vs
13 ms, and attention at parity. The matmul kernel is decode and latency bound:
114 instructions per 4096-MAC WMMA, of which 54 are VALU and 10 LDS, at 2922
wave cycles per WMMA and 39% issue occupancy, against about 22 instructions per
WMMA in llama.cpp's int8 MMQ kernel (128-row x 128-token tile, Q8_1 shared
staging, 256-wide K step). `launch_gemm_blas_f16` also requests fp32 compute
with fp32 output, which selects Tensile kernels at 3.3-4.7 TFLOP/s; the same
GEMM with fp16 compute and output reaches 20-33 TFLOP/s. Both the library
datatype and the tile/int8 geometry are open items. See
[benchmarks.md](benchmarks.md#prefill-gap-attribution-vs-llamacpp-2026-11-16).

A 2026-11-16 experiment rejected the FP16 accumulator and the 128-row tile.
Dequantize-to-F16 plus fp16-compute rocBLAS runs the 512-token prefill in
0.330 s (1552 tok/s, 1.76x) and is 1.6-3.3x faster per GEMM from 256 rows
upward, but it moves the chunk-512 logits by 0.305 against the 0.05 gate, and
restricting it to the shallow-`K` or the deep-`K` matmul family alone still
drifts 0.319 and 0.296. FP16 accumulation is therefore rejected for every
prefill matmul; the FP32 accumulator stays. The 128-row tile gains 6-26% on wide
shallow-`K` shapes and loses 0.5-12% on tall-`K` and 3-15% at 128 rows, and its
end-to-end range (0.553-0.572 s) stays inside the 2.4% run-to-run band of the
baseline (0.559-0.589 s), so it is not selected.

The remaining matmul bottleneck is instructions per MAC, but the datatype item
is only half closed. rocBLAS accepts FP16 operands with FP32 compute and an FP16
destination, which keeps the FP32 accumulator and still measures 22.0 TFLOP/s on
`2560 -> 9216` at `M = 512`, against 8.6 TFLOP/s for the selected WMMA kernel
and 35.9 TFLOP/s for the rejected FP16 accumulator. That combination is the
cheapest next measurement: it removes the error this experiment rejected and
leaves only the FP16 rounding of the result. The int8 MMQ geometry remains the
larger rewrite, and its int8 activation quantization adds its own error term, so
it must be measured against the same gate rather than assumed exact. See
[benchmarks.md](benchmarks.md#prefill-matmul-route-experiments-2026-11-16).

### 4.8 B5 status: fused gate/up (2026-11-15)

The FFN gate/up pair is fused when both projections select the custom WMMA
path. One kernel stages the activation tile and both weight tiles, accumulates
both matrices, and applies SwiGLU in the epilogue. The separate gate GEMM, up
GEMM, and `SILU_MUL` pass are removed from that plan. The fused kernel uses a
32-wide K step: with a 64-wide step it measured 528 ms for the 32 fused
submissions, and with 32-wide staging it measures 308 ms against 385 ms for the
three removed operations.

The position-0 chunk total moves from 994 ms to 938 ms and the 512-token prefill
rate from 515.9 to 549.8 tok/s; 2048 tokens reaches 511.0 tok/s. The 32768 rate
is 209.3 tok/s, unchanged within noise because attention dominates at that
length. Greedy tokens and chunk-size equivalence are unchanged. Q/K/V
projection fusion, beta/alpha projection fusion, and RMSNorm epilogue fusion
remain unstarted.

### 4.9 B4 status: not accepted (2026-11-15)

The int8 tensor-core path is not implemented. The gfx1151 dense WMMA reference
lists both `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` and
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` at 8192 operations per 32-cycle
issue, so true int8 does not raise the instruction issue rate on this device.
The precision-ladder ratios cited in section 4.5 compare int8 against an
int8-to-FP16 conversion path; they do not apply to the packed-weight kernel,
which already decodes into an FP16 WMMA operand.

The measured kernel is not operand-bandwidth bound: the 512-row chunk GEMM
consumed about 38 GB/s of the 221 GB/s read bandwidth before the tile changes,
and the later tile/fusion work reduced that further. An int8 path would add
per-32 activation quantization and Q4_K/Q5_K min-term correction to the
epilogue without a measured issue-rate gain, and would require a token-level
tolerance change. It stays rejected until a register-blocked GEMM makes the
kernel compute-bound; the FP16 WMMA path remains the fallback either way.

Correction (2026-11-16): the rejection rested on the MMA issue rate and on DRAM
bandwidth, and both observations stand, but the kernel turned out to be neither
MMA bound nor DRAM bound. Instruction counters for the 64x64/K32 Q4_K kernel on
`2560 -> 9216, M=512` measure 114 instructions per 4096-MAC WMMA (54 VALU, 10
LDS) at 2922 wave cycles per WMMA and about 39% issue occupancy, so the limiter
is weight decode plus shared-memory staging and the latency around them. An int8
staging path halves the stored bytes and the per-value conversion work even
though it does not raise the MMA issue rate, so the operand-density argument
does apply. llama.cpp's int8 MMQ kernel reaches 24.9 TFLOP/s against yuuka's 8.5
TFLOP/s on the same device. See
[benchmarks.md](benchmarks.md#prefill-gap-attribution-vs-llamacpp-2026-11-16).

## 5. Workstream C: prefill attention

### 5.1 Problem

The prefill attention kernel stages only K in shared memory. V is read from
global memory inside the per-key softmax loop, so the load is on the dependency
chain. Eight query rows in a workgroup read the same V data independently, and
four query heads mapping to one KV head read the same K/V data independently.
At position 7680 the kernel accounts for about 58% of the chunk.

### 5.2 C1: stage V in shared memory

Load K and V for a key block into shared memory together, using 16 or 32 keys
per block and F16 storage where the KV format allows it. The inner loop then
reads V from shared memory instead of global memory.

### 5.3 C2: query tile and GQA reuse

Process a tile of 16-32 query rows per workgroup. Share the K/V tile across
all query rows in the tile, and, when register pressure permits, across the
query heads that map to the same KV head.

Targets:

- K/V shared-memory reuse across the query tile.
- V traffic reduced by at least 8x, and by up to 32x if query heads are
  grouped.
- Vectorized loads and value accumulation (`float4`, `half2`).

### 5.4 C3: KV format for long context

Use F16 KV as the measured long-context prefill option. The diagnostic run
showed about 27% less attention time at position 7680 with no change at
position 0. Keep F32 as the default until a long-context benchmark under the
full protocol confirms the gain and the logit tolerance.

Q8_0 KV remains a memory-capacity option. It needs a separate long-context
measurement because its dequantization cost currently offsets the bandwidth
saving at short context.

### 5.5 C4: native matrix path only if measured

Do not use rocWMMA fragments for attention on gfx1151. llama.cpp measured its
rocWMMA flash-attention kernel up to 8.6x slower than its native tile path on
this device, with an up-to-42% prefill regression at 8192 tokens. If tensor
cores are needed for QK or PV, use the native WMMA builtins with an explicit
register/LDS layout and compare against the scalar tile path.

### 5.6 Acceptance

- Position-7680 chunk attention is at most 800 ms with F32 KV, or at most
  600 ms with F16 KV, measured under the full benchmark protocol.
- The 32768-token prompt prefill curve improves without changing greedy
  tokens.
- The attention reference tolerance `2e-4 * (1 + |expected|)` still passes.
- Chunked-prefill equivalence across chunk sizes still passes.

### 5.7 Status (2026-11-15)

C1 and C2 are implemented. A workgroup now owns 16 query rows for one head
(512 threads), stages K and V together in 16-key blocks, and reuses the staged
V tile across all 16 rows. The per-lane shared stride is padded by one float;
without that padding the staged-V version still measured 1484 ms at position
7680, and with it the time is 643 ms. The shared staging reduces global V
traffic by 16x, not the 32x that grouping the four query heads of one KV head
would allow; those heads still rely on L2 reuse.

C3 is measured but not adopted as a default: F16 KV measures 657.5 ms against
643.1 ms for F32 at position 7680, so F32 stays the default and the F16 target
of at most 600 ms is not met. C4 remains unstarted; no rocWMMA attention path is
used.

Results: position-7680 attention is 643 ms with F32 KV (target: at most 800 ms,
met), and 657.5 ms with F16 KV (target: at most 600 ms, not met). End-to-end
prefill reaches 515.9 tok/s at 512 tokens, 480.2 tok/s at 2048, 382.9 tok/s at
8192, and 208.9 tok/s at 32768; the 32768 sequencing target of at least
250 tok/s is not met. `TestGPUChunkedPrefillLongContext` reports bitwise
identical logits at chunk sizes 128, 512, and 1024 with the same top-1 token,
and the attention reference test passes at `2.98e-08` maximum absolute
difference. See
[benchmarks.md](benchmarks.md#prefill-attention-tiling-2026-11-15).

## 6. Workstream D: dispatch and planning

These changes are small and must follow, not lead, the kernel work:

- Keep the default chunk size at 512 until a full sweep proves another value.
- Update `backend/hip.SelectPrefillPath` only with measured thresholds for the
  new kernels. A new tile or dtype path must be selected only where it wins
  under the same conditions.
- Add a prefill-path diagnostic that reports the selected path and the
  measured per-operation time, so future changes can be attributed without a
  temporary test.
- Do not add a runtime autotuner. Dispatch remains offline and documented.

### 6.1 Status (2026-11-15)

The chunk size remains 512; the 2026-11-14 chunk sweep did not justify a
different value. `backend/hip.SelectPrefillPath` is unchanged because the
measured batched-GEMV/WMMA/rocBLAS crossovers did not move. The WMMA path's
internal 64x64-versus-32x64 tile selection is documented in
[kernels.md](kernels.md) and follows the same measured row and shape classes.
`PrefillPath.String` and the logged cases in `TestSelectPrefillPath` report the
selected path, while `TestGPUProfilePrefill` reports per-operation time through
`Plan.ExecuteProfiled`. No runtime autotuner was added.

## 7. Sequencing and provisional targets

Dependencies are strict: B4 depends on B1-B3, and C4 depends on C1-C2.

| Order | Workstream | Estimated effect on the 512-token chunk | Provisional gate |
| ---: | --- | --- | --- |
| 1 | A1-A3 | gated delta-net 550 ms to 50-150 ms | total chunk at most 1050 ms |
| 2 | B1-B3 | GEMM 900 ms to 300-500 ms | total chunk at most 550 ms |
| 3 | C1-C3 | 8k-position attention 2030 ms to 400-800 ms | 32768-token prefill at least 250 tok/s |
| 4 | B4, B5 | GEMM 300-500 ms to 200-300 ms | total chunk at most 450 ms |
| 5 | D | dispatch and fusion cleanup | no shape-class regression |

The estimates are planning targets derived from the measured shares and from
published same-device kernel results. They are not performance claims. Each
gate is provisional and must be revised when the full-protocol measurement is
available. A stage is complete only when its gate is measured, its correctness
tests pass, and the owning documents are updated.

The 512-token prefill rate that corresponds to the final target is roughly
1000+ tok/s, compared with the recorded 313 tok/s baseline. Reaching it would
still leave a gap to llama.cpp HIP at about 2163 tok/s, so the plan does not
claim parity.

## 8. Verification requirements

Every accepted change must satisfy the following:

- Correctness:
  - `ctest --test-dir build --output-on-failure`
  - `go test ./...`
  - `go test -race ./...` for CPU-side changes
  - `go test -tags hip ./engine/gpu -run 'TestTinyGPUChunkedPrefillMatchesCPU|TestGPUChunkedPrefillLongContext'` with `YUUKA_TEST_MODEL` set
  - The native prefill and paged tests that cover the changed kernel family
- Determinism:
  - Run the same chunk twice and compare logits and recurrent or KV state.
  - For split reductions, document whether the result is deterministic.
- Resources:
  - `scripts/kernel_resources.sh`
  - Record VGPR, SGPR, LDS, scratch, and occupancy for every new kernel.
- Performance:
  - Follow [testing.md](testing.md).
  - Record model hash, quantization, ROCm/HIP version, device, kernel version,
    power profile, warmup, repetitions, and concurrency.
  - Use the same binary and weights for A/B runs, interleaved where possible.
  - Separate prefill from decode and single-request latency from aggregate
    throughput.
  - Do not record a performance result without the measured table and the
    environment metadata.
- Documentation:
  - Update [kernels.md](kernels.md) for a changed kernel contract or dispatch
    policy.
  - Update [benchmarks.md](benchmarks.md) for a new measured result.
  - Update this document when a gate is met, revised, or abandoned.

## 9. Risks and open questions

| Risk | Mitigation |
| --- | --- |
| Gated delta-net register array spills and lowers occupancy | Measure VGPR and LDS first; fall back to an LDS-resident tile or split state rows |
| Larger GEMM tile exceeds the VGPR budget | Start at 64x64, then 128x64; require resource and end-to-end evidence |
| int8 activation quantization moves logits | Keep FP16 path as a fallback; require token equality and an explicitly reviewed tolerance |
| Attention shared-memory budget blocks K+V staging | Use 16-key blocks and F16 storage; trade tile width against occupancy |
| F16 KV changes long-context logits beyond tolerance | Keep F32 the default; measure the long-context logit delta before changing the default |
| Event profiling is mistaken for a benchmark | Keep the diagnostic label and reproduce with the full protocol |
| L2/MALL behavior distorts small-shape microbenchmarks | Cycle weight pools larger than 64 MB, as noted in the gfx1151 tuning references |

Open questions:

- Which gated delta-net mapping wins on gfx1151: one thread per column with
  128 registers, a 4-row LDS tile, or a row-split register tile?
- Does a 128x128 GEMM tile fit the VGPR budget at useful occupancy on gfx1151,
  or is 128x64 the practical maximum?
- Does int8 WMMA plus Q8 activation quantization stay inside the token-level
  correctness gate for Q4_K_M?
- Can attention group the four query heads that share a KV head without
  spilling the online-softmax accumulators?
- Is the long-context gain from F16 KV stable across prompts and chunk sizes?
- Would an int4 WMMA path for Q4_K pay off after the scales and mins are
  handled in the epilogue, or is int8 the practical stopping point?

## 10. References

Repository documents:

- [kernels.md](kernels.md): GEMM, attention, gated delta-net, fusion, dispatch.
- [benchmarks.md](benchmarks.md): the recorded prefill curves, comparisons, and
  the diagnostic tables behind this plan.
- [testing.md](testing.md): benchmark metadata, tolerances, and regression
  gates.
- [roadmap.md](roadmap.md): the completed Phase 3 scope and its exit criteria.
- `backend/hip/dispatch.go`: the measured prefill path selector.
- `kernels/hip/gemm/wmma_gemm.hip`: the current WMMA GEMM.
- `kernels/hip/prefill/prefill.hip`: the chunk kernels, including the gated
  delta-net and prefill attention kernels.

External references:

- ROCm gfx1151 WMMA precision ladder:
  <https://github.com/rocm/rocm-libraries/blob/b9a2faf423a2372c5ca575f603ede2e8cae1a9ca/dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/examples/gfx1151/gemm/README.md>
- RDNA3 dense WMMA builtins, including int8 and int4:
  <https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/_sources/compiler-builtins/rdna/rdna3-dense-wmma-builtins.rst>
- gfx1151 roofline:
  <https://raw.githubusercontent.com/shisa-ai/hipEngine/refs/heads/main/docs/reference/ROOFLINE-gfx1151.md>
- gfx1151 tuning notes, including the TileLang 128x128x64 result:
  <https://raw.githubusercontent.com/shisa-ai/hipEngine/refs/heads/main/docs/reference/TUNING-gfx1151.md>
- llama.cpp gfx1151 flash-attention routing:
  <https://github.com/ggml-org/llama.cpp/pull/24562>
- llama.cpp chunked gated delta-net prefill:
  <https://github.com/ggml-org/llama.cpp/pull/24561>
- hipfire RDNA3 gated delta-net implementation:
  <https://raw.githubusercontent.com/warpfront/hipfire/45f0a9cfccb41853eefccbf77c79b4732c250c98/docs/DELTANET.md>
- llama.cpp Vulkan gfx1151 K-quant tile result:
  <https://github.com/ggml-org/llama.cpp/issues/27553>
