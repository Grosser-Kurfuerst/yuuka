# GPU Kernels

## 1. Scope

This document specifies the device-side execution strategy for gfx1151:
operator families, prefill/decode separation, quantized GEMV and GEMM,
attention, fusion, dispatch, and numerical requirements. It does not specify
the Go scheduler or API; those are in [serving.md](serving.md).

Implementation status: decode-oriented kernels for `F16`, `Q8_0`, `Q4_0`,
`Q5_0`, `Q4_K`, `Q5_K`, and `Q6_K` GEMV, RMSNorm, L2Norm, gated RMSNorm, RoPE,
decode attention, convolution state, gated delta-net recurrence, embedding,
SwiGLU, residual add, and copy are implemented and validated against CPU
references. Batched GEMV
reuses each packed weight element across up to 16 rows; its measured thresholds
are recorded in [benchmarks.md](benchmarks.md), and
`backend/hip.SelectPrefillPath` chooses the batched path for the small row
classes it covers.

Phase 3 adds the prefill family: rocWMMA FP16 and quantized GEMM for `F16`,
`F32`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, and `Q6_K`; a rocBLAS FP16
baseline with an explicit FP32-to-FP16 cast; causal GQA Flash Attention over
prompt chunks; chunked prefill with per-chunk KV writes; row-batched RMSNorm,
head RMSNorm, RoPE, embedding, and KV-write kernels; and a fused gated delta-net
chunk kernel that performs convolution, L2 normalization, state update, and
gated RMSNorm in two launches per layer. `backend/hip.SelectPrefillPath` is the
measured dispatch table described in section 9.

Phase 4 adds the paged KV family: decode and prefill attention kernels that
resolve every key/value position through an int32 block table, paged KV-write
kernels for one row and for a chunk, dynamic-parameter variants that read the
token and position from device memory, and a C ABI (version 4) for
device-to-device block copies and HIP Graph capture/replay. KV pools support
F32 (default), F16, and Q8_0; see section 7.3. Attention uses scalar per-lane
kernels for F32/Q8_0 and for F16 contexts below 2048 tokens; F16 contexts at or
above 2048 tokens use a matrixized rocBLAS path by default (section 7.5).
Continuous batching is scheduler-level interleaving of per-request plans;
mixed-shape prefill/decode kernel batches and split-KV decode are not
implemented.

The guiding rule is that **prefill and decode are separate operator families**.
They may share data layouts and helper primitives, but they are not the same
GEMM with a different `M`.

## 2. gfx1151 execution model

| Constraint | Kernel consequence |
| --- | --- |
| 40 CUs (reference configuration) | Tiles and workgroups must provide enough parallelism to fill the GPU; tiny grids waste the device. |
| Wave32-first | Prefer 32-lane workgroups and reductions tuned for Wave32. |
| RDNA WMMA | Use WMMA for prefill matrix tiles. Do not reuse CDNA MFMA kernels. |
| LDS capacity | Double-buffering and tile size trade against occupancy; avoid LDS spills. |
| VGPR budget | Aggressive fusion can reduce occupancy; measure VGPR usage per kernel. |
| UMA / LPDDR5x | Decode weights are read from shared system memory; effective bandwidth is the primary limit. |
| Kernel launch cost | Decode launches many small kernels; batching and Graph replay matter. |

Every kernel must be evaluated on the target device with recorded ROCm and
driver versions. Occupancy and bandwidth estimates from other RDNA parts are not
accepted as evidence.

## 3. Prefill versus decode

### 3.1 Prefill

Prefill processes the prompt with a relatively large `M`:

```text
Y[M, N] = X[M, K] * W[K, N]
```

The critical resources are matrix throughput, weight reuse across the `M`
dimension, attention I/O, and tile scheduling. The primary tools are WMMA,
quantized GEMM, Flash Attention, larger tiles, and software pipelining.

### 3.2 Decode

Decode generates one token at a time, or a small batch:

```text
y[N] = x[K] * W[K, N]       (per request, M = 1)
```

The critical resources are quantized weight-read bandwidth, KV-cache reads,
reduction efficiency, and launch overhead. The primary tools are fused
dequantize-GEMV, GQA-aware paged attention, and shape-dependent batching.

### 3.3 Roofline intuition

For a 4.5 GB weight set and a theoretical 256 GB/s memory bandwidth, a fully
streaming single-request decode has a rough upper bound near 57 tokens/s:

```text
256 GB/s / 4.5 GB ≈ 57 token/s
```

This is an order-of-magnitude estimate, not a benchmark. Real throughput is
lower and depends on effective bandwidth, KV reads, dequantization cost, kernel
efficiency, context length, and host scheduling. With larger batches, weight
reuse can make **aggregate throughput** exceed the single-request estimate.

### 3.4 Routing

| Condition | Preferred path |
| --- | --- |
| `M = 1` | Quantized GEMV, fused unpack/scale/reduce |
| Small batch (`M = 2..16`) | GEMV or small-M GEMM, chosen by measured threshold |
| Larger `M` | Quantized WMMA GEMM or rocBLAS/rocWMMA baseline |
| Prefill | WMMA GEMM + Flash Attention |
| Decode | GEMV + Paged Attention |

The thresholds are **measured per quantization type and shape class**, not
hard-coded from another GPU.

## 4. Operator family map

| Kernel family | Main scenario | Optimization focus |
| --- | --- | --- |
| `gemv_q4_k` | Single-token decode | Packed reads, fused dequantization, Wave reduction |
| `gemv_q6_k` | Decode | High-precision K-quant layout |
| `gemv_q8_0` | Decode / baseline | Simple, stable high-bandwidth quantization |
| `gemv_q4_0` / `gemv_q5_0` | Decode / broader GGUF coverage | Nibble and high-bit unpack, batch reuse |
| `gemm_fp16_wmma` | Prefill baseline | Tiling, WMMA, LDS, pipelining |
| `gemm_q4_k_wmma` | Quantized prefill | Tiled dequantization plus WMMA |
| `gemm_q6_k_wmma` | Quantized prefill | Scale handling, tile reuse |
| `flash_attention` | Prefill | Online softmax, causal mask, GQA |
| `attention_prefill_paged_blas_f16` | F16 prefill, 2048+ positions | Gathered K/V, rocBLAS QK/PV, FP32 softmax |
| `paged_attention` | Decode | Block table, shared KV heads, long-context split |
| `rmsnorm` / `rope` | All stages | Minimize memory traffic and launches |
| `fused_swiglu` | FFN | SiLU, gate, and multiply |
| `moe_route` / `moe_expert_*` | Sparse MoE | Top-k routing, per-expert assignment bins, packed expert reads |
| `sampling` | Output | GPU sampling or a low-cost host path |

## 5. Decode: quantized GEMV

### 5.1 Data flow

```text
GGUF packed Q4_K / Q6_K / Q8_0
            |
 Coalesced read + unpack + scale
            |
 Multiply-accumulate in registers
            |
 Wave / block reduction
            |
 Output FP16 / FP32
```

The kernel must not expand the full weight matrix into FP16 before computation.
Unpack, scale application, multiply-accumulate, and reduction are fused as far
as the register and LDS budget allows.

### 5.2 Q4_K specifics

Q4_K uses 256-weight superblocks with encoded scales and mins. Implementation
requirements:

- respect superblock and sub-block boundaries;
- load packed words with alignment-safe, coalesced accesses;
- apply the correct scale/min encoding per sub-block;
- map workgroups across output channels and the reduction dimension so that
  neighboring lanes read neighboring data;
- keep partial sums in registers and reduce within a wave where possible;
- avoid per-element branches on quantization metadata.

### 5.3 Workgroup mapping

The main design axes to benchmark:

- one wave per output channel versus multiple output channels per workgroup;
- vectorized loads versus narrower packed loads;
- register reuse versus LDS staging for scales and partial sums;
- batch sizes 1, 2, 4, 8, and 16 to find the GEMV/GEMM crossover;
- accumulation precision and deterministic reduction order.

### 5.4 Q4_0 and Q5_0

Q4_0 and Q5_0 use 32-value blocks. Q4_0 stores one FP16 scale and sixteen packed
nibble bytes; values are `d * (q - 8)`. Q5_0 adds a four-byte high-bit plane;
values are `d * (q - 16)`. Both are supported by the CPU reference, single and
batched HIP GEMV, embedding, and WMMA prefill.

### 5.5 IQ2 and IQ3 reference support

The CPU reference dequantizes `IQ2_XXS`, `IQ2_S`, and `IQ3_S` blocks with the
ggml codebooks and sign tables. `IQ2_XXS` and `IQ3_S` derive sub-block scales
from packed auxiliary words; `IQ2_S` uses one 4-bit scale pair per 32 values.
`tensor.DequantizeRow` and `tensor.Matrix.MatVec` use the same block decoders,
so the reference path never expands a whole model to FP16. These types are not
implemented in HIP kernels yet; a HIP plan rejects them during validation.
`Q4_1`, `Q5_1`, `Q2_K`, `Q3_K`, `Q8_K`, `IQ1`, `IQ2_XS`, `IQ3_XXS`, `IQ4_NL`,
and `IQ4_XS` remain capability failures.

### 5.6 Precision

For FP16/BF16 inputs, accumulate in FP32 unless a measured, documented
tradeoff justifies otherwise. Quantization scales and mins must be applied
before the final rounding step; avoid double rounding between unpack and
accumulate. Current compute kernels write FP32 outputs. An output-dtype plan
field must be added before FP16 outputs are exposed.

## 6. Prefill: WMMA and quantized GEMM

### 6.1 Implementations

Prefill uses a separate operator family from decode. The runtime selects one of:

- **Batched GEMV**: the Phase 2 kernel, limited to 16 rows. It is retained as
  the baseline for very small prompt chunks.
- **rocWMMA GEMM**: the custom kernel in `kernels/hip/gemm/wmma_gemm.hip`. It
  accepts FP32 activations and every supported weight dtype.
- **rocBLAS FP16**: the external baseline in `kernels/hip/gemm/gemm_blas.hip`.
  It requires an FP16 activation buffer; the planner inserts an explicit cast.
- **Int8 MMQ**: `kernels/hip/gemm/mmq_gemm.hip` keeps Q4_K/Q5_K/Q6_K weights
  quantized, quantizes the activations to Q8_1, and accumulates exact int32
  products in 16x16x16 int8 WMMA. See 6.3.

All four are validated against a float reference built from the exact packed
weights. The rocBLAS baseline is a comparison point and a dispatch choice for
the measured F16 shape classes, not a fallback for unsupported quantization.

### 6.2 Custom WMMA GEMM

Two rocWMMA tile variants share the packed-load helpers and are selected inside
`launch_gemm_wmma_f32out` from the measured row and shape class:

- `gemm_wmma_k32_kernel`: 32x64 output tile, eight waves, one 16x16 fragment
  per wave, 32-wide K step. This is the original Phase 3 kernel.
- `gemm_wmma_kernel<DType, 64, 2, 32>`: 64x64 output tile, eight waves, one A
  fragment and two B fragments per wave, 32-wide K step with double-buffered
  shared-memory staging. One decoded weight tile feeds two accumulator
  fragments and 64 activation rows.

Both variants:

1. Load the FP32 `X` tile and convert it to FP16 in shared memory.
2. Stream and decode the weight tile directly into the FP16 shared tile, one
   contiguous span per thread.
3. Accumulate in FP32 with 16x16x16 WMMA operations.
4. Stage the FP32 accumulator tile in shared memory and write only valid
   output elements, so non-aligned `M` and `N` edges are safe.

`load_b_tile` gives each of the 64 weight rows four threads, so one thread
decodes a contiguous span of 8 values at a 32-wide K step (16 at a 64-wide
step). A span stays inside one 32-value quantization segment, so the block
header, scale, and min are loaded once per span. Q4_K and Q5_K read their
packed values with `uint2`/`uint4` global loads: their row byte sizes are
multiples of 16 and the span offsets are 8/16-byte aligned by construction.
Decoded halves are written straight to the shared tile, so no span is staged in
registers or scratch. An earlier version that buffered the span in a local
array spilled 48 B/lane to scratch and lost 6-12% on the wide 512-row shapes,
so the direct shared-memory write is the accepted form. Q4_0, Q5_0, and Q8_0
decode the span with scalar loads.

Q6_K decodes four values per thread from one `ql` word and one `qh` word
(`q6k_span4`), which retires the per-value byte loads of the element mapping.
The group shares one scale byte and one block header, and its four values always
sit in one 32-value segment, so the 16-bit halves of the two words are all the
block reads it needs. A block is 210 bytes, so the words are assembled from two
16-bit loads: a 32-bit load would be undefined-aligned on the odd blocks. Eight
lanes cover one 64-column row and a pass stages 32 rows. This replaced the
element mapping for Q6_K, which measured 6.115 ms on `9216 -> 2560` at `M = 512`
and 4.882 ms on `2560 -> 9216`; the span loader measures 3.589 ms and 3.175 ms
on the same cells, and the earlier span walk's 54% regression on small batches
does not recur because the span now amortizes the header over four values
instead of eight. The decoded values are bit-identical to `q6k_value`, so the
chunk gate did not move.

F32 and F16 keep the element mapping, in which consecutive lanes read
consecutive row bytes. They have no block header worth hoisting above the
per-value cost, and the span walk measured 12-16% slower for F16 on tall-K
shapes because a thread's bytes are then strided across rows instead of
coalesced across lanes. This kernel never materializes a dequantized weight
matrix: every supported dtype shares the tile path and differs only in the
packed-load helper. The separate rocBLAS route does
materialize one FP16 weight tile per call, in a caller-owned scratch sized from
the largest quantized matrix, and it is used only for the wide K-quant class
described below.

The FP32 accumulator is a measured requirement, not a default. FP16 compute
(after dequantizing into an FP16 scratch) is 1.6x faster in the library but
moved the 512-token prefill logits by 0.305 against the 0.05 chunk gate, so the
wide K-quant class instead runs rocBLAS with FP32 compute and an FP16
destination: 22.0 TFLOP/s on `2560 -> 9216` at `M = 512` against 8.6 for this
kernel, with one rounding of each output instead of one per K step. Tall-`K`
shapes keep this kernel because rocBLAS splits `K` there and the FP16
destination forces that reduction through FP16 (1.1-1.5e-3 measured against
4.0-4.9e-4 on the wide shapes). A 128-row tile with four accumulator fragments
was also measured: it halves the weight-tile reloads but needs a two-half
epilogue for the 24 KiB staging alias, and it loses on tall-`K` shapes where the
extra CTA count matters more. All of these are recorded in
[benchmarks.md](benchmarks.md#prefill-matmul-route-experiments-2026-11-16).

The 64x64/K32 variant is selected for `rows >= 64` when `K <= N`; the 32x64
variant remains selected for tall-K projections and smaller batches. Tall-K was
re-measured with the span loader instead of assumed: at `M = 64` and `M = 128`
the 64-row tile costs 40-90% on F16 and Q6_K tall-K and about 10% on Q4_K,
because halving the output columns per CTA doubles the number of CTAs that hide
the long K loop. At `M = 512` tall-K the two tiles are within 10%, so the 32x64
tile stays for every `K > N` shape. A 64-wide K step was also measured for the
64x64 tile and is 15-30% slower than the 32-wide step with the span loader
(Q4_K 2560 -> 9216 at `M = 512`: 3.3 ms against 2.85 ms), so only `BlockK = 32`
is instantiated.

The 64x64/K32 kernel stages 16 KiB, so four CTAs fit the 64 KiB LDS of a CU
instead of two for the previous 32 KiB, 64-wide staging, and reports 84-86
VGPRs, 28-48 SGPRs, no spills, no scratch, and 16 waves/SIMD. The 32x64 variant
reports 42-43 VGPRs, 25-46 SGPRs, 14 KiB static LDS, no spills, and 16
waves/SIMD. Measured on 2026-11-16 with the same binary pair, `M = 512` shapes
improve 48-50% for Q4_K, 49-54% for Q5_K, 32-49% for Q8_0, 3-23% for Q6_K, and
14-27% for F16, and the 512-token prefill moves from 544 to 881 tok/s. See
[benchmarks.md](benchmarks.md#prefill-gemm-span-decode-2026-11-16). The 500 ms
chunk GEMM gate is still not met. The wide Q4_K shape now runs at 8.5 TFLOP/s,
which is 2.4x below what hipBLAS reaches on the same shape with fp16 compute
(33.6 TFLOP/s) and 2.9x below llama.cpp's 24.9 TFLOP/s prefill matmul rate; the
gap attribution is in
[benchmarks.md](benchmarks.md#prefill-gap-attribution-vs-llamacpp-2026-11-16).

A CTA-order experiment that mapped consecutive blocks to M tiles was measured
and rejected: it reduced the tall-K time but raised the wide-projection time,
and the position-0 chunk GEMM total regressed to 947 ms. The kernels keep `N`
fastest in `blockIdx.x`.

`gemm_wmma_swiglu_kernel` fuses the FFN gate and up projections. It stages one
activation tile and both weight tiles, accumulates with two pairs of fragments,
and applies `silu(gate) * up` to the accumulator elements before the single
output store. Both weights must share dtype and shape. The fused kernel uses a
32-wide K step (24 KiB shared memory for the 64-row tile) because a 64-wide step
needed 48 KiB and was 72% slower for the fused pair. The plan selects it only
when both projections resolve to `PrefillWMMA`; otherwise the separate gate,
up, and `SILU_MUL` operations remain. Q/K/V, beta/alpha, and RMSNorm fusions are
not implemented.

Larger tiles, an int8 tensor-core path, and the remaining projection fusions
remain candidates under [roadmap2.md](roadmap2.md); they are not implemented and
are not claimed in the benchmark results.

### 6.3 Int8 MMQ GEMM

`mmq_gemm.hip` follows llama.cpp's MMQ design at the algorithm level (MIT,
commit `b1b770d90`); staging, tile shape, scale layout, and epilogue are an
independent implementation validated against the CPU reference in
`tests/hip/prefill_test.cpp`.

Activation quantization (`quantize_q8_1_kernel`, public ABI
`yuuka_quantize_q8_1`, plan op `YUUKA_OP_QUANT_Q8_1`): one wave handles 128
contiguous values, four per lane. Each 32-value block keeps `d = amax/127` and
`d * sum(q)`, so a Q4_K/Q5_K weight minimum becomes a dot product with the
activation block sum. Ties round to even (`rintf`). The width must be a multiple
of 128, so a wave never crosses a row.

`gemm_mmq_kernel<DType>` computes a 64 output-channel by 128 token tile with
eight waves, a 256-wide K step, and 64512 bytes of dynamic shared memory:

| Slice | Layout | Bytes |
| --- | --- | ---: |
| A, decoded weights | `[64][272]` int8 | 17408 |
| B, Q8_1 activations | `[128][272]` int8 | 34816 |
| A scales | `[16][64]` float, scale-slot major | 4096 |
| B scales | `[8][128]` float2, block major | 8192 |

Each wave owns one 16-row weight fragment at `row_base = (warp % 4) * 16` and
four 16-column activation fragments at `col_base = (warp / 4) * 64`. The int32
accumulator mapping is fixed by the builtin and was measured against the CPU
product on this device: lane `l`, element `e` holds output row
`2*e + l/16` of column `l%16`, so a lane applies one activation scale (its
column) and eight weight scales (its rows) per block.

Both tiles use a 272-byte row stride, not 256. At 256 every row of a tile starts
on the same shared-memory bank and the 16x16 int8 fragment loads conflict 16
ways: the identical compute loop measures 12.5 TFLOP/s at 256 bytes against
48.1 TFLOP/s at 272 bytes. The padding costs 1 KiB per tile.

The Q4_K/Q5_K scale (one per 32 values) is applied after each pair of 16-wide
slices, which is exact because the int32 partial sum is exact and the scale is
constant across the block; the minimum term adds `-(d_m * m) * d_a * sum(q)`.
Q6_K has one scale per 16 values and no minimum, so it applies the scale per
slice. The weight scales load once per block and are reused by all four
accumulator tiles. `MMA -> accumulate -> scale` dependencies are what limit this
kernel: the MMA-only ceiling for the same geometry is 52 TFLOP/s (no fragment
loads) and 48 TFLOP/s with loads, while the full kernel measures 10.4-14.8
TFLOP/s. Software-pipelining the scale application behind the next block's MMAs
was measured and rejected: it pushed the kernel to 192 VGPRs with 11 spills and
lost 3-6%.

Tails are masked: rows outside the batch stage zeros (their scales and sums are
zero) and output rows and channels outside `n_out` are not written. `n_in` must
be a multiple of 256, which every K-quant row layout already requires. The
epilogue reuses the operand staging area as a `[128][65]` float tile, then all
256 threads write coalesced over the output channel.

Measured resources on gfx1151: 163-178 VGPRs, 48-52 SGPRs, no spills, no
scratch, 8 waves/SIMD, dynamic LDS 64512. The quantizer reports 20 VGPRs and 16
waves/SIMD.

### 6.4 Dispatch

Dispatch is based on `M`, `N`, `K`, quantization type, and the measured row
class. The table is in `backend/hip.SelectPrefillPath`; measurements are in
[benchmarks.md](benchmarks.md).

| Condition | Selected path |
| --- | --- |
| `M < 4` | Batched GEMV |
| Quantized, `M < 8`, and `K > N` | Batched GEMV |
| K-quant, `N >= 64`, `M >= 128` | Int8 MMQ (Q8_1 activations, int32 WMMA) |
| Quantized, other `M >= 4` | Custom rocWMMA |
| F16, `4 <= M <= 64`, `K > N` | rocBLAS + cast |
| F16, `M < 8`, `K <= N` | Batched GEMV |
| F16, `M >= 128`, `K <= N` | rocBLAS + cast |
| F16, other `M >= 8` | Custom rocWMMA |

The custom kernel is selected only where the sweep showed it faster than the
compared baseline, and a library route only where it is both faster and inside
the numeric gates. The K-quant MMQ row is bounded from below by the Q8_1 pass
and the 128-token tile (it loses to WMMA below 128 rows) and requires
`N >= 64` because narrower output tiles waste more than half the tile. It
replaces the dequantize-plus-rocBLAS route for K-quants entirely: that route was
2-17% faster on wide shapes at 512 rows but is numerically rejected on tall
shapes (see 6.2), and it needed an FP16 copy of the weight matrix per call. F16
baselines include the cast cost. A dispatch table entry
is valid only for the recorded gfx1151 environment; the kernel is not selected
from assumptions about other RDNA parts.

Weight reuse is a secondary effect, not the binding limit. A 32-row M tile
makes a 512-row chunk read every weight tile 16 times, and the 64-row tile
halves that, but halving it again with the measured 128-row tile only bought
6-26% on wide shapes and lost 0.5-15% elsewhere. The kernel is instruction-issue
bound: 114 instructions per 4096-MAC WMMA, of which 54 are VALU and 10 LDS.
The 2026-11-15 measurements put a 512-token prompt at 516 tok/s; the
2026-11-16 span-decoding build measures 881 tok/s on the same GGUF and device,
and the Q6_K span loader plus the parallel convolution measure 1209 tok/s on a
512-token prompt, against llama.cpp HIP at 2175 tok/s for the 512-token row of
the current reference build. The 2026-11-16 split is matmul 293 ms, the gated
delta-net chunk 50 ms, and prefill attention 19 ms at position 0 and 562 ms at
position 7680. Attention dominates at long positions and was reduced from
2030 ms to 643 ms at position 7680. The ordered optimization plan and its
per-workstream status are in [roadmap2.md](roadmap2.md).

## 7. Attention

### 7.1 Prefill: Flash Attention

`attention_prefill_f32_kernel` (and its paged counterpart) processes a prompt
chunk with one wave per query row. A workgroup owns 16 query rows for one head
with 512 threads.

- Q is read into registers, with each lane holding `ceil(head_dim/32)` values.
- K and V are staged together, 16 keys at a time, in dynamic shared memory as
  FP32. The shared layout stores each lane's `vec` values with a stride of
  `vec + 1`, so the lane-parallel dot-product and value reads are bank-conflict
  free. The workgroup reuses one staged V tile across all 16 query rows, which
  cuts global V traffic by 16x relative to the previous one-row tile. Query
  heads that share a KV head still stage their own tile; they are not grouped
  in one workgroup, and only L2 reuse applies across them.
- Scores are reduced within the wave; online softmax keeps per-row max, sum,
  and FP32 value accumulators, so the full attention matrix is never
  materialized.
- Causality uses the chunk position: query row `r` may attend to keys
  `0..position+r`. Waves in a workgroup share the K/V-block loop to keep
  `__syncthreads` convergent and mask keys beyond their own position.
- GQA maps query head `h` to KV head `h / (heads/kv_heads)`.
- The Qwen3.5 attention gate (the second half of each query vector) is applied
  as `sigmoid(gate)` before the output write. The contiguous op implements only
  this gated layout; `ATTENTION_PREFILL` rejects `u[3] != 0` with an unsupported
  capability error. The dense query-layout flag is retained by the paged
  variants described in section 7.4.

Dynamic shared memory is `2 * 16 * 32 * (vec + 1) * 4` bytes, or 36 KiB at
`head_dim = 256`. The kernels report 49-53 VGPRs, 48-63 SGPRs, no spills, and
16 waves/SIMD. F16 KV is still loaded and staged as FP32; with the staged-V
kernel it measured slightly slower than F32 KV at position 7680 (657.5 ms
versus 643.1 ms), so F32 remains the default and F16 stays a capacity option.
The measured position-7680 attention time is 643 ms with F32 KV, down from
2030 ms before this change.

Supported prefill head dimensions are 1-256; the plane-stage kernel reports an
unsupported capability error above 256. Decode attention remains the
per-head kernel described in section 7.2. Flash Attention is not used for
decode.

Chunked prefill splits a prompt into at most `chunk-size` rows (default 512,
maximum 4096). KV writes for a chunk happen before its attention, so later
chunks see the full causal prefix. The runtime builds and caches one plan per
chunk length; only the final chunk executes the output tail.

### 7.2 Decode: Paged Attention

`attention_paged_f32_kernel` runs one workgroup per query head:

- The workgroup stages the query head in shared memory and reduces the score
  dot product across 256 threads.
- Each logical position `p` maps to
  `block_table[p / block_size] * block_size + p % block_size`; K and V are read
  from the pool through that row. The block table and pool are device buffers.
- Scores for one head use a per-head stride so one scores workspace serves all
  heads. The causal length is `position + 1`.
- GQA maps query head `h` to KV head `h / (heads/kv_heads)`.
- The attention gate is applied as `sigmoid(gate)` before the output write.
- The dynamic variant reads `position` from a device parameter buffer so a
  captured HIP Graph can be replayed without changing kernel arguments.

`attention_prefill_paged_f32_kernel` keeps the same eight-query-row workgroup
tiling and online softmax as the contiguous prefill kernel, but resolves each
staged K row and each V read through the block table. `launch_kv_write_paged_f32`
and `launch_kv_write_paged_dynamic_f32` map written positions through the same
table; the batch case is selected by the row count, not a separate kernel name.

`attention_prefill_paged_gqa_f32_kernel` runs the four query heads that share one
KV head in one workgroup, so a staged K/V tile serves all four instead of being
reloaded by four workgroups. Each of the 16 waves owns one head and four of the
block's 16 query rows; the tile layout, staging walk, and online-softmax update
are the ones the one-head kernel uses, applied to the four rows independently
with their own causal stops. It is selected only when `heads / kv_heads == 4`
and `position + rows >= 2048`: below that the staging cost is small and the
larger workgroup count of the one-head kernel wins, so a 512-token prompt at
position 0 is unchanged. The kernel has two staged-tile variants:

- **Packed K.** When `head_dim == 32 * vec` with an even `vec >= 2` (so every
  lane covers whole FP16 pairs), K is staged as `__half2` and each score
  accumulates with `v_dot2` in FP32. V stays FP32 because converting a half2 V
  tile costs more instructions than the shared memory it saves. The packed tile
  is also what makes a 32-key tile fit: at `head_dim` 256 it needs 55 KiB of the
  64 KiB dynamic limit, while two FP32 tiles of 32 keys need 73.7 KiB.
- **FP32 tile.** Everything else stages K and V as FP32.

`gqa_tile()` picks 32 keys when the variant's tile fits the 64 KiB dynamic shared
limit and 16 keys otherwise, so the barrier count per staged key halves wherever
the budget allows. Two keys are scored before either softmax update, which
overlaps the two shuffle chains. `tests/hip/paged_test.cpp` covers both variants
(`head_dim` 4 FP32, `head_dim` 64 packed) against the same reference attention
sweep at 2e-4. Measured effect and the rejected row-phase restructure are in
[benchmarks.md](benchmarks.md#long-context-attention-packed-k-tile-2026-11-16).

### 7.3 KV cache formats

The paged KV family accepts an F32, F16, or Q8_0 pool selected by the plan. The
same block table and pool geometry are used for every format; only the row byte
size changes:

| Format | Row layout |
| --- | --- |
| F32 | `kv_dim` FP32 values |
| F16 | `kv_dim` FP16 values |
| Q8_0 | `ceil(kv_dim/32)` groups of one FP16 scale plus 32 int8 values |

`kv_write_paged` quantizes per 32-value group for Q8_0; attention kernels decode
K and V through the same row format. Prefix-cache keys include the format so
cached blocks are never reused across formats.

### 7.4 Attention query layout

The paged attention kernels assume the Qwen3.5 query layout with a sigmoid
output gate in the second head-sized half. The plan carries a dense flag in
`u[3]` that skips the gate and uses `heads * head_dim` query stride; the ABI
retains that variant, but no supported model selects it. The `qwen35` path uses
the gated layout exclusively.

### 7.5 Matrixized paged prefill attention

`attention_prefill_paged_blas_f16` is selected by default when the KV pool is
F16 and `position + rows >= 2048`; F32, Q8_0, shorter F16 contexts, and models
opened with `DisableBlasAttention` keep the tiled paged prefill kernels. Decode
attention is unaffected.

The kernel packs the query values to FP16, gathers the K/V prefix through the
block table into reusable contiguous FP16 buffers, runs QK^T and PV as rocBLAS
GEMMs with FP32 accumulation, applies the causal mask and softmax in a
256-thread workgroup per `(query row, head)` with an FP32 reduction, writes
probabilities as FP16, and applies the Qwen3.5 attention gate during unpack.
The FP16 score and probability matrices trade accuracy for the ability to use
rocBLAS; the recorded logit difference stays below the chunk gate.

Q, K, V, scores, probabilities, and results share one scratch allocation sized
`6*q + 4*kv + 6*score` elements, where `q = heads*rows*head_dim`,
`kv = kv_heads*capacity*head_dim`, and `score = heads*rows*capacity`. The model
keeps the tiled kernel when that allocation would exceed 4096 MiB, so large KV
pools remain openable without an opt-in switch.

The native test at position 2048 validates against the float reference within
2e-4. A full-model F16-KV differential test records maximum logit differences
of 0.0132079 after a 2048-token prompt and 0.00757194 after an 8192-token
prompt against the tiled path, with the same top-1 token and below the 0.05
chunk gate. Performance and reproduction steps are in
[benchmarks.md](benchmarks.md#2026-11-17-gfx1151-matrixized-attention-default-for-f16-kv).

Current limits:

- Q8_0 KV is supported but not faster than F32 at short context; its benefit is
  cache memory.
- The tiled kernels run one workgroup per head, or per group of four query
  heads above 2048 positions; split-KV is not implemented. The tiled path is
  6.5%-of-peak-issue bound on FP32 scalar per-lane arithmetic at long context,
  which is the remaining long-context gap for F32/Q8_0 and short F16 contexts.
- No mixed prefill/decode batch. Each step belongs to one request plan.
- Block-table entries are immutable for a request; all blocks are allocated at
  admission, so replay and kernels never observe a changing table.

| Aspect | Prefill | Decode |
| --- | --- | --- |
| Query length | Prompt chunk | 1 token (or small batch) |
| Main limit | Matrix throughput and attention I/O | KV reads, weight bandwidth, launch overhead |
| KV access | Contiguous prompt range | Block table indirection |
| Parallelism | Tiles over query/key dimensions | Heads and KV splits |
| Masking | Causal, chunk-aware | Causal with cached prefix |

The logical KV view is block-based and independent of physical placement.
Kernel-specific block layouts are backend-internal.

## 8. Fusion

### 8.1 MoE expert pipeline

`moe_route` selects top-k experts, normalizes the selected softmax weights, and
bins `(row, slot)` assignments into one bucket per expert using atomics. The
grouped kernels then index blocks by `(expert, output row)` and walk that
expert's assignment list:

- `moe_gate_up_grouped` reads one packed gate and up row per block and computes
  all assigned token rows, so the packed weight row is reused across rows.
- SwiGLU uses the existing fused `silu_mul` kernel.
- `moe_down_grouped` performs the expert down projection over the same buckets.
- `moe_reduce_add` applies routing weights and accumulates into the residual
  stream.

The routing and assignment buffers are sized `experts * rows * used`, which is
safe because one row contributes exactly `used` assignments. A malformed plan
that would exceed a bucket is rejected at plan construction. The grouped kernels
remain covered by native F32 and Q4_K kernel tests, but no supported model
runtime executes them: `qwen35moe` is CPU-only and the removed Qwen3 MoE adapter
is not reachable.

### 8.2 Speculative verification

Speculative decoding reuses the chunked prefill graph with a tail that computes
logits for every candidate row instead of only the last row. Verification reads
those rows, accepts the matching greedy prefix, and on a mismatch restores the
session snapshot and replays the accepted prefix plus the correction token.
This is exact for greedy sampling and does not change the F32 KV or decode
kernels.

Implemented fusions, chosen because they remove launches or intermediate
traffic without reducing occupancy:

- Dequantization + GEMV and dequantization + WMMA GEMM.
- RMSNorm row batching, head batching, and fused RoPE over a chunk.
- KV-write batching for a prompt chunk.
- Prefill attention online softmax, value reduction, and attention gate.
- Gated delta-net chunk: convolution state update, q/k L2 normalization, the
  128x128 recurrent update, and gated RMSNorm per head. The convolution is a
  plain `kernel-1`-tap FIR over the raw chunk, because the carry state holds
  only the previous raw inputs, so one thread per output element replaces a
  serial per-channel walk over the chunk; a second small kernel writes the
  trailing raw inputs for the next chunk. That is four launches per layer on the
  register path (convolution, carry state, recurrence, gated RMSNorm) against
  three before, and the convolution dropped from 26.4 ms to 8.7 ms per 512-row
  chunk, with the outputs unchanged because the tap summation order is the
  same. The register-resident path (`gdn_chunk_splitreg_f32_kernel`) covers
  state size 128, where four lanes cooperate on one state column and each lane
  holds 32 of the 128 state rows in registers for the whole chunk. A layer
  exposes `heads * (state / 32)` workgroups instead of `heads` workgroups; q/k
  normalization is recomputed per workgroup, and the per-row cost is two block
  barriers for the q/k and decay/beta handoff. The q and k partial reductions
  are wave shuffles combined through shared memory, and the per-column `S^T k`
  and `S^T q` dot products are summed across the four lanes with shuffles.
  Gated RMSNorm runs as a separate row kernel over each `(row, head)` pair,
  because splitting columns across workgroups would otherwise break the
  per-head normalization domain. State sizes other than 128 use the fused
  one-column-per-thread kernel, which reads and writes the state for every row.
  Both paths keep the row-major state layout shared with the decode kernel.
  The register path is the measured fast path; see the attribution in
  [benchmarks.md](benchmarks.md).

SwiGLU remains an elementwise pass over the chunk. Fusing it into the GEMM
epilogue, together with fusing the gate/up GEMMs themselves, is a candidate
under [roadmap2.md](roadmap2.md); no fused implementation is claimed. Every
fusion must be evaluated on all of:

- kernel launch count;
- VGPR usage and occupancy;
- LDS consumption;
- memory traffic;
- end-to-end latency, not just isolated kernel time.

Over-fusion can reduce occupancy or increase register pressure enough to lose
performance. A fused kernel that wins on one shape may lose on another; retain
the unfused path as a fallback when measurements justify it.

## 9. Dispatch and tuning

### 9.1 Shape signature

Dispatch decisions and plan validation use these shape and capability fields:

```text
operator, M, N, K, input dtype, weight dtype, batch, sequence class, backend
```

There is no generic kernel registry in this release. Go selectors map the
fields they need to a concrete path, and plan construction validates supported
dtypes, row layout, buffer ranges, and dimensions before submission. Lookup is
per plan, not per tensor per token.

### 9.2 Dispatch selection and validation

Dispatch is recorded in one explicit Go selector:
`backend/hip.SelectPrefillPath` chooses batched GEMV, custom rocWMMA, or the
rocBLAS FP16 baseline from dtype, rows, and the input/output widths.
Single-token decode submits one GEMV per matrix; there is no separate runtime
GEMV batch selector.

Each plan addition validates the dtype, quantization row layout, shape
constraints, and buffer ranges. Unsupported combinations fail during planning;
there is no size-keyed registry, scratch registry, or graph-compatibility
registry to consult at runtime.

### 9.3 Offline tuning

Dispatch decisions are offline. Thresholds measured on gfx1151 are encoded in
`backend/hip.SelectPrefillPath`, keyed by dtype, shape class, and row class, and
recorded with the measurement environment in [benchmarks.md](benchmarks.md).
There is no runtime autotuner. Adding a device requires re-running the sweeps
and updating the selector and tests.

### 9.4 Fallback

If a custom kernel is unavailable, unsupported for a shape, or slower than the
library baseline, the planner selects the documented fallback. Silent numerical
substitution is forbidden. A fallback must be visible in diagnostics and
benchmarks.

## 10. Numerical requirements

- Prefill converts FP32 activations and dequantized weights to FP16 for WMMA
  input and accumulates products in FP32. This is a documented prefill policy,
  not an implicit fallback. Tensor tests compare against a float reference and
  use `2e-2 * (1 + |expected|)` for GEMM outputs; the observed worst case for
  the checked-in shapes is under `1.2e-2`.
- The prefill attention reference comparison uses
  `2e-4 * (1 + |expected|)`; the current checked-in shape reports about
  `3.0e-8` because both sides use FP32.
- Chunked prefill must produce the same greedy token as unchunked CPU prefill.
  The GPU tests compare final logits within `2e-3` and require an identical
  top-1 token for chunk sizes 1, 3, 7, 16, 17, 33, and 64 on the tiny fixture.
- Define a tolerance per stage: tensor-level outputs, logits, and final token
  selection.
- Use stable softmax (max subtraction) and stable normalization.
- Handle non-aligned dimensions, zero-length edge cases, and extreme values.
- Check for NaN/Inf propagation and overflow in reduction accumulators.
- Specify whether a kernel is deterministic. Split reductions and atomics may
  produce run-to-run variation; tests must use a deterministic mode or an
  explicit tolerance.
- Compare against the CPU reference and at least one independent implementation
  for critical operators.
- Do not tune a kernel against a single prompt or a single shape.

## 11. Kernel acceptance criteria

A kernel is not complete until:

- [x] It matches the CPU reference on representative shapes and buffer-boundary
      cases; randomized and non-aligned shape sweeps cover the GEMV, GEMM,
      prefill, and MoE families.
- [x] Its numerical tolerance is documented and tested.
- [x] Its relevant shape classes are covered by a benchmark or profile.
- [x] Its VGPR, LDS, scratch, and occupancy figures are recorded for the Phase
      0-3 kernel families (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] New Phase 5 kernels have recorded VGPR, LDS, and occupancy figures for the
      GEMV, WMMA, and MoE variants (`scripts/kernel_resources.sh`, results in
      [benchmarks.md](benchmarks.md)).
- [x] The planner selects it; quantized GEMV and GEMM have measured fallbacks.
- [x] It is integrated into an end-to-end model run.
- [x] It does not regress a previously accepted shape class.
