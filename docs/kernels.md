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
F32 (default), F16, and Q8_0; see section 7.3. Prefill attention runs on the
FP16 MMA kernel for F32 and F16 pools whose shape it covers, and on the scalar
per-lane tiled kernel otherwise (section 7.5).
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
| `lm_head_stream` | Decode matvec and both output projections | One wave per four output channels, 16-byte packed reads, LDS activation staging, wave reduction, no block barrier in K |
| `gemv_q4_k` / `gemv_q6_k` / `gemv_q8_0` / `gemv_q4_0` / `gemv_q5_0` | Decode fallback, narrow projections, kernel reference | Packed reads, fused dequantization, block reduction |
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

Two mappings exist and `backend/hip.SelectDecodeGemvPath` is the measured rule
between them. Both compute the same per-weight products through `decode_span4`;
they differ in how much of the machine one output row keeps busy.

**Per-row GEMV** (`gemv_*_kernel`): one 256-thread block per output row, one
packed element per thread per K step, a block-wide tree reduction, and for
Q4_K/Q5_K two barriers per 256-value step for the scale staging. It reads the
activation once per output row, so a Q4_K row reads 10 KiB of activation against
1.44 KiB of weights. Measured 25-76 GB/s on the decode shapes.

**Streaming** (`lm_head_kernel`, section 6.5): one wave owns four output
channels and walks the K axis; a lane reads 16 bytes of packed weights per step,
the activation row is staged in shared memory once per block, accumulators stay
in registers, and the only reduction is one wave shuffle per channel. No block
barrier sits inside the K loop. Measured 90-287 GB/s on the same shapes at 2.1-8.6
instructions per weight fewer, and 0.16 LDS instructions per weight against up to
2.66.

The rule, in output width at `n_in = 2560` and one row: the streaming form wins
from 512 outputs for Q4_K/Q5_K/Q6_K and from 1536 for Q8_0/F16 (5.3.1), so the
selector requires `n_out >= 1024`, a K step of 256 values, a dtype in the GEMV
family, and 1-16 rows. Everything else keeps the per-row kernel. The 32-output
recurrent projections are the only decode shapes that land on the fallback.

Numerics. The per-row kernel sums a strided set of K values per thread and
reduces over the block; the streaming kernel sums four neighbours per lane and
reduces across the wave. Both accumulate in FP32, so the difference is that
reassociation: `TestGPUStreamingDecodeMatchesGemv` measures 8.34e-06 maximum
absolute logit difference over eight greedy decode steps of the real model, with
identical tokens, and `tests/hip/gemv_test.cpp` checks every dtype against the
host dot product at 2e-3 relative and against the per-row kernel at 2.4e-4
absolute.

#### 5.3.1 Measured decode shapes

One submission per plan, matrix resident, five warmups and twenty-five timed
submissions; `kernel` is the traced kernel time from
`scripts/decode_profile.sh`, which excludes the launch the host column includes:

| Shape | dtype | per-row GB/s | streaming GB/s |
| --- | --- | ---: | ---: |
| `ffn_gate` 2560 -> 9216 | Q4_K | 40.5 | 286.6 |
| `ffn_down` 9216 -> 2560 | Q6_K | 81.6 | 256.1 |
| `ffn_down` 9216 -> 2560 | Q4_K | 43.6 | 192.6 |
| `attn_q` 2560 -> 8192 | Q4_K | 40.6 | 258.2 |
| `attn_qkv` 2560 -> 8192 | Q5_K | 43.1 | 248.6 |
| `attn_gate` 2560 -> 4096 | Q4_K | 39.9 | 254.8 |
| `attn_output` 4096 -> 2560 | Q4_K | 42.6 | 203.2 |
| `ssm_out` 4096 -> 2560 | Q5_K | 44.2 | 203.4 |
| `attn_k` 2560 -> 1024 | Q4_K | 30.3 | 90.4 |
| `attn_v` 2560 -> 1024 | Q6_K | 51.4 | 136.7 |
| `ssm_beta` 2560 -> 32 | Q8_0 | 26.9 | 5.0 |
| `lm_head` 2560 -> 248320 | Q6_K | 46.5 | 184.2 |

The three cells below 150 GB/s are the 1024-output attention projections: a
32-block grid cannot fill a 40-compute-unit device, and one such submission costs
16 microseconds, so the decode step spends 0.26 ms of its 33.4 ms on the sixteen
of them. Closing that needs split-K or fewer channels per block, which is not
taken yet (roadmap4 5.5).

Inside the decode step the same kernels run at 94-187 GB/s rather than the
isolated figures, because a token rotates through 2.7 GB of weights and every
matrix is cold in the 32 MB MALL (roadmap4 2.4).

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
  quantized, stages both operands in WMMA fragment order, quantizes the
  activations to Q8_1, and accumulates exact int32
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
[benchmarks.md](benchmarks.md#decisions-and-rejected-routes).

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
14-27% for F16, and the 512-token prefill moved from 544 to 881 tok/s at the
time of that change. See [benchmarks.md](benchmarks.md#isolated-kernels).

The int8 MMQ route (6.3) now carries K-quants from 128 rows up, so the current
numbers for the wide shapes come from it: 17.3-18.3 TFLOP/s on Q4_K
`2560 -> 9216` at `M = 512` against 8.4 for this WMMA tile, and a 512-token
prefill of 1612 tok/s end to end against llama.cpp's 2346 tok/s on the same
device. The gap record is in
[benchmarks.md](benchmarks.md#llamacpp-comparison).

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
eight waves, a 128-value K step, and 30720 bytes of dynamic shared memory:

| Slice | Layout | Bytes |
| --- | --- | ---: |
| A, decoded weights | fragment order, `[4][4][512]` | 8192 |
| B, Q8_1 activations | fragment order, `[8][4][512]` | 16384 |
| A scales | `[8][64]` float, scale-slot major | 2048 |
| B scales | `[4][128]` float2, block major | 4096 |

Each wave owns one 16-row weight fragment at `row_base = (warp % 4) * 16` and
four 16-column activation fragments at `col_base = (warp / 4) * 64`. The int32
accumulator mapping is fixed by the builtin and was measured against the CPU
product on this device: lane `l`, element `e` holds output row
`2*e + l/16` of column `l%16`, so a lane applies one activation scale (its
column) and eight weight scales (its rows) per block.

Both operands are staged in the fragment register order of the int8 WMMA, not in
row-major tiles. The Q8_1 buffer itself stays row-major, so the public quantizer
contract and the digit check in `tests/hip/prefill_test.cpp` are unchanged; the
swizzle happens in the LDS copy of `stage_activations`. A 16x16 int8 fragment is 256 bytes and `rocwmma` loads eight
bytes per lane, where lane `l` holds row `l%16` of the 8-value K half `l/16`.
Staging therefore groups one 32-value K pair into a 512-byte region whose lane
`l` slot holds 16 bytes: the first 16-value slice in bytes 0-7 and the second in
bytes 8-15. One `ds_read_b128` then feeds two MMAs, five 128-bit loads serve the
eight MMAs of a K pair, and the staging stores become 16-byte lane stores.

Staging is assigned per warp: each warp owns whole 512-byte regions and every
lane fills one 16-byte slot, so a warp store covers one contiguous region. The
measured `LDSBankConflict` rate is 0.0% against 35.9% for the previous
272-byte-stride row-major staging. Q4_K/Q5_K spans are decoded with a
word-wide nibble expansion (a shift and one `0x0F0F0F0F` mask for eight
digits, plus a bit-plane move for the Q5_K fifth bit); Q6_K block starts are only
2-byte aligned, so its spans are still assembled from 16-bit halves.

The K step is 128 values rather than 256. The staged operands then take 30 KiB,
which keeps two blocks resident per CU (measured `OccupancyPercent` 45.6%
against 23.9%), and the staging stalls of one block are covered by the other
block's MMAs. A 256-value step needs 60 KiB and one block, and a 64-value step
adds enough barriers to lose the tall-K shapes; both were measured and rejected
(see [benchmarks.md](benchmarks.md#decisions-and-rejected-routes)).

The Q4_K/Q5_K scale (one per 32 values) is applied after each pair of 16-wide
slices, which is exact because the int32 partial sum is exact and the scale is
constant across the block; the minimum term adds `-(d_m * m) * d_a * sum(q)`.
Q6_K has one scale per 16 values and no minimum, so it applies the scale per
slice. The weight scales load once per block and are reused by all four
accumulator tiles.

The epilogue writes the accumulators straight to global memory. For a fixed
fragment the warp's 16 low lanes cover 16 consecutive tokens of one output
channel, so each store instruction covers two 64-byte runs. Transposing through
a shared-memory tile instead needs 33 KiB, which would break the two-block
residency, and the direct stores measure the same within noise.

The loop is limited by VALU issue, not by LDS: 22.2 VALU and 4.8 LDS
instructions per WMMA on Q4_K `2560 -> 9216` at `M=512`. Two costs dominate.
First, rocWMMA duplicates gfx11 wave32 operands in registers, so every
`mma_sync` costs 2.5 `ds_swizzle_b32` on the LDS pipe; feeding the builtin
directly needs the duplicated 16-byte operands staged in LDS, which doubles the
operand budget to 51 KiB and drops to one block per CU, and reconstructing the
duplicate in registers costs the same two moves per fragment on the VALU pipe.
Second, the scale application is four VALU per accumulator element per block
(`int32 -> float` conversion, the scale product, the main FMA, and the minimum
FMA), which is 16 VALU per WMMA by construction: each block updates eight
elements per fragment while issuing two MMAs per fragment. `v_fma_mix_f32` does
not convert an integer operand on gfx1151, so the conversion cannot be folded.

Tails are masked: rows outside the batch stage zeros (their scales and sums are
zero) and output rows and channels outside `n_out` are not written. `n_in` must
be a multiple of 128; every K-quant row layout is a multiple of 256.

Measured resources on gfx1151: 133 VGPRs, 38-40 SGPRs, no spills, no scratch,
10 waves/SIMD resource limit with two blocks resident per CU, dynamic LDS
30720. The quantizer reports 20 VGPRs and 16 waves/SIMD.

Measured throughput at `M = 512` (sweep, five timed submissions after two
warmups, including the Q8_1 pass): Q4_K 17.3-18.3, Q5_K 17.5-18.1, and Q6_K
12.4-14.4 TFLOP/s across the three shapes, against 13.5-14.1, 12.6-13.4, and
11.0-11.6 TFLOP/s before the layout and occupancy change. The route is still
1.2-1.3x below llama.cpp's MMQ kernel on the same device; the remaining
attribution is in [roadmap3.md](roadmap3.md#4-stage-s1-mmq-gemm-highest-value-73-of-the-gap).

The q/k/v and qkv/gate groups read the same normalized tile, so the plan
quantizes it once per group instead of once per projection
(`prefillProjections` in `engine/gpu/plan.go`); a 512-row chunk runs 128
`quant_q8_1` operations instead of 168.

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

The prefill tail (the output projection over the last chunk row, or over the
candidate rows of a speculative block) is not on this table: it is a `N >> M`
shape with `M <= 16`, where every route above wastes the tile on rows that do
not exist. It runs `lm_head_stream` instead (6.5) whenever the dtype is
supported, `n_in` is a multiple of 256, `M` is in `1..16`, and
`N >= 1024`; anything else keeps one decode GEMV per row. The decode path itself
is unaffected: it always uses the GEMV family.

Weight reuse is a secondary effect, not the binding limit. A 32-row M tile
makes a 512-row chunk read every weight tile 16 times, and the 64-row tile
halves that, but halving it again with the measured 128-row tile only bought
6-26% on wide shapes and lost 0.5-15% elsewhere. The kernel is instruction-issue
bound: 114 instructions per 4096-MAC WMMA, of which 54 are VALU and 10 LDS.
The 2026-11-15 measurements put a 512-token prompt at 516 tok/s, the
2026-11-16 span-decoding build at 881 tok/s, and the int8 MMQ route now
measures 1324 tok/s on the same GGUF and device, against llama.cpp HIP at
2346 tok/s for the same 512-token prompt in the current reference build. The
2026-11-16 split was matmul 293 ms, the gated delta-net chunk 50 ms, and
prefill attention 19 ms at position 0 and 562 ms at position 7680; attention at
position 7680 dropped from 2030 ms to 643 ms. The ordered optimization plan and
per-workstream status are in [roadmap2.md](roadmap2.md).

### 6.5 Streaming LM head

`lm_head_kernel` in `kernels/hip/gemm/lm_head.hip` computes the output
projection of both graphs (the prefill tail and the decode tail) and, since
roadmap4 D2, every decode matvec that `SelectDecodeGemvPath` covers:

```text
out[m][n] = sum_k W[n][k] * x[m][k],   m in [0, rows), rows <= 16
```

`x` and `out` are row-major FP32 blocks, `W` is any packed dtype in the GEMV
family, and `n_in` must be a multiple of 256 so that every K step is one
quantization block. The kernel is not a tile GEMM: with `N` in the hundreds of
thousands and `M = 1` every tile form wastes 15 of 16 accumulator rows, and the
operator is pure weight bandwidth.

Mapping:

- One wave owns four output channels and walks the whole K axis itself. A lane
owns four consecutive K values of each of its channels, so 32 lanes cover a
128-value half without overlap and the packed reads of the wave are contiguous
32-byte sectors of the four arrays a Q6_K half touches (`ql`, `qh`, `scales`,
`d`).
- The activation row block for the current 256-value K step is staged in
dynamic shared memory once per workgroup, so the activations are read from L2
once per 32 output channels instead of once per channel. The bytes are
`rows * 256 * 4`, that is 1 KiB for the single-row tail and 16 KiB at the
16-row cap.
- `decode_span4` in `quant.cuh` decodes a lane's four packed values straight
into registers, with the block base, nibble selection, and K-quant scale and
minimum hoisted out of the four values. Each decoder restates the arithmetic of
the per-value decoder of the same dtype, so both produce the same digits; the
per-value form is 3.2x slower here because it recomputes the block arithmetic
for every value.
- `rows` is a compile-time bound, not a runtime loop bound: the single-row
instantiation uses 45-67 VGPRs and the 16-row one 97-127, where a runtime bound
indexes the accumulator array dynamically and spills 272 bytes per lane to
scratch (measured 19.9 GB/s against 88 GB/s).
- The reduction is one `shfl_xor` tree per channel per row, so a second
identical submission is bitwise identical.

Measured with the `LMHEAD` cells of `build/yuuka_prefill_sweep` (median of five
timed submissions after two warmups) on the recorded gfx1151 device, `n_in`
2560:

| dtype | N | M | per-row GEMV | streaming LM head |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 248320 | 1 | 15.01 ms (24 GB/s) | 2.36 ms (152 GB/s) |
| Q4_K | 248320 | 16 | 240.82 ms | 6.59 ms |
| Q6_K | 248320 | 1 | 11.94 ms (44 GB/s) | 2.90 ms (180 GB/s) |
| Q6_K | 248320 | 16 | 191.00 ms | 7.67 ms |
| Q8_0 | 248320 | 1 | 7.22 ms (93 GB/s) | 3.11 ms (217 GB/s) |
| Q8_0 | 248320 | 16 | 115.51 ms | 7.14 ms |

In the 512-row prefill plan of `TestGPUProfilePrefill` the tail costs 2.84-2.92 ms
of a 287 ms chunk, against 11.3 ms for the single-row GEMV tail before this
kernel ([roadmap3.md](roadmap3.md#7-stage-s4-prefill-lm-head)). The kernel is
bandwidth bound: the same 521 MB Q6_K matrix streams at 244 GB/s in a pure read
loop on this device, so 2.90 ms is about 85% of that ceiling.

The decode tail runs the same single-row instantiation. In the profiled decode
step of `TestGPUProfileDecode` it costs 2.79 ms at 187 GB/s against 11.23 ms at
46 GB/s for the per-row GEMV, which is 8.4 ms of a 97.4 ms step
([roadmap4.md](roadmap4.md#45-implementation-2026-11-18)).

Numerics. The per-element products are the same FP32 products the decode GEMV
forms, including the K-quant scale-then-minimum order; only the summation order
differs, because a lane accumulates four neighbours and the last five additions
are a shuffle tree. `tests/hip/prefill_test.cpp` checks all eight dtypes against
the independent host dot products for one, three, and sixteen rows and requires
a repeated run to be bitwise identical; `TestPlanLMHeadMatchesDequant` checks
Q6_K and Q8_0 against dequantized CPU dots (Q8_0 is bitwise equal), and the
model-level gates are unchanged (`TestGPUMatchesCPU` 0.00309944,
`TestGPUChunkedPrefillLongContext` top-1 `33075` with zero difference).

A shape the kernel does not cover keeps the per-row GEMV tail; nothing silently
changes numerically. `backend/hip.SelectLMHeadPath` is the one rule both tails
use: dtype in the GEMV family, `n_in` a multiple of 256, `n_out` at or above
`lmHeadMinOutputs`, and 1-16 rows.

The crossover was re-measured at the decode width (`n_in = 2560`, one row) with
`build/yuuka_gemv_decode_sweep --threshold`, five warmups and twenty-five timed
submissions per cell. The streaming kernel wins from 512 outputs upward for
Q4_K, Q5_K, and Q6_K, and from 1536 upward for Q8_0 and F16, where the per-row
GEMV's vectorized path is still ahead at 1024 (0.022 ms against 0.035 ms). The
threshold stays at 1024 because one number covers every dtype without
regressing Q8_0, which is level there, and because no model shape sits in the
512-1536 range: an output projection is vocabulary sized. The table is in
[benchmarks.md](benchmarks.md#decode-matvec).

Numerics of the decode route. The per-row GEMV sums a strided set of K values
per thread and reduces over the block; the streaming kernel sums four
neighbours per lane and reduces across the wave. Both accumulate in FP32, and
`TestGPUStreamingTailMatchesGemv` measures the difference on the real model:
4.8e-06 maximum absolute logit difference over eight greedy decode steps, with
identical tokens.

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

### 7.5 FP16 MMA paged prefill attention

`attention_prefill_paged_wmma_kernel` in
`kernels/hip/prefill/attention_wmma.hip` is the default prefill attention
kernel. `launch_attention_prefill_paged_f32` selects it when all of the
following hold, and falls back to the tiled per-head kernel otherwise:

- the pool is F32 or F16, so an FP16 tile staging is either lossless or a
  documented rounding step (Q8_0 stays on the tiled kernel);
- `heads == 4 * kv_heads`, so one workgroup can serve the four query heads that
  share a KV head;
- `head_dim` is 64, 128, or 256, the widths the kernel is instantiated for;
- `rows >= 64` or `position + rows >= 256`: below that the chunk offers too few
  16-row query tiles to fill 40 CUs, and the tiled kernel measures faster.

Geometry. A workgroup is four waves, one per query head of the group, and owns a
16-row query tile. Each wave keeps the whole `16 x head_dim` output block in
accumulator registers (128 registers per lane at head width 256), so the value
product never leaves the wave. K and V tiles of 16 keys are staged in shared
memory once per workgroup and reused by all four heads.

Score orientation. The score product runs as `S^T = K Q^T` rather than
`S = Q K^T`, so both operands are row major in shared memory: K keeps the
key-row layout it is gathered in, and Q is stored transposed once per workgroup
instead of once per key tile. A score lane then owns one query row and eight
keys, so a row statistic is eight register operations and one cross-lane step
instead of a four-step butterfly, and the running maximum and sum are scalars
rather than per-row arrays. The transposed P the value product needs is written
back to shared memory from the same lane mapping.

Online softmax. The mask is applied inside the tile: a lane's element is `-inf`
when its key is beyond the query row or beyond the tile's causal bound. The
softmax runs in base-2 units, so an exponent is one `v_exp_f32` after a scaled
multiply, and the accumulator is rescaled by the row factor before each product.
The running statistics stay FP32, the accumulator is FP32, and only the staged
Q/K/V tiles and the probability tile are FP16.

Shared memory. At head width 256 the workgroup uses `(4*16 + 2*16)*256` halves
for the transposed Q and the K/V tiles plus four 16x16 probability tiles, 50 KiB
in total, which keeps two workgroups resident per CU. The kernel declares
`__launch_bounds__(128, 2)` so the compiler can use the full register file for
the accumulator block: at 192 registers it spilled 21 slots into the key loop,
and removing that spill was most of the 1.5x the stage measured.

Crossover and measurements. Isolated prefill attention, one layer, FP32 pool,
median of 20 launches after a warmup (full table in
[benchmarks.md](benchmarks.md#prefill-attention)):

| Shape | Tiled | MMA |
| --- | ---: | ---: |
| 16 rows at position 0 | 0.016 ms | 0.039 ms |
| 48 rows at position 0 | 0.044 ms | 0.052 ms |
| 64 rows at position 0 | 0.075 ms | 0.057 ms |
| 512 rows at position 0 | 2.468 ms | 0.507 ms |
| 32 rows at position 7680 | 14.651 ms | 3.266 ms |
| 512 rows at position 7680 | 70.243 ms | 14.815 ms |

A 512-token chunk therefore spends 4.1 ms in attention at position 0 and
117.6 ms at position 7680 over the eight full-attention layers, against 19.7 ms
and 562 ms for the tiled kernel.

Numerics. Staging rounds Q, K, and V to FP16, so an F32 pool gains the rounding
step an F16 pool already has. The native test measures 9.3e-5 maximum absolute
difference against the FP32 reference and against the tiled kernel on the same
plan input at head width 256, and the output is bitwise reproducible on a second
run. The full-model differential test records 0.015 maximum logit difference at
a 64-token prompt with the top-1 token unchanged; that is the same magnitude as
changing the KV storage format from F32 to F16 on the tiled kernel (0.011 on the
same prompt), which is the accuracy regime the current route runs in.

The earlier group-of-four tiled kernel (`attention_prefill_paged_gqa_f32_kernel`)
is removed: the MMA kernel serves the same group at every shape the tiled kernel
won on, so keeping it would have left a slower path behind a silent condition.

Current limits:

- Q8_0 KV is supported but not faster than F32 at short context; its benefit is
  cache memory.
- The tiled fallback runs one workgroup per head; split-KV is not implemented.
  It is 6.5%-of-peak-issue bound on FP32 scalar per-lane arithmetic, which is why
  Q8_0 pools and short chunks keep the long-context gap.
- The MMA kernel splits the value dimension across the four waves of a
  workgroup only through the accumulator layout, not across workgroups, so a
  16-row query tile is the smallest unit of parallelism.
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

## 9. Gated delta-net

### 9.1 Chunkwise form

`kernels/hip/prefill/gdn_chunk.hip` implements the chunkwise WY form of the same
recurrence and is validated against `gdn_chunk_splitreg_f32_kernel` and the CPU
reference, but it is **not the selected path**: it is slower than the per-token
kernel on gfx1151 (see the measurements below). The op selects it only when the
caller supplies the scratch buffer of `yuuka_op.buf[12]`; the prefill graph
leaves that slot null.

With `p_r = gamma_1 .. gamma_r`, `g_r = log p_r`, and `g_c` the last cumulative
decay of a chunk of `n` tokens, the recurrence

```text
S_r = gamma_r (I - beta_r k_r k_r^T) S_{r-1} + beta_r k_r v_r^T
```

becomes, with `K/Q/V` the normalized chunk rows and `M = I + B` where
`B[r][s] = beta_r (k_r . k_s)` for `s < r`:

```text
S_C = exp(g_c) S_0 + K^T (M^-1 diag(beta) (exp(g_c - g) V - exp(g_c) K S_0))
o_r = exp(g_r) (q_r S_0 + Z (diag(beta exp(-g)) V - K S_0))_r
Z   = (Q K^T on the lower triangle) M^-1
```

The decay of a row is a scalar for the whole state matrix, so it factors out of
the triangular solve and only rows of the chunk interact. The chunk length is
fixed at 32 so the triangular system is one warp-wide inversion and the state
projection is a 32-row tile.

Three kernels run per prompt chunk:

- `gdn_prepare_rows_f32_kernel` normalizes the q and k segments of every row and
  replaces the raw alpha and beta projections with the log decay and the sigmoid
  gate. It consumes `conv`, `alpha`, and `beta` in place; the callers treat them
  as scratch.
- `gdn_prep_f32_kernel` runs once per (chunk, head) and writes `M^-1` and `Z`.
  One thread owns one column of the system and a quarter of the state
  dimension, so a dot product keeps 32 accumulators live and resolves with two
  shuffles. The inverse is a column-wise forward substitution on one wave.
- `gdn_scan_f32_kernel` walks the chunks of one head in order with one state
  column block per workgroup (32 of 128 columns, 8 column groups by 8 state
  parts, two columns per thread). The state slice stays in registers, the state
  projections reduce across the eight parts of a column group with shuffles,
  and the triangular products hold only the thread's own rows. Chunk-row tiles
  are staged in LDS as `tile[(i / 16) * 8 + (i % 16)][r]` so the eight parts of
  a column read windows that differ by the padded stride and land on different
  banks.

Measured with `build/yuuka_hip_prefill_sweep`-style isolation
(`/tmp/bench_gdn`, one op in a plan, 24 layers, 512 rows): the chunked path
takes 3.10 ms per layer, the per-token split-register path 2.01 ms. Per kernel,
the scan is 1.73 ms, the prep 0.85 ms, and the existing convolution, gated
RMSNorm, and preparation kernels add 0.36 ms. The scan reaches 0.4 IPC per CU:
it is LDS-latency bound, with 106-164 VGPRs and three 8-wave workgroups resident
per CU against a 47-67 cycle LDS latency, and it issues 2.7M instructions per CU
per layer. Every product is an FP32 FMA-pipe loop; the reference implementation
instead runs its dense products on the FP16 WMMA path (`mmb_dense`), which is
the step that would make this form competitive (see
[roadmap3.md](roadmap3.md#51-implementation-2026-11-18)).

Correctness: `tests/hip/prefill_test.cpp` runs both paths against the same CPU
reference (chunked maximum absolute difference `1.6e-7`, per-token `2.1e-7`) and
checks that a repeated execution reproduces the outputs bitwise.

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

## 10. Dispatch and tuning

### 10.1 Shape signature

Dispatch decisions and plan validation use these shape and capability fields:

```text
operator, M, N, K, input dtype, weight dtype, batch, sequence class, backend
```

There is no generic kernel registry in this release. Go selectors map the
fields they need to a concrete path, and plan construction validates supported
dtypes, row layout, buffer ranges, and dimensions before submission. Lookup is
per plan, not per tensor per token.

### 10.2 Dispatch selection and validation

Dispatch is recorded in one explicit Go selector:
`backend/hip.SelectPrefillPath` chooses batched GEMV, custom rocWMMA, or the
rocBLAS FP16 baseline from dtype, rows, and the input/output widths.
Single-token decode submits one matvec per matrix. `SelectDecodeGemvPath`
chooses between the per-row GEMV and the streaming kernel per operation
(section 5.3), and the output projection takes the same rule; there is no
separate runtime GEMV batch selector.

Each plan addition validates the dtype, quantization row layout, shape
constraints, and buffer ranges. Unsupported combinations fail during planning;
there is no size-keyed registry, scratch registry, or graph-compatibility
registry to consult at runtime.

### 10.3 Offline tuning

Dispatch decisions are offline. Thresholds measured on gfx1151 are encoded in
`backend/hip.SelectPrefillPath`, keyed by dtype, shape class, and row class, and
recorded with the measurement environment in [benchmarks.md](benchmarks.md).
There is no runtime autotuner. Adding a device requires re-running the sweeps
and updating the selector and tests.

### 10.4 Fallback

If a custom kernel is unavailable, unsupported for a shape, or slower than the
library baseline, the planner selects the documented fallback. Silent numerical
substitution is forbidden. A fallback must be visible in diagnostics and
benchmarks.

## 11. Numerical requirements

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

## 12. Kernel acceptance criteria

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
