# Prefill Gap Closure Roadmap

## 1. Purpose and scope

[roadmap2.md](roadmap2.md) closed the prefill work with absolute gates: a
512-token chunk at 1324 tok/s and position-7680 attention at 643 ms. Against
the llama.cpp reference on the same device and the same GGUF the path is still
about 1.8x slower at 128-512 prompt tokens and about 2.0x slower at 2048, so
the remaining work is best expressed as a gap against the reference, not as
another absolute target.

This document records the 2026-11-18 attribution of that gap and turns it into
an ordered plan with per-stage acceptance gates. It supersedes the ordering in
[roadmap2.md](roadmap2.md) section 2 for prefill work; roadmap2's per-stage
history and rejected variants remain valid and are not repeated here.

Scope:

- `qwen35` single-request prefill on gfx1151: the int8 MMQ GEMM family, the
  gated delta-net chunk kernel, prefill attention, and the prefill LM head.
- The dispatch, counter, and correctness gates needed to accept a change.

Non-goals:

- Decode kernels. Decode has its own operator family; the LM head change in
  Stage 4 is limited to the prefill graph.
- Expanding quantized weights to FP16, or re-quantizing the GGUF.
- Porting CDNA MFMA tiles or W7900 numbers to gfx1151.
- NPU work, multi-request batching, speculative decoding.

Ownership: [kernels.md](kernels.md) owns kernel contracts,
[benchmarks.md](benchmarks.md) owns the measurement record, and
[testing.md](testing.md) owns the benchmark protocol and regression gates. If a
stage changes a documented kernel contract or tolerance, update the owning
document in the same change.

### 1.1 Implementation status (2026-11-18)

| Stage | Status | Gate |
| --- | --- | --- |
| S0: counter and attribution tooling in `scripts/` | Implemented (2026-11-18), see 3.1 | Reproducible counters for both engines |
| S1: MMQ GEMM layout, epilogue, and K-stage pipelining | Not started | Isolated Q4_K `2560 -> 9216` at `M=512` at 19 TF/s or more |
| S2: gated delta-net chunked form | Not started | 512-row chunk at 15 ms or less for 24 layers |
| S3: MMA prefill attention | Not started | 512-token chunk at 6 ms or less; position 7680 at 150 ms or less |
| S4: prefill LM head via batched GEMM | Not started | 4 ms or less per chunk |

Only S0 is implemented; S1-S4 are not. The tables in section 2 are the decision
summary; [benchmarks.md](benchmarks.md) stays the authoritative record of
measured throughput, and this document records the attribution that motivates
the order.

## 2. Baseline attribution (2026-11-18)

### 2.1 Conditions

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 recurrent), hidden 2560, FFN 9216, 16 heads, 4 KV heads, head dim 256 |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | Ryzen AI Max+ 395, 120 GiB shared memory, Linux 7.0.14-28-pve |
| Code | `kernels/`, `engine/gpu/`, `backend/hip/` at `9767284` |
| Reference | `strix-llama.cpp` `c8912d7f5` (build 12216), ROCm backend, `-ngl 99 -b 512 -ub 512` |
| yuuka KV | F32 (llama.cpp uses its default F16; `llama-bench` rejects `f32`) |
| Chunk / batch | 512 tokens on both sides |
| Sampling | Greedy, temperature 0 |

Methods:

- end-to-end: `yuuka bench` (warmup 1, 3 repetitions) and `llama-bench`
  (3 repetitions); both median.
- yuuka per-operation time: `TestGPUProfilePrefill`, which wraps every
  `ExecutionPlan` operation in a HIP event pair (651 operations per 512-row
  chunk).
- llama.cpp per-kernel time: `rocprofv3 -f rocpd --kernel-trace true` around
  `llama-bench -p 512 -n 0 -r 1`, divided by the two traced prefills (warmup
  plus one measured run).
- instruction mix and stalls: `rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_LDS
  SQ_WAVES` and `--pmc LDSBankConflict ALUStalledByLDS LdsLatency
  OccupancyPercent`, on the native `yuuka_prefill_sweep` (whose per-MMA
  denominators are known from the sweep's cell list) and on `llama-bench`.
- `rocprofv3` does not flush its output database for a Go process, because the
  Go runtime exits without running glibc `atexit` handlers. yuuka per-kernel
  counters therefore come from the native sweep binary, not from
  `yuuka bench`.

Neither run pins clocks, power, or thermal state. Treat the shares as a work
order, not as release numbers.

### 2.2 End-to-end

| Test | yuuka t/s | llama.cpp t/s | yuuka/llama |
| --- | ---: | ---: | ---: |
| pp128 | 1063.7 | 1778.3 | 0.60 |
| pp512 | 1333.8 | 2373.3 | 0.56 |
| pp2048 | 1202.1 | 2357.6 | 0.51 |

### 2.3 Per-family attribution for one 512-token chunk

yuuka figures are operation times at position 0; llama.cpp figures are kernel
times divided by two traced prefills. Families are grouped so that the same
mathematical work is compared, not the same kernel count.

| Family | yuuka | llama.cpp | Delta |
| --- | ---: | ---: | ---: |
| Quantized matmul (body) | 291.0 ms | 167.0 ms | +124.0 ms |
| Gated delta-net chunk | 47.9 ms | 12.1 ms | +35.8 ms |
| Prefill attention | 17.6 ms | 3.7 ms | +13.9 ms |
| LM head | 11.3 ms | 2.7 ms | +8.6 ms |
| RMSNorm (all variants) | 3.2 ms | 5.0 ms | -1.7 ms |
| SwiGLU / gated activation | 6.9 ms | 10.2 ms | -3.3 ms |
| Q8_1 activation quantization | 4.0 ms | 7.0 ms | -3.0 ms |
| Residual add | 2.8 ms | 3.1 ms | -0.3 ms |
| RoPE | 0.4 ms | 1.2 ms | -0.7 ms |
| KV write, embedding, casting, misc | 0.3 ms | 1.6 ms | -1.4 ms |
| Inter-kernel gaps | 0.0 ms | 2.3 ms | -2.3 ms |
| **Total** | **385.3 ms** | **215.7 ms** | **+169.6 ms** |

So 73% of the gap is the quantized matmul, 21% is the gated delta-net chunk,
8% is attention, and 5% is the LM head. Elementwise and normalization work is
about 10 ms in yuuka's favour, against about 2 ms of inter-kernel gaps on the
reference side. Kernel launch count is not a factor: yuuka submits 651
operations per chunk, llama.cpp dispatches 1098 kernels per chunk.

The same grouping explains the pp2048 difference (1.704 s against 0.878 s):
matmul about +496 ms, gated delta-net about +143 ms, attention about +141 ms,
and the LM head about +34 ms, partly offset by yuuka's cheaper elementwise
kernels.

### 2.4 Quantized matmul: instruction mix and stalls

Both engines run the same algorithm (Q8_1 activations, K-quant weights decoded
to int8, 16x16x16 int8 WMMA, per-32-value FP32 scales) with the same
`I=64` x `J=128` output tile and the same 256-value K step. Isolated kernel,
Q4_K, `K=2560`, `N=9216`, `M=512`, same device:

| Metric | yuuka `gemm_mmq_kernel<Q4_K>` | llama.cpp `mul_mat_q<Q4_K,128>` |
| --- | ---: | ---: |
| VALU instructions per MMA | 24.8 | 17.1 |
| LDS instructions per MMA | 4.9 | 2.0 |
| `LDSBankConflict` | 35.9% | 11.7% |
| ALU stalled by LDS | 0.3% | 4.3% |
| `LdsLatency` (cycles) | 67.6 | 127.3 |
| `OccupancyPercent` | 23.9% | 16.7% |
| Resident waves per CU | 8 (one 256-thread block) | 4 (one 128-thread block) |
| Achieved | 12.6 TF/s | 21.9 TF/s |

Occupancy is not the deficit: yuuka keeps twice as many waves resident and is
still 1.74x slower, so the loss is inside each wave's instruction stream. The
four causes below are ordered by expected cost, not by an isolated measurement
of each one:

1. **No MMA data layout in LDS.** `kernels/hip/gemm/mmq_gemm.hip:37` stages
   both operands as plain int8 rows with a 272-byte stride and loads fragments
   with `rocwmma::load_matrix_sync` (row-major A, col-major B). Each fragment
   becomes several generic LDS reads plus per-lane address arithmetic, and a
   272-byte stride places rows 8 apart on the same bank, which is the measured
   35.9% conflict rate. llama.cpp stores both tiles in a swizzled MMA layout
   and reads each fragment with one 128-bit LDS access (2.0 LDS per MMA, 11.7%
   conflicts).
2. **Expensive weight unpacking.** `stage_weight_span` and `q6k_span8`
   (`mmq_gemm.hip:124`) read 16-bit halves because K-quant blocks are only
   2-byte aligned and then shift and OR per value. llama.cpp reads 16 bytes at
   a time and unpacks four 4-bit fields per int32.
3. **No K-stage pipelining.** `mmq_gemm.hip:331` runs
   `stage_weights -> stage_activations -> __syncthreads -> MMA -> __syncthreads`
   for every 256-value step, with a single operand buffer and one block per CU
   (64,512 B of the 64 KiB LDS budget). Every step exposes the full weight-load
   latency with no other block to fill it. llama.cpp has an explicit register
   prefetch path on RDNA 3.5 and splits the activation tile into halves staged
   around the two `vec_dot` calls.
4. **Epilogue arithmetic.** The accumulator update is three VALU operations per
   element per 32-value block (`wd[e]*dsb.x` is not hoisted out of the
   four-accumulator loop), against llama.cpp's two; that is about 12 of the
   24.8 VALU per MMA.

Two smaller dispatch issues belong to the same stage: `ssm_beta`/`ssm_alpha`
(`nOut = 32`) still select a 64x128 tile whose grid has four blocks, and the
q/k/v projections in a full-attention layer each run their own Q8_1 pass over
the same normalized tile.

### 2.5 Gated delta-net chunk

`kernels/hip/prefill/prefill.hip:392 gdn_chunk_splitreg_f32_kernel` keeps the
state column in registers, but it walks the chunk token by token:
`for (int r = 0; r < rows; ++r)` at line 423 with two `__syncthreads` per row,
so a 512-row chunk has 1024 serialization points per layer and no tensor-core
work. The layer grid is `heads * State/32 = 128` blocks of 128 threads (the
recurrent `heads` is the time-step rank, 32), that is 512 waves on a device
with about 1280 wave slots, and each block is one strictly serial dependency
chain.

Measured 2.0 ms per layer, 47.9 ms for 24 layers. llama.cpp expresses the same
recurrence in chunked WY form (`gdn_chunk_prep`, four small `mmb_dense`
matmuls, `gdn_chunk_scan`) so the intra-chunk work becomes MMA GEMM and only a
short scan is sequential: 12.1 ms for the same 24 layers (prep, scan, dense
products, concat, and the bf16 conversion of the dense inputs). This is the
stage A4 that roadmap2 deferred as unnecessary for its absolute gate; against
the reference it is now the second largest gap.

### 2.6 Prefill attention

`prefill.hip:170` and `prefill.hip:816` use one wave per query row and one
dot product per key: eight values per lane, a five-step `__shfl_xor`
reduction, a scalar softmax update, then eight FMA for V. K and V are staged
16 keys at a time in FP32 with two barriers per tile, and the four query heads
that share a KV head only share the staged tile above `kGQAThreshold = 2048`.
No stage uses tensor cores.

llama.cpp runs `flash_attn_ext_f16<256,256,16,4>` (`fattn-mma-f16.cuh`, the
RDNA WMMA path) on F16 KV. Measured attention per 512-token chunk: 17.6 ms
against 3.7 ms at a cold context; the same chunk at position 7680 takes
537 ms (59% of that chunk), while a whole 2048-token prefill on the reference
spends 33.4 ms on attention across its four chunks. Attention is 4.6% of the
chunk at a cold context and the dominant term at long context, which is why
pp8192 has the worst ratio in [benchmarks.md](benchmarks.md).

### 2.7 Prefill LM head

`engine/gpu/plan.go` finishes the prefill graph with `AddRMSNorm` plus
`AddGemv` on the last row. The model has no `output.weight`, so the tied Q6_K
`token_embd.weight` (521 MB) is read by the decode GEMV at about 46 GB/s:
11.3 ms. llama.cpp's `mul_mat_vec_q_fq` takes 2.7 ms for the same matrix
(about 227 GB/s). This is the same single-row bandwidth problem as decode and
it costs about 9% of the chunk at 128 prompt tokens.

## 3. Stage S0: reproducible counter attribution

Do this first so every later stage is judged on one script instead of ad-hoc
commands.

1. Add `scripts/prefill_counters.sh` that:
   - requires a built `libyuuka_hip.so` and prints the recorded GGUF SHA-256,
     ROCm version, and `gfx1151` from `rocminfo`;
   - collects `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQ_WAVES` and
     `LDSBankConflict`, `ALUStalledByLDS`, `LdsLatency`,
     `OccupancyPercent` with `rocprofv3 -f rocpd --kernel-trace true` around
     `build/yuuka_prefill_sweep`, and around `llama-bench -p 512 -n 0 -r 1`
     when a reference binary is provided;
   - reduces the database to the per-MMA table in section 2.4 and prints a
     machine-readable summary.
2. Record the reduction denominator in the script: the sweep runs seven
   submissions per cell (two warmups plus five timed), and the per-cell MMA
   count is `n_in * n_out * rows / 4096`, so the script must pair
   `gemm_mmq` dispatches with cells in submission order rather than aggregate
   across cells.
3. Document the Go `rocprofv3` limitation (no `atexit` flush) in
   [testing.md](testing.md) next to the existing profiling notes.

Acceptance: one command reproduces section 2.4 within run-to-run noise on two
consecutive runs, and the script fails loudly when the model hash or the
device does not match.

### 3.1 Implementation (2026-11-18)

`scripts/prefill_counters.sh` runs `build/yuuka_prefill_sweep` once unprofiled
to read the cell list and achieved throughput, then twice under `rocprofv3`:
`SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES`, and `LDSBankConflict ALUStalledByLDS
LdsLatency OccupancyPercent`. It prints the date, commit, host, device,
performance level, ROCm and HIP version, and the recorded GGUF SHA-256, which it
verifies against a supplied `--model`. It exits with status 2 when the GPU is
not `gfx1151` or the hash differs, and when a profiled run does not exercise the
same cells as the unprofiled one. `scripts/prefill_counters.py` reduces the
databases to the section 2.4 table plus one machine-readable CSV row per cell.

Reduction: each cell owns seven `gemm_mmq` dispatches in submission order,
instruction counters are summed over them, and percentage and cycle counters are
averaged over them. The per-MMA denominator is the executed MMA count, because
sweep cells with fewer than 128 rows still run the padded tile; for the
section 2.4 cell the executed and useful counts agree at 2,949,120 per
submission. The reduction prints how many cells the two counts disagree on, and
the CSV carries both counts.

Two consecutive runs at `80ab905` on `gfx1151`, ROCm 10.1.0, HIP 7.16.26385,
`power_dpm_force_performance_level=auto`, both with the GGUF SHA-256 verified
through `--model`:

| Metric | Run 1 | Run 2 | Section 2.4 |
| --- | ---: | ---: | ---: |
| VALU per MMA | 24.77 | 24.77 | 24.8 |
| LDS per MMA | 4.89 | 4.89 | 4.9 |
| `LDSBankConflict` | 35.94% | 35.94% | 35.9% |
| ALU stalled by LDS | 0.29% | 0.29% | 0.3% |
| `LdsLatency` | 68.21 cycles | 68.64 cycles | 67.6 cycles |
| `OccupancyPercent` | 23.95% | 23.95% | 23.9% |
| Waves per workgroup | 8.00 | 8.00 | 8 |
| Achieved | 13.80 TF/s | 14.40 TF/s | 12.6 TF/s |

The instruction counters reproduce to 0.1%, and `LdsLatency` moves by about 0.6%
between runs. The achieved throughput is the current tree's measurement (13.7
TF/s in [benchmarks.md](benchmarks.md), 13.8-14.4 TF/s in these runs) rather than
the 12.6 TF/s recorded in section 2.4, so that row tracks code and clock state,
not the counters.

The llama.cpp column is filled only with `--llama-bench PATH`. Normalizing the
reference per MMA needs one executed MMA count per dispatch, which depends on
the reference's matrix shapes and is not visible in a kernel trace; the script
therefore reports the reference per wave unless `--reference-mma-counts`
supplies those counts. That path has not been run against a llama.cpp binary on
this host, so the recorded 17.1 VALU and 2.0 LDS per MMA in section 2.4 remain
the only reference measurements. The Go `rocprofv3` limitation is documented in
[testing.md](testing.md#63-what-to-measure-separately).

## 4. Stage S1: MMQ GEMM (highest value, 73% of the gap)

Target: 12.6 -> 19 TF/s or more on the isolated cells, which is about
`19/21.9 = 87%` of the reference kernel and 1.4x on the chunk's matmul time.

### 4.1 S1a: swizzled MMA operand layout

1. Change `quantize_q8_1_kernel` (`mmq_gemm.hip:254`) to write activations in
   the layout the fragment loader consumes, so the copy into LDS is linear.
   This changes the contract of `launch_quantize_q8_1` and the Q8_1 buffer
   reuse between the gate/up pair; update [kernels.md](kernels.md).
2. Replace the 272-byte-stride tiles with a swizzled layout that maps each
   16x16 fragment to one contention-free 128-bit access per lane, and load
   fragments with explicit `ds_read_b128` instead of `load_matrix_sync`.
   Keep the exact int32 fragment-to-output mapping documented above
   `gemm_mmq_kernel` (`mmq_gemm.hip:291`); the numeric result must not change.
3. Keep the epilogue tile alias-free under the new layout, or fall back to a
   separate 32 KiB epilogue buffer if the LDS budget requires it.

Acceptance: `LDSBankConflict` at or below 15%, `LDS/MMA` at or below 2.5, and
`tests/hip/prefill_test.cpp` still within its recorded gates (maximum absolute
difference 6.7e-5 for Q4_K, 1.5e-4 for Q5_K) with identical Q8_1 digits.

### 4.2 S1b: vectorized weight unpacking

Replace the per-value 16-bit reads in `stage_weight_span` / `q6k_span8` with
aligned 16-byte loads and int32 field extraction, handling the 2-byte block
alignment with a single unaligned head or tail load per row. The decoded int8
digits and the scale/min staging must stay bit-identical to the current code;
`tests/hip/prefill_test.cpp` already compares digits, and that check must keep
passing unchanged.

Acceptance: `VALU/MMA` at or below 20 with the epilogue change from S1c, and no
new VGPR spills in `scripts/kernel_resources.sh`.

### 4.3 S1c: hoist the epilogue scale products

Precompute `wd[e] * dsb.x` once per 32-value block for the eight accumulator
elements instead of recomputing it per accumulator tile, so the inner update
becomes two FMA per element. This is arithmetic reordering of a float product
followed by a multiply-accumulate; verify with the existing logit and chunk
gates rather than assuming bit-identity.

Acceptance: VALU per MMA drops by at least 8 with no change in the
`TestGPUChunkedPrefillLongContext` top-1 token and no growth of the maximum
logit difference beyond the recorded 0.05 chunk gate.

### 4.4 S1d: overlap the K stage with the MMA

LDS budget check first. At K step 256 the A tile is 64 rows x 256 int8
(16 KiB) plus 4 KiB of weight scales, and the B tile is 128 rows x 256 int8
(32 KiB) plus 8 KiB of activation scales, that is 60 KiB of the 64 KiB budget
before padding and before the epilogue aliasing. A full double buffer does not
fit, and neither does a double-buffered B tile alone. Acceptable variants, in
order of preference:

1. register prefetch of the next iteration's weight span, as llama.cpp does on
   RDNA 3.5, keeping one LDS buffer;
2. split the K step to 128 values, which makes both tiles 30 KiB, and double
   buffer them; measure the extra barriers against the latency they hide;
3. keep K step 256 with a single buffer and prefetch only the activation tile
   into registers, leaving weight staging on the critical path.

Acceptance: the first two variants must beat the S1a-S1c kernel on all six
`Q4_K`/`Q5_K`/`Q6_K` cells of the sweep at `M=512`, not only on the widest
shape; record the losing variant in the rejected-routes list rather than
keeping it behind a flag.

### 4.5 S1e: narrow-N and shared Q8_1 cleanup

- Route `ssm_beta`/`ssm_alpha` (`nOut = 32`) through the batched GEMV path or a
  narrow-N tile instead of a 64x128 tile with four blocks.
- Quantize the normalized tile once per layer for q, k, and v, as the gate/up
  pair already does, so a full-attention layer runs one Q8_1 pass instead of
  three.

Acceptance: `quant_q8_1` operation time for a 512-row chunk drops by at least
25% and the chunk total improves by at least 1 ms, with `dispatch_test.go`
updated for the new selection.

## 5. Stage S2: gated delta-net chunked form

Implement the chunkwise gated delta rule described in roadmap2 section 3.5,
using the Stage S1 GEMM for the intra-chunk products. Chunk size 32 matches
the reference and keeps the triangular solve small.

Steps:

1. Add a CPU reference for the chunked form next to the existing per-token
   recurrence and require agreement to 1e-4 on random `q`/`k`/`v`/decay/beta
   before any device work.
2. Implement the device path as prep, dense products, solve, and scan kernels;
   keep `gdn_chunk_splitreg_f32_kernel` as the correctness oracle and as the
   fallback for shapes the chunked path does not cover.
3. Keep the decode recurrence untouched; the two paths must produce the same
   state matrix within the recorded gate.

Acceptance: 512-row chunk at 15 ms or less for 24 layers (from 47.9 ms), same
greedy tokens as the split-register kernel over a 1024-token prompt, and no
`__syncthreads` loop that scales with chunk length.

Risk: the chunked form changes the summation order of the state update. Gate it
on logits, not on bitwise identity, and state the measured maximum difference
in the commit body.

## 6. Stage S3: MMA prefill attention

1. Add a WMMA attention kernel for the paged prefill path: 16x16 QK^T over a
   staged K tile, FP32 online softmax, 16-key V tiles, and one workgroup
   serving the four query heads that share a KV head regardless of context
   length.
2. Default to F16 K/V staging for the attention tile, reusing the existing
   `cast` kernels; keep the F32 tiled kernel as the fallback for formats the
   MMA path cannot stage.
3. Keep the current tiled kernel for contexts below the crossover only if it
   measures faster; do not keep it as a silent fallback.

Acceptance: attention at or below 6 ms for a 512-token chunk at position 0 and
at or below 150 ms at position 7680 (from 17.6 ms and 537 ms), maximum logit
difference against the tiled kernel inside the recorded 1.12e-3 F16 KV gate,
and identical greedy tokens on the 1024-token chunked-prefill test.

## 7. Stage S4: prefill LM head

Replace the last-row GEMV in `buildPrefillPlan` with a batched matmul over the
final row block (or the whole chunk when the logits are needed for
speculative verification). Reuse the Stage S1 MMQ kernel or the existing WMMA
kernel, whichever the shape dispatch selects.

Acceptance: 4 ms or less per chunk (from 11.3 ms), unchanged top-1 token for
the 128/512/2048 prompt rows in [benchmarks.md](benchmarks.md), and no change
to the decode-path GEMV dispatch.

## 8. Cross-cutting gates

Every stage lands with:

- `go build ./...`, `go vet ./...`, `go test ./...`, `go test -race ./...`.
- `cmake --build build -j"$(nproc)"` and `ctest --test-dir build
  --output-on-failure` (9 tests today).
- `go test -tags hip ./...` with `YUUKA_TEST_MODEL` set, including
  `TestGPUMatchesCPU` and `TestGPUChunkedPrefillLongContext`.
- `scripts/kernel_resources.sh` output for every touched kernel, with zero new
  spills.
- `build/yuuka_prefill_sweep` before/after for the six K-quant cells at
  `M=512`, plus `yuuka bench` rows at 128, 512, 2048 and 8192 under the
  [testing.md](testing.md) protocol when the stage claims an end-to-end gain.
- Updated [benchmarks.md](benchmarks.md) rows and an updated status row in
  section 1.1 of this document. Performance claims without the recorded
  model hash, quantization, ROCm version, device, and power state are not
  accepted.

Do not weaken a tolerance or delete a counter check to land a stage. If a stage
fails its gate, record the measured result and the rejected variant in
Section 10 instead of loosening the gate.

## 9. Order and expected effect

| Order | Stage | Chunk time | pp512 |
| --- | --- | ---: | ---: |
| 0 | Baseline | 385 ms | 1334 t/s |
| 1 | S1 MMQ GEMM at its 19 TF/s gate | 287 ms | 1784 t/s |
| 2 | S2 delta-net at 15 ms | 254 ms | 2016 t/s |
| 3 | S3 attention at 6 ms (cold context) | 243 ms | 2109 t/s |
| 4 | S4 LM head at 4 ms | 236 ms | 2174 t/s |

These are projections from the measured per-family times at the stage gate
values, not measurements. They assume every stage reaches its gate and nothing
regresses. The long-context gain is larger because S3 removes the term that
dominates at 8192 tokens.

## 10. Rejected routes and open questions

Rejected under this plan (record the measurement in the commit body if a
variant is tried again):

- Adding warps or blocks before fixing the instruction mix: yuuka already
  keeps twice the reference's resident waves.
- Keeping the 272-byte-stride tile and only changing the padding: the conflict
  pattern is a consequence of the layout, not of the pad width.
- Moving the delta-net recurrence to more workgroups without changing the
  algorithm: the serialization is per token row, not per block.

Open questions:

1. Does the LDS budget allow a swizzled layout plus any prefetch variant, or
   must prefetch live in registers? Section 4.4 gives the budget arithmetic;
   confirm it with the compiler's LDS report before committing to a variant.
2. Should the Q8_1 layout stay a stable ABI, or is it internal to the plan? The
   gate/up sharing depends on it, and `tests/hip/prefill_test.cpp` compares
   digits produced by the host quantizer.
3. For S2, is one triangular solve per 32-token chunk cheap enough at rank 32,
   or is a smaller chunk needed to keep the solve off the critical path?
4. For S3, does an F16 K/V tile in a 64 KiB LDS budget cover head dim 256 with
   16-key tiles, or does the tile have to shrink to 8 keys at that head size?
