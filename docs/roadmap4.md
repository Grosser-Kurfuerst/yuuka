# Decode Gap Closure Roadmap

## 1. Purpose and scope

[roadmap3.md](roadmap3.md) closed the prefill gap with per-stage gates and left
prefill at 0.66-0.77 of the llama.cpp reference on the same device and GGUF.
Decode is the remaining large gap: 9.61 tok/s against 62.39 tok/s, a factor of
6.5. Unlike prefill, that factor is not spread over several operator families.
The 2026-11-18 attribution below puts 89.7% of one decode step in the
single-row quantized GEMV family, which reaches 31 GB/s on a device whose
measured streaming read rate is 244 GB/s and whose own streaming LM head kernel
already reaches 180-226 GB/s on the same matrix.

This document records that attribution and turns it into an ordered plan
(D0-D5) with per-stage acceptance gates. It supersedes no earlier document:
[roadmap.md](roadmap.md) owns phases, roadmap2/roadmap3 own prefill.

Scope:

- `qwen35` single-request decode on gfx1151: the decode GEMV family, the decode
  output projection route, the gated delta-net per-token kernel, per-step launch
  structure, and the tooling and gates needed to accept a change.
- The dispatch thresholds and correctness gates that a decode kernel change
  must not break.

Non-goals:

- Prefill kernels, dispatch, and gates. [roadmap3.md](roadmap3.md) owns them.
- Fused mixed-shape multi-request decode batching, KV quantization policy,
  speculative decoding (already implemented and opt-in), and `qwen35moe` HIP
  execution.
- Expanding quantized weights to FP16, re-quantizing the GGUF, or any change to
  the GGUF arithmetic that is not measured against the CPU reference.
- Porting CDNA MFMA assumptions or W7900/CDNA bandwidth numbers to gfx1151.
- Copying implementation code from llama.cpp or GGML. Those trees are reference
  material for algorithms and for measurement; a route derived from them is
  reimplemented against this repository's ABI, layouts, and tests, and is
  attributed in the commit body.

Ownership: [kernels.md](kernels.md) owns kernel contracts and numerics,
[benchmarks.md](benchmarks.md) owns the measurement record,
[testing.md](testing.md) owns the benchmark protocol and regression gates, and
[serving.md](serving.md) owns scheduler and API behavior. If a stage changes a
documented kernel contract, dispatch rule, or tolerance, update the owning
document in the same change.

### 1.1 Implementation status (2026-11-18)

| Stage | Status | Gate |
| --- | --- | --- |
| D0: reproducible decode attribution tooling | **Done** (2026-11-18) | One command reproduces 2.3 and 2.4 within noise |
| D1: decode output projection on the streaming kernel | **Done** (2026-11-18) | Decode p50 13.21 s -> 12.2 s or less; tokens unchanged |
| D2: channel-parallel decode GEMV | Not started | Isolated decode shapes at 150 GB/s or more; decode at 28 tok/s or more |
| D3: Q8_1 activations with `dp4a` | Not started | Isolated decode shapes at 185 GB/s or more; decode at 31 tok/s or more |
| D4: gated delta-net per-token kernel | Not started | 24 layers at 2.0 ms or less (from 7.23 ms); decode at 37 tok/s or more |
| D5: launch gaps and operator fusion | Not started | Inter-operation gaps at 1.5 ms or less; decode at 45 tok/s or more |

D0 is implemented; its record is 3.4 and its measurements are in
[benchmarks.md](benchmarks.md#decode-matvec). D1 is implemented and meets its
gate; its record is 4.5. D2-D5 are not started. The measurements in section 2
describe the tree at commit `bfd36c0` and are the baseline each gate is compared
against; each stage record states what that stage moved.

## 2. Baseline attribution (2026-11-18)

### 2.1 Conditions

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model file size | 2,740,937,888 bytes; quantized matmul weights 2,726 MB per token |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 gated delta-net) |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs, Wave32 |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 120 GiB shared memory |
| Go | go1.24.4 |
| Commit | `bfd36c0` |
| Sampling | Greedy, temperature 0 |
| Prompt / decode | 128 prompt tokens, 128 generated tokens |
| Warmup / repetitions | 1 / 2 for the yuuka row; 3 for the `llama-bench` row |
| llama.cpp reference | `strix-llama.cpp` commit `c8912d7f5`, build 12216, ROCm backend, `-ngl 99 -b 512 -ub 512` |

Commands:

```bash
./yuuka bench -backend hip -model "$YUUKA_TEST_MODEL" \
    -prompt-tokens 128 -decode-tokens 128 -reps 2 -warmup 1

../strix-llama.cpp/build/bin/llama-bench -m "$YUUKA_TEST_MODEL" \
    -p 128 -n 128 -r 3 -ngl 99 -b 512 -ub 512

YUUKA_TEST_MODEL="$YUUKA_TEST_MODEL" \
    go test -tags hip ./engine/gpu -run TestGPUProfileDecode -v
```

Only one benchmark process may use the device at a time. Device clock, power,
and thermal state are not pinned, so the ratios below are the stable quantity,
not the absolute numbers.

### 2.2 End to end

| Test | llama.cpp | yuuka | yuuka / llama |
| --- | ---: | ---: | ---: |
| pp128 | 1766.62 +- 233.55 t/s | 1359.91 t/s | 0.77 |
| tg128 | **62.39 +- 0.11 t/s** | **9.61 t/s** | **0.15** |

Derived:

| Quantity | llama.cpp | yuuka |
| --- | ---: | ---: |
| Time per generated token | 16.03 ms | 104.1 ms |
| Effective quantized-weight read rate | about 170 GB/s | about 26 GB/s |
| Share of the 244 GB/s streaming read ceiling | about 0.70 | about 0.11 |

The 244 GB/s ceiling is the measured pure streaming read recorded in
[benchmarks.md](benchmarks.md#prefill-lm-head); it is the practical roofline of
this device, not a specification number.

### 2.3 Per-operation attribution

`TestGPUProfileDecode` instruments every plan operation with a HIP event and
profiles one decode step after a real-model forward pass. The step contains 579
operations.

| Operation | Time | Share |
| --- | ---: | ---: |
| **GEMV (quantized matvec)** | **87.556 ms** | **89.7%** |
| gated delta-net per-token | 7.233 ms | 7.4% |
| RMSNorm | 0.768 ms | 0.8% |
| embedding lookup | 0.541 ms | 0.6% |
| residual add | 0.349 ms | 0.4% |
| L2Norm | 0.254 ms | 0.3% |
| gated RMSNorm | 0.205 ms | 0.2% |
| conv1d | 0.168 ms | 0.2% |
| SwiGLU multiply | 0.158 ms | 0.2% |
| other (KV write, RoPE, scale) | 0.136 ms and below | 0.1% |
| profiled total | 97.605 ms | 100% |

The end-to-end step is 104.1 ms, so about 6.5 ms (6.2%) is not inside any
operation: it is the submission gap between 579 dependent launches on one
stream. Event instrumentation makes the per-operation numbers a diagnosis, not
a release measurement.

### 2.4 Isolated decode GEMV shapes

One `YUUKA_OP_GEMV` per submission, one row, the exact shapes and dtypes the
model dispatches, three warmups and ten timed submissions, median. These
numbers are an upper bound for the shape in the real step because the matrix is
re-read while hot; the real step is slower (2,726 MB / 87.556 ms = 31 GB/s).

| Tensor | dtype | n_in -> n_out | ms | GB/s | ns per row |
| --- | --- | --- | ---: | ---: | ---: |
| `ffn_gate` | Q4_K | 2560 -> 9216 | 0.3460 | 38.4 | 37.5 |
| `ffn_up` | Q4_K | 2560 -> 9216 | 0.3503 | 37.9 | 38.0 |
| `ffn_down` | Q6_K | 9216 -> 2560 | 0.2544 | 76.1 | 99.4 |
| `attn_q` | Q4_K | 2560 -> 8192 | 0.3142 | 37.5 | 38.4 |
| `attn_k` | Q4_K | 2560 -> 1024 | 0.0700 | 21.1 | 68.3 |
| `attn_v` | Q6_K | 2560 -> 1024 | 0.0604 | 35.6 | 59.0 |
| `attn_output` | Q4_K | 4096 -> 2560 | 0.1464 | 40.3 | 57.2 |
| `attn_gate` | Q4_K | 2560 -> 4096 | 0.1689 | 34.9 | 41.2 |
| `attn_qkv` (recurrent) | Q5_K | 2560 -> 8192 | 0.3529 | 40.9 | 43.1 |
| `ssm_out` | Q5_K | 4096 -> 2560 | 0.1838 | 39.2 | 71.8 |
| `ssm_beta` | Q8_0 | 2560 -> 32 | 0.0095 | 9.2 | 296.5 |
| `lm_head` (tied `token_embd`) | Q6_K | 2560 -> 248320 | 11.2383 | 46.4 | 45.3 |

The model carries mixed per-tensor dtypes inside one quantization preset:
`ffn_down` is Q6_K in 16 layers and Q4_K in 16, and `attn_v` is Q6_K in 5
layers and Q4_K in 3. The table lists each shape at the Q6_K/Q4_K variant
named; the byte totals below are the exact per-token sum.

The per-token row count is 1,380,352 output rows (671,744 FFN, 102,400 full
attention, 357,888 recurrent, 248,320 output projection) over 2,726 MB of
packed weights. Weighting the table by each shape's byte count gives a
projected GEMV time of about 63 ms against the measured 87.6 ms; the
difference is the cold-weight effect in the real step, where every matrix is
read once.

These cells are reproduced by `build/yuuka_gemv_decode_sweep`, including the
batched and streaming forms on the same shapes, and recorded in
[benchmarks.md](benchmarks.md#decode-matvec).

The `ssm_beta`/`ssm_alpha` cell is not bandwidth at all: 32 output rows cost
9.5 microseconds, which is the launch and block-reduction floor, repeated 48
times per step.

### 2.5 Why the kernel is the limiter, not the device

Four independent observations, all measured on the current tree:

| Observation | Measurement | Consequence |
| --- | --- | --- |
| Equal weight bytes per row, different decode cost | Q4_0 `2560 -> 2560` 1440 B/row at 0.062 ms; Q4_K same 1440 B/row at 0.125 ms | The cost is in the per-weight instructions and the per-block barriers, not in the bytes |
| Same weight bytes, more rows amortized | `yuuka_gemv_sweep`, `batch 16` is 3.9x faster per row than one submission per row for Q4_K | Per-row fixed cost (block reduction, small block) dominates the single-row form |
| The same device and the same matrix, different kernel | `lm_head_kernel` on the tied 521 MB Q6_K matrix: 2.896 ms (180 GB/s), per-row GEMV: 11.937 ms (44 GB/s) in the same sweep | The hardware can stream this model at 4-5x the current decode rate |
| dtype ladder at identical structure | Q8_0 76, Q4_0 59, Q6_K 46, Q4_K 29 GB/s (`2560 -> 2560`) | Throughput falls with the number of scalar operations per weight and with the number of barriers per K step |

### 2.6 Code-level causes

`kernels/hip/gemv/gemv_quant.hip`:

1. **One 256-thread block per output row** (`gemv_packed_kernel`,
   `gemv_q6_k_kernel`, `gemv_k_kernel`, launchers below). For `n_in = 2560`
   and Q4_K a row is 1440 bytes, so each thread holds 5.6 bytes: one or two
   scalar loads. There is no per-thread memory-level parallelism to cover a
   DRAM round trip.
2. **Two `__syncthreads()` per 256-value K block** in `gemv_k_kernel`
   (`gemv_quant.hip:119` and `:138`, the scale/min staging). Ten blocks are 20
   barriers per row. `block_reduce` (`:24-35`) adds eight more per row, so a
   Q4_K row costs 28 block-wide barriers. Barriers serialize the weight loads
   inside a block, which is why the barrier-free kernels (Q6_K, Q8_0, Q4_0) are
   faster at the same or larger byte count.
3. **Scalar, element-addressed loads.** The inner loop indexes
   `i / 256`, `i % 256`, `j / 64`, and a nibble branch per element. No 16-byte
   vector load, no aligned block load, no read-only cache hint.
4. **The activation is FP32 and is re-read once per output row.** A row reads
   10 KB of `x` (`sum += x[b * 256 + tid] * value`) against 1.44 KB of weights,
   so 7 of every 8 bytes the block requests are activation.
5. **No quantized activation, therefore no `dp4a`.** Each weight costs a nibble
   extract, an `int`-to-`float` convert, and two FMA; llama.cpp spends roughly
   0.3-0.5 instructions per weight by quantizing the activation to Q8_1 once per
   token and using `v_dot4_i32_i8` (`__builtin_amdgcn_sudot4` on RDNA3/3.5).

Two secondary routes:

6. **The decode output projection still uses the per-row GEMV** although the
   streaming kernel exists: `engine/gpu/plan.go:149` calls `AddGemv`, while the
   prefill tail uses `AddLMHeadAt` behind `hip.SelectLMHeadPath`
   (`backend/hip/dispatch.go:106`, threshold `lmHeadMinOutputs = 1024`). The
   tied 248,320-row matrix is therefore 11.24 ms instead of 2.9 ms. D1 fixed
   this; the baseline description stays because it is what the gain is measured
   against (4.5).
7. **`gdn_f32_kernel`** (`kernels/hip/ssm/ssm.hip:26-78`) launches one
   128-thread block per head and traverses the 128x128 FP32 recurrent state
   five times per token (decay, `S^T k`, update, `S^T q`, plus the write-back).
   Each head moves 384 KiB, so 24 layers x 32 heads is about 302 MB of state
   traffic per token at 42 GB/s; the floor at 244 GB/s is about 1.2 ms against
   the measured 7.23 ms.

### 2.7 Reference structure (llama.cpp MMVQ)

Read from `../strix-llama.cpp` at `c8912d7f5`:

- One block is one wave (32 threads) and computes **two output rows**
  (`VDR_Q4_K_Q8_1_MMVQ = 2`, `calc_rows_per_block(RDNA3_5) == 1`,
  `calc_nwarps(RDNA3_5) == 1`; `ggml/src/ggml-cuda/mmvq.cu:483-520, 1283-1300`).
  9216 rows are 4608 blocks, and each lane holds about 90 bytes of packed
  weights, that is 5-6 sixteen-byte loads in flight (`vecdotq.cuh:969-1016`).
- **No shared memory and no barrier in the K loop**; the only reduction is a
  wave shuffle at the end.
- **The activation is quantized to Q8_1 once per token** and the inner loop is
  `dp4a`, so four weights retire per instruction; the RDNA3.5 `_fq` variant
  fuses that quantization into the matvec block and prefetches two K iterations
  before the barrier.
- Upstream and fork work shows where the tuning effort sits: `nwarps` for
  batch-1 decode (`llama.cpp` PR 19478), the regression that wide `nwarps`
  causes on narrow matrices (PR 20831), a gfx1151 parameter table (PR 21344),
  the fused-quantization variants (fork commits `9b4944f0c`, `d74a20895`), and
  exactness of the reduction order (fork commit `f9c096c42`). Commit `9b4944f0c`
  records 1480 kernel launches per token with a 2 microsecond gap on gfx1151,
  which is what makes launch structure worth a stage here too.

### 2.8 Derived limits

Per decode step on this model and device:

| Term | Traffic | Floor at 244 GB/s | Current |
| --- | ---: | ---: | ---: |
| Quantized matvec weights | 2,726 MB | 11.2 ms | 87.556 ms |
| Tied output projection | 521 MB | 2.1 ms | 11.238 ms |
| Gated delta-net state (24 layers x 32 heads x 384 KiB) | 302 MB | 1.2 ms | 7.233 ms |
| Attention, 128-token context, 8 layers (F32 KV) | 1.05 MB per layer, up to 4.2 MB with the 4:1 GQA reuse | below 0.2 ms | below 0.2 ms (below the profile's reporting cut) |
| Launch gaps, 579 operations | none | about 1.2 ms at 2 microseconds | about 6.5 ms |

A step at 13 ms of weight traffic plus 2 ms of state traffic plus 2.5 ms of
overheads is about 19 ms, which is the reference's 16 ms plus its larger
non-matvec work. Every stage below is aimed at one row of this table. The
"Current" column is the state at `bfd36c0`; D1 moved the output-projection row
to 2.79 ms (4.5) and left the rest untouched.

## 3. Stage D0: reproducible decode attribution tooling

### 3.1 Problem

The current tooling cannot support the later stages:

- `TestGPUProfileDecode` aggregates by operation code, so a GEMV row cannot be
  attributed to the tensor that produced it, and every stage below needs a
  per-tensor before/after.
- `backend/hip.Plan` keeps only the operation count (`hip.go:353-358`), so the
  Go side cannot name the operation it just timed.
- `rocprofv3` cannot write its database for a Go process (recorded in
  [testing.md](testing.md#63-what-to-measure-separately)), so counter
  attribution has to run on a native binary.

### 3.2 Work

1. Keep a Go-side shadow of the plan: record `code`, `n[0]`, `n[1]`, `u[0]`,
   and the buffer role in each `Add*` method of `backend/hip`, and expose it as
   `Plan.Ops() []OpInfo`. No C ABI change.
2. Extend `TestGPUProfileDecode` to print one line per operation index with the
   tensor role resolved from the model, sorted by time, and to print the
   per-dtype GB/s of every GEMV. Gate the printed totals against 2.3.
3. Add `bench/hip/gemv_decode_sweep.cpp` (built as
   `build/yuuka_gemv_decode_sweep`) that measures one dispatch per row, the
   batched form, and the streaming form for the shapes in 2.4, prints
   `name,dtype,n_in,n_out,ms,GB/s,ns_per_row`, and keeps the matrix resident so
   the numbers are the hot-matrix upper bound. This is the native target for
   `rocprofv3`.
4. Add `scripts/decode_profile.sh` that requires the built library and sweep,
   prints the GGUF SHA-256, `rocminfo` target, and ROCm version, runs the sweep
   under `rocprofv3 -f rocpd --kernel-trace true` with the counters from
   `scripts/prefill_counters.sh`, and reduces the database to GB/s per shape.
   It fails when the device is not gfx1151 or the model hash does not match.
5. Record in [testing.md](testing.md) how decode attribution is produced and why
   it is a diagnostic, not a release measurement.

### 3.3 Acceptance

- One command reproduces the per-tensor table of 2.3 and the GB/s column of 2.4
  within run-to-run noise on two consecutive runs.
- The sweep binary and the profiled step agree on a matrix that exceeds the
  device cache, and the difference on the smaller matrices is measured and
  attributed rather than left open: at the 521 MB `lm_head` matrix they agree
  within 1%, and the 6-20 MB per-layer matrices read 1.3-1.6x faster in the
  sweep because they fit the 32 MB MALL cache that the real step rotates
  through. The comparison is recorded in
  [benchmarks.md](benchmarks.md#decode-matvec).
- No change to measured behavior: `go test ./...`, `go test -tags hip ./...`,
  and `ctest --test-dir build` stay green.

### 3.4 Implementation (2026-11-18)

Added:

- `backend/hip/opinfo.go`: `OpInfo`, `OpRole`, the op-code constants, `OpName`,
  and `opRoles`. `backend/hip.Plan` keeps a shadow slice and exposes
  `Ops() []OpInfo` and `OpCount()`; `Buffer.ID()` gives a slot identity.
  `ExecuteProfiled` now sizes its output from the shadow, and its doc states
  that a `runTail=false` run lines up with a prefix of the shadow.
- `engine/gpu/decode_profile_test.go`: `TestGPUProfileDecode` reports the
  per-operation-code table, the per-tensor table, the per-dtype table, and one
  line per plan index sorted by time, all resolved from the model buffers, and
  gates the result against 2.3.
- `bench/hip/gemv_decode_sweep.cpp` (`build/yuuka_gemv_decode_sweep`): the
  twelve shapes of 2.4 in the per-row, batched, and streaming forms.
- `scripts/decode_profile.sh` and `scripts/decode_profile.py`: one command that
  runs the sweep unprofiled, traces it with `SQ_WAVES SQ_INSTS_VALU
  SQ_INSTS_LDS`, reduces the database to GB/s and per-weight instruction counts,
  and then runs `TestGPUProfileDecode`.
- `docs/testing.md` 6.3 records how the decode attribution is produced;
  [benchmarks.md](benchmarks.md#decode-matvec) holds the measurements.

Three decisions differ from 3.2 as written:

1. **Roles are recorded per op code, not per `Add*` call site.** A buffer role is
   a property of the ABI opcode: every launcher in `bridge.cpp` reads buf[0..2]
   with one meaning for a given code. Recording it once in `opRoles` keeps the
   forty `Add*` methods unchanged, and `opinfo_test.go` fails when a code the C
   ABI declares has no name or no first-slot role, so the table cannot drift
   silently. What 3.2 needs from the shadow is that the Go side can name an
   operation and the buffers it reads; that holds.
2. **Slot identity is a `uintptr` in `OpInfo`, not a `*Buffer`.** The plan
   records what it handed to the C ABI, and inverting a C buffer pointer back to
   a Go `*Buffer` would need a registry or an aliasing map. `Buffer.ID()` returns
   the same value, so the caller resolves names with a map of its own; the test
   builds that map from the model weights, the activation structs, the KV pool,
   and the session state.
3. **The sweep-to-step agreement is stated per matrix size, not as a flat 5%.**
   The acceptance criterion as first written assumed a hot matrix and a real
   step measure the same thing. They do not for a matrix that fits the cache:
   the sweep re-reads one matrix while it is resident, and a decode step rotates
   through 2.7 GB of weights per token. Measured, the 521 MB `lm_head` matrix
   agrees within 1% (46.5 against 46.4 GB/s) and the 6-20 MB per-layer matrices
   read 1.3-1.6x faster in the sweep. The criterion above therefore compares the
   two on the cache-exceeding matrix and records the ratio for the rest, and 2.4
   already stated the upper-bound caveat that the original wording contradicted.

Measured reproduction (two consecutive runs of `scripts/decode_profile.sh` on
the reference device):

| Quantity | Recorded in 2.3 | Run 1 | Run 2 |
| --- | ---: | ---: | ---: |
| Plan operations | 579 | 579 | 579 |
| Profiled step | 97.605 ms | 97.421 ms | 97.721 ms |
| GEMV family | 87.556 ms (89.7%) | 87.599 ms (89.9%) | 87.501 ms (89.5%) |
| Gated delta-net | 7.233 ms | 7.130 ms | 7.878 ms |
| Matmul bytes per token | 2,726 MB | 2,726.1 MB | 2,726.1 MB |
| Matmul bandwidth | 31 GB/s | 31.1 GB/s | 31.2 GB/s |

The per-cell GB/s of 2.4 reproduces within 9% on every swept cell and within
0.3% on the profiled step. The largest residual is on the sub-100 microsecond
cells, where the host timing of a submission is dominated by the launch cost;
the traced kernel time is stable and is reported next to it.

What the new measurements add to 2.4-2.6, recorded in
[benchmarks.md](benchmarks.md#decode-matvec):

- The streaming kernel that D1 and D2 target already reaches 172-199 GB/s on
  every shape the model dispatches except the 32-output `ssm_beta` cell, where
  the per-row GEMV wins 2.6x. That is the measured basis for the D1 threshold
  requirement in 4.2 and for keeping a shape rule in D2.
- The per-row GEMV executes 18-29 VALU and 0.1-2.7 LDS per weight against
  8.5-13.8 VALU and 0.16 LDS for the streaming kernel. The instruction ratio is
  about two, while the time ratio is four to six, so the remaining loss is the
  per-row block structure (one block per output row, a tree reduction, two
  barriers per K step) and the activation re-read, not the arithmetic alone.
- The batched form repeats the K loop per row and therefore costs 3.5x the
  instructions of the single-row form; decode at batch 1 cannot use it as a
  substitute for the mapping change in D2.
- Wall time minus kernel time is 20-30 microseconds per submission on the small
  cells, which is the per-operation launch cost that D5 has to remove.

## 4. Stage D1: decode output projection on the streaming kernel

### 4.1 Problem

The decode tail runs `AddGemv` on the widest matrix in the model
(`engine/gpu/plan.go:149`). At 46.4 GB/s that is 11.238 ms of an 87.556 ms GEMV
budget, 10.8% of the step, while `lm_head_kernel` already achieves 180 GB/s on
the same matrix (2.896 ms) and is already dispatched in the prefill graph.

### 4.2 Work

1. In `buildDecodePlan`, replace the unconditional `AddGemv` with the same rule
   the prefill tail uses: `hip.SelectLMHeadPath(output.dtype, 1, hidden, vocab)`
   selects `AddLMHeadAt`, otherwise keep `AddGemv`.
2. Re-measure the threshold for `rows = 1` across dtypes at the decode width so
   `lmHeadMinOutputs` stays a measured value; the existing rule was measured
   for the prefill tail and the sweeps differ by activation layout only, so
   expect no change, but the number must be re-derived rather than assumed.
3. Keep the fast path out of any shape the kernel does not cover
   (`n_in % 256 != 0`, unsupported dtype, `n_out` below the threshold); silent
   fallback stays forbidden and the per-row GEMV remains the documented
   fallback.

### 4.3 Numerics

The streaming kernel sums four K values per lane and then reduces across the
wave, so the reduction order differs from the GEMV tail. The existing prefill
checks bound this at `1.3e-4` for 1/3/16 rows
([benchmarks.md](benchmarks.md#correctness-gates)). Add the decode-side
equivalent: the same logits check as `TestPlanLMHeadMatchesDequant` for
`rows = 1` at the real width, and the recorded greedy-token comparison before
and after the change.

### 4.4 Acceptance

- Decode p50 at 128 prompt / 128 decode: 13.21 s becomes 12.2 s or less
  (10.5 tok/s or more, +9%): the projected step is 104.1 - 8.3 = 95.6 ms.
- Greedy tokens identical to the current tree for the recorded prompts; maximum
  logit difference recorded in the test.
- Prefill rows unchanged (the prefill tail already uses this kernel).
- Isolated measurement of the matrix at 180 GB/s or more, recorded in
  [benchmarks.md](benchmarks.md).

### 4.5 Implementation (2026-11-18)

`buildDecodePlan` now applies the prefill tail's rule to the output projection:
`hip.SelectLMHeadPath(output.dtype, 1, hidden, vocab)` selects `AddLMHeadAt` and
anything else keeps `AddGemv`. The plan shape does not change, so the decode
step still submits 579 operations; only the tail's kernel changes.
`Options.ForceGemvTail` keeps the per-row route available so a differential test
can compare the two on identical plan inputs, the same way
`ForceTiledAttention` does for prefill attention.

Threshold re-derivation. `build/yuuka_gemv_decode_sweep --threshold` measures
the rows = 1 crossover at the decode width, and `scripts/decode_profile.sh` runs
it as part of the decode attribution. Measured over two runs, the streaming
kernel wins from 512 outputs upward for Q4_K, Q5_K, and Q6_K and from 1536
upward for Q8_0 and F16, where the vectorized per-row GEMV is still ahead at
1024 (116.8 GB/s against 115.2 GB/s for Q8_0, 233.6 against 149.6 for F16).
`lmHeadMinOutputs` therefore stays at 1024: it is the smallest single width that
does not regress a dtype, and no output projection sits between 512 and 1536
outputs. The table is in [benchmarks.md](benchmarks.md#decode-matvec).

Numerics. `TestGPUStreamingTailMatchesGemv` runs the real model twice, once per
tail, and compares the logits of eight greedy decode steps:
4.77e-06 maximum absolute difference with identical tokens, against a 0.02
gate that is two orders of magnitude wider than the measured value. Its first
assertion is that the two plans really do run different kernels (it reads the
last operation of each decode plan through the D0 shadow), so the comparison
cannot pass vacuously. The prefill tail is unchanged, and `TestGPUMatchesCPU`
keeps its 0.00309944 first-step bound.

Measured, same session, `yuuka bench --chunk-size 512 --prompt-tokens 128
--decode-tokens 128 --warmup 2 --reps 3`:

| Quantity | Before | After | Gate |
| --- | ---: | ---: | ---: |
| Decode p50, 128 tokens | 13.241 s | 12.079 s | 12.2 s or less |
| Decode tokens/s | 9.59 | 10.51 | 10.5 or more |
| Decode p50, 256 tokens | 26.854 s | 24.531 s | - |
| Profiled step (579 ops) | 97.421 ms | 89.350 ms | - |
| Output projection | 11.231 ms (46.4 GB/s) | 2.806 ms (185.9 GB/s) | 180 GB/s or more |
| Matmul family | 87.599 ms (89.9%) | 79.197 ms (88.6%) | - |
| Prefill p50, 128 tokens | 0.085 s | 0.087 s | unchanged |
| Prefill p50, 256 tokens | 0.088 s | 0.088 s | unchanged |

The gate is met. Both columns were measured with the same flags in the same
session, the "before" one from a build of the three changed files reverted. The
tail accounts for 8.43 ms of the 9.15 ms per-token gain measured end to end
(9.11 ms at 256 tokens), and 10.51 tokens/s is +9.6% against 9.59. The remaining
0.7 ms is not attributed by these runs: the instrumented step moved by 8.07 ms
between the same two builds, so the two methods disagree on the residue by more
than the residue itself.

Two follow-ups this stage does not take, both deferred to D2:

- The per-row GEMV is still the fastest route below 1024 outputs and for the
  `ssm_beta`/`ssm_alpha` projections, which are 32 outputs wide; D2 folds those
  narrow projections into one submission.
- The 88.9% matmul share is unchanged, so the remaining gap in section 2.6 is
  still the per-row GEMV family, not this tail. D2 addresses it.

## 5. Stage D2: channel-parallel decode GEMV

### 5.1 Problem

The decode GEMV family is one output row per 256-thread block with a block-wide
tree reduction and, for Q4_K/Q5_K, two barriers per K block (2.6). This is the
single change that closes most of the 6.5x gap: the same mapping that
`lm_head_kernel` uses reaches 168-226 GB/s on decode-like widths.

### 5.2 Work

1. Add a decode GEMV family that keeps the streaming kernel's mapping but
   serves the whole model, not only the output projection:
   - one wave owns four output channels and walks the K axis; every lane loads
     16 bytes of packed weights per step, so a 32-byte sector is fully consumed
     by the lanes that read it;
   - the activation row is staged in shared memory once per block and read as
     `float4` from there instead of once per output channel from L1;
   - accumulators stay in registers, and the only reduction is one wave shuffle
     per channel, so there is no block barrier in the K loop;
   - reuse `decode_span4` from `quant.cuh` so the arithmetic per weight is
     identical to the current GEMV and the numerics change is limited to the
     summation order.
2. Cover the dtypes the decode dispatch actually selects (Q4_K, Q5_K, Q6_K,
   Q8_0, Q4_0, Q5_0, F16, F32) and the `n_in` values the model uses (2560,
   4096, 9216). A shape outside the covered set keeps the current per-row GEMV.
3. Handle the very narrow projections (`ssm_alpha`/`ssm_beta`, `n_out = 32`)
   in the same submission as the neighbouring projection, or in one launch for
   both, instead of 48 separate launches per step.
4. Update `backend/hip/dispatch.go`: the decode path selects the channel
   parallel kernel by dtype and shape; the per-row GEMV stays as the fallback
   and as the reference for the kernel test. Add the crossover to
   [kernels.md](kernels.md) section 4.
5. Keep the arithmetic contract in [kernels.md](kernels.md) updated: same
   products, different summation order, tolerance per dtype.

### 5.3 Acceptance

- Isolated decode shapes (2.4) at 150 GB/s or more for Q4_K, Q5_K, Q6_K.
- Decode at 28 tok/s or more (from 9.61), that is 2.9x: the projected step is
  2,726 MB / 150 GB/s + 7.2 ms of delta-net + 2.6 ms of other operations +
  6.5 ms of gaps = 35 ms.
- `TestGPUMatchesCPU` keeps its recorded `0.00309944` first-step maximum logit
  difference or better, with greedy tokens equal.
- `tests/hip/gemv_test.cpp` covers the new family against the per-row GEMV and
  the host dot product for every covered dtype at aligned and unaligned widths.

## 6. Stage D3: Q8_1 activations with `dp4a`

### 6.1 Problem

Stage D2 removes the structural loss but keeps the arithmetic: one nibble
extract, one convert, and two FMA per weight, plus 4 bytes of activation per
value. The reference spends about 0.3-0.5 instructions per weight because the
activation is Q8_1 and four weights retire per `dp4a`. That difference is the
step from roughly 150-180 GB/s to the 185 GB/s and above the reference reaches,
and it is the same machinery the prefill MMQ path already uses
(17-18 TFLOP/s, [kernels.md](kernels.md#63-int8-mmq-gemm)).

### 6.2 Work

1. Per token, quantize each activation vector that more than one projection
   consumes to Q8_1 once, reusing the exact quantizer contract of the prefill
   path (same `amax`, division, and `roundf`) so the two paths validate each
   other.
2. Add the int8 inner loop over the Q8_1 activation and the packed K-quant
   weights, with the block scale and min applied per 32-value sub-block as
   today. Start with Q4_K and Q5_K, which carry about 1,880 MB of the 2,726 MB
   a token reads.
3. Measure the two candidate shapes before committing to one: a separate small
   quantization kernel per activation vector, and the fused variant that
   quantizes the activation inside the matvec block (as the reference does for
   Q8_0 and Q6_K). Pick per dtype from measurement, not per preference.
4. Decide and record whether the int8 route also serves `n_out = 32`
   projections and the delta-net inputs; those are too narrow to pay for a
   separate quantization pass, so they stay FP32 unless measured otherwise.

### 6.3 Numerics

This is the first stage that changes the arithmetic, not only the summation
order. Requirements:

- A kernel-level test per dtype that compares the int8 route against the FP32
  dequantized dot product for 1, 3, and 16 rows, with an absolute tolerance
  recorded in the test.
- The prefill int8 route already moves logits by up to 0.0031 on a short prompt
  and 0.264 at 2048 tokens with top-1 preserved
  ([benchmarks.md](benchmarks.md#correctness-gates)); record the decode
  equivalent, and keep top-1 identical on the recorded prompts.
- Document the tolerance and the dispatch condition in
  [kernels.md](kernels.md). Do not widen any existing tolerance.

### 6.4 Acceptance

- Isolated decode shapes at 185 GB/s or more on Q4_K/Q5_K/Q6_K.
- Decode at 31 tok/s or more: the projected step is 14.7 ms of weights plus the
  unchanged 16.3 ms of delta-net, other operations, and gaps = 31 ms.
- Greedy tokens identical on the recorded prompts; new tolerances recorded with
  their measurement.
- No prefill regression: the two paths share the Q8_1 layout, so the MMQ
  correctness gates remain green.

## 7. Stage D4: gated delta-net per-token kernel

### 7.1 Problem

24 of 32 layers are gated delta-net. The per-token kernel
(`kernels/hip/ssm/ssm.hip:26-78`) launches one 128-thread block per head, keeps
one state column per thread, and traverses the 128x128 FP32 state five times:
decay, `S^T k`, the rank-1 update, `S^T q`, and the write-back. That is about
302 MB of state traffic per token at 42 GB/s, 7.23 ms of a 97.6 ms step, against
a 1.2 ms floor.

### 7.2 Work

1. Keep the whole 128x128 state of a head in registers or in shared memory for
   the duration of the token: 64 KB per head fits in the 64 KB LDS budget only
   with 32-bit-per-lane splitting, so measure the register-resident variant
   (one column per thread, 128 columns) against a shared-memory variant before
   choosing.
2. Fuse decay, the two `S^T` products, and the rank-1 update into one traversal
   so the state is read once and written once per token instead of five times.
3. Fuse the neighbouring elementwise work into the same launch where it does
   not change the contract: `conv1d` (0.168 ms), the decay/beta computation
   (`softplus`, `exp`, `sigmoid`, currently one lane per head), and the gated
   RMSNorm at the output.
4. Keep the state layout and the per-token result identical to what the
   chunked prefill path consumes ([kernels.md](kernels.md) section 9 owns that
   contract); a change of layout must be recorded there and validated against
   `TestGPUChunkedPrefillLongContext`.

### 7.3 Acceptance

- 24 layers at 2.0 ms or less, that is 3.6x, with the state traffic at
  150 GB/s or more in the isolated `RECURRENT` cell of the sweep.
- Decode at 37 tok/s or more: 14.7 ms of weights plus 2.0 ms of delta-net plus
  9.1 ms of other operations and gaps = 26 ms.
- Greedy tokens identical; hidden-state comparison against the CPU reference
  inside the existing bound.
- The chunked prefill recurrent path is unchanged or faster.

## 8. Stage D5: launch gaps and operator fusion

### 8.1 Problem

With D1-D3 in place the step is about 14.7 ms of weight traffic, 7.2 ms of
delta-net state traffic, and about 9 ms of everything else: 579 dependent
launches with about 6.5 ms of submission gap, plus 2.6 ms of separate RMSNorm,
L2Norm, gated RMSNorm, residual add, SwiGLU, conv1d, RoPE, and embedding
operations.

### 8.2 Work

1. Reduce the operation count rather than the launch cost: fold the activation
   quantization of D3 into the consumers, fuse RMSNorm into the GEMV prologue
   where the weight row layout allows it, fuse the SwiGLU multiply into the
   `ffn_down` prologue, and fuse RoPE and the KV write into the attention
   prologue.
2. Merge the narrow projections (`ssm_alpha`, `ssm_beta`, `ssm_a`/`ssm_dt`) into
   one launch.
3. Re-measure opt-in HIP Graph replay on the reduced plan
   ([serving.md](serving.md) owns the contract) and record whether it now pays;
   it is currently opt-in with a documented fallback.
4. Any fusion that changes a documented operator boundary updates
   [kernels.md](kernels.md) and the plan validation test
   (`tests/hip/plan_validation_test.cpp`) in the same change.

### 8.3 Acceptance

- Inter-operation time at 1.5 ms or less (from about 6.5 ms), measured as
  end-to-end minus the profiled operation sum.
- Decode at 45 tok/s or more: 14.3 ms of weights at 190 GB/s plus 2.0 ms of
  delta-net plus about 2.5 ms of fused operations and gaps = about 19 ms,
  which is 52 tok/s. The gate stays at 45 so that a partial fusion win still
  accepts the stage. Reaching the reference's 62 tok/s additionally requires
  the matvec to hold 200 GB/s or more with the non-matvec work at about 1 ms,
  which this plan does not assume.
- Cancellation, event-lifetime, and leak checks from
  [testing.md](testing.md) stay green, including `go test -race ./...`.

## 9. End-to-end acceptance

The stage gates are individually necessary; the plan is done when the whole
matrix holds on the reference device, measured under
[testing.md](testing.md)'s protocol with the metadata of 2.1:

| Test | Current | After D1 (measured) | After D2 | After D3 | After D4 | After D5 | llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tg128, 128 prompt | 9.61 | **10.51** | 28 | 31 | 37 | 45 | 62.39 |
| Step time, release | 104.1 ms | **95.1 ms** | 35 ms | 31 ms | 26 ms | 19 ms | 16.03 ms |
| Step time, profiled | 97.6 ms | **89.0 ms** | - | - | - | - | - |
| Decode p50, 128 tokens | 13.21 s | **12.08 s** | - | - | - | - | 2.05 s |
| Matvec bandwidth | 31 GB/s | **34.5** | 150 | 185 | 185 | 190 | about 170 |
| Prefill pp128 tok/s | 1359.9 | **1465** | unchanged | unchanged | unchanged | unchanged | 1766.6 |

The After-D1 column is measured (4.5); the D2-D5 columns are arithmetic from the
traffic floors of 2.8 and the gates of sections 5 to 8, not measurements, and
every cell must be replaced by a measured value before this document claims it.
The tg128 columns after D1 carry the gate value, which is deliberately below the
projection: section 8.3 projects 52 tok/s from a 19 ms step and gates at 45. The
reference column is measured.

Additional acceptance:

- `go build ./...`, `go vet ./...`, `go test ./...`, `go test -race ./...`,
  `go test -tags hip ./...`, and `ctest --test-dir build` pass.
- Greedy tokens on the recorded prompts are identical to `bfd36c0`, and every
  arithmetic change is covered by a tolerance test.
- [benchmarks.md](benchmarks.md) carries the new decode table, the per-tensor
  GB/s table, and a same-session `llama-bench` row with build and ROCm version.
- No regression larger than 2% on any prefill row of
  [benchmarks.md](benchmarks.md#prefill-prompt-length).

## 10. Rejected routes and constraints

| Route | Decision | Reason |
| --- | --- | --- |
| Raise the thread count per row or `nwarps` for batch-1 decode | Rejected as the primary fix | Measured upstream as regressive on narrow matrices because the extra warps still pay the barrier and the shared-memory reduction (llama.cpp PR 20831); the loss here is structural, not occupancy |
| Expand quantized weights to FP16 and use the WMMA path | Forbidden | [AGENTS.md](../AGENTS.md) section 5.2; it also multiplies the per-token weight traffic by 2.8-4 |
| Route single-row decode through the prefill MMQ GEMM | Not yet | At `M = 1` the MMQ tile cannot amortize its staging; keep it as a measured ablation in D3 rather than a default |
| Batch several requests to amortize the weight pass | Out of scope here | It changes the scheduler contract ([serving.md](serving.md)); it is worth revisiting after the single-request path reaches its floor |
| Copy the reference kernel implementation | Forbidden | [AGENTS.md](../AGENTS.md) section 16; reimplement against this repository's ABI, layouts, tests, and record the algorithm's origin in the commit body |
| Tune dispatch thresholds from a single session | Forbidden | Thresholds come from a sweep recorded with the measurement metadata, per [testing.md](testing.md) |

## 11. References

In-tree:

- [kernels.md](kernels.md): quantized GEMV and GEMM contracts, the streaming LM
  head, the gated delta-net contract, and the dispatch tables.
- [benchmarks.md](benchmarks.md): the current measurement record, the streaming
  LM head table, the batched GEMV crossovers, and the llama.cpp comparison.
- [testing.md](testing.md): the benchmark protocol, the profiling split, and the
  regression gates.
- [roadmap3.md](roadmap3.md): the prefill stages, the counter tooling that D0
  extends, and the per-MMA attribution method.

External, read for algorithm and measurement only:

- `llama.cpp` PR 19478 (`nwarps` for batch-1 decode), PR 20831 (the narrow-matrix
  regression of a wide `nwarps`), PR 21344 (gfx1151 MMQ/MMVQ parameter table),
  and `ggml/src/ggml-cuda/mmvq.cu` for the MMVQ mapping.
- `ggml/src/ggml-cuda/vecdotq.cuh` for the per-type `dp4a` formulation and the
  RDNA3/3.5 `__builtin_amdgcn_sudot4` lowering.
- The `strix-llama.cpp` fork commits `9b4944f0c` and `d74a20895` (fused
  activation quantization for decode matvecs) and `f9c096c42` (reduction-order
  exactness on RDNA3.5).
