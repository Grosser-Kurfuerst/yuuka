# Benchmark Report: Qwen3.5-4B Q4_K_M

This report records single-request CPU-versus-HIP measurements, the Phase 2
decode baseline, the Phase 3 prefill results, Phase 4 serving results, the
release decode-length matrix, and the Phase 5 KV-format,
speculative-decoding, MoE, and quantization measurements. It
follows the metadata requirements in [testing.md](testing.md) as far as the
current tooling allows. The early decode section is a correctness-phase
baseline, not a tuned performance result; the Phase 3 section records measured
prefill behavior and its known gap to llama.cpp.

## Environment

| Field | Value |
| --- | --- |
| Date | 2026-11-11 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Architecture | `qwen35`, 32 layers, 8 full-attention, 24 gated delta-net |
| Quantization | Q4_K_M, mixed per-tensor dtypes |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | Ryzen AI MAX+ 395, 32 threads, 120 GiB shared memory |
| Kernel | 6.17.13-21-pve |
| Sampling | Greedy, temperature 0 |
| Warmup | 1 request |
| Repetitions | 3 measured requests |
| Context capacity | CPU 88, HIP 96 |
| Prompt | 16 tokens from repeated English prose |
| Decode | CPU 8 tokens, HIP 16 tokens |

Current-host note (2026-11-13): the machine now reports kernel
`7.0.14-28-pve`; the tables below were recorded on `6.17.13-21-pve` and have
not been re-baselined after the kernel change.

The CPU and HIP runs use different decode lengths, so compare throughput and
latency, not total wall time. The prompt construction and sampling settings are
identical.

## Results

| Metric | CPU reference p50 | CPU p95 | HIP p50 | HIP p95 |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 7.260 s | 7.350 s | 1.433 s | 1.433 s |
| Prefill throughput | 2.20 tok/s | — | 11.17 tok/s | — |
| TTFT | 7.260 s | 7.350 s | 1.433 s | 1.433 s |
| Decode | 3.631 s / 8 tokens | 3.644 s | 1.519 s / 16 tokens | 1.521 s |
| Decode throughput | 1.93 tok/s | — | 9.87 tok/s | — |

Effective weight-read bandwidth during decode, estimated as model bytes times
tokens per second:

| Backend | Effective bandwidth |
| --- | ---: |
| CPU reference | 5.3 GB/s |
| HIP | 27.0 GB/s |

This is an effective estimate, not a hardware counter. The HIP GEMV path is
still far below the measured `hipMemset` rate (about 190-230 GB/s across the
recorded runs). The current kernel launches one 256-thread workgroup per output
channel and performs a block-wide reduction per quantization block. Reducing
launch and reduction overhead, increasing per-workgroup output reuse, and
vectorizing packed loads are the obvious next optimizations.

## Per-quantization decode baseline

The Q4_K_M source model was requantized with llama.cpp to Q8_0, Q6_K, and a
Q5_K/Q6_K mix. These files are test variants, not independently quality-tuned
models. Conditions: `gfx1151`, HIP backend, prompt 16 tokens, decode 16 tokens,
greedy sampling, context 128, warmup 1, repetitions 2.

| Model variant | File type | File size | Prefill p50 | Prefill tok/s | Decode p50 | Decode tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | `15` | 2,740,937,888 | 1.878 s | 8.52 | 1.731 s | 8.67 |
| Q8_0 | `7` | 4,482,403,488 | 0.807 s | 19.82 | 1.053 s | 14.24 |
| Q6_K | `18` | 3,464,055,968 | 1.528 s | 10.47 | 1.311 s | 11.44 |
| Q5_K/Q6_K mixed | `18` | 3,074,987,168 | 1.572 s | 10.18 | 1.651 s | 9.09 |

SHA-256 values: Q4_K_M `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
Q8_0 `dc3021458088a6fb0e8629c096ea7e95f743e434d96dc4351d2107c669be1fa9`,
Q6_K `438c752e79f8cfdf80483b7017ebc672f16eaf717f198c298740eb6caec097e4`,
Q5_K `68060caab98d455ed40683ab44181599d9ac528ec2378f1c10683b8b8c790637`.
The CPU/GPU agreement test passes for all four variants.

Effective weight-read bandwidth, estimated as model bytes times decode tok/s:
Q4_K_M 23.8 GB/s, Q8_0 63.8 GB/s, Q6_K 39.6 GB/s, and the Q5_K mix 28.0 GB/s.
These remain below the measured memset range of about 190-230 GB/s.

## Decode kernel attribution

`TestGPUProfileDecode` runs one ordinary decode step after a fixed prompt with
HIP events around every submitted plan operation. This is event-instrumented
attribution, not a hardware-counter profile; total time includes event overhead.

| Operation | Time | Share |
| --- | ---: | ---: |
| GEMV | 92.859 ms | 90.5% |
| GDN | 7.308 ms | 7.1% |
| RMSNorm | 0.807 ms | 0.8% |
| residual add | 0.374 ms | 0.4% |
| L2Norm | 0.258 ms | 0.3% |
| gated RMSNorm | 0.237 ms | 0.2% |
| conv1d | 0.174 ms | 0.2% |
| SwiGLU | 0.161 ms | 0.2% |
| attention | 0.139 ms | 0.1% |
| head RMSNorm | 0.096 ms | 0.1% |

The profiled step submitted 579 operations and took 102.580 ms. The GEMV path is
the clear decode bottleneck, consistent with the effective bandwidth remaining
far below device copy rates.

## Batched GEMV dispatch

`build/yuuka_gemv_sweep` compares one GEMV per batch row against the batched
GEMV kernel for `n_in = n_out = 2560`, two warmups, ten repetitions, on gfx1151.
Values are the batched speedup over per-row submission; host timing includes the
completion wait for both paths.

| Dtype | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 | Batched path from |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q4_K | 0.98 | 1.58 | 2.34 | 3.20 | 3.88 | batch 2 |
| Q5_K | 1.02 | 1.71 | 2.57 | 3.55 | 4.22 | batch 2 |
| Q6_K | 1.02 | 1.25 | 1.90 | 2.49 | 3.16 | batch 2 |
| Q8_0 | 0.61 | 0.86 | 1.26 | 1.59 | 1.71 | batch 4 |
| F16 | 0.81 | 0.94 | 1.22 | 1.37 | 1.51 | batch 4 |

The table is the measured record behind the batched-GEMV path;
`backend/hip.SelectPrefillPath` selects that path for the small row classes it
covers, and batches above 16 fall back to per-row submission. The current
single-token decode path always uses batch 1, so the batched path is a
prefill/batching building block. There is no separate runtime GEMV batch
selector.
`tests/hip/gemv_test.cpp` checks the batched kernel against the reference for
F32, F16, Q8_0, Q4_K, Q5_K, and Q6_K at batch sizes 1, 2, 3, 4, 8, and 16.

## Correctness evidence

For the prompt `The capital of France is`:

- CPU and HIP greedy tokens are identical for the first six generated tokens:
  `[11751, 13, 198, 32, 13, 2912]`.
- A direct sequential `Step` run of the decode graph gives a first-step maximum
  absolute logit difference of `4.58e-5`; `TestGPUDecodePathMatchesCPU` covers
  that path. Phase 3 routes multi-token prompts through chunked prefill; the
  same prompt in that path differs from the CPU reference by `3.65e-3` in the
  worst logit and selects the same top-1 token. Both numbers are observations
  under the current kernels; the chunked-path test enforces a loose `5.0` logit
  guard, and the decode-path test enforces `1e-3`.
- The checked-in `TestGPUMatchesCPU` compares chunked-prefill final logits and
  greedy tokens with the CPU reference and reports `3.65e-3` for the same
  prompt, not `4.58e-5`. The checked-in `TestGPULayerTraceMatchesCPU` compares
  every layer's hidden state for the same prompt; the worst recorded layer
  difference is `5.53e-5`.
- Golden CPU token tests also cover `1+1=` and `Once upon a time`.

## Phase 3 prefill

### Environment

Same model, device, ROCm/HIP, and host as the top of this document. Conditions
for the prefill measurements: Qwen3.5-4B Q4_K_M, `gfx1151`, greedy sampling,
`chunk-size` 512 except where a full prompt chunk is smaller, two warmup
iterations for the GEMM sweep and one for prompt curves unless stated
otherwise. The prompt text is the repeated-prose corpus used by `bench`.

### Prefill GEMM path comparison

`build/yuuka_prefill_sweep` compares, for one GEMM submission followed by a
completion wait:

- `row`: one plan containing `M` single-row GEMV operations (the pre-Phase-3
  prefill path);
- `batch`: the Phase 2 batched GEMV (`M <= 16`);
- `wmma`: the custom rocWMMA kernel with FP32 activations converted to FP16 in
  LDS;
- `rocblas`: the rocBLAS FP16 baseline, including the FP32-to-FP16 cast
  submitted immediately before the GEMM.

Values are milliseconds per submission, the mean of five measured iterations
after two warmups. `N` is output width and `K` is input width. Host timing
includes the completion wait.

`Q4_K`, milliseconds:

| M | 2560→9216 row | batch | wmma | 9216→2560 row | batch | wmma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.364 | 0.364 | 0.362 | 0.315 | 0.333 | 0.685 |
| 4 | 1.385 | 0.572 | 0.376 | 1.231 | 0.507 | 0.730 |
| 8 | 2.748 | 0.862 | 0.384 | 2.447 | 0.795 | 0.711 |
| 16 | 5.468 | 1.461 | 0.400 | 4.876 | 1.298 | 0.724 |
| 32 | 10.945 | — | 0.392 | 9.752 | — | 0.813 |
| 64 | 21.902 | — | 0.734 | 19.448 | — | 0.900 |
| 128 | 43.823 | — | 1.520 | 38.978 | — | 1.261 |

`Q6_K`, milliseconds:

| M | 2560→9216 row | batch | wmma | 9216→2560 row | batch | wmma |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.283 | 0.359 | 0.418 | 0.261 | 0.328 | 0.688 |
| 4 | 1.106 | 0.593 | 0.429 | 1.032 | 0.528 | 0.726 |
| 8 | 2.294 | 0.888 | 0.432 | 2.103 | 0.787 | 0.698 |
| 16 | 4.618 | 1.471 | 0.443 | 4.279 | 1.322 | 0.739 |
| 32 | 9.253 | — | 0.474 | 8.574 | — | 0.845 |
| 64 | 19.526 | — | 0.974 | 17.488 | — | 1.058 |
| 128 | 38.525 | — | 1.861 | 35.854 | — | 1.451 |

`F16`, milliseconds, including the cast for `rocblas`:

| M | 2560→9216 row | batch | wmma | blas | 9216→2560 row | batch | wmma | blas |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.295 | 0.370 | 0.606 | 0.313 | 0.286 | 0.389 | 1.149 | 0.290 |
| 4 | 1.163 | 0.577 | 0.695 | 0.930 | 1.125 | 0.839 | 0.987 | 0.573 |
| 8 | 2.310 | 0.919 | 0.705 | 0.942 | 2.261 | 1.401 | 1.640 | 0.575 |
| 16 | 4.571 | 1.606 | 0.716 | 0.977 | 4.498 | 2.162 | 1.138 | 0.582 |
| 32 | 9.158 | — | 0.831 | 1.042 | 9.052 | — | 1.500 | 0.602 |
| 64 | 18.245 | — | 1.537 | 1.476 | 18.128 | — | 1.301 | 1.079 |
| 128 | 36.424 | — | 2.751 | 1.612 | 35.975 | — | 1.581 | 2.298 |

`Q8_0`, `Q5_K`, and `Q4_K` follow the same crossover: the small-row paths are
fastest at `M = 1` (per-row GEMV for `Q8_0`, batched GEMV for `Q5_K`, and
effectively tied for `Q4_K`); WMMA wins from `M = 4` for wide projections
(`K <= N`) and from `M = 8` for tall projections (`K > N`). Rows 2560→9216 and
2560→8192 were measured; the shorter `Q8_0` table is omitted because the
selected path is identical. `SelectPrefillPath` uses batched GEMV below four
rows for every dtype; at `M = 1` the sweep shows per-row submission is faster
for `F16`, `Q8_0`, and `Q6_K` (about 20-30%), so the small-row dispatch remains
a follow-up.

The dispatch table in `backend/hip.SelectPrefillPath` encodes these crossovers.
For F16 it selects batched GEMV below `M = 8` for wide projections, rocBLAS
where the measured baseline wins, and WMMA otherwise; for quantized dtypes it
selects batched GEMV only at the measured small-row points. Differences below
5% are treated as run-to-run noise: the table selects rocBLAS only where it was
decisively faster and WMMA otherwise. The 2560→9216 `M = 64` case is a 4%
rocBLAS advantage left on WMMA by that rule; the tall `M = 4` cases select
batched GEMV because it is decisively faster than WMMA.

### Prompt-length performance

`yuuka` HIP, Qwen3.5-4B Q4_K_M, greedy sampling, chunk size 512. Prefill and
decode use the same process; context capacity is `prompt + 16 + 64`. Prompts up
to 2048 use one warmup and three measured repetitions; 8192 and 32768 use one
measured repetition without warmup because each run is long. P50 values:

| Prompt tokens | Prefill | Prefill tok/s | TTFT | Decode tok/s (16 tokens) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.409 s | 313.05 | 0.409 s | 9.70 |
| 512 | 1.635 s | 313.06 | 1.636 s | 9.16 |
| 2048 | 6.690 s | 306.14 | 6.690 s | 7.47 |
| 8192 | 37.687 s | 217.37 | 37.687 s | 4.31 |
| 32768 | 340.357 s | 96.28 | 340.357 s | 1.60 |

Prefill throughput is flat through 2048 tokens and then declines as attention
work grows with context length. Decode throughput declines for the expected
reason: the Phase 2 decode-attention kernel is linear in the cached context and
is not part of the Phase 3 prefill work. The 32768 run completed with correct
state; its long duration is a measured limitation of the scalar Flash Attention
tile, not a failure or truncation.

### llama.cpp comparison

Same GGUF (`00fe7986...`), same device, greedy prompt processing only
(`llama-bench -n 0`). `llama.cpp` used its default prompt batch 2048 and ubatch
512 with flash attention `auto`; `yuuka` used chunk size 512. The build is
`strix-llama.cpp` commit `c8912d7f5` (build 12216, version 0.6.0-dev). HIP used
ROCm/HIP 7.16 on gfx1151; Vulkan used Mesa RADV 25.0.7 on gfx1151. Each
`llama-bench` cell is the mean and standard deviation of two measured
repetitions after its default warmup.

| Prompt tokens | yuuka HIP tok/s | llama.cpp HIP tok/s | HIP ratio | llama.cpp Vulkan tok/s | Vulkan ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 313.05 | 1589.53 ± 265.12 | 5.1x | 2154.59 ± 5.92 | 6.9x |
| 512 | 313.06 | 2163.12 ± 94.78 | 6.9x | 2307.82 ± 7.93 | 7.4x |
| 2048 | 306.14 | 2171.11 ± 24.63 | 7.1x | 2235.83 ± 16.82 | 7.3x |
| 8192 | 217.37 | 2077.64 ± 10.40 | 9.6x | 2114.75 ± 22.70 | 9.7x |
| 32768 | 96.28 | 1494.85 ± 12.68 | 15.5x | 1702.18 ± 31.82 | 17.7x |

The custom path does not close the gap to llama.cpp. The gap is widest at long
context, where the scalar Flash Attention tile and its single-wave-per-query
structure dominate. At short and mid context the remaining gap is smaller. The
GEMM itself is weight-reuse limited by its 32-row M tile: a 512-row chunk
reloads each weight tile 16 times. llama.cpp's HIP and Vulkan kernels use
larger tiles, matrix-core layouts, and pipelined loads. Closing the GEMM gap
requires a larger M tile (with more accumulator fragments per wave) and
double-buffered LDS; closing the long-context gap additionally requires a
matrix-core or multi-query-tile attention kernel. The measured attribution and
ordered work are in [roadmap2.md](roadmap2.md); none of those optimizations is
claimed here.

The comparison uses different prompt tokenization only in the sense that each
runner builds its own prompt from the same token count; both use the same GGUF
tokenizer and greedy prompt processing. `llama-bench` reports prefill only, so
these ratios are not end-to-end generation comparisons.

### Prefill attribution diagnostic (2026-11-14)

This section records a diagnostic attribution of one 512-token prefill chunk.
It is event-instrumented evidence for planning, not a benchmark under the full
[testing.md](testing.md) protocol.

| Field | Value |
| --- | --- |
| Date | 2026-11-14 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Comparison model | `Qwen3.5-4B-Q8_0.gguf`, SHA-256 `dc3021458088a6fb0e8629c096ea7e95f743e434d96dc4351d2107c669be1fa9` |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host kernel | Linux 7.0.14-28-pve |
| Backend / KV | HIP, F32 KV unless stated |
| Chunk | 512 rows |
| Method | `Plan.ExecuteProfiled` with HIP events around every operation |
| Warmup | one plan execution before the measured executions |
| Repetitions | one diagnostic execution per position and KV format |

Event overhead is included. The profiled total for the position-0 chunk was
1534 ms against a 1553 ms wall time in the same run. Clocks, power, and
thermal state were not controlled. Run-to-run variation of a few percent was
observed across repeated diagnostics.

Q4_K_M, 512 rows at position 0:

| Operation | Time | Share |
| --- | ---: | ---: |
| WMMA GEMM | 909 ms | 59.3% |
| Gated delta-net chunk | 551 ms | 36.0% |
| Prefill attention | 48 ms | 3.1% |
| LM-head GEMV | 12 ms | 0.8% |
| Other operations | 14 ms | 0.9% |

The same 512-row chunk on Q8_0 measured 723 ms of GEMM, 551 ms of gated
delta-net work, and 48 ms of attention.

Attention scales with the chunk's absolute position; all other phases stay
essentially constant:

| Chunk start position | F32 KV attention | F16 KV attention |
| ---: | ---: | ---: |
| 0 | 48.7 ms | 45.3 ms |
| 2048 | 421.5 ms | 388.3 ms |
| 7680 | 2030.1 ms | 1475.1 ms |

At position 7680 attention is about 58% of the chunk with F32 KV and about 50%
with F16 KV. The F16 pool reduces attention time by about 27% at that position
and is within noise at position 0.

Chunk-size scaling, one measured execution per size:

| Chunk rows | Time | Throughput |
| ---: | ---: | ---: |
| 128 | 392.5 ms | 326.2 tok/s |
| 512 | 1505.4 ms | 340.1 tok/s |
| 1024 | 3132.6 ms | 326.9 tok/s |
| 2048 | 6812.6 ms | 300.6 tok/s |

These are single diagnostic runs. The formal Phase 3 curve above remains the
reference throughput table.

Derived limits (from GGUF tensor sizes, model dimensions, and kernel loop
counts; not hardware counters):

- The prefill matmul weights total about 2.161 GB. A 32-row M tile reads every
  weight once per group of 32 tokens, so the amortized traffic is about
  67.5 MB per token and about 34.6 GB per 512-token chunk. 909 ms of GEMM
  corresponds to roughly 38 GB/s, below the measured read bandwidth of about
  221 GB/s.
- The same chunk performs about 3.66 TFLOP of prefill matmul work; 909 ms
  corresponds to about 4.0 TFLOP/s, or about 6.8% of the documented
  59.4 TFLOP/s FP16 WMMA peak.
- The gated delta-net kernel visits each head's 128x128 FP32 state four times
  per token row, about 155 GB of state traffic for the chunk, and performs
  about 51.5 GFLOP. 551 ms corresponds to roughly 280 GB/s of state traffic
  and about 94 GFLOP/s.

The attribution shows three distinct limits: GEMM is limited by tile reuse,
unpacking, and unpipelined loads; the gated delta-net kernel is limited by
repeated state traffic and a 32-workgroup grid; long-context attention is
limited by V traffic, exposed load latency, and the lack of query tiling. The
ordered optimization work is in [roadmap2.md](roadmap2.md).

Reproduction:

```bash
# Q4_K_M, F32 KV, three positions and per-operation attribution.
YUUKA_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf \
  go test -tags hip ./engine/gpu -run TestGPUProfilePrefill -v -count=1

# Same diagnostic with F16 KV.
YUUKA_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf \
YUUKA_TEST_KV=f16 \
  go test -tags hip ./engine/gpu -run TestGPUProfilePrefill -v -count=1

# Q8_0 comparison.
YUUKA_TEST_MODEL=/path/to/Qwen3.5-4B-Q8_0.gguf \
  go test -tags hip ./engine/gpu -run TestGPUProfilePrefill -v -count=1
```

The test is opt-in because it loads the full model. It is a diagnostic path:
do not use its numbers as release benchmarks without the metadata and
repetition requirements in [testing.md](testing.md).

### Gated delta-net register state (2026-11-15)

The `gdn_chunk_splitreg_f32_kernel` path replaced the per-row global state
traversal described above. Measurement conditions match the attribution run:
`Qwen3.5-4B-Q4_K_M.gguf`, `gfx1151`, HIP, F32 KV, chunk size 512, one warmup,
one diagnostic `ExecuteProfiled` execution. The same test command as the
attribution section was used.

| Operation | Before | After | Change |
| --- | ---: | ---: | ---: |
| Recurrent chunk, 24 layers | 551 ms | 67.6 ms | 8.1x faster |
| WMMA GEMM, 248 submissions | 909 ms | 944 ms | unchanged path |
| Prefill attention, position 0 | 48.7 ms | 50.2 ms | unchanged path |
| Profiled chunk total, position 0 | 1534 ms | 1086 ms | 1.41x faster |
| Wall chunk total, position 0 | 1553 ms | 1077 ms | 1.44x faster |

The recurrent-chunk time is constant at positions 0, 2048, and 7680 (67.6,
67.7, and 67.7 ms), so the register path removed the state traffic that scaled
only with rows. This meets the A1 acceptance target of at most 150 ms and is
below the sequencing target of 50-150 ms. The provisional A1-A3 total-chunk
target of 1050 ms was not met at 1077 ms; the remaining gap is the GEMM path.

`tests/hip/prefill_test.cpp` checks the register path against its float
reference at state 128 and the fallback fused kernel at state 64, and repeats
the same execution to require bitwise-identical output and conv scratch. The
resource report for the new kernels:

| Kernel | VGPR | SGPR | LDS | Scratch | Occupancy | Spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gdn_chunk_splitreg_f32_kernel<128>` | 93 | 34 | 1,064 B | 0 | 16 waves/SIMD | 0 |
| `gdn_norm_gate_rows_f32_kernel` | 17 | 18 | 0 | 0 | 16 waves/SIMD | 0 |

An earlier one-column-per-lane variant kept all 128 state values in one thread
and compiled to 192 VGPRs with 90 spills, so it was rejected; the four-lane
column split is the accepted mapping.

### Prefill GEMM tiles (2026-11-15)

`bench/hip/prefill_sweep` now includes `M = 256` and `M = 512`. The 64x64,
two-fragment, K=64 double-buffered tile was compared with the original 32x64,
one-fragment, K=32 tile on the same binary build for the selected-shape sweep.
Q4_K, milliseconds per submission, mean of five measured iterations after two
warmups:

| Shape | 32x64 K32 | Selected path | Change |
| --- | ---: | ---: | ---: |
| 2560 → 9216, M=512 | 5.94 | 5.76 (64x64) | 3.1% faster |
| 2560 → 8192, M=512 | 5.46 | 5.07 (64x64) | 7.2% faster |
| 9216 → 2560, M=512 | 5.49 | 5.59 (32x64, unchanged) | within noise |
| 9216 → 2560, M=256 | 2.45 | 2.42 (32x64, unchanged) | within noise |

The 64x64 tile is selected only for `rows >= 64` with `K <= N`. At `M <= 32`,
and for every `K > N` shape, the original 32x64 tile remains selected because
the wider tile was slower there: for example, 2560 → 9216 at `M=32` moved from
0.383 ms to 0.622 ms, and 9216 → 2560 at `M=512` moved to 6.65 ms when the
64x64 tile was forced. A 64-row tile with a 2x2 fragment wave, a 128-row tile,
and a 64x128 tile were also measured and were slower than both selected
variants; they are not present in the kernel.

Chunk effect, same diagnostic protocol as the gated delta-net table:

| Measurement | Before | After |
| --- | ---: | ---: |
| GEMM, 248 submissions, position 0 | 944 ms | 895 ms |
| Recurrent chunk, 24 layers | 67.6 ms | 63.7 ms |
| Prefill attention, position 0 | 50.2 ms | 48.3 ms |
| Profiled chunk total, position 0 | 1086 ms | 1030 ms |
| Wall chunk total, position 0 | 1077 ms | 1025 ms |

The GEMM change is a 5% improvement on the chunk, not the 2-3x that workstream B
targeted: the 500 ms gate is not met. The isolated wide shapes followed the same
4.6-4.7 TFLOP/s figure as the sweep's rocBLAS column, which suggested a
hardware ceiling. The 2026-11-16 attribution shows that figure is the fp32
compute path of both `launch_gemm_blas_f16` and the earlier WMMA tile, not the
device: hipBLAS with fp16 compute and fp16 output reaches 20-33 TFLOP/s on the
same shapes. See
[prefill gap attribution](benchmarks.md#prefill-gap-attribution-vs-llamacpp-2026-11-16).
The CTA-order experiment that mapped consecutive blocks to M tiles was rejected:
it improved the tall-K isolated shape but raised the wide shape, and the chunk
GEMM total regressed to 947 ms.

Resource report for the two selected variants:

| Kernel | VGPR | SGPR | LDS | Scratch | Occupancy | Spills |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_wmma_kernel<Q4_K,64,2>` | 72 | 29 | 32,768 B | 0 | 16 waves/SIMD | 0 |
| `gemm_wmma_k32_kernel<Q4_K>` | 48 | 30 | 14,336 B | 0 | 16 waves/SIMD | 0 |

### Prefill GEMM span decode (2026-11-16)

The weight loader in both rocWMMA kernels decodes one contiguous span per
thread instead of one element, and the 64x64 tile now stages a 32-wide K step,
16 KiB of shared memory instead of 32 KiB. Environment: `gfx1151`, the pinned
ROCm/HIP toolchain, `Qwen3.5-4B-Q4_K_M.gguf` for the end-to-end number and
in-process random weights for the isolated GEMM, FP32 activations, one stream,
no concurrent work.

Two isolation protocols were run on the same binary pair (the
`libyuuka_hip.so` built before and after the change, selected with
`LD_PRELOAD`):

- `bench/hip/prefill_sweep`: two warmups and five measured submissions per
  shape, exactly as committed; each binary was run three times and the per-shape
  median is reported.
- Focused A/B: five warmups and 40 measured submissions per shape (200 for the
  small-batch checks), two passes.

`bench/hip/prefill_sweep`, milliseconds per submission, `M = 512`:

| Shape | Q4_K | Q5_K | Q8_0 | Q6_K | F16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2560 → 9216 | 5.60 → 2.78 | 6.26 → 2.89 | 5.46 → 2.81 | 6.23 → 4.78 | 6.86 → 5.24 |
| 2560 → 8192 | 4.93 → 2.45 | 5.47 → 2.69 | 4.63 → 2.39 | 5.45 → 4.23 | 6.08 → 5.24 |
| 9216 → 2560 | 5.58 → 2.91 | 6.23 → 3.16 | 4.26 → 2.91 | 6.25 → 6.05 | 6.13 → 7.28 |

Q4_K improves 48-50%, Q5_K 49-54%, Q8_0 32-49%, Q6_K 3-23%, and F16 14% on both
wide shapes. The sweep's five-submission window is too short for the 0.4-1 ms
shapes and for the 7 ms F16 tall-K shape, so four shapes it reports as
regressions were re-measured with 200 submissions after ten warmups:

| Shape | Baseline | New | Change |
| --- | ---: | ---: | ---: |
| Q6_K 9216 → 2560, M=16 | 0.718 | 0.702 | 2.2% faster |
| Q6_K 9216 → 2560, M=32 | 0.812 | 0.807 | 0.6% faster |
| Q6_K 9216 → 2560, M=512 | 6.348 | 5.795 | 9.7% faster |
| Q6_K 2560 → 9216, M=16 | 0.438 | 0.422 | 3.7% faster |
| F16 8192 → 2560, M=8 | 0.840 | 0.782 | 7.4% faster |
| F16 9216 → 2560, M=16 | 1.244 | 1.107 | 11.0% faster |
| F16 9216 → 2560, M=512 | 7.472 | 7.438 | 0.5% faster |

The sweep under-reports the F16 tall-K baseline (6.13 ms against the 7.47 ms
steady-state value), which is what produced the apparent 19% regression there.
The remaining sweep variance at small `M` is clock-state noise: the same kernel
binary reproduces within 2% under the longer protocol. F32, F16, and Q6_K run
the element mapping in both binaries, so their kernels are unchanged apart from
the shared helper signature.

End-to-end, same model, backend, KV format, chunk size, and repetition count as
the 2026-11-15 tables, `--prompt-tokens 512 --decode-tokens 1 --warmup 1
--reps 3 --chunk-size 512`:

| Measurement | 2026-11-15 build | 2026-11-16 build |
| --- | ---: | ---: |
| Prefill p50 | 0.940 s | 0.581 s |
| Prefill rate p50 | 544.3 tok/s | 880.8 tok/s |
| TTFT p50 | 0.940 s | 0.581 s |

That is 38% lower TTFT and 62% higher prefill throughput, above the 10% gate and
above the 750 tok/s stretch target. Correctness is unchanged: `ctest` passes
9/9, and `TestGPUChunkedPrefillLongContext` keeps top-1 token 33075 with a zero
logit difference against the CPU reference at chunk sizes 128, 512, and 1024.

The sweep was not re-tuned for this change. Because its five-submission window
cannot separate a 2% kernel change from clock-state noise at small shapes,
dispatch decisions that depend on a few percent should use a longer protocol
like the focused A/B above.

### Prefill gap attribution vs llama.cpp (2026-11-16)

Same machine, same GGUF, both backends on the GPU: llama.cpp `b1b770d90` (build
12359) run as `llama-bench -p 512 -n 1 -r 3 -ngl 99` reports `pp512 = 2216.25
± 39.88 t/s`; `yuuka bench --prompt-tokens 512 --decode-tokens 1 --warmup 1
--reps 3 --chunk-size 512` reports 881 t/s (0.581 s against 0.233 s). This
section attributes the 2.5x gap.

Per-operation split. yuuka: `TestGPUProfilePrefill` HIP events around every plan
operation, 512 rows, F32 KV, position 0. llama.cpp: `rocprofv3
--kernel-trace` over one `pp512` run, aggregated by kernel name. Both in
milliseconds per 512-token prefill:

| Component | yuuka | llama.cpp | Ratio |
| --- | ---: | ---: | ---: |
| Matmul (incl. LM head, activation quantize, Q6_K dequant, F16 cast) | 476.0 | 177.9 | 2.7x |
| Gated delta-net chunk | 67.4 | 13.0 | 5.2x |
| Prefill attention (incl. llama.cpp KV concat) | 18.8 | 17.6 | 1.07x |
| Norm, RoPE, elementwise, copy, KV write | 6.7 | 24.5 | 0.3x |
| Total | 568.9 | 233.0 | 2.44x |

The matmul gap is 298 ms of the 336 ms difference and the delta-net gap is 54
ms, partly offset by yuuka's 18 ms advantage on the elementwise kernels:
llama.cpp pays a `concat_non_cont` KV copy (12.1 ms), a per-matmul activation
quantize (`quantize_mmq_q8_1`, 6.5 ms), and a gate kernel for the SiLU product
(9.7 ms).

**llama.cpp mixes two matmul paths at 512 tokens.** `ggml_cuda_should_use_mmq`
keeps Q4_K/Q5_K on the int8 `mul_mat_q` (MMQ) kernel but sends Q6_K to
dequantize + hipBLAS when the batch exceeds 256 rows. In the trace that is
`mul_mat_q<(ggml_type)12>` 205.1 ms and `<(ggml_type)13>` 71.7 ms over two runs,
plus `dequantize_block_q6_K` 3.9 ms and the `Cijk_Alik_Bljk_HB_MT128x128x32`
hipBLAS kernel 19.6 ms per run. The MMQ kernel is configured with I=128 weight
rows, J=128 token columns, 256 threads, occupancy 2, 240 VGPR, K step 256, and
Q8_1 shared staging: weights are dequantized to int8 once per 128-token pass and
the activation is quantized once per matmul, so per 4096-MAC WMMA-equivalent the
kernel issues about 22 instructions against yuuka's 114.

**yuuka's kernel is decode and latency bound, not MMA bound.** Counters for the
64x64/K32 Q4_K kernel on `2560 -> 9216, M=512` (`rocprofv3 --pmc`,
2,949,120 WMMA instructions per launch):

| Counter | Per WMMA |
| --- | ---: |
| `SQ_INSTS_ALL` | 114.0 |
| `SQ_INSTS_VALU` | 54.0 |
| `SQ_INSTS_LDS` | 9.9 |
| `SQ_WAVE_CYCLES` | 2,922.6 |

That launch moves 336.1 M instructions, 0.028 per MAC, for 8.5 TFLOP/s; issue
occupancy is about 39% of the 160 SIMD issue slots. The same math in llama.cpp
costs 0.0054 instructions per MAC. Three structural causes, in order of
measured weight:

1. Token-tile reuse: yuuka decodes each weight for 64 accumulation rows and
   re-reads every weight tile 8 times for 512 tokens; llama.cpp's J=128 halves
   both.
2. Decode cost per weight: yuuka converts to FP16 and stores two bytes per value
   into shared memory (9.9 LDS instructions per WMMA); llama.cpp stores one byte
   as int8 and applies the block scale in the dot product.
3. Synchronization granularity: yuuka synchronizes twice per 32 K values,
   llama.cpp twice per 256.

**The 4.6-4.7 TFLOP/s "ceiling" reported in the 2026-11-15 sections is a
rocBLAS wrapper artifact, not hardware.** `launch_gemm_blas_f16` passes
`rocblas_datatype_f32_r` as the compute type and `f32` output, which makes
rocBLAS select fp32-compute Tensile kernels. The same GEMM with fp16 compute
and fp16 output, which is what `ggml-cuda` requests, on the two prefill shape
classes:

| Shape (F16 in) | f32 compute, f32 out (yuuka) | f16 compute, f16 out (llama.cpp) |
| --- | ---: | ---: |
| 2560 -> 9216, M=512 | 5.14 ms, 4.70 TFLOP/s | 0.72 ms, 33.6 TFLOP/s |
| 9216 -> 2560, M=512 | 7.29 ms, 3.31 TFLOP/s | 1.16 ms, 20.6 TFLOP/s |

So gfx1151 reaches 20-33 TFLOP/s through the library on F16, while yuuka's own
kernels reach 8.5 TFLOP/s on the wide Q4_K shape and 4.0 TFLOP/s on the Q6_K
tall-K shape the model actually uses (`9216 -> 2560`, 6.05 ms against 1.16 ms
for hipBLAS f16). Two implementation paths follow and are not yet implemented:

- Fix `launch_gemm_blas_f16` to request fp16 compute and fp16 output with an
  explicit cast, and route Q6_K through dequantize + hipBLAS as llama.cpp does.
  Q6_K is about 100 ms of the 476 ms matmul total, so this is an estimated
  70-80 ms of the chunk, roughly +12-15% prefill throughput.
- Rebuild the custom kernel to the MMQ geometry for Q4_K/Q5_K: 128-token tile,
  int8 shared staging, 256-wide K step. A dequantize-to-F16 route is not
  obviously better for these types because the expanded weights are 2x larger
  and are re-read once per 128-token pass, which makes that route memory limited
  at 512 tokens (about 120 ms of traffic against about 110 ms of compute).

The delta-net gap (67 against 13 ms) is independent of both.

Measurement notes: the per-op split adds HIP event overhead and does not
control clocks, so it is diagnostic under [testing.md](testing.md), not a
release benchmark. The counter run uses the same shape in isolation with two
warmups and three measured launches.

### Prefill matmul route experiments (2026-11-16)

The attribution above leaves 476 ms of a 569 ms 512-token prefill in quantized
matmul, and llama.cpp runs the same math in 178 ms. This section records the
route that was adopted, the two variants it beat, and the one tile experiment
that was rejected. Conditions: same machine, GGUF, and CLI as [the
attribution](#prefill-gap-attribution-vs-llamacpp-2026-11-16); gfx1151,
Q4_K_M, F32 KV.

#### Numeric gate before implementation

A throwaway Go program decoded the first eight rows of three real weight
matrices with `tensor.DequantizeRow`, built one deterministic activation row
(`x[i] = sin(0.37 i) cos(0.11 i)`), and compared three dot products against a
float64 reference: an FP32 accumulator, a simulated FP16 accumulator that
rounds the running sum after every product, and one that rounds every 16
products, which is the MMA shape. The entry is the max over the eight rows of
`|acc - ref| / (1 + |ref|)`.

| Tensor | Dtype | `K` | `N` | FP16, per product | FP16, per 16 | FP32 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `blk.0.ffn_up.weight` | Q4_K | 2560 | 9216 | 1.94e-3 | 4.20e-4 | 1.5e-7 |
| `blk.0.ffn_down.weight` | Q6_K | 9216 | 2560 | 4.98e-3 | 4.07e-3 | 8.8e-7 |
| `blk.0.attn_qkv.weight` | Q5_K | 2560 | 8192 | 2.13e-3 | 7.20e-4 | 4.4e-7 |

The per-matmul gate that was set from this table, 5e-3 on the FP16
accumulator, turned out to be too loose for the network: see the chunk-gate
table below.

#### Adopted: FP32-compute dequant + rocBLAS

rocBLAS rejects FP16 compute with an FP32 destination on this build
(`rocblas_status` 2) and rejects nothing for FP32 compute with an FP16
destination, which is the combination that matters here. Same shape
(`2560 -> 9216`, `M = 512`), one warmup call then three timed repeats:

| rocBLAS types | Time | Rate | Adopted |
| --- | ---: | ---: | --- |
| FP16 compute, FP16 out | 0.672 ms | 35.9 TFLOP/s | No: 0.305 logits of drift |
| FP32 compute, FP16 out | 1.098 ms | 22.0 TFLOP/s | Yes |
| FP32 compute, FP32 out (the 2026-11-15 wrapper) | 5.107 ms | 4.7 TFLOP/s | No: slower than the WMMA kernel |

FP32 compute keeps the FP32 accumulator and rounds each output once into an
FP16 scratch, which a cast kernel converts to the FP32 result. The dequantized
tile matches an independent format decoder within one FP16 unit (max relative
error 2.4e-4 over every element of a 2560x256 Q4_K, Q5_K, and Q6_K matrix), so
the route's only numeric difference from the WMMA path is that single rounding
of the result.

Isolated GEMM, milliseconds, 5 repeats, median. `dequant+rocBLAS` includes the
weight expansion and the activation cast on every call:

| Dtype | Shape | `M` | WMMA 64x64 | dequant + rocBLAS | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_K | 2560 -> 9216 | 128 | 0.716 | 0.733 | 0.98x |
| Q4_K | 2560 -> 9216 | 256 | 1.339 | 1.077 | 1.24x |
| Q4_K | 2560 -> 9216 | 512 | 2.767 | 1.607 | 1.72x |
| Q4_K | 2560 -> 8192 | 512 | 2.440 | 1.479 | 1.65x |
| Q4_K | 9216 -> 2560 | 128 | 0.606 | 1.313 | 0.46x |
| Q4_K | 9216 -> 2560 | 512 | 2.865 | 1.619 | 1.77x |
| Q5_K | 2560 -> 9216 | 512 | 2.857 | 1.566 | 1.82x |
| Q5_K | 2560 -> 8192 | 512 | 2.529 | 1.441 | 1.76x |
| Q5_K | 9216 -> 2560 | 512 | 3.113 | 1.611 | 1.93x |
| Q6_K | 2560 -> 9216 | 128 | 1.172 | 1.053 | 1.11x |
| Q6_K | 2560 -> 9216 | 512 | 4.567 | 1.882 | 2.43x |
| Q6_K | 2560 -> 8192 | 512 | 4.018 | 1.587 | 2.53x |
| Q6_K | 9216 -> 2560 | 512 | 5.758 | 1.854 | 3.11x |

Below 128 rows the expansion dominates (0.31-0.64x at `M = 32` and `M = 64`),
and from 128 rows up the route is neutral (0.95-1.16x) and then faster.

End to end, `bench --prompt-tokens N --chunk-size 512 --warmup 2 --reps 5`,
sequential runs on the same machine:

| Prompt tokens | WMMA | Adopted route | Delta |
| ---: | ---: | ---: | ---: |
| 32 | 0.081 s / 393 tok/s | 0.082 s / 391 tok/s | -0.3% |
| 100 | 0.139 s / 717 tok/s | 0.139 s / 717 tok/s | 0 |
| 128 | 0.149 s / 858 tok/s | 0.157 s / 818 tok/s | -5% |
| 512 | 0.583 s / 878 tok/s | 0.463 s / 1107 tok/s | **+26%** |

The 128-token row is slower because the fused gate/up kernel is only selected
on the WMMA path: at 128 rows the GEMMs are neutral and the route pays for two
separate GEMMs plus `SILU_MUL` instead of one fused epilogue.

#### The route is bounded by the chunk-invariance gate, not by the per-GEMM gate

`TestGPUChunkedPrefillLongContext` requires the chunk-512 prefill to stay within
0.05 logits of the chunk-128 prefill. It has no slack: the WMMA path is bitwise
identical across chunk boundaries (0). Measured drift for the route variants,
each with the same top-1 token (33075):

| Route selection | Max logit diff | Gate |
| --- | ---: | ---: |
| WMMA everywhere (baseline) | 0 | 0.05 |
| FP16 compute, every matmul, `M >= 256` | 0.305 | 0.05 |
| FP32 compute, every matmul, `M >= 256` | 0.0686 | 0.05 |
| FP32 compute, `K <= N`, `M >= 256` | 0.0577 | 0.05 |
| FP32 compute, `K <= N`, `M >= 256`, `N <= 8192` | 0.0846 | 0.05 |
| FP32 compute, `K <= N`, `M >= 128` (adopted) | **0.0419** | 0.05 |

Two effects explain the shape of the table. First, a route that depends on the
row count makes the chunk size change the numerics, and that difference
(0.058) is larger than the route's own batch-size sensitivity (0.042), so
the adopted rule keeps the row threshold at the smallest chunk size the gate
uses. Second, the metric is a maximum over 248,320 logits, so it is an
extreme-value statistic rather than a norm: the `N <= 8192` subset lands on a
worse extreme than the full wide set even though it routes fewer matmuls.

The tall-`K` class is excluded for a numeric reason, not a speed one: it
measured 1.44-1.77x faster on `9216 -> 2560` at `M = 256` and above, but
rocBLAS splits `K` there and the FP16 destination forces the split reduction
through FP16. The native route test measures 1.1-1.5e-3 relative error on those
shapes against 4.0-4.9e-4 on the wide ones, so tall-`K` keeps the WMMA path.

Residual error of the adopted route, from
`tests/hip/prefill_test.cpp`: 4.0-4.9e-4 relative per GEMM on the wide shapes,
which is one FP16 unit roundoff of the result, and 1.1-1.5e-3 on the tall shapes
the engine does not route. The test bound is 2e-3 per element.

Known limitation: a prefill chunk below 128 rows (a prompt shorter than that, or
an explicitly small `--chunk-size`) selects WMMA while larger chunks select the
route, so those two produce logits that differ by about the 0.058 measured
above. The gate compares 128 and 512, which both use the route.

#### Prompt-length scaling

The route's saving is a fixed amount of matmul work per 512-row chunk, so its
relative value falls as the prompt grows and attention takes over. Same machine,
`--chunk-size 512`, sequential runs, `--warmup 1 --reps 2` up to 8192 and
`--warmup 1 --reps 1` above it:

| Prompt tokens | WMMA | Route | Delta | Saving per chunk |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.575 s / 890 tok/s | 0.475 s / 1077 tok/s | +21% | 0.100 s |
| 2048 | 2.513 s / 815 tok/s | 2.096 s / 977 tok/s | +20% | 0.104 s |
| 8192 | 13.795 s / 594 tok/s | 12.818 s / 639 tok/s | +7.6% | 0.061 s |
| 16384 | 40.105 s / 409 tok/s | 37.302 s / 439 tok/s | +7.5% | 0.088 s |
| 32768 | 129.245 s / 254 tok/s | 123.898 s / 264 tok/s | +4.3% | 0.084 s |

The absolute saving is 0.08-0.10 s per 512-token chunk at every length, so the
gain is `saving / total` and the total is dominated by attention from about 8k
tokens on. The 8192 row measures a lower per-chunk saving than its neighbours;
two repetitions cannot separate that from run-to-run noise, and the surrounding
rows bracket it.

The saving is bounded by the traffic the expansion costs. The routed class is
120 tensors, 1.431 GB packed and 2432.7 M elements, so per chunk the route reads
1.431 GB of packed weights, writes 4.865 GB of FP16 scratch, and the GEMM reads
that scratch back: 9.73 GB more DRAM traffic than the WMMA path, which reads the
1.431 GB once. At the 190 GB/s the hardware probe measures for a streaming
kernel, that is 51 ms per chunk against a 0.13-0.15 s compute saving, which is
why the route is 1.2x on the matmul and not the 2.5x the isolated GEMM shows. A kernel that keeps the weights packed
(the int8 MMQ geometry) removes that term entirely, which is the strongest
argument for it beyond the instruction count.

Above 16k tokens the matmul is no longer where the time goes: 64 chunks of the
route save 5.3 s of a 129 s prefill, so attention and the gated delta-net are the
levers there.

#### Rejected: 128-row WMMA tile

The attribution ranks token-tile reuse first: the 64-row tile re-reads every
weight tile eight times for 512 tokens, so a 128-row tile (`BlockM = 128`,
four accumulator fragments per wave, 32-wide K step) halves that. The new tile
stages 24 KiB of A/B double buffer (`2 * (128 + 64) * 32 * 2`), which leaves two
CTAs per LDS CU instead of four, and its 128x64 FP32 result does not fit the
staging alias, so the epilogue writes two 128x32 column halves with a barrier
between them.

Isolated GEMM, milliseconds, median of three sweep runs, against the committed
64-row tile:

| Dtype | Shape | `M` | WMMA 64x64 | WMMA 128x64 | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4_K | 2560 -> 9216 | 128 | 0.687 | 0.779 | -13.4% |
| Q4_K | 2560 -> 9216 | 256 | 1.361 | 1.186 | +12.9% |
| Q4_K | 2560 -> 9216 | 512 | 2.755 | 2.519 | +8.6% |
| Q4_K | 2560 -> 8192 | 128 | 0.670 | 0.769 | -14.8% |
| Q4_K | 2560 -> 8192 | 512 | 2.499 | 2.153 | +13.8% |
| Q4_K | 9216 -> 2560 | 256 | 1.263 | 1.419 | -12.4% |
| Q4_K | 9216 -> 2560 | 512 | 2.791 | 3.030 | -8.6% |
| Q5_K | 2560 -> 9216 | 256 | 1.485 | 1.215 | +18.2% |
| Q5_K | 2560 -> 9216 | 512 | 3.033 | 2.621 | +13.6% |
| Q5_K | 9216 -> 2560 | 512 | 3.266 | 3.281 | -0.5% |
| Q6_K | 2560 -> 9216 | 256 | 2.290 | 1.700 | +25.8% |
| Q6_K | 2560 -> 9216 | 512 | 4.658 | 3.727 | +20.0% |
| Q6_K | 9216 -> 2560 | 512 | 6.119 | 6.000 | +1.9% |

The tile wins on wide, shallow-`K` shapes (+6% to +26%) and loses on tall-`K`
shapes (-0.5% to -12%) and at 128 rows (-3% to -15%), where halving the CTA
count costs more than the weight reuse saves.

End to end with the tile selected for every `K <= N` shape at `M >= 128`:
0.553 s / 926 tok/s. Restricted to `M >= 256`, `K <= 2560`, and `N >= 8192`:
0.572 s / 895 tok/s. WMMA baseline over three consecutive runs: 0.559 s,
0.563 s, 0.589 s (916, 909, 869 tok/s). Both variants sit inside that 2.4%
run-to-run band while the sweep shows regressions outside it, so the tile is not
selected, and the two-half epilogue was reverted.

**What the tile experiment rules out.** Weight traffic is not the prefill matmul
bottleneck: halving it buys 6-26% on the wide shapes and nothing on the tall
ones. The route that was adopted instead removes work per weight rather than
weights per work: the library kernel converts each packed weight to FP16 once
per call and then runs a 128-column-token tile in registers, which is why it
reaches 22 TFLOP/s where the custom 64-row kernel reaches 8.6.

Reproduction: `./build/yuuka_prefill_sweep` prints the isolated columns, and the
end-to-end numbers are the `bench` commands above. The numeric gate is a scratch
Go module that replaces `github.com/yuuka-ai/yuuka` and uses `gguf.Open` plus
`tensor.DequantizeRow`.

### Prefill attention tiling (2026-11-15)

The prefill attention kernels now process 16 query rows per workgroup with 512
threads, stage K and V together in 16-key blocks, and pad the per-lane shared
stride by one float to remove the 8-way bank conflict in the lane-parallel dot
product and value accumulation. Measurement conditions match the attribution
run; the F32 KV diagnostic uses the same `ExecuteProfiled` command.

| Chunk start position | F32 KV before | F32 KV after | F16 KV after |
| ---: | ---: | ---: | ---: |
| 0 | 48.7 ms | 17.8 ms | 18.5 ms |
| 2048 | 421.5 ms | 159.4 ms | 174.3 ms |
| 7680 | 2030.1 ms | 643.1 ms | 657.5 ms |

The F32 position-7680 acceptance target of at most 800 ms is met; the F16
target of at most 600 ms is not. F16 KV is slightly slower at every measured
position after V staging because the staged tile is FP32 and the F16 load path
adds conversion work without reducing the now-cached shared traffic. F32 KV
therefore remains the default. The lane-padded shared layout is responsible for
most of the gain; an intermediate version that staged K and V but kept the
unpadded layout measured 1484 ms at position 7680.

End-to-end prompt curves, `yuuka` HIP, Q4_K_M, greedy, chunk size 512, one
measured repetition without warmup. These are the workstream C numbers before
the FFN gate/up fusion; the final numbers are in the next section:

| Prompt tokens | Before | After | Throughput after |
| ---: | ---: | ---: | ---: |
| 512 | 1.635 s | 0.992 s | 515.9 tok/s |
| 2048 | 6.690 s | 4.265 s | 480.2 tok/s |
| 8192 | 37.687 s | 21.394 s | 382.9 tok/s |
| 32768 | 340.357 s | 156.868 s | 208.9 tok/s |

The 32768-token run completes with unchanged greedy processing; the sequencing
target of at least 250 tok/s at that length is not met, but the prefill time is
2.17x shorter than the recorded Phase 3 run.

The 512-token prefill rate improved from 313.06 to 515.9 tok/s. The
position-0 chunk profile dropped from 1086 ms to 1001 ms profiled, and from
1077 ms to 994 ms wall, with GEMM still at 895 ms of the total.

At position 7680 the chunk profile is 1639 ms, with attention at 643 ms (39%),
GEMM at 905 ms (55%), and the recurrent chunk at 66 ms. Resource report:

| Kernel | VGPR | SGPR | Static LDS | Dynamic LDS | Spills |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention_prefill_f32_kernel` | 49 | 48 | 0 | 36,864 B at head_dim 256 | 0 |
| `attention_prefill_paged_f32_kernel<dense>` | 51 | 63 | 0 | 36,864 B at head_dim 256 | 0 |
| `attention_prefill_paged_f32_kernel<gated>` | 53 | 61 | 0 | 36,864 B at head_dim 256 | 0 |

### Fused FFN gate/up (2026-11-15)

The prefill FFN now submits one fused gate/up GEMM when both projections select
the custom WMMA path. The kernel stages one activation tile and both weight
tiles, accumulates both matrices, and applies SwiGLU in the epilogue, so no gate
or up intermediate is written and the separate `SILU_MUL` pass is removed. The
fused kernel uses a 32-wide K step because staging three tiles at the 64-wide
step raised shared memory to 48 KiB and measured 528 ms for the 32 fused ops;
the 32-wide step uses 24 KiB and measures 308 ms.

Position-0 chunk, same diagnostic protocol:

| Operation | Before fusion | After fusion |
| --- | ---: | ---: |
| Remaining GEMM submissions | 248 / 895 ms | 184 / 534 ms |
| Fused gate/up submissions | — | 32 / 308 ms |
| Separate `SILU_MUL` | 32 / 6.3 ms | removed |
| Recurrent chunk | 63.7 ms | 64.0 ms |
| Prefill attention | 17.8 ms | 18.1 ms |
| Profiled chunk total | 1001 ms | 942 ms |
| Wall chunk total | 994 ms | 938 ms |

The fused path replaces 64 GEMM submissions and 32 elementwise submissions with
32 fused submissions. Its combined FFN gate/up time is 308 ms against 379 ms
for the removed GEMM submissions, a 19% reduction for that pair and 5.6% for
the whole chunk. Greedy output is unchanged:
`TestGPUChunkedPrefillLongContext` still reports bitwise identical logits at
chunk sizes 128, 512, and 1024 with top-1 token 33075, and the native fused test
passes F32/F16/Q4_K/Q6_K at 33 and 70 rows.

Resource report: `gemm_wmma_swiglu_kernel<Q4_K,64,2>` uses 93 VGPRs, 38 SGPRs,
24,576 B LDS, no spills, and 16 waves/SIMD; the 32-row variant uses the same
VGPR count and 20,480 B LDS.

End-to-end prefill with the fused path, one measured repetition without warmup:

| Prompt tokens | Before all workstreams | After fusion |
| ---: | ---: | ---: |
| 512 | 1.635 s / 313.06 tok/s | 0.931 s / 549.8 tok/s |
| 2048 | 6.690 s / 306.14 tok/s | 4.008 s / 511.0 tok/s |
| 8192 | 37.687 s / 217.37 tok/s | 19.854 s / 412.6 tok/s |
| 32768 | 340.357 s / 96.28 tok/s | 156.562 s / 209.3 tok/s |

The 32768 result is essentially unchanged from the pre-fusion C measurement
(208.9 tok/s): at that length attention dominates and the FFN pair is a smaller
share of the chunk.

### Prefill gap closure: Q6_K span loader, parallel convolution, grouped attention (2026-11-16)

Same machine, GGUF, and protocol as the sections above: `yuuka bench --backend
hip`, `--chunk-size 512`, sequential runs, `--warmup 2 --reps 5` at 512 tokens
and `--warmup 1 --reps 2` at 8192. llama.cpp `b1b770d90` (build 12359) reports
`pp512 = 2174.81 ± 35.77 t/s` and `pp8192 = 1974.74 ± 18.98 t/s`
(`llama-bench -m <gguf> -p 512,8192 -n 0 -r 3 -ngl 99`).

| Change | 512 p50 | 8192 p50 | 8192 tok/s |
| --- | ---: | ---: | ---: |
| Before | 0.472 s / 1085 tok/s | 12.688 s | 646 |
| Q6_K span loader | 0.444 s / 1154 tok/s | — | — |
| Parallel chunk convolution | 0.423 s / 1209 tok/s | 11.694 s | 701 |
| Grouped prefill attention | 0.423 s / 1209 tok/s | 11.056 s | 741 |

The 512-token rows are 49 ms faster than the starting point, 11% more
throughput, and the 8192-token row is 15% faster. Neither row reaches
llama.cpp's 2175/1975 tok/s.

Per-operation split from `TestGPUProfilePrefill` (512 rows, F32 KV, HIP events
around every plan operation), before against after:

| Component | Position 0 before | Position 0 after | Position 7680 before | Position 7680 after |
| --- | ---: | ---: | ---: | ---: |
| Matmul (`gemm`) | 373.3 ms (248) | 293.4 ms (248) | 372.6 ms | 284.9 ms |
| Gated delta-net chunk | 66.3 ms | 50.2 ms | 66.7 ms | 50.2 ms |
| Prefill attention | 18.0 ms | 19.5 ms | 614.8 ms | 561.7 ms |
| Q6_K dequant (`dequant_f16`) | 36.0 ms (179) | 26.1 ms (120) | 35.8 ms | 26.2 ms |
| F16 cast (`cast_f16`) | 11.7 ms | 11.5 ms | 11.5 ms | 11.5 ms |
| Total (profiled) | 480.1 ms | 421.9 ms | 1025.1 ms | 952.2 ms |

The matmul total falls even though the Q6_K kernel itself is faster, because 59
of the 179 K-quant calls now take the measured-fastest path for their shape: the
engine re-measures the isolated per-shape timings, and the improved WMMA Q6_K
kernel wins shapes that the dequant route used to hold.

Isolated Q6_K cells from `yuuka_prefill_sweep` (`wmma` column, same binary pair):

| Shape | Before | After |
| --- | ---: | ---: |
| `M=512 N=2560 K=9216` | 6.115 ms | 3.589 ms |
| `M=512 N=9216 K=2560` | 4.882 ms | 3.175 ms |

Decoder gates for this round, and their outcome:

| Gate | Target | Measured | Result |
| --- | --- | --- | --- |
| Tall K-quant dequant route | max logit diff <= 0.045, top-1 33075 | Q4_K 0.047 (FFN down only), 0.061 (whole tall class); Q5_K 0.079 | rejected, route removed |
| Q6_K loader | isolated tall cell <= 4.5 ms | 3.589 ms | met, (-41%) |
| Gated delta-net chunk | `recurrent_chunk` <= 40 ms | 50.2 ms | not met; convolution 26.4 -> 8.7 ms kept, recurrence restored |
| Grouped attention | position-7680 attention <= 450 ms | 561.7 ms | not met, (-9%); kernel kept because staging is only 70 ms of the old 615 ms |
| 512-token prefill | <= 0.340 s, else >= 35 ms below 0.480 s | 0.423 s | met on the fallback gate (-49 ms) |
| 8192-token prefill | >= 750 tok/s | 741 tok/s | not met (-1.2%) |
| Greedy tokens and chunk invariance | same top-1, max logit diff <= 0.045 | top-1 33075, 0.0419 (baseline 0.042) | met |

Two rewrite attempts were measured and rejected on correctness, not speed. An
eight-lane split of the recurrent column (16 state rows per lane, 256
workgroups) reduces `recurrent_chunk` to 27.0 ms, but the reduction reassociates
the per-column dot products and the 1024-token chunk gate moves from 0.042 to
0.081 logits; a four-lane variant with pre-normalized q/k, one block barrier per
row, and shuffle-derived decay and beta preserves the summation order and
measures 0.0419 on the same gate, but it is no faster than the in-kernel form
(40.3 ms against 39.9 ms), so the original kernel stayed. The grouped attention
kernel keeps the reference at 4.1e-07 maximum absolute difference on a native
2048-position case and is bit-identical to the one-head kernel on the 1024-token
chunk gate.

Reproduction:

```bash
MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --prompt-tokens 512 \
  --decode-tokens 1 --warmup 2 --reps 5 --chunk-size 512
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --prompt-tokens 8192 \
  --decode-tokens 1 --warmup 1 --reps 2 --chunk-size 512
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run TestGPUProfilePrefill -v -count=1
./build/yuuka_prefill_sweep | grep Q6_K
```

### Short-prompt latency and correctness

The Phase 2 baseline at the top of this document submitted 16 prompt tokens
through the decode graph and measured 1.433 s prefill (11.17 tok/s) and 9.87
decode tok/s on HIP. With chunked prefill on the same model and prompt length,
`bench --prompt-tokens 16 --decode-tokens 8 --chunk-size 16` measures 0.137 s
prefill (117.02 tok/s) and 9.88 decode tok/s. Decode is unchanged; short-prompt
TTFT improved by 10.5x.

Chunked prefill correctness is checked at two levels:

- `tests/hip/prefill_test.cpp` compares the fused chunk kernels, WMMA GEMM, and
  Flash Attention against independent float references.
- `engine/gpu/prefill_test.go` runs the tiny hybrid fixture with chunk sizes 1,
  3, 7, 16, 17, 33, and 64, requires the same top-1 token as the CPU reference,
  and enforces a `2e-3` maximum logit difference. It also decodes one token
  after each chunked prefill and compares that logit vector with the CPU state
  produced by a single unchunked prefill.
- `TestGPUChunkedPrefillLongContext` prefills 1024 real-model tokens with chunk
  sizes 128, 512, and 1024. In the recorded run the three logit vectors were
  bitwise identical (maximum absolute difference `0`) and selected the same
  top-1 token. The checked-in test enforces the same top-1 token and a `0.05`
  maximum absolute logit difference, so it does not by itself guarantee
  bitwise identity. This is the KV/recurrent-state equivalence check at a
  context length where the earlier shared-scratch race was observable; the
  fixed kernel is also checked for bitwise repeatability in the native test.

A 32-token real-model prompt ("Write a short list of colors...") produces the
same greedy continuation with chunk sizes 7, 16, and 32 as the CPU reference.
Chunk 7 exercises a partial final chunk (7,7,7,7,4) and mixes WMMA for wide
projections with batched GEMV for tall ones; chunks 16 and 32 select WMMA for
every projection. Prefill time is 0.526 s, 0.265 s, and 0.167 s respectively.

### Kernel resource usage

`scripts/kernel_resources.sh` compiles the dispatched kernel families with
`-Rpass-analysis=kernel-resource-usage`. All listed kernels report zero scratch,
zero VGPR spills, and 16 waves/SIMD occupancy.

| Kernel | VGPR | SGPR | Static LDS | Dynamic LDS |
| --- | ---: | ---: | ---: | ---: |
| `gemm_wmma_kernel<*,64,2>` | 64-75 | 28-34 | 32,768 B | 0 |
| `gemm_wmma_k32_kernel<*>` | 39-50 | 26-34 | 14,336 B | 0 |
| `gemm_wmma_swiglu_kernel<*,64,2>` | 93 | 38 | 24,576 B | 0 |
| `gemm_wmma_swiglu_kernel<*,32,1>` | 93 | 38 | 20,480 B | 0 |
| `attention_prefill_f32_kernel` | 49 | 48 | 0 | 36,864 B at head_dim 256 |
| `attention_prefill_paged_f32_kernel<dense>` | 51 | 63 | 0 | 36,864 B at head_dim 256 |
| `attention_prefill_paged_f32_kernel<gated>` | 53 | 61 | 0 | 36,864 B at head_dim 256 |
| `gdn_chunk_f32_kernel` | 37 | 70 | 8 B | 2,048 B at state 128 |
| `gdn_chunk_splitreg_f32_kernel<128>` | 93 | 34 | 1,064 B | 0 |
| `gdn_norm_gate_rows_f32_kernel` | 17 | 18 | 0 | 0 |
| `rmsnorm_batch_f32_kernel` | 9 | 18 | 1,024 B | 0 |
| `rmsnorm_heads_batch_f32_kernel` | 9 | 20 | 1,024 B | 0 |
| `rope_batch_f32_kernel` | 19 | 18 | 0 | 0 |
| `kv_write_batch_f32_kernel` | 14 | 34 | 0 | 0 |
| `embedding_batch_f32_kernel` | 21 | 19 | 0 | 0 |
| `conv1d_chunk_f32_kernel` | 26 | 34 | 0 | 0 |
| `cast_f32_to_f16_kernel` | 7 | 18 | 0 | 0 |

Dynamic LDS is set by the launcher and does not appear in the compiler's static
figure.

### Phase 3 reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka

build/yuuka_prefill_sweep
scripts/kernel_resources.sh

for p in 128 512 2048; do
  /tmp/yuuka-hip bench \
    --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
    --prompt-tokens "$p" --decode-tokens 16 \
    --warmup 1 --reps 3 --chunk-size 512
done

# The recorded 8192 and 32768 runs used one measurement and no warmup.
for p in 8192 32768; do
  /tmp/yuuka-hip bench \
    --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
    --prompt-tokens "$p" --decode-tokens 16 \
    --warmup 0 --reps 1 --chunk-size 512
done

# llama.cpp comparison, same GGUF and device.
build/bin/llama-bench -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  -p 128,512,2048,8192,32768 -n 0 -r 2 -o md
LD_LIBRARY_PATH=$PWD/build-vk/bin build-vk/bin/llama-bench \
  -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  -p 128,512,2048,8192,32768 -n 0 -r 2 -o md
```

The llama.cpp Vulkan build requires the Vulkan loader and a gfx1151 ICD. The
recorded run used Mesa RADV 25.0.7. Warmup and repetition counts differ for the
longest prompts in the yuuka curves; the values reported below state the actual
count used.


## Phase 4 serving

### Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-11 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host CPU / kernel | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S; Linux 6.17.13-21-pve |
| Commit / Go | `6a9127e` plus the Phase 4 fixes in this change; Go 1.24.4. `bench/serving` prints `vcs.revision`, `-dirty`, and the toolchain metadata |
| Power performance level | `auto` reported by the amdgpu sysfs interface |
| Backend | HIP, paged FP32 KV, block size 16 |
| Context | 256 positions |
| Sampling | Greedy, temperature 0 |
| Prompt | 64 tokens from repeated English prose |
| Decode | 16 tokens per request |
| Prefix cache | disabled unless stated |
| Graph replay | enabled unless stated |
| Warmup | one request for latency/throughput runs; none for prefix runs |

Measurements use `bench/serving`, which reports queue wait, TTFT, prefill,
decode, and sampling separately. TTFT is measured from request arrival and
therefore includes queue wait; queue wait is also reported on its own. The
benchmark prints the model hash and size, VCS revision (with `-dirty` when
applicable), Go version, host CPU, kernel, `hipconfig` version, amdgpu power
performance level, sampling mode, warmup, requests, and concurrency. Run one
benchmark process at a time; two processes sharing the GPU invalidate both
results.

### Single-user latency and graph replay

Concurrency 1, 3 measured requests, one warmup request.

| Configuration | Latency P50 | TTFT P50 | Prefill P50 | Decode P50 | Sampling P50 | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph replay | 1.796 s | 0.257 s | 0.245 s | 1.530 s | 1.69 ms | 8.91 |
| Plain stream | 1.797 s | 0.258 s | 0.246 s | 1.529 s | 3.04 ms | 8.91 |

Graph replay changes decode time by less than run-to-run noise at this context,
so the server default is the plain stream path. Queue wait was about 2 µs in
both runs.

### Multi-request throughput

Concurrency 4, 8 measured requests, 64 KV blocks, graph replay enabled.

| Metric | P50 | P95 |
| --- | ---: | ---: |
| End-to-end latency | 6.919 s | 7.212 s |
| Queue wait | 0.205 s | 0.452 s |
| TTFT | 1.010 s | 1.084 s |
| Prefill | 0.239 s | 0.245 s |
| Decode | 1.533 s | 1.534 s |
| Sampling | 1.64 ms | 1.65 ms |

Aggregate throughput was 0.557 requests/s and 8.915 output tokens/s over a
14.358 s wall interval. Per-request decode time is unchanged from the
single-request run. TTFT is higher than in the single-user run because it
includes queue wait and waits for decode/prefill opportunities on the shared
stream. The scheduler interleaves request plans on one stream, so concurrency
improves admission fairness and exposes queue wait, but does not raise aggregate
token throughput; fused mixed-shape batched kernels are the missing
optimization.

### Prefix cache

Three identical 64-token prompts at concurrency 1 with the prefix cache enabled
and no warmup: one miss and two full-prompt hits. The first request's TTFT was
0.265 s; the two hits had TTFT P50 of 4.78 ms and zero prefill time because the
cached recurrent state and prompt logits were restored. TTFT includes arrival,
admission, and sampling overhead, so it is not zero even for a full cache hit.
Decode time remained 1.533 s. Aggregate throughput was 9.813 output tokens/s.

### Phase 4 correctness evidence

- `go test -tags hip ./engine/gpu ./model/qwen35 ./serving` with
  `YUUKA_TEST_MODEL` set passes on the Q4_K_M model with the paged kernels and
  graph replay enabled in the GPU serving tests.
- Concurrent CPU and HIP requests reproduce isolated-run token sequences and
  text at the configured concurrency (3 in the checked-in serving tests).
- Full and partial prefix hits reproduce a fresh run's tokens and text on both
  backends.
- Cancellation leaves zero used KV blocks, no active requests, and balanced
  allocator counters on both backends.
- Twelve sequential request cycles with the prefix cache disabled leave zero
  used blocks and no active requests.
- Native `hip_paged_graph_correctness` covers paged KV writes, paged decode and
  prefill attention against a float reference, device copies, dynamic graph
  replay at several positions, and rejection of non-dynamic plans.

### Phase 4 reproduction

```bash
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving

# Single user, graph replay.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph

# Same run on the plain stream path.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph=false

# Four concurrent requests.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 64 \
  --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph

# Prefix reuse.
/tmp/yuuka-serving-bench --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --ctx 256 --block-size 16 --kv-blocks 16 \
  --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 0 --graph --prefix-cache
```

## Phase 5 expansion features

### Environment and protocol

Phase 5 measurements use the same model, device, ROCm/HIP version, greedy
sampling, warmup, and repetition settings as the Phase 2/4 sections unless a
row states otherwise. The KV-format runs use a 128-token prompt and 32 generated
tokens. The speculative HIP rows use the same 128/32 protocol; the CPU rows in
that table are a 16-prompt/8-decode negative control. The MoE sweep uses
synthetic Q4_K weights at Qwen3-30B-A3B-like single-layer shapes because no
standard `qwen3moe` GGUF is available on the target host. NPU support is a future
task and is not part of the Phase 5 benchmark set.

### KV cache formats

Qwen3.5-4B has 8 full-attention layers, 4 KV heads, and head size 256. One KV
row is 1024 values per layer. The three formats store:

| Format | Bytes per token (K+V) | Relative to F32 | 4096-token request |
| --- | ---: | ---: | ---: |
| F32 | 65,536 | 1.00x | 256 MiB |
| F16 | 32,768 | 0.50x | 128 MiB |
| Q8_0 | 17,408 | 0.266x | 68 MiB |

| KV format | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| F32 | 0.412 s | 0.412 s | 3.203 s | 9.68 |
| F16 | 0.411 s | 0.411 s | 3.192 s | 9.71 |
| Q8_0 | 0.428 s | 0.428 s | 3.180 s | 9.75 |

The latency differences are within run-to-run noise. KV format is a memory
feature, not a decode throughput feature, at this context length. The real-model
correctness test generated the same six greedy tokens for F32, F16, and Q8_0;
maximum logit differences from F32 were `1.12e-3` (F16) and `4.53e-2` (Q8_0).

### Speculative decoding

Exact greedy prompt-lookup speculation with `--speculative`, four draft tokens,
and two-token minimum match. The HIP comparison uses the 128-token
repeated-prose prompt; the CPU negative control uses 16 prompt tokens and 8
generated tokens:

| Path | Prompt/decode | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s | Acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HIP plain greedy | 128/32 | 0.412 s | 0.412 s | 3.203 s | 9.68 | — |
| HIP speculative | 128/32 | 0.429 s | 0.575 s | 1.024 s | 30.26 | 32/32 draft tokens |
| CPU plain greedy | 16/8 | 7.328 s | 7.329 s | 3.637 s | 1.92 | — |
| CPU speculative | 16/8 | 7.327 s | 9.996 s | 6.277 s | 1.12 | 0/10 draft tokens |

The generated token sequence is identical to the plain greedy path. TTFT rises
because the first step drafts and verifies a block before emitting its first
token. The measured decode speedup is about 3.1x; repeated runs ranged from 30.3
to 31.2 tok/s. On a 24-token real-model test the same settings produced 2.333 s
baseline decode versus 0.719 s speculative decode with identical tokens.
`TestSpeculativePartialAcceptanceMatchesGreedy` additionally forces a block
with one accepted and one rejected draft through a scripted runtime and checks
the rollback/replay result against plain greedy tokens.

The CPU reference path is not a speculative target. On the same model with a
16-token prompt and 8 generated tokens, plain CPU decode was 3.637 s (1.92
tok/s); speculative decode was 6.277 s (1.12 tok/s) with 10 draft tokens and 0
accepted. Per-candidate LM-head work and no accepted draft made it slower. GPU
speedup therefore depends on batched verification and real acceptance; the CPU
result is recorded as a negative experiment, not a claim.

Acceptance is prompt-dependent: a prompt with no repeated n-grams falls back to
normal decode.

### MoE grouped expert kernels

Synthetic single-layer Q4_K shapes: 128 experts, top-8, hidden 2048, expert FFN
768. Times are one route + grouped gate/up + SwiGLU + grouped down + weighted
reduce on one HIP stream. Per-op times come from `yuuka_plan_execute_profiled`;
total is the steady-state wall time including the plan submission.

| Rows | Route ms | Gate/up ms | Down ms | Reduce ms | Total ms | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.225 | 0.380 | 0.411 | 0.016 | 1.055 | 948 |
| 2 | 0.231 | 1.022 | 0.778 | 0.018 | 2.061 | 970 |
| 4 | 0.240 | 1.969 | 1.470 | 0.021 | 3.670 | 1090 |
| 8 | 0.245 | 3.731 | 2.747 | 0.026 | 6.749 | 1185 |
| 16 | 0.251 | 7.274 | 4.870 | 0.029 | 12.409 | 1289 |
| 32 | 0.253 | 13.806 | 9.453 | 0.038 | 23.462 | 1364 |

These are kernel-pipeline numbers, not end-to-end model throughput. Routing and
reduction are negligible; gate/up and down are limited by packed-weight reads.
The F32 synthetic fixture CPU reference and HIP pipeline agree within `3.9e-5`
maximum logit difference, and the native F32 MoE test agrees within `8.7e-9` on
its direct kernel comparison. The Q4_K synthetic fixture test
`TestTinyMoEQuantizedGPUMatchesCPU` compares chunked prefill and
prefill-plus-decode against the CPU decoder; the observed maximum logit
difference is `1.19e-6`. No standard `qwen3moe` GGUF is available on the target
host, so real-model MoE validation remains open.

### Q4_0 and Q5_0 GEMV

`bench/hip/gemv_sweep` at 2560x2560, one projection per measurement:

| dtype | Batch 1 per-row ms | Batch 1 batched ms | Batch 8 batched ms | Batch 16 batched ms | Weight MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 0.060 | 0.093 | 0.240 | 0.409 | 3.5 |
| Q5_0 | 0.102 | 0.092 | 0.263 | 0.433 | 4.3 |

Batched GEMV wins from batch 4 for Q4_0. The Q5_0 batch-1 difference was within
run-to-run noise in repeated sweeps; the dispatch table uses batch 4 for both
Q4_0 and Q5_0. The tensor, GEMV, embedding, and WMMA paths pass randomized
tests; GEMV and WMMA additionally cover non-aligned dimensions. The embedded
Q4_0/Q5_0 row test covers both `EMBEDDING` and `EMBEDDING_BATCH`.

### Phase 5 kernel resource usage

`scripts/kernel_resources.sh` also compiles the Phase 5 GEMV, WMMA, and MoE
variants. All rows report zero scratch, zero VGPR spills, and 16 waves/SIMD
occupancy.

| Kernel | VGPR | SGPR | Static LDS | Dynamic LDS |
| --- | ---: | ---: | ---: | ---: |
| `gemv_packed_kernel<Q4_0>` | 13 | 17 | 1,024 B | 0 |
| `gemv_packed_kernel<Q5_0>` | 15 | 17 | 1,024 B | 0 |
| `gemv_batch_kernel<Q4_0>` | 28 | 37 | 1,024 B | 0 |
| `gemv_batch_kernel<Q5_0>` | 29 | 37 | 1,024 B | 0 |
| `gemm_wmma_kernel<Q4_0>` | 46 | 26 | 14,336 B | 0 |
| `gemm_wmma_kernel<Q5_0>` | 47 | 26 | 14,336 B | 0 |
| `moe_route_kernel` | 10 | 32 | 0 | 0 |
| `moe_gate_up_grouped_kernel` | 23 | 48 | 1,024 B | 0 |
| `moe_down_grouped_kernel` | 21 | 42 | 1,024 B | 0 |
| `moe_reduce_add_kernel` | 9 | 21 | 0 | 0 |

### Phase 5 reproduction

```bash
go test ./...
go test -tags hip ./...
ctest --test-dir build --output-on-failure

go run -tags hip ./cmd/yuuka bench --model "$YUUKA_TEST_MODEL" --backend hip \
  --prompt-tokens 128 --decode-tokens 32 --warmup 1 --reps 3 --kv-type f16
go run -tags hip ./cmd/yuuka bench --model "$YUUKA_TEST_MODEL" --backend hip \
  --prompt-tokens 128 --decode-tokens 32 --warmup 1 --reps 3 --speculative

./build/yuuka_gemv_sweep
./build/yuuka_moe_sweep
scripts/kernel_resources.sh
go test ./engine -run TestSpeculativePartialAcceptanceMatchesGreedy -v
go test -tags hip ./engine/gpu -run TestTinyMoEQuantizedGPUMatchesCPU -v
YUUKA_TEST_MODEL="$YUUKA_TEST_MODEL" go test -tags hip ./engine/gpu/... \
  -run 'TestGPURealModelKVFormats|TestGPURealModelSpeculativeMatchesGreedy' -v
```

## Release decode-length matrix

This matrix completes the output-length dimension of the first-release
performance gate. The Phase 3 curves cover the prompt-length dimension; this
run covers the required 128- and 256-token decode lengths on the validated
Qwen3.5-4B Q4_K_M file.

| Field | Value |
| --- | --- |
| Date | 2026-11-13 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host CPU / kernel | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S; Linux 6.17.13-21-pve |
| Code commit / Go | `a67ec00`; Go 1.24.4 |
| Backend / KV | HIP, F32 KV |
| Execution path | Single request, plain stream (no HIP Graph replay) |
| Prompt | 128 tokens from the bench prompt generator |
| Sampling | Greedy, temperature 0 |
| Warmup / repetitions | 1 / 3 |
| Power performance level | `auto` reported by the amdgpu sysfs interface |

| Decode tokens | Context | Prefill p50 / p95 | TTFT p50 / p95 | Decode p50 / p95 | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 320 | 0.409 s / 0.409 s | 0.409 s / 0.410 s | 13.223 s / 13.231 s | 9.60 |
| 256 | 448 | 0.411 s / 0.412 s | 0.411 s / 0.412 s | 26.838 s / 26.846 s | 9.50 |

Decode throughput uses `(decode_tokens - 1) / decode p50`, matching the CLI
report. Decode is weight-bandwidth bound; the two rows differ by about 1%.

Reproduction:

```bash
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka

for n in 128 256; do
  /tmp/yuuka-hip bench \
    --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
    --prompt-tokens 128 --decode-tokens "$n" --warmup 1 --reps 3
done
```

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

go build -o /tmp/yuuka-cpu ./cmd/yuuka
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka

/tmp/yuuka-cpu bench \
  --model /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt-tokens 16 --decode-tokens 8 --warmup 1 --reps 3 --hash

/tmp/yuuka-hip bench \
  --model /path/to/Qwen3.5-4B-Q4_K_M.gguf --backend hip \
  --prompt-tokens 16 --decode-tokens 16 --warmup 1 --reps 3 --hash
```

The benchmark tool reports P50 and P95 over the measured repetitions. It does
not yet record power, clock, or thermal state.
