# Stage 1b-2 Direct FP16 WMMA

This step implements [sub-128 projection routing](../stage1bc-sub128-rows.md).
The control is [Stage 1b-1](04a-streaming-short-tails.md) at `ac659b7`.
The measurements use the same Qwen3.5-4B Q4_K_M and Phase 1 token fixture
hashes as 1b-1. AI has a Radeon 8060S (`gfx1151`), ROCm 10.1.0, HIP
7.16.26385 and Go 1.24.4. The GPU runs in automatic power mode, with one
GPU workload at a time. Both libraries are Release gfx1151 builds.

## Implementation and experiments

The [WMMA launcher](../../../../kernels/hip/gemm/wmma_gemm.hip) selects
Direct FP16 for eligible Q4_K, Q5_K and Q6_K projections at 17–127 rows.
The Go planner and C ABI use the existing WMMA and SwiGLU operations.
A compact final-layer projection still has actual `M=1` and uses the
parent's FP16 arithmetic through the legacy kernel. The exact build
target, supported widths and pointer alignment guard the native route.

The first Direct prototype used the same K order in both half-waves.
Although the GEMM produced the expected dot products, the order of
terms in the instruction changed. The model's maximum logit changes
were 0.230990 at 96 tokens and 0.258722 at 127. Ordering the second set
of eight K values first in the upper half-wave matches legacy rocWMMA
and reproduced the full control logits byte for byte at 17, 18, 64,
96, 127, 530 and 600 tokens. All eight continuation steps also matched.
This ordering is retained in the final kernel.

The isolated SwiGLU comparison used the same 2560 → 9216 Q4_K
matrices and matched F32 input. Both prototypes use 32- or 64-token
tiles. The 64-channel tile doubled the K step and produced the same
127-token model result as the 128-channel trial.

| Fused geometry | 18 rows (ms) | 32 rows (ms) | 64 rows (ms) | 96 rows (ms) | 127 rows (ms) | 127-token model p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 channels, K=32 | 0.389 | 0.403 | 0.420 | 0.749 | 0.842 | 76.877 |
| 64 channels, K=64 | 0.273 | 0.264 | 0.348 | 0.535 | 0.512 | 69.974 |

The selected fused tile stays at 64 channels with a direct register
SwiGLU epilogue. The single GEMM has 128 channels. Both use K=64.
Hoisting each scale through its entire 32-value Q4_K/Q5_K or
16-value Q6_K span reduced repeated scale work further. The swizzle
uses the low K-span bits in each LDS row index to distribute both
vector stores and fragment loads. There is no intermediate FP16
weight allocation or accumulator tile in LDS.

## Verification method

- [Native tests](../../../../tests/hip/prefill_test.cpp) exercise both
  ordinary and fused WMMA for Q4_K, Q5_K and Q6_K at 17, 18, 32,
  33, 64, 65, 96 and 127 rows with `256 → 128` and `512 → 256`
  shapes. The independent reference rounds the operands to FP16,
  uses a host dot product and checks the SwiGLU product. Both
  repetitions must have identical bits and preserve the trailing
  sentinel. The existing FP32-dot tests retain their prior checks.
- [The real-model tests](../../../../engine/gpu/prefill_real_test.go)
  compare 17 and 64 tokens to the CPU with F16 KV. For the same-chunk
  control, set `YUUKA_TEST_PREFILL_FP16_CONTROL` to a wrapper around
  `prefill-compare` built at `ac659b7` and linked to its own HIP
  library. Use the same wrapper construction as 1b-1. The suite
  compares full logit rows and nine greedy tokens, including the first
  top-1, for 17, 18, 64, 96, 127, 530 and 600 tokens and pins chunk 512.
- `scripts/kernel_resources.sh` compiles the same gfx1151 path as
  CMake. `rocprofv3 --pmc LDSBankConflict --kernel-include-regex
  gemm_wmma_direct_kernel -f rocpd` records the bank-conflict metric
  for the production templates in the sweep and in native tests.

## Native performance

Both libraries ran the same `yuuka_prefill_sweep --sub128` source.
Each cell has the median of three process means, with two warmups
and five timed submissions per process. Entries are legacy / Direct
milliseconds. The FP16 dequantization plus rocBLAS rows included
dequantization and input casts. Direct was faster than that route in
all 50 single-projection cells below.

| Dtype | Input → output | Operation | 18 rows | 32 rows | 64 rows | 96 rows | 127 rows |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2560 → 1024 | WMMA | 0.114 / 0.058 | 0.133 / 0.057 | 0.131 / 0.092 | 0.135 / 0.081 | 0.143 / 0.093 |
| Q4_K | 2560 → 4096 | WMMA | 0.135 / 0.067 | 0.150 / 0.071 | 0.250 / 0.100 | 0.289 / 0.120 | 0.297 / 0.131 |
| Q4_K | 2560 → 8192 | WMMA | 0.167 / 0.083 | 0.184 / 0.089 | 0.311 / 0.130 | 0.589 / 0.200 | 0.671 / 0.225 |
| Q4_K | 2560 → 9216 | WMMA | 0.176 / 0.091 | 0.197 / 0.094 | 0.367 / 0.129 | 0.638 / 0.245 | 0.697 / 0.250 |
| Q4_K | 2560 → 9216 | SwiGLU | 0.488 / 0.233 | 0.501 / 0.235 | 0.659 / 0.279 | 1.018 / 0.444 | 1.123 / 0.453 |
| Q4_K | 4096 → 2560 | WMMA | 0.183 / 0.080 | 0.216 / 0.084 | 0.219 / 0.128 | 0.237 / 0.142 | 0.274 / 0.147 |
| Q4_K | 9216 → 2560 | WMMA | 0.387 / 0.165 | 0.459 / 0.170 | 0.475 / 0.254 | 0.514 / 0.280 | 0.601 / 0.309 |
| Q5_K | 2560 → 8192 | WMMA | 0.175 / 0.110 | 0.195 / 0.110 | 0.324 / 0.145 | 0.614 / 0.240 | 0.690 / 0.263 |
| Q5_K | 4096 → 2560 | WMMA | 0.194 / 0.106 | 0.226 / 0.104 | 0.232 / 0.144 | 0.253 / 0.161 | 0.293 / 0.166 |
| Q6_K | 2560 → 1024 | WMMA | 0.130 / 0.070 | 0.144 / 0.070 | 0.148 / 0.100 | 0.157 / 0.099 | 0.164 / 0.105 |
| Q6_K | 9216 → 2560 | WMMA | 0.451 / 0.225 | 0.516 / 0.239 | 0.562 / 0.342 | 0.670 / 0.367 | 0.794 / 0.387 |

The 55 recorded cells were 1.42–2.98× faster. The three target shape
classes each met 1.5× across all five row counts. Reported SwiGLU
bandwidth and arithmetic throughput count both weight matrices and
both GEMMs.

## Model and serving performance

The unprofiled A/B uses F16 KV and chunks of up to 512 tokens. Each
length ran control, candidate, candidate, control with three warmups
and seven timed prefills per process. Reported medians pool 14 samples
per build. Temperature before process starts was 49–65 °C.

| Prompt tokens | Stage 1b-1 p50 (ms) | Direct FP16 p50 (ms) | Change |
| ---: | ---: | ---: | ---: |
| 18 | 58.216 | 35.482 | −39.05% |
| 64 | 80.714 | 47.539 | −41.10% |
| 96 | 107.973 | 60.077 | −44.36% |
| 127 | 117.919 | 66.295 | −43.78% |
| 128 | 68.204 | 68.141 | −0.09% |
| 512 | 217.347 | 217.381 | +0.02% |
| 530 | 275.372 | 253.476 | −7.95% |
| 600 | 321.792 | 274.990 | −14.54% |
| 2048 | 904.158 | 912.455 | +0.92% |
| 8192 | 4109.704 | 4106.951 | −0.07% |

The short-prompt sequence is monotonic. The 127-row chunk meets the
82 ms gate, both 512-row tails improve in both process pairings and
the long-prompt cells remain inside ±1%.

The same serving configuration as [1b-1](04a-streaming-short-tails.md)
used `-prefix-cache -append-tokens 18` with a verified 512-token
prefix. Per-process TTFT medians in process order were 66.753,
43.231, 43.150 and 67.443 ms.

For HTTP SSE the same nonce-bearing chat request used 50 repetitions
of the benchmark phrase instead of 202. It rendered 530 tokens. The
per-process first-content medians were 286.721, 258.428, 259.184 and
282.448 ms. The pooled medians were 283.387 and 258.906 ms. The
requests ran at concurrency one, with two warmups and nine timed
requests per process, an eight-token completion cap and greedy
sampling. All server processes had prefix caching disabled.

## Resource measurements

All 12 compiled Direct templates use 81–120 VGPRs, zero VGPR
spills and zero private scratch. The 32-token tile uses 20 KiB
and the 64-token tile 24 KiB of static LDS. The launch reserves
no dynamic LDS. The final model-shape sweep recorded 0.0%
`LDSBankConflict` in all 539 Direct dispatches.

The native synthetic profile captured all 12 templates. It recorded
0.0% in 197 of 198 dispatches, with one Q5_K `512 → 256` fused
dispatch at 65 rows reporting 100%. Three repeat passes on the mixed
native suite returned five zero values and one 100% value for that
exact cell (mean 16.67%). In the nonzero sample the numerator and
denominator were both 9,728. Some other mixed-suite dispatches returned
zero active LDS cycles.

To measure valid counters for that cell, a separate probe reused one
plan and buffers and dispatched Q5_K `512 → 256` SwiGLU with 65 rows
64 times. All 64 dispatches had positive active cycles, with 4,980,736
active cycles and zero bank conflicts in aggregate. The same isolated
probe ran Q5_K and Q6_K `2560 → 9216` SwiGLU at 18 and 127 rows.
Each of those four 64-dispatch cells also had positive active cycles
and zero bank conflicts. The model sweep and isolated valid samples
meet the <5% diagnostic gate.

## Correctness and review

The final same-chunk model comparisons at 17, 18, 64, 96, 127,
530 and 600 tokens were exact on the full logit vector, with the
same prefill top-1 and eight greedy continuations. The F16 KV CPU
oracle comparisons at 17 and 64 tokens measured maximum errors of
0.00704288 and 0.0216442, with identical greedy sequences.

The unmodified 64-token attention comparison passed at 0.0147166.
The five-token model gate reported 0.00004673, and speculative
verification accepted 24 of 24 draft tokens. The full HIP suite
(including both control wrappers), CTest 10/10, both Go vet
variants, Go build and the untagged race suite passed on AI. The
MoE and two SentencePiece golden tests skipped for missing optional
fixtures. The changed Go and native files in the AI checkout
matched the implementation files by SHA-256.

Two independent reviews ended in `NO_BLOCKING_ISSUES`. After the
first round, fixed-size memcpy replaced the typed reinterpretation
of the temporary FP16 array, and the SwiGLU sweep began counting
both GEMMs' throughput. The final sweeps, resource checks, matrix
and model tests, and server A/B were repeated after those fixes.
