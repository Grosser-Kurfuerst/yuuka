# Iteration 2 Q4_K FFN projection

## Scope and gate

Evaluate the Q4_K MMQ kernel at `M=512, K=2560, N=9216`. The stage
requires at most **1.05 ms** per projection and a repeatable improvement
in unprofiled pp512. The [projection plan](../projections-and-ffn.md)
describes the original eight-wave mapping and the proposed four-wave
mapping.

## Measurement

All GPU work ran in the AI container on the [iteration 1 device, toolchain,
model, and power profile](01-baseline.md#environment). The candidates
were built in a source tree separate from the iteration 1 control.
The native sweep reused the same `run_shape` implementation and seed for
every trial, restricting its entry point to the target shape. Each A/B ran
control, candidate, candidate, control. Each sweep process used two warmups
and five timed samples. Pure MMQ medians come from the five timed dispatches
in the ROCm kernel trace. Unprofiled pp512 used the real GGUF, F16 KV,
chunk 512, two warmups and seven timed repetitions per process.

| Experiment | Control MMQ medians (ms) | Candidate MMQ medians (ms) | Control pp512 (tok/s) | Candidate pp512 (tok/s) |
| --- | --- | --- | --- | --- |
| Four waves, same tile and LDS layout | 1.281, 1.327 | 1.772, 1.801 | 1774.20, 1766.90 | 1574.24, 1570.96 |
| Eight waves, paired scale/min in LDS | 1.274, 1.310 | 1.316, 1.292 | 1762.37, 1768.66 | 1747.17, 1753.34 |
| Eight waves, direct WMMA with LDS broadcast | 1.259, 1.296 | 1.167, 1.176 | 1763.99, 1767.66 | 1809.17, 1806.85 |
| Four waves, direct WMMA with shorter-lived accumulators | 1.291, 1.303 | 1.319, 1.303 | 1768.77, 1750.86 | 1749.68, 1737.00 |
| Eight waves, direct WMMA with shorter-lived accumulators | 1.303, 1.321 | 1.163, 1.191 | 1773.32, 1760.92 | 1805.56, 1809.05 |
| Eight waves, fixed-shape direct WMMA | 1.316, 1.310 | 1.105, 1.114 | 1769.88, 1760.22 | 1834.44, 1824.82 |
| Eight waves, fixed shape and shorter-lived accumulators | 1.276, 1.291 | 1.098, 1.134 | 1775.08, 1755.64 | 1843.87, 1841.55 |

- **Four waves.** Each lane owns eight token fragments instead of four.
  VGPR count rose from 136 to 192 and the trace reported 100 bytes of
  scratch. The end-to-end slowdown agrees with the isolated regression.
- **Paired scales.** The writer and matching reader used `float2` for
  weight scale and signed minimum. VGPR count and LDS size matched the
  control. Both pp512 medians were below their controls.
- **Direct WMMA.** Store each 16-value K slice contiguously per channel
  or token in LDS. Lanes `i` and `i+16` read the same operand address,
  delivering replicated operands to the gfx1151 WMMA instruction. This
  keeps the working group's LDS allocation at 30 KiB. VGPR count dropped
  to 128 with zero scratch. The complete target-shape CPU comparison
  passed at the existing tolerance with a maximum absolute error of
  approximately `1.3e-4`. The candidate improved standalone pp512 by
  about 2.4% in this A/B.
- **Shorter-lived accumulators.** Finish both 16-wide WMMA instructions
  and the scale correction for one token fragment before starting its
  neighbor. This removed scratch from the four-wave candidate and lowered
  its VGPR count to 152. Its pp512 declined in both trials. With eight
  waves the candidate used 112 VGPRs with zero scratch and delivered a
  similar pp512 gain to the first direct-WMMA version. Both passed the
  full target-shape CPU comparison at the original tolerance.
- **Fixed shape.** Limit the direct variant to the exact dimensions and
  use them as compile-time constants. This removes edge checks for the
  fully populated tiles. The candidate passed the complete CPU comparison
  with a maximum absolute difference of about `1.3e-4`. It used 128
  VGPRs and no scratch. The average of the two pp512 medians was 3.7%
  above the average of its two control medians.
- **Fixed shape with shorter-lived accumulators.** This combination used
  120 VGPRs and no scratch. Every output of the full target shape was
  compared against the CPU reference at the existing tolerance. The
  maximum absolute difference was about `1.3e-4`. Average pp512 throughput
  across the two run medians rose from 1765.36 to 1842.71 tok/s (4.4%).
  Some traced dispatches took less than 1.05 ms, while both medians
  exceeded the target.

The best initial combination reduced the median VALU count from 22.19 to 18.14
and the LDS count from 4.75 to 2.88 per 4096 useful MACs. Both dispatched
4608 waves for the cell. The counts come from five timed dispatches under
`rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES`.

Raw files, source variants, and temporary probe sources are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-02/` on AI.
An in-run telemetry snapshot recorded automatic power mode and 49 °C.

## Initial decision

The initial candidates demonstrated a repeatable pp512 gain and exceeded
the 1.05 ms time limit. The runtime continued to use the original kernel
while the tested candidate and exact-shape CTest remained in the trial
source on AI.

## Initial verification on AI

- The final combination passed the exact-shape CPU comparison through the
  native MMQ entry point. The comparison covered all 4,718,592 outputs.
- The complete native CTest suite passed 9/9 on the final combination.
  The review revision also passed 9/9 with the exact target shape added
  to CTest. The expanded comparison reported maximum absolute error
  `1.14e-4` and checked that the error was finite.
- `go vet -tags hip ./...`, `go test -p 1 -tags hip ./... -count=1 -v`,
  `go build ./...`, `go test ./...`, and `go test -race ./...` passed
  on the final combination. The real-model HIP run used the verified
  Qwen GGUF. The optional MoE golden test and two SentencePiece tests
  skipped because their separate fixtures were absent.
- The review revision repeated the five real-model and tiny-fixture
  gates listed in the [baseline verification](01-baseline.md#verification).
  All five passed with zero skips.

## Initial complete FFN and prompt-length check

The review revision used the same model and F16 KV with fixed chunk 512.
A Go test overlay on `TestGPUProfilePrefill` summed the eight consecutive
plan operations of each FFN, from its input normalization to its residual
addition. It matched all 32 FFNs in each position-zero sample. The
three-run median fell from **143.773 ms** to **130.732 ms**. The
best candidate's sweep averaged 1.093 ms and 1.100 ms in the two processes
when including Q8_1. Its controls averaged 1.318 ms and 1.342 ms.
Those are wall-clock means from the native sweep. The primary gate uses
pure-MMQ profiler medians from the separate traces above.

Unprofiled CLI, two warmups and five timed repetitions per cell, in a
single control-first comparison:

| Prompt | Control tok/s | Candidate tok/s |
| ---: | ---: | ---: |
| 128 | 1500.32 | 1487.23 |
| 512 | 1741.00 | 1808.88 |
| 2048 | 1710.67 | 1776.96 |
| 8192 | 1498.63 | 1544.30 |

The 128-token benchmark does not meet the candidate dispatch condition.
In a candidate/control/control/candidate repeat with two warmups and seven
timed samples per process, p50 was 1506.50 and 1501.82 tok/s for the
candidate and 1506.91 and 1507.32 tok/s for the control. All four
latencies rounded to 85 ms. Raw files are in the `review-fix/` subdirectory.

## Initial review

Two read-only review rounds. Round 1 requested coverage of the exact
dispatch cell in CTest, finite-value checks, additional model benchmarks,
and updated kernel documentation if the candidate ships. The experimental
revision includes the CTest and finite-value checks, shares shape constants
between device and host, and removes an unreachable branch. The verification
and performance sections report the follow-up tests.

Round 2 verdict for the evaluation record: `NO_BLOCKING_ISSUES`.

## Retry on gfx1151

The retry kept the shape, model, scale arithmetic and measurement
conditions. Each screening figure is the median of five traced MMQ
dispatches after two warmups. These first measurements select candidates
for repeated GPU tests.

| Experiment | Output channels × tokens per workgroup | Pure MMQ screening median (ms) |
| --- | --- | ---: |
| Prior fixed-shape direct WMMA | 64 × 128 | 1.077 |
| Paired Q4_K nibble loads with 16-byte writes | 64 × 128 | 1.068 |
| Vectorized Q8_1 activation staging | 64 × 128 | 1.016 |
| Both vectorized stages | 64 × 128 | 1.058 |
| Wider output tile | 128 × 64 | 1.251 |
| Wider output tile with both vectorized stages | 128 × 64 | 0.943 |
| Compact tile | 64 × 64 | 1.233 |
| Compact tile with both vectorized stages | 64 × 64 | 1.107 |
| Compact tile with both vectorized stages and K step 64 | 64 × 64 | 1.340 |

Both sub-1.05 ms candidates passed the full CPU tensor comparison.
The 128 × 64 prototype reached 0.990 and 0.977 ms in the repeated traces.
That prototype changed the generic tile as well and its two pp512 runs
fell to about 1051 tok/s. The 64 × 128 activation-vector candidate
reached 1.009 and 1.014 ms, with 1882.80 and 1874.54 tok/s in the same
comparison. The iteration 1 controls reached 1765.44 and 1750.81 tok/s.

The integrated 64 × 128 trial gave 1.015 and 1.065 ms. To gain margin,
the next trial scoped the 128 × 64 tile and both vectorized stages to the
target shape, preserving the generic tile for all other shapes. Its
three independent MMQ medians were 0.962, 0.995 and 1.019 ms. The
respective pp512 medians were 1876.73, 1864.78 and 1857.40 tok/s, compared
with control medians of 1747.80 and 1746.17 tok/s. The full CPU comparison
passed with a maximum absolute difference of approximately `1.3e-4`.

The production implementation uses the narrow 128 × 64 variant for the
measured shape on gfx1151, with 16-byte aligned packed weights and Q8_1
digits. The host grid is unchanged. The device maps each of its two x
halves to an independent 64-token half of a 128-token group. This covers
all 576 tiles exactly once. The original Q8_1 format, packed GGUF
weights, FP32 scale arithmetic and output layout carry over.
The generic path serves other shapes and device targets.

The native test suite now runs Q4_K MMQ at the exact dispatch shape and
rejects nonfinite errors before applying the existing CPU tolerance.
Raw experiment files are in the `retry-nov19/` subdirectory on AI.

## Final Release verification

The shipping source was rebuilt for gfx1151 in `retry-nov19/final-wide/`
using ROCm 10.1 and HIP 7.16.26385. The run verified the original GGUF
SHA-256. The native sweep used two warmups and five timed dispatches per
process. Model pp512 used F16 KV, chunk 512, one request, two warmups and
seven timed repetitions per process. Controls used the same iteration 1
Release build and model file.

| Trial | Pure MMQ trace median (ms) | Unprofiled pp512 median (tok/s) |
| --- | ---: | ---: |
| Control 1 | 1.29543 | 1764.33 |
| Candidate 1 | 0.94602 | 1868.22 |
| Candidate 2 | 0.98947 | 1869.35 |
| Control 2 | 1.32793 | 1763.04 |
| Candidate 3 | 0.98129 | 1859.38 |

All three candidate medians were below the 1.05 ms limit. The mean of
the first two matched pp512 run medians rose by 6.0%. The dispatch trace
reported 120 VGPRs versus 136 for the control, 30 KiB LDS, and no scratch.

Release CTest passed 9/9 with zero skips. The exact-shape test compared
all 4,718,592 outputs with the CPU reference and reported a maximum
absolute error of `1.14e-4`. A complete grid check independently
confirmed the coordinates of all 576 direct tiles.

The native sweep means including Q8_1 were 1.388 and 1.365 ms for the
controls and 0.971, 0.962 and 1.004 ms for the candidates. The power
profile was automatic. Benchmark boundary temperature readings were
46–61 °C. The native sweep uses seeded synthetic packed weights. The
prompt runs use the verified Qwen3.5-4B Q4_K_M GGUF.

The three-sample profile at position zero summed all 32 complete FFNs.
Its median fell from **144.568 ms to 125.297 ms**. The unprofiled controls
at 128, 2048 and 8192 used two warmups and five timed repetitions.

| Prompt | Control tok/s | Candidate tok/s |
| ---: | ---: | ---: |
| 128 | 1509.29 | 1503.52 |
| 2048 | 1708.62 | 1807.29 |
| 8192 | 1498.16 | 1571.10 |

Both pp128 runs rounded to 85 ms and use the generic dispatch.

`go build ./...`, the HIP CLI build, `go vet ./...`,
`go vet -tags hip ./...`, `go test ./...`, `go test -race ./...`, and
`go test -p 1 -tags hip ./... -count=1 -v` passed on AI. The HIP suite
passed the real-model long-context, CPU logits, greedy decode, speculative
and tiny-model logits tests. The three optional MoE and SentencePiece
tests skipped for lack of their separate fixtures.

## Acceptance and final review

Accepted. All three final pure MMQ medians meet 1.05 ms and the
unprofiled pp512 runs repeat the gain. Both the
[kernel contract](../../../kernels.md#gfx1151-direct-mmq-specialization) and the
[benchmark index](../../../benchmarks.md#prefill-prompt-length) link the
implemented path and measurements.

Three read-only review rounds total. The final implementation verdict was
`NO_BLOCKING_ISSUES`. After the final round, the opening kernel comment
was clarified to label the generic path's historical measurements.
The local source matched the tested AI source by SHA-256 after excluding
that comment edit. `git diff --check` and Markdown link checks passed.
