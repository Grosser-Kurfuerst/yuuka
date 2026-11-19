# Stage 6a Prepared Q8_1 GEMV for the compact output tail

The [Stage 6 plan](../stage6-compact-tail-and-norm.md) compares the
prepared single-row GEMM and GEMV for the five cropped projections.
The production control is `8d50b1a` with the same kernels as
`e4cde6f`. The [reference model and tokens](05-cli-chunk-policy.md#baseline-and-protocol)
run on AI gfx1151 with ROCm 10.1.0, HIP 7.16.26385 and Go 1.24.4.
Clocks and power are automatic. The paired trials use Release builds,
F16 KV, greedy output and one active request.

## Route and tensor correctness

The planner recognizes the explicitly compacted final full-attention
layer within the normal last-logit plan. The parent chunk still decides
which products use MMQ. The query quantizes the selected last FP32
normalized row into zero-based Q8_1 scratch. The output projection
quantizes the compact attention output. Gate and up consume the same
one-row Q8_1 buffer, and down consumes the fused SwiGLU Q8_1 result.
Full K/V writes and the final query's absolute displacement run as
before. Non-final chunks skip the marked tail.

The plan audit finds exactly five `GEMV_Q8_1` operations in the tail
for parents 128, 512, 1024 and 2048. The baseline has five GEMMs there.
Both arms use three explicit quantizations plus one fused SwiGLU/quant
operation. Parents 1, 16 and 127 have no `GEMV_Q8_1` in either arm, and
an all-logit four-row plan has none. A non-final 512-row step followed
by a true 17-row remainder yields identical hashes for the recurrent
snapshot, populated K/V bytes, and remainder logits.

The integer CPU oracle multiplies signed activation digits by
unpacked weight digits and applies the block scales and minimum
corrections. GEMM and GEMV read the same digits and scales.
All five tested shape/dtype cells pass
`2e-4 * (1 + abs(reference))`. The initial trial's largest CPU
difference is `1.37e-5` and its largest difference between paths
is `3.05e-5`.

## Default GEMV trial

The baseline and first GEMV candidate run two paired orders,
old/new/new/old then new/old/old/new. Each process uses two warmups
and five samples. Chunks for the four lengths are 128, 512, 1024
and 1024. Times are medians with full sample range. The fixed
saving is control p50 minus candidate p50. Each pair compares the
final full logit row and three greedy continuation tokens.

| Tokens | Order | GEMM p50 [range], ms | GEMV p50 [range], ms | Saving, ms | Max logit delta |
| ---: | --- | --- | --- | ---: | ---: |
| 128 | A | 62.749 [61.982–63.710] | 61.661 [61.206–62.136] | 1.088 | 0.028345 |
| 128 | B | 62.479 [62.271–62.882] | 61.609 [61.253–61.993] | 0.870 | 0.028345 |
| 512 | A | 201.700 [200.932–202.312] | 199.701 [199.125–203.822] | 1.998 | 0.000006 |
| 512 | B | 201.459 [200.515–203.455] | 200.538 [199.338–203.885] | 0.921 | 0.000006 |
| 1024 | A | 397.995 [396.578–403.781] | 396.554 [394.740–401.334] | 1.441 | 0.000008 |
| 1024 | B | 399.665 [397.918–405.462] | 398.189 [396.254–401.093] | 1.476 | 0.000008 |
| 2048 | A | 811.053 [806.841–825.033] | 814.197 [812.449–830.010] | -3.144 | 0.000006 |
| 2048 | B | 819.438 [813.694–836.241] | 816.619 [812.723–826.051] | 2.819 | 0.000006 |

The default tile passes the 0.05 real-model logit gate. The
saving falls below 1.0 ms in the reverse order at 128 and 512.
The 2048-token results vary in sign.

## Compact-tail tile sweep

An isolated HIP probe interleaves eight workgroup shapes in two
orders. All use the same populated packed weights, Q8_1 digits
and block scales. The probe checks finite outputs and bitwise
equivalence for every tile. It uses two warmups and 21 samples per
shape and order. The table reports median kernel microseconds by
order. The production decode mapping is eight waves with four
channels per wave.

| Shape | Dtype | Original 8×4 | Tuned candidate | Tuned time |
| --- | --- | --- | --- | --- |
| 2560 → 8192 | Q4_K | 46.57 / 46.09 | 8×8 | 44.72 / 44.44 |
| 4096 → 2560 | Q4_K | 33.62 / 33.74 | 8×2 | 23.12 / 22.80 |
| 4096 → 2560 | Q5_K | 35.02 / 34.98 | 8×2 | 24.45 / 24.16 |
| 2560 → 9216 | Q4_K | 46.89 / 47.61 | 8×8 | 46.37 / 46.45 |
| 9216 → 2560 | Q6_K | 81.79 / 83.15 | 8×2 | 54.50 / 55.06 |

The tuned trial changes workgroup mapping only for these prefill
tail shapes. Each output channel has the same summation order.
The backend distinguishes the prefill tail through the existing
tail flag. The regular decode op retains its original tile.

## Tuned model trial

The tuned binaries rerun the same two process orders with two
warmups and seven samples per process. All final logits are
finite, all max deltas pass 0.05, and all four greedy IDs match.

| Tokens | Order | GEMM p50 [range], ms | Tuned p50 [range], ms | Saving, ms | Max logit delta |
| ---: | --- | --- | --- | ---: | ---: |
| 128 | A | 62.664 [62.472–63.008] | 61.472 [61.332–62.169] | 1.192 | 0.028345 |
| 128 | B | 62.765 [62.450–63.136] | 61.593 [61.265–64.644] | 1.173 | 0.028345 |
| 512 | A | 202.647 [201.056–207.477] | 201.711 [201.282–207.547] | 0.937 | 0.000006 |
| 512 | B | 203.030 [202.672–208.428] | 202.701 [201.138–207.014] | 0.328 | 0.000006 |
| 1024 | A | 402.543 [400.266–409.578] | 400.883 [397.846–406.635] | 1.660 | 0.000008 |
| 1024 | B | 402.909 [399.951–409.936] | 400.388 [398.693–411.248] | 2.521 | 0.000008 |
| 2048 | A | 820.283 [813.706–832.841] | 818.109 [812.577–829.641] | 2.173 | 0.000006 |
| 2048 | B | 823.747 [819.460–841.310] | 822.029 [816.585–836.183] | 1.718 | 0.000006 |

The instrumented median for the five projections moves from
1.502 to 0.486 ms at 128 rows and from 1.622 to 0.504 ms at
512. The complete marked tail moves from 4.538 to 3.508 ms and
from 4.772 to 3.595 ms. Profiling introduces device events between
operations. These results attribute the savings to the five
projections.

For a tighter whole-prefill pair, one runtime builds and caches
both plans before measurement. It resets the recurrent state and
alternates the plan used by the normal session prefill entry
point. The model, token IDs, allocations and HIP library are
shared. Two starting orders use two warmups and 15 unprofiled
samples per arm.

| Tokens | Order | GEMM p50 [range], ms | Tuned p50 [range], ms | Saving, ms |
| ---: | --- | --- | --- | ---: |
| 128 | 1 | 62.739 [62.473–66.792] | 61.761 [61.496–62.362] | 0.977 |
| 128 | 2 | 62.854 [62.645–65.446] | 61.897 [61.665–62.601] | 0.958 |
| 512 | 1 | 205.900 [205.041–211.098] | 205.086 [204.131–208.463] | 0.814 |
| 512 | 2 | 206.684 [205.845–212.692] | 206.568 [204.797–240.638] | 0.116 |

## Decision and verification

Keep the production compact row on the existing GEMM. The tuned
inter-process series passes the fixed-saving threshold at 128,
1024 and 2048. At 512 the savings are 0.937 and 0.328 ms. The
same-process pairs fail to reproduce 1.0 ms at both tested
lengths. Stage 6b starts with the accepted compact-tail GEMM policy.

The tuned trial passes Release CTest (10/10), HIP Go build and
vet, the native dot products, the real-model final-prefill
short/displaced/multi-chunk/separate-prefix cases, the 8K
chunked-prefill regression, and real and tiny speculative
verification. The three non-final-state hashes match the control
in both trial builds.

The experiment root is `/tmp/yuuka-phase2-5-11.3wZQZG/6a` on AI.
The default and tuned sources are under `src` and `tuned/src`.
Use `-kv-type f16 -greedy-steps 3` with the matched-token compare
driver, explicit chunk, and the corresponding Release library.
Run `go test -tags hip ./engine/gpu -run '^TestStage6ATailAudit$' -count=1 -v`
with `YUUKA_TEST_MODEL` and `YUUKA_6A_CANDIDATE=1` for the two
GEMV candidates. Omit the latter for the original plan.
`mv_probe.hip` and `mv_probe.csv` hold the isolated tile results.
`tuned/paired-src` and `tuned/cached.log` hold the two-plan
diagnostic. Run `go test -tags hip ./engine/gpu -run '^TestStage6APairedCachedPlans$' -count=1 -v`
from that tree. The diagnostic tokenizes its own repeated text,
and both cached plans use those same IDs. The test-only toggle
is confined to that tree.

The read-only review ended with `NO_BLOCKING_ISSUES` after one round.
