# Stage 3B Direct Q6_K attention V

## Implementation and controls

The [Stage 3 plan](../stage3-q6k-direct-mmq.md) adds only the measured
`2560 → 1024` Q6_K attention V shape to the Direct MMQ dispatcher.
It uses the same packed kernel as [Stage 3A](07a-direct-q6k-ffn.md).
The shape covers five V projections in the verified Qwen3.5-4B
Q4_K_M GGUF. FFN down stays enabled independently. The launcher
also requires at least 128 rows and 16-byte alignment of weights
and activation digits. The original route covers smaller rows,
other shapes and packed pointers without that alignment.

The measurements use AI gfx1151, ROCm 10.1.0, HIP 7.16.26385, Go
1.24.4 and automatic power. The [model and token hashes](05-cli-chunk-policy.md#baseline-and-protocol)
and model settings match Stage 3A. Stage 2 `2bfabbb` is the cumulative
parent. Stage 3A `bb56fec` isolates the V dispatch in the microbenchmark.

## Native tensor and dispatch verification

The full CPU outputs, repeated device outputs and trailing sentinels
matched for both Q6_K shapes at 127, 128, 129, 192, 511 and 512 rows.
Both 129-row weight-offset-two requests passed against the same CPU
tensors. All 14 native comparisons reported max absolute error zero
against the `2e-4` gate. The 127-row and offset cases exercise the
generic route. The ineligible `768 → 70` test also remains in the
native matrix. Complete tiles and masked tiles report 110 and 112
VGPRs, zero scratch and zero VGPR spills. Dynamic LDS is 30,720 bytes.
Release CTest passed 10/10.

`rocprofv3` traced the native suite with the MMQ kernel filter.
The V shape had a 4096-thread X grid and the FFN down shape a
10,240-thread X grid. For each, the 127-row control used the
generic Q6_K template with Y grid 1, and offset two used the
generic template with Y grid 2. The aligned 129 and 192-row
cases used the masked Direct template at Y grid 2. The ineligible
70-channel geometry also ran the generic template.

## Isolated V sweep

The same `yuuka_prefill_sweep --direct-mmq` build ran on the three
libraries in parent/FFN/both/both/FFN/parent order. The activation
quantizer runs before timing. Each process reports nine submissions
after two warmups and includes host submit/wait. Each arm below is
the median of two process means, in milliseconds.

| Token rows | Stage 2 | Stage 3A | Both shapes | Reduction vs Stage 2 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.100656 | 0.154499 | 0.071357 | 29.11% |
| 129 | 0.129314 | 0.138421 | 0.099058 | 23.40% |
| 192 | 0.144440 | 0.144802 | 0.120667 | 16.46% |
| 511 | 0.207204 | 0.206168 | 0.157130 | 24.17% |
| 512 | 0.207223 | 0.207697 | 0.162198 | 21.73% |
| 1024 | 0.389725 | 0.394262 | 0.279266 | 28.34% |
| 2048 | 0.727032 | 0.744500 | 0.511128 | 29.70% |
| 4096 | 1.421158 | 1.416167 | 1.051659 | 26.00% |

The 128-row Stage 3A V control had 0.208255 and 0.100743 ms
process means. The same generic path at Stage 2 measured 0.099843
and 0.101468 ms. Both Direct process means beat the faster controls.

## Cumulative model A/B against Stage 2

The [matched-token driver](../../../../bench/prefill-compare/main.go)
used identical chunks and F16 KV. It ran control/both/both/control
and then both/control/control/both, two warmups and five samples per
process (three at 8192). The numbers below are medians and complete
ranges in milliseconds. All runs were unprofiled. The whole Stage 3
result includes the contribution from FFN down.

| Prompt | Chunk | A control [range] | A both [range] | A reduction | B control [range] | B both [range] | B reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 64.145 [63.862–64.864] | 63.409 [62.604–65.595] | 1.15% | 64.356 [64.056–64.545] | 63.432 [62.899–64.796] | 1.44% |
| 512 | 512 | 213.940 [211.069–218.869] | 203.367 [202.668–208.970] | 4.94% | 211.774 [210.045–216.473] | 207.353 [203.403–211.081] | 2.09% |
| 2048 | 1024 | 846.489 [841.812–863.047] | 825.926 [815.844–875.474] | 2.43% | 844.178 [840.682–851.000] | 820.232 [815.914–850.418] | 2.84% |
| 8192 | 1024 | 3867.984 [3854.905–3875.873] | 3762.003 [3745.229–3764.541] | 2.74% | 3857.004 [3850.618–3866.927] | 3743.555 [3738.273–3753.453] | 2.94% |

Every prefill logit dump and all nine greedy IDs matched across both
process orders. The nonaligned dispatch smoke used one warmup and
three samples per process at chunk 512. At 129 tokens the median was
85.226/80.927/80.724 ms for Stage 2/3A/3B. At 511 tokens it was
216.424/206.815/206.822 ms. These logits and greedy sequences matched
across all three builds.

## Q4_K and Q5_K controls

The 64 Q4_K/Q5_K cells in the full direct sweep had a median
change of +0.05% versus Stage 3A. The six largest apparent
slowdowns were repeated in isolated cells with the same fixture
and run protocol.

| Dtype | Input → output | Rows | Stage 3B latency change on repeat |
| --- | --- | ---: | ---: |
| Q4_K | 9216 → 2560 | 511 | +1.29% |
| Q4_K | 2560 → 1024 | 192 | +0.01% |
| Q5_K | 4096 → 2560 | 129 | +0.41% |
| Q4_K | 9216 → 2560 | 192 | +1.67% |
| Q4_K | 2560 → 9216 | 129 | +3.21% |
| Q4_K | 9216 → 2560 | 2048 | +1.04% |

A further reversed-order repeat of the 129-row wide cell yielded
0.372373 ms for Stage 3A and 0.369810 ms for Stage 3B. Its sign
flipped to −0.69%. The sweeps show run-to-run variation in the
controls. The host dispatch change is confined to Q6_K.

## Acceptance and review

Go build, vet, test and race, HIP build and vet, and the full
serial real-model HIP suite passed on AI. Both original long F16
attention and narrow projection regression tests ran. The two
optional Step 1b control wrappers, MoE model and two SentencePiece
fixtures were unavailable. The four cumulative prompt lengths
improved in both orders, and the isolated V route passed the
numerical, dispatch and resource gates.

One independent read-only review returned `NO_BLOCKING_ISSUES`.
The raw CTest, sweep, trace, model and control logs are under
`/tmp/yuuka-phase2-5-11.3wZQZG/3b` on AI.
