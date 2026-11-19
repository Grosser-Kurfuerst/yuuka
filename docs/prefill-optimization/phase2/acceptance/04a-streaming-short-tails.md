# Stage 1b-1 FP32 streaming short tails

The [updated plan](../stage1bc-sub128-rows.md)
sets a 0.002 maximum CPU logit difference on the short-prompt
fixture and requires no regression at 16 tokens. This record
records the rejected cap experiments and the accepted staging change.

## Setup

The control uses the source of `260094f` and its Release HIP build
on AI (Radeon 8060S, `gfx1151`, ROCm 10.1.0, HIP 7.16.26385, and
Go 1.24.4). GPU power was automatic and one GPU process ran at a time.
The model and the [Phase 1 token fixture](../../phase1/acceptance/08-upstream-comparison.md)
had SHA-256 hashes:

- Qwen3.5-4B Q4_K_M `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`
- Token fixture `e613216db780a72fdbe796b31c63b4afabf273199dbc67f799b5c082d18e3e30`

The candidate lowered eligible `Q4_K`, `Q5_K`, and `Q6_K`
projections to the existing stream matvec plan operation, with
the 16-row cap for `nIn >= nOut`. Other dtypes, narrow outputs,
and the 128-row MMQ crossover kept their previous rules.

## Measured wide crossover

F16 KV, greedy top-1, two warmups and five timed samples per process
in [prefill-compare](../../../../bench/prefill-compare/main.go).
Two processes measured each route for each cell. The wide cap on
the candidate was four or eight.

| Prompt tokens | Control p50 (ms) | Cap 4 p50 (ms) | Cap 8 p50 (ms) |
| ---: | ---: | ---: | ---: |
| 4 | 56.632 | 32.338 | 32.251 |
| 8 | 49.458 | 42.367 | 41.394 |
| 12 | 54.685 | 47.001 | 46.985 |
| 16 | 55.247 | 49.355 | 49.912 |

Both mixed routes fail numerical acceptance at 16 tokens.
The real-model CPU comparison uses the same F32 or F16 KV format
in each pair and checks the complete prefill logit row, its top-1
and eight further greedy tokens. Where the wide 4- and 8-row
choices both preserve the wide FP16 WMMA path, they produce the
same 16-token maximum differences.

| Wide cap | Prompt tokens | F32 KV maximum absolute difference | F16 KV maximum absolute difference |
| ---: | ---: | ---: | ---: |
| 4 | 5 | 0.005983 | 0.005142 |
| 4 | 8 | 0.005491 | 0.005210 |
| 4 or 8 | 16 | 0.003724 | 0.004332 |
| 16 | 16 | 0.000489 | 0.001063 |

The 16-row shape ablation kept all tall shapes on streaming and
varied the wide groups by output count (4096, 8192, 9216).
With F16 KV, the best mixed cases reached 0.002104 and 0.002066.
Only streaming for all wide groups passed the 0.002 gate on both
KV formats. The candidate used shape and row count, with no prompt
or layer exceptions.

## Full-streaming experiment

The experiment selected the existing FP32 stream matvec for both
shape classes through 16 rows. It used the same HIP library as
control. The latency comparison ran control, candidate, candidate,
control with F16 KV, a 512-row chunk, three warmups and seven
measured prefills per process. The table pools the 14 samples per
build. Temperatures before each process were 55–67 °C.

| Prompt tokens | Control p50 (ms) | Full streaming p50 (ms) | Change |
| ---: | ---: | ---: | ---: |
| 2 | 85.020 | 28.796 | −66.13% |
| 16 | 55.372 | 56.449 | +1.95% |
| 64 | 80.818 | 80.867 | +0.06% |
| 127 | 119.076 | 119.032 | −0.04% |
| 128 | 68.479 | 68.772 | +0.43% |
| 512 | 219.795 | 220.087 | +0.13% |
| 513 | 293.957 | 240.052 | −18.34% |
| 2048 | 898.170 | 898.404 | +0.03% |
| 2050 | 994.426 | 941.651 | −5.31% |
| 8192 | 4128.249 | 4125.188 | −0.07% |

The same-chunk control comparison dumped the full prefill logit
row and recorded top-1 followed by eight generations using
`Session.Decode`. The logit maxima were 0.00066936 at 513
tokens and 0.00042534 at 2050 tokens. Both sequences matched
control at every step. This cross-binary comparison was performed
with a temporary diagnostic extension to `prefill-compare`.

The [serving benchmark](../../../../bench/serving/main.go) used
`-backend hip -kv-type f16 -ctx 640 -kv-blocks 72 -prompt-tokens 512
-decode-tokens 2 -concurrency 1 -warmup 3 -requests 9 -prefix-cache
-append-tokens 2`, checking that every timed request hit the exact
512-row prefix. The four process TTFT medians were 91.774, 35.485,
35.404 and 91.689 ms.

The server A/B used loopback, F16 KV, context 2200, chunk 512,
concurrency one and disabled prefix caching. Each process used two
warmups and nine timed requests with temperature zero, disabled
thinking, nonce values 1000–1010, and an eight-token completion
limit. The input was `Request nonce <nonce>. `, followed by 202
repetitions of `The quick brown fox jumps over the lazy dog. ` and
`Please now really do summarize this in one word.`. Every request
rendered exactly 2050 tokens. The client measured from HTTP POST to
the first SSE event with nonempty `delta.content`. The four process
medians were 1006.101, 942.766, 941.884 and 995.627 ms. Pooled
medians were 1000.093 and 942.437 ms.

The full-streaming candidate meets the tail, cache, SSE, and
unchanged-length latency gates. The 16-token latency regression
exceeds the plan's no-regression gate.

### Additional Go-only diagnostics

A second shape ablation separated the Q4_K and Q5_K versions of
`2560 → 8192`. With F16 KV, streaming the Q5_K version alongside
`2560 → 4096` and `2560 → 9216` measured 0.001714 versus CPU, but
the same-process diagnostic median rose from 55.475 to 55.982 ms.
Streaming the Q4_K version instead measured 54.348 ms and 0.002159.
Neither resolves both gates.

A temporary planner also tested the existing streaming residual
epilogue for attention outputs and FFN down projections, with all
wide projections streaming at 16 rows. It reused
`AddGemvResidualAt` for the projection plus residual sum. Its
four-process median was 56.538 ms versus 54.644 ms for control.
The prototype did not recover the 16-token latency regression.

## Verification before the staging change

- The full-streaming candidate passed `go test -tags hip ./backend/hip/...`,
  `go vet -tags hip ./...`, `go test -race ./...`, and all ten CTest
  suites on AI.
- `YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip -v ./...`
  passed against the tracked source tree. The initial copy included two
  untracked historical diagnostic tests that expected private fixtures;
  the clean run excluded those two files. Three tracked tests skipped
  for missing optional fixtures (one MoE and two SentencePiece).
- The short-prompt CPU test passed for all configured lengths and both
  KV formats on the full-streaming candidate. The 64-row attention
  comparison remained 0.0147166, the five-token golden test measured
  0.00004673, and real-model speculative verification accepted 24 of
  24 draft tokens. The candidate's golden value is recorded here for
  comparison with the deployed route.
- Two independent review rounds ended in `NEEDS_FIX`. The first review
  confirmed routing, non-K precedence and compact activation offsets.
  The candidate test was updated to propagate native session reset
  errors and rerun. The 513- and 2050-token full-vector differential
  was measured across binaries. Its permanent regression harness
  was added with the staging change below.

## Aligned 16-row activation staging

The approved 16-row cap applies to both shape classes. The native change
is in [lm_head.hip](../../../../kernels/hip/gemm/lm_head.hip). Each block
stages its complete aligned K-quant activation tile using `float4` loads.
A 16-row × 256-value tile takes four vector transfers per thread. The scalar
stager serves smaller and unaligned blocks. The dequantization, FP32 dot
products and wave reduction are shared.

The pilot used the same Go candidate against the original and new HIP
libraries. The original stager measured 56.16–56.23 ms at 16 tokens and
the vector stager measured 47.43–47.69 ms (process p50). The baseline
with its prior Go planner measured 54.76–55.25 ms.

### Reproduction

Build the control planner at `260094f` and the candidate as separate
Release gfx1151 builds. Use the candidate version of
[bench/prefill-compare](../../../../bench/prefill-compare/main.go) to
build the comparison binary in both trees. Only the benchmark runner
needs the added `-greedy-steps` flag in the control tree. Keep the control
wrapper bound to its native library:

```sh
#!/bin/sh
LD_LIBRARY_PATH="/path/to/control/build:/opt/rocm/lib" exec /path/to/control/prefill-compare "$@"
```

Run the real-model suite in the candidate tree with
`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16
YUUKA_TEST_PREFILL_CONTROL=/path/to/control-wrapper go test -p 1 -tags hip -v ./...`.
`TestGPUStreamingShortPrefillMatchesCPU` checks both KV formats and
`TestGPUStreamingTailsMatchControl` pins both processes to chunk 512 and
compares every prefill logit and nine greedy outputs. The control
subprocess enters and exits before the candidate runtime opens.

### Accepted performance run

This run used the same F16 KV, process order, warmups, repetitions,
prefix-cache checks and rendered 2050-token server input as the first
full-streaming experiment. GPU temperature before process starts was
41–66 °C. The Go candidate and final native stager ran together.

| Prompt tokens | Control p50 (ms) | Candidate p50 (ms) | Change |
| ---: | ---: | ---: | ---: |
| 2 | 84.805 | 29.349 | −65.39% |
| 16 | 54.909 | 47.910 | −12.75% |
| 64 | 80.761 | 80.836 | +0.09% |
| 127 | 116.700 | 116.599 | −0.09% |
| 128 | 68.008 | 68.316 | +0.45% |
| 512 | 216.760 | 216.126 | −0.29% |
| 513 | 292.453 | 238.689 | −18.38% |
| 2048 | 902.530 | 899.412 | −0.35% |
| 2050 | 994.276 | 936.828 | −5.78% |
| 8192 | 4119.515 | 4109.407 | −0.25% |

The verified cached `512+2` request TTFT medians were 91.867, 35.828,
35.778 and 91.547 ms. At 2050 tokens, the SSE medians per process
were 1004.379, 938.743, 942.901 and 991.705 ms. Pooled SSE medians
were 996.531 and 940.279 ms. Every server log reported 2050 input
tokens for all requests.

### Accepted correctness run

- The full model CPU comparison passed for both KV formats and all
  configured prompts. At 16 tokens the maximum was 0.00048923 with
  F32 KV and 0.00106335 with F16 KV. The golden five-token model
  difference was 0.00004673.
- The permanent control comparison returned 0.00066936 at 513 and
  0.000425339 at 2050 tokens with the same nine greedy tokens. The
  64-token attention gate passed at 0.0147166 and speculative prefill
  accepted all 24 draft tokens.
- The full real-model HIP suite with the control wrapper, native CTest
  (10/10), HIP dispatch tests, `go build ./...`, both Go vet variants
  and `go test -race ./...` passed on AI. The MoE and two SentencePiece
  golden tests skipped for missing optional fixtures.
- The third independent review returned `NO_BLOCKING_ISSUES` on the
  approved scope. The two earlier rounds covered the rejected caps.
  Following the final review, the test explicitly pins the candidate
  chunk to 512 and isolates each baseline output file. The full
  suite above ran after those two test changes.
