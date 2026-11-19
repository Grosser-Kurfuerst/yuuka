# Stage 1b short-chunk projections

This record covers the [Stage 1b plan](../stage1bc-sub128-rows.md).
Control is `260094f` with the [Stage 1c prepared recurrence](03-sub128-gdn.md).
Stage 1b is blocked on numerical acceptance. The experimental
projection dispatch has not landed.

## Environment and measurement

AI container, Radeon 8060S (`gfx1151`), ROCm 10.1.0, HIP 7.16.26385,
Go 1.24.4, Release HIP library and automatic power. One GPU workload
ran at a time. Model was Qwen3.5-4B Q4_K_M, SHA-256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
Model A/B used F16 KV, a 512-row chunk, the [Phase 1 benchmark token
fixture](../../phase1/acceptance/08-upstream-comparison.md), and greedy
continuations. Per-row-count benchmarks used three warmups and seven
timed prefills per process in control, candidate, candidate, control
order. The time includes reading the final logits to the host.

## Route crossover

An isolated build temporarily removed the 128-row native Direct floor.
The existing [sweep](../../../../bench/hip/prefill_sweep.cpp) ran
`--sub128` (including quantization in the MMQ path) and
`--direct-mmq --sub128` (prequantized MMQ) three times. The table reports
medians of the three process medians in microseconds. Each entry gives
FP32 streaming / FP16 WMMA / Q8_1 MMQ. Streaming covers at most 16 rows.
The Q4_K and Q5_K entries use Direct MMQ, and the Q6_K entries use
Generic MMQ.

| Weight | Input → output | 4 rows | 8 rows | 16 rows | 18 rows | 64 rows | 127 rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2560 → 1024 | 39 / 94 / 71 | 54 / 87 / 70 | 94 / 105 / 71 | — / 117 / 71 | — / 134 / 76 | — / 141 / 76 |
| Q4_K | 2560 → 4096 | 78 / 115 / 100 | 109 / 115 / 100 | 172 / 125 / 101 | — / 135 / 100 | — / 249 / 104 | — / 296 / 148 |
| Q4_K | 2560 → 8192 | 119 / 150 / 146 | 155 / 154 / 146 | 250 / 164 / 147 | — / 166 / 147 | — / 313 / 145 | — / 668 / 265 |
| Q4_K | 2560 → 9216 | 118 / 158 / 148 | 164 / 165 / 149 | 254 / 175 / 148 | — / 176 / 146 | — / 361 / 149 | — / 692 / 273 |
| Q4_K | 4096 → 2560 | 69 / 143 / 103 | 94 / 147 / 108 | 136 / 172 / 106 | — / 181 / 105 | — / 222 / 112 | — / 277 / 145 |
| Q4_K | 9216 → 2560 | 124 / 291 / 198 | 173 / 303 / 199 | 262 / 361 / 200 | — / 383 / 200 | — / 466 / 209 | — / 604 / 290 |
| Q5_K | 2560 → 8192 | 135 / 154 / 151 | 172 / 162 / 151 | 264 / 173 / 150 | — / 170 / 151 | — / 323 / 157 | — / 676 / 276 |
| Q5_K | 4096 → 2560 | 76 / 153 / 107 | 102 / 159 / 106 | 145 / 185 / 107 | — / 195 / 105 | — / 233 / 113 | — / 293 / 149 |
| Q6_K | 2560 → 1024 | 38 / 102 / 96 | 53 / 106 / 99 | 91 / 120 / 99 | — / 130 / 102 | — / 147 / 106 | — / 162 / 111 |
| Q6_K | 9216 → 2560 | 110 / 357 / 380 | 170 / 374 / 380 | 238 / 430 / 384 | — / 449 / 377 | — / 547 / 399 | — / 775 / 422 |

The initial candidate keeps the FP32 streaming route through 16 rows
for eligible K-quant matrices. In this range the FFN's streaming gate
and up use separate operations, and speculative verification retains
FP32 activations. It enables Direct and the two measured Q6_K Generic
shapes from 17 rows. The isolated 18-row cells and the model's 17-row
A/B test bracket the transition. The candidate keeps the F32, F16 and
Q8_0 dispatch rules.

## Initial numerical results

The full FP32 output logit row was compared after prefill. The same
top-1 and all eight subsequent greedy tokens were observed in each
pair below. The CPU path uses the independent FP32 forward pass with
the same F16 KV storage format.

| Prompt tokens | Candidate vs same-chunk control | Control vs CPU | Candidate vs CPU |
| ---: | ---: | ---: | ---: |
| 1 | 0.00005364 | 0.00014836 | 0.00014108 |
| 2 | 0.00006962 | 0.00009966 | 0.00013208 |
| 5 | 0.00518596 | 0.00546932 | 0.00063860 |
| 16 | 0.00683498 | 0.00700855 | 0.00106335 |
| 17 | 0.197486 | 0.007043 | 0.194796 |
| 64 | 0.650768 | 0.021644 | 0.657772 |
| 96 | 0.611968 | — | — |
| 127 | 2.805252 | — | — |
| 513 | 0.000669 | — | — |
| 530 | 0.308279 | — | — |
| 600 | 0.301544 | — | — |
| 2050 | 0.000425 | — | — |

The 127-row difference was broad. Its RMS was 0.522533 and 139,001
logits differed by more than 0.3. The 530- and 600-token cases exceeded
0.3 on two and one logits, respectively.

Comparing a single chunk with a 512-row chunk plus the tail yielded
the same top-1 at 513 and 530 tokens. Maximum differences were
1.067446 and 0.308279 for control, and 1.066872 and 0 for the candidate.

FP32 KV corroborated the Q8_1 difference at 64 rows (0.657094 versus
CPU). The 1- and 2-row streaming runs also slightly increased the
FP32 KV errors, from 0.00006485 to 0.00006640 and from 0.00005007
to 0.00005102, respectively. In an isolated 1-row trial, preserving
batched GEMV for FFN projections gave deviations of 0.00014544 with
F16 KV and 0.00006187 with FP32 KV. In a 2-row trial, preserving
batched GEMV for the `4096 → 2560` output projections gave 0.00009823
with F16 KV and 0.00004911 with FP32 KV. These are fixture-scoped
results for potential streaming-only routing.

### Isolating the Q8_1 difference

The same Go candidate loaded the control HIP library, which
selects Generic MMQ for actual batches below 128 rows. At 17, 64,
96, and 127 rows all logits were byte-identical to the Direct-enabled
build, with the same continuation. This isolates the numerical change
to the switch from FP16/FP32 projection activations to Q8_1 and the
resulting downstream computations. The partial Direct tile does not
explain these differences.

Eleven temporary projection ablations allowed or excluded MMQ by
shape or dtype while retaining the previous path for the affected
matmuls. The smallest 127-row difference was 2.412715 when
`n_out = 1024` projections kept the prior path. The Q6_K-only and
FFN-only variants gave 4.896219 and 7.297421, respectively. At 96
rows even the lowest result was 0.416036 (Q6_K-only). The
`2560 → 8192` and `2560 → 4096` FP16 input projection ablation
reduced the 64-row difference against control to 0.192466. That
isolated improvement does not resolve the 96- and 127-row cases.

`TestGPUAttentionPathsAgree` also failed with the candidate. The
64-row MMA-versus-tiled comparison rose from 0.0147166 on control to
0.495347, above the existing 0.05 gate. Top-1 stayed 33075. Both
attention arms used the same short-row projections, so the result
shows that the new projection arithmetic amplifies differences
between the attention routes. The proposed max-absolute logit bound
of 0.3 could not be confirmed. The 5-token historical
`TestGPUMatchesCPU` value of 0.00309944 was taken on the prior short
FP16/FP32 projection routes, where the MMQ threshold was 128.

## End-to-end timing for the initial candidate

The table pools the 14 samples per build. Temperature before
process starts was 54–66 °C. Prefill latency is exclusive of
model loading and includes the final logits.

| Prompt tokens | Control p50 (ms) | Candidate p50 (ms) | Change |
| ---: | ---: | ---: | ---: |
| 2 | 85.037 | 28.876 | −66.04% |
| 17 | 57.858 | 42.200 | −27.06% |
| 18 | 58.721 | 42.285 | −27.99% |
| 64 | 80.975 | 47.459 | −41.39% |
| 96 | 107.659 | 66.082 | −38.62% |
| 127 | 118.797 | 69.291 | −41.67% |
| 128 | 68.353 | 68.435 | +0.12% |
| 512 | 217.962 | 219.465 | +0.69% |
| 513 | 295.581 | 240.860 | −18.51% |
| 530 | 276.321 | 259.920 | −5.94% |
| 600 | 323.501 | 285.771 | −11.66% |
| 2048 | 902.096 | 904.908 | +0.31% |
| 2050 | 994.044 | 938.834 | −5.55% |
| 8192 | 4123.330 | 4115.523 | −0.19% |

The [serving benchmark](../../../../bench/serving/main.go) built
separately against each planner. It checked the exact 512-token hit
on every extension, using `-backend hip -kv-type f16 -ctx 640
-kv-blocks 72 -prompt-tokens 512 -decode-tokens 2 -concurrency 1
-warmup 3 -requests 9 -prefix-cache -append-tokens 2` and then 18.
Each server used `-kv-type f16 -ctx 2200 -prefill-chunk 512
-max-concurrency 1 -prefix-cache=false` on loopback. Requests used
temperature zero, disabled thinking and up to eight output tokens.
The server verified the rendered lengths on all requests. The chat
prompt was `Request nonce 1000. ` (incremented per request), 50 or
202 repetitions of `The quick brown fox jumps over the lazy dog. `,
and `Please now really do summarize this in one word.`. SSE timing
ends at the first nonempty `delta.content`. Each process used two
warmups and nine timed requests per length.

| Scenario | Control process p50 (ms) | Candidate process p50 (ms) |
| --- | ---: | ---: |
| Cached 512 + 2 | 91.666 / 91.615 | 35.193 / 35.190 |
| Cached 512 + 18 | 67.168 / 67.026 | 48.990 / 48.945 |
| SSE, 530 tokens | 281.156 / 283.586 | 264.495 / 262.986 |
| SSE, 2050 tokens | 989.717 / 993.527 | 939.560 / 939.943 |

The end-to-end latency gates passed for this candidate. Its numerical
gates still block acceptance.

## Route and regression checks

- `go test -tags hip ./backend/hip/...` passed under the candidate.
  The selector checks covered the two row bands for each profiled
  dtype and projection shape, the prior non-K routes and the
  default dispatch for other GPUs.
- On the normal plan, `rocprofv3 --kernel-trace true -f rocpd`
  recorded the following. A temporary benchmark build called
  libc `exit` after normal return to flush the profiler.

  | Rows | Streaming projections | Q4_K/Q5_K Direct MMQ | Q6_K Generic MMQ | Compact Q4_K Generic MMQ |
  | ---: | ---: | ---: | ---: | ---: |
  | 2 | 200 | 0 | 0 | 0 |
  | 18 | 0 | 175 | 21 | 4 |
  | 64 | 0 | 175 | 21 | 4 |
  | 127 | 0 | 175 | 21 | 4 |

  The table excludes the single streaming LM head dispatch. The
  48 narrow Q8_0 projections used batched GEMV at 2 rows and their
  WMMA path at 18, 64 and 127. The compact projections inherit the
  Q8_1 arithmetic and use the existing Generic kernel at actual M=1.
- Release CTest passed 10/10. In the isolated candidate tree, the native
  sentinel experiment at
  17, 63, 64, 65 and 127 rows matched the complete Direct tile
  reference for valid rows and preserved every trailing sentinel.
  The real-model compact-versus-full-row tests at 17, 64 and 127,
  including KV bytes and three decode steps, also passed with their
  existing tolerance.
- The full real-model suite ran with
  `YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip -v ./...`.
  `TestGPUMatchesCPU` passed (first-step maximum difference
  0.00004673 in the candidate). The hidden-state, chunked-prefill,
  tiny fixture and real-model speculative tests also passed.
  Speculative verification accepted 24 of 24 draft tokens.
  `TestGPUAttentionPathsAgree` failed as detailed above. The missing
  separate MoE and SentencePiece fixtures caused three skips.
- The untagged `go test -race ./...` passed on AI. A local attempt
  hit the sandbox's socket restriction in the HTTP server package.
  Local Go build, vet and Markdown link checks passed.

Two independent read-only reviews ended in `NEEDS_FIX`. The first
verified K-quant-only routing, the matching Go and native allowlists
and compact tail arithmetic. The Direct-versus-Generic comparison
and isolated native and compact model checks addressed its boundary
test findings. The second confirmed the unresolved numerical
blocker and the need for a scope decision. The numerical gates
and the active-path benchmark values retain their prior settings.

## FP32-only short-tail trial

This isolated trial leaves the projection arithmetic unchanged from
17 rows onward. It uses streaming FP32 for eligible K-quants through
16 rows, with the existing batched GEMV for FFN projections at one
row and for the `4096 → 2560` projections at two rows. These
exceptions were tested against both KV formats on the original prompt.

With the Phase 1 inputs at 1, 2, 5 and 16 tokens, maximum errors
against the CPU were at or below control with both KV formats,
with the same top-1 and eight greedy continuation steps. The F16 KV
513- and 2050-token comparisons against the same-chunk control
had maximum differences of 0.000776 and 0.000306, with the same
greedy tokens. The 17-, 64-, 96-, 127-, 530- and 600-token F16 KV
runs matched the control logits byte for byte. The targeted real-model
`TestGPUAttentionPathsAgree`, `TestGPUMatchesCPU` and speculative test
passed. These results are for the isolated prototype.

Single-process medians with three warmups and seven samples were
57.316 ms at one row, 33.518 ms at two rows, 273.737 ms for 513
tokens, and 927.479 ms for 2050 tokens. A verified cached 512 + 2
run had 40.058 ms TTFT p50 versus about 91.6 ms in the earlier
interleaved control. This exploratory performance run has less
replication than the original four-process A/B. The original 64–127
cliff and the 18- and 88-row tails require a separate projection
solution if this trial is split off.

Two additional F16 KV inputs checked whether the one- and two-row
exceptions generalized. Each input repeated a sentence about a falling
leaf (English) or a prompt about planting lettuce (Chinese). All pairs
agreed on top-1 and the eight subsequent greedy tokens. Maximum
absolute differences against CPU and the matched-chunk control were
as follows.

| Input | Rows | Reference | Control vs reference | FP32 trial vs reference |
| --- | ---: | --- | ---: | ---: |
| English | 1 | CPU | 0.000121117 | 0.000124693 |
| English | 2 | CPU | 0.000106812 | 0.000120044 |
| Chinese | 1 | CPU | 0.000130177 | 0.000116110 |
| Chinese | 2 | CPU | 0.000151873 | 0.000156879 |
| English | 512 + 1 | Control | 0 | 0.000473976 |
| English | 512 + 2 | Control | 0 | 0.000378609 |
| Chinese | 512 + 1 | Control | 0 | 0.000456691 |
| Chinese | 512 + 2 | Control | 0 | 0.000522256 |

The new prompts show that the fixture-selected exceptions are insufficient
for a strict no-increase CPU error condition across inputs. The FP32 trial
has no production threshold or acceptance gate.
