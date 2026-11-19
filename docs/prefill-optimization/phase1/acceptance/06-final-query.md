# Iteration 6 final query

## Implementation

The [prefill plan](../../../../engine/gpu/plan.go) runs final-layer K/V
projection, key normalization and RoPE, and the paged K/V write for
every prompt row. Subsequent operators enter the output tail. An
ordinary final chunk moves the last residual to row zero, projects its
normalized Q and gate, applies attention against the entire prefix,
and runs the output projection and FFN for one row. The speculative
all-logits plan keeps full-row execution.

The [ABI 7 descriptor](../../../../backend/hip/bridge/yuuka_hip.h) carries
token displacement in `ROPE_BATCH.u[0]` and
`ATTENTION_PREFILL_PAGED.n[5]`. The [executor](../../../../backend/hip/bridge/bridge.cpp)
checks the adjusted absolute positions before narrowing, and checks the
adjusted attention range against the KV capacity. It uses the parent
chunk's attention preference when computing the compact query. This
preserves FP16 MMA for short parent chunks such as pp128. The Q and
output projections likewise use their parent shape's dispatch.

## Correctness

All nine native tests passed. A compact attention query at dispatch
position 10 and displacement 5 matched the CPU at position 15 with
maximum absolute error 0.0000000447 and capacity 16. Increasing the
dispatch position by one failed. The Go RoPE test matched rotation at
position 5 and exercised signed position overflow rejection.

The [real-model test](../../../../engine/gpu/prefill_real_test.go) compared
the final logits, three greedy decode steps and full last-layer K/V
buffers against the full-row plan with F16 KV. It covered one token,
128 tokens, a full chunk, one- and three-token tails, two full chunks,
and separate nonfinal and final calls. All K/V buffers matched byte
for byte, the maximum final logit difference was 0.000003815, and all
three decode steps matched exactly. The all-logits fixture passed.

## Complete prompt timings

Both builds used the validated [AI setup and model](01-baseline.md#environment),
F16 KV, equal chunk sizes, one request, two warmups and one greedy output.
The automatic power mode was active, with pre-run temperatures of
42–61 °C. Reported rates are unprofiled process medians.

| Prompt | Chunk | Samples | Control tok/s | Compact query tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 5 | 2318.94 | 2359.41 | +1.75% |
| 2048 | 2048 | 5 | 2367.46 | 2380.10 | +0.53% |
| 8192 | 512 | 3 | 1909.98 | 1956.53 | +2.44% |
| 8192 | 2048 | 3 | 2009.43 | 2065.91 | +2.81% |

The first pp128 pair at chunk 128 gave 1870.39 versus 1643.49 tok/s.
Four interleaved repeats per build with seven samples and two warmups
gave 1864.55, 1873.62, 1862.85 and 1867.16 tok/s for the control, and
1853.31, 1823.25, 1867.01 and 1863.38 tok/s for the crop. The mean of
medians declined 0.82%. The large initial gap did not recur. A paired
pp128 trace showed a small reduction in last-layer time from 1.788 to
1.744 ms and attention time from 0.094 to 0.068 ms. That isolated
reduction is small relative to the measured whole-prompt variation.

## Output work per chunk

A diagnostic build profiled every chunk with actual populated K/V.
Each cell is the median of three complete runs following one warmup
on each build. The last-layer span starts at its input RMSNorm and ends
at its FFN residual. Total attention includes all full-attention layers.

| Prompt / chunk | Last-layer query rows | Last-layer FFN rows | Last-layer K/V rows | All K/V writes | Logit rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 512 | 512 → 1 | 1 → 1 | 512 → 512 | 8 → 8 | 1 → 1 |
| 8192 / 512 | 8192 → 1 | 1 → 1 | 8192 → 8192 | 128 → 128 | 1 → 1 |
| 8192 / 2048 | 8192 → 1 | 1 → 1 | 8192 → 8192 | 32 → 32 | 1 → 1 |

| Prompt / chunk | Last layer (ms) | All attention (ms) | Attention in last chunk (ms) |
| --- | ---: | ---: | ---: |
| 512 / 512 | 3.775 → 2.276 | 5.235 → 4.838 | 5.235 → 4.838 |
| 8192 / 512 | 148.141 → 11.711 | 930.855 → 797.644 | 123.825 → 107.688 |
| 8192 / 2048 | 119.510 → 10.955 | 699.865 → 606.826 | 335.798 → 289.128 |

On the crop path, every nonfinal chunk executed its final K/V write
and skipped final-layer attention, FFN and output logits.

## Verification

The Release build with `-DCMAKE_HIP_ARCHITECTURES=gfx1151`,
`ctest --test-dir build --output-on-failure -V`, Go build, the HIP CLI
build, `go vet ./...`, `go vet -tags hip ./...`, `go test ./...`,
`go test -race ./...` and
`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -count=1 -v`
passed on AI. The HIP suite exercised the full-row oracle, RoPE
displacement, tiny all-logits fixture, long context, CPU logits,
separate decode and speculative greedy checks. The optional MoE and
two SentencePiece tests skipped because their fixtures were absent.

The control is iteration 5 at `2d6969b`. Raw timings and diagnostic
output are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-06-MAsA5B/`
on AI.

## Review and decision

Accepted the final-query tail. The first read-only review requested the
whole-prompt trace, pp128 repeat and unified ABI descriptor documentation.
The second round returned `NO_BLOCKING_ISSUES` after those updates.
`git diff --check` passed.
