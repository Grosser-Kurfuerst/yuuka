# Iteration 3 final FFN

## Implementation

The ordinary prefill plan places the final FFN in the output tail when
the last layer has full attention. The final chunk copies its last
residual into row zero and evaluates one row of the FFN and output head.
The FFN chooses its projection routes using the parent chunk's rows. The
all-logits plan and plans ending in a recurrent layer process every row.

- [Plan construction](../../../../engine/gpu/plan.go) — Tail placement, row compaction and projection dispatch.
- [Session](../../../../engine/gpu/session.go) — Cached last-row and all-logits modes.
- [HIP plan builder](../../../../backend/hip/hip.go) — Checked byte-offset copies using the existing native operation.
- [Tests](../../../../engine/gpu/prefill_real_test.go) — Full-row oracle, F16 KV, five chunk layouts, and greedy continuation.
- [Copy test](../../../../backend/hip/hip_test.go) — In-place disjoint copy, sentinel row, overlap and bounds validation.

## Arithmetic A/B

The reference is iteration 2 at `0e323b6`. The tests used the verified
Qwen3.5-4B Q4_K_M GGUF on the same [AI container](01-baseline.md#environment),
with F16 KV and chunk 512. The real-model test compared one token, a full
chunk, a one-token tail, two complete chunks, and separate nonfinal and
final calls. With the parent projection routes, the prompt logits and
three following decode steps matched the full-row reference exactly.
Both final-layer K/V buffers matched byte for byte, after clearing them
between runs. The all-logits fixture passed.

The single-row GEMV experiment changed the activation arithmetic of
K-quant projections. It exceeded the existing 0.05 logit tolerance
with a maximum difference of 0.08342 at 512 tokens and 0.06125 at 1024.
The implementation uses the parent routes, including Q8_1 for MMQ.

Raw artifacts are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-03-47UVL5/`
on AI.

## Whole-prompt timings

Both Release CLIs used the same stage 2 native library, verified GGUF,
F16 KV, one request and two warmups. The 512-token comparison used the
order control, candidate, candidate, control with seven timed samples
per process. Each 8K configuration used three timed samples. All figures
are unprofiled median tokens/s. Automatic power mode was active and
starting temperatures were 40–61 °C.

| Prompt | Chunk | Control | Cropped |
| ---: | ---: | ---: | ---: |
| 512 | 512 | 1891.15, 1884.73 | 1905.12, 1905.99 |
| 8192 | 512 | 1582.75 | 1607.79 |
| 8192 | 2048 | 1591.00 | 1607.31 |

The mean of the two pp512 medians improved by 0.93%. The observed 8K
medians improved by 1.58% at chunk 512 and 1.03% at chunk 2048.

## Tail profile

The diagnostic test in the run directory used HIP events for three
whole-prompt passes at chunk 512, from token position zero. The FFN
timings include its residual and, on the cropped path, the row copy.

| Prompt | Final FFN input rows | Final FFN median time | Last attention rows | Last K/V rows | Logit rows |
| ---: | --- | --- | ---: | ---: | ---: |
| 512 | 512 → 1 | 4.112 → 1.090 ms | 512 | 512 | 1 |
| 8192 | 8192 → 1 | 64.138 → 1.044 ms | 8192 | 8192 | 1 |

The operation trace retained all eight paged K/V writes and all eight
attention operations per chunk. Intermediate cropped chunks ended before
the FFN. The profile's instrumented times are diagnostic.

## Verification and review

- The final HIP CLI build and all nine CTest tests passed with zero skips.
  Both builds used the stage 2 native library.
- `go build ./...`, `go vet ./...`, `go vet -tags hip ./...`,
  `go test ./...`, `go test -race ./...` and
  `YUUKA_TEST_MODEL="$MODEL" go test -p 1 -tags hip ./... -count=1 -v`
  passed on AI. The HIP suite exercised the real-model long-context,
  decode, speculative and CPU parity checks. The optional MoE golden
  test and two SentencePiece tests skipped due to absent fixtures.
- The copy and exact real-model full-row comparisons passed in the
  final suite. `git diff --check` and the documentation link checks
  also passed.

Accepted after one read-only review round with verdict
`NO_BLOCKING_ISSUES`.
