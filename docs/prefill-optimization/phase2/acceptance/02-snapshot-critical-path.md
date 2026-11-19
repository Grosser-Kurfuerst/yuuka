# Stage 5a and 5b prefix snapshot

## Environment and method

The Stage 5a control was `3caea06`. Both builds used the same gfx1151 Release
library on AI (Ryzen AI Max+ 395 / Radeon 8060S, Linux `7.0.14-28-pve`,
ROCm 10.1.0, HIP 7.16.26385, Go 1.24.4). The Qwen3.5-4B Q4_K_M GGUF had
SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
The GPU ran in automatic power mode with F16 KV, chunk 512, one request at
a time, greedy sampling, and HIP Graph disabled. Starting temperatures
for the 5a internal benchmark were 37–58 °C.

The [serving benchmark](../../../../bench/serving/main.go) used
`-ctx <prompt+64> -prompt-tokens <prompt> -decode-tokens 2 -concurrency 1
-warmup 3 -requests 11 -kv-type f16 -unique-prompts`. The control used the
same benchmark source and flags. Each request owned a prompt copy and its
first token was the request index across warmup and measured requests.
Start four processes in control, candidate, candidate, control order.
Internal results are medians of the two process p50 values per build.

For the [server](../../../../cmd/yuuka-server/main.go), both builds used
`--ctx 2176 --max-concurrency 1 --prefill-chunk 512 --kv-type f16`. The
loopback client sent streaming requests at temperature 0 with disabled
thinking and a four-token output limit. User messages began with
`Request nonce 1000. ` using sequential nonces, followed by 11, 50 or 202
repetitions of `The quick brown fox jumps over the lazy dog. ` and
`Summarize this in one word.`. Each process ran three warmups and nine
timed requests per length. SSE results pool the 18 timed samples per build
in the same process order. Time runs from HTTP POST to the first SSE
event with nonempty `delta.content`.

## Stage 5a capture gating

The [manager](../../../../serving/manager.go) tells the
[generation task](../../../../serving/task.go) whether it can retain a
prefix. Without a cache, the final chunk skips both recurrent-state
download and prompt-logit serialization. The
[ordering regression test](../../../../serving/task_test.go) covers
prefill, emission, decode and the absence of cache data.

Both timing arms disabled prefix caching.

| Internal prompt | Control TTFT (ms) | Candidate TTFT (ms) | Control total (ms) | Candidate total (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 140 | 94.473 | 91.347 | 121.560 | 117.905 |
| 530 | 288.408 | 278.254 | 322.476 | 310.659 |
| 2050 | 982.425 | 982.838 | 1036.566 | 1037.204 |

| Rendered SSE prompt | Control first content (ms) | Candidate first content (ms) | Latency saved (ms) |
| ---: | ---: | ---: | ---: |
| 138 | 96.243 | 93.485 | 2.758 |
| 528 | 282.469 | 278.241 | 4.228 |
| 2048 | 906.596 | 904.466 | 2.129 |

A temporary diagnostic control timed snapshot and serialization separately
inside the same requests. With three warmups and 15 measured requests
per length, median snapshot time was 2.556, 2.568 and 3.274 ms. Per-request
serialization medians were 0.138, 0.145 and 0.149 ms. The median combined
cost was 2.706, 2.702 and 3.466 ms respectively. An isolated
`testing.B` run over 248,320 logits also measured 0.094 ms per operation
across five repetitions.

The original 7 ms requirement came from older isolated snapshot data.
The owner approved acceptance against the observed cost after reviewing
the new data. Both SSE process pairs improved at every measured length.
The internal 2050-token p50 varied with prefill time and showed no
repeatable end-to-end gain.

### Stage 5a verification

- On AI, `go build ./...`, `go vet ./...`, `go test ./...`,
  `go test -race ./...`, and `go test -tags hip -v ./serving/...` passed.
  The HIP serving suite ran `TestGPUPrefixCacheMatchesFreshRun` and the
  concurrency, cancellation and allocation checks. CTest passed 10/10.
- Locally, `go test -race ./serving/... ./bench/serving` passed. A CPU
  tiny-fixture benchmark under the race detector used four concurrent
  requests, three warmups, 12 measured requests and cache enabled.
  It completed with zero hits and 15 misses.
- One independent read-only review returned `NO_BLOCKING_ISSUES`.
  The reviewer independently reduced the SSE samples and capture timings.
