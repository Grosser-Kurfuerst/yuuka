# Stage 4c Q8_0 unsplit WMMA prefill

The [Stage 4 plan](../stage4-long-context-attention.md) compares
unsplit WMMA and tiled attention within the same packed Q8_0 KV
format. The reference is `ccf2ea2`. Tests ran in the AI container
with the [same model and platform](07b-direct-q6k-attention.md) as the
other Phase 2 changes. The matched prompt uses the published token
file and Qwen3.5-4B Q4_K_M GGUF. The KV pool stays in Q8_0 format.

## Implementation

Each workgroup gathers a K/V tile from physical page rows. The
reader loads the scale and two digit pairs from two-byte-aligned
addresses, multiplies the signed values in FP32 and stages the
products in FP16. Widths 64, 128 and 256 admit four query heads
per shared KV head, gated or dense. Masked key slots are zero-filled.
The same geometry and start-position decision is used in Go and HIP.
WMMA attribute and launch failures propagate to the plan executor.

The production policy keeps chunks beginning before position 512
on tiled attention. The 1024-row parent has a separate start
position of 2048. It uses unsplit WMMA for eligible later Q8_0
chunks. The cached plan evaluates the absolute parent chunk start
when executing. F32/F16 keep their existing arithmetic and selection.

## Choosing the starting position

The initial probe extended the F32/F16 WMMA selection to Q8_0
without a distinct prefix rule. With 64 tokens the max logit delta
was 0.0840435 against the 0.05 gate. With 8192 tokens the same input's
prefill delta was 0.408962 and the worst of three following decode
steps reached 0.527487. The first greedy token still matched in
both cases. The unrestricted CLI reached 2123.22 tok/s, so the
limiting criterion was model numerics.

The position probe compares two populated texts at 8192 tokens
and fixed chunk 512. Control is all tiled. The two inputs are
the same A and B texts as [Stage 4b](08b-f16-split-cells.md).
The probe switches the attention route only at chunk boundaries
and reports prefill through final host logits. It records one
screening sample per text and route. The max covers all logits
in prefill and the next three steps.

| WMMA positions | A tok/s | A max delta | A greedy | B tok/s | B max delta | B greedy |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| None | 1101.10 | 0.000 | Match | 1106.94 | 0.000 | Match |
| `[0, 8192)` | 2131.54 | 0.527 | Match | 2122.54 | 0.259 | Match |
| `[512, 8192)` | 2122.48 | 0.269 | Match | 2118.64 | 0.181 | Match |
| `[1024, 8192)` | 2106.91 | 0.344 | Match | 2077.11 | 0.166 | Match |
| `[1536, 8192)` | 2060.90 | 0.283 | Match | 2055.99 | 0.149 | Match |
| `[2048, 8192)` | 2015.81 | 0.268 | Match | 2004.32 | 0.155 | Match |
| `[3072, 8192)` | 1881.99 | 0.222 | Match | 1876.99 | 0.182 | Match |
| `[4096, 8192)` | 1727.13 | 0.286 | Match | 1717.18 | 0.196 | Match |
| `[0, 7680)` | 1915.59 | 0.511 | Match | 1902.28 | 0.251 | Match |
| `[512, 7680)` | 1908.73 | 0.383 | Match | 1900.93 | 0.183 | Match |
| `[1024, 7680)` | 1888.58 | 0.352 | Match | 1883.30 | 0.175 | Match |

Position 512 is the fastest screened start that meets both the
0.30 model gate and 1850 tok/s in both texts at chunk 512. The final
compiled dispatch reproduces the screened results. The short 64-token run
stays tiled and matches the control through all four logit rows.

The first position-only rule also dispatched 1024-row chunks
starting at position 1024. The real 8192-token control at that
chunk size showed a 0.34388 delta on text A and 0.16619 on B.
The 2048-row chunk passed for both inputs, at 0.26777 and 0.15454.
The 1024-row policy keeps both leading chunks tiled, preserving
the chunk-512 route and its throughput budget.

## Native and model verification

The native fixture permutes the pages and compares the same Q8_0
bytes against a CPU oracle and the forced tiled device reader.
It verifies the gated 64-row case at start 512 and a dense 19-row
case at start 527, including partial query and key tiles. Outputs
are deterministic on repeat. The error limit is the existing
`5e-3 * (1 + abs(reference))`. Final Release CTest passed 10/10. The gfx1151 head-width-256 gated
Q8_0 kernel uses 254 VGPRs and 32 SGPRs, with zero spills and zero
private scratch. `ship/resources.log` records the compiler diagnostics.

| Head width | Gated max vs CPU / tiled | Dense max vs CPU / tiled | Dynamic LDS |
| ---: | --- | --- | ---: |
| 64 | 2.27e-5 / 2.27e-5 | 3.22e-5 / 3.22e-5 | 14,336 B |
| 128 | 2.08e-5 / 2.08e-5 | 3.54e-5 / 3.54e-5 | 26,624 B |
| 256 | 2.08e-5 / 2.09e-5 | 3.86e-5 / 3.86e-5 | 51,200 B |

`go build -tags hip ./...` and `go vet -tags hip ./...` passed.
Run `YUUKA_TEST_ATTENTION_LONG=1 go test -tags hip ./engine/gpu -run '^TestGPU(Q8AttentionPathsAgree|RealModelKVFormats)$' -count=1 -v`
with the model and matching library. All four greedy IDs match
for each same-format comparison. The 64-token control matches the
short-prefill path exactly.

| Chunk | 8192-token text A max logit delta | 8192-token text B max logit delta |
| ---: | ---: | ---: |
| 512 | 0.268958 | 0.181484 |
| 1024 | 0.267765 | 0.154540 |
| 2048 | 0.267765 | 0.154540 |

The KV-format regression recorded Q8_0 versus F32 max 0.0497253
under the existing 0.5 gate and matched six greedy tokens.
`go test -tags hip ./backend/hip -run '^TestPrefillAttentionMMAQ8Prefix$' -count=1`
passes both sides of the 512 and 2048 position cutoffs. The F16
attention control reports a maximum difference of 0.220348 and
matching top-1 at 2048 tokens.

The unrestricted trial and screening source are under
`/tmp/yuuka-phase2-5-11.3wZQZG/4c` on AI. Run
`go test -tags hip ./engine/gpu -run '^TestStage4CQ8Crossover$' -count=1 -v`
from its `src` directory with `YUUKA_TEST_MODEL` set. The source is
`engine/gpu/attention_crossover_test.go`. The final build and logs
are under `ship`.

## Paired throughput

The [comparison driver](../../../../bench/prefill-compare/main.go)
accepts `-kv-type q8_0` and forces the scalar reader with
`-tiled-attention`. Both arms use the same production build,
8192 IDs, chunk 512, one warmup and two samples per process.
The order is tiled/auto/auto/tiled, then auto/tiled/tiled/auto.
The end point is the final host logit row. The CLI also runs
`yuuka bench --backend hip --kv-type q8_0 --prompt-tokens 8192 --chunk-size 512 --decode-tokens 4 --warmup 1 --reps 3`
with the same GGUF.

| Order | Tiled median [range], ms | Auto median [range], ms | Throughput, tok/s | Latency reduction |
| --- | --- | --- | --- | ---: |
| A | 7410.379 [7392.026–7434.740] | 3861.036 [3842.189–3871.043] | 1105.48 → 2121.71 | 47.90% |
| B | 7415.782 [7409.023–7433.906] | 3875.254 [3865.158–3882.195] | 1104.67 → 2113.93 | 47.74% |

All eight runs returned greedy IDs `[13477, 37550, 33075, 888]`
with `-greedy-steps 3`. The input file and GGUF match the [Phase 2
hashes](05-cli-chunk-policy.md#baseline-and-protocol). The CLI prefill
p50 was 3.873 s (2114.95 tok/s), with 3.910 s p95. Both comparisons
and the CLI clear the 1850 tok/s gate. Raw samples are in
`ship/{tiled,wmma}-*.json` and `ship/cli.log`.

The read-only review ended with `NO_BLOCKING_ISSUES` after two
rounds. The second round verified the 1024-row cutover and tests.
