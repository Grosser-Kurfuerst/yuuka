# Iteration 7 long-context attention

## Scope

The control is iteration 6 at `93c9ba1`. The ordinary final layer
already consumes one query. The 8K trace at chunk 512 put 107.688 ms
in all attention operations of the last chunk and 797.644 ms in
attention over the full request. This stage measures split-key WMMA
over the full-row earlier attention layers with F16 KV, head width
256 and four query heads per KV head.

## Independently normalized ranges

The prototype divides each causal key interval into contiguous
16-key tile ranges. Each workgroup writes its maximum, sum and
unnormalized FP32 values. A merge kernel weights the nonempty
records by their base-2 maximum and applies the query gate once.
The layout reserves 32.25 MiB for four splits at chunk 512.

The native tests used 17 queries at positions 0 and 270, permuted
physical pages, a partial last page and splits with no visible keys.
One, two and four splits passed at the existing attention tolerance.
At position 270 one split matched unsplit WMMA exactly and two and
four splits differed by at most 0.0000215 and 0.0000263. A second
execution reproduced the outputs. Exact-fit scratch passed and the
short span failed. Release CTest passed 9/9.

The native probe uses the target model dimensions, synthetic Q and
F16 K/V, and two warmups followed by seven samples. The table gives
the means of two process medians for the full attention operation,
including the merge. Times are in milliseconds.

| New rows | Start position | Direct | One split | Two splits | Four splits |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0 | 0.081 | 0.068 | 0.083 | 0.118 |
| 512 | 0 | 0.477 | 0.526 | 0.613 | 0.939 |
| 512 | 2048 | 3.241 | 3.419 | 3.319 | 3.304 |
| 512 | 7680 | 15.259 | 15.474 | 11.751 | 10.137 |
| 1024 | 7168 | 26.586 | 25.757 | 19.392 | 18.560 |
| 2048 | 6144 | 44.341 | 43.804 | 34.430 | 33.155 |
| 4096 | 4096 | 63.882 | 63.488 | 55.588 | 56.643 |

The direct and four-split kernels used 50 KiB of dynamic LDS.
The compiler reported 256 VGPRs for direct WMMA, 254 for the partial
kernel and 15 for the initial merge. All reported zero VGPR spills
and zero private scratch.

### Model sensitivity

Two 8192-token prompts used fixed token IDs per prompt and F16 KV,
with one prefill logit row and three subsequent greedy steps. The
two-split trial starting at position 2048 preserved top-1 tokens on
all four chunk configurations. Its maximum logit difference across
the prefill and continuation was 0.39797. Moving onset to position
6144 gave maxima 0.44455 for two splits and 0.35025 for four splits
at chunk 512 and 2048. Limiting four splits to the terminal 8K
window gave a maximum of 0.38925. These exceed the 0.3 tolerance
used for existing long attention route comparisons.

## Initial maximum scan

The seeded trial scanned preceding key ranges for maxima before
computing the partials. Starting from the original running maximum
preserves rounding of the FP16 probability tile within each range.
The FP32 reduction still groups the values differently. It added
128 KiB of maximum scratch at chunk 512. At position 270 the
native maximum errors dropped to 0.0000000782 for two splits and
0.0000000969 for four splits. The tested seeded four-split
dispatch preserved every greedy token, with a 0.27609 maximum
logit difference across both prompts and all measured chunk sizes.
The alternative that scans 32 keys per block iteration and uses
one wave to merge each row and head gave the same model logits.

The complete maximum scan cost as much as the partial-kernel saving.
For 512 rows at position 7680 the direct and seeded two/four split
medians were 14.355, 13.985 and 14.315 ms in the first trial. With the
32-key scan and wave merge the times were 15.119, 15.258 and 15.378 ms.
Its 8K throughput was 1942.99, 1795.93, 2057.50 and 1812.87 tok/s at
chunks 512, 1024, 2048 and 4096, versus 1957.54, 1813.70, 2071.45
and 1808.70 for direct WMMA in the same session.

## Leading-prefix seed

The final candidate uses one score-only maximum scan over up to the
leading 512 keys of the first range. The later ranges share this seed.
The shorter scan reduces the change in the FP16 probability rounding
and costs less than computing a separate maximum for each range. The
512-key seed held the largest logit difference to 0.25627 with the
same greedy outputs at chunk 512 and 2048. The 1024-key seed reached
0.37293, above the existing 0.3 gate.

The [native dispatcher](../../../../kernels/hip/prefill/prefill.hip)
chooses four splits for 512, 1024 and 2048 query rows at execution
positions 6144 and later when the full query range ends by token
8191. The [model](../../../../engine/gpu/model.go) reserves a reusable
split buffer on gfx1151 for that geometry and F16 KV. The records
and the one maximum per row and head share this allocation. At chunk
512 it is 32.28125 MiB. The one-query tail and the 4096-row chunk
continue to use direct attention.

## Prompt and chunk sweep

The [AI configuration and verified model](01-baseline.md#environment)
used F16 KV, one request, one greedy output and two warmups. Each short
or boundary cell had five timed samples. Each 8K cell had three timed
samples. Values are unprofiled medians.

| Prompt | Chunk | Direct tok/s | Candidate tok/s | Change | TTFT direct → candidate (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1879.42 | 1871.22 | -0.44% | 0.068 → 0.069 |
| 512 | 512 | 2346.98 | 2368.54 | +0.92% | 0.218 → 0.216 |
| 2048 | 2048 | 2402.33 | 2390.91 | -0.48% | 0.853 → 0.857 |
| 2049 | 2048 | 2177.86 | 2175.84 | -0.09% | 0.941 → 0.942 |
| 8192 | 512 | 1961.03 | 2020.44 | +3.03% | 4.178 → 4.055 |
| 8192 | 1024 | 1800.78 | 1835.57 | +1.93% | 4.549 → 4.463 |
| 8192 | 2048 | 2080.98 | 2107.57 | +1.28% | 3.937 → 3.887 |
| 8192 | 4096 | 1813.07 | 1817.76 | +0.26% | 4.518 → 4.507 |

Use the measured chunk 2048 setting for the tuned 8K comparison. The
4096-row setting and the four short and boundary cases use direct
attention. The benchmark ran in automatic power mode at starting
sensor readings of 49–65 °C.

## Final attention and scratch measurements

Three actual-populated whole-prompt traces, after one warmup, give
the following attention medians. They include all attention layers,
including the compact final-layer query.

| Prompt / chunk | Total attention direct → split (ms) | Last chunk direct → split (ms) |
| --- | ---: | ---: |
| 512 / 512 | 4.838 → 4.772 | 4.838 → 4.772 |
| 8192 / 512 | 797.644 → 708.625 | 107.688 → 81.678 |
| 8192 / 2048 | 606.826 → 570.484 | 289.128 → 252.483 |

The fixed-512 last chunk clears the 100 ms target. The native trace at
512 rows and position 7680 reports 0.678 ms for the score-only scan,
9.796 ms for the four partials and 0.409 ms for merge. These are
separate medians of seven samples after two warmups. The probe's full
device-time medians were 14.239 ms direct and 10.950 ms with four splits.
At 2048 rows and position 6144, two splits measured 36.036 and
34.550 ms in consecutive prototype and final runs. Four splits
measured 35.555 and 35.467 ms. Their order changed within the
observed spread. The runtime uses the four-split configuration
verified by the real-model comparison.

The compiler resource script reports 98 VGPRs for the seed, 256 for
the partial kernel and 23 for merge, with no VGPR spills or private
scratch. The three kernels reserve 48 KiB, 50 KiB and zero dynamic
LDS respectively. In the populated 8K run the prefill allocations
totaled 191,924,224 bytes at chunk 512 and 767,696,896 bytes at 2048.
The split buffer accounts for 33,849,344 and 135,397,376 bytes of those
totals. The figures cover prefill work buffers and exclude weights,
session state and the KV pool.

## Verification

A gfx1151 Release build and all nine CTest tests passed on AI. The
shipping prefix-seed variant passed the empty and partially masked
native tests, the exact-fit and short scratch checks and repeatability.
The [long-model differential](../../../../engine/gpu/attention_compare_test.go)
compared the 8K prompt and three greedy steps with direct WMMA, with
the same top-1 tokens and a maximum difference of 0.25627.

The HIP CLI and Go builds, `go vet ./...`, `go vet -tags hip ./...`,
`go test ./...`, `go test -race ./...` and
`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -count=1 -v`
passed. The HIP suite includes the new differential, the final-query
oracle, long-context, CPU logits, separate decode and speculative
greedy checks. One MoE and two SentencePiece tests skipped because
their fixtures were absent.

After the empty-record check and nil-scratch validation were added,
Release CTest passed again. Repeating the two 8K prompt comparisons at
chunks 512, 1024, 2048 and 4096 produced the same top-1 at every
prefill and decode step. The maximum logit difference was 0.25627 at
all three dispatch sizes. The 4096-row direct route matched exactly.
The native test also downloaded an empty split and checked the
`(-inf, 0, zeros)` record literally.
The rebuilt CLI reported 1990.19 and 2104.73 tok/s at 8K for chunks
512 and 2048 in the final confirmation.

Raw logs, probe source, build directories and logits are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-07-to3OFt/`
on AI.

## Review and decision

Accepted the four-split route with the 512-key prefix seed in the
measured window. The fixed-512 attention target and real-model
continuation gate passed. One read-only review round returned
`NO_BLOCKING_ISSUES`. `git diff --check` passed.
