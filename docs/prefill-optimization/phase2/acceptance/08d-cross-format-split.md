# Stage 4d Split-key attention across KV formats

This slice extends the [Stage 4 plan](../stage4-long-context-attention.md)
from the `a7eca5d` control. Tests use the
[reference Qwen3.5-4B Q4_K_M](05-cli-chunk-policy.md#baseline-and-protocol)
on AI gfx1151, ROCm 10.1.0, HIP 7.16.26385 and Go 1.24.4 with
automatic power. Each timing arm uses identical tokens and the
same KV storage format. Each process uses one active session.

## Implementation

The Go planner passes its pool dtype to
`AddAttentionPrefillPagedSplitTypedAt`. The existing F16 append
method forwards unchanged. The native descriptor reuses its
pool dtype and four-range scratch encoding. The seed stages K
alone, including Q8_0 digits times their FP16 block scales. The
partial reads K and V in the indicated format and the FP32
merge consumes the same layout across all three formats.

The bridge checks both pool offsets, their byte spans and the
alignment required by the gather. Split scratch is available
only for gated 256-wide GQA, with the expected four-query-head
group, sufficient aligned bytes and no alias to live operands.
Forced split rejects missing scratch and incompatible flags.

| Format | Measured query rows | Accepted start range | Scratch in the reference model |
| --- | ---: | --- | ---: |
| F32 | 512 | 6144–14336 inclusive | 33,849,344 B |
| Q8_0 | 512 | 8192–14336 inclusive | 33,849,344 B |

The scratch footprint is `rows * heads * (4 * (head_dim + 2) + 1) * 4`
bytes. The two new formats reserve it only on gfx1151 when the
configured chunk is 512 and capacity can fit an eligible chunk.
F16 retains the [Stage 4b policy](08b-f16-split-cells.md). The
native launcher selects by execution position, preserving the
single cropped query and the shorter tails on unsplit attention
for F32 and Q8_0.

## Same-format trial and decision

The test repeats two populated prompts A and B from [Stage 4b](08b-f16-split-cells.md)
at chunk 512. Calls use the same model throughout each probe. One arm
clears the scratch binding before plan creation, which chooses unsplit
WMMA and preserves the Q8_0 tiled prefix. The other arm binds scratch
and selects auto. Each trial resets recurrent state and prefill plans
before timing through final host logits. The two prompts use opposite
interleaving order. Full-logit max includes three following decode
steps. A table p50 is the median of two samples per arm, with the
full range in brackets.

The initial Q8_0 window started at 6144. At 8192 prompt tokens text A
reached a 0.36178 continuation delta even though all four greedy IDs
matched. Text B's max was 0.16059. The selected Q8_0 cutoff is 8192,
so a prompt ending at that point uses unsplit throughout its WMMA
region. This retains the typed split path at the separately verified
longer positions.

| Dtype | Text | Tokens | Unsplit, ms | Auto, ms | Reduction | Max logit delta |
| --- | --- | ---: | --- | --- | ---: | ---: |
| F32 | A | 8192 | 4042.7 [4032.7–4052.8] | 3922.4 [3916.4–3928.4] | 2.98% | 0.256796 |
| F32 | A | 16384 | 10028.9 [9998.9–10058.9] | 9399.4 [9397.5–9401.3] | 6.28% | 0.143852 |
| F32 | B | 8192 | 4078.7 [4067.9–4089.5] | 3969.2 [3967.4–3971.0] | 2.69% | 0.133584 |
| F32 | B | 16384 | 10072.1 [10070.7–10073.6] | 9440.4 [9437.9–9442.9] | 6.27% | 0.158485 |
| Q8_0 | A | 16384 | 9195.5 [9190.5–9200.5] | 8915.3 [8914.0–8916.7] | 3.05% | 0.174742 |
| Q8_0 | B | 16384 | 9193.5 [9184.8–9202.3] | 8923.5 [8920.9–8926.1] | 2.94% | 0.170582 |

## Isolated operation and resources

The native probe uses identical generated pools and a reversed page
table per format, with 16 heads, four KV heads and width 256. It
times synchronous plan submit/wait, including seed, partials and
merge. Order is unsplit/four/four/unsplit, with two warmups and
seven samples per process. Values are pooled medians and ranges.
Forced Q8_0 rows at 6144 and 7680 serve as controls before its
production cutoff.

| Dtype | Rows | Start | Unsplit, ms | Split 4, ms | Reduction |
| --- | ---: | ---: | --- | --- | ---: |
| F32 | 512 | 6144 | 12.545 [12.448–12.642] | 9.438 [9.202–9.571] | 24.77% |
| F32 | 512 | 7680 | 16.185 [15.954–16.405] | 11.657 [11.402–12.073] | 27.98% |
| F32 | 512 | 8192 | 17.322 [17.181–17.461] | 12.435 [12.135–13.362] | 28.21% |
| F32 | 512 | 14336 | 30.070 [30.030–30.119] | 23.449 [23.116–23.749] | 22.02% |
| Q8_0 | 512 | 6144 | 8.203 [8.095–8.364] | 8.417 [8.334–8.553] | -2.62% |
| Q8_0 | 512 | 7680 | 10.177 [10.084–10.367] | 10.219 [10.129–10.320] | -0.41% |
| Q8_0 | 512 | 8192 | 10.889 [10.697–10.982] | 10.782 [10.682–10.873] | 0.98% |
| Q8_0 | 512 | 14336 | 22.589 [22.486–22.681] | 17.753 [17.654–17.879] | 21.41% |

The Q8_0 isolated change at 8192 is small relative to sample spread.
The 16K model comparison evaluates the full measured long window.

| Format | Seed VGPR / SGPR | Partial VGPR / SGPR | Dynamic LDS seed / partial | Spills |
| --- | ---: | ---: | ---: | ---: |
| F16 | 98 / 40 | 254 / 48 | 49,152 / 51,200 B | 0 |
| F32 | 97 / 40 | 254 / 48 | 49,152 / 51,200 B | 0 |
| Q8_0 | 99 / 40 | 254 / 48 | 49,152 / 51,200 B | 0 |

The shared merge has no dynamic LDS. All seed and partial variants
report zero private scratch. Compiler data are in `ship/resources.log`.

## Verification

Release CTest passed 10/10. The native suite exercised forced 1,
2 and 4 ranges at position 0 and position 527 for the two new
formats using permuted pages and uneven tails. At 527, all three
F32 outputs stayed within `2.2e-6` of unsplit WMMA. Q8_0 stayed
within `1.72e-5`. Repeatability and empty partials passed. Native
automatic route comparisons passed at 6143/6144 for F32 and
8191/8192 for Q8_0, along with the original F16 edges.

Both F32 and Q8_0 validation cases accepted exact-fit aligned
offsets and rejected one-byte-short key/value pools and scratch,
misalignment, scratch aliasing, missing forced scratch, unsupported
head width and dtype, dense queries and incompatible flags.

HIP Go build and vet passed, and both F16 attention controls
passed. The two-format run of
`YUUKA_TEST_ATTENTION_LONG=1 go test -tags hip ./engine/gpu -run '^TestGPULongSplitAttentionMatchesDirect$/^(F32|Q8_0)$' -count=1 -v`
passed all eight subtests. At 8192, both Q8_0 sets match exactly.
The F32 maximum over both lengths and continuations is 0.256796.
Both Q8_0 16K sets retain the same four greedy token IDs.

The experiment root on AI is
`/tmp/yuuka-phase2-5-11.3wZQZG/4d`. The model timing probe is
`src/engine/gpu/attention_format_probe_test.go`. Use
`YUUKA_SPLIT_PROBE_DTYPE=f32` or `q8_0` and
`go test -tags hip ./engine/gpu -run '^TestStage4DMatchesUnsplit$' -count=1 -v`
with the matching library and model. The final Q8_0 timings use the
`ship` build and only the 16K subtests. The original wider Q8_0 timings
remain in `q8_0.log`.

The native probe is `cell.cpp` in the experiment root. Compile it
against `ship/build/libyuuka_hip.so`. Invoke
`./cell 512 START SPLITS 256 DTYPE 7 -` from `ship` with `SPLITS=0`
or 4 and `DTYPE=0` for F32 or 8 for Q8_0.

The read-only review ended with `NO_BLOCKING_ISSUES` after one
round. The C descriptor comment records the validated alignments.
