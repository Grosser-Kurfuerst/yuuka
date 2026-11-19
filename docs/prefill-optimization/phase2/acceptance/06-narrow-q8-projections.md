# Stage 2 narrow Q8_0 projection trial

## Setup

The [Stage 2 plan](../stage2-narrow-q8-projections.md) sets a
`1e-5` CPU tensor gate, a `0.05` same-chunk model-logit gate with
identical top-1 and eight greedy continuation tokens, and a paired
5% throughput lead over upstream at four prompt lengths.

The control is Step 1d `ba37822`. Control and candidate are separate
Release gfx1151 builds on AI, with the [Step 1d model, token fixture and
toolchain](05-cli-chunk-policy.md#baseline-and-protocol). Runs use F16
KV, one request and automatic GPU power. The official ggml-org/llama.cpp
`v1.0.0` tag and commit were verified at
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` before building the reference
driver. Measurement starts immediately before prefill and ends with
the final logit row on the host.

## Candidate 2a

The isolated candidate added one batch-pair operation after the QKV
and gate group in [the recurrent prefill planner](../../../../engine/gpu/plan.go).
It binds beta then alpha to distinct outputs and reads their common
FP32 normalized input. The runtime eligibility is gfx1151, input width
2560, rank 32, both weights Q8_0 and 128–2048 rows. The native
operation uses packed 34-byte Q8_0 blocks, two FP32 output spans and
no intermediate activation or weight buffer. One wave holds a channel
from each matrix across a token tile and reuses each FP32 activation
vector between the dots. It reduces at the end of K and masks the
final token tile.

The experimental operation was appended at opcode 44 with ABI 10. Its
native validation checks the two dtypes, dimensions, buffer bounds,
alignment and output overlap. The decode pair kept its existing
descriptor and dispatch. The candidate changes are isolated from the
accepted implementation unless a candidate passes all gates.

### Native product and trace

The independent CPU product used different weighted signed digits in
the two matrices, zeros and signed extremes, nonuniform FP32 activations,
and output sentinels. Both outputs passed in each cell. All ten Release
CTests passed, including native pair validation with a disjoint alias
case and overlap, alignment, truncation and dtype rejection.

| Token rows | Beta max absolute error | Alpha max absolute error |
| ---: | ---: | ---: |
| 128 | 1.34e-6 | 1.34e-6 |
| 131 | 1.18e-6 | 1.27e-6 |
| 512 | 1.70e-6 | 1.46e-6 |

`TestGPUProfilePrefill` with `YUUKA_TEST_PROFILE_ROWS=128,512` and
`YUUKA_TEST_KV=f16` saw 24 pairs, with the total plan decreasing from
581 to 557 operations. Both builds issued 97 `quant_q8_1` ops.
At position zero the paired ops took 0.783 ms at 128 rows and
4.006 ms at 512. Across positions 0, 2048 and 7680 those times
were 0.777–0.792 ms and 4.006–4.032 ms. The 1.2 ms diagnostic
stretch target passed at 128 and missed at 512.

### Same-chunk numerical screen

These initial A/B processes each ran one warmup and three timed
prefills. Both builds used the same token IDs, chunk and F16 pool.
All nine greedy IDs matched in every cell. The full prefill logits
exceeded 0.05 at every tested length.

| Prompt | Chunk | Control p50 (ms) | FP32 pair p50 (ms) | Max logit difference |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 68.106 | 65.060 | 0.950872 |
| 129 | 512 | 87.037 | 82.979 | 0.636650 |
| 511 | 512 | 216.751 | 213.476 | 0.567958 |
| 512 | 512 | 213.952 | 210.117 | 0.449573 |
| 1024 | 512 | 431.524 | 427.476 | 0.401648 |
| 2048 | 1024 | 850.666 | 843.795 | 0.509289 |
| 8192 | 1024 | 3872.042 | 3881.381 | 0.409032 |

### Official llama.cpp v1.0.0 speed comparison

The same dtypes, prompt IDs, capacity request and greedy top-1 endpoint
were used. Yuuka and llama.cpp used the chunks reported below, with the
reference logical batch and microbatch equal. Each process used two
warmups and five timed samples (three at 8192). Each cell pools two
processes per engine. Order A was Yuuka/llama/llama/Yuuka. Order B
reversed that order. These are unprofiled runs.

| Prompt | Yuuka / llama chunks | A Yuuka / llama p50 (ms) | A throughput gain | B Yuuka / llama p50 (ms) | B throughput gain |
| ---: | --- | --- | ---: | --- | ---: |
| 128 | 128 / 128 | 63.712 / 68.290 | +7.19% | 64.078 / 68.395 | +6.74% |
| 512 | 512 / 512 | 213.844 / 232.260 | +8.61% | 212.904 / 231.303 | +8.64% |
| 2048 | 1024 / 512 | 854.364 / 940.204 | +10.05% | 853.978 / 943.474 | +10.48% |
| 8192 | 1024 / 256 | 3924.364 / 4159.148 | +5.98% | 3905.908 / 4168.063 | +6.71% |

The throughput target was reached in both orders. The native CPU
product and throughput results therefore cannot justify promotion
without a same-chunk numerical solution.

### Kernel tile and operand ablations

The isolated probe times the two original WMMA operations together
or the fused launch through HIP operation events. Each geometry uses
ten warmups and 50 measured submissions. The packed Q8_0 matrices
and FP32 input are identical across arms. Values are median
microseconds for both outputs.

| Rows | Two original launches | Pair tile 1 | Pair tile 2 | Pair tile 4 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 219.680 | 34.264 | 27.009 | 23.483 |
| 129 | 209.823 | 32.620 | 43.479 | 23.503 |
| 512 | 212.188 | 115.050 | 86.698 | 81.428 |
| 1024 | 237.133 | 266.147 | 163.258 | 139.960 |
| 2048 | 280.192 | 495.007 | 349.920 | 279.942 |

The tile-2 129-row run ranged from 26.849 to 57.856 μs. Tile 4
had the best median in this screening sweep. At 2048 rows its
median nearly equaled the original pair.

The [original WMMA kernel](../../../../kernels/hip/gemm/wmma_gemm.hip)
rounds normalized input and each dequantized Q8_0 weight to FP16
before WMMA and accumulates in FP32. Candidate 2a keeps both
operands in FP32 and also changes summation order. Temporary
variants restored the original rounding for the activation, the
weight, or both, while retaining the scalar dot and wave reduction.
Each variant used the same chunks and two warmups with five timed
prefills. All nine greedy IDs still matched. Restoring both
operand conversions left a substantial model-logit delta.

| Prompt | FP32 pair | Round activation | Round weight | Round both |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.950872 | 0.439721 | 0.993468 | 0.399395 |
| 512 | 0.449573 | 0.394662 | 0.431151 | 0.325426 |
| 2048 | 0.509289 | 0.389121 | 0.639953 | 0.408677 |

These are full-logit maxima against the original same-chunk path.
Rounding alone did not recover the existing arithmetic. The
recurrence applies sigmoid to beta and transforms alpha into
exponential state decay. Subsequent layers and activation
quantization can further amplify small differences in those
inputs. [GDN preparation](../../../../kernels/hip/prefill/gdn_chunk.hip)

### Attention regression and compatible WMMA probe

The full real-model Go suite also rejected 2a. The existing long
`TestGPUAttentionPathsAgree` reported 0.309254 against its 0.30
bound. `TestGPULongSplitAttentionMatchesDirect` reported 0.439037
against 0.30 on its first decode step. Both failures preserved
greedy IDs. The project keeps the existing gates.

A temporary paired LDS WMMA probe kept the original FP16 operands
and the ordered WMMA operations. Both 32- and 64-wide K tiles
matched the original bitwise at pp128 and pp512, with identical
eight-step continuations. At pp128 and pp512 the K64 probe timed
at 98.420 and 97.899 μs in isolation. The two original narrow
operations timed at 219.680 and 212.188 μs in the tile sweep.
This confirmed that changing the launch grouping while retaining
the original WMMA arithmetic preserves model outputs.

## Candidate 2c with equivalent WMMA arithmetic

The user approved this arithmetic-preserving follow-up for Stage 2.
Candidate 2a had already met the upstream performance condition
for 2b, so no Q8_1 trial was required by the plan. The production
candidate [pairs the narrow projections](../../../../engine/gpu/plan.go)
and [feeds WMMA from registers](../../../../kernels/hip/gemm/wmma_gemm.hip).
One wave produces a 16-token by 16-channel tile for each matrix.
It shares the activation fragment, decodes the Q8_0 halves from
two-byte-aligned packed reads, preserves the GFX11 half-wave K
orientation and writes directly from the two FP32 accumulators.
The launcher has no LDS tile or temporary dequantized allocation.
C validation restricts the op to gfx1151 and validates both spans
before either output can be written.

An initial CPU math check on broad random FP16 inputs differed by
at least 2.49e-5 even though the two device output arrays matched
legacy WMMA bitwise. A three-case probe verified that legacy WMMA
preserved FP16 subnormal activation and weight operands in simple
products, so an assumed flush-to-zero rule would have been an
incorrect oracle. The final native check retains the general
fixture and bitwise check, and adds a separate dyadic fixture for
the independent `1e-5` CPU test. Its operand grids bound the
integer numerator of any partial sum below 2^24. It isolates
matrix indexing, signed decode and operand conversion from the
original WMMA's rounding in the general fixture.

| Prompt | Chunk | Prefill logit bytes vs control | Greedy top-1 and eight steps |
| ---: | ---: | --- | --- |
| 128 | 128 | Match | Match |
| 129 | 512 | Match | Match |
| 511 | 512 | Match | Match |
| 512 | 512 | Match | Match |
| 1024 | 512 | Match | Match |
| 2048 | 1024 | Match | Match |
| 8192 | 1024 | Match | Match |

### 2c native checks and speed

Both outputs matched native legacy WMMA bitwise for the general
and dyadic fixtures at 128, 131 and 512 rows. The dyadic CPU
maxima were 1.19e-7 for each output in all six comparisons. The
general fixture's CPU maxima ranged from 4.51e-5 to 6.66e-5.
Release CTest passed 10/10. The dispatcher uses the matched
geometry and keeps the existing route above 2048 rows.

The isolated microbenchmark uses the same inputs, ten warmups
and 50 timed submissions as the original pair sweep.

| Rows | Original pair p50 (μs) | Equivalent WMMA pair p50 (μs) |
| ---: | ---: | ---: |
| 128 | 219.680 | 57.104 |
| 129 | 209.823 | 53.358 |
| 512 | 212.188 | 59.730 |
| 1024 | 237.133 | 76.280 |
| 2048 | 280.192 | 102.451 |

The v1.0.0 comparison repeated the same protocol and chunks as
2a. All eight paired throughput margins clear 5%. Each bracket
is the full observed range in milliseconds.

| Prompt | A Yuuka p50 [range] | A llama p50 [range] | A gain | B Yuuka p50 [range] | B llama p50 [range] | B gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 64.253 [64.045–64.969] | 67.792 [67.495–69.651] | +5.51% | 64.462 [64.093–65.736] | 68.418 [67.006–69.310] | +6.14% |
| 512 | 211.166 [209.909–218.273] | 228.637 [225.832–231.563] | +8.27% | 210.852 [210.224–212.982] | 228.947 [228.319–231.604] | +8.58% |
| 2048 | 850.749 [836.463–858.313] | 938.311 [929.001–959.128] | +10.29% | 852.787 [848.668–870.051] | 944.914 [942.132–951.120] | +10.80% |
| 8192 | 3876.212 [3866.477–3880.027] | 4185.495 [4168.534–4190.520] | +7.98% | 3884.187 [3864.107–3900.043] | 4195.371 [4171.114–4226.019] | +8.01% |

Both engines agreed on the first greedy token at every length.
The model equivalence probe used the [matched-token driver](../../../../bench/prefill-compare/main.go)
with a temporary diagnostic extension. The alternate prompt repeats
`Please summarize this paragraph and list the three most important ideas. `.
Its 2048-ID file SHA-256 is
`d8a595a030b9161f98ffba972dcb5eddfa10d2b5f1c484434946b800fe343345`.

On AI the diagnostic extension is
`/tmp/yuuka-phase2-5-11.3wZQZG/2a/diagnostic.py`. The separately built
baseline and candidate drivers are `2a/validation/ctrl` and
`2c/verify/diagnostic` under that directory. Run both with the
model, identical `-tokens` file, `-prompt N -chunk C -warmup 0 -reps 1
-greedy-steps 8 -diagnostics DIR`, each with its own HIP library.
The extension prefills one chunk at a time, takes the interleaved
GDN snapshot without resetting the state, and dumps the final
prefill and all eight subsequent full-vocabulary rows. Compare
the directories with `diff -qr`. Both prompts passed at 128,
512, two explicit 512-token calls, and 2048 with chunk 1024.
The original prompt also passed at 2048 with chunk 2048 and
8192 with chunk 1024. Every snapshot and logit file matched.

### 2c regression and operation trace

The AI build passed `go build ./...`, `go vet ./...`,
`go test ./...`, `go test -race ./...`, HIP build and vet, and
`go test -tags hip -p 1 ./... -count=1 -v` on the verified real
model. Set `YUUKA_TEST_KV=f16`, `YUUKA_TEST_ATTENTION_LONG=1`,
and `YUUKA_TEST_PREFILL_NARROW_CONTROL` to the baseline wrapper
under `2c/verify/control`. The new six-case same-chunk control
regression and both original attention gates ran and passed.
The optional Step 1b comparison wrappers, MoE model, and two
SentencePiece fixtures were unavailable.

`TestGPUProfilePrefill` repeated the 128- and 512-row profiles.
The original 581-op plan became 557 ops, with 24 paired narrow
ops and the same 97 activation quantizer ops. The 24 pair times
at position 0 totaled 1.902 and 3.748 ms. The 1.2 ms target is
diagnostic. The primary upstream throughput gate passed at
every length. HIP Clang reported 92 VGPRs, 26 SGPRs, zero spills,
zero scratch and zero LDS for the final paired kernel.

The original-route comparison used control/candidate/candidate/control,
two warmups and five timed prefills per process (three at 8192).
Both arm pairs produced identical prefill logits and greedy IDs.
The matched chunks and input tokens isolate the paired projection
from the prior CLI policy and upstream batch settings.

| Prompt | Chunk | Control p50 [range] (ms) | 2c p50 [range] (ms) | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 69.119 [68.452–70.605] | 64.955 [64.515–68.048] | 6.02% |
| 512 | 512 | 221.519 [218.948–226.129] | 218.633 [213.339–223.494] | 1.30% |
| 2048 | 1024 | 879.908 [860.217–896.693] | 869.964 [855.062–882.063] | 1.13% |
| 8192 | 1024 | 3946.428 [3882.787–4006.868] | 3878.244 [3863.497–3892.824] | 1.73% |

The raw timing, profile and diagnostic data are under
`/tmp/yuuka-phase2-5-11.3wZQZG/2a` and
`/tmp/yuuka-phase2-5-11.3wZQZG/2c` on AI.

## Review and decision

Candidate 2c passed the native, state, logit and throughput gates.
The two read-only review rounds both returned `NO_BLOCKING_ISSUES`.
The first round prompted a documentation correction, full
reproduction details and a separate RNG copy to preserve the
existing native test sequence. The final Release CTest rerun
passed 10/10 after the fixture change.
