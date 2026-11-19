# Stage 4a F16 attention LDS padding trial

## Candidate

The [Stage 4 plan](../stage4-long-context-attention.md) compares the
existing 256-half K/V leading dimension and a 264-half trial for
F16 KV at head width 256. The Stage 3B control is `704f211`. Both
Release builds run on AI gfx1151 with ROCm 10.1.0, HIP 7.16.26385,
Go 1.24.4 and automatic power. The model, input IDs, F16 KV pool
and concurrency-one settings match the [prior stage](07b-direct-q6k-attention.md).

The trial changes the F16 K/V copy, masked zero writes, score and
value fragments, seed's second K tile and all subsequent LDS base
offsets. Query and probability layouts retain their original sizes.
The shared-memory calculation uses the selected K/V stride for
normal, partial and max-only launches. Ordinary and partial kernels
reserve 51,712 bytes (previously 51,200). The seed reserves 49,664
(previously 49,152). F32 and other head widths retain their geometry.

## Native bitwise and isolated timing

The [native plan](../../../../backend/hip/bridge/yuuka_hip.h)
uses 16 query heads, four KV heads, 16-position pages in reverse
physical order and deterministic populated query and KV buffers.
It times the complete host submit/wait interval with two warmups
and seven repetitions per process. A forced split includes seed,
partials and merge. The probe selects unsplit WMMA by omitting split scratch.
The parity check compares the full output arrays for each cell.
All eight output pairs were finite and bitwise equal, including the 17-row partial-page
two-split case and unchanged F32 and head-width controls.

The two process orders are control/pad/pad/control and
pad/control/control/pad. Each cell reports the pooled median and full sample range
in milliseconds and percentage latency reduction.

| Rows | Start | Splits | KV / head width | A control → pad | A reduction | B control → pad | B reduction |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 512 | 0 | direct | F16 / 256 | 0.495118 [0.468655–0.532281] → 0.499482 [0.481969–0.507085] | −0.88% | 0.503549 [0.491366–0.507616] → 0.495489 [0.485134–0.508738] | +1.60% |
| 512 | 2048 | direct | F16 / 256 | 3.054762 [2.975296–3.110455] → 2.946678 [2.851056–3.011092] | +3.54% | 3.078706 [2.981808–3.116857] → 3.008267 [2.977951–3.048952] | +2.29% |
| 512 | 7680 | direct | F16 / 256 | 15.508550 [15.256750–15.782159] → 15.121782 [14.817340–15.376331] | +2.49% | 15.530040 [15.246061–15.806964] → 15.103308 [14.809685–15.408390] | +2.75% |
| 512 | 7680 | 4 | F16 / 256 | 10.887254 [10.836601–11.178741] → 10.377284 [10.133274–10.549781] | +4.68% | 10.875427 [10.525968–11.230978] → 10.452026 [10.254938–10.597540] | +3.89% |
| 2048 | 6144 | 4 | F16 / 256 | 35.696162 [35.064321–36.464142] → 34.183994 [33.661635–34.801628] | +4.24% | 36.264544 [35.781594–36.828785] → 34.305692 [33.844462–34.487498] | +5.40% |
| 17 | 270 | 2 | F16 / 256 | 0.117798 [0.103070–0.122666] → 0.115879 [0.101046–0.121444] | +1.63% | 0.118258 [0.103862–0.141952] → 0.116064 [0.101146–0.121824] | +1.86% |
| 512 | 0 | direct | F32 / 256 | 0.527748 [0.519007–0.537110] → 0.521442 [0.502036–0.541318] | +1.19% | 0.525208 [0.512756–0.536560] → 0.516858 [0.501965–0.526090] | +1.59% |
| 128 | 500 | direct | F16 / 128 | 0.138375 [0.126544–0.182907] → 0.138736 [0.126844–0.147151] | −0.26% | 0.138435 [0.126883–0.146631] → 0.137900 [0.127155–0.146631] | +0.39% |

## Whole-prompt timing

Both libraries run the same [matched-token driver](../../../../bench/prefill-compare/main.go)
with F16 KV and greedy sampling. Each process uses two warmups.
The samples per process are five at 512 tokens and three at
8192. Ranges are the full observed milliseconds.

| Prompt | Chunk | A control [range] | A pad [range] | A reduction | B control [range] | B pad [range] | B reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 512 | 201.666 [200.731–207.201] | 201.612 [200.991–203.488] | 0.03% | 202.375 [201.875–204.860] | 202.300 [201.941–203.774] | 0.04% |
| 8192 | 1024 | 3746.578 [3700.088–3761.968] | 3686.079 [3674.142–3715.008] | 1.61% | 3739.486 [3736.987–3749.625] | 3712.000 [3701.367–3727.243] | 0.74% |

All prefill logits matched bitwise and the four greedy IDs matched in
both orders. At 512 tokens the changes of 0.054 and 0.075 ms are
inside both arms' observed spread. At native position zero the
512-row timing changes sign across the two orders. At position
7680 the four-split operation improves in both orders.

## Profiler and decision

Separate `rocprofv3 --pmc LDSBankConflict` runs record six dispatches
per template. The medians below are in percent. Seed, partial and
direct keep the same VGPR allocations. All have zero spills and
private scratch.

| Template | Control conflicts | Pad conflicts | VGPRs |
| --- | ---: | ---: | ---: |
| Direct, 512 at position 0 | 43.852% | 36.315% | 256 |
| Prefix seed, 512 at position 7680 | 51.085% | 34.310% | 98 |
| Split partials, 512 at position 7680 | 44.803% | 37.188% | 254 |

The padding reduces measured bank conflicts and speeds the tested
long prompt. The short-prompt and early isolated timing gates are
inconclusive. Step 4b starts from the original 256-half layout,
as specified by the Stage 4 plan. The trial passed Release CTest
10/10 and the matched-model smoke. The unchanged production build
had already passed the full HIP suite in Stage 3B.

The trial code, native probe, timing samples, counter data and
compiler resource reports are under
`/tmp/yuuka-phase2-5-11.3wZQZG/4a` on AI.

The native probe source is `cell.cpp` in that directory. Invoke
`./cell ROWS POSITION SPLITS HEAD_DIM DTYPE REPS OUTPUT_F32` with
the corresponding Release library in `LD_LIBRARY_PATH`. The
direct setting is `SPLITS=0`. F32 and F16 use `DTYPE=0` and
`DTYPE=1` respectively. For the long model comparison use
`-prompt 8192 -chunk 1024 -warmup 2 -reps 3 -greedy-steps 3` on
each build's `compare` driver with the same model and token file.

The read-only review ended with `NO_BLOCKING_ISSUES` after two rounds.
