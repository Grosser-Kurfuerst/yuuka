# Stage 7 Wide Direct MMQ workgroup mapping

The linear control uses the executable source from `e4cde6f`. Stage 5c,
6a and 6b recorded trials and kept that executable behavior.
Tests ran in the AI container on the Radeon 8060S (`gfx1151`) with
ROCm 10.1.0, HIP 7.16.26385, rocprofv3 1.4.1, Go 1.24.4 and automatic
power and clocks. See the [model and token hashes](05-cli-chunk-policy.md#baseline-and-protocol).
All timings are Release, with one GPU test process at a time.

## Tile mapping and correctness

All paths launch the same physical grid. One workgroup computes 128
output channels by 64 token rows. Linear mapping advances in channel
order for a token tile, then advances to the next token tile. Grouped
mapping walks 8 or 16 channel tiles at a time and revisits those channels
across token tiles before moving to the next channel group. In the
9216-channel shape, group 16 ends with a partial group of eight.
The staging, K order, activation layout and stores stay the same.

Host enumeration checked both widths (8192 and 9216) at
128/129/192/511/512/1024/2048/4096 rows for each mapping. Each logical
tile had one visit. For 129 or 192 rows at width 9216 the physical grid
launches 288 groups, of which 216 contain live rows. At 511 and 512 it
launches 576 useful groups. At 2048 it launches 2304 useful groups.
Inactive workgroups return before reading the operands, and partial
groups preserve the existing activation zero fill and store mask.

Three wide dtype/width pairs each ran full product checks at 129, 511,
512 and 2048 rows. The fixture used varying populated inputs, verified
every live output finite and written, and required all 256 trailing
guard values to survive. The 12 FP32 product buffers were bitwise
identical across linear, groups of eight, and groups of 16. The 9216-output
Q4_K 512-row case exercises the fixed-shape kernel.

## Isolated timing

Each cell uses two warmups and nine timed multiplies on the same packed
weights and prequantized input. The table reports the median [range] of
the per-process means in ms across three sweeps for each mapping.
Full timing data for all 80 direct cells are in the experiment logs.

| Dtype | Output channels | Rows | Linear | Group 8 | Group 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 8192 | 129 | 0.323 [0.323–0.326] | 0.309 [0.306–0.310] | 0.313 [0.313–0.317] |
| Q4_K | 8192 | 511 | 0.851 [0.821–0.882] | 0.815 [0.815–0.821] | 0.833 [0.827–0.839] |
| Q4_K | 8192 | 512 | 0.839 [0.823–0.854] | 0.815 [0.809–0.835] | 0.821 [0.819–0.834] |
| Q4_K | 8192 | 2048 | 3.531 [3.520–3.571] | 3.381 [3.366–3.383] | 3.459 [3.448–3.470] |
| Q4_K | 9216 | 129 | 0.364 [0.362–0.395] | 0.351 [0.347–0.353] | 0.351 [0.349–0.359] |
| Q4_K | 9216 | 511 | 1.014 [0.989–1.025] | 0.932 [0.925–0.933] | 0.932 [0.919–0.946] |
| Q4_K | 9216 | 512 | 0.992 [0.978–1.019] | 0.925 [0.915–0.926] | 0.951 [0.931–0.952] |
| Q4_K | 9216 | 2048 | 3.965 [3.877–3.970] | 3.794 [3.790–3.832] | 3.847 [3.837–3.855] |
| Q5_K | 8192 | 129 | 0.366 [0.359–0.366] | 0.325 [0.325–0.340] | 0.338 [0.336–0.372] |
| Q5_K | 8192 | 511 | 0.914 [0.904–0.951] | 0.855 [0.854–0.877] | 0.882 [0.875–0.884] |
| Q5_K | 8192 | 512 | 0.920 [0.907–0.927] | 0.871 [0.867–0.888] | 0.877 [0.872–0.877] |
| Q5_K | 8192 | 2048 | 3.760 [3.755–3.793] | 3.573 [3.569–3.583] | 3.663 [3.654–3.685] |

## Cache attribution

The three exploratory builds ran two separate rocprofv3 passes each.
Every pass produced the same 80 ordered sweep cells and 880 MMQ
dispatches. The parser joins by event ID within each database, drops
the first two dispatches per cell and takes the median of nine.
`L2CacheHit` is a percentage. `FETCH_SIZE` is reported in KiB
(the metric divides read volume by 1024). The table reports
linear / 8 / 16. These are profiled device counters.

| Dtype | Output channels | Rows | L2 hit % | Fetch KiB |
| --- | ---: | ---: | --- | --- |
| Q4_K | 8192 | 512 | 42.71 / 85.62 / 74.62 | 211958 / 37037 / 79240 |
| Q4_K | 8192 | 2048 | 40.52 / 87.70 / 70.64 | 935711 / 134113 / 435946 |
| Q4_K | 9216 | 512 | 41.23 / 85.72 / 78.76 | 250838 / 41269 / 81289 |
| Q4_K | 9216 | 2048 | 40.27 / 87.93 / 70.38 | 1065089 / 150658 / 441833 |
| Q5_K | 8192 | 512 | 37.37 / 85.35 / 70.35 | 272625 / 44648 / 107022 |
| Q5_K | 8192 | 2048 | 36.12 / 84.63 / 61.18 | 1212923 / 204484 / 723928 |

The eight-tile group gives higher measured reuse at the wider row
counts and cuts traffic on all three projections. The fixed Q4_K
512-row kernel uses 119 VGPRs in each exploratory build. The linear
build uses 24 SGPRs and each exploratory grouped build uses 26.
There are zero spills. Dynamic LDS remains 30 KiB. The full-group
specialization restores the fixed kernel to 24 SGPRs and retains
119 VGPRs with zero spills. The exact full-group build was profiled
again with the same protocol. In the fixed Q4_K FFN cell, the
result is a hit rate of 85.81% versus 41.23% for linear and
43,423 KiB fetched versus 250,838 KiB. At 2048 rows the same
projection rises from 40.27% to 87.76% and fetch volume falls
from 1,065,089 KiB to 148,876 KiB.

## Whole-prefill shortlist

The comparison driver measures through the final host logit row.
F16 KV, chunk 512, two warmups and five timed repetitions per
process apply to every arm. Two process rounds each contain two
runs per mapping in forward and reverse orders. Times are
medians [full range] in ms. Full final logits and four greedy IDs
match for both lengths.

| Tokens | Round | Linear | Group 8 | Group 16 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 206.158 [205.573–214.707] | 200.000 [197.755–203.626] | 200.531 [199.538–205.636] |
| 512 | 2 | 205.008 [204.142–208.917] | 199.404 [197.753–201.879] | 200.978 [199.805–204.688] |
| 2048 | 1 | 853.531 [848.922–864.853] | 813.691 [807.843–828.242] | 824.150 [819.575–829.101] |
| 2048 | 2 | 849.246 [841.478–854.092] | 821.933 [814.837–883.804] | 829.609 [824.794–835.931] |

## Full-group specialization

The [MMQ launcher](../../../../kernels/hip/gemm/mmq_gemm.hip) selects
a separate template instance for Direct Q4_K/Q5_K when the measured
wide output width fills groups of eight tiles and M is at most 4096.
The existing Direct admission gate supplies the lower limit of 128.
The specialized device index calculation uses complete groups, so it
needs one quotient and remainder over the token-tile count. The fixed
512-row FFN and the aligned and masked runtime shapes use that same
mapping. The other routes keep the original device mapping.

The full-group candidate repeats the populated product and output
guard comparisons and preserves every byte. The [committed native
regression](../../../../tests/hip/prefill_test.cpp) extends the
masked-row check to the wide Q4_K and Q5_K kernels and verifies
that the complete product writes a finite value at every output.
The two template variants agree at every live row in the partial
runs, with guard values untouched.

Two independently reversed control/candidate process orders
gave the following medians [range] in ms. The reference GGUF,
F16 KV, fixed chunk 512, token IDs, two warmups and five timed
runs per process are the same for both arms. The complete final
FP32 logit bytes and four greedy IDs match.

| Tokens | Order | Control | Full-group 8 | Saving, ms |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 206.348 [204.309–208.864] | 197.626 [197.155–199.177] | 8.722 |
| 512 | 2 | 204.986 [204.041–210.763] | 197.908 [197.196–198.587] | 7.077 |
| 2048 | 1 | 842.346 [838.589–866.966] | 817.788 [808.978–831.801] | 24.559 |
| 2048 | 2 | 846.402 [843.111–850.836] | 812.743 [810.049–817.925] | 33.659 |

A separate extraction of the gfx1151 device code from the two shared
libraries identified 19 code objects. Eighteen are byte-identical.
The MMQ object adds the swizzled kernels. All ten original linear
MMQ functions and both Q8_1 quantization functions have
byte-identical instruction bodies.

The narrower candidate reran the complete 80-cell Direct sweep with
three interleaved processes per arm. The 24 changed cells ranged
from 0.892 to 1.001 times the paired linear median, within the 1%
limit for every changed cell. Eleven other cells varied by more than
1%. Each also varied by more than 1% across separate runs of the
same linear library. For example, the original Q6_K 2560 → 1024
192-row control went from 0.101 to 0.130 ms in separate rounds,
while the candidate measured 0.120 ms.

The standard sweep ran all 171 printed shape/LM-head lines in each
of three independent processes per arm. It produced 653 operation
timings. The nine changed MMQ cells, which include activation
quantization in this sweep, were 1.5–5.7% faster. The raw 123
differences above 1% occurred on unchanged routes, and the six
control samples from the two separate linear rounds also span
more than 1% for those cells. The additional focused run isolates
these 123 cells and the 11 Direct-sweep outliers. It takes 21
samples per process at the original operation and dimensions in three
interleaved process pairs. The candidate median is still more than 1% slower for 33 cells.
The full sample ranges of 29 of those cells overlap. The remaining
four were replayed after 32 warmups with 101 timed calls per
process in eight reversed-order process pairs. The Q8_0 LM head
and the two short Q4_K BLAS cells then measured +0.34%, +0.95%
and -1.18%, respectively. The 16-row Q8_0 WMMA cell still
measured +1.64% in the pooled medians and required a shared-process
check to control for run-to-run differences.

Loading two libraries with independently allocated buffers also
produced shifts from +3.71% to +8.12% in the median paired ratios
for two copies of the same control library. A tighter probe
used the same GPU stream and allocations for both libraries.
It alternates which library submits first, uses the same prepared
weights and activations and times through event completion.
Each process runs 12 warmup pairs and 101 pairs of seven calls.
Four processes per library load order gave median paired changes
of +0.001% and +0.019%. The two control-only load orders gave
-0.016% and -0.006%. Holding memory placement and stream fixed reduces the
difference to the same scale as the control-only variation.
The shared-allocation comparison stays within the 1% gate.

## Decision

Enable the full-group-eight variant in the measured gfx1151
Direct Q4_K 2560 → 8192/9216 and Q5_K 2560 → 8192 cells with
128–4096 rows. It improves both whole-prefill lengths in both
orders, preserves the logits and output guards, and meets the
changed-cell limit in both matrices. The unchanged routes retain
their original device instructions. The group of eight passes the quiet-host gate.
CPU memory-contention performance is unmeasured.

## Verification

The checked-in native regression passed on the exact final AI build.
Release CTest passed 10/10. HIP Go build, vet and
`go test -tags hip ./... -count=1` passed with the reference
model loaded. The AI source tree included the earlier 16K cross-format
attention and GDN state lifecycle probes. The optional historical
binary comparisons require their respective prefill control environment
variables. For Stage 7 the new same-chunk matched-input comparisons
supply the direct control. Local non-HIP Go build, vet, test and race
test passed. Three read-only review rounds returned
`NO_BLOCKING_ISSUES`. The second round clarified the description of
linear launch geometry and raw timing outliers. The third round
confirmed those edits and the upstream timing table and protocol.

The AI experiment root is `/tmp/yuuka-phase2-5-11.3wZQZG/7`.
Use `build.sh`, `measure.sh` and `report.py` for the exploratory
mapping variants. The production source and library were verified
by `verify-specialized.sh`. The `specialized-report.py` report compares the
two matrices and both prompt pairs. The outlier probes use
`finish-evidence.sh`, `check-outliers.sh` and `shared-wmma.sh`.
Their raw logs are under `g0`, `g8`, `g16` and `ship`. The
`device-identity.log` records the device-object comparison.

## Final comparison with official llama.cpp v1.0.0

AI ran the [matched-input protocol](../../phase1/acceptance/08-upstream-comparison.md)
with the final Group 8 library and official `ggml-org/llama.cpp`
`v1.0.0` at `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`.
Both drivers use the verified GGUF, F16 KV, identical integer prompt
IDs, and time to the final host logit row. Batch limits are the
previously measured choices for each engine. Each round interleaves
two processes per engine, with reverse starting order in the
second round. Processes use two warmups and five measured prefills,
or three measured prefills at 8192 tokens. Times are medians
[full ranges] in ms. Throughput gain is the llama.cpp median
divided by the Yuuka median, minus one. Both engines returned
the same final top-1 token in every process at each length.

| Tokens | Yuuka / llama.cpp chunk | Round | Yuuka ms | llama.cpp ms | Yuuka throughput gain |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 128 / 128 | 1 | 59.081 [58.875–60.008] | 68.533 [67.972–69.632] | +16.00% |
| 128 | 128 / 128 | 2 | 59.499 [58.900–62.701] | 67.957 [67.428–68.335] | +14.21% |
| 512 | 512 / 512 | 1 | 195.141 [194.451–198.149] | 228.578 [225.973–230.247] | +17.13% |
| 512 | 512 / 512 | 2 | 196.271 [195.688–199.141] | 230.809 [227.681–234.565] | +17.60% |
| 2048 | 1024 / 512 | 1 | 808.156 [801.927–810.911] | 937.983 [927.276–951.311] | +16.06% |
| 2048 | 1024 / 512 | 2 | 811.868 [805.485–820.589] | 939.982 [934.537–953.650] | +15.78% |
| 8192 | 1024 / 256 | 1 | 3685.340 [3679.532–3705.444] | 4163.147 [4134.108–4170.712] | +12.97% |
| 8192 | 1024 / 256 | 2 | 3703.992 [3698.203–3747.320] | 4163.955 [4151.261–4175.006] | +12.42% |

The model and input token SHA-256 values were rechecked after both
rounds. To repeat this comparison in the AI experiment, run
`bash final-upstream.sh ship`. Raw samples are under `upstream`.
