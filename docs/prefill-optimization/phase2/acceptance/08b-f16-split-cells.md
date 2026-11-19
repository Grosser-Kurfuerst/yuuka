# Stage 4b F16 split-key attention cells

The [Stage 4 plan](../stage4-long-context-attention.md) holds the F16
K/V leading dimension at 256 halves and the seed at 512 keys. The
production reference is `ab398ac`, which has the same kernels as
`704f211`. The [Stage 3B environment](07b-direct-q6k-attention.md)
provides the model hash and device configuration. These experiments
used the same AI container, Radeon 8060S gfx1151, ROCm 10.1.0,
HIP 7.16.26385, Go 1.24.4, Release builds and automatic power.

## Native route scan

The probe builds the production attention source with an isolated
eight-entry merge specialization. Direct, two-, four- and eight-way
attention use identical deterministic F16 pages, a reversed page
table, gated queries, head width 256 and 16 query heads over four KV
heads. The recorded GPU event interval includes every attention
launch: split samples include the seed, all partials and the merge.
The probe rotates the route order over two warmups and seven samples
per cell. A second process reverses the cell order and starts each
rotation two arms later. All results were finite and reproducible.
The maximum absolute delta across the 84 split-route comparisons was
`5.07e-6` versus direct WMMA. The 127-, 511- and 513-row cells
exercise partially occupied query tiles and nonuniform split ends.

Times are forward median / reverse median [pooled min–max] in ms.
| Rows | Start | Direct | Split 2 | Split 4 | Split 8 |
| ---: | ---: | --- | --- | --- | --- |
| 64 | 7680 | 3.121/3.289 [3.051–3.647] | 1.974/1.815 [1.766–2.062] | 1.837/1.882 [1.798–2.077] | 1.806/1.848 [1.776–1.994] |
| 64 | 8192 | 3.353/3.460 [3.266–3.879] | 1.990/1.948 [1.933–2.247] | 2.114/2.226 [2.100–2.255] | 1.921/2.019 [1.825–2.159] |
| 64 | 14336 | 6.177/6.168 [5.868–6.612] | 3.455/3.422 [3.356–3.662] | 3.534/3.402 [3.389–3.694] | 3.292/3.299 [3.208–3.476] |
| 127 | 7680 | 3.243/3.200 [3.157–3.699] | 3.316/3.389 [3.273–3.549] | 3.271/3.288 [3.222–3.519] | 2.987/3.002 [2.888–3.212] |
| 127 | 8192 | 3.414/3.422 [3.393–3.933] | 3.643/3.637 [3.601–3.857] | 3.446/3.544 [3.315–3.934] | 3.216/3.057 [3.034–3.329] |
| 127 | 14336 | 6.445/6.438 [6.282–6.700] | 6.966/6.778 [6.666–7.056] | 6.345/6.248 [5.808–6.616] | 5.387/5.276 [5.177–5.508] |
| 511 | 7680 | 15.378/14.685 [14.485–15.455] | 12.135/12.010 [11.489–13.209] | 10.801/10.758 [10.638–11.106] | 10.943/10.909 [10.789–11.961] |
| 511 | 8192 | 16.309/15.924 [15.734–16.360] | 12.585/12.720 [12.309–13.051] | 11.154/11.458 [10.992–11.642] | 11.544/11.717 [11.459–11.935] |
| 511 | 14336 | 28.720/28.041 [28.014–28.792] | 23.844/24.335 [23.643–24.478] | 19.511/19.248 [19.092–19.705] | 18.841/18.947 [18.689–19.177] |
| 512 | 4096 | 6.433/7.492 [6.372–7.607] | 6.102/6.606 [5.974–6.801] | 6.402/6.520 [6.059–6.605] | 6.795/6.775 [6.715–7.396] |
| 512 | 6144 | 11.677/11.868 [11.624–11.902] | 9.397/10.254 [9.275–10.527] | 8.825/9.260 [8.728–9.381] | 9.164/9.316 [9.085–9.455] |
| 512 | 7680 | 15.138/14.366 [14.074–15.168] | 11.187/12.022 [11.044–12.103] | 11.097/10.569 [10.426–11.282] | 10.978/10.789 [10.650–11.098] |
| 512 | 8192 | 16.056/16.716 [16.029–16.743] | 12.667/13.260 [12.489–13.448] | 11.565/11.825 [11.326–12.243] | 11.709/11.659 [11.501–11.845] |
| 512 | 12288 | 25.730/25.075 [25.048–25.764] | 21.567/19.971 [19.885–21.634] | 17.315/16.976 [16.925–17.917] | 17.098/16.811 [16.452–17.201] |
| 512 | 14336 | 29.203/29.357 [29.109–29.416] | 24.181/23.768 [23.642–24.617] | 20.115/19.639 [19.427–20.598] | 18.910/18.852 [18.701–19.080] |
| 513 | 7680 | 14.923/15.403 [14.738–15.446] | 11.497/12.524 [11.396–12.594] | 10.858/11.366 [10.766–11.816] | 11.157/11.138 [10.920–11.263] |
| 513 | 8192 | 15.850/16.671 [15.658–16.727] | 12.847/13.540 [12.770–13.633] | 11.886/11.734 [11.401–12.400] | 11.943/11.628 [11.448–12.249] |
| 513 | 14336 | 28.018/28.736 [27.909–28.776] | 24.020/24.218 [23.757–24.779] | 20.319/20.293 [19.743–20.749] | 19.347/19.340 [18.968–19.754] |
| 1024 | 6144 | 22.418/20.596 [20.087–22.584] | 17.566/17.389 [17.068–18.121] | 17.314/17.258 [16.926–18.227] | 18.203/18.211 [18.133–19.019] |
| 1024 | 7168 | 25.874/26.787 [25.766–26.827] | 20.331/21.709 [20.233–22.522] | 19.933/19.863 [19.703–20.383] | 20.798/20.880 [20.548–21.828] |
| 1024 | 8192 | 30.966/28.925 [28.863–31.065] | 26.057/23.418 [23.250–26.826] | 22.143/22.945 [21.830–23.121] | 23.042/23.276 [22.496–23.768] |
| 1024 | 14336 | 50.395/52.639 [50.344–53.441] | 44.872/45.706 [44.792–46.217] | 37.933/39.139 [37.481–39.349] | 36.338/36.567 [35.816–36.940] |
| 2048 | 4096 | 27.459/23.698 [23.262–27.608] | 26.938/25.794 [25.348–27.108] | 26.895/25.986 [25.829–27.244] | 29.360/28.698 [28.494–29.926] |
| 2048 | 6144 | 41.665/44.515 [41.055–44.564] | 36.296/35.860 [34.945–36.841] | 34.850/35.250 [34.535–35.695] | 37.217/37.536 [37.130–37.658] |
| 2048 | 7680 | 57.074/52.035 [51.802–57.189] | 43.204/43.363 [42.884–44.175] | 42.331/42.219 [41.560–42.772] | 43.670/43.784 [43.388–44.029] |
| 2048 | 8192 | 55.173/60.201 [55.003–60.483] | 45.243/46.561 [44.858–47.072] | 43.940/44.460 [43.413–44.880] | 45.657/45.679 [45.143–46.015] |
| 2048 | 12288 | 86.346/86.068 [85.787–86.668] | 73.914/77.894 [73.833–78.072] | 63.945/66.999 [63.487–68.850] | 63.967/64.348 [63.175–65.189] |
| 2048 | 14336 | 103.500/101.539 [101.359–103.771] | 92.867/94.277 [90.390–95.117] | 75.240/75.200 [73.772–77.483] | 72.190/72.185 [71.756–73.511] |

Two splits reduce scratch but rarely minimize time for 512 or 2048
query rows. Eight splits help some long cells, especially at position
14336. Eight splits do little for many 512- and 1024-row cells and
increase partial storage. The runtime scratch formula is
`rows * heads * (splits * (256 + 2) + 1) * 4` bytes. For 512 rows
and 16 heads this gives 16,941,056 / 33,849,344 / 67,665,920 bytes
for 2 / 4 / 8 ranges. The 2048-row allocations are four times those
sizes. The probe reserves eight ranges once per cell to compare routes.

| Kernel | VGPRs | SGPRs | Dynamic LDS (bytes) | Spill counts |
| --- | ---: | ---: | ---: | --- |
| Direct WMMA | 256 | 33 | 51,200 | 0 |
| Seed | 98 | 40 | 49,152 | 0 |
| Split partial | 254 | 48 | 51,200 | 0 |
| Merge up to four | 23 | 19 | 0 | 0 |
| Merge up to eight | 26 | 34 | 0 | 0 |

Reproduction files are in `/tmp/yuuka-phase2-5-11.3wZQZG/4b` on AI.
From that directory compile `cell.hip` with
`/opt/rocm/bin/hipcc -std=c++17 -O3 --offload-arch=gfx1151 -I../3b/src/kernels/hip/prefill -I../3b/src/backend/hip/bridge cell.hip -o cell`.
Run `./cell 7 0` and `./cell 7 1` serially. Use
`-Rpass-analysis=kernel-resource-usage` for the compiler resource report.

## Whole-model trial

The trial broadens the four-way route to all tested aligned and
shortlisted tail row counts from position 6144 through an end at
16384. The direct arm omits split scratch. Both arms run the same
model, token IDs, chunk sizes, F16 pages and greedy continuation.
Prompt A repeats `The quick brown fox jumps over the lazy dog. `.
Prompt B repeats `The only global variable sets the program's default
concurrency limit. A second workload starts on Tuesday at six o'clock. `.
The test runs two pairs in direct/split/split/direct order for A and
split/direct/direct/split for B. Each measurement ends when the
final host logit row is available. The maximum covers prefill and
three greedy decode steps. The four token IDs match for each pair.

The trial compares cumulative split dispatch to entirely direct
WMMA. Earlier chunks may also split. In the 513-row case at
position 7680 the test explicitly submits the prefix first to
keep the final row count at 513. Whole-prompt measurements include
both submissions. The tail replay below controls for earlier work.

Times are median [min–max] in ms. Percentages are elapsed-time
reductions.
| Rows | Start | Chunk | Prompt A | A max logit delta | Prompt B | B max logit delta |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 64 | 7680 | 512 | 3819.7 [3769.3–3870.1] → 3687.1 [3686.2–3688.0] (+3.47%) | 0.239 | 3755.4 [3745.3–3765.4] → 3695.3 [3684.8–3705.9] (+1.60%) | 0.095 |
| 64 | 14336 | 512 | 8610.6 [8374.3–8847.0] → 7667.2 [7663.4–7671.0] (+10.96%) | 0.115 | 8337.7 [8329.7–8345.8] → 7644.4 [7640.7–7648.0] (+8.32%) | 0.035 |
| 511 | 7680 | 512 | 4006.7 [4003.7–4009.8] → 3914.5 [3913.3–3915.8] (+2.30%) | 0.162 | 3999.2 [3993.7–4004.6] → 3903.6 [3901.2–3906.0] (+2.39%) | 0.141 |
| 511 | 14336 | 512 | 8994.3 [8699.7–9289.0] → 8187.0 [8032.2–8341.7] (+8.98%) | 0.237 | 8645.0 [8642.8–8647.2] → 7917.0 [7916.5–7917.5] (+8.42%) | 0.150 |
| 512 | 6144 | 512 | 3041.8 [3031.5–3052.1] → 3014.5 [3013.6–3015.4] (+0.90%) | 0.142 | 3116.9 [3113.7–3120.0] → 3105.8 [3104.3–3107.3] (+0.36%) | 0.191 |
| 512 | 7680 | 512 | 3935.3 [3925.3–3945.3] → 3852.6 [3842.4–3862.9] (+2.10%) | 0.256 | 4017.7 [4016.7–4018.6] → 3934.5 [3927.8–3941.2] (+2.07%) | 0.134 |
| 512 | 8192 | 512 | 4268.9 [4268.1–4269.8] → 4215.6 [4196.2–4235.0] (+1.25%) | 0.152 | 4343.3 [4336.4–4350.2] → 4215.5 [4200.6–4230.5] (+2.94%) | 0.148 |
| 512 | 14336 | 512 | 8601.8 [8598.2–8605.3] → 7898.8 [7876.4–7921.2] (+8.17%) | 0.177 | 8722.3 [8717.1–8727.5] → 7958.2 [7956.3–7960.0] (+8.76%) | 0.167 |
| 513 | 7680 | 1024 | 3866.3 [3864.7–3867.9] → 3793.6 [3792.8–3794.4] (+1.88%) | 0.187 | 3868.6 [3859.7–3877.5] → 3845.4 [3820.0–3870.8] (+0.60%) | 0.161 |
| 513 | 14336 | 1024 | 8230.9 [8230.6–8231.2] → 7677.1 [7670.7–7683.5] (+6.73%) | 0.229 | 8220.2 [8207.7–8232.7] → 7694.1 [7683.9–7704.3] (+6.40%) | 0.119 |
| 1024 | 7168 | 1024 | 3791.7 [3781.5–3802.0] → 3733.9 [3729.6–3738.1] (+1.53%) | 0.256 | 3820.7 [3818.2–3823.2] → 3764.3 [3761.8–3766.8] (+1.48%) | 0.134 |
| 1024 | 14336 | 1024 | 8518.3 [8502.0–8534.5] → 7923.4 [7918.7–7928.1] (+6.98%) | 0.185 | 8561.5 [8560.0–8563.0] → 7984.5 [7974.4–7994.7] (+6.74%) | 0.176 |
| 2048 | 6144 | 2048 | 3809.1 [3806.9–3811.3] → 3773.1 [3771.8–3774.4] (+0.94%) | 0.256 | 3847.1 [3846.8–3847.4] → 3795.9 [3788.1–3803.8] (+1.33%) | 0.134 |
| 2048 | 8192 | 2048 | 5008.7 [5004.1–5013.2] → 4879.5 [4870.5–4888.4] (+2.58%) | 0.161 | 5025.6 [5020.0–5031.2] → 4887.8 [4887.4–4888.1] (+2.74%) | 0.146 |
| 2048 | 14336 | 2048 | 9227.7 [9221.2–9234.2] → 8628.7 [8610.1–8647.3] (+6.49%) | 0.192 | 9260.9 [9248.7–9273.0] → 8681.2 [8678.8–8683.6] (+6.26%) | 0.215 |

Prompt B's 1024-row position-14336 and 2048-row position-6144
samples overlapped a compilation in the first run. The table uses
the idle-host repeat for both cells and the repeat of 2048 rows at
position 14336. The original measurements remain in `model-cells.log`.

### Whole-prompt gain beyond the original automatic window

The final-candidate probe runs the same two text inputs and aligned
chunks. Both arms prefill the first 8192 tokens with the existing
four-range policy. The previous-policy arm omits split scratch after
8192, which reproduces the parent revision's direct dispatch there
for these aligned cases. The expanded-policy arm keeps the
production scratch binding. Both timings include both submissions
and end at host logits. The old-policy arm invalidates cached plans
at the boundary to change scratch binding. The first prefill and
the rest of the device computation otherwise use identical settings.

The probe source is `ship/src/engine/gpu/attention_previous_test.go`.
Run `go test -tags hip ./engine/gpu -run '^TestStage4BPreviousAuto$' -count=1 -v`
against the final Release build. Process samples are in
`ship/previous-auto.log`. The two texts use opposite arm order.
Timings are median [min–max] in ms.

| Rows | Start | Chunk | Prompt A | A max logit delta | Prompt B | B max logit delta |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 512 | 14336 | 512 | 8481.9 [8477.7–8486.1] → 7849.0 [7834.8–7863.1] (+7.46%) | 0.140 | 8510.8 [8506.1–8515.6] → 7898.3 [7891.5–7905.2] (+7.20%) | 0.155 |
| 1024 | 14336 | 1024 | 8420.0 [8412.0–8428.1] → 7914.0 [7909.0–7919.0] (+6.01%) | 0.203 | 8414.0 [8403.2–8424.9] → 7917.5 [7916.4–7918.5] (+5.90%) | 0.205 |
| 2048 | 14336 | 2048 | 9143.0 [9121.6–9164.3] → 8605.6 [8603.2–8608.1] (+5.88%) | 0.158 | 9129.3 [9126.2–9132.4] → 8625.0 [8619.7–8630.4] (+5.52%) | 0.279 |

### Isolated model suffix from the same prefix

For each tail the experiment saves a single populated prefix's GDN
snapshot and reuses the corresponding KV pages. Before each timed
suffix, it restores that snapshot and absolute position. The two
arms then perform precisely the same suffix submission with or
without split scratch. Snapshot and restore are outside the timer.
Each route gets two warmups and five interleaved repetitions.

| Rows | Start | Chunk | Prompt A | A max logit delta | Prompt B | B max logit delta |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 64 | 7680 | 512 | 76.0 [75.0–76.1] → 65.8 [63.6–71.2] (+13.33%) | 0.006 | 76.3 [76.2–76.3] → 66.1 [66.0–66.1] (+13.34%) | 0.009 |
| 64 | 14336 | 512 | 100.5 [100.1–100.6] → 80.2 [79.9–80.3] (+20.21%) | 0.006 | 100.6 [100.6–100.7] → 80.1 [80.0–80.3] (+20.37%) | 0.005 |
| 511 | 7680 | 512 | 318.1 [317.3–319.4] → 284.3 [283.8–288.1] (+10.64%) | 0.151 | 318.8 [317.8–321.5] → 286.8 [285.6–288.2] (+10.05%) | 0.114 |
| 511 | 14336 | 512 | 416.0 [414.6–417.0] → 349.3 [348.9–349.6] (+16.04%) | 0.328 | 418.1 [416.6–419.3] → 350.3 [349.7–351.1] (+16.22%) | 0.140 |
| 513 | 7680 | 1024 | 345.2 [341.1–348.6] → 315.9 [311.9–316.8] (+8.49%) | 0.204 | 339.0 [338.3–340.6] → 307.6 [307.2–308.2] (+9.27%) | 0.132 |
| 513 | 14336 | 1024 | 439.2 [437.6–443.2] → 378.1 [377.5–380.2] (+13.91%) | 0.198 | 438.0 [437.7–440.3] → 375.4 [375.0–377.8] (+14.30%) | 0.140 |

## Dispatch decision

| Effective query rows | Inclusive start position | Automatic route |
| --- | --- | --- |
| 512, 1024, 2048 | 6144–14336 | Four ranges |
| 64, 513 | 7680–14336 | Four ranges |
| Remaining rows or positions | — | Unsplit |

This dispatch applies to the existing gfx1151 F16 gated GQA scratch
path. The Go planner binds scratch only to eligible query shapes.
Chunks configured to 512, 1024 or 2048 rows keep the existing scratch
reservation (33,849,344 / 67,698,688 / 135,397,376 bytes for this
model). A 64-row tail requires 4,231,168 bytes and a 513-row tail
requires 33,915,456 bytes, both within the four-range allocation of
the measured parent chunks. A configured 512-row chunk cannot produce
513 query rows. The native launcher decides from the execution
position and leaves the single cropped query on the unsplit path.

The 511-row suffix at position 14336 reaches a logit delta of 0.328
on prompt A despite faster attention. Keep 511 rows unsplit. Four
ranges give stable gains for the enabled shapes without increasing
the maximum scratch reservation. Some eight-range cells improve by
a further few percent in isolation and would double the partial
scratch. Eight ranges remain probe-only. The F16 512-key seed,
layout and ABI remain the same.

## Verification

The probe extensions in `src/engine/gpu/attention_cell_test.go` and
`src/engine/gpu/attention_tail_test.go` on AI produce the tables with
`go test -tags hip ./engine/gpu -run '^TestStage4B(Cells|TailReplay)$' -count=1 -v`.
Set `YUUKA_TEST_MODEL` to the [reference GGUF](05-cli-chunk-policy.md#baseline-and-protocol)
and link and load the matching gfx1151 HIP library. The final build
and test logs use the `ship` subdirectory of the experiment root.

The final candidate passed Release build and CTest (10/10), with
exact route comparisons at positions 6143/6144, 7679/7680 and
14336/14337 and a direct control at 511 rows. HIP-tagged Go build
and vet passed. The real-model command was
`YUUKA_TEST_ATTENTION_LONG=1 go test -tags hip ./engine/gpu -run '^TestGPU(AttentionPathsAgree|LongSplitAttentionMatchesDirect)$' -count=1 -v`.
The original 8192-token split case and the 7744-, 14849- and
16384-token configurations passed on both input texts. Final prefill
and all three continuation logits stayed within 0.257 with identical
greedy tokens. Go build, vet, test and race also passed without HIP
tags on the local checkout.

The read-only review ended with `NO_BLOCKING_ISSUES` after two
rounds. The second round verified the added old-policy comparison.
