# Stage 3A Direct Q6_K FFN down

## Scope and implementation

The [Stage 3 plan](../stage3-q6k-direct-mmq.md) enables the reference
model's Q6_K FFN down shape first. The control is the accepted narrow
projection build at `2bfabbb`. Both are AI Release gfx1151 builds
with the same [model and token hashes](05-cli-chunk-policy.md#baseline-and-protocol),
ROCm 10.1.0, HIP 7.16.26385, Go 1.24.4 and automatic GPU power.
Model timing uses F16 KV and one greedy request at a time.

The [Direct MMQ kernel](../../../../kernels/hip/gemm/mmq_gemm.hip)
stages one 32-value Q6_K region as eight packed four-byte words.
The decoder uses the same nibble layout as the existing span decoder.
Each packed word uses two 16-bit loads because each block has a
210-byte stride. The scale slots store independent factors for
each 16-value slice. The kernel issues two signed WMMA operations
with separate zero accumulators and applies their factors in
low-then-high order. The activation contribution uses only the
Q8_1 scale, with no minimum correction.

The selector enables the existing runtime-shape Direct tile at
`9216 → 2560` from 128 rows onward with the established alignment
requirements. Keep `2560 → 1024` on Generic MMQ for this iteration.
The [native sweep](../../../../bench/hip/prefill_sweep.cpp) now
includes both Q6_K entries so the second shape is also available
as a control.

## Native verification

The [CPU reference](../../../../tests/hip/prefill_test.cpp) unpacks
every signed weight digit and the alternating positive and negative
16-value scales. It computes all outputs for 512 rows from the same
valid Q8_1 digits and scale/sum pairs sent to the device. For partial
launches it compares the complete live prefix and the output sentinel
after the last token. Each GPU case runs twice and the two buffers must
match byte for byte. The absolute CPU error bound is `2e-4`.

| Token rows | Full-tensor max absolute error | Guard |
| ---: | ---: | --- |
| 128 | 0 | Pass |
| 129 | 0 | Pass |
| 192 | 0 | Pass |
| 511 | 0 | Pass |
| 512 | 0 | Pass |

The launcher binds 30,720 bytes of dynamic LDS, with zero static LDS.

## Isolated timing

Both libraries ran the same `yuuka_prefill_sweep --direct-mmq` source.
The tool quantizes the activation once, then reports the mean of
nine timed multiplies after two warmups. Each arm here pools two
process means in control/candidate/candidate/control order on AI.
Timing includes host submit and wait overhead. Both arms use the
same packed weights and activation and run without profiling.

| Rows | Generic median (ms) | Vector Direct median (ms) | Reduction |
| ---: | ---: | ---: | ---: |
| 128 | 0.409 | 0.320 | 21.82% |
| 129 | 0.646 | 0.403 | 37.64% |
| 192 | 0.699 | 0.428 | 38.77% |
| 511 | 1.701 | 1.215 | 28.56% |
| 512 | 1.739 | 1.205 | 30.73% |
| 1024 | 3.518 | 2.697 | 23.35% |
| 2048 | 7.199 | 5.646 | 21.58% |
| 4096 | 15.143 | 11.226 | 25.87% |

The first candidate called the eight-value decoder four times per
32-value pair. Its 512-row median was 1.751 ms against 1.647 ms for
Generic, with additional losses at 1024 and 2048. Vectorizing the
unpack into four digits per operation reversed those regressions.
The 512-row result is 1.205 ms. It missed the 1.15 ms diagnostic
target by 0.055 ms and improved by 30.73% against the same-session
Generic control.

A second ABBA compared the vector kernel with an alternative that
applies the low-scale FMAs before issuing the second WMMA. It did
not provide a distinct crossover. At 512 rows the two process
means were 1.303/1.177 ms for the selected scheduling and
1.198/1.239 ms for the alternative. The final kernel retains the
single low/high epilogue loop.

## Dispatch trace and real-model smoke

The final build passed Release CTest 10/10. Resource output came from
`hipcc --offload-arch=gfx1151 -O3 -Rpass-analysis=kernel-resource-usage`.
The selected Q6_K templates use 110 VGPRs for complete rows and
112 VGPRs for masked rows, with zero spills and zero scratch on both.
A separate `rocprofv3 --kernel-trace --kernel-include-regex gemm_mmq_kernel`
run recorded `gemm_mmq_kernel<14u, true, false, false>` for complete
tiles and `<14u, true, false, true>` for masked tiles. The generic
Q6_K template remains active in the V projection control.

The unprofiled same-chunk smoke uses one warmup and three timed
prefills per process, with F16 KV and the shared fixture. Both
arms returned identical full prefill logit bytes and all nine greedy
IDs. Use the complete paired timings in Stage 3B for product-level
attribution.

| Prompt | Chunk | Control p50 (ms) | Direct FFN p50 (ms) |
| ---: | ---: | ---: | ---: |
| 128 | 128 | 64.424 | 62.973 |
| 129 | 512 | 85.148 | 80.227 |
| 511 | 512 | 211.456 | 201.850 |
| 512 | 512 | 208.993 | 202.777 |
| 2048 | 1024 | 831.875 | 814.185 |
| 8192 | 1024 | 3818.624 | 3741.755 |

Raw data and isolated builds are under
`/tmp/yuuka-phase2-5-11.3wZQZG/3a` on AI.

## Acceptance and review

Go build, vet, test and race passed on AI. HIP build, vet and
the full real-model `go test -tags hip -p 1 ./... -count=1 -v`
passed with the long F16 attention gate and the Stage 2 narrow
projection control wrapper. Two optional Step 1b wrappers and
three optional model/tokenizer fixtures skipped.

One read-only review returned `NO_BLOCKING_ISSUES`. The timing
description now lists both warmups and the host wait boundary.
The measured FFN down speedup and exact CPU and model parity
pass this iteration. The V projection remains for Stage 3B.
