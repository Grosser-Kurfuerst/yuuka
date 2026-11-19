# Stage 7 — Swizzle wide Direct MMQ workgroups for L2 reuse

Execution order 11. Run after the accepted Stage 1a row mapping and Stages 2–6, on their actual merged revision. Limit the trial to gfx1151 Direct MMQ with `n_out >= 8192` and multiple 64-token tiles. The principal cells are Q4_K 2560 → 9216 and Q4_K/Q5_K 2560 → 8192. The historical wide Q4_K trace gives a reason to try a different schedule. Its counter values are not the control for this experiment. [Roadmap attribution](roadmap.md)

## Iteration 1 — Establish the current mapping and cost

1. In [gemm_mmq_kernel](../../../kernels/hip/gemm/mmq_gemm.hip), the native launch grid is shared with generic MMQ. A Direct workgroup computes 128 output channels by 64 tokens. The two halves of physical grid X select consecutive token tiles within a 128-row Y band. The fixed Q4_K 512-row specialization hardcodes its geometry. Trace both that specialization and the runtime-shaped path for aligned and masked rows.
2. Use the [direct-only sweep](../../../bench/hip/prefill_sweep.cpp) with identical weights and prequantized input on both builds. Record M=128,129,192,511,512,1024,2048,4096 with width and dtype. Use two warmups/nine timed multiplies for these cells, one exclusive process. Record per-shape median/spread and **unprofiled** pp512/2048 final-host-logit time. Separately collect `rocprofv3` `L2CacheHit` and `FETCH_SIZE` per kernel, resource counts and launched vs useful tiles. Use the [counter scripts](../../../scripts/prefill_counters.sh) with the matching sweep ordering; the existing reducer assumes the default sweep.

## Iteration 2 — Test groups of 8 and 16 channel blocks

Keep each workgroup's inputs, K summation order, output layout and tile mask identical. Change only the calculation of its two tile coordinates before staging. One candidate mapping using the existing physical launch is:

```text
id      = blockIdx.y * gridDim.x + blockIdx.x
N       = n_out / 128                 # admitted shapes fill the channel tile
T       = 2 * gridDim.y               # physical token tiles, including padding
g       = id / (groupSize * T)
firstN  = g * groupSize
width   = min(groupSize, N - firstN)
within  = id - g * groupSize * T
n0      = 128 * (firstN + within % width)
m0      = 64 * (within / width)
```

For each channel group this cycles through nearby channel tiles, revisiting them on successive token tiles before moving to the next group. The last group uses its actual width (9216 outputs give 72 channel tiles, so group 16 ends with 8). The integer domain has exactly `N*T` workgroups. With a padded token grid, retain the uniform return for `m0 >= rows`, zero-fill inactive activation lanes, and predicate final stores for a partial tile. Use a compile-time specialization or one launch-time dispatch per shape. Keep linear mapping available for paired tests and all ineligible shapes. Verify the fixed-shape branch selects the candidate during its trial.

**Mapping gate.** Enumerate physical IDs on the host and require exactly one visit per logical tile, including masked/padded cases, both group widths and the partial last group. Use full product and output-guard tests at 129 and 511 rows, plus 512/2048 for both wide output sizes. For identical inputs require bitwise-identical outputs to the linear kernel, including the last live row. Run the [native product suite](../../../tests/hip/prefill_test.cpp), same-chunk full-logit byte comparisons at pp512/2048, and the HIP model tests. Reduction order is local to each tile.

## Iteration 3 — Decide whether to ship a mapping

- Compare linear, group 8 and group 16 with fixed chunk, IDs, F16 KV, Release flags and power settings. Shortlist by isolated unprofiled timings, then interleave linear/candidate/candidate/linear whole-prompt tests, repeating in reversed order. Require a repeatable unprofiled wall-time win on the pp512/2048 series or in the documented UMA contention scenario below. Record both prompt lengths. A change within control-to-control timing noise is inconclusive.
- For a contention-only win, define the CPU load before measuring: for example a pinned, fixed-thread-count sequential copy over two resident arrays larger than the last-level cache. Start it before warmup and keep it running throughout both arms. Record array sizes, affinity, achieved CPU bandwidth, temperature and GPU clocks/power. Repeat quiet and loaded pairs with identical load parameters. Report which workload improved.
- Run the **complete** `build/yuuka_prefill_sweep` and `--direct-mmq` regression matrix. Per cell require candidate median ≤ 1.01 × control median. Repeat noisy cells rather than averaging away a regression. Record compiler resources and paired L2 hit/fetch counters to explain observed timings. Better counters alone do not pass the gate.
- Enable only a passing group size for the measured wide shapes. If neither passes, keep linear dispatch and record the negative result, tested shapes and environment in Phase 2 acceptance. Use the [matched-input comparison protocol](../phase1/acceptance/08-upstream-comparison.md) for end-to-end reporting.

Estimate: 20–45 production HIP lines in one kernel file if a mapping passes, plus a small coverage fixture and the acceptance table. No planner, ABI, scratch or weight-layout change is necessary.
