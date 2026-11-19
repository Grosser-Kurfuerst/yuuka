# Stage 1a: Direct MMQ for arbitrary row counts

Implementation plan for Step 1a of the [Phase 2 roadmap](roadmap.md#stage-1-p1-eliminate-prompt-length-and-tail-chunk-cliffs-4-independent-steps).

## 1. Goal and scope

Route every Q4_K and Q5_K projection with `rows >= 128` through the gfx1151
Direct MMQ tile, regardless of whether `rows` is exactly `128`, `512`, or
`2048`.

The problem this removes, measured at `9da6e77` on the reference container:
`pp128` takes `68.01 ms` while `pp129` takes `115.28 ms`. GDN stays on the
prepared path in both cases (`4.94 ms` and `5.33 ms`), but projections grow
from `57.80 ms` to `103.72 ms` (`+45.92 ms`) because `measured_direct_shape`
only accepts `rows` in `{128, 512, 2048}`. Every other row count falls back
to the generic `rocwmma` MMQ tile and is padded to a multiple of `128` rows.
The same allowlist makes `chunk=1024` slower than both `chunk=512` and
`chunk=2048` at 1024, 2048, and 8192 tokens.

Out of scope for this stage:

| Item | Reason | Handled by |
| --- | --- | --- |
| `rows < 128` projections and the GDN preparation threshold | Separate thresholds that need their own crossover sweeps | Step 1b / Step 1c |
| Q6_K projections (21 tensors) | Q6_K has no Direct specialization yet and stays generic | Stage 3 |
| 48 narrow Q8_0 `ssm_alpha` / `ssm_beta` projections | Independent of row count | Stage 2 |
| Go dispatch | [dispatch.go](../../../backend/hip/dispatch.go) already selects MMQ for every `rows >= 128`; the restriction exists only in the native launcher | No change |
| C ABI | No new opcode or descriptor field | No change |

## 2. Design

### 2.1 Keep the launch grid; exit and mask inside the kernel

The host grid is `(ceil(n_out / 64), ceil(rows / 128))`. The Direct kernel
remaps each workgroup to 128 output channels by one 64-token half of a
128-token group (see `gemm_mmq_kernel` in
[mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip)).

Two options compute the same set of 64-token tiles:

- **A. Native 64-row grid**: launch `(n_out / 128, ceil(rows / 64))`.
- **B. Unchanged grid (selected)**: a workgroup whose 64-token half starts at
  or beyond `rows` returns before it touches shared memory, and a partially
  filled half masks the absent token rows.

Both execute three 64-token tiles for 129 rows instead of the four that the
current padding implies. B is selected because:

- Full-tile cells (`128`, `512`, `2048`, `1024`, `4096`) keep the exact same
  workgroup-to-tile mapping and dispatch order, so there is no L2 locality
  risk for already-accepted shapes.
- The host grid geometry that
  [prefill_counters.py](../../../scripts/prefill_counters.py) validates stays
  unchanged, so the counter reduction needs no update.
- The early return is uniform across the workgroup and precedes the first
  `__syncthreads`, so it cannot deadlock a barrier.

### 2.2 Expected numerical behavior

Generic and Direct MMQ perform the same arithmetic in the same order:

- Both accumulate each 32-value K pair exactly in int32 with two
  `16x16x16` int8 WMMA operations.
- Both apply the per-32-value FP32 epilogue as
  `fmaf(wd * dsb.x, acc, fmaf(wm, dsb.y, sum))` in ascending K order.
- The Direct minimum `(-dm) * m` equals the generic `-(dm * m)` in IEEE
  arithmetic.

Switching `129`, `256`, `511`, and similar row counts from generic to Direct
is therefore **expected to leave logits bitwise identical** at the same chunk
size. This is a prediction derived from the code, not a measurement. If the
measured logits are not bitwise identical, stop and explain the difference
before accepting the change.

Zeroed digits and zeroed `float2` scales on masked rows contribute exactly
zero to valid outputs, and each output element depends only on its own token
row, so masking cannot change valid rows.

## 3. Changes

### 3.1 [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip) (~25 lines)

1. **Row mask in activation staging.** Give `stage_direct_activations` a
   `MaskRows` template parameter and a `rows` argument. A token row at or
   beyond `rows` stores zero digits and a zero `float2` scale:

   ```cpp
   template <bool MaskRows>
   __device__ __forceinline__ void stage_direct_activations(
       int8_t * tile_b, float2 * scale_b, const int8_t * xs, const float2 * xds,
       int64_t n_in, int64_t m0, int64_t k0, int64_t rows) {
       ...
       if (MaskRows && row >= rows) {
           dst[0] = make_uint4(0, 0, 0, 0);
           dst[kPairBytes / (2 * sizeof(uint4))] = make_uint4(0, 0, 0, 0);
           scale_b[pair * kDirectTileJ + token] = make_float2(0.0f, 0.0f);
           continue;
       }
       ... // existing vector loads
   }
   ```

2. **Early exit for empty halves.** Immediately after `m0` is computed in
   `gemm_mmq_kernel`:

   ```cpp
   if constexpr (kUseDirect && !FixedShape) {
       if (m0 >= rows) {
           return;  // whole 64-token half lies beyond the batch
       }
   }
   ```

3. **Staging call.** Call `stage_direct_activations<!FixedShape>(..., rows)`.
   The `FixedShape` variant (`M=512, K=2560, N=9216`) compiles without mask
   code, so the pp512 hot path is unchanged.

4. **Store guard.** Change the epilogue row check from
   `if (!kUseDirect && gm >= rows)` to
   `if ((!kUseDirect || !FixedShape) && gm >= rows)`. The channel check stays
   as is because Direct only accepts `n_out` divisible by 128.

5. **Allowlist.** In `measured_direct_shape`, replace
   `if (rows != 128 && rows != 512 && rows != 2048) return false;` with
   `if (rows < 128) return false;`, and update the comment above it, which
   currently states that every selected cell fills complete token tiles.

### 3.2 [prefill_test.cpp](../../../tests/hip/prefill_test.cpp) (~70 lines)

1. **Direct cells with partial token tiles**, chosen with narrow output widths
   to keep the CPU reference cost low:

   | Weights | Shape (K -> N) | Rows |
   | --- | --- | --- |
   | Q4_K | 2560 -> 1024 | 129, 192, 511 |
   | Q4_K | 2560 -> 4096 | 129 |
   | Q5_K | 4096 -> 2560 | 140 |

   Each case uses the existing `test_gemm_mmq`, which checks the Q8_1 digits
   exactly and the GEMM against the CPU product at the existing tolerance.

2. **`test_gemm_mmq_row_tail`**, a new mask test:
   - Quantize 256 activation rows once.
   - Run the GEMM with `rows = R` for `R` in `{129, 191, 255}` and with
     `rows = 256`, into output buffers sized for 256 rows and pre-filled with a
     sentinel value.
   - Require the first `R` rows to be **bitwise identical** to the 256-row
     result (row independence), and every row at or beyond `R` to still hold
     the sentinel (no out-of-range stores).

### 3.3 [prefill_sweep.cpp](../../../bench/hip/prefill_sweep.cpp) (~10 lines)

Add boundary row counts `{129, 192, 511, 1024}` for Q4_K and Q5_K at the
model's Direct shapes only. The existing all-dtype row list stays as is so the
sweep runtime does not grow for F16 and Q8_0.

### 3.4 Documentation (~60 lines)

- [kernels.md](../../kernels.md#gfx1151-direct-mmq-specialization): change the
  selection condition from "`M` in 128, 512 or 2048" to "`M >= 128`", and
  describe the early exit and row masking.
- Add `phase2/acceptance/01-direct-rows.md` with the measured results and
  environment metadata, add a `phase2/acceptance/README.md` index, and link it
  from the [module index](../README.md).
- Record the final numbers in the roadmap Step 1a entry.

**Total scope:** 4 file groups, about 165 lines. No new opcode, no Go changes.

## 4. Steps and acceptance

| Step | Work | Acceptance |
| --- | --- | --- |
| 1. Kernel | Section 3.1 | Release build passes. `scripts/kernel_resources.sh` reports no VGPR spills or private scratch for the Direct variants, dynamic LDS stays 30 KiB, and VGPRs do not rise enough to reduce residency below two workgroups per CU. |
| 2. Native tests | Section 3.2 | CTest passes 9/9. New cells stay within the existing MMQ tolerance against the CPU product. The row-tail test is bitwise identical on valid rows and leaves every sentinel intact. |
| 3. Kernel A/B | Section 3.3; control (`HEAD` before the change) and candidate built side by side on the container; `rocprofv3` confirms that 129- and 511-row calls dispatch `gemm_mmq_kernel<…, true, false>` | Every Q4_K and Q5_K model shape at boundary row counts is faster than control. `128`, `512`, and `2048` rows stay within `±1%`. `OccupancyPercent` stays near the 47% baseline. |
| 4. Model numerics | [prefill-compare](../../../bench/prefill-compare/main.go) exports logits from control and candidate at the same chunk size | At 129, 140, 256, 511, 513 (`chunk=2048`), and 1024 (`chunk=1024`) tokens, candidate logits are expected to be bitwise identical to control. The real-model HIP Go suite passes. |
| 5. End-to-end and TTFT | Unprofiled timing in control, candidate, candidate, control order; `yuuka-server` SSE first-content latency for chat prompts of about 140 and 300 tokens | See gates below. |
| 6. Docs and commit | Section 3.4 with measured data | Zero broken relative links, `git diff --check` passes, one atomic Conventional Commits change. |

Step 5 gates:

- **Primary:** a repeatable improvement over control at every tested
  non-allowlisted length `>= 128` (129, 140, 192, 256, 511, 513 at
  `chunk=2048`, and 1024 at `chunk=1024`).
- **No regression:** 128, 512, 2048, and 8192 tokens stay within `±1%` of
  control.
- **Chunk 1024:** no more than 3% slower than `chunk=512` at 1024, 2048, and
  8192 tokens.
- **Server:** at least 10 ms lower SSE first-content latency for the
  ~140-token chat prompt.
- **Control case:** `512 + 18` tails are outside this stage; record them
  unchanged as a reference.

## 5. Expected gains

Estimates derived from the `9da6e77` trace breakdown; acceptance uses the
measured gates above, not these numbers.

- **129 tokens: about 87–95 ms (from 115.3 ms).** The Direct-eligible part
  (Q4_K + Q5_K) costs 42.6 ms at 128 rows and needs one more 64-token tile at
  129 rows, estimated at 1.3–1.5x. Q6_K stays generic and padded to 256 rows
  (about 14 ms), Q8_0 stays at about 5.4 ms, and the remaining non-projection
  work stays at about 13 ms.
- **511 tokens: about 220–224 ms (from 252–255 ms).** It runs the same Direct
  tiles as 512 tokens, using the non-fixed-shape variant, which costs a few
  milliseconds more than the fixed-shape kernel.
- **`chunk=1024`:** expected to stop being slower than `chunk=512` and
  `chunk=2048`.

Step 1a alone cannot bring 129 tokens down to 78 ms: 129 rows need one more
64-token tile than 128 rows, and Q6_K still runs generic MMQ padded to 256
rows. Further reductions at this length come from Stage 3 (Q6_K Direct) and
Stage 2 (narrow Q8_0 projections).

## 6. Measurement environment

- Reference container `192.168.254.101:11022` (gfx1151, ROCm 10.1.0,
  HIP 7.16.26385), model SHA-256
  `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
  F16 KV, automatic power mode, one GPU process at a time.
- Build control and candidate in an isolated `/tmp` directory with a private
  `GOCACHE`, and remove every build artifact, profiler database, and process
  after the run.
