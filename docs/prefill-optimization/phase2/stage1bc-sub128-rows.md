# Stage 1b/1c: Sub-128-row projections and GDN preparation

> **Status:** Instrumentation landed in `b833b4e` and **Step 1c landed in `260094f`**
> ([Stage 1c acceptance](acceptance/03-sub128-gdn.md)). **Step 1b-1** uses FP32
> streaming for eligible K-quant projections through 16 rows. See the
> [accepted implementation](acceptance/04a-streaming-short-tails.md).
> The short Q8_1 MMQ route described below is a rejected trial.
> **Step 1b-2** uses Direct FP16 WMMA for 17–127 rows on gfx1151.
> See the [accepted implementation](acceptance/04b-direct-fp16-wmma.md).

Implementation plan for Steps 1b and 1c of the
[Phase 2 roadmap](roadmap.md#stage-1-p1-eliminate-prompt-length-and-tail-chunk-cliffs-4-independent-steps).
It depends on [Stage 1a](stage1a-direct-rows.md), which adds the row mask that
lets Direct MMQ run partial 64-token tiles.

## 1. Goal and scope

Make prefill chunks with fewer than 128 rows cost no more than a 128-row
chunk, and make small tail chunks cost about as much as a decode step instead
of a full slow pass. These chunks appear as short prompts, as the new tokens
appended after a prefix-cache hit, and as the remainder of a prompt split by
`--chunk-size` (`512 + 1`, `512 + 18`, `512 + 88`, `2048 + 2`).

Measured at `9da6e77` on the reference container (F16 KV, `chunk=512`):

| Prompt (chunks) | Median | Projections | GDN (`recurrent_chunk`) | Notes |
| ---: | ---: | ---: | ---: | --- |
| 64 | 84.63 ms | — | — | Slower than 128 tokens |
| 96 | 112.50 ms | — | — | Slower than 128 tokens |
| 127 | 124.17–125.86 ms | 109.84 ms | 11.90 ms | FP16 WMMA projections, inline GDN |
| 128 | 67.57–68.01 ms | 57.80 ms | 4.94 ms | Direct MMQ, prepared GDN |
| 513 (`512 + 1`) | 293.41 ms | — | — | The 1-row tail costs about 75 ms |
| 600 (`512 + 88`) | 320.90 ms | — | — | The 88-row tail costs about 103 ms |

Projections and GDN cross separate thresholds at 128 rows, so this plan splits
them into two independent changes:

- **Step 1c (GDN):** lower the `rows >= 128` threshold that selects the prepared
  recurrence in [plan.go](../../../engine/gpu/plan.go). Pure performance change;
  logits are expected to stay byte-identical.
- **Step 1b (projections):** route sub-128-row Q4_K/Q5_K/Q6_K projections to the
  fastest measured implementation. This changes projection arithmetic for these
  chunks and needs its own numerical gate.

**Commit order: Step 1c first, then Step 1b.** Step 1c is expected to be
byte-identical, so landing it first keeps Step 1b's numerical comparisons
limited to the projection route change.

Out of scope:

| Item | Reason | Handled by |
| --- | --- | --- |
| Q8_0 `ssm_alpha` / `ssm_beta` (`2560 -> 32`) | Too narrow for the streaming kernel (`n_out < 1024`) and for MMQ (`n_out < 64`) | Stage 2 |
| Prefill attention below 64 rows | Already under 1 ms per chunk at these lengths | — |
| Default CLI chunk size | Must be swept after the new kernels land | Step 1d |
| Compact final-layer `M = 1` tail | Keeps following the parent chunk's route, as today | Stage 6a |

## 2. Current routing below 128 rows

From `SelectPrefillPath` in [dispatch.go](../../../backend/hip/dispatch.go) and
`addPrefillRecurrent` in [plan.go](../../../engine/gpu/plan.go):

| Rows | K-quant projection route | Activation arithmetic | GDN |
| --- | --- | --- | --- |
| 1–3 | Batched GEMV | FP32 | Inline split-register |
| 4–7, `K > N` | Batched GEMV | FP32 | Inline split-register |
| 4–7, `K <= N` | rocWMMA GEMM | FP16 | Inline split-register |
| 8–127 | rocWMMA GEMM (gate/up fused with SwiGLU) | FP16 | Inline split-register |
| ≥ 128 | Int8 MMQ (Direct where the shape is allowlisted) | Q8_1 | Prepared (gfx1151) |

The candidate route families are:

- **Streaming matvec** (`lm_head_kernel` in
  [lm_head.hip](../../../kernels/hip/gemm/lm_head.hip)): reads every packed
  weight row once for a block of 1–16 rows with FP32 activations. It already
  serves the whole decode matvec family (23.4 ms for a profiled decode step) and
  the prefill output projection. Eligible when `n_in % 256 == 0`,
  `n_out >= 1024`, and `rows <= 16` (`SelectLMHeadPath`).
- **Direct MMQ with partial tiles** (after Stage 1a): one 64-token tile per 128
  output channels; the empty half of each 128-token group exits early.
- **Generic MMQ**: a 128-token tile. For 65–127 rows it executes exactly one
  token tile, the same as at 128 rows.

## 3. Design

### 3.1 Step 1c: prepared GDN below 128 rows

Replace the hard-coded `rows >= 128` in the prepared-recurrence condition with
a measured constant `gdnPreparedMinRows`. The prepared span reuses the FFN gate
scratch, and its width (`convDim - inner + 2 * rank = 4160` floats per row) is
below the FFN width (9216) at every row count, so the existing capacity check
still passes.

Expected numerics: the [Phase 1 prepared-GDN record](../phase1/acceptance/04b-gdn.md)
found that the four-lane prepared route matches the inline route byte for byte
in logits and in convolution and matrix state, including across chunk
boundaries. Lowering the threshold runs the same kernels on fewer rows, so
logits and state are expected to stay byte-identical. This is a prediction to
verify, not a measurement.

### 3.2 Step 1b: sub-128 projection routes

Add one route and two thresholds:

1. **`PrefillStreaming`.** A new `PrefillPath` that `prefillMatmulPath` lowers
   to `plan.AddLMHeadAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)`.
   Selected for K-quant projections when `rows <= streamingMaxRows`
   (at most 16) and `SelectLMHeadPath` accepts the shape. The FFN then runs
   gate, up, `SiluMul`, and down as separate operations, because the fused
   SwiGLU kernel is WMMA-only.
2. **Direct MMQ below 128 rows.** For shapes on the native Direct allowlist,
   select MMQ from `directMMQMinRows`. The native lower bound in
   `measured_direct_shape` ([mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip))
   moves from 128 to the same constant.
3. **Generic MMQ below 128 rows.** For K-quant shapes without a Direct
   specialization (in this model, the Q6_K `ffn_down` and `attn_v`), select MMQ
   from `genericMMQMinRows`, which is expected to sit just above 64 rows because
   the generic 128-token tile then wastes at most half its rows.

Go needs to know which shapes have a Direct specialization. Add
`DirectMMQShape(dtype, nIn, nOut)` to [dispatch.go](../../../backend/hip/dispatch.go)
as an explicit mirror of the native `measured_direct_shape` shape list,
following the existing pattern of `PrefillAttentionMMA`, which mirrors the
native attention rule. Both sides cite each other in comments, and
[dispatch_test.go](../../../backend/hip/dispatch_test.go) lists every model
shape.

All three thresholds are chosen from the sweep in Step 3 below, per dtype and
shape class, not assumed. The thresholds are valid for gfx1151 only; other
targets compile the generic MMQ kernel, as documented for the existing
dispatch table.

### 3.3 Numerical policy

Step 1b replaces FP16-activation WMMA and FP32 batched GEMV with FP32 streaming
or Q8_1 MMQ for these chunks, so logits are not expected to be bitwise equal to
control. The policy separates three comparisons:

- **Against the CPU reference** (the FP32 oracle), for prompts where the CPU
  forward pass is affordable (`1`, `2`, `5`, `16`, `17`, `64` tokens): same top-1
  token and same greedy continuation. For prompts that land on the FP32
  streaming route, the maximum absolute logit difference against CPU must not
  exceed control's. For prompts on the Q8_1 route, the bound is the open
  question in section 7.
- **Against control at the same chunking** (`96`, `127` tokens and the tail
  cases): same top-1 token and same 8-step greedy continuation; the maximum
  absolute logit difference is recorded and held to the same bound.
- **Across chunkings** (`512 + 1` vs one 513-row chunk, `512 + 18` vs one
  530-row chunk): same top-1 token; the maximum absolute difference is recorded
  as a diagnostic. Control already differs by 1.0674 at 513 tokens because the
  1-row tail uses FP32 GEMV while a 513-row chunk uses Q8_1 MMQ.

Speculative verification prefill runs 2–16 rows through the same projections,
so it moves to the streaming route as well. `TestGPURealModelSpeculativeMatchesGreedy`
must keep passing, and the acceptance rate is recorded.

## 4. Changes

### 4.1 Instrumentation (~130 lines)

- [prefill_sweep.cpp](../../../bench/hip/prefill_sweep.cpp):
  - Time the streaming route in `run_shape` when `rows <= 16`,
    `n_out >= 1024`, and `n_in % 256 == 0`.
  - Run the model's K-quant shapes (the Direct shapes listed in Stage 1a plus
    Q6_K `9216 -> 2560` and `2560 -> 1024`) at rows
    `{1, 2, 4, 8, 16, 18, 32, 48, 64, 88, 96, 127}`.
  - Add `RECURRENT` cells that time one `RECURRENT_CHUNK` plan operation at the
    model geometry (16 groups, 32 heads, state 128, convolution width 8192,
    inner 4096, kernel 4) with and without the prepared span, at the same row
    list.
- [prefill_profile_test.go](../../../engine/gpu/prefill_profile_test.go): accept
  a `YUUKA_TEST_PROFILE_ROWS` list so the per-operation attribution can run at
  1, 2, 18, 64, 96, and 127 rows instead of only 512.
- [bench/serving](../../../bench/serving/main.go): add `-append-tokens K`. A
  warmup request inserts the base prompt into the prefix cache; each measured
  request appends `K` request-specific tokens to it, so TTFT measures a
  prefix-cache hit followed by a `K`-row prefill. Reuse `-unique-prompts` from
  [Stage 5a/5b](stage5ab-snapshot-critical-path.md) for cache-miss runs, or add
  it here if that stage has not landed.

### 4.2 Step 1c (~5 lines)

- [plan.go](../../../engine/gpu/plan.go): replace `rows >= 128` in the
  prepared-recurrence condition with `rows >= gdnPreparedMinRows`, and document
  the measured crossover next to the constant.

### 4.3 Step 1b (~110 lines)

- [dispatch.go](../../../backend/hip/dispatch.go): `PrefillStreaming` and its
  `String`; `DirectMMQShape`; `streamingMaxRows`, `directMMQMinRows`, and
  `genericMMQMinRows` with measurement references; the updated
  `SelectPrefillPath` rules and doc comment.
- [dispatch_test.go](../../../backend/hip/dispatch_test.go): replace the
  existing `rows=64 -> WMMA` and small-row cases with the measured routes, and
  add one case per model shape on both sides of every threshold.
- [plan.go](../../../engine/gpu/plan.go): the `PrefillStreaming` case in
  `prefillMatmulPath`.
- [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip): set the
  `measured_direct_shape` lower bound to `directMMQMinRows` and cite the Go
  mirror.

### 4.4 Tests (~70 lines)

- A real-model Go test (opt-in through `YUUKA_TEST_MODEL`) that runs prompts of
  `1`, `2`, `5`, `16`, `17`, and `64` tokens on GPU and CPU and applies the
  section 3.3 checks, plus the tail and cross-chunk comparisons.
- Existing gates that must keep passing: `TestGPUMatchesCPU` (its recorded
  first-step difference, 0.00309944 on a 5-token prompt, is expected to change
  and is re-recorded), the layer-by-layer hidden-state check (`<= 0.1`),
  `TestGPUChunkedPrefillLongContext`, the tiny-fixture chunk tests, and
  `TestGPURealModelSpeculativeMatchesGreedy`. The tiny fixture's hidden size
  (32) is not a multiple of 256, so its routes do not change.

### 4.5 Documentation (~70 lines)

- [kernels.md](../../kernels.md): the section 6.4 dispatch table, the Direct
  MMQ row condition, and the prepared-GDN row condition in section 8.
- [docs/README.md](../../README.md) and [benchmarks.md](../../benchmarks.md):
  re-recorded `TestGPUMatchesCPU` and prefill-path logit differences.
- `phase2/acceptance/03-sub128-gdn.md` and `phase2/acceptance/04-sub128-projections.md`,
  listed in the phase 2 acceptance index.

**Total scope:** about 385 lines: about 130 lines of measurement tooling, about
115 lines of source, about 70 lines of tests, and about 70 lines of
documentation. No new opcode or ABI change: the streaming kernel already has a
plan operation.

## 5. Steps and acceptance

| Step | Work | Acceptance |
| --- | --- | --- |
| 1. Instrumentation | Section 4.1, on top of Stage 1a | Tools run on the container; control numbers recorded for every new sweep cell, the profile at each row count, and `-append-tokens 2` and `18` over a 512-token prefix. |
| 2. Step 1c | Measure `RECURRENT` cells; set `gdnPreparedMinRows` to the smallest row count from which prepared is not slower at any larger row count; section 4.2 | Logits and convolution and matrix state byte-identical to control at 1, 2, 17, 64, 127 tokens and at `512 + 1` and `512 + 18`. `recurrent_chunk` at 127 rows drops from 11.90 ms to at most 5.5 ms (prepared costs 4.94 ms at 128). No change at `>= 128` rows. |
| 3. Step 1b sweep | Experimental build with the native Direct lower bound removed; sweep every route at every row count and model shape | A recorded crossover table per dtype and shape class from which `streamingMaxRows`, `directMMQMinRows`, and `genericMMQMinRows` are read. |
| 4. Step 1b dispatch | Sections 4.3 | `go test ./backend/hip/...` passes; `rocprofv3` confirms the expected kernel per row band at 2, 18, 64, and 127 rows. |
| 5. Step 1b numerics | Section 4.4 | The section 3.3 checks pass; all existing gates pass; the bound in section 7 is met. |
| 6. End-to-end and TTFT | Unprofiled, control, candidate, candidate, control | See gates below. |
| 7. Docs and commits | Section 4.5 | Zero broken relative links, `git diff --check`; separate Conventional Commits for instrumentation, Step 1c, and Step 1b. |

Step 6 gates:

- **Short prompts:** monotonic latency `t(64) <= t(96) <= t(127) <= 1.05 × t(128)`.
- **Tails:** repeatable improvement over control for `512 + 1`, `512 + 18`,
  `512 + 88` (600 tokens), and `2048 + 2` at `chunk=512`.
- **Prefix-cache increments:** repeatable TTFT improvement over control for
  `-append-tokens 2` and `18` over a cached 512-token prefix.
- **Server:** repeatable SSE first-content latency improvement for chat prompts
  of about 530 and 2050 tokens.
- **No regression:** 128, 512, 2048, and 8192 tokens stay within `±1%`.

## 6. Expected gains

Estimates; acceptance uses the measured gates above.

- **127 tokens: close to `pp128`, about 68–72 ms (from about 125 ms).** With
  prepared GDN (about 5 ms instead of 11.90 ms) and MMQ at 127 rows, the work
  matches a 128-row chunk: Direct MMQ runs the same two 64-token tiles, and
  generic MMQ for Q6_K runs the same single 128-token tile.
- **64 and 96 tokens: below `pp128`.** One or two 64-token Direct tiles instead
  of FP16 WMMA; the exact values come from the sweep.
- **1- and 2-row tails: about 25–30 ms instead of about 75 ms.** A 1-row pass on
  the streaming route performs the same weight traffic as a decode step, which
  profiles at 23.4 ms with fused streaming matvecs. This would bring 513 tokens
  at `chunk=512` to about 245 ms (from 293 ms). Prefill does not use the fused
  pair and residual epilogues, so the tail may cost a few milliseconds more than
  a decode step.
- **88-row tail: about the cost of a 128-row chunk.** Two Direct 64-token tiles
  and prepared GDN put 600 tokens at about 285 ms (from 321 ms).
- **18-row tail:** one partial Direct tile or the WMMA route, whichever the sweep
  selects; no estimate before the sweep.

## 7. Open question

**Numerical bound for the Q8_1 route on short chunks.** The proposal is: same
top-1 and same 8-step greedy continuation as the CPU reference (or control,
where CPU is too slow), and a maximum absolute logit difference of at most
0.3, the bound Phase 1 used for route comparisons. For reference, the Q8_1 route
was measured at 0.0031 against an FP16-activation reference on a tiny prompt
and at 0.264 on a 2048-token prompt. This bound must be confirmed before Step 5
is accepted.

## 8. Measurement environment

- Reference container `192.168.254.101:11022` (gfx1151, ROCm 10.1.0,
  HIP 7.16.26385), model SHA-256
  `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
  F16 KV, automatic power mode, one GPU process at a time.
- Build control and candidate in an isolated `/tmp` directory with a private
  `GOCACHE`, and remove every build artifact, profiler database, server
  process, and log after the run.
