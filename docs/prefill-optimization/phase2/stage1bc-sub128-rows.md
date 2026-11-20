# Stage 1b/1c: Sub-128-row projections and GDN preparation

> **Status:**
> - **Instrumentation** landed in `b833b4e` and **Step 1c (prepared GDN)** landed in `260094f` ([Stage 1c acceptance](acceptance/03-sub128-gdn.md)).
> - **Original Step 1b (`Q8_1` MMQ below 128 rows)** was rejected on numerical acceptance ([Stage 1b trial record](acceptance/04-sub128-projections.md)).
> - **Step 1b landed as two accuracy-preserving steps:**
>   1. **Step 1b-1 landed in `ac659b7`** — [FP32 streaming through 16 rows](acceptance/04a-streaming-short-tails.md) (fixed `<= 2e-3` FP32 gate, with aligned `float4` 16-row activation staging).
>   2. **Step 1b-2 landed in `c7dc298`** — [gfx1151 Direct FP16 WMMA for 17–127 rows](acceptance/04b-direct-fp16-wmma.md) (native kernel optimization preserving FP16 activation arithmetic and existing `<= 0.05` `TestGPUAttentionPathsAgree` gate).

Implementation plan for Steps 1c, 1b-1, and 1b-2 of the
[Phase 2 roadmap](roadmap.md#stage-1-p1-eliminate-prompt-length-and-tail-chunk-cliffs-4-independent-steps).

## 1. Goal and scope

Make prefill chunks with fewer than 128 rows fast without weakening any existing
model-level numerical gate. Sub-128 chunks appear as short prompts, as the new
tokens appended after a prefix-cache hit, and as the remainder of a prompt split
by `--chunk-size` (`512 + 1`, `512 + 18`, `512 + 88`, `2048 + 2`).

Pre-Step-1b baseline timings at `260094f` (after Stage 1a + Step 5a/5b + Step 1c,
F16 KV, `chunk=512`; see [04a](acceptance/04a-streaming-short-tails.md) and
[04b](acceptance/04b-direct-fp16-wmma.md) for the accepted post-change runs):

| Prompt (chunks) | `260094f` baseline median | Pre-Step-1b bottleneck | Handled by |
| ---: | ---: | --- | --- |
| 2 | 85.34 ms | 2-row batched GEMV over all 32 layers | Step 1b-1 |
| 18 | 58.25 ms | Naive FP16 WMMA (`32×64`, `K=32`, LDS `c_tile`) | Step 1b-2 |
| 64 | 80.67 ms | Naive FP16 WMMA (`64×64` wide / `32×64` tall-K) | Step 1b-2 |
| 96 | 108.83 ms | Naive FP16 WMMA | Step 1b-2 |
| 127 | 116.10–119.04 ms | Naive FP16 WMMA (vs `67.70 ms` at 128) | Step 1b-2 |
| 513 (`512 + 1`) | 292.09 ms | 1-row tail costs `~76.5 ms` on batched GEMV | Step 1b-1 |
| 530 (`512 + 18`) | 273.36 ms | 18-row tail costs `~57.8 ms` on naive FP16 WMMA | Step 1b-2 |
| 600 (`512 + 88`) | 320.45 ms | 88-row tail costs `~104.9 ms` on naive FP16 WMMA | Step 1b-2 |
| 2050 (`2048 + 2`) | 987.76 ms | 2-row tail costs `~88.4 ms` on batched GEMV | Step 1b-1 |
| Cached `512 + 2` | 91.88 ms TTFT | 2-row increment on batched GEMV | Step 1b-1 |
| Cached `512 + 18` | 66.92 ms TTFT | 18-row increment on naive FP16 WMMA | Step 1b-2 |

### Why `Q8_1` MMQ below 128 rows is excluded

In the [Stage 1b trial](acceptance/04-sub128-projections.md), routing `17..127`
rows to `Q8_1` MMQ met every latency gate (`127` rows dropped to `69.29 ms`)
but failed numerical acceptance:

- Switching `17..127` rows from FP16 activations (`~5e-4` relative step) to
  `Q8_1` int8 activations (`~4e-3` relative step) produced up to **`2.805252`**
  max absolute logit delta at `127` rows (`139,001` logits exceeding `0.3`) and
  `0.650768` at `64` rows.
- `TestGPUAttentionPathsAgree` ([attention_compare_test.go](../../../engine/gpu/attention_compare_test.go))
  checks `64` rows under a `0.05` gate that explicitly assumes the FP16 prefill
  route; under `Q8_1` projections, MMA-vs-tiled attention delta jumped from
  `0.0147` to `0.495347`.
- Running control's Generic MMQ gave byte-identical logits to Direct MMQ, and
  11 partial ablations still exceeded `2.41` at `127` rows, proving the drift
  is intrinsic to `Q8_1` activation quantization rather than a kernel bug.

Therefore, sub-128 optimization keeps **FP32 activations for `1..16` rows**
(Step 1b-1) and **FP16 activations for `17..127` rows** (Step 1b-2).

Out of scope:

| Item | Reason | Handled by |
| --- | --- | --- |
| Q8_0 `ssm_alpha` / `ssm_beta` (`2560 -> 32`) | Too narrow for streaming (`n_out < 1024`) and `128`-channel Direct tiles | Stage 2 |
| Prefill attention below 64 rows | Already under 1 ms per chunk | — |
| Default CLI chunk size | Swept after Steps 1b-1 and 1b-2 landed | Step 1d ([05](acceptance/05-cli-chunk-policy.md)) |
| Compact final-layer `M = 1` tail at `rows >= 128` | Keeps following the parent chunk's route | Stage 6a ([10a](acceptance/10a-compact-tail-gemv.md)) |

## 2. Step 1c summary (landed in `260094f`)

Step 1c lowered the prepared-GDN guard in [plan.go](../../../engine/gpu/plan.go)
from `rows >= 128` to `rows >= gdnPreparedMinRows` (`2`). Isolated sweeps showed
prepared GDN beating inline from `2` rows upward; real-model 127-row recurrent
time dropped from `12.596 ms` to `4.668 ms`, and all checked logits and
recurrent state digests matched control byte for byte. See
[03-sub128-gdn.md](acceptance/03-sub128-gdn.md).

## 3. Step 1b-1: FP32 `PrefillStreaming` for `1..16` rows

### 3.1 Design and measured crossover

`lm_head_kernel` in [lm_head.hip](../../../kernels/hip/gemm/lm_head.hip) streams
packed K-quant weights once for `1..16` rows with FP32 activations and FP32
accumulation (`SelectLMHeadPath`: `n_in % 256 == 0`, `n_out >= 1024`,
`1 <= rows <= 16`). No new kernel, opcode, or C ABI change is needed.

The isolated `yuuka_prefill_sweep --sub128` measurements already recorded in
[03-sub128-gdn.md](acceptance/03-sub128-gdn.md) show a shape-class split
between **tall-K / narrow-N** (`n_in >= n_out`, plus `2560 -> 1024`) and
**wide-N** (`n_out > n_in`: `2560 -> 4096 / 8192 / 9216`) K-quant matrices:

| Shape class (K-quant) | `M=1` Stream / Prev | `M=2` Stream / Prev | `M=4` Stream / Prev | `M=8` Stream / WMMA | `M=16` Stream / WMMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q4_K 2560 -> 1024` | 24 / 58 us | 36 / 66 us | 45 / 91 us | 53 / 92 us | **94 / 106 us** |
| `Q4_K 4096 -> 2560` | 47 / 156 us | 53 / 177 us | 69 / 226 us | 93 / 146 us | **138 / 172 us** |
| `Q4_K 9216 -> 2560` | 95 / 334 us | 110 / 372 us | 126 / 504 us | 187 / 306 us | **261 / 361 us** |
| `Q5_K 4096 -> 2560` | 51 / 179 us | 68 / 195 us | 77 / 240 us | 102 / 158 us | **144 / 184 us** |
| `Q6_K 9216 -> 2560` | 93 / 320 us | 107 / 376 us | 128 / 495 us | 167 / 380 us | **255 / 424 us** |
| `Q6_K 2560 -> 1024` | 23 / 57 us | 34 / 66 us | 38 / 104 us | 53 / 106 us | **89 / 118 us** |
| `Q4_K 2560 -> 4096` | 37 / 177 us | 66 / 206 us | 79 / 115 us | 115 / 115 us | 169 / **125 us** |
| `Q4_K 2560 -> 8192` | 71 / 323 us | 101 / 382 us | 120 / 149 us | 156 / 154 us | 255 / **164 us** |
| `Q4_K 2560 -> 9216` | 66 / 356 us | 103 / 423 us | 120 / 163 us | 159 / 166 us* | 259 / **175 us*** |
| `Q5_K 2560 -> 8192` | 80 / 348 us | 117 / 402 us | 136 / 153 us | 173 / **162 us** | 262 / **172 us** |

*\*Note: on `2560 -> 9216` FFN `gate/up`, `PrefillWMMA` fuses `gate + up + SwiGLU` into one `AddGemmSwigluAt` launch, whereas `PrefillStreaming` runs two `AddLMHeadAt` launches plus `AddSiluMul`. Therefore, wide-N shapes (`n_out > n_in`) lose to WMMA before `M = 16`.*

Dispatch rules for Step 1b-1 in [dispatch.go](../../../backend/hip/dispatch.go):

1. Add `PrefillStreaming` to `PrefillPath` (`String() -> "streaming"`).
2. Restrict `PrefillStreaming` strictly to **K-quant dtypes (`Q4_K`, `Q5_K`, `Q6_K`)**
   that pass `SelectLMHeadPath(dtype, rows, nIn, nOut)` (`nIn % 256 == 0`,
   `nOut >= 1024`, `1 <= rows <= MaxLMHeadRows`), leaving `F32`, `F16`, and
   `Q8_0` (`ssm_alpha`/`ssm_beta`) rules completely unchanged. Place this K-quant
   check **before** the existing `if rows < 4` and `if nIn > nOut && rows < 8`
   `PrefillBatchedGemv` early returns in `SelectPrefillPath` (after the
   `rows <= 0` and `dtype == tensor.F16` branches) so `1..7`-row K-quant chunks
   actually reach `PrefillStreaming`.
3. Use the single `MaxLMHeadRows` cap for every eligible shape. The measured
   wide caps of 4 and 8 failed the 16-token CPU gate. The approved extension
   to 16 rows uses aligned `float4` activation staging for complete K-quant
   tiles to meet the 16-token latency gate. Smaller or unaligned tiles retain
   scalar staging. The dot products and wave reduction are unchanged.
4. In [plan.go](../../../engine/gpu/plan.go) `prefillMatmulPath`, lower
   `hip.PrefillStreaming` to:
   ```go
   case hip.PrefillStreaming:
       return plan.AddLMHeadAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
   ```
   - `addPrefillFFN` already gates fused SwiGLU on `gatePath == hip.PrefillWMMA`,
     so `PrefillStreaming` cleanly runs separate `gate`, `up`, `AddSiluMul`, and `down`.
   - `addPrefillFullAttention` (`cropQuery`) passes `xOff = lastOff`
     (`uint64(rows-1)*hidden*4`, always 16-byte aligned for `hidden % 4 == 0`)
     and `queryRows = 1`, satisfying `validate_lm_head_plan_op`.
   - Tiny test fixture (`hidden = 32`) has `nIn % 256 != 0`, so `SelectLMHeadPath`
     returns `false` and tiny-fixture routes do not change.

### 3.2 Numerical policy for Step 1b-1

Because `lm_head_kernel` computes the same FP32 weight dequantization and FP32
accumulation as `gemv_batch` (differing only in lane/shuffle reduction order),
the earlier "never exceed control's CPU error on any prompt" clause caused
`4e-6` IEEE-754 rounding noise to fail on extra prompts
([04-sub128-projections.md](acceptance/04-sub128-projections.md#fp32-only-short-tail-trial)).
Step 1b-1 replaces that clause with fixed, reproducible FP32 bounds:

- **Against CPU FP32 reference** (`1`, `2`, `5`, `8`, `16` tokens on the Phase 1
  fixture, plus two independent English and Chinese prompts, on both F32 and F16 KV):
  identical top-1 token, identical 8-step greedy continuation, and
  **max absolute logit difference `<= 2e-3`** (expected `~1e-4 .. 1.1e-3`).
- **Against same-chunk control (`260094f`)** (`513 = 512+1`, `2050 = 2048+2` at
  `chunk=512`): identical top-1 token, identical 8-step greedy continuation, and
  **max absolute logit difference `<= 2e-3`** (observed `0.0003..0.0008`).
- **Existing real-model gates (all unchanged)**: `TestGPUMatchesCPU` (re-record
  the improved first-step delta in [kernels.md](../../kernels.md) and
  [benchmarks.md](../../benchmarks.md)), layer hidden-state check (`<= 0.1`),
  `TestGPUAttentionPathsAgree` (`<= 0.05` at 64 rows),
  `TestGPUChunkedPrefillLongContext`, and
  `TestGPURealModelSpeculativeMatchesGreedy` (verify 24/24 draft acceptance).

### 3.3 Changes and scope for Step 1b-1 (~120 LoC)

- [dispatch.go](../../../backend/hip/dispatch.go) (~25 LoC): `PrefillStreaming`,
  `String`, K-quant branch in
  `SelectPrefillPath` with measurement comment.
- [dispatch_test.go](../../../backend/hip/dispatch_test.go) (~40 LoC): update
  sub-16 K-quant expectations and verify `F32`, `F16`, `Q8_0`, and `nOut < 1024`
  stay on their prior routes.
- [plan.go](../../../engine/gpu/plan.go) (~5 LoC): `case hip.PrefillStreaming`
  in `prefillMatmulPath`.
- Real-model Go test (~50 LoC, opt-in via `YUUKA_TEST_MODEL`): verify `1, 2, 5, 16`
  against CPU (`<= 2e-3`) and `513, 2050` against control (`<= 2e-3`), plus
  English/Chinese short prompts.
- Documentation: update [kernels.md](../../kernels.md),
  [benchmarks.md](../../benchmarks.md) (`TestGPUMatchesCPU` recorded delta),
  [roadmap.md](roadmap.md), and add `phase2/acceptance/04a-streaming-short-tails.md`
  in [acceptance/README.md](acceptance/README.md).

### 3.4 Step 1b-1 acceptance gates

1. `go test -race ./...`, `go test -tags hip ./backend/hip/...`, full real-model
   `YUUKA_TEST_MODEL=... YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./...`, and
   CTest `10/10` all pass (including unmodified `TestGPUAttentionPathsAgree`).
2. Interleaved (`ctrl, cand, cand, ctrl`) `yuuka-compare` (F16 KV, `chunk=512`):
   - `2` tokens drops from `~85.3 ms` to `<= 35 ms`.
   - `513 (512+1)` drops from `~292 ms` to `<= 255 ms` (saving `>= 35 ms`).
   - `2050 (2048+2)` drops from `~988 ms` to `<= 950 ms` (saving `>= 35 ms`).
   - `16` tokens does not regress vs control.
   - `64`, `127`, `128`, `512`, `2048`, `8192` stay within `±1%`.
3. `bench/serving -prefix-cache -append-tokens 2` (512-token prefix) TTFT p50
   drops from `~91.9 ms` to `<= 42 ms`.
4. `yuuka-server` SSE first-content latency at `~2050` tokens (`4×512 + 2`)
   improves repeatably over `260094f` control (`>= 30 ms` saved).

---

## 4. Step 1b-2: `gfx1151` Direct FP16 WMMA for `17..127` rows

### 4.1 Why current `wmma_gemm.hip` is slow and how Direct FP16 WMMA fixes it

For `17..127` rows,
Go dispatch already selects `PrefillWMMA` (`YUUKA_OP_GEMM_WMMA`) and fused FFN
`AddGemmSwigluAt` (`YUUKA_OP_GEMM_SWIGLU`). Both enter
[wmma_gemm.hip](../../../kernels/hip/gemm/wmma_gemm.hip) with **zero C ABI or
Go planner changes**.

Four bottlenecks explain why current `wmma_gemm.hip` costs `109 ms` at `127`
rows vs `58 ms` for Direct MMQ at `128` rows:

1. **Narrow output tile and short K step**: `kBlockN = 64` and `BlockK = 32`
   (half the channel reuse of Direct MMQ's `128` channels, and double the barrier
   frequency).
2. **Row-major LDS layout and `c_tile` round-trip**: `rocwmma::load_matrix_sync`
   on row-major `32`-wide tiles incurs LDS bank conflicts, and the epilogue
   writes accumulators through shared memory `c_tile` with an extra
   `__syncthreads()` ([wmma_gemm.hip](../../../kernels/hip/gemm/wmma_gemm.hip#L387-L403))
   instead of storing directly from registers.
3. **Unpipelined `32×64` fallback on all tall-K matrices**: `launch_wmma_selected`
   ([wmma_gemm.hip](../../../kernels/hip/gemm/wmma_gemm.hip#L618-L626)) sends
   every `n_in > n_out` shape (`4096->2560` and `9216->2560`, covering all
   `attn_output`, `ssm_out`, and `ffn_down`) to single-buffered, 1-fragment
   `gemm_wmma_k32_kernel` even at `64..127` rows.
4. **Fused `gemm_wmma_swiglu_kernel` shares the same narrow tile and `c_tile`
   epilogue**, and accounts for 32 layers of `Q4_K 2560 -> 9216` (`64` weight
   matrices per chunk).

### 4.2 Kernel design (`kernels/hip/gemm/wmma_gemm.hip`, `#if defined(__gfx1151__)`)

Keep the exact FP16-operand / FP32-accumulator arithmetic (`__float2half(x)`
and `__float2half(scale * q - min)`), so every layer stays in the same FP16
activation domain as control and `TestGPUAttentionPathsAgree` stays at `<= 0.05`.

1. **Operand instruction and verified fragment mapping**:
   - Use the Wave32 RDNA 3.5 FP16 WMMA instruction
     (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` or `rocwmma` fragment loads
     over direct conflict-free LDS). First validate the exact lane-to-element
     mapping (`lane % 16` row/col and `2*e + (lane >> 4)` accumulator index,
     matching [attention_wmma.hip](../../../kernels/hip/prefill/attention_wmma.hip#L26-L28))
     against CPU dot products in [prefill_test.cpp](../../../tests/hip/prefill_test.cpp).
2. **Direct FP16 single-GEMM kernel (`gemm_wmma_direct_kernel<DType, TileM>`)**:
   - **Geometry (8 waves = 256 threads)**:
     - Output tile `128` output channels × `TileM` tokens (`TileM = 64` for
       `rows > 32`, `TileM = 32` for `rows <= 32` to halve wasted work on
       `17..32` row chunks).
     - K step `kStepK = 64` FP16 values (`128` bytes per row):
       - `A` weight tile (`128 × 64 × 2 B`) = `16 KiB`
       - `B` activation tile (`64 × 64 × 2 B`) = `8 KiB` (`4 KiB` for `TileM = 32`)
       - **Total LDS = `24 KiB`** (`20 KiB` for `TileM = 32`) — no scale array
         needed in LDS because weight scales/mins are folded when decoding to
         `__half`. Keeps **2 blocks resident per CU** (`48 KiB <= 64 KiB` LDS).
   - **Conflict-free LDS staging**:
     - Store `16`-row × `16`-K FP16 slices with XOR 16-byte chunk swizzling (or
       measured stride skew) so `ds_read_b128` reads stay bank-conflict-free
       (verified with `rocprofv3` `LDSBankConflict < 5%`).
     - Vectorized activation loads (`float4` -> `2 × __floats2half2_rn`) with
       row-bounds zero masking (`gm >= rows`).
     - Vectorized K-quant weight decode straight into LDS:
       - `Q4_K` and `Q5_K`: read packed 32-value subblock pairs with `uint4`,
         hoist `d`/`dm`/`sc`/`mn` once per 32-value subblock, decode both
         nibbles to `__half2` directly into LDS.
       - `Q6_K`: decode `16`-value groups via 16-bit-aligned `ql`/`qh` loads
         with hoisted per-16-value `d * sc` directly into LDS (accelerating the
         16 `Q6_K` `ffn_down` and 5 `Q6_K` `attn_v` layers below 128 rows as
         well).
   - **Direct register epilogue**:
     - Each wave owns 16 channels × `TileM` tokens (`4` or `2` 16-token
       fragments, `8` FP32 accumulators per fragment); write directly to global
       `out[gm * n_out + gn]` with `if (gm < rows)` mask — zero `c_tile` LDS
       round-trip and zero extra `__syncthreads()`.
   - **Dispatch eligibility inside `launch_gemm_wmma_f32out`**:
     - `#if defined(__gfx1151__)`
     - `dtype` in `{YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q5_K, YUUKA_DTYPE_Q6_K}`
     - `n_in % 256 == 0`, `n_out % 128 == 0`, `17 <= rows <= 127` (the
       final-layer cropped `rows == 1` uses `launch_wmma_selected`)
     - Pointer alignment: `uintptr_t(x) % 16 == 0` for all three dtypes, and
       `uintptr_t(w) % 16 == 0` for `Q4_K`/`Q5_K` (whose row strides are
       multiples of 16 bytes). For `Q6_K`, each 210-byte block and row stride
       (`2100`, `7560` bytes) is only 2-byte aligned, so `Q6_K` must use
       16-bit-aligned loads (as `q6k_span4` already does) instead of `uint4`
       casts.
     - Everything else (other GPUs, `F32`/`F16`/`Q4_0`/`Q5_0`/`Q8_0`, unaligned
       shapes, or `rows < threshold`) keeps the existing `launch_wmma_selected`.
3. **Direct FP16 fused SwiGLU kernel (`gemm_wmma_swiglu_direct_kernel<DType, TileM>`)**:
   - Because 32 FFN `gate/up` pairs (`Q4_K 2560 -> 9216`) are the single largest
     sub-128 projection family, upgrade `launch_gemm_wmma_swiglu_f32out` with
     the same direct register-epilogue pattern:
     - `128` channels × `TileM` tokens at `kStepK = 32` (`A_gate` `8 KiB` +
       `A_up` `8 KiB` + `B` `4 KiB` = **`20 KiB` LDS**, 2–3 blocks/CU), OR
       `64` channels × `TileM` tokens at `kStepK = 64` (`24 KiB` LDS) —
       benchmark both in Step 1b-2 and keep the faster one with zero VGPR spills.
     - Compute `silu(gate[t][e]) * up[t][e]` directly in registers and store to
       `out[gm * n_out + gn]` with `if (gm < rows)` mask (no `c_tile` LDS).
   - Also check whether `dequant_f16 + rocBLAS` or Direct FP16 WMMA wins on
     every model shape; keep Direct FP16 WMMA everywhere it matches or beats
     the prior kernel so no extra `w16` weight scratch buffer is allocated.

### 4.3 Changes and scope for Step 1b-2 (~320 LoC)

- [wmma_gemm.hip](../../../kernels/hip/gemm/wmma_gemm.hip) (~230 LoC):
  `gemm_wmma_direct_kernel`, `gemm_wmma_swiglu_direct_kernel`, vectorized
  `Q4_K`/`Q5_K`/`Q6_K` FP16 LDS stagers, and `#if defined(__gfx1151__)`
  eligibility dispatch in `launch_gemm_wmma_f32out` and
  `launch_gemm_wmma_swiglu_f32out`.
- [prefill_test.cpp](../../../tests/hip/prefill_test.cpp) (~60 LoC):
  add `Q4_K`, `Q5_K`, and `Q6_K` native tests for `yuuka_gemm_wmma` and
  `YUUKA_OP_GEMM_SWIGLU` at `n_in=256/512`, `n_out=128/256`, and partial row
  counts `rows = 17, 18, 32, 33, 64, 65, 96, 127`, including trailing output
  sentinel checks and bitwise reproducibility on repeated runs.
- [prefill_sweep.cpp](../../../bench/hip/prefill_sweep.cpp) (~30 LoC):
  add a `swiglu` timing cell for `Q4_K 2560 -> 9216` under `--sub128` so
  Direct FP16 SwiGLU is timed alongside `wmma` and `lmhead`.
- [dispatch.go](../../../backend/hip/dispatch.go) (optional, ~5 LoC):
  any proposed change below 17 rows requires a new numerical and latency
  comparison against the accepted FP32 streaming route.
- Documentation: update [kernels.md](../../kernels.md) (Section 6.2 WMMA GEMM
  and Section 6.4 dispatch table), [roadmap.md](roadmap.md), and add
  `phase2/acceptance/04b-direct-fp16-wmma.md` in
  [acceptance/README.md](acceptance/README.md).

### 4.4 Numerical and performance acceptance gates for Step 1b-2

1. **Native correctness & resources**:
   - CTest `10/10` passes (including all new partial-row `Q4_K`/`Q5_K`/`Q6_K`
     `wmma` and `swiglu` cells + sentinel preservation + bitwise repeat parity).
   - `scripts/kernel_resources.sh`: zero VGPR spills, zero private scratch,
     dynamic/static LDS `<= 30 KiB` (confirming `>= 2` blocks/CU).
   - Diagnostic `rocprofv3`: `LDSBankConflict < 5%` on the Direct FP16 kernels.
2. **Model-level numerical gates (zero tolerance relaxation)**:
   - **`TestGPUAttentionPathsAgree`** (`64` rows, `<= 0.05` gate): **PASSES
     unmodified** (expected `~0.015`, same FP16 activation regime as control).
   - **Same-chunk control comparison (`17`, `18`, `64`, `96`, `127`, `530`, `600`
     tokens, F16 KV)**: identical top-1 and 8-step greedy continuation, and
     **max absolute logit difference `<= 0.05`** against Step 1b-1 control.
   - **Against CPU FP32 reference (`17`, `64` tokens)**: identical top-1 and
     8-step greedy continuation, max absolute logit difference `<= 0.05`.
   - All existing real-model tests (`TestGPUMatchesCPU`, hidden-state `<= 0.1`,
     `TestGPUChunkedPrefillLongContext`, `TestGPURealModelSpeculativeMatchesGreedy`)
     pass unmodified.
3. **End-to-end and TTFT performance gates (interleaved `ctrl, cand, cand, ctrl`, F16 KV, `chunk=512`)**:
   - **Isolated sweep (`yuuka_prefill_sweep --sub128`)**: Direct FP16 WMMA beats
     legacy `wmma` across all 10 K-quant model shapes at `rows = 18, 32, 64, 96, 127`
     (target `>= 1.5×` faster on `2560 -> 9216`, `2560 -> 8192`, `9216 -> 2560`).
   - **Short prompts**: monotonic latency `t(18) <= t(64) <= t(96) <= t(127)`,
     with `t(127)` dropping from `~116–119 ms` to `<= 82 ms` (eliminating most
     of the `127 -> 128` cliff without changing activation precision).
   - **Tails**: repeatable reduction over Step 1b-1 control on `530 (512+18)` and
     `600 (512+88)`.
   - **Prefix-cache increment**: `bench/serving -prefix-cache -append-tokens 18`
     TTFT p50 drops repeatably from `~66.9 ms` (`<= 52 ms` target).
   - **Server SSE**: repeatable first-content latency improvement on `~530`-token
     chat prompts.
   - **No regression**: `128`, `512`, `2048`, and `8192` stay within `±1%`.

## 5. Staged execution order

| Step | Scope | Deliverable & commit |
| :---: | --- | --- |
| **1b-1** | `1..16` rows FP32 `PrefillStreaming` (`dispatch.go`, `plan.go`, `lm_head.hip`, tests) | Landed in `ac659b7` ([04a-streaming-short-tails.md](acceptance/04a-streaming-short-tails.md)) |
| **1b-2** | `17..127` rows `gfx1151` Direct FP16 WMMA + SwiGLU (`wmma_gemm.hip`, native/model tests) | Landed in `c7dc298` ([04b-direct-fp16-wmma.md](acceptance/04b-direct-fp16-wmma.md)) |

## 6. Expected vs. measured outcomes

Pre-implementation estimates compared against the accepted runs in
[04a](acceptance/04a-streaming-short-tails.md) and
[04b](acceptance/04b-direct-fp16-wmma.md):

- **After Step 1b-1 (`1..16` rows FP32 streaming, `ac659b7`)**:
  - `2` tokens: `29.35 ms` measured (est. `~29–34 ms`, from `84.81 ms` control)
  - `16` tokens: `47.91 ms` measured (from `54.91 ms` control)
  - `513 (512+1)` tokens: `238.69 ms` measured (est. `~242–250 ms`, from `292.45 ms` control)
  - `2050 (2048+2)` tokens: `936.83 ms` measured (est. `~935–945 ms`, from `994.28 ms` control)
  - Cached `512 + 2` TTFT: `35.8 ms` measured (est. `~35–40 ms`, from `91.7 ms` control)
- **After Step 1b-2 (`17..127` rows Direct FP16 WMMA, `c7dc298`)**:
  - `18` tokens: `35.48 ms` measured (est. `~42–48 ms`, from `58.22 ms` control)
  - `64` tokens: `47.54 ms` measured (est. `~50–58 ms`, from `80.71 ms` control)
  - `96` tokens: `60.08 ms` measured (est. `~68–76 ms`, from `107.97 ms` control)
  - `127` tokens: `66.30 ms` measured (est. `~72–82 ms`, from `117.92 ms` control)
  - `530 (512+18)` tokens: `253.48 ms` measured (est. `~255–265 ms`, from `275.37 ms` control)
  - `600 (512+88)` tokens: `274.99 ms` measured (est. `~285–298 ms`, from `321.79 ms` control)
  - Cached `512 + 18` TTFT: `43.2 ms` measured (est. `~48–54 ms`, from `67.1 ms` control)

## 7. Measurement environment

- Reference container `AI` (`192.168.254.101:11022`, `gfx1151`, ROCm 10.1.0,
  HIP 7.16.26385, Go 1.24.4), model SHA-256
  `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
  F16 KV, automatic power mode, one exclusive GPU process at a time.
- Build control and candidate in an isolated `/tmp` directory with a private
  `GOCACHE`, and remove every build artifact, profiler database, server
  process, and log after the run.
