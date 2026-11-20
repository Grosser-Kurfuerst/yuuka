# Phase 2: Real-world prefill optimization roadmap

## 1. Objective and scope

[Phase 1](../phase1/README.md) optimized single-request prefill on four power-of-two
benchmark lengths (`128`, `512`, `2048`, `8192`) with tuned chunk sizes, clearing the
+5% target against upstream `llama.cpp v1.0.0` at `512`, `2048`, and `8192` tokens and
reaching +0.61% (and -1.22% in a later paired check) at `128` tokens.

**Phase 2 targets real-world time-to-first-token (TTFT)** across interactive CLI runs,
multi-turn prefix-cache increments, and `yuuka-server` streaming requests, where prompt
lengths are arbitrary integers rather than exact multiples of `128`, `512`, or `2048`.

Scope: `qwen35` dense hybrid HIP execution on `gfx1151`. Previously measured negative
experiments remain excluded; see [decisions and rejected routes](../../benchmarks.md#decisions-and-rejected-routes)
and the [Phase 1 acceptance records](../phase1/acceptance/README.md).

---

## 2. Diagnostic measurement baseline

All measurements below were collected on `2026-11-19` at commit `9da6e77` on the
reference `gfx1151` container:

| Field | Value |
| --- | --- |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` (`2,740,937,888` bytes) |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 gated delta-net), hidden `2560`, FFN `9216`, vocab `248320` |
| Weight dtype mix | `Q4_K=131`, `Q5_K=48`, `Q6_K=22` (`16` `ffn_down`, `5` `attn_v`, `1` `token_embd`), `Q8_0=48` (`24` `ssm_alpha`, `24` `ssm_beta`), `F32=177` |
| Hardware / stack | AMD Ryzen AI Max+ 395 w/ Radeon 8060S (`gfx1151`, 40 CUs, Wave32), Linux `7.0.14-28-pve`, ROCm `10.1.0`, HIP `7.16.26385`, Go `1.24.4` |
| Power level | `auto` (`amdgpu` sysfs), single exclusive GPU process |
| Timing protocol | Fixed token stream through final host logit availability; `2` warmups + `5` reps for short/edge lengths, `1` warmup + `3` reps for `>=2048` tokens; `rocprofv3` traces use `1` warmup + `1` timed pass |

### 2.1 Unprofiled prompt-length and boundary-cliff sweep

| Prompt | Max chunk (chunks run) | KV | Median (ms) | Throughput (tok/s) | Mechanism isolated by trace |
| ---: | ---: | :---: | ---: | ---: | --- |
| 64 | 512 (`64`) | F16 | 84.63 | 756.2 | `<128` falls back to FP16 WMMA + unprepared GDN; slower than `128` tokens |
| 96 | 512 (`96`) | F16 | 112.50 | 853.3 | Same `<128` fallback path |
| **127** | 512 (`127`) | F16 | **124.17–125.86** | **1009.1–1022.8** | **Crosses two thresholds at once**: projections `109.84 ms` (FP16 WMMA) + GDN `11.90 ms` (unprepared) |
| **128** | 512 (`128`) | F16 | **67.57–68.01** | **1882.2–1894.4** | Hits `rows == 128` Direct MMQ (`57.80 ms` projections) + prepared GDN (`4.94 ms`) |
| **129** | 512 (`129`) | F16 | **115.28** | **1119.1** | **Pure MMQ allowlist cliff**: GDN stays prepared (`5.33 ms`), but projections jump `+45.92 ms` (`103.72 ms`) on Generic MMQ padded to `256` rows |
| 192 | 512 (`192`) | F16 | 123.62 | 1553.2 | Generic MMQ padded to `256` rows (`grid.y = 2`) |
| 256 | 512 (`256`) | F16 | 134.75–136.21 | 1879.4–1899.9 | `!= {128, 512, 2048}` runs Generic `rocwmma` MMQ (`117.02 ms` projections) |
| **511** | 512 (`511`) | F16 | **252.34–255.19** | **2002.5–2025.0** | 1 token below `512` loses fixed-shape Direct MMQ (`212.74 ms` vs `177.98 ms` projections) |
| **512** | 512 (`512`) | F16 | **216.10–217.60** | **2353.0–2369.3** | Hits fixed-shape Direct MMQ (`177.98 ms` projections, `21.82 ms` GDN) |
| **513** | 512 (`512 + 1`) | F16 | **293.41** | **1748.4** | `1`-row tail chunk executes a second full 32-layer pass (`~75 ms`, batched GEMV + unprepared GDN) |
| **513** | 2048 (`513`) | F16 | **290.63** | **1765.2** | Single chunk avoids second pass, but `513 != 512` falls back to Generic MMQ padded to `640` rows |
| **600** | 512 (`512 + 88`) | F16 | **320.90** | **1869.8** | `88`-row tail chunk hits `<128` FP16 WMMA + unprepared GDN (`+103.30 ms`) |
| 1024 | 512 (`2×512`) / **1024 (`1024`)** | F16 | 438.26 / **491.59** | 2336.5 / **2083.0** | `chunk=1024` misses `measured_direct_shape` allowlist (`397.97 ms` projections) |
| 2048 | 512 / **1024** / 2048 | F16 | 888.33 / **997.98** / 855.29 | 2305.5 / **2052.1** / 2394.5 | `chunk=2048` hits Direct MMQ; `chunk=1024` regresses on Generic MMQ |
| 8192 | 512 / **1024** / 2048 | F16 | 4039.59 / **4437.61** / 3874.33 | 2027.9 / **1846.0** / 2114.4 | `chunk=2048` wins; `chunk=1024` regresses on Generic MMQ |
| 8192 | 2048 (`4×2048`) | F32 | 4036.38 | 2029.5 | Default `F32` KV disables Split-Key Attention (`+162 ms` vs F16) |
| 8192 | 512 (`16×512`) | Q8_0 | 7609.02 | 1076.6 | `Q8_0` KV lacks WMMA Attention, falling back to scalar tiled kernel |
| 8256 | 2048 (`4×2048 + 64`) | F16 | 4009.28 | 2059.2 | `64`-row tail hits `<128` projection/GDN fallback plus unsplit `pos=8192` attention (`+134.95 ms` total; does not isolate the `>8192` split cap because `rows=64` is already ineligible for split) |

Cross-chunk numerical baseline note: on unmodified `9da6e77`, running the identical
`513`-token prefix as `chunk=512` (`512 + 1`, where row `513` uses FP32 `PrefillBatchedGemv`)
versus `chunk=2048` (`513` rows all on `Q8_1` MMQ) already produces a `1.0674` maximum
absolute logit difference (same top-1 token `37550`), whereas `1024` at `chunk=512` vs
`chunk=1024` (both `Q8_1` MMQ) has `0.0` difference. Differential gates must therefore
separate **same-chunking route parity** from **cross-chunking arithmetic differences**.

### 2.2 Session lifecycle and `yuuka-server` critical-path breakdown

Isolated session lifecycle medians at `prompt=128, chunk=512, kv=f16` (`1` warmup, `5` reps):

- `NewSession` (`50` `hipMalloc` + `48` `hipMemsetAsync` + `Synchronize` + decode plan build): **`2.87 ms`**
- First-prefill cold plan build delta (`first_prefill - warm_prefill`): **`0.32 ms`**
- `Snapshot` (`48` synchronous device-to-host copies totaling `48 MiB` of GDN state): **`7.95 ms`**
- `Close` (`50` `hipFree`): **`2.24 ms`**

Streaming HTTP SSE TTFT on `yuuka-server` (`--kv-type f16 --prefix-cache=false --max-concurrency 1`):
`132.79 ms` (`~140` tokens), `314.65 ms` (`~530` tokens), and `1042.59 ms` (`~2050` tokens).
Two distinct causes explain this gap against synthetic CLI numbers:

1. **Boundary cliffs on chat-rendered lengths**: `~140` tokens lands on Generic MMQ
   padded to `256` rows; `~530` (`512 + 18`) and `~2050` (`4×512 + 2`) pay `~75–80 ms`
   for their small tail chunk.
2. **Unconditional pre-emission `Snapshot()`**: [serving/task.go](../../../serving/task.go)
   calls `t.seq.Snapshot()` (`7.95 ms`) and `serializeLogits()` (`~0.5 ms`) synchronously
   *before* `sampleAndEmit(logits)` on every final prefill chunk, even when
   `--prefix-cache=false` (because cache enablement is only checked later in
   [serving/manager.go](../../../serving/manager.go) `Manager.finalize`).

### 2.3 `rocprofv3` kernel trace and hardware counter attribution

Timed single-pass kernel breakdown from `rocprofv3 --kernel-trace` and `--pmc` on
the real GGUF (`F16` KV):

| Kernel family / group | `pp128` ms (%) | `pp512` ms (%) | `pp8192` (`c=2048`) ms (%) | `pp512` hardware counters (`rocprofv3`) |
| --- | ---: | ---: | ---: | --- |
| `Q4_K` Direct MMQ (`2560->9216`, `n=62`) | 21.32 (29.4%) | 63.22 (30.3%) | 1073.19 (25.7%) | `Occ=47.3%`, `LDSConflict=0%`, `L2Hit=45.2%`, `MemBusy=42.5%`, `FETCH=14.77 GiB (244.8 GB/s)` |
| `Q4_K` Direct MMQ (other 5 shapes, `n=65`) | 12.21 (16.9%) | 36.17 (17.3%) | 657.30 (15.8%) | `Occ=43.2%`, `L2Hit=73.1%`, `MemBusy=42.3%`, `FETCH=3.51 GiB (103.1 GB/s)` |
| `Q5_K` Direct MMQ (`8192` & `2560`, `n=48`) | 11.35 (15.7%) | 32.38 (15.5%) | 571.35 (13.7%) | `Occ=47.2%`, `L2Hit=40.5%` (`qkv`) / `84.1%` (`out`), `FETCH=6.64 GiB (217.6 GB/s)` |
| **`Q6_K` Generic MMQ** (`ffn_down`×16, `v`×5) | **8.44 (11.7%)** | **25.32 (12.1%)** | **511.91 (12.3%)** | `Occ=42.3%`, `L2Hit=80.2%`, `FETCH=61.9 GB/s`, `VALU=4.85M/call` (`2.6×` `Q4_K` Direct) |
| **`Q8_0` WMMA K32** (`ssm_alpha/beta` `[2560,32]`, `n=48`) | **6.17 (8.5%)** | **8.79 (4.2%)** | 59.89 (1.4%) | `Occ=9.6%` (`grid=1×16`), `L2Hit=6.0%`, `LDSConflict=35.4%`, `0.16–0.46 TF/s` |
| GDN prepared core (`gdn_chunk_prepared`, `n=24`) | 2.96 (4.1%) | 11.46 (5.5%) | 215.58 (5.2%) | `Occ=38.2%`, `MemBusy=95.9%`, `L2Hit=70.9%` |
| GDN `conv1d` + `prepare` + `norm_gate` (`n=72`) | 2.18 (3.0%) | 9.97 (4.8%) | 192.20 (4.6%) | `conv1d`: `MemBusy=97.8%`, `L2Hit=64.2%` |
| Fused `SwiGLU + Q8_1` (`quantize_q8_1<true>`, `n=32`) | 0.77 (1.1%) | 6.68 (3.2%) | 100.37 (2.4%) | `MemBusy=94.9%`, `L2Hit=16.3%`, `FETCH=1.12 GiB (175.6 GB/s)` |
| Paged WMMA Attention (incl. 8K split + merge) | 0.76 (1.0%) | 4.57 (2.2%) | **579.91 (13.9%)** | `Occ=8.7%` (256 VGPRs), **`LDSConflict=44.0%`**, `MemBusy=56.5%` |
| Streaming LM Head (`2560->248320`, `n=1`) | 2.98 (4.1%) | 2.88 (1.4%) | 2.87 (0.1%) | `Occ=98.9%`, `MemBusy=99.1%`, `FETCH=545.6 MiB (198.4 GB/s)` |
| Elementwise / Norm / Q8_1 / RoPE / KV write | 2.01 (2.8%) | 6.45 (3.1%) | 203.46 (4.9%) | Memory-bandwidth bound (`Add` `315 GB/s`, `RMSNorm` `234 GB/s`) |
| Compact tail `M=1` Generic MMQ (`n=4–5`) | 1.61 (2.2%) | 1.46 (0.7%) | 1.47 (0.04%) | 1 active row in a `64×128` tile (>98% wasted MACs) |
| Inter-kernel GPU idle gaps (`676` gaps/chunk) | 1.53 (2.1%) | 1.61 (0.7%) | 8.51 (0.2%) | Median gap `2.1–2.3 us`; single-call plan submission is not launch bound |

---

## 3. Priority ranking

Items are ranked by **real-world TTFT impact × engineering ROI**:

| Priority | Optimization slice | Target real-world workload | Expected gain | Estimated scope |
| :---: | --- | --- | --- | --- |
| **P1** | **Eliminate non-power-of-two and tail-chunk cliffs** (arbitrary-row Direct MMQ, `<128` projection & GDN crossovers, post-kernel CLI chunk rule) | Arbitrary-length prompts, multi-turn prefix-cache increments, and non-multiple tail chunks | **Save 30–85 ms** across boundary and tail cases (`127`, `129`, `512+1`, `512+18`, `2048+2`) | 4 sub-steps, 3–4 files, ~160 LoC |
| **P2** | **Optimize narrow `Q8_0` projections (`ssm_alpha` / `ssm_beta` `[2560, 32]`)** | All prompt lengths (highest relative share on short/medium prompts) | **Save ~4–8 ms** per chunk; closes the remaining `pp128` +5% gap | 2–3 files, ~120 LoC |
| **P3** | **`gfx1151` Direct WMMA specialization for `Q6_K` (`9216->2560`, then `2560->1024`)** | All Direct-eligible chunks (`16` `ffn_down` and `5` `attn_v` tensors) | **Save ~3.0 ms** at `pp128`, **~10.0 ms** per 512 rows, **~180 ms** at 8K | 2 files, ~150 LoC |
| **P4** | **Long-context Attention fixes** (F16 LDS skew padding, `>8K` Split-Key window, `Q8_0` WMMA, cross-format Split-Key) | Long-context prompts (`>=2K` to `32K+`) and `F32` / `Q8_0` KV pools | **15%–35% faster** 4K–32K attention; **~2× faster** 8K `Q8_0` KV | 4 sub-steps, 6–8 files, ~280 LoC |
| **P5** | **Server prefix-cache `Snapshot` gating, off-critical-path ordering, and contiguous GDN state** | All `yuuka-server` streaming requests (both `--prefix-cache=false` and `true`) | **Save ~8.0–8.5 ms** on first-token SSE latency | 3 sub-steps, 3–4 files, ~110 LoC |
| **P6** | **(a) Compact `M=1` tail `GEMV_Q8_1`**, then **(b) fused `Residual Add + RMSNorm (+ Q8_1)`** | All requests (fixed single-row tail + sub-layer activation traffic) | **(a) Save ~1.1 ms** fixed tail; **(b) Save ~2.5–3.0 ms** per 512 rows | (a) 1 file, ~30 LoC; (b) 5–6 files, ~230 LoC |
| **P7** | **2D workgroup L2 cache swizzling for wide Direct MMQ tiles** | Sustained prefill or concurrent CPU/GPU contention on shared UMA memory | Lower GL2C miss traffic; retained only if unprofiled wall time improves | 1 file, ~30 LoC |

### Ranking rationale and flexible ordering notes

1. **P1 comes first** because non-allowlisted row counts (`129`, `140`, `256`, `511`, `1024`)
   and remainder tail chunks (`512+1`, `512+18`, `600`, `2048+2`) cause `30%–95%` slowdowns
   in interactive CLI and server traffic, and P3 (`Q6_K` Direct) only benefits arbitrary
   lengths once P1 routes them into Direct MMQ.
2. **Two lightweight slices can be pulled forward opportunistically**:
   - **P5a (skip `Snapshot` when `--prefix-cache=false`)** is a ~15-line fix in `serving`
     with zero kernel risk and immediately removes `~8.5 ms` of wasted work from uncached
     server runs; it can land alongside P1 when measuring server SSE TTFT.
   - **P6a (compact `M=1` tail `GEMV_Q8_1`)** touches only Go plan construction for five
     single-row ops and can be pulled ahead of P3 if P2 leaves `pp128` within `~1 ms` of
     the paired +5% upstream gate.
3. **P2 ranks above P3** because `48` tiny `Q8_0` `[2560, 32]` ops (`0.11%` of FLOPs) cost
   `6.2–9.3 ms` per chunk due to `9.6%` occupancy and `6.0%` L2 hit rate.
4. **P4, P6b, and P7** follow in order of workload breadth versus ABI/plumbing complexity.

### Recommended execution order

While Section 3 ranks items by impact category, concrete implementation follows the
dependency- and risk-ordered sequence below so each step cleans up measurements for the
next and isolates numerical changes:

| Order | Slice | Plan | Status | Why here |
| :---: | --- | --- | --- | --- |
| **1** | **Step 1a** (`rows >= 128` Direct MMQ) | [stage1a-direct-rows.md](stage1a-direct-rows.md) | **Landed** (`3caea06`, [01](acceptance/01-direct-rows.md)) | Removes the `{128, 512, 2048}` cliff (`128 -> 129` `+45.92 ms` projections) and adds the row-masking mechanism in Direct MMQ that Step 1b later reuses for `<128` rows. |
| **2** | **Steps 5a & 5b** (prefix-cache `Snapshot` gating & ordering) | [stage5ab-snapshot-critical-path.md](stage5ab-snapshot-critical-path.md) | **Landed** (`6e267d7`, `c09cd06`, [02](acceptance/02-snapshot-critical-path.md)) | Pure Go `serving` change (`~22` production LoC, zero kernel/ABI/numerical risk) pulled ahead of `1c/1b` to eliminate dead/critical-path D2H + logit serialization before benchmarking server SSE TTFT on `1c/1b`. |
| **3** | **Step 1c** (`rows < 128` GDN crossover) | [stage1bc-sub128-rows.md](stage1bc-sub128-rows.md#2-step-1c-summary-landed-in-260094f) | **Landed** (`b833b4e`, `260094f`, [03](acceptance/03-sub128-gdn.md)) | ~5-line threshold change in `plan.go` (`gdnPreparedMinRows = 2`), byte-identical to control (per Phase 1 `04b-gdn.md`); landing it before `1b` fixes GDN while isolating `1b`'s projection arithmetic changes. |
| **4a** | **Step 1b-1** (`1..16` rows FP32 `PrefillStreaming`) | [stage1bc-sub128-rows.md](stage1bc-sub128-rows.md#3-step-1b-1-fp32-prefillstreaming-for-116-rows) | **Landed** (`ac659b7`, [04a](acceptance/04a-streaming-short-tails.md)) | Routes `1..16` K-quant rows to `lm_head_kernel` with aligned 16-row `float4` activation staging under the `<= 2e-3` FP32 error bound; cuts `2`, `512+1`, `2048+2`, and cached `512+2` tails with all existing gates unmodified. |
| **4b** | **Step 1b-2** (`17..127` rows Direct FP16 WMMA) | [stage1bc-sub128-rows.md](stage1bc-sub128-rows.md#4-step-1b-2-gfx1151-direct-fp16-wmma-for-17127-rows) | **Landed** (`c7dc298`, [04b](acceptance/04b-direct-fp16-wmma.md)) | Upgrades `wmma_gemm.hip` (128-channel GEMM + 64-channel fused SwiGLU, `K=64`, conflict-free LDS, register epilogue), cutting `127` rows to `66.30 ms` while keeping FP16 activation logits bitwise identical to control. |
| **5** | **Step 1d** (CLI default chunk re-sweep) | [Stage 1d — CLI chunk policy](stage1d-cli-chunk-policy.md) | **Landed** (`ba37822`, [05](acceptance/05-cli-chunk-policy.md)) | The gfx1151 Qwen3.5-4B Q4_K_M CLI uses chunk 1024 for 1024–8192 input tokens with F16 KV when the flag is omitted. The 17-length sweep includes aligned and boundary prompts. |
| **6** | **Stage 2** (`ssm_alpha`/`ssm_beta` `Q8_0`) | [Stage 2 — Narrow Q8_0 projections](stage2-narrow-q8-projections.md) | **Landed** (`2bfabbb`, [06](acceptance/06-narrow-q8-projections.md)) | The equivalent WMMA pair keeps model logits and snapshots bitwise identical. Both upstream comparisons clear the 5% throughput gate at all four lengths. |
| **7** | **Stage 3** (`Q6_K` Direct WMMA) | [Stage 3 — Direct Q6_K MMQ](stage3-q6k-direct-mmq.md) | **Landed** (`bb56fec`, `704f211`, [07a](acceptance/07a-direct-q6k-ffn.md), [07b](acceptance/07b-direct-q6k-attention.md)) | Direct MMQ covers the aligned FFN down and attention V shapes from 128 rows. Both repeated full-prompt pairs improved at all four lengths with bitwise logits. |
| **8** | **Stage 4** (Long-context Attention) | [Stage 4 — Long-context attention](stage4-long-context-attention.md) | **Complete** (`ab398ac`, `ccf2ea2`, `a7eca5d`, `e4cde6f`; [4a](acceptance/08a-f16-lds-padding.md), [4b](acceptance/08b-f16-split-cells.md), [4c](acceptance/08c-q8-kv-wmma.md), [4d](acceptance/08d-cross-format-split.md)) | Step 4a kept the 256-half LDS layout after inconclusive early-cell timing; 4b/4c/4d landed F16 split extension, `Q8_0` unsplit WMMA, and typed `F32`/`Q8_0` split-key in independently verified ranges. |
| **9** | **Step 5c** (Contiguous GDN state) | [Stage 5c — Contiguous GDN state](stage5c-contiguous-gdn-state.md) | **Measured** (`8d50b1a`, [09](acceptance/09-contiguous-gdn-state.md)) | A single allocation preserved snapshot bytes and reduced API calls. The snapshot and lifecycle medians exceeded 1.5 ms, so per-layer allocations remain in production. |
| **10** | **Steps 6a & 6b** (`M=1` tail GEMV & norm fusion) | [Stage 6 — Compact tail and norm fusion](stage6-compact-tail-and-norm.md) | **Measured** (`8762a72`, `e548993`; [6a](acceptance/10a-compact-tail-gemv.md), [6b](acceptance/10b-residual-norm.md)) | Compact GEMV and residual norm fusion each pass numerics in their equivalent variants. Neither repeatedly meets its full-prefill timing gate, so existing dispatch remains. |
| **11** | **Stage 7** (2D L2 cache swizzling) | [Stage 7 — Direct MMQ workgroup swizzle](stage7-direct-mmq-swizzle.md) | **Landed** (`9aaf432`, [11](acceptance/11-direct-mmq-swizzle.md)) | Groups of eight channel tiles improve pp512 and pp2048 in both orders with bitwise logits and output guards. Dispatch is confined to the measured wide Direct Q4_K/Q5_K cells. |

---

## 4. Phased implementation plan

Each sub-stage lands as an atomic commit with before/after measurements on `gfx1151`.

### Stage 1 (P1): Eliminate prompt-length and tail-chunk cliffs (4 independent steps)

Because `128 -> 129` isolates the MMQ allowlist/padding cliff while `127 -> 128` crosses
both projection and GDN thresholds simultaneously, Stage 1 is split into four separately
verified steps:

- **Step 1a — Arbitrary-row Direct MMQ for `rows >= 128`**
  ([implementation plan](stage1a-direct-rows.md)):
  - *Changes*: In [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip), keep the host
    grid, return early from Direct workgroups whose 64-token half starts at or beyond
    `rows`, mask absent token rows in `stage_direct_activations` and the Direct store
    epilogue, and replace the `rows != 128 && rows != 512 && rows != 2048` allowlist in
    `measured_direct_shape` with `rows < 128` (keep `FixedShape = true` scoped to exact
    `M=512, K=2560, N=9216`). Add partial-tile Direct cells and a row-tail mask test to
    [prefill_test.cpp](../../../tests/hip/prefill_test.cpp).
  - *Acceptance*: Same-chunking logits vs control are expected to be bitwise identical;
    every tested non-allowlisted length `>= 128` improves repeatably over control
    (estimated `pp129` about `87–95 ms` from `~115.3 ms`, `pp511` about `220–224 ms`),
    `chunk=1024` is no more than 3% slower than `chunk=512`, the ~140-token server SSE
    first-content latency drops by at least 10 ms, and `pp128`, `pp512`, `pp2048`, and
    `pp8192` stay within `±1%`. Step 1a alone cannot reach `pp129 <= 78 ms` because Q6_K
    stays on generic MMQ padded to 256 rows until Stage 3.
  - *Measured*: pp129 improved from 113.107 to 86.931 ms, pp511 from 256.411
    to 217.648 ms, and the 140-token SSE first-content run from 125.826 to
    97.752 ms. At pp8192, chunk 1024 improved from 4517.265 to 3906.133 ms
    and chunk 4096 from 4514.581 to 3957.490 ms. Same-chunk logits matched
    bitwise in all 18 cells. [Stage 1a acceptance](acceptance/01-direct-rows.md)
- **Step 1b — Sub-128 projections split into two accuracy-preserving steps**
  ([implementation plan](stage1bc-sub128-rows.md), lands after Step 1c):
  - *Background*: The initial `Q8_1` MMQ (`>= 17` rows) trial was rejected because
    quantizing sub-128 activations to `Q8_1` moved `127`-row logits by `2.805252` and
    broke `TestGPUAttentionPathsAgree` (`0.495347 > 0.05` at `64` rows; see
    [Stage 1b trial record](acceptance/04-sub128-projections.md)). Step 1b therefore
    keeps FP32/FP16 activations and splits into:
  - **Step 1b-1 (`1..16` rows FP32 `PrefillStreaming`)**: Route eligible K-quant
    projections in [dispatch.go](../../../backend/hip/dispatch.go) and
    [plan.go](../../../engine/gpu/plan.go) to existing `AddLMHeadAt`
    ([lm_head.hip](../../../kernels/hip/gemm/lm_head.hip)). Both shape
    classes use the 16-row cap, with vectorized activation staging at 16 rows.
    - *Acceptance*: Fixed `<= 2e-3` max absolute logit difference vs CPU FP32 (`1, 2, 5, 16`
      tokens plus independent English/Chinese prompts) and vs same-chunk control (`513`,
      `2050`), identical top-1 and 8-step greedy tokens, all existing real-model gates
      unmodified; `2` rows `<= 35 ms`, `513` `<= 255 ms`, `2050` `<= 950 ms`, cached
      `512+2` TTFT `<= 42 ms`.
    - *Measured (`ac659b7`)*: **Landed.** `2` rows `84.81 -> 29.35 ms`, `16` rows
      `54.91 -> 47.91 ms`, `513` `292.45 -> 238.69 ms`, `2050` `994.28 -> 936.83 ms`,
      cached `512+2` TTFT `91.7 -> 35.8 ms`, and `~2050`-token SSE `996.53 -> 940.28 ms`;
      16-token CPU max delta `0.000489` (F32 KV) / `0.001063` (F16 KV).
      [Step 1b-1 acceptance](acceptance/04a-streaming-short-tails.md)
  - **Step 1b-2 (`17..127` rows `gfx1151` Direct FP16 WMMA)**: Upgrade
    [wmma_gemm.hip](../../../kernels/hip/gemm/wmma_gemm.hip) (`launch_gemm_wmma_f32out`
    and `launch_gemm_wmma_swiglu_f32out`) with Direct 128-channel GEMM and 64-channel SwiGLU tiles,
    vectorized `Q4_K`/`Q5_K`/`Q6_K` LDS decode, conflict-free staging, and direct
    register epilogue — zero C ABI or Go planner changes.
    - *Acceptance*: Native CTest `10/10` pass with partial-row sentinels, zero VGPR
      spills, LDS `<= 30 KiB`, `LDSBankConflict < 5%`; unmodified
      `TestGPUAttentionPathsAgree` (`<= 0.05` at 64 rows); `<= 0.05` max logit diff vs
      Step 1b-1 control (`17..127`, `530`, `600`); `t(127) <= 82 ms`, monotonic
      `t(18) <= t(64) <= t(96) <= t(127)`, cached `512+18` TTFT `<= 52 ms`, and `>= 128`
      lengths within `±1%`.
    - *Measured (`c7dc298`)*: **Landed.** Bitwise-identical logits to Step 1b-1 control
      at `17..127`, `530`, and `600` tokens; `18` `58.22 -> 35.48 ms`, `64`
      `80.71 -> 47.54 ms`, `96` `107.97 -> 60.08 ms`, `127` `117.92 -> 66.30 ms`, `530`
      `275.37 -> 253.48 ms`, `600` `321.79 -> 274.99 ms`, cached `512+18` TTFT
      `67.1 -> 43.2 ms`, and `~530`-token SSE `283.39 -> 258.91 ms`; `0.0%` bank
      conflicts across 539 model sweep dispatches.
      [Step 1b-2 acceptance](acceptance/04b-direct-fp16-wmma.md)
- **Step 1c — Sub-128 GDN preparation crossover**
  ([implementation plan](stage1bc-sub128-rows.md), lands before Step 1b because it is
  expected to be byte-identical):
  - *Changes*: Sweep `gdn_prepare_splitreg` + `gdn_chunk_prepared` versus inline
    `gdn_chunk_splitreg` at `rows = 8, 16, 18, 32, 64, 96, 127` and lower the `rows >= 128`
    guard in [plan.go](../../../engine/gpu/plan.go) to the measured win point.
  - *Acceptance*: Byte-identical logits and recurrent state against the same
    chunks with the old threshold. The sum of `127`-row recurrent operations
    must fall to `<= 5.5 ms`. The end-to-end short prompt, tail and cache
    increment gates apply after Step 1b, as specified in the sub-128 plan.
    [Step 1c acceptance](acceptance/03-sub128-gdn.md)
- **Step 1d — Re-sweep default CLI chunk policy on updated kernels**:
  - *Changes*: Re-sweep `chunk = 512, 1024, 2048` across `512..8192` (including `2049` and
    `4097` boundary lengths) *after* Steps 1a–1c land; only wire an automatic length-based
    default into [cmd/yuuka/main.go](../../../cmd/yuuka/main.go) (preserving explicit
    `--chunk-size`) if it wins across both aligned and `+1`/`+18` boundary lengths.
  - *Acceptance*: Recorded sweep table shows net TTFT improvement without hurting boundary
    prompts or exceeding transient memory budgets.

### Stage 2 (P2): Optimize narrow `Q8_0` `ssm_alpha` and `ssm_beta` projections

Evaluate two distinct arithmetic candidates separately because `alpha` and `beta` drive
exponential state decay and gating across hundreds of recurrent steps:

- **Candidate 2a — Shared-input FP32-equivalent narrow `Q8_0` pair**:
  - Pair `ssm_beta` and `ssm_alpha` (`2 × [2560, 32]` `Q8_0`) in [plan.go](../../../engine/gpu/plan.go)
    into one launch over the shared FP32 `p.norm` activation, using coalesced `Q8_0` block
    loads and warp reductions across `(64 channels, row_tile)` workgroups while preserving
    exact FP32 dequantized dot-product semantics.
- **Candidate 2b — `Q8_1` activation int8 dot-product route**:
  - Consume the group's existing `p.xQ8` / `p.xDS` buffers with the signed-operand
    `__builtin_amdgcn_sudot4` form used by [the gfx1151 matvec](../../../kernels/hip/gemv/gemv_i8.hip)
    over `Q8_0` weights. Because this quantizes the activation feeding `alpha`/`beta`, it
    changes GDN gate inputs relative to FP32.
- **Acceptance criteria**:
  - **Numerics**: Candidate 2a must match CPU `Q8_0` reference within `1e-5` tensor error
    and `0.05` model logit bound. Candidate 2b is eligible *only* if multi-chunk `conv` and
    `recState` matrices, full-vocabulary logits at `pp128`/`pp512`/`pp2048` (including
    two-chunk `512+512`), and `8` subsequent greedy decode tokens stay within the `0.05`
    logit gate and preserve identical tokens (cf. [Phase 1 GDN state gate](../phase1/acceptance/04b-gdn.md)).
  - **Performance**:
    - *Product gate*: Paired interleaved same-session comparison clears
      **`Yuuka median <= llama.cpp median / 1.05`** at `pp128` (and preserves >= +5% leads
      at `pp512`, `pp2048`, `pp8192`) rather than pinning a fixed millisecond constant that
      drifts with APU boost clocks.
    - *Diagnostic stretch target*: `<= 1.2 ms` combined GPU time for the 48 projections at
      `pp128`/`pp512` (down from `6.17–9.33 ms`). An implementation slightly above `1.2 ms`
      is still accepted if it cleanly passes the paired product gate and numerical checks.

### Stage 3 (P3): Add `gfx1151` Direct WMMA specialization for `Q6_K`

- **Concrete changes**:
  - [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip): Add `YUUKA_DTYPE_Q6_K` support to
    `stage_direct_weights` (16-bit aligned unpack of 32 contiguous values per 210-byte
    block) and the `kUseDirect` loop (two `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`
    instructions per 32-value pair, retaining **independent per-16-value FP32 scale
    application** and omitting the unused `wm` term). Enable the model's actual `Q6_K`
    projection shapes in order: first **`9216 -> 2560`** (`16` `ffn_down` layers), then
    **`2560 -> 1024`** (`5` `attn_v` layers). (`2560 -> 9216` does not appear as `Q6_K` in
    this GGUF and is omitted from dispatch.)
  - [prefill_test.cpp](../../../tests/hip/prefill_test.cpp): Full CPU product checks for both
    `Q6_K` shapes across multiple 256-value superblocks and non-aligned row counts.
- **Acceptance criteria**:
  - Full CPU tensor error `<= 2e-4` maximum absolute difference, zero VGPR spills, dynamic
    LDS `<= 30 KiB`.
  - **Independent two-level timing verification**: (1) isolated `Q6_K` `9216 -> 2560` at
    `M=512` microbenchmark (`<= 1.15 ms` diagnostic target, down from `~1.57–1.68 ms`), and
    (2) separate unprofiled end-to-end A/B at `pp128`, `pp512`, `pp2048`, and `pp8192`
    showing a repeatable whole-prompt improvement (do not extrapolate whole-prompt gain
    from the isolated kernel sweep).

### Stage 4 (P4): Long-context Attention optimization and KV-format expansion (4 steps)

Because `AddAttentionPrefillPagedSplitFormatAt` in [hip.go](../../../backend/hip/hip.go),
validation in [bridge.cpp](../../../backend/hip/bridge/bridge.cpp), `KeyOnly` staging, and
split template instantiations in [attention_wmma.hip](../../../kernels/hip/prefill/attention_wmma.hip)
are currently hardcoded to `F16`, Stage 4 is executed as four incremental steps across
`6–8` files:

- **Step 4a — `F16` LDS skew-padding experiment**:
  - Pad `kh` and `vh` row strides in [attention_wmma.hip](../../../kernels/hip/prefill/attention_wmma.hip)
    (e.g. `256 -> 264` halves, keeping total dynamic LDS `<= 64 KiB`).
  - *Acceptance*: Unprofiled whole-prompt `pp512`/`pp8192` and isolated attention latency
    improve repeatably, with `rocprofv3` `LDSBankConflict` (`< 5%` vs `44%`) recorded as
    diagnostic confirmation; bitwise or `< 1e-4` native parity.
- **Step 4b — Expand `F16` Split-Key to verified `>8K` positions and tail row counts**:
  - Measure isolated `512`- and `2048`-row chunks placed at `position = 4096, 6144, 7680, 8192, 12288, 14336`
    (direct vs 2/4/8 splits) so `>8K` behavior is isolated at equal eligible row counts.
    Verify whether the `512`-key prefix-maximum seed holds model logit drift within `<= 0.30`
    at `8K–16K` (in Phase 1, changing seed length moved max logit delta from `0.256` to
    `0.373`; see [Phase 1 long-attention record](../phase1/acceptance/07-long-attention.md)).
  - *Acceptance*: Update [prefill.hip](../../../kernels/hip/prefill/prefill.hip) only for
    `(rows, position)` cells with measured whole-attention wins and verified `<= 0.30`
    long-prompt logit differential and identical greedy continuation.
- **Step 4c — `Q8_0` KV support in unsplit WMMA prefill attention**:
  - Add `Q8_0 -> FP16` block decoding in `stage_kv_tile` and instantiate unsplit
    `attention_prefill_paged_wmma_kernel` for `YUUKA_DTYPE_Q8_0`.
  - *Acceptance*: `Q8_0` native and real-model parity within `Q8_0` KV tolerances; `pp8192`
    with `--kv-type q8_0` improves from `~1077 tok/s` to `>= 1850 tok/s`.
- **Step 4d — Cross-format (`F32` / `Q8_0`) Split-Key plumbing**:
  - Thread explicit KV `dtype` through [model.go](../../../engine/gpu/model.go),
    [plan.go](../../../engine/gpu/plan.go), [hip.go](../../../backend/hip/hip.go),
    [bridge.cpp](../../../backend/hip/bridge/bridge.cpp), `KeyOnly` max-scan staging, and
    partial WMMA instantiations.
  - *Acceptance*: Plan-validation negative tests pass; `F32` and `Q8_0` KV pools match
    their unsplit WMMA paths within `<= 0.30` max logit delta at `8K–16K` and improve
    unprofiled long-prompt latency.

### Stage 5 (P5): `yuuka-server` prefix-cache `Snapshot` and contiguous GDN state (3 steps)

- **Step 5a — Skip `Snapshot` and logit serialization when prefix cache is disabled**
  ([implementation plan for 5a/5b](stage5ab-snapshot-critical-path.md)):
  - *Changes*: Pass a `capturePrefix bool` flag from [serving/manager.go](../../../serving/manager.go)
    (`m.cache != nil`) into `newGenerationTask` in [serving/task.go](../../../serving/task.go) so
    `PrefillOnce` skips `t.seq.Snapshot()` and `serializeLogits(logits)` entirely when
    `--prefix-cache=false` (can be pulled forward alongside Stage 1).
  - *Acceptance*: Removes `~8.5 ms` of dead D2H/CPU work on `--prefix-cache=false` SSE TTFT;
    all `serving` tests pass.
  - *Measured (`6e267d7`)*: **Landed.** In-process snapshot cost was `2.56–3.27 ms` + `~0.14 ms`
    serialization; SSE first-content latency improved by `2.13–4.23 ms` (`138`/`528`/`2048`
    tokens) under `cache=false`, accepted against observed capture cost.
    [Stage 5a/5b acceptance](acceptance/02-snapshot-critical-path.md)
- **Step 5b — Move enabled prefix-cache `Snapshot` after first-token emission**:
  - *Changes*: When `capturePrefix` is true, run `sampleAndEmit(logits)` first in
    `PrefillOnce` so the first SSE token chunk flushes immediately, then capture
    `t.seq.Snapshot()` and `serializeLogits(logits)` *before* `PrefillOnce` returns (strictly
    before any subsequent `DecodeOnce` call mutates device `conv` / `recState`).
  - *Acceptance*: Measurable `~7–8 ms` reduction in user-observed first-token SSE latency
    with `--prefix-cache=true`; subsequent requests hitting the cached prefix produce
    bit-identical logits and greedy tokens against the pre-change order; passes `go test -race ./serving/...`.
  - *Measured (`c09cd06`)*: **Landed.** Saved `9.43 ms` (`138` tokens) and `6.47 ms` (`528`
    tokens) in SSE, `2.87 ms` (`TTFT - prefill` delta at `2050` tokens in same-process A/B),
    with identical cached/uncached tokens and zero total-latency regression.
    [Stage 5a/5b acceptance](acceptance/02-snapshot-critical-path.md)
- **Step 5c — Consolidate 48 per-layer GDN buffers into contiguous session storage**:
  - *Changes*: In [session.go](../../../engine/gpu/session.go), allocate one contiguous buffer
    laid out in per-layer interleaved `[conv_0, rec_0, conv_1, rec_1, ...]` order and bind
    per-layer sub-buffer/offset views. Preserving the interleaved byte order keeps
    `Snapshot()` and `Restore()` wire-compatible while collapsing `48` `hipMalloc`,
    `hipFree`, `Memset`, `Download`, and `Upload` calls into single contiguous operations.
  - *Acceptance*: `Snapshot()` time drops from `~7.95 ms` to `<= 1.5 ms`, `NewSession + Close`
    drops below `1.5 ms`, and snapshot/restore round-trip tests match byte-for-byte.

### Stage 6 (P6): Split into (6a) compact `M=1` tail `GEMV_Q8_1` and (6b) sub-layer norm/quant fusion

- **Step 6a — Compact `M=1` tail `GEMV_Q8_1` (pull-forward candidate)**:
  - *Changes*: In [plan.go](../../../engine/gpu/plan.go), route the five cropped final-layer
    `rows == 1` K-quant projections (`attn_q`, `attn_output`, `ffn_gate`, `ffn_up`,
    `ffn_down`) to `AddGemvQ8_1At` (`YUUKA_OP_GEMV_Q8_1`) over the same `Q8_1` activation
    instead of launching `64×128` Generic MMQ tiles.
  - *Acceptance*: Saves `~1.0–1.2 ms` per prefill; real-model logits at `128`, `512`, `1024`,
    and `2048` tokens stay within the `0.05` logit gate with identical greedy continuation.
- **Step 6b — Fused `Residual Add + RMSNorm (+ Q8_1)`**:
  - *Changes*: Add fused `ADD_RMSNORM_BATCH` and `ADD_RMSNORM_QUANT_Q8_1` ops across
    [norm.hip](../../../kernels/hip/ops/norm.hip), [yuuka_hip.h](../../../backend/hip/bridge/yuuka_hip.h),
    [bridge.cpp](../../../backend/hip/bridge/bridge.cpp), [hip.go](../../../backend/hip/hip.go),
    and [plan.go](../../../engine/gpu/plan.go). **Mixed-consumer constraint**: in layers where
    `prefillProjections` feeds both `PrefillMMQ` consumers (`xQ8`/`xDS`) and non-MMQ
    consumers requiring FP32 `p.norm` (e.g. `Q8_0` `ssm_alpha`/`beta` if on Candidate 2a, or
    small-row WMMA/streaming routes), emit both FP32 `p.norm` and `xQ8`/`xDS` outputs; omit
    the `p.norm` store only when every consumer of that norm group reads `xQ8`/`xDS`.
  - *Acceptance*: Full bounds/alignment/overlap coverage in `plan_validation_test.cpp`;
    bitwise-identical `Q8_1` digits and FP32 norms; repeatable `>= 2.0–3.0 ms` unprofiled
    improvement at `pp512`.

### Stage 7 (P7): Workgroup L2 cache swizzling for wide Direct MMQ

- **Concrete changes**:
  - After the preceding stages, collect `L2CacheHit` and `FETCH_SIZE` on the direct MMQ
    sweep for each mapping. Measure isolated and end-to-end latency in separate unprofiled
    runs. Follow the [Stage 7 counter procedure](stage7-direct-mmq-swizzle.md#collect-l2-counters-on-gfx1151).
  - [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip): Swizzle 2D workgroup indices into
    channel-tile groups (`8` or `16` channel blocks across token tiles) for wide matrices
    (`n_out >= 8192`) with multiple token tiles.
- **Acceptance criteria**:
  - **Primary gate**: Repeatable unprofiled wall-clock improvement on `pp512` / `pp2048` or
    under a documented concurrent CPU memory-stream benchmark on the shared UMA host, with
    zero regression (`<= 1%`) on any shape in `yuuka_prefill_sweep`.
  - **Diagnostic evidence**: `rocprofv3` `L2CacheHit` and `FETCH_SIZE` are recorded to
    explain the mechanism, not used as standalone acceptance criteria.

---

## 5. Final Phase 2 summary (commit `9aaf432`)

All 11 ordered execution slices (P1–P7) are complete on the reference `gfx1151`
container (`Qwen3.5-4B-Q4_K_M`, SHA-256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`, ROCm `10.1.0`,
HIP `7.16.26385`, Go `1.24.4`, `power=auto`, `concurrency=1`, Release CTest `10/10`
and full HIP Go test suite passing).

### 5.1 Final comparison against official `llama.cpp v1.0.0` (`b1b770d90`)

Interleaved matched-input comparison at `9aaf432` using F16 KV, identical integer
token IDs, and two reversed-start rounds (see
[Stage 7 acceptance](acceptance/11-direct-mmq-swizzle.md#final-comparison-with-official-llamacpp-v100)
for full sample ranges and protocol):

| Prompt | Yuuka / llama.cpp chunk | Round | `9aaf432` median (ms) | `llama.cpp v1.0.0` median (ms) | Throughput gain | Top-1 |
| ---: | :---: | ---: | ---: | ---: | ---: | :---: |
| 128 | 128 / 128 | 1 / 2 | 59.081 / 59.499 | 68.533 / 67.957 | **+16.00% / +14.21%** | Match |
| 512 | 512 / 512 | 1 / 2 | 195.141 / 196.271 | 228.578 / 230.809 | **+17.13% / +17.60%** | Match |
| 2048 | 1024 / 512 | 1 / 2 | 808.156 / 811.868 | 937.983 / 939.982 | **+16.06% / +15.78%** | Match |
| 8192 | 1024 / 256 | 1 / 2 | 3685.340 / 3703.992 | 4163.147 / 4163.955 | **+12.97% / +12.42%** | Match |

### 5.2 Key real-world outcomes and recorded negative trials

- **Boundary and short-chunk cliffs eliminated (Stage 1a, 1c, 1b-1, 1b-2, 1d)**:
  - Arbitrary `>= 128` rows route into masked Direct MMQ (`pp129` `113.11 -> 86.93 ms`,
    `pp511` `256.41 -> 217.65 ms`, `140`-token SSE `125.83 -> 97.75 ms`;
    [01](acceptance/01-direct-rows.md)).
  - Prepared GDN starts at `2` rows (`127`-row recurrence `12.60 -> 4.67 ms`, bitwise;
    [03](acceptance/03-sub128-gdn.md)).
  - `1..16` rows use FP32 streaming (`2` rows `84.81 -> 29.35 ms`, `513`
    `292.45 -> 238.69 ms`, `2050` `994.28 -> 936.83 ms`, cached `512+2` TTFT
    `91.7 -> 35.8 ms`, `2050`-token SSE `996.53 -> 940.28 ms`;
    [04a](acceptance/04a-streaming-short-tails.md)).
  - `17..127` rows use `gfx1151` Direct FP16 WMMA (`18/64/96/127` rows drop to
    `35.48 / 47.54 / 60.08 / 66.30 ms`, `530` `275.37 -> 253.48 ms`, `600`
    `321.79 -> 274.99 ms`, cached `512+18` TTFT `67.1 -> 43.2 ms`, `530`-token SSE
    `283.39 -> 258.91 ms`, bitwise to Step 1b-1 control;
    [04b](acceptance/04b-direct-fp16-wmma.md)).
  - CLI omits `--chunk-size` -> automatically selects `1024` for `1024..8192` tokens on
    the verified `gfx1151` F16 KV configuration (`3.18%–5.01%` faster across aligned and
    `+1`/`+18` boundaries; [05](acceptance/05-cli-chunk-policy.md)).
- **Core projection, attention, L2, and serving gains (Stages 2, 3, 4b–4d, 5a–5b, 7)**:
  - Paired narrow `Q8_0` WMMA (`ssm_beta`/`ssm_alpha`) matched logits bitwise and
    cleared the +5% upstream gate at all four benchmark lengths
    ([06](acceptance/06-narrow-q8-projections.md)).
  - Direct `Q6_K` MMQ landed for `ffn_down` and `attn_v` with bitwise parity and
    cumulative improvements across all four benchmark lengths
    ([07a](acceptance/07a-direct-q6k-ffn.md), [07b](acceptance/07b-direct-q6k-attention.md)).
  - Long-context attention extended F16 split-key windows, added packed `Q8_0` unsplit
    WMMA (`pp8192` `Q8_0` KV `1105 -> 2114–2122 tok/s`), and enabled typed `F32`/`Q8_0`
    split-key in verified ranges ([08b](acceptance/08b-f16-split-cells.md),
    [08c](acceptance/08c-q8-kv-wmma.md), [08d](acceptance/08d-cross-format-split.md)).
  - Server prefix capture skips dead `Snapshot` when caching is off and emits the first
    SSE token before capturing when caching is on
    ([02](acceptance/02-snapshot-critical-path.md)).
  - Wide Direct `Q4_K`/`Q5_K` 8-tile workgroup swizzle raised L2 hit rate from
    `~36%–43%` to `~85%–88%`, cut fetch volume ~6–7×, and saved `7.1–8.7 ms` at `pp512`
    and `24.6–33.7 ms` at `pp2048` bitwise ([11](acceptance/11-direct-mmq-swizzle.md)).
- **Measured trials kept out of production (by acceptance gate)**:
  - Original `<128` `Q8_1` MMQ trial rejected on numerical drift
    ([04](acceptance/04-sub128-projections.md)).
  - Step 4a (`F16` attention LDS padding), Step 5c (contiguous GDN state buffer),
    Step 6a (compact `M=1` tail `GEMV_Q8_1`), and Step 6b (fused `Residual Add + RMSNorm
    (+ Q8_1)`) all passed numerical checks but missed their repeatability or absolute
    latency gates, so existing production routes were retained
    ([08a](acceptance/08a-f16-lds-padding.md), [09](acceptance/09-contiguous-gdn-state.md),
    [10a](acceptance/10a-compact-tail-gemv.md), [10b](acceptance/10b-residual-norm.md)).
