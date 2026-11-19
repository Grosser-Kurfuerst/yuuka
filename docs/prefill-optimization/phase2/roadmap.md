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
| **3** | **Step 1c** (`rows < 128` GDN crossover) | [stage1bc-sub128-rows.md](stage1bc-sub128-rows.md#31-step-1c-prepared-gdn-below-128-rows) | **Landed** (`b833b4e`, `260094f`, [03](acceptance/03-sub128-gdn.md)) | ~5-line threshold change in `plan.go` (`gdnPreparedMinRows = 2`), byte-identical to control (per Phase 1 `04b-gdn.md`); landing it before `1b` fixes GDN while isolating `1b`'s projection arithmetic changes. |
| **4** | **Step 1b** (short projection routing) | [Sub-128 projections](stage1bc-sub128-rows.md) | **1b-1 landed**, 1b-2 pending | FP32 streaming accelerates eligible K-quants at 1–16 rows. FP16 WMMA handles the medium short chunks. [Acceptance](acceptance/04a-streaming-short-tails.md) |
| **5** | **Step 1d** (CLI default chunk re-sweep) | [Stage 1d — CLI chunk policy](stage1d-cli-chunk-policy.md) | Pending (`1b`) | Must run *after* `1a–1c` so the `chunk = 512 / 1024 / 2048` crossover is tuned on the updated tail and sub-128 kernels rather than stale penalties. |
| **6** | **Stage 2** (`ssm_alpha`/`ssm_beta` `Q8_0`) | [Stage 2 — Narrow Q8_0 projections](stage2-narrow-q8-projections.md) | Pending | Highest remaining per-chunk tax (`6.2–9.3 ms`) once boundary cliffs are gone; Candidate 2a precedes Candidate 2b to keep FP32 gate precision unless 2b is needed. |
| **7** | **Stage 3** (`Q6_K` Direct WMMA) | [Stage 3 — Direct Q6_K MMQ](stage3-q6k-direct-mmq.md) | Pending | Benefits all arbitrary `>= 128` lengths automatically after Step 1a routes them to Direct MMQ (`9216->2560` first, then `2560->1024`). |
| **8** | **Stage 4** (Long-context Attention) | [Stage 4 — Long-context attention](stage4-long-context-attention.md) | Pending | Sub-steps `4a -> 4b -> 4c -> 4d`: F16 LDS padding -> `>8K` Split-Key -> `Q8_0` WMMA -> cross-format Split-Key plumbing. |
| **9** | **Step 5c** (Contiguous GDN state) | [Stage 5c — Contiguous GDN state](stage5c-contiguous-gdn-state.md) | Pending | Separated from `5a/5b` because merging 48 buffers changes `session.go` allocation layout and per-layer offsets; shrinks the off-critical-path copy itself. |
| **10** | **Steps 6a & 6b** (`M=1` tail GEMV & norm fusion) | [Stage 6 — Compact tail and norm fusion](stage6-compact-tail-and-norm.md) | Pending | `6a` (pull-forward fallback if `pp128` misses +5% by `~1 ms` after Stage 2), then `6b` fused `Add + RMSNorm (+ Q8_1)`. |
| **11** | **Stage 7** (2D L2 cache swizzling) | [Stage 7 — Direct MMQ workgroup swizzle](stage7-direct-mmq-swizzle.md) | Pending | Final pass after Step 1a's new grid mapping and Stages 2–6 stabilize memory traffic, judged strictly by unprofiled wall time. |

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
- **Step 1b — Sub-128 projection crossover (`rows < 128`, GDN held fixed)**
  ([implementation plan](stage1bc-sub128-rows.md), lands after Step 1c):
  - *Changes*: Benchmark masked `64`-row Direct MMQ, `wmma_k32`, `gemv_batch`, and the
    multi-row streaming kernel ([lm_head.hip](../../../kernels/hip/gemm/lm_head.hip), `1..16` rows)
    across `rows = 1, 2, 4, 8, 16, 18, 32, 48, 64, 88, 96, 127` with GDN held constant.
    Update [dispatch.go](../../../backend/hip/dispatch.go) and [plan.go](../../../engine/gpu/plan.go)
    with the measured crossover rules.
  - *Acceptance*: Numerical checks separate **same-chunking route comparisons** (streaming
    FP32 matches FP32 reference within `2e-3`; `Q8_1` MMQ matches CPU `Q8_1` product within
    existing MMQ bounds) from **cross-chunking comparisons** (`512+1` vs single-chunk `513`,
    `512+18` vs `530`: verify identical greedy top-1 and 3-step continuation, and record
    logit max-abs delta). Projection time on `127`, `512+1`, `512+18`, `600` (`512+88`),
    and `2048+2` drops monotonically with row count.
  - *Status*: FP32 streaming through 16 rows passed acceptance in Step 1b-1
    ([record](acceptance/04a-streaming-short-tails.md)). The earlier short Q8_1
    MMQ candidate remains rejected ([trial](acceptance/04-sub128-projections.md)).
    Step 1b-2 covers FP16 WMMA for medium short chunks.
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

## 5. Current progress snapshot (commit `260094f`)

Cumulative interleaved verification on the reference `gfx1151` host (`2026-11-19`,
`Qwen3.5-4B-Q4_K_M`, SHA-256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
`F16 KV`, `chunk=512`, `concurrency=1`, greedy, `power=auto`, temp `38 -> 63 °C`, CTest
`10/10` pass):

### 5.1 Upstream `llama.cpp v1.0.0` (`b1b770d90`) vs `HEAD` (`260094f`)

Using `bench/prefill-compare` in `yuuka-1, llama-1, llama-2, yuuka-2` order (`2` warmups,
`5` reps per process, pooled `10` samples):

| Prompt | `260094f` (`chunk=512`) | `llama.cpp v1.0.0` | Speedup | Top-1 | `260094f` (`chunk=1024`) |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 127 | 116.10 ms (`1093.9 tok/s`) | 67.19 ms (`1890.1 tok/s`) | `-42.12%` | Match | — |
| 128 | 67.70 ms (`1890.8 tok/s`) | 67.38 ms (`1899.7 tok/s`) | `-0.47%` | Match | — |
| 129 | 86.82 ms (`1485.8 tok/s`) | 73.26 ms (`1760.9 tok/s`) | `-15.62%` | Match | — |
| 140 | 88.02 ms (`1590.5 tok/s`) | 74.26 ms (`1885.3 tok/s`) | `-15.64%` | Match | — |
| **511** | **219.43 ms** (`2328.7 tok/s`) | 234.52 ms (`2179.0 tok/s`) | **`+6.87%`** | Match | — |
| **512** | **215.58 ms** (`2375.0 tok/s`) | 231.67 ms (`2210.0 tok/s`) | **`+7.46%`** | Match | — |
| 513 (`512+1`) | 292.09 ms (`1756.3 tok/s`) | 245.96 ms (`2085.7 tok/s`) | `-15.79%` | Match | — |
| 530 (`512+18`) | 273.36 ms (`1938.9 tok/s`) | 259.17 ms (`2045.0 tok/s`) | `-5.19%` | Match | — |
| 600 (`512+88`) | 320.45 ms (`1872.4 tok/s`) | 286.40 ms (`2094.9 tok/s`) | `-10.62%` | Match | — |
| 1024 | — | — | — | — | **425.71 ms (`2405.4 tok/s`)** |
| **2048** | **899.37 ms** (`2277.2 tok/s`) | 948.31 ms (`2159.6 tok/s`) | **`+5.44%`** | Match | **862.68 ms (`2374.0 tok/s`)** |
| 2050 (`2048+2`) | 987.76 ms (`2075.4 tok/s`) | 960.09 ms (`2135.2 tok/s`) | `-2.80%` | Match | — |
| **8192** | **4122.63 ms** (`1987.1 tok/s`) | 4215.05 ms (`1943.5 tok/s`) | **`+2.24%`** | Match | **3911.58 ms (`2094.3 tok/s`)** |

### 5.2 Cumulative pre-Phase-2 control (`846fc4f`) vs `260094f` (`1a + 5ab + 1c`)

All 18 lengths in `bench/prefill-compare` (`ctrl-1, cand-1, cand-2, ctrl-2`, `chunk=512`)
remained **bitwise identical (`maxdiff = 0`)**:

- **Non-power-of-two `>= 128` (Step 1a)**: `129` `114.94 -> 88.65 ms` (`-22.87%`), `140`
  `116.57 -> 89.22 ms` (`-23.47%`), `256` `138.36 -> 117.78 ms` (`-14.88%`), `511`
  `259.78 -> 223.59 ms` (`-13.93%`). Aligned `128`, `512`, `1024` (`442.79 ms` at `c=512`),
  `2048`, and `8192` stayed within `-0.80%` to `+0.12%`.
- **Sub-128 prepared GDN (Step 1c)**: `18` `60.24 -> 58.25 ms` (`-3.31%`), `64`
  `84.62 -> 80.67 ms` (`-4.66%`), `96` `115.67 -> 108.83 ms` (`-5.91%`), `127`
  `126.55 -> 119.04 ms` (`-5.93%`).
- **`yuuka-server` SSE first-content TTFT (`ctrl-1, cand-1, cand-2, ctrl-2`, `14` samples)**:
  - `--prefix-cache=false`: `~140` `127.63 -> 93.44 ms` (`-34.19 ms`), `~530`
    `284.46 -> 278.86 ms` (`-5.60 ms`), `~2050` `916.84 -> 912.39 ms` (`-4.45 ms`).
  - `--prefix-cache=true` (misses): `~140` `129.12 -> 93.45 ms` (`-35.67 ms`), `~530`
    `287.39 -> 278.94 ms` (`-8.45 ms`), `~2050` `917.92 -> 915.20 ms` (`-2.72 ms`).
- **Prefix-cache increment (`bench/serving` on `260094f`, 512-token prefix, `hits=11 misses=1`)**:
  `append=2` TTFT `91.881 ms` (prefill `87.524 ms`); `append=18` TTFT `66.922 ms` (prefill
  `62.688 ms`).
- **Remaining gap**: Because Step 1b is blocked on numerical acceptance, `2`-row prefill
  (`85.34 ms`), `127`-row prefill (`116.10–119.04 ms`), and `1–2` row tails (`513` at
  `292.09 ms`, `+76.5 ms` over `512`) still use the pre-Phase-2 sub-128 projection paths.
