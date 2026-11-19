# Quantized projections and FFN

## Start with the repeated Q4_K shape

At pp512 the gate and up projections multiply 512 rows of width 2560 by two
separate Q4_K matrices, each with 9216 output channels. A single full-row
projection took 1.406 ms in yuuka and 0.942 ms upstream. The full-row totals
were 89.5 and 58.6 ms, respectively. Yuuka ran 64 full-row projections.
Upstream ran 62 and computed the last FFN on the requested output row.
Use equal shapes when tuning the kernel and equal output requirements when
timing the whole FFN.

The other Q4_K projections account for a further 16.1 ms of the pp512 gap
and Q5_K for 10.9 ms. The Q6_K paths, including upstream's conversions,
differ by 0.1 ms at this length. Prioritize Q4_K, then Q5_K.

1. **Establish the native A/B.** Extend the existing
   [prefill sweep](../../../bench/hip/prefill_sweep.cpp#L98) to run the existing and
   candidate MMQ on identical packed weights and Q8_1 input. Use
   `M=512, K=2560, N=9216` first. Report both MMQ-only time and the path
   including activation quantization. The 1.406 ms baseline is MMQ-only.
2. **Build one four-wave variant.** In
   [mmq_gemm.hip](../../../kernels/hip/gemm/mmq_gemm.hip#L343), keep
   `launch_gemm_mmq_f32out` as the entry point. Specialize
   `launch_mmq_tile` and `gemm_mmq_kernel` on the wave count. Use the
   following mapping for the first Q4_K experiment. Both launches produce
   the same 64 output channels by 128 tokens per workgroup, advance K by
   128 and use the existing 30 KiB LDS layout.

   | Part of the kernel | Current eight-wave launch | Proposed four-wave launch |
   | --- | --- | --- |
   | Threads, channel base, token base within tile | 256, `16*(wave%4)`, `64*(wave/4)` using integer division | 128, `16*wave`, 0 |
   | Fragments per wave | Four 16×16 token fragments | Eight, with token offsets 0, 16, …, 112 |
   | Staging per wave per K step | Two weight and four activation regions | Four weight and eight activation regions |
   | Live FP32 dot totals per lane | `sum[4][8]` | `sum[8][8]` |

   Pass the GEMM wave count into `stage_weights` and `stage_activations`.
   Each region is still 512 bytes. Size `b_base`, `bv`, `c`, `sum` and
   every matching loop, including the final stores, by the fragment count.
   For fragment `t` and lane element `e`, store token
   `m0 + token_base + 16*t + (lane%16)`, channel
   `n0 + channel_base + 2*e + lane/16`. Use integer division and retain
   both edge masks. Instantiate the HIP function attribute and launch size
   for each specialization. Keep the standalone Q8_1 quantizer at its own
   eight waves per block. Its warp index also uses the current `kWarps`.
   [Staging](../../../kernels/hip/gemm/mmq_gemm.hip#L229) · [Quantizer](../../../kernels/hip/gemm/mmq_gemm.hip#L299) · [Launcher](../../../kernels/hip/gemm/mmq_gemm.hip#L486)
3. **Measure the inner loop.** At the target shape, yuuka/upstream issued
   22.19/17.50 VALU and 4.75/2.05 LDS instructions per 4096 useful MACs.
   Yuuka's measured LDS bank conflicts were already 0%. The kernel already
   hoists `wd[e]` and `wm[e]` outside the token loop. The prototype reuses
   that weight fragment and its corrections across eight token fragments.
   Its risk is increased register pressure. Compare VGPRs, spills, LDS
   instructions and total time with the eight-wave control. For a second
   independent test, pack each Q4_K weight scale and signed minimum as
   one aligned `float2` at `[pair][channel]`, written by the existing
   `within == 0` producer. Read that pair once per lane element and inspect
   the generated LDS load. Keep Q6_K's scale-per-16 layout on its own
   path. Test the scale layout on the faster geometry from the first A/B.
   Normalize counters by
   the work of the actual tile. Update the geometry and dispatch pairing in
   [the counter reducer](../../../scripts/prefill_counters.py) when extending it
   to a new tile or sweep order. The old reducer assumes eight waves and
   seven submissions per cell. Earlier double-buffer and register-prefetch
   experiments are recorded in [decisions and rejected routes](../../benchmarks.md#decisions-and-rejected-routes).
4. **Promote by shape.** The first target is at most **1.05 ms** and a
   reproducible improvement in unprofiled pp512. Select the winning native
   variant inside the MMQ launcher if the Go-level route is still MMQ.
   Update [Go dispatch](../../../backend/hip/dispatch.go#L60) and its
   [tests](../../../backend/hip/dispatch_test.go) only when changing route
   selection. At the 1.05 ms target, 64 equal-shaped calls would save about
   23 ms of traced projection time (`64*(1.406-1.05)`). Recount calls after
   final-layer cropping. The original pp512 projection gap alone was 62.7 ms.
5. **Extend packed staging to the next Q4_K and Q5_K shapes.** Inspect
   `stage_weight_span`. Adjacent 32-value subblocks of a weight row
   read the same `ql` address at a given within-block offset. They use
   its low and high nibbles respectively. In `stage_weights`, for each
   pair of regions in the same 16-channel group, load the packed bytes
   once per lane and emit both fragment slots. For Q5_K, the extra `qh`
   bytes at that within-block offset are also shared, with a distinct bit
   for each subblock. Reuse the loaded word across the owned pair and
   apply the corresponding bit masks. Keep separate scale/min corrections
   per 32-value subblock. A/B this change alone at M=128 and 512 for the
   wide and tall shapes in the sweep, including K=9216 → N=2560. Check
   the high bit and scale/min at subblocks 3/4 and the next 256-value
   block in the product test. Compare loads, instructions and time on
   model shapes. Enable only cells with a measured benefit.
   [Packed span decode](../../../kernels/hip/gemm/mmq_gemm.hip#L158)

## Fuse where the intermediate has a single consumer

In [addPrefillFFN](../../../engine/gpu/plan.go#L653), the MMQ branch quantizes the
normalized input once, submits two `AddGemmMmqAt` operations, applies
`AddSiluMul`, then calls `prefillMatmul` for down. The existing
`AddGemmSwigluAt` fusion serves the FP16 WMMA route. Two FP32 gate/up
outputs of shape [512, 9216] occupy 18 MiB each. Writing and rereading both
accounts for 72 MiB of logical traffic per layer. Cache residency determines
actual DRAM traffic.

1. **Fuse SwiGLU with the down-input quantizer.** Start where down selects
   MMQ. Read corresponding gate/up values, compute `SiLU(gate) * up`, and
   quantize a complete 32-channel block using the rounding in
   [quantize_q8_1_kernel](../../../kernels/hip/gemm/mmq_gemm.hip#L299). Write the
   packed digits and `(scale, scale * sum(digits))` into the existing
   `xQ8` and `xDS` scratch. Feed these buffers directly to down through
   `AddGemmMmqAt`, bypassing the second input quantization. Both gate and
   up must finish consuming the shared input before reusing that scratch.
   For the first implementation require all three projection routes to
   select MMQ. After the shared input quantization and the two gate/up
   GEMMs, replace the following plan segment.

   | Current operations | Proposed operations |
   | --- | --- |
   | `AddSiluMul(ffnGate, ffnUp → ffnGate)` then `AddQuantizeQ8_1At(ffnGate → xQ8, xDS)` inside `prefillMatmul(down)` | One proposed `AddSiluMulQuantQ8_1At(ffnGate, ffnUp → xQ8, xDS)` |
   | `AddGemmMmqAt(down, xQ8, xDS → tmp)` then `Add(x, tmp)` | Emit those same two operations directly |

   Read four FP32 gate/up pairs per lane. For each pair compute
   `(g / (1 + expf(-g))) * up` with the existing FP32 evaluation order.
   Use the current eight-lane maximum and sum reductions, `rintf` with
   the original clamp and the zero-block rule. The fused pass removes one
   write and one read of the 18 MiB activated tensor at 512 rows, or
   **36 MiB of logical traffic per eligible layer**. The two GEMMs still
   materialize gate and up. Count the launches and compare full FFN time.
   [Current SwiGLU arithmetic](../../../kernels/hip/ops/norm.hip#L101) · [Down projection](../../../engine/gpu/plan.go#L705)

2. **Carry the fused operation through the plan.** Append a new opcode to
   the [C ABI](../../../backend/hip/bridge/yuuka_hip.h). Give it
   `buf[0..3] = {gate, up, digits, scale_sum}` and
   `n[0..1] = {ffn_width, rows}`. Require positive rows, width divisible
   by 128, and checked spans at the supplied byte offsets of respectively
   `4*rows*width`, `4*rows*width`, `rows*width` and
   `8*rows*(width/32)` bytes. Require alignment for the vector loads and
   stores used by the kernel.
   Add native buffer-span validation and dispatch in the
   [bridge](../../../backend/hip/bridge/bridge.cpp), the launcher declaration
   in [yuuka_kernels.h](../../../kernels/hip/yuuka_kernels.h), and one
   checked plan builder in [hip.go](../../../backend/hip/hip.go). Update the
   [profile labels](../../../engine/gpu/prefill_profile_test.go) so its time is
   included in the complete FFN comparison.
3. **Evaluate paired gate/up as a separate change.** For two Q4_K weights
   of identical shape, have each workgroup own the same output tile in
   both matrices. Per K step stage the shared activation B tile once.
   Stage gate weights into A, accumulate the gate product, synchronize the
   block, then reuse the A and weight-scale storage for up. Keep separate
   FP32 totals until the final K step. Apply SwiGLU elementwise in the
   epilogue and write one FP32 `p.ffnGate` tile. Follow it with the
   existing down quantizer and MMQ. This first paired prototype can reuse
   `xQ8/xDS` for down after the paired kernel returns. Writing down's
   quantized output into those buffers inside the paired kernel would race
   other workgroups still reading the input activation. Measure register
   spills, added barriers and complete FFN time against the simpler fusion.

## Acceptance

- Extend [test_gemm_mmq](../../../tests/hip/prefill_test.cpp#L534) for the
  geometry A/B. Add a case beside [test_gemm_swiglu](../../../tests/hip/prefill_test.cpp#L586)
  comparing `SILU_MUL → QUANT_Q8_1 → MMQ(down)` with the fused route
  from identical gate/up buffers. Check the existing
  per-GEMM tolerance, zero and signed activation blocks, Q8_1 rounding,
  edge tiles, final down output and repeatability. An isolated fused
  quantizer fed the same gate/up values should reproduce the reference
  digits. Use model-level gates after any change in projection reduction order.
- Cover buffer offsets, short buffers and invalid widths for added plan
  operations in [plan validation](../../../tests/hip/plan_validation_test.cpp).
  Run the native sweep, real-model pp128/512/2048/8192, and greedy
  generation. Record MMQ time, complete FFN time, scratch, compiler spills
  and unprofiled throughput for each accepted change.
