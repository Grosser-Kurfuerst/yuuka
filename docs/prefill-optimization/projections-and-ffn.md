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
   [prefill sweep](../../bench/hip/prefill_sweep.cpp#L98) to run the existing and
   candidate MMQ on identical packed weights and Q8_1 input. Use
   `M=512, K=2560, N=9216` first. Report both MMQ-only time and the path
   including activation quantization. The 1.406 ms baseline is MMQ-only.
2. **Build one four-wave variant.** In
   [mmq_gemm.hip](../../kernels/hip/gemm/mmq_gemm.hip#L343), keep
   `launch_gemm_mmq_f32out` as the entry point. For the target cell, remap
   output fragments among four Wave32 waves, assign contiguous packed loads
   to lanes, and stage the two operands in the order consumed by gfx11 int8
   WMMA. Start with the current K-step and output coverage as the control.
   Inspect the generated operand loads, lane swizzles and register spills
   before changing the K-step. Keep int32 dot accumulation, FP32 scale/min
   correction and the existing Q8_1 digits and sums.
3. **Measure the inner loop.** At the target shape, yuuka/upstream issued
   22.19/17.50 VALU and 4.75/2.05 LDS instructions per 4096 useful MACs.
   Yuuka's measured LDS bank conflicts were already 0%. Test scale/min reuse
   across token fragments and activation-scale reuse across output channels,
   one change per sample. Inspect resource usage and normalize counters by
   the work of the actual tile. Update the geometry and dispatch pairing in
   [the counter reducer](../../scripts/prefill_counters.py) when extending it
   to a new tile or sweep order. The old reducer assumes eight waves and
   seven submissions per cell. Earlier double-buffer and register-prefetch
   experiments are recorded in [the previous roadmap](../roadmap3.md#10-rejected-routes-and-open-questions).
4. **Promote by shape.** The first target is at most **1.05 ms** and a
   reproducible improvement in unprofiled pp512. Select the winning native
   variant inside the MMQ launcher if the Go-level route is still MMQ.
   Update [Go dispatch](../../backend/hip/dispatch.go#L60) and its
   [tests](../../backend/hip/dispatch_test.go) only when changing route
   selection. Repeat for the high-cost Q4_K and Q5_K projection shapes in
   the real graph, testing M=128 separately and checking partial chunks.
   If the variant loses outside the measured cell, limit its dispatch to the
   winning shapes. Consider load-time reordering of packed weight blocks
   only if unpacking remains costly. Measure load time and additional
   resident bytes before adopting such a layout.

## Fuse where the intermediate has a single consumer

In [addPrefillFFN](../../engine/gpu/plan.go#L653), the MMQ branch quantizes the
normalized input once, submits two `AddGemmMmqAt` operations, applies
`AddSiluMul`, then calls `prefillMatmul` for down. The existing
`AddGemmSwigluAt` fusion serves the FP16 WMMA route. Two FP32 gate/up
outputs of shape [512, 9216] occupy 18 MiB each. Writing and rereading both
accounts for 72 MiB of logical traffic per layer. Cache residency determines
actual DRAM traffic.

1. **Fuse SwiGLU with the down-input quantizer.** Start where down selects
   MMQ. Read corresponding gate/up values, compute `SiLU(gate) * up`, and
   quantize a complete 32-channel block using the rounding in
   [quantize_q8_1_kernel](../../kernels/hip/gemm/mmq_gemm.hip#L299). Write the
   packed digits and `(scale, scale * sum(digits))` into the existing
   `xQ8` and `xDS` scratch. Feed these buffers directly to down through
   `AddGemmMmqAt`, bypassing the second input quantization. Both gate and
   up must finish consuming the shared input before reusing that scratch.
2. **Carry the fused operation through the plan.** Add its descriptor,
   buffer roles and width constraints to the [C ABI](../../backend/hip/bridge/yuuka_hip.h).
   Add native buffer-span validation and dispatch in the
   [bridge](../../backend/hip/bridge/bridge.cpp), the launcher declaration
   in [yuuka_kernels.h](../../kernels/hip/yuuka_kernels.h), and one
   checked plan builder in [hip.go](../../backend/hip/hip.go). Append the
   op code without renumbering existing codes. Update the
   [profile labels](../../engine/gpu/prefill_profile_test.go) so its time is
   included in the complete FFN comparison.
3. **Evaluate paired gate/up as a separate change.** Use the winning Q4_K
   layout to consume one activation fragment and two packed weight tiles.
   Keep both accumulators live until SwiGLU, then write the activation or,
   if a workgroup owns a complete quantization block, emit Q8_1 directly.
   Gate on supported dtypes, shapes and the actual down route. Time the
   entire segment from normalized input to down output. Retain the simpler
   fusion if doubling accumulators introduces spills or loses end-to-end.

## Acceptance

- Extend [native prefill tests](../../tests/hip/prefill_test.cpp) with identical
  input blocks for old/new MMQ and fused/unfused FFN. Check the existing
  per-GEMM tolerance, zero and signed activation blocks, Q8_1 rounding,
  edge tiles, final down output and repeatability. An isolated fused
  quantizer fed the same gate/up values should reproduce the reference
  digits. Use model-level gates after any change in projection reduction order.
- Cover buffer offsets, short buffers and invalid widths for added plan
  operations in [plan validation](../../tests/hip/plan_validation_test.cpp).
  Run the native sweep, real-model pp128/512/2048/8192, and greedy
  generation. Record MMQ time, complete FFN time, scratch, compiler spills
  and unprofiled throughput for each accepted change.
