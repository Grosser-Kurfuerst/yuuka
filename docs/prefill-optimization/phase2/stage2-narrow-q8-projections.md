# Stage 2 — Pair the narrow Q8_0 recurrent projections

Execution order 6. Start from the accepted Stage 1 routes and freeze chunk size in each A/B. Each of the 24 recurrent layers reads one normalized tile for QKV, gate, beta and alpha. Beta and alpha each have 32 outputs for 2560 inputs on the reference model. The original 48 launches consumed 6.17 ms at pp128 and 8.79 ms at pp512. The candidates below evaluate distinct arithmetic contracts. [Roadmap attribution](roadmap.md)

## 2a — One paired launch over FP32 input

1. At `addPrefillRecurrent`, keep QKV and gate in the existing shared-quantization group. Bind the narrow weights and outputs explicitly as `beta → p.beta` and `alpha → p.alpha`. Select the pair only for matching Q8_0 weights, measured row counts, width 2560 and rank 32 on gfx1151. Other configurations keep the existing projection group. The decode pair in `lm_head.hip` takes one activation row. Use its packed dequantization as a reference for a batched implementation. [Planner](../../../engine/gpu/plan.go) · [Streaming kernels](../../../kernels/hip/gemm/lm_head.hip)
2. Tile independent token rows and the 64 logical output channels. Assign each wave a row/channel dot, split K across lanes, and reuse the FP32 input stripe across the two matrices within the workgroup. Sweep small row tiles (e.g. 1, 2, 4) against the current two ops. Decode each 34-byte block as a half scale plus 32 signed bytes, apply the scale in FP32, accumulate/reduce in FP32 and store two row-major FP32 arrays. Mask the final row tile. This avoids an activation cast and a concatenated weight allocation.
3. Add a prefill pair plan descriptor (two weight handles, FP32 tile, two outputs, width, rank and rows), a Go append method and profile name. Reuse the existing plan validation and launch switches, preserve existing opcode values and the decode pair semantics. Validate dtype, packed row spans, dimensions, FP32 spans and distinct output ranges before appending. [ABI](../../../backend/hip/bridge/yuuka_hip.h) · [Bridge](../../../backend/hip/bridge/bridge.cpp) · [Go binding](../../../backend/hip/hip.go) · [Profile names](../../../backend/hip/opinfo.go)
4. Compare both full outputs with the CPU dequantized Q8_0 product. Use different beta/alpha weights, signed extremes, zero blocks, nonuniform FP32 activations, a partial row tile and guards after both destinations. Require max absolute error ≤ 1e-5 for each tensor. Compare same-chunk real-model logits to the pre-stage route (max absolute ≤ 0.05, same top-1 and eight greedy tokens). A trace must show 24 paired launches in place of 48, with the same QKV/gate preparation count. [Native product tests](../../../tests/hip/prefill_test.cpp)

## 2b — Conditional trial with Q8_1 activations

Open this trial only if 2a misses the product gate. Give it an independent forced A/B against the unmodified projection route and against 2a. Use it only where the group already produced valid digits and scale/sum pairs for this exact tile. Schedule the pair before their next overwrite. Keep 2a or the original path when the group has no prepared input.

For each 32-value block, pack the signed weight and activation digits in the same byte order. Compute eight four-byte dot products using the gfx1151 form `__builtin_amdgcn_sudot4(true, w4, true, x4, 0, false)` and sum them in int32. Both `true` arguments select signed bytes. Apply `weightScale * activationScale` once to the block dot and accumulate in FP32. The existing matvec shows the operand packing and builtin call. [Signed dot implementation](../../../kernels/hip/gemv/gemv_i8.hip)

Q8_0 has no minimum correction. Verify both outputs against a CPU product of the **same quantized bytes**. Even a correct integer product can move the exponential decay and sigmoid inputs. Save control/candidate snapshots after each chunk with a nonzero carried state and parse each layer's convolution history and recurrent matrix separately. Compare every value, report max absolute error and the responsible layer/chunk. Proposed additional state guard for this trial: finite values, ≤ 0.05 max absolute difference per state tensor against the original route, alongside the existing native recurrence gates. [Recurrence state checks](../phase1/acceptance/04b-gdn.md)

## 2c — Preserve the original WMMA arithmetic

The FP32 pair passed the CPU tensor and throughput gates but exceeded the
same-chunk model-logit bound. The approved equivalent candidate retains the
current FP16 operand conversions and ordered K16 WMMA operations with FP32
accumulation. Share the activation fragment between the two projections.
Decode their Q8_0 blocks in registers and write the outputs to independent
row-major FP32 spans. Preserve the GFX11 upper half-wave K order.

The CPU tensor oracle applies FP16 rounding to both operands before the
dot and gates at `1e-5` on a dyadic fixture whose partial sums fit in
FP32 in any order. Also run the broader random fixture and require both
native output arrays to match legacy WMMA bitwise on both fixtures.
Require the same bitwise agreement for full model logits through eight
decode steps and recurrent snapshots. Include both input prompts and the
two-call 512+512 test. The paired upstream 5% gate still applies at all four
lengths. Retain the verified original path for unmeasured or losing row
bands. [Trial record](acceptance/06-narrow-q8-projections.md)

## Promotion decision

- At pp128, pp512, pp2048 and 512+512 (two calls carrying the same state), use identical IDs, positions and chunking. For 2b require finite full-vocabulary logits within 0.05 at final prefill and each of eight decode steps, with identical greedy tokens throughout. Include a second prompt with different gate activations. A failing 2b cannot replace a passing 2a.
- Use the [matched upstream driver protocol](../phase1/acceptance/08-upstream-comparison.md) on gfx1151, same GGUF, F16 KV, one request. Record chunk per length and both engines' settings. Interleave yuuka/upstream/upstream/yuuka in one measurement session, 2 warmups and at least 5 samples per short arm (3 at 8K), reversing order to confirm. At **each** of 128/512/2048/8192 require `median_yuuka_ms <= median_llama_ms / 1.05` against upstream v1.0.0. Separately compare candidate/control on unchanged chunks to attribute the improvement. The ≤ 1.2 ms combined narrow-projection GPU time at pp128/512 is a diagnostic stretch target.
- Extend native plan/product tests and the [real-model suite](../../../engine/gpu/prefill_real_test.go), run Release CTest and the full HIP suite with the verified GGUF. Record which candidate, row bands and gates passed. If the product gate remains open, report the remaining gap before considering the roadmap's limited 6a pull-forward.

Estimate: 100–180 production lines for 2a across the kernel, plan ABI, bindings and planner. Allow roughly 50–100 more for a justified 2b trial. Tests and benchmarks are additional.
