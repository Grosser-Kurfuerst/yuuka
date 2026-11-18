# Gated delta-net recurrence

The 24 recurrent layers spend 53.9 ms at pp512 versus 35.1 ms upstream.
The split-register core accounts for 45.7 vs 13.2 ms, and the surrounding
convolution, preparation and gating for 8.2 vs 21.9 ms. Yuuka has room to
win on the complete operation if it preserves its inexpensive preparation.
The proposed budget is **30 ms** for all 24 layers. At the baseline support
cost this leaves about 22 ms for the core including any new preparation.

## Implement in two measured changes

1. **Preserve state ownership and the reference.** Start in
   [launch_recurrent_chunk_f32](../../kernels/hip/prefill/prefill.hip#L825).
   The selected state-size-128 path runs convolution, advances convolution
   history, runs the split-register recurrence, then applies gated RMSNorm.
   Keep this sequence as the A/B oracle. The current non-null triangular
   scratch already selects a separate chunkwise algorithm that measured
   74.5 ms. Give new temporary storage an explicit role if required by the
   plan. Keep persistent matrix state and convolution history on the
   session. [Current op binding](../../engine/gpu/plan.go#L616) · [Previous chunkwise results](../roadmap3.md#51-implementation-2026-11-18)
2. **Prepare read-only recurrence inputs over the whole chunk.** After
   convolution, compute normalized Q/K per (token, key group) and compute
   decay and beta per (token, value head). Match the split-register formula
   for epsilon, query scaling, softplus and sigmoid. The current kernel
   recomputes these values in each workgroup and uses two workgroup barriers
   per token. Let a first prototype write prepared inputs, then have the
   existing four-lane-per-column mapping read them directly. Preserve V
   and the convolution history needed by the next chunk. Measure preparation
   plus recurrence before considering fusion with the convolution output.
3. **Test eight lanes per state column.** For state size 128 this leaves
   16 FP32 state elements per lane, compared with 32 in the current
   mapping. Combine lane-local dot products using wave shuffles within
   each eight-lane segment. For every token in order compute the decayed
   state/key dot, the beta-scaled correction, the updated state, and the
   state/query dot. Load a state shard once on entry and store the final
   shard once on exit. Preserve the physical state layout consumed by
   decode. Vary only lane mapping in this experiment. Check actual global
   load coalescing, VGPR count, spills and duration on gfx1151.
4. **Integrate the winning combination.** Route by supported geometry in
   the prefill launcher. The model planner already emits one recurrent op
   per recurrent layer. Reuse transient space with a proven end of lifetime
   or size it once in [Model.configurePrefill](../../engine/gpu/model.go#L349)
   and bind it in [addPrefillRecurrent](../../engine/gpu/plan.go#L558). Validate
   prepared buffer lengths for rows, groups, heads and state width in the
   [bridge](../../backend/hip/bridge/bridge.cpp). Check HIP launch errors
   for each new kernel. Preserve explicit-stream ordering and the existing
   completion event for scratch reuse.

The upstream trace on this device executed a per-token kernel with eight
lanes per column. It is a useful geometry to test. At baseline the two
cores issued about 9.50M vs 3.15M LDS instructions per layer. Moving input
preparation changes the core boundary, so count the new preparation
dispatches in the comparison. [Upstream implementation at v1.0.0](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/ggml/src/ggml-cuda/gated_delta_net.cu#L39-L119)

## Verify and decide

- Extend [prefill_test.cpp](../../tests/hip/prefill_test.cpp) to compare
  convolution history, raw recurrence outputs, final matrix state and gated
  outputs of both variants with the CPU reference. Include a nonzero
  initial state, heads sharing a key group and a sequence split into two
  chunks. Check repeatability and the current operator tolerances.
- Run real-model chunked prefill, greedy continuation and the separate
  decode-path check. Inspect layer state at the chunk boundary if the first
  token agrees and subsequent tokens diverge. [CPU recurrence](../../model/qwen35/recurrent.go)
  · [CPU/GPU comparison tests](../../engine/gpu/runtime_test.go)
- Time four-lane baseline, four-lane with prepared inputs, and eight-lane
  with the same inputs. Include convolution, preparation, core, output gate
  and total. An intermediate milestone is **42 ms** at pp512. Revisit
  the 30 ms budget against the end-to-end result. Retain the existing path
  for a geometry where the experiment loses.
