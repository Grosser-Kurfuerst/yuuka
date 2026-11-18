# Long-context attention and chunk size

At chunk 512 the eight attention layers together took 5.4 ms at pp512
versus 6.0 ms upstream, and 959.6 ms at pp8192 versus 662.7 ms upstream.
The last 512 rows alone cost 127.3 vs 86.1 ms. The latter figures include
the upstream partial-result merge. Re-measure after final-layer cropping
because it changes the number of query rows.

## Split the key range for long prefixes

1. **Keep the short-prefix kernel as the control.** Start from
   [attention_prefill_paged_wmma_kernel](../../kernels/hip/prefill/attention_wmma.hip#L107).
   The target is F16 paged KV, head width 256, with four query heads sharing
   one KV head. One workgroup handles 16 query rows and stages 16 keys per
   loop. At this width its dynamic LDS footprint is 50 KiB. Increasing the
   key tile to 32 in the same layout would require 68 KiB, above the dispatch
   limit. Keep the tile size for the first experiment.
2. **Split the existing WMMA key loop.** Extend the launch grid from
   `(kv_heads, ceil(rows/16))` to `(kv_heads, ceil(rows/16), S)`, first
   testing `S=2`, then 4. Within each query tile compute
   `T = ceil((position + min(16*(blockIdx.y+1), rows))/16)` visible
   key tiles. Split `s = blockIdx.z` visits tile indices in
   `[floor(s*T/S), floor((s+1)*T/S))`. For tile index `i`, stage paged
   K/V starting at `16*i` with the original tail bound. Score each key
   only up to that query row's absolute position. All threads participate
   in block barriers even for a row whose keys are entirely masked.
   [Key loop and online softmax](../../kernels/hip/prefill/attention_wmma.hip#L164) · [Current grid](../../kernels/hip/prefill/attention_wmma.hip#L325)

   Use the existing scaled base-2 scores, FP16 probability tile and FP32
   accumulators. In a tile with no valid key for a row, write zero
   probabilities, rescale the previous accumulator by one and preserve
   its maximum and sum. This avoids `exp2(-inf - -inf)`. At the end,
   write the partial record indexed by `((s*rows+r)*heads+h)*(head_dim+2)`
   in floats. Its fields at offsets 0, 1 and 2 onward are maximum, sum,
   and the **unnormalized** value vector. One lane per real row writes the
   scalar pair using the score-lane row mapping. Store vector components
   using the existing accumulator-to-output mapping. An empty split
   writes `(-inf, 0, zeros)`. Skip padded query rows. A split grid of one
   can be forced in the test to check record layout against the unsplit output.
3. **Merge on the same stream.** Launch one workgroup per `(r,h)`.
   Have one lane compute `M = max(m_i)` over records with `l_i > 0`,
   the participating `a_i = exp2(m_i - M)` and denominator
   `L = sum(a_i*l_i)`. Broadcast these through shared memory with one
   barrier. Threads cover dimensions in strides of the block size and
   write `sum(a_i*v_i[d])/L` over the same participating splits, in
   ascending split order. Every real query can attend to its own token.
   Apply the Qwen sigmoid gate once from the original Q/gate row after
   normalization. Time both kernels including their intermediate traffic.
4. **Bind scratch and select at runtime.** The logical scratch requirement
   is `splits * rows * query_heads * (head_dim + 2) * sizeof(float)`,
   before alignment. At 512 rows, 16 query heads and four splits this is
   32.25 MiB. Size it for the actual configured chunk and supported split
   count. Add a `prefillActivations` buffer to the existing allocation
   list for the experimental F16, four-heads-per-KV, head-width-256
   route. Bind the optional buffer at `ATTENTION_PREFILL_PAGED.buf[6]`
   in the Go plan builder. Slot 5 holds BLAS scratch, `u[3]` selects
   the dense/gated layout and `n[5]` is reserved by the proposed cropped
   query for its position displacement. With slot 6 present, validate
   its aligned byte range for four splits using checked products at plan
   creation. Reuse it across layers on the model stream and release it
   with the other prefill allocations. Cover an exact-fit span and a span
   one byte short. [Allocation](../../engine/gpu/model.go#L349) · [Plan binding](../../engine/gpu/plan.go#L546) · [Go paged descriptor](../../backend/hip/hip.go#L1458) · [Native validation](../../backend/hip/bridge/bridge.cpp#L956)

   Inside `launch_attention_prefill_paged_f32`, after the WMMA geometry
   check, choose the split count using the **execution-time** query position
   and row count. Sweep forced counts 1/2/4 at the positions below before
   setting any crossover. For each measured row/prefix class, select the
   lowest median total time (direct kernel or partial plus merge). Use the
   direct kernel on ties within observed run-to-run spread and for unmeasured
   shapes, including the compact one-query tail. Feed the winning table
   back into this launcher. The same cached 512-row plan executes at
   positions 0 and 7680. [Native dispatch](../../kernels/hip/prefill/prefill.hip#L791)

## Verify the attention change

Compare one-, two- and four-split results with the unsplit reference on
identical paged KV. Cover a partial last page, noncontiguous physical pages,
query rows on both sides of a split boundary, and an entirely masked split
for an early query. Check the existing native attention tolerances, gated
output, deterministic reruns, real-model logits and greedy continuation.
Extend [test_attention_prefill_mma](../../tests/hip/prefill_test.cpp#L1299)
with an F16 pool. Round K/V once before feeding the same values to both
the reference and the candidate. With 17 new queries at position zero,
run four forced splits to exercise empty intervals and the one-row tail.
At a nonzero start, test a row whose final tile is partially masked.
Also run the [paged tests](../../tests/hip/paged_test.cpp) and the
[model attention comparison](../../engine/gpu/attention_compare_test.go).

Measure at positions 0, 2048 and 7680 with 512 new tokens. Use the full
prompt trace and a fresh unprofiled pp8192 comparison to confirm the win.
The first target is **at most 100 ms** for the last 512 rows under baseline
geometry. Record merge time and peak scratch. Include the new kernel in
the resource report when measuring spills and dynamic LDS. The current
[resource script](../../scripts/kernel_resources.sh) needs the attention
WMMA source added to its compilation list for this measurement.

## Select a chunk size for single-request generation

1. Use the [observed sweep](measurement.md#baseline-used-for-this-plan)
   as a starting point. Chunk 2048 is available today through
   `yuuka bench --chunk-size 2048` and improved the 8K result by 5.9%
   over 512 in the same session. It is a configuration candidate for a
   long request. Short prompts still need their own measurement.
2. After the projection, recurrence and attention changes, run
   512/1024/2048/4096 chunks again at 8K and include lengths around a
   boundary (for example 2048 and 2049). Check pp128/512, TTFT, peak
   transient bytes and the cost of a short final chunk. The model
   currently clamps to capacity and a maximum of 4096. Log the effective
   size in the result. [Runtime configuration](../../engine/gpu/model.go)
3. If an automatic choice is justified, apply a measured length-based rule
   once before the single-request prefill plan and scratch are built. Keep
   an explicit `--chunk-size` override authoritative. The existing CLI
   defaults to 512, so distinguish an omitted option from an explicit 512
   when wiring the rule into [run and bench](../../cmd/yuuka/main.go).
   Reallocating shared scratch or invalidating session plans per chunk
   would defeat the policy. Evaluate scheduler latency separately before
   applying a single-request result to serving.
