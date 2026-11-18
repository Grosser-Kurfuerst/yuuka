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
2. **Implement partial attention.** Add a split coordinate to the grid.
   Try two disjoint key ranges, then four. Align interior boundaries to
   16 keys. Fetch each tile through the existing page table and apply the
   causal mask using each query's absolute position. For each split,
   query and head, write a local maximum `m`, exponential sum `l` and
   unnormalized value vector `v` in FP32. Keep the current base-2 score
   convention. A wholly masked range contributes zero mass and a zero
   vector. Handle it before evaluating any difference of infinities.
3. **Merge on the same stream.** For each query/head with participating
   splits, compute `M = max(m_i)` over nonempty splits and
   `a_i = exp2(m_i - M)`. The output is
   `sum(a_i * v_i) / sum(a_i * l_i)`. Apply the Qwen sigmoid gate once
   after normalization. Preserve head sharing, row strides and page order.
   Time the partial and merge kernels together.
4. **Bind scratch and select at runtime.** The logical scratch requirement
   is `splits * rows * query_heads * (head_dim + 2) * sizeof(float)`,
   before alignment. At 512 rows, 16 query heads and four splits this is
   32.25 MiB. Size it for the actual configured chunk and supported split
   count. Allocate once during [prefill configuration](../../engine/gpu/model.go#L349),
   reuse across sequential layers and validate capacity in the
   [paged plan op](../../backend/hip/bridge/yuuka_hip.h). Bind it from
   [addPrefillFullAttention](../../engine/gpu/plan.go#L462). Pass the dynamic
   execution position to the [native launcher](../../kernels/hip/prefill/prefill.hip#L791)
   to choose the measured crossover. A plan cached by row count can run at
   multiple prefix lengths. Freezing the decision at plan creation would
   apply the first chunk's route to later chunks.

## Verify the attention change

Compare one-, two- and four-split results with the unsplit reference on
identical paged KV. Cover a partial last page, noncontiguous physical pages,
query rows on both sides of a split boundary, and an entirely masked split
for an early query. Check the existing native attention tolerances, gated
output, deterministic reruns, real-model logits and greedy continuation.
Extend the [paged tests](../../tests/hip/paged_test.cpp) and the
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
