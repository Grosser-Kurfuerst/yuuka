# Final-layer output rows

In the measured model the 32nd and final block uses full attention. Earlier
layers produce hidden rows needed by later layers. In the final layer,
future decode steps need the complete K/V history, while ordinary generation
needs the output for the last prompt row only. Today `buildPrefillPlan`
places `MarkTail` after every layer's FFN. At chunk 512 the final FFN alone
cost 4.9 ms at pp512. At 8K the last layer's attention through its FFN
residual cost about **188 ms across the first 15 nonfinal chunks**. This
includes their FFNs. Remeasure the end-to-end saving after cropping.
[Layer loop](../../engine/gpu/plan.go#L376) · [Session execution](../../engine/gpu/session.go#L218)

## Work required by output mode

| Call and chunk | Final-layer cache work | Final-layer output work after both steps |
| --- | --- | --- |
| Ordinary nonfinal chunk, including `Prefill(tokens, false)` | Normalize inputs, project K/V for every row, normalize/rotate K and write each position | Stop after the cache write |
| Ordinary final chunk | Same K/V work for all rows | Compute Q and its gate for the last row, attend to the full prefix, apply output projection, residual, FFN, output norm and LM head for that row |
| `PrefillLogits` for speculative verification | Update state for every supplied token | Produce every requested logit row in order |

Apply this specialization only when the actual last layer uses full attention.
Use the existing general graph for other last-layer types and the all-logits
plan. [Layer selection](../../model/qwen35/config.go#L280) · [Speculative path](../../engine/gpu/session.go#L246)

## Step 1. Move the final FFN into the output tail

1. In `buildPrefillPlan`, run the last layer's attention and residual as
   today. For ordinary generation, mark its subsequent FFN and output ops
   as tail work. `Session.Prefill` already passes
   `final && start+n == len(tokens)` to `Plan.Execute`. Keep the cached
   plan reusable for a different absolute position or final flag.
2. Select the last hidden row at byte offset
   `(rows - 1) * hidden * sizeof(float)`. Give the single-row FFN a
   well-defined input and destination through offset-aware operations or
   one plan-local row copy. Read the FFN residual from the same selected
   row. Point output norm and the LM head at the actual result buffer.
   The current output norm already applies a last-row offset. Applying it
   again to compacted output would read stale data.
3. Verify single-row arithmetic against the final row of the existing full
   graph. Reducing the shape to M=1 changes the default dispatch to GEMV.
   The original K-quant MMQ path quantizes its input to Q8_1. If the change
   in arithmetic breaks the existing numerical gates, preserve the parent
   projection's activation quantization for the selected row before
   enabling the route. [Dispatch](../../backend/hip/dispatch.go) · [Plan builders](../../backend/hip/hip.go)

## Step 2. Stop nonfinal chunks immediately after the final K/V write

1. Split `addPrefillFullAttention` at the K/V write for this terminal
   case. Keep normalization, K and V projection, K head norm and K RoPE for
   **all** rows before the tail marker. Write each paged slot at the chunk's
   absolute start plus row. Preserve Q8_1 input sharing where applicable.
2. Move the final Q projection, Q head norm/RoPE and gated attention after
   the marker. Read only the selected input row and retain the interleaved
   Q/gate stride. Both Q RoPE and causal attention require the absolute
   position `chunk_start + rows - 1`. Current batch ops receive the dynamic
   chunk start. Carry a checked per-op row displacement for the cropped
   query if needed, and apply it only to these operations. Test the memory
   offset and the position adjustment independently at a nonzero chunk start.
3. Reuse the existing paged cache and output tail. Apply the attention
   output projection and first residual to the selected input row, then
   run the single-row FFN. If a bridge descriptor gains a displacement or
   offset, specify its units, validate its span and cover default-zero
   behavior. [Paged op contracts](../../backend/hip/bridge/yuuka_hip.h) ·
   [Native dispatch](../../backend/hip/bridge/bridge.cpp)

## Acceptance

Use the full-row plan as a test oracle. Cover one token, exactly one chunk,
one chunk plus one token, and two full chunks. Compare final logits and the
next several greedy tokens. Exercise a nonfinal call followed by a separate
final call and verify persisted K/V by reading populated pages or comparing
subsequent attention against the full-row oracle. A recurrent-state snapshot
alone omits the full-attention KV pool. Verify every row in the speculative
path. [Chunk tests](../../engine/gpu/prefill_test.go) · [All-logits tests](../../engine/gpu/spec_fixture_test.go)

Trace pp512 and 8K. Count final-layer FFN projection rows, final-layer query
rows and K/V writes. Confirm that nonfinal chunks retain every cache write
and that the final logit is computed once. Time the full prompt again at
chunk 512 and the current best measured chunk size.
