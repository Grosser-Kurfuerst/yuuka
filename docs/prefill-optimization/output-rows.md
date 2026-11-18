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
2. Use row zero of `p.x` as the compact working row. After `MarkTail`,
   when `rows > 1`, copy `hidden` FP32 elements from `p.x` byte offset
   `lastOff = (rows-1)*hidden*4` to byte offset zero in the same buffer.
   The ranges are disjoint and row zero's old residual has no remaining
   reader. For `rows == 1` skip the copy. Add a Go builder
   `AddCopyAt(dst, dstOff, src, srcOff, count)` and have `AddCopy` call
   it with zero offsets. Existing native `COPY` uses `off[0]` for the
   destination and `off[1]` for the source, and validates both spans.
   Add a plan test with a sentinel middle row to verify the copy direction.
   [Go copy builder](../../backend/hip/hip.go#L678) · [Native copy](../../backend/hip/bridge/bridge.cpp#L1868) · [Span validation](../../backend/hip/bridge/bridge.cpp#L1263)
3. Emit the FFN over this compact row and add its projected delta from
   `p.tmp` into row zero of `p.x`. In this first step the attention
   residual still runs on **all** rows before the copy. On the compact
   path output RMSNorm reads `p.x` at byte offset **zero**. Feed its
   `p.norm[0]` result to the current head.
   Keep `lastOff` for plans whose final hidden rows are still full-sized.
4. Separate row selection from the change in arithmetic. In the first A/B,
   pass the original chunk row count to projection **route selection** and
   one row to the **kernel** for each cropped projection. For a parent MMQ
   gate/up pair, emit the existing Q8_1 quantizer once with one row and
   issue both MMQs on its output. Do the same for a parent MMQ down
   projection on its own input. Carry through the original WMMA, BLAS or
   batched GEMV route on other weights. Compare against the uncropped last
   row. Then benchmark the usual one-row dispatch (batched GEMV), which
   changes K-quant activation arithmetic. Select it only where the
   logits and decode continuation pass the same gates.
   [Dispatch](../../backend/hip/dispatch.go#L60) · [Projection builders](../../engine/gpu/plan.go#L13)

## Step 2. Stop nonfinal chunks immediately after the final K/V write

Generate the following plan in `buildPrefillPlan` for the final full-attention
layer. Lines below represent plan operations. `MarkTail` applies to every
following operation. Emit it once on each plan-building branch.

```text
Earlier layers: usual attention or recurrence, then FFN, all rows
Ordinary generation, final layer:
    RMSNorm(p.x -> p.norm, rows)
    project K,V(p.norm -> p.k,p.v, rows)
    K head norm and RoPE(rows, position=chunk_start)
    paged KV write(rows, position=chunk_start)
    MarkTail()                                  # nonfinal execution stops here
    copy last p.x row -> p.x row 0 if rows > 1  # residual for both adds
    project Q/gate(p.norm at lastOff -> p.q row 0, one row)
    Q head norm and RoPE(one row, position=chunk_start+rows-1)
    paged gated attention(one query, same absolute position -> p.attn)
    output projection(p.attn -> p.tmp), add p.tmp into p.x row 0
    FFN(one row, including its residual), output norm(row 0), LM head(row 0)
PrefillLogits:
    usual full final attention and FFN, all rows
    MarkTail(), output norm and LM head for all requested rows in order
```

Keep the K/V pair grouped through `prefillProjections`. Emit the single
Q/gate projection separately, reading the FP32 normalized row at
`lastOff` before FFN overwrites `p.norm`. If Q needs MMQ, quantize that
one row at its FP32 byte offset into the start of `p.xQ8, p.xDS` after
the K/V consumers have finished. Q head norm and RoPE operate on the first
`headDim` entries of each `2*headDim` head. Attention consumes the gate
in the second half. [Existing Q/K/V path](../../engine/gpu/plan.go#L495)

Extend the two batch descriptors with a displacement in **tokens**. Use
the currently unused `u[0]` of `ROPE_BATCH` and `n[5]` of
`ATTENTION_PREFILL_PAGED`. Carry the value through the Go builder and the
native executor. Default zero preserves the full-row call. Before narrowing
to the native launcher's `int`, check the adjusted first and last positions
in wide arithmetic, and require `first + queryRows <= capacity` for paged
attention. Use the adjusted first position in the attention dispatch and
launch. K RoPE and the K/V write receive zero displacement. Version this
descriptor change in the C ABI and Go's `RequiredABI` together. The old
library accepts these opcodes and would ignore the new displacement.
[ABI compatibility check](../../backend/hip/hip.go#L51)
[Batch descriptor layouts](../../backend/hip/bridge/yuuka_hip.h#L312) · [RoPE execution](../../backend/hip/bridge/bridge.cpp#L1708) · [Paged attention execution](../../backend/hip/bridge/bridge.cpp#L1960)

**Worked case.** With chunk start 512, three rows and hidden width 2560,
K/V fills logical slots 512, 513 and 514. Read Q's input at byte offset
20,480 in `p.norm`. Its compact output starts at zero in `p.q`. Apply
Q RoPE at position 514 and let attention read slots 0 through 514. At
capacity 515 this call must succeed. A displacement of three would request
slot 515 and must fail the capacity check.

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
