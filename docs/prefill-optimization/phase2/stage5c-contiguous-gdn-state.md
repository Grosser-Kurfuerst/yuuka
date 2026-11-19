# Stage 5c — One recurrent-state allocation per GPU session

Execution order 9. Steps 5a/5b already gate prefix capture and emit the first token before taking an enabled snapshot. Step 5c reduces allocation and copy time for the remaining capture, restore, reset and session lifecycle. The historical isolated snapshot median was 7.95 ms. The serving A/B saw 2.56–3.27 ms for capture in process. Measure those workloads separately. [Serving acceptance](acceptance/02-snapshot-critical-path.md)

## 1. Preserve the binary layout and bind offsets

For each recurrent layer in increasing **model layer index**, append its convolution history then its matrix state. Skip full-attention layers. From the config compute:

- `convDim = 2 * SSMGroupCount * SSMStateSize + SSMInnerSize`
- `C = 4 * (SSMConvKernel - 1) * convDim` bytes
- `R = 4 * SSMTimeStepRank * SSMStateSize²` bytes

For recurrent ordinal `j`, use `convOff = j*(C+R)` and `recOff = convOff+C`. The length is `recurrentCount*(C+R)`. Check products, additions, FP32 alignment and host slice representability before allocation. Preserve this packed layout byte for byte, without headers or padding. Store offsets by model-layer index to avoid counting full-attention layers as state. A model with no recurrent layers has an empty snapshot and no state allocation.

Use an owning `*hip.Buffer` in [Session](../../../engine/gpu/session.go) and ordinary byte offsets in its [plans](../../../engine/gpu/plan.go). Today a HIP buffer handle owns its allocation and release. Representing a slice as another owning handle would introduce a double free. The existing C descriptor already has `off[]`. Add offset-taking Go append forms (keep existing methods forwarding with zero) for fused `AddConvL2Norm` / `AddRecurrentTokenAt` and legacy `AddConv1D` / `AddGDNAt`. Extend `RecurrentChunkArgs` with state offsets for slots 0/1. Trace, graph capture, prepared, chunkwise and ordinary paths must bind the same layer ranges. Audit each touched native validator and launch pointer to honor those offsets. Check exact-fit spans and reject overlapping simultaneously writable conv/matrix spans. [Go binding](../../../backend/hip/hip.go) · [Descriptor and validation](../../../backend/hip/bridge/bridge.cpp)

**First verification.** Exercise nonzero offsets and guards on either side in the native plan and recurrent tests, including the second recurrent layer with an intervening attention layer. Compare outputs and both states to independent buffers. Update the [decode profiler](../../../engine/gpu/decode_profile_test.go) to attribute state by owner/offset instead of counting the shared allocation 48 times.

## 2. Switch the lifecycle as one change

| Operation | Required implementation |
| --- | --- |
| NewSession | Allocate once through `allocTransient`; assign the owner immediately for cleanup. For a new state, clear the full range and observe completion before use. For supplied bytes, validate length then restore the entire initialized range. Keep table/dynamic buffers separate. |
| Reset | Under the existing step lock, issue one clear and synchronize. Only then reset position and decode counters. Keep the device address stable for cached plans/graphs. |
| Snapshot | Under the existing lock and after prior writes complete, allocate one caller-owned host byte slice and issue one synchronous D2H. |
| Restore | Validate the entire length before the first write. Perform one synchronous H2D into the existing owner, serialized with use of the same state. Preserve the caller's position/table contract. Audit existing lock ownership before adding synchronization. |
| Close / failed construction | Release graph/plans that reference the address before releasing the owner once. Preserve idempotent close and cleanup on partial allocation or plan-build failure. |

Update [snapshot/restore callers and tests](../../../engine/gpu/session_test.go). Prefix capture still finishes before the next decode mutates the session. A snapshot serializes only GDN state. An integration restore must bind the corresponding paged KV table and absolute prefix position before comparing continuation. Speculative rollback also consumes this byte layout.

## Acceptance

1. Generate a control snapshot with the old per-layer serializer after multiple chunks of nonuniform tokens. Verify every byte against a new snapshot for the same inputs, restore old → new and new → old using matching KV and compare full logits and subsequent greedy tokens exactly. Include reset-to-zero, two sessions with distinct state, malformed lengths (no mutation), speculative rollback and graph replay. Exercise a post-allocation failure and repeated create/close cycles. [Graph tests](../../../engine/gpu/graph_replay_test.go) · [Real-model checks](../../../engine/gpu/runtime_test.go)
2. On the reference GGUF, trace the **state subset** of calls: 48 → 1 each for allocating/freeing state, clearing a new/reset state, downloading a snapshot and uploading a restore. Device bytes per live session equal the old sum. Table and dynamic allocations still appear in the whole-session trace.
3. Reproduce prompt 128, chunk 512, F16 KV, one session at a time. Time synchronous snapshot through host bytes available. Time complete NewSession plus Close including plan setup/teardown. Pair pre/post binaries, ≥1 warmup + 5 reps per arm, record bytes, medians/spread and reverse the order. Require **snapshot ≤ 1.5 ms** and **NewSession + Close < 1.5 ms**. Report Restore/Reset independently. An isolated speedup below these gates remains a partial result.
4. Smoke cache-off, cache-on miss and populated-prefix hit via the serving path. Record first-content SSE, total latency, hit count and exact continuation. Keep the ordering from 5a/5b and verify `go test -race ./serving/...`, native CTest and HIP session/graph/speculative tests on gfx1151. A lifecycle win is measured independently of any SSE change.

Estimate: 90–160 production lines across session layout, plan binding and offset-aware Go methods, plus targeted validation, profiler and compatibility tests. No device kernel algorithm change is required.
