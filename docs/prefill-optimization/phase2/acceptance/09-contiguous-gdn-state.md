# Stage 5c Contiguous recurrent state trial

The [Stage 5c plan](../stage5c-contiguous-gdn-state.md) sets two
synchronous latency gates. Snapshot must finish within 1.5 ms
and NewSession plus Close must finish below 1.5 ms. The trial
reduced both operations but missed both gates in both process
orders. Keep the original per-layer state ownership. This result
completes the experiment and provides a control for further work.

## Implementation and compatibility

The experimental state layout allocated one owning HIP buffer per
session. It placed convolution history and matrix state in model
layer order, skipping full-attention layers. The plan passes
FP32-aligned byte offsets through existing descriptors. The bridge
checks the spans and rejects overlapping convolution and matrix
outputs. Decode (fused and legacy), trace, graph replay and all
three chunked recurrence paths used the same spans. The owner
lives until after all corresponding plans and graphs close.

The serialized layout matches the base revision `e4cde6f` byte
for byte. The reference [Qwen3.5-4B Q4_K_M GGUF](05-cli-chunk-policy.md#baseline-and-protocol)
has 24 recurrent layers. Per layer the convolution history is
98,304 bytes and the matrix is 2,097,152 bytes. Both arms hold
52,690,944 bytes per session, excluding table and dynamic controls.

After two nonuniform chunks of 61 and 67 tokens, the old and
contiguous snapshots have SHA-256
`b74a16bb0c3e3d410a20fd0d5665f2cec41a5a150376d18bbec1ce0f52ad6d78`.
Each binary cleared its own recurrent state, restored the
128-token position with the matching populated paged KV, then
restored the other's bytes and downloaded a new state snapshot.
Both restored snapshots matched the original byte for byte. Both produced
the greedy IDs `[1581, 383, 7670, 6353]`. The four complete
logit arrays, including the prefill row, have SHA-256
`76a898e8e718014ffe6bf8085188e1d1b765382827dc56ed7a8d850d43a8b589`.

## Paired performance

Tests ran in the AI container on gfx1151 with ROCm 10.1.0, HIP
7.16.26385, Go 1.24.4, Release builds and automatic power.
The request uses 128 tokens, chunk 512, F16 KV, greedy sampling,
and one active session. Snapshot timing includes creation of the
caller-owned byte slice and transfer. Lifecycle timing includes
plan construction and teardown and preserves table/dynamic handling.
The lifecycle sample in this first pair retains the prefill fixture's
idle session while creating and closing a second session.
Each process uses one warmup and seven timed samples. Order A is
old/new/new/old. Order B is new/old/old/new.

| Order | Operation | Old p50 [range], ms | Contiguous p50 [range], ms | Reduction |
| --- | --- | --- | --- | ---: |
| A | Snapshot | 7.708 [2.310–14.283] | 6.615 [1.903–11.159] | 14.2% |
| A | Restore | 1.029 [0.984–3.288] | 0.666 [0.656–4.806] | 35.3% |
| A | Reset | 0.266 [0.261–0.278] | 0.179 [0.169–0.226] | 33.0% |
| A | NewSession + Close | 3.815 [3.125–5.166] | 3.070 [2.838–6.235] | 19.5% |
| B | Snapshot | 6.873 [2.286–12.317] | 3.020 [1.837–7.931] | 56.1% |
| B | Restore | 1.066 [0.981–1.204] | 0.668 [0.663–6.832] | 37.3% |
| B | Reset | 0.270 [0.261–0.280] | 0.180 [0.173–0.220] | 33.2% |
| B | NewSession + Close | 3.850 [3.109–5.172] | 3.121 [2.730–7.144] | 19.0% |

The solo lifecycle probe closes the fixture session and retains only
its block table before timing. With one live session at a time,
the same paired protocol records 4.426 [3.533–5.604] → 2.935
[2.803–4.451] ms in order A and 4.460 [3.615–5.473] → 2.971
[2.542–3.654] ms in order B. Both candidate medians exceed the gate.

An isolated component probe measured a median 1.649 ms for
`hipMalloc` of the full 52,690,944-byte state even without decode
plan construction. Clear plus synchronization was 0.377 ms, and
free was 0.101 ms. A copy into a reused host destination whose
pages were already touched took 0.817 ms median (0.816–1.611 ms).
That copy timing excludes the host buffer costs of Snapshot. The
full operation with caller-owned bytes had a minimum of 1.837 ms
across the paired samples. The trace confirms that launch and copy
count reduction alone does not meet either latency budget.

| State API calls | Old | Contiguous |
| --- | ---: | ---: |
| NewSession allocation | 48 | 1 |
| NewSession clear | 48 | 1 |
| Close free | 48 | 1 |
| Snapshot download | 48 | 1 |
| Restore upload | 48 | 1 |
| Reset clear | 48 | 1 |

The trace brackets each phase with test-only allocations and
selects state allocations and transfers by their exact byte size.
The close count joins deallocation addresses to those state
allocations. The original rocprofv3 run initialized successfully
but did not serialize a trace artifact on Go test process exit.
The replacement is a temporary HIP API forwarding counter that
records each successful call as it completes. All timed samples
run without that instrumentation.

## Serving smoke

The localhost SSE test uses the same model, F16 KV, chunk 512,
context 576, concurrency one and greedy requests for both builds.
There are three measured requests per mode and one server warmup.
The off and miss inputs are unique within each run. Each hit is
the second submission of its matching miss.

| Mode | Old first-content / total p50, ms | Contiguous first-content / total p50, ms | Hits per arm |
| --- | --- | --- | ---: |
| Cache off | 84.369 / 111.296 | 86.629 / 111.836 | 0 |
| Cache-on miss | 84.389 / 121.053 | 86.974 / 119.445 | 0 |
| Populated prefix | 4.944 / 31.950 | 5.280 / 30.729 | 3 |

All streamed continuations match across arms and cache modes.
These smoke samples verify behavior and do not establish a
first-content latency improvement.

## Verification and reproduction

- Release CTest passed 10/10. Nonzero offsets produce bitwise
  identical output and state to independent buffers in all four
  existing recurrent fixture variants. The guards, overlapping,
  unaligned and truncated span cases pass.
- HIP-tagged Go build and vet passed. The layout, malformed
  restore, isolated sessions, reset, failed construction, repeated
  lifecycle, real-model fused/legacy, graph, graph fallback and
  speculative tests passed. `go test -tags hip ./serving` and
  `go test -race ./serving/...` passed in AI.
- The experimental sources, raw samples and intermediates are at
  `/tmp/yuuka-phase2-5-11.3wZQZG/5c` in AI. Use the matching HIP
  library for each binary. The old tree and library are in
  `../4d/ship`. Run
  `go test -tags hip ./engine/gpu -run '^TestStage5CStateLifecycle$' -count=1 -v`
  with `YUUKA_TEST_MODEL`. `YUUKA_5C_SAVE` writes the prefetched
  snapshot and `YUUKA_5C_RESTORE` loads the other binary's state
  after checking byte parity and clearing the local GDN state.
  `TestStage5CSoloLifecycle` captures the separate one-live-session
  pairs. `TestStage5CStateComponents` times the allocation and
  warm-host-buffer transfer independently.
- `hip_calls.cpp`, `state-api-{old,new}.log`, `sse.py` and
  `sse-samples.json` record the trace and actual serving test.

The read-only review ended with `NO_BLOCKING_ISSUES` after two
rounds. The second round verified the solo lifecycle and
clear-before-restore controls.
