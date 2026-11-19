# Stage 5a/5b: Prefix snapshot off the first-token path

> **Status:** Steps 5a and 5b landed in `6e267d7` and `c09cd06`. See
> [Stage 5a/5b acceptance](acceptance/02-snapshot-critical-path.md).

Implementation plan for Steps 5a and 5b of the
[Phase 2 roadmap](roadmap.md#stage-5-p5-yuuka-server-prefix-cache-snapshot-and-contiguous-gdn-state-3-steps).
Step 5c (contiguous GDN state storage) stays a separate change.

## 1. Goal and scope

Remove the recurrent-state snapshot and logit serialization from the critical
path between the last prefill chunk and the first streamed token in
`yuuka-server`.

The problem, read from the code at `846fc4f`:

- `generationTask.PrefillOnce` in [task.go](../../../serving/task.go), after the
  final chunk, calls `t.seq.Snapshot()` and `serializeLogits(logits)` and only
  then calls `t.sampleAndEmit(logits)`, which samples the first token and
  invokes the sink.
- Whether the prefix cache is enabled is only checked later, in
  `Manager.finalize` in [manager.go](../../../serving/manager.go), when the
  request closes. With `--prefix-cache=false` the snapshot and serialization are
  computed and then discarded.
- The `yuuka-server` sink writes the SSE chunk and calls `http.Flusher.Flush`
  synchronously on the scheduler goroutine, so the client cannot see the first
  token until the snapshot has finished.

Measured cost at `9da6e77` on the reference container (`prompt=128`,
`kv=f16`, median of 5): `Session.Snapshot()` takes **7.95 ms** (48 synchronous
device-to-host copies, 48 MiB of convolution and matrix state). This cost does
not depend on prompt length. `serializeLogits` converts 248,320 FP32 values on
the host; its cost has not been measured separately and is recorded in Step 1.

Out of scope:

| Item | Reason | Handled by |
| --- | --- | --- |
| Consolidating the 48 per-layer GDN buffers | Changes session allocation and the per-layer offsets; answers "how fast is the snapshot" rather than "how early is the first token" | Step 5c |
| Prefix-cache lookup, insertion, or eviction policy | Unchanged | — |
| CLI (`yuuka run`, `yuuka bench`) | The engine path does not call `Snapshot` on prefill | — |

## 2. Design

### 2.1 Step 5a: capture only when a cache exists

Pass a `capturePrefix` flag into the generation task when the manager builds
it. The flag is `m.cache != nil`. When it is false, `PrefillOnce` skips
`Snapshot()` and `serializeLogits()` entirely. `PrefixReady` already returns
false without a snapshot, and `Manager.finalize` already returns early without
a cache, so no other path changes.

### 2.2 Step 5b: emit first, then capture

When `capturePrefix` is true, `PrefillOnce` samples and emits the first token
first, then captures the snapshot and serializes the logits, and only then
returns.

Correctness conditions, checked against the code:

- **State is still at the prompt boundary.** The snapshot must be taken before
  any `Decode` mutates the convolution and matrix state. `PrefillOnce` runs on
  the scheduler goroutine and the scheduler does not call `DecodeOnce` for the
  task until `PrefillOnce` returns, so capturing before the return is
  sufficient.
- **Logits are unchanged by sampling.** `Sampler.Sample` in
  [sampling.go](../../../engine/sampling.go) reads the logits slice and builds
  its own candidate list; it does not write the slice. The session-owned logits
  buffer is only overwritten by the next `Decode`. Serializing after sampling
  therefore stores the same bytes as today.
- **Prefix eligibility is unchanged.** Today the snapshot is captured even when
  the first token is a stop token, when `MaxTokens` is 1, or when the sink
  returns an error. The new order captures in all of these cases as well, after
  `sampleAndEmit` returns and before the error is propagated, so the set of
  requests that insert a cache entry is identical.

Total request latency does not change in Step 5b: the same work runs, only
after the first token instead of before it. Step 5a removes the work outright
when no cache exists.

## 3. Changes

### 3.1 [task.go](../../../serving/task.go) (~20 lines)

- Add a `capturePrefix bool` field to `generationTask` and a matching
  parameter to `newGenerationTask`.
- Move the snapshot and serialization into a helper and call it after the
  first token:

  ```go
  t.promptDone = true
  sampleElapsed, err := t.sampleAndEmit(logits)
  t.sampling += sampleElapsed
  stats.SamplingTime = sampleElapsed
  if t.capturePrefix {
      // The first token is already streamed. Capture the prompt boundary
      // before returning, because the next DecodeOnce mutates the state.
      t.capturePromptPrefix(logits)
  }
  if err != nil {
      t.failure = err
      return stats, true, err
  }
  return stats, true, t.failure
  ```

  `capturePromptPrefix` performs the existing `Snapshot()` call (setting
  `promptState` and `snapshotOK` on success) and the existing
  `serializeLogits(logits)` assignment, with unchanged error semantics: a
  snapshot failure only disables cache insertion.

### 3.2 [manager.go](../../../serving/manager.go) (~2 lines)

Pass `m.cache != nil` as `capturePrefix` from `Manager.build`.

### 3.3 Tests (~90 lines)

In the `serving` package, using the tiny fixture tokenizer that
[manager_test.go](../../../serving/manager_test.go) already builds:

1. **Recording sequence.** A test `Sequence` that returns fixed logits from
   `Prefill`, appends `"prefill"`, `"snapshot"`, and `"decode"` to a shared
   event log, and a sink that appends `"emit"` to the same log.
2. **Cache disabled.** With `capturePrefix=false`, a full request records no
   `"snapshot"` event and `PrefixReady` reports false.
3. **Cache enabled, order.** With `capturePrefix=true`, the log reads
   `prefill, emit, snapshot, decode, …`: the first emission precedes the
   snapshot, and the snapshot precedes the first decode.
4. **Terminal first token.** With `MaxTokens=1`, and separately with a stop
   token as the first sample, the snapshot is still captured and `PrefixReady`
   reports true.
5. **Cached logits.** The serialized prompt logits equal the logits returned by
   `Prefill`, byte for byte.

The existing [TestGPUPrefixCacheMatchesFreshRun](../../../serving/gpu_test.go)
must keep passing; it checks that a cached-prefix request generates the same
tokens as a fresh run.

### 3.4 [bench/serving](../../../bench/serving/main.go) (~15 lines)

Add a `-unique-prompts` flag that replaces the first prompt token of each
request with a request-specific token, so no request is a prefix-cache hit
even with `-prefix-cache`. Without the flag, every request reuses the same
prompt and the cache would serve them, which hides the snapshot cost.

### 3.5 Documentation (~40 lines)

- [serving.md](../../serving.md) section 7: state that the snapshot is
  captured only when the prefix cache is enabled, and that it is taken after
  the first token is emitted and before the first decode step.
- Add `phase2/acceptance/02-snapshot-critical-path.md` with the measurements,
  and list it in the phase 2 acceptance index.

**Total scope:** about 170 lines across 2 source files, tests, one benchmark
tool, and documentation. No kernel, ABI, or session-layout change.

## 4. Steps and acceptance

| Step | Work | Acceptance |
| --- | --- | --- |
| 1. Step 5a | Sections 3.1 (flag only), 3.2, tests 1–2, 3.4 | `go test -race ./serving/...` passes. On the reference container, `bench/serving -prefix-cache=false -unique-prompts` TTFT p50 drops by at least 7 ms at 140, 530, and 2050 prompt tokens against control, and total latency drops by a similar amount. `serializeLogits` time is recorded. |
| 2. Step 5b | Section 3.1 (reorder), tests 3–5 | `go test -race ./serving/...` and `go test -tags hip ./serving/...` pass, including `TestGPUPrefixCacheMatchesFreshRun`. With `-prefix-cache -unique-prompts`, TTFT p50 drops by at least 7 ms against control while total latency stays within noise. |
| 3. Real-model prefix reuse | `yuuka-server` with the real GGUF and `--prefix-cache=true`: request a prompt, repeat it (full hit), then extend it (partial hit) | Generated tokens of the hit requests are identical to the same requests on a `--prefix-cache=false` server. |
| 4. Server SSE | `yuuka-server --kv-type f16`, cache off and on, chat prompts of about 140, 530, and 2050 tokens with a per-request nonce in the user message | Repeatable reduction of first-content SSE latency by at least 7 ms in both cache modes against control. |
| 5. Docs and commit | Section 3.5 | Zero broken relative links, `git diff --check` passes, Conventional Commits. Steps 5a and 5b may land as two commits. |

## 5. Expected gains

- **Cache disabled:** about 8–8.5 ms lower TTFT and total latency on every
  request: the measured 7.95 ms snapshot plus the not yet measured logit
  serialization.
- **Cache enabled, cache miss:** the same TTFT reduction; total latency is
  unchanged because the work still runs before the first decode step.
- **Cache hit:** unchanged; a full hit already skips prefill and snapshot.

Relative effect on the measured server TTFT: about 6% at ~140 tokens
(132.8 ms), about 3% at ~530 tokens, under 1% at ~2050 tokens.

## 6. Measurement environment

- Reference container `192.168.254.101:11022` (gfx1151, ROCm 10.1.0,
  HIP 7.16.26385), model SHA-256
  `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
  F16 KV, automatic power mode, one GPU process at a time.
- Build control and candidate in an isolated `/tmp` directory with a private
  `GOCACHE`, and remove every build artifact, server process, and log after
  the run.
