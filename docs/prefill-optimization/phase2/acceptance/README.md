# Phase 2 progress and acceptance

- [Stage 1a Direct MMQ rows](01-direct-rows.md) — Partial token tiles, projection timings, same-chunk logits, and server first-content latency.
- [Stage 5a and 5b prefix snapshot](02-snapshot-critical-path.md) — Prefix capture gating, first-content timing, and cached generation validation.
- [Stage 1c short-chunk GDN](03-sub128-gdn.md) — Short-chunk measurement tools, prepared recurrence crossover, and model equivalence.
- [Stage 1b short-chunk projection trial](04-sub128-projections.md) — Route crossover, numerical blockers, and measured serving latency.
- [Stage 1b-1 FP32 streaming short tails](04a-streaming-short-tails.md) — Wide-shape crossover, CPU comparisons, and short-tail latency.
- [Stage 1b-2 Direct FP16 WMMA](04b-direct-fp16-wmma.md) — Medium short-chunk tiles, numerical parity, and prefill and serving acceptance.
- [Step 1d CLI prefill chunk sweep](05-cli-chunk-policy.md) — Paired chunk timing, boundary prompts, CLI first output, and device workspace.
- [Stage 2 narrow Q8_0 projections](06-narrow-q8-projections.md) — FP32 trial, equivalent WMMA pair, original-route numerics, and upstream prefill timing.
- [Stage 3A Direct Q6_K FFN down](07a-direct-q6k-ffn.md) — Packed staging, per-16 scales, CPU tensor parity, and FFN down timing.
- [Stage 3B Direct Q6_K attention V](07b-direct-q6k-attention.md) — V projection dispatch, alignment controls, and cumulative model timing.
- [Stage 4a F16 attention LDS padding](08a-f16-lds-padding.md) — K/V stride trial, numerical parity, attention timing, and bank-conflict measurements.
- [Stage 4b F16 split-key cells](08b-f16-split-cells.md) — Isolated direct and split timings, long-prompt numerics, and accepted dispatch ranges.
- [Stage 4c Q8_0 unsplit WMMA](08c-q8-kv-wmma.md) — Packed reader, measured tiled-prefix cutoff, numerical checks, and paired throughput.
- [Stage 4d Cross-format split-key](08d-cross-format-split.md) — Typed seed and partials, validation, and independently measured F32 and Q8_0 route windows.
- [Stage 5c Contiguous GDN state](09-contiguous-gdn-state.md) — Snapshot compatibility, lifecycle timings, HIP call reduction, and the latency gate decision.
- [Stage 6a Compact final-row GEMV](10a-compact-tail-gemv.md) — Five prepared projections, tail tile sweep, full-prefill timings, and the dispatch decision.
- [Stage 6b Residual and RMSNorm fusion](10b-residual-norm.md) — Bitwise tensor and model checks, launch attribution, and the paired pp512 decision.
- [Stage 7 Wide Direct MMQ workgroup mapping](11-direct-mmq-swizzle.md) — Group-size sweep, exact products, L2 counters and paired whole-prefill acceptance.
