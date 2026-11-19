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
