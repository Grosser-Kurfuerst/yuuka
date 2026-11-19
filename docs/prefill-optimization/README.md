# Qwen prefill optimization

Module index for Qwen3.5 (`qwen35`) HIP prefill optimization on `gfx1151`,
organized into Phase 1 (synthetic power-of-two benchmark parity and lead over
upstream `llama.cpp v1.0.0`) and Phase 2 (real-world arbitrary prompt lengths,
tail chunks, and serving TTFT).

## Phase 2: Real-world prefill optimization

- [Phase 2 priority roadmap](phase2/roadmap.md) — Diagnostic `rocprofv3` attribution, P1–P7 priority ranking, execution order, phased plan, and current `260094f` benchmark progress.
- [Phase 2 progress and acceptance](phase2/acceptance/README.md) — Validation records for landed steps (`1a`, `5a/5b`, `1c`) and the blocked `1b` trial.
- [Stage 1a: Direct MMQ for arbitrary row counts](phase2/stage1a-direct-rows.md) — Design, change points, and acceptance gates for `rows >= 128` Direct MMQ (landed in `3caea06`).
- [Stage 5a/5b: Prefix snapshot off the first-token path](phase2/stage5ab-snapshot-critical-path.md) — Server prefix-capture gating and post-emission ordering plan (landed in `6e267d7`, `c09cd06`).
- [Stage 1b/1c: Sub-128-row projections and GDN preparation](phase2/stage1bc-sub128-rows.md) — Prepared GDN crossover (`260094f`) and sub-128 projection routes (currently blocked on `1b` numerical gates).

## Phase 1: Upstream benchmark optimization (completed)

- [Phase 1 implementation plan](phase1/README.md) — Target margins, family time budgets, and 8-step delivery order against upstream `llama.cpp v1.0.0`.
- [Progress and acceptance](phase1/acceptance/README.md) — Index of per-iteration validation records and review decisions for Phase 1.
- [Measurement and acceptance](phase1/measurement.md) — Reference versions, baseline attribution, and matched-input comparison protocol.
- [Quantized projections and FFN](phase1/projections-and-ffn.md) — `Q4_K`/`Q5_K` Direct MMQ specialization and fused SwiGLU quantization design.
- [Final-layer output rows](phase1/output-rows.md) — Plan-tail placement and final-row cropping for last-layer query and FFN.
- [Gated delta-net recurrence](phase1/recurrence.md) — Per-token/group input preparation and split-register recurrence mapping.
- [Long-context attention and chunk size](phase1/long-context.md) — Split-key WMMA prefill attention, prefix-maximum seeding, and chunk selection.
