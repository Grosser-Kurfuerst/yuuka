# Qwen prefill optimization

Module index for Qwen3.5 (`qwen35`) HIP prefill optimization on `gfx1151`,
organized into Phase 1 (synthetic power-of-two benchmark parity and lead over
upstream `llama.cpp v1.0.0`) and Phase 2 (real-world arbitrary prompt lengths,
tail chunks, and serving TTFT).

## Phase 2: Real-world prefill optimization

- [Phase 2 priority roadmap](phase2/roadmap.md) — Diagnostic `rocprofv3` attribution, real-world length cliffs, P1–P7 priority ranking, and phased implementation plan.
- [Stage 1a: Direct MMQ for arbitrary row counts](phase2/stage1a-direct-rows.md) — Design, change points, staged steps, acceptance gates, and expected gains for extending Direct MMQ to every `rows >= 128`.

## Phase 1: Upstream benchmark optimization (completed)

- [Phase 1 implementation plan](phase1/README.md) — Target margins, family time budgets, and 8-step delivery order against upstream `llama.cpp v1.0.0`.
- [Progress and acceptance](phase1/acceptance/README.md) — Index of per-iteration validation records and review decisions for Phase 1.
- [Measurement and acceptance](phase1/measurement.md) — Reference versions, baseline attribution, and matched-input comparison protocol.
- [Quantized projections and FFN](phase1/projections-and-ffn.md) — `Q4_K`/`Q5_K` Direct MMQ specialization and fused SwiGLU quantization design.
- [Final-layer output rows](phase1/output-rows.md) — Plan-tail placement and final-row cropping for last-layer query and FFN.
- [Gated delta-net recurrence](phase1/recurrence.md) — Per-token/group input preparation and split-register recurrence mapping.
- [Long-context attention and chunk size](phase1/long-context.md) — Split-key WMMA prefill attention, prefix-maximum seeding, and chunk selection.
