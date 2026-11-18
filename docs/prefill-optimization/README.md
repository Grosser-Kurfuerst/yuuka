# Qwen prefill optimization

**Objective.** Run Qwen3.5-4B Q4_K_M prefill faster than upstream llama.cpp
v1.0.0 on gfx1151 at 128, 512, 2048 and 8192 input tokens. The proposed
acceptance margin is 5% higher median tokens/s **at each length**, using F16
KV, one request, an empty prefix cache and equal output requirements. The
scope is the dense hybrid `qwen35` HIP path. Other Qwen models, including
MoE, need their own supported runtime and measurements.

**Status.** This is a follow-up plan based on measurements at `bfd36c0`.
The implementation steps below are pending. The existing MMQ, split-register
recurrence, paged attention and streaming LM head are the starting point.
The earlier experiments and rejected variants are recorded in the
[previous roadmap](../roadmap3.md).

## Document index

- [Measurement and acceptance](measurement.md) — Reference versions, baseline timings, reproducible commands and the final comparison protocol.
- [Quantized projections and FFN](projections-and-ffn.md) — Q4_K and Q5_K MMQ specialization, dispatch and gate/up/down fusion.
- [Final-layer output rows](output-rows.md) — Plan-tail placement, final-query positions, residual offsets and state-equivalence tests.
- [Gated delta-net recurrence](recurrence.md) — Input preparation, wave-level state updates and validation of the complete recurrent operation.
- [Long-context attention and chunk size](long-context.md) — Split-key attention, partial softmax reduction, scratch ownership and chunk selection.

## Implementation order

At pp512 yuuka achieved **1765 vs 2203 tok/s** (80.1% of upstream).
The fixed-chunk trace puts **62.7 ms** of excess GPU time in projections
and **18.8 ms** in complete recurrence. Attention was already 0.6 ms
faster at this length. At 8192 tokens attention alone cost an extra
296.9 ms. Start with the repeated FFN projection, then crop work that
ordinary generation does not consume. Split-key attention has its own
long-prefix gate. [Timing conditions and family breakdown](measurement.md#attribution)

Each row is an independently measurable change. File counts are planning
estimates covering source and tests. Record the commit, numerical result and
before/after timings when a row lands.

| Order | Deliverable | Decision to make on gfx1151 | Estimated scope |
| --- | --- | --- | --- |
| 1 | Reproduce the fixed-512 baseline and the tuned 8K comparison | Establish unprofiled medians, warm GPU state and the current dispatch | Benchmark work |
| 2 | [Specialize one Q4_K FFN projection](projections-and-ffn.md#start-with-the-repeated-q4_k-shape) | Advance at ≤1.05 ms per call with a repeatable pp512 gain | 2–4 files |
| 3 | [Crop the final-layer FFN](output-rows.md#step-1-move-the-final-ffn-into-the-output-tail) | Compare logits, subsequent decode and last-layer time on single and multiple chunks | 2–5 files |
| 4 | Extend winning MMQ shapes and [move GDN preparation out of the token loop](recurrence.md#step-1-prepare-once-per-token-and-key-group), in separate changes | Aim for 160 ms of projections and 30 ms of complete recurrence in the pp512 trace | 3–6 files per change |
| 5 | [Fuse the FFN activation and quantizer](projections-and-ffn.md#fuse-where-the-intermediate-has-a-single-consumer), then evaluate paired gate/up | Keep the variant that improves complete FFN time and unprofiled prefill | 4–8 files |
| 6 | [Crop the final query after all K/V writes](output-rows.md#step-2-stop-nonfinal-chunks-immediately-after-the-final-kv-write) | Measure again at 8K to identify the remaining attention cost | 3–6 files |
| 7 | [Split long-prefix attention](long-context.md#split-the-key-range-for-long-prefixes) and sweep chunk sizes again | Include merge cost, scratch bytes and the short-context control | 4–8 files |
| 8 | Run the four-length comparison with matching token IDs and requested logits | Accept a lead only at lengths that clear the same-session margin | Benchmark and results |

At pp512, the two proposed family budgets plus the measured 25.5 ms of other
kernels give about **215.5 ms** versus the reference's **230.6 ms** of traced
kernels. This is a planning calculation. End-to-end acceptance uses the
unprofiled clock. Cropping a row also removes work attributed to MMQ, FFN or
attention. Re-profile after each change and calculate the next budget from
the new trace.
