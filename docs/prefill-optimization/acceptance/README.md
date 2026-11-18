# Prefill optimization progress and acceptance

The implementation follows the [Qwen prefill plan](../README.md) in order.
Each completed iteration includes its acceptance record and one commit.
The [measurement protocol](../measurement.md) defines the target model,
reference build, and comparison conditions.

## Document index

- [Iteration 1 baseline](01-baseline.md) — Fixed-chunk throughput, 8K chunk sweep, and GPU verification.

## Progress

| Plan step | Scope | Status |
| --- | --- | --- |
| 1 | Reproduce the pre-optimization baseline | Accepted |
| 2 | Specialize the Q4_K FFN projection | Pending |
| 3 | Crop the last-layer FFN | Pending |
| 4a | Extend measured MMQ improvements to additional shapes | Pending |
| 4b | Reduce GDN input preparation and synchronization | Pending |
| 5 | Fuse SwiGLU with down-input quantization and evaluate paired gate/up | Pending |
| 6 | Crop the last-layer query after writing K/V | Pending |
| 7 | Split long-prefix attention and repeat the chunk sweep | Pending |
| 8 | Compare the four prompt lengths using identical inputs and output requirements | Pending |
