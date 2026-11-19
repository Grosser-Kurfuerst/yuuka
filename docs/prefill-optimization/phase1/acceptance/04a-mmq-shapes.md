# Iteration 4a MMQ shapes

## Implementation

The [MMQ kernel](../../../../kernels/hip/gemm/mmq_gemm.hip) reuses the gfx1151
direct WMMA tile for additional full-tile Q4_K and Q5_K projections. A
staging thread reads a pair of low and high nibbles together and applies
separate FP32 scales and minima. For Q5_K it also reads the packed
high bits once for that pair and extracts the bit plane for each
subblock. The [dispatch table](../../../kernels.md#gfx1151-direct-mmq-specialization)
lists the measured cells. The earlier FFN specialization uses its
compile-time dimensions.

The [native tests](../../../../tests/hip/prefill_test.cpp) add full CPU
product checks for one new Q4_K and one new Q5_K cell. They include
subblocks on both sides of the 128-value staging boundary and across
multiple 256-value quantization blocks.

## Shape screening

The shape list comes from the committed model's verified GGUF tensor
descriptors on AI. The sweep used seeded synthetic packed weights
and repeated the same cells in control, candidate, candidate, control
order. Each process excluded two warmups and traced five timed MMQ
dispatches per cell. The table gives the mean of two independent
medians for each build, in milliseconds, control → candidate.

| Weights | Input → output | M=128 | M=512 | M=2048 |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 2560 → 1024 | 0.061 → 0.044 | 0.153 → 0.106 | 0.554 → 0.383 |
| Q4_K | 2560 → 4096 | 0.145 → 0.110 | 0.608 → 0.394 | 2.196 → 1.746 |
| Q4_K | 2560 → 8192 | 0.317 → 0.222 | 1.120 → 0.835 | 4.439 → 3.678 |
| Q4_K | 2560 → 9216 | 0.324 → 0.229 | 0.973 → 0.954 | 5.082 → 4.133 |
| Q4_K | 4096 → 2560 | 0.142 → 0.103 | 0.489 → 0.347 | 2.209 → 1.735 |
| Q4_K | 9216 → 2560 | 0.300 → 0.234 | 1.173 → 0.910 | 5.154 → 4.210 |
| Q5_K | 2560 → 8192 | 0.328 → 0.238 | 1.171 → 0.910 | 4.710 → 3.920 |
| Q5_K | 4096 → 2560 | 0.147 → 0.110 | 0.516 → 0.389 | 2.412 → 1.879 |

`M=2048` covers the strongest previously observed long-prompt setting.
Raw traces, the GGUF shape inventory and the sweep source are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-04a-99NiWj/`
and the preceding iteration directory on AI.

## Final validation

The allowlisted build and the stage 3 control used the same Qwen3.5-4B
Q4_K_M model and F16 KV on the AI Ryzen AI Max+ 395 (gfx1151). The
model SHA-256 is `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
ROCm 10.1 reports HIP 7.16.26385. The device was in automatic power
mode and the pre-run sensor readings covered 42–56 °C. Both builds used
one request, fixed synthetic prompt tokens, one greedy output token
and two warmups per process.

| Prompt | Chunk | Timed runs per process | Control tok/s | Candidate tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 512 | 5 | 1506.17 | 1700.01 | +12.87% |
| 512 | 512 | 5 | 1908.38 | 2078.82 | +8.93% |
| 2048 | 2048 | 5 | 1802.37, 1797.45 | 2047.16, 2037.67 | +13.47% |
| 8192 | 2048 | 3 | 1621.11 | 1801.70 | +11.14% |

The pp2048 change compares the means of two process medians. The process
order was control, candidate, candidate, control.

The three pp512 position-zero profiles had the same 248 GEMM operations.
The median GEMM sum moved from 196.962 to 175.962 ms, down 10.66%.
The projection family is 15.962 ms above the 160 ms planning budget.
Unprofiled throughput is an independent measurement from the trace.
An additional three-sample trace instrumented the operation range of
each of the 32 FFNs, including their residual adds and the final row
copy. The median complete FFN sum fell from 121.202 to 117.852 ms.

The compiler reports 133 → 125 VGPRs for both of the newly dispatched
Q4_K and Q5_K variants, zero VGPR spills, and zero private scratch.
The launch reserves the existing 30 KiB of dynamic LDS per workgroup.
The fixed FFN cell stays at 119 compiler VGPRs. Persistent and transient
buffer capacities at the plan level are the same in both builds.

The profiler recorded the following median counts per MMQ dispatch in
millions after two warmups and five samples. Input data and launch order
match the isolated shape sweep. Wave counts are unchanged. The load
counter measures instructions, and the read counter measures requests
at the vector cache.

| M=512 shape | VALU | LDS | Flat loads | Vector cache reads |
| --- | ---: | ---: | ---: | ---: |
| Q4_K 2560 → 4096 | 29.082 → 22.469 | 6.226 → 3.727 | 0.942 → 0.451 | 7.537 → 7.209 |
| Q4_K 9216 → 2560 | 64.169 → 50.118 | 14.008 → 8.387 | 2.120 → 1.014 | 16.957 → 16.220 |
| Q5_K 2560 → 8192 | 63.734 → 48.951 | 12.452 → 7.455 | 2.212 → 1.065 | 17.695 → 17.039 |
| Q5_K 4096 → 2560 | 31.575 → 24.375 | 6.226 → 3.727 | 1.106 → 0.532 | 8.847 → 8.520 |

Release build and all nine native tests passed. The new Q4_K and Q5_K
checks had maximum absolute errors of 0.0000801 and 0.000275.
The stage 2 FFN cell checked at 0.000114. Go build, Go vet with and
without the HIP tag, Go tests, race tests, and the HIP Go suite passed.
The real-model suite covered long chunked prefill, cropped final FFN,
CPU/GPU logits, separate decode and speculative greedy continuation.
One optional MoE test and two optional SentencePiece tests skipped
because their external fixtures were unavailable.

## Review and decision

Accepted for the measured dispatch cells. The projection budget remains
open. The first read-only review requested the pp2048, complete FFN,
compiler and hardware counter measurements, which were added in the
follow-up. The second round returned `NO_BLOCKING_ISSUES`.
`git diff --check` passed.
