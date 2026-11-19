# Iteration 5 FFN fusion

## Implementation

The [prefill plan](../../../../engine/gpu/plan.go) uses the fused quantizer
where gate, up and down all select MMQ. After both gate and up GEMMs
complete, the [kernel](../../../../kernels/hip/gemm/mmq_gemm.hip) reads their
FP32 projections and writes the down-input digits and scales into the
same Q8_1 buffers. There is no activated FP32 write and read. At
`M=512, FFN=9216` the change removes one launch and 36 MiB of logical
traffic per eligible full-row FFN. The persistent and transient buffer
capacities are unchanged. ABI 6 adds the opcode and checks allocation
spans, alignment and output overlap.

The [native comparison](../../../../tests/hip/prefill_test.cpp) covered a
zero block, signed and exact halfway values, 17 rows, 65 down-output
channels and nonzero offsets. Fused Q8_1 digits, scale/sum pairs and
the following Q4_K down output matched the separate activation and
quantizer exactly across two runs. The [plan tests](../../../../tests/hip/plan_validation_test.cpp)
covered all four short or unaligned buffer spans, overlapping output
and unsupported width. All nine CTest tests passed.

## Initial pp512 comparison

The control was iteration 4b at `557c7a3`. All timings used the same
[AI environment and model](01-baseline.md#environment), F16 KV, one
request, one greedy output, two warmups and five samples. The process
order was control, fused, fused, control. The control medians were
2337.05 and 2310.96 tok/s, and the fused medians were 2383.57 and
2366.39 tok/s. The mean of medians improved by 2.19%.

## Complete FFN and paired gate/up trial

The isolated paired kernel retained the `128×64` direct MMQ tile.
Per K step it staged one activation tile, accumulated gate, reused
weight LDS for up, and applied SwiGLU on the two FP32 totals at the
end. Down used the existing separate quantizer on that result. The
experiment covered the measured Q4_K `M=512, K=2560, N=9216` pair,
which applied to 31 of the 32 FFNs in the ordinary prefill plan.

Three position-zero profiles summed the operations from the FFN's
input norm through its residual add, with the compact row copy
included for the last layer. Times are diagnostic medians.

| Path | FFN ops | Projection launches | Projection sum (ms) | Complete FFN (ms) |
| --- | ---: | ---: | ---: | ---: |
| Separate SwiGLU and Q8_1 | 257 | 96 | 109.089 | 120.659 |
| Fused SwiGLU and Q8_1 | 225 | 96 | 106.948 | 116.636 |
| Paired gate/up and separate Q8_1 | 194 | 65 | 217.003 | 223.457 |

The compiler reported 39 VGPRs and zero spills or private scratch for
the fused quantizer. The plain quantizer used 20 VGPRs and zero spills.
The paired prototype reserved the same 30 KiB of dynamic LDS as the
separate direct MMQ tile. Keeping both products live took 192 VGPRs
with 168 spills and 676 bytes of private scratch per lane. The fixed
single-product tile uses 119 VGPRs and zero spills. The paired trial
pp512 medians were 1595.34 and 1580.70 tok/s versus 2348.09 and 2372.62
tok/s for the fused quantizer in the same A/B. The mean of paired
medians was 32.72% lower. The fused quantizer is the selected path.

## Final build

Both the fused and paired builds matched the control's full-vocabulary
logits, three subsequent greedy decode steps, and the checked recurrent
layer states byte for byte at pp128, pp512, two calls of 512 tokens and
pp2048. The control is the accepted iteration 4b build. The final
unprofiled comparison used equal chunk sizes and the same settings as
the initial pp512 comparison.

| Prompt | Chunk | Samples | Control tok/s | Fused tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 5 | 1866.58 | 1881.08 | +0.78% |
| 512 | 512 | 5 | 2323.50 | 2319.18 | -0.19% |
| 2048 | 2048 | 5 | 2301.27 | 2354.99 | +2.33% |
| 8192 | 2048 | 3 | 2001.28 | 2021.55 | +1.01% |

The single final pp512 pair was effectively flat. The initial four
processes and a second interleaved run both showed a gain. The second
run used two warmups and seven samples per process. The mean of the
two medians rose 1.19% from the control medians of 2286.06 and
2296.35 tok/s to the fused medians of 2332.74 and 2304.31 tok/s. The
device used automatic power mode. The final four-length measurement
had starting sensor readings of 45–58 °C.

Release build, `ctest --test-dir build --output-on-failure -V`, Go and HIP
CLI builds, `go vet ./...`, `go vet -tags hip ./...`, `go test ./...`,
`go test -race ./...` and
`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -count=1 -v`
passed on AI. The HIP run exercised the CPU/GPU, separate decode,
long-context, final FFN and speculative tests. The optional MoE and two
SentencePiece cases skipped because their fixtures were unavailable.

The full `build/yuuka_prefill_sweep` completed for both builds. Its
standalone MMQ path includes the plain Q8_1 quantizer and reports the
mean of five sweeps per shape after two warmups. Representative M=512
times in ms were 0.984 → 0.982 for Q4_K 2560 → 9216, 0.978 → 1.022
for Q4_K 9216 → 2560, 0.916 → 0.932 for Q5_K 2560 → 8192, and
1.688 → 1.675 for Q6_K 2560 → 9216. These standalone calls do not
exercise SwiGLU. The raw scan logs are `native-sweep-control.log` and
`native-sweep-fused.log`.

Raw measurements, the profiling modifications and the isolated paired
prototype are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-05-JnUaw0/`
on AI.

## Review and decision

Accepted fused SwiGLU quantization for FFNs with three MMQ routes. Two
read-only review rounds were completed. The first requested the native
sweep. Both builds ran the sweep and the second round returned
`NO_BLOCKING_ISSUES`. `git diff --check` passed.
