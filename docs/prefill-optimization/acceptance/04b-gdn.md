# Iteration 4b prepared GDN

## Implementation

The [prefill plan](../../../engine/gpu/plan.go) binds the existing FFN gate
scratch for gfx1151 recurrence with 128 state rows, at least 128 tokens and
sufficient capacity. The [HIP descriptor](../../../backend/hip/bridge/yuuka_hip.h)
and [native validator](../../../backend/hip/bridge/bridge.cpp) use slot 13
for the prepared span and reject short, misaligned, overlapping and
conflicting bindings. Go and the native library require ABI 5.

[The prepared path](../../../kernels/hip/prefill/prefill.hip) computes the
FP32 Q/K scales once per token and key group and the decay and beta once per
token and value head. The four-lane core then runs the existing register
recurrence and reduction order with no block barriers in its token loop.
The prepared data for pp512 takes 8,519,680 bytes inside the existing
scratch allocation. Convolution history and matrix state remain
session-owned. The complete operation uses five launches per layer.

## Lane mapping trial

The control was the accepted iteration 4a build at `f9c5ade`. All runs used
the validated [AI environment and model](01-baseline.md#environment), F16
KV, chunk 512, one request and greedy generation. Unprofiled runs used
two warmups and five timed samples per process. The order was control,
four lanes, eight lanes, eight lanes, four lanes, control. The diagnostic
time is the median of three position-zero traces of all 24 recurrence
operations, through the gated RMSNorm.

| Mapping | Complete recurrence (ms) | Unprofiled pp512 medians (tok/s) |
| --- | ---: | ---: |
| Inline preparation, four lanes | 51.494 | 2077.50, 2070.16 |
| Prepared, four lanes | 21.328 | 2352.48, 2343.81 |
| Prepared, eight lanes | 20.355 | 2357.62, 2331.36 |

Both prepared routes met the 42 ms intermediate milestone and the 30 ms
complete recurrence budget. The four-lane route matched the control byte
for byte on full-vocabulary logits at pp128, pp512, two calls of 512
tokens and pp2048, including three subsequent greedy decode steps per
case. Convolution history and recurrent matrices matched exactly for
the first and last recurrent layers, including the two-call boundary.
The eight-lane route preserved the same top-1 tokens in this trial and
changed logits by up to 1.10760 and the checked matrices by up to
0.14030. It exceeds the 0.05 model comparison limit.

### Kernel attribution

A native probe replayed the model's 512-token recurrence geometry with
synthetic inputs. Each entry is the median of five dispatches after two
warmups, multiplied by 24 layers. These isolated times supplement the
real-model complete-operation traces above.

| Kernel (ms per 24 layers) | Inline four lanes | Prepared four lanes | Prepared eight lanes |
| --- | ---: | ---: | ---: |
| Convolution | 6.440 | 5.662 | 6.092 |
| Write convolution history | 0.041 | 0.032 | 0.031 |
| Prepare normalized inputs and gates | — | 1.749 | 1.623 |
| Recurrent core | 42.145 | 10.656 | 9.949 |
| Output gated RMSNorm | 1.582 | 1.881 | 1.832 |

The compiler reports 93, 88 and 57 core VGPRs respectively, and 31
for the preparer. All core variants and the preparer report zero
VGPR spills and zero private scratch. The prepared cores use no LDS.

## Final four-lane build

The Release build with the chosen mapping reproduced the exact state
and logit comparison. The median of three complete recurrence traces
was 21.572 ms at pp512. Unprofiled prompt rates were measured with equal
chunk sizes, the same verified model and F16 KV, one greedy output
token, one request and two warmups per process. The device used automatic
power mode and the sensor read 50–59 °C before the timed processes.

| Prompt | Chunk | Samples | Control tok/s | Prepared tok/s | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 5 | 1692.70 | 1863.42 | +10.09% |
| 512 | 512 | 5 | 2051.83 | 2312.37 | +12.70% |
| 2048 | 2048 | 5 | 2045.20 | 2288.33 | +11.89% |
| 8192 | 2048 | 3 | 1805.14 | 1988.19 | +10.14% |

The native suite passed 9/9. The two-chunk recurrence test used 57 and
71 tokens, a nonzero starting state, and two value heads per key group.
It checked both convolution and matrix state and all outputs against a
CPU reference. The gated output maximum absolute error was 0.000000179.
A second execution reproduced all outputs and states. The test also
checked the prepared scratch offset and the rejected bindings above.
An isolated diagnostic build omitted the final RMSNorm from the same
two-chunk test. Raw recurrence output versus the CPU reference had
maximum absolute error 0.0000000335 with four lanes and 0.0000000373
with eight lanes.

The following checks passed on AI with the verified model.

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1151`
  with the container's ROCm toolchain, `cmake --build build -j8`,
  and `ctest --test-dir build --output-on-failure -V`.
- `go build ./...`, `go build -tags hip ./cmd/yuuka`, `go vet ./...`,
  `go vet -tags hip ./...`, `go test ./...` and `go test -race ./...`.
- `YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -count=1 -v`
  with the temporary cross-build diagnostic run separately. The suite
  exercised chunked prefill, CPU/GPU logits, the separate decode path,
  the final FFN comparison and speculative greedy continuation. Its
  MoE golden test and two SentencePiece tests skipped for absent fixtures.

Raw artifacts, the binary state comparison and the profiler traces are
under `/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-04b-ouURUW/`
on AI.

## Review and decision

Accepted the prepared four-lane core. Pp512 meets the complete
recurrence budget and the lane ordering preserves model arithmetic for
all tested layouts. Both read-only reviews returned
`NO_BLOCKING_ISSUES`. The second round verified the prepared group
count guard and its regression case. Release CTest and the real-model
chunked prefill comparison passed again. `git diff --check` passed.
