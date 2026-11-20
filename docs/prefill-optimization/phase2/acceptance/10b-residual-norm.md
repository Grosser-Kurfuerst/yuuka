# Stage 6b Residual add and prefill RMSNorm

The production control is `8762a72`. The [compact GEMV trial](10a-compact-tail-gemv.md)
left its five projections on GEMM. Tests ran in the AI container on
Radeon 8060S (`gfx1151`), ROCm 10.1.0, HIP 7.16.26385, and Go 1.24.4
with automatic clocks and power. Model and token SHA-256 values are in
the [reference protocol](05-cli-chunk-policy.md#baseline-and-protocol).

## Candidate and correctness

Two candidate operations consumed the pending residual add at its next
normalization site. The FP32 operation stored the rounded residual and the
normalized row. The Q8_1 operation also produced quantized digits and block
scales directly. It kept the FP32 normalization result for mixed consumer
groups, including the recurrent beta/alpha pair and the cropped final
query. The first embedding norm, final norm and final FFN residual remained
separate. The original 256-thread sum, epsilon and normalization
multiplication order were preserved, together with quantization rounding
and the per-block integer sum.

AI Release CTest passed 10/10. After synchronizing the two added test
sources, the rebuilt prefill and plan-validation targets passed 2/2.
The native differential test compared
residuals, FP32 norms, digits, and both fields of the scale pair bitwise
against the three separate operations. It covered FP32-only, Q8_1-only
and mixed outputs, 127, 128 and 2560 columns, a zero row and cancellation
and rounding inputs. The ABI test exercised aliasing, capacity, alignment
and optional-output validation.

A three-plan test used the same model and allocations to compare the
unfused, FP32 fusion, and quantized fusion schedules. The complete finite
prefill logit rows and three subsequent greedy logit rows were bitwise
identical at 128, 512, 529 (512 + 17), and 2048 tokens, as were the four-row
all-logit outputs. Independent executions with the checked-in compare
driver also matched the final logit bytes and all four greedy IDs at
16, 128, 512, 530 (512 + 18), and 2048 tokens.

## Launch attribution

At 512 rows, the 64 ADD operations and 64 batched RMSNorms each fall to
one. The fused quantized operation also absorbs 63 of 97 standalone
Q8_1 quantizers. Total executed operations fall from 557 to 431. A
position-zero diagnostic with inter-op HIP events measured 6.898 ms
for the three original operation families and 4.333 ms for the fused
and remaining operations. The 17-row tail replaces 63 ADD/norm pairs
with FP32-only fused operations and issues no Q8_1 quantization.
The diagnostic adds timing events and is separate from the wall-time gate.

## Unprofiled acceptance

Each independent process used F16 KV, chunk 512 (clamped at shorter
lengths), one request, two warmups and five measured repeats. Order 1
was control, candidate, candidate, control. Order 2 reversed the sequence.
Each table cell pools ten samples. Times are medians [full range] in ms.
The saving is the control median minus the candidate median.

| Tokens | Order | Control | Fused Q8_1 or FP32 | Saving |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 48.005 [47.920–48.156] | 47.932 [47.747–47.987] | 0.074 |
| 16 | 2 | 47.941 [47.833–49.134] | 47.831 [47.676–49.853] | 0.110 |
| 128 | 1 | 63.089 [62.750–64.088] | 61.933 [61.380–64.150] | 1.156 |
| 128 | 2 | 62.802 [62.417–63.091] | 62.261 [62.105–64.793] | 0.541 |
| 512 | 1 | 203.968 [202.511–208.257] | 200.203 [199.765–201.298] | 3.765 |
| 512 | 2 | 203.520 [202.191–209.560] | 202.797 [201.326–214.847] | 0.723 |
| 530 | 1 | 236.647 [235.845–244.644] | 236.851 [235.148–237.771] | -0.203 |
| 530 | 2 | 237.706 [237.116–239.366] | 235.554 [235.076–241.099] | 2.152 |
| 2048 | 1 | 837.324 [835.306–848.641] | 840.585 [831.165–849.209] | -3.261 |
| 2048 | 2 | 844.861 [835.007–866.360] | 839.288 [837.325–851.814] | 5.573 |

The three-plan control shared weights and scratch allocations, prebuilt
each plan, and alternated routes within the same process. After two
warmups, five samples per arm and sequence gave the following pp512
medians in ms.

| Sequence | Control | FP32 add/norm | Add/norm/Q8_1 | Full fusion saving |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 199.714 | 199.680 | 198.447 | 1.267 |
| 2 | 200.252 | 199.896 | 198.574 | 1.678 |

A 2560-column tuning trial cached each thread's rounded residual values
in registers and used one wave per Q8_1 block. The first pp128 model
comparison differed at logit 0 (control 8.845961, candidate 9.063799).
The numerical gate stopped that variant before timing.

## Decision and reproduction

Keep separate ADD, RMSNorm and Q8_1 quantization. Full fusion has a
positive pp512 signal, with two independent-process savings of 3.765 and
0.723 ms and two same-process savings below 2.0 ms. The planned gate
requires repeatable savings of at least 2.0 ms. The long-run medians also
vary in sign. The production plan used by Stage 7 therefore has the same
executable code as the 6b control.

The candidate changes span the HIP ABI, bridge and Go bindings,
prefill projections and residual scheduling, the norm and MMQ quantization
kernels, and the native plan and tensor tests. The read-only review found
`NO_BLOCKING_ISSUES` in one round for the bitwise-equivalent candidate.
The register-caching experiment stopped at the model comparison.

The AI experiment root is `/tmp/yuuka-phase2-5-11.3wZQZG/6b`. The
experiment is preserved in `candidate.patch`, with `candidate-sha256`
verifying the trial sources.
Use `src` with the Release HIP library in `build` for the first candidate
and `../4d/ship` for the production kernel control. Run
`ctest --test-dir build --output-on-failure`, then
`go build -tags hip ./...`, `go vet -tags hip ./...` and
`go test -tags hip ./... -count=1` from the source tree with
`YUUKA_TEST_MODEL` and the matching library paths. All passed for the
first candidate. The optionally supplied pre-1b control binary was absent,
so its historical short-path comparison skipped. Local non-HIP build,
vet, test and race test passed. See `cached.log`, `*-profile.log`,
`model-tests.log` and `tuned/cached.log` for the diagnostic runs.
