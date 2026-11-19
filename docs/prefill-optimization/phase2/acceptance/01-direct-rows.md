# Stage 1a Direct MMQ rows

## Implementation

The [Stage 1a plan](../stage1a-direct-rows.md) extends the existing Q4_K and
Q5_K Direct MMQ shape table to any token count of at least 128 on gfx1151.
The native launcher retains the original grid and the exact-size Q4_K
FFN specialization. The masked variant returns before any block barrier
when its 64-token half is empty, zeros invalid activation digits and scales,
and suppresses invalid stores. The launcher uses the original mask-free
Direct variants when the batch row count is a multiple of 128.
[Kernel and dispatch contract](../../../kernels.md#gfx1151-direct-mmq-specialization)

The [native test](../../../../tests/hip/prefill_test.cpp) checks the
Q8_1 encoding and the CPU matrix product for Q4_K and Q5_K with partial
rows. It also compares 129, 191 and 255 rows against the corresponding
prefix of a 256-row run and checks every output sentinel after the prefix.
The [native sweep](../../../../bench/hip/prefill_sweep.cpp) exposes
`--direct-mmq` for all eight measured shapes at eight row counts.

## Environment

Both gfx1151 Release builds ran in the AI container with ROCm 10.1.0,
HIP 7.16.26385, Go 1.24.4 and Linux `7.0.14-28-pve`. The control was
`846fc4f`. Runs used F16 KV, one request, automatic GPU power mode and
one GPU process at a time. The comparison's starting sensor readings
were 40–66 °C. The GGUF and fixed token file matched the SHA-256 values in
the [Phase 1 comparison](../../phase1/acceptance/08-upstream-comparison.md#runtime-and-input).

## Native correctness and compiler resources

All 10 CTest suites passed. The new partial tiles produced maximum
absolute CPU product errors of `9.16e-5` for Q4_K and `2.98e-4` for
Q5_K at the existing tolerance. Every tested valid-row prefix matched
bit for bit and all sentinels survived. The plan's nine-suite count
predates the current CTest registration.

`scripts/kernel_resources.sh` reported 113 VGPRs for each masked variant,
125 for each regular Direct variant and 119 for the fixed FFN. Each has
zero VGPR spills and zero private scratch. Static LDS is zero and the
launcher reserves 30 KiB of dynamic LDS per workgroup.

## Isolated MMQ performance

The first unprofiled sweep timed each submit-and-wait sequence. One of the
40 newly enabled cells was 2.25% slower in that series. Four interleaved
`rocprofv3` processes then timed the same multiplies on the GPU, excluding
quantization and host waits. All 40 new cells improved by 15.82–45.33%.
The disputed Q4_K cell at `2560→1024, M=192` took 0.08816 ms in the control
and 0.06738 ms in the candidate. The candidate trace confirmed Direct with
masking at M=129, 192 and 511, and mask-free Direct at M=1024 and 4096.

`OccupancyPercent` at M=512 was 46.85% → 47.06% for the fixed Q4_K FFN
and 47.01% → 47.20% for the wide Q5_K projection. Narrow and partial tiles
have fewer active workgroups and are evaluated by duration. Extracting
the gfx1151 code objects confirmed byte-for-byte equality of the GPU
instructions of all six pre-existing MMQ variants. The extra mask flag
compiled two separate device functions for the new partial row counts.

The process-level traces at M=128, 512 and 2048 exceeded the ±1% band
in some cells, including with 16 warmups and 101 samples per process.
A second probe submitted both builds through the same stream using
the same buffers. It linked against the control and loaded the candidate
with local, deep symbol binding. The trace confirmed alternating
control/candidate and candidate/control dispatches by their distinct
template symbols. Each of the 24 cells used 16 warmup pairs and 96 timed
pairs. The difference between the two median durations was −0.56% to
+0.36%. Medians of the per-pair ratios varied from −0.51% to +0.30%.

## Matched-input model comparison

The [prefill comparison driver](../../../../bench/prefill-compare/main.go)
used the Phase 1 token stream with positions starting at zero and timed
through the final host logit row. Each cell ran in control, candidate,
candidate, control order. Each process used two warmups and seven timed
samples for prompts below 2048 tokens and one warmup and three samples for
longer prompts. The table gives the median across both processes per
version. Chunk is the configured limit.

| Prompt | Chunk | Control (ms) | Candidate (ms) | Latency change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 68.270 | 68.129 | −0.21% |
| 129 | 512 | 113.107 | 86.931 | −23.14% |
| 140 | 512 | 114.668 | 88.220 | −23.06% |
| 192 | 512 | 124.567 | 94.871 | −23.84% |
| 256 | 512 | 135.678 | 115.759 | −14.68% |
| 511 | 512 | 256.411 | 217.648 | −15.12% |
| 512 | 512 | 218.608 | 218.745 | +0.06% |
| 513 | 2048 | 291.033 | 242.445 | −16.70% |
| 530 | 512 | 279.173 | 279.481 | +0.11% |
| 1024 | 512 | 439.274 | 438.116 | −0.26% |
| 1024 | 1024 | 493.434 | 423.575 | −14.16% |
| 2048 | 512 | 893.676 | 898.600 | +0.55% |
| 2048 | 1024 | 1000.997 | 859.313 | −14.15% |
| 2048 | 2048 | 872.599 | 873.021 | +0.05% |
| 8192 | 512 | 4137.669 | 4127.052 | −0.26% |
| 8192 | 1024 | 4517.265 | 3906.133 | −13.53% |
| 8192 | 2048 | 3897.016 | 3915.704 | +0.48% |
| 8192 | 4096 | 4514.581 | 3957.490 | −12.34% |

The candidate's chunk 1024 was faster than its chunk 512 by 3.32%, 4.37%
and 5.35% at 1024, 2048 and 8192 prompt tokens respectively. The 18-row
tail at 530 tokens served as the small-tail control. All 18 cells returned
identical complete FP32 logit rows across both builds and both processes.

## Server first-content latency

Both Release HIP servers listened on loopback with F16 KV, context 1024,
chunk 512, concurrency one and prefix caching disabled. Each request had
one user message containing 12, 28 or 51 repetitions of
`The quick brown fox jumps over the lazy dog. ` followed by
`Summarize this in one word.`. The request used streaming, temperature
zero, disabled thinking and an eight-token output limit. The server
reported the rendered token counts below.

The client timed from HTTP POST to the first SSE event with a nonempty
`delta.content`. Each process used two warmups and seven timed requests per
prompt. Results are pooled medians across the same interleaved process
order as the prompt benchmark.

| Rendered prompt tokens | Control (ms) | Candidate (ms) | Latency change (ms) |
| ---: | ---: | ---: | ---: |
| 140 | 125.826 | 97.752 | −28.074 |
| 300 | 197.128 | 157.792 | −39.336 |
| 530 | 287.907 | 288.668 | +0.761 |

## Reproduction

Build both trees in Release mode with `-DCMAKE_HIP_ARCHITECTURES=gfx1151`.
Use the Stage 1a sweep source with each HIP library so that both binaries
test the same cells. Run `build/yuuka_prefill_sweep --direct-mmq` for
unprofiled MMQ timings and repeat under `rocprofv3 --kernel-trace true -f
rocpd` for dispatch and device timings. Each cell quantizes once before
two warmups and nine timed multiplies. Select only `gemm_mmq_kernel` and
discard the first two dispatches per cell. The default sweep still uses
five timed calls. Its [counter reducer](../../../../scripts/prefill_counters.py)
expects the default sweep and counts launched tiles even when a masked
kernel returns before issuing MMAs.

Build [prefill-compare](../../phase1/acceptance/08-upstream-comparison.md#reproduce-the-matched-test)
and `./cmd/yuuka-server` with `-tags hip` against each library. Pass
`-model`, `-tokens`, `-prompt`, `-chunk`, `-warmup`, `-reps` and
`-logits` to the comparison binary. Use `cmp` on each paired logit
file and compare the unprofiled `samples_ns` medians.

## Verification

- The gfx1151 Release build and `ctest --test-dir build --output-on-failure`
  passed all ten registered suites.
- `go build ./...`, `go vet ./...`, `go vet -tags hip ./...`,
  `go test ./...`, and `go test -race ./...` passed on AI with the Qwen model.
- `YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -v -count=1`
  passed. One MoE and two SentencePiece tests skipped because their
  separate fixtures were absent.

## Review and decision

Accepted. The two independent read-only reviews returned
`NO_BLOCKING_ISSUES`. The second review independently reduced the 18
prompt cells and the 5,376 dispatches from the same-stream probe. A final
wording change clarified that mask-free selection applies to the entire
batch. The checked-in source matched the tested source byte for byte.
Relative link checks and `git diff --check` passed.
