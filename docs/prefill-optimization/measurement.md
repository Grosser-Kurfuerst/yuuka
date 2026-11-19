# Measurement and acceptance

## Baseline used for this plan

The 2026-11-18 comparison used yuuka `bfd36c0` and upstream
[llama.cpp v1.0.0](https://github.com/ggml-org/llama.cpp/tree/b1b770d90d94cad9e9ea5b993bdbc383a522fb28).
The older tables in the [benchmark report](../benchmarks.md) use a
strix-llama.cpp fork. Reproduce the upstream tag when evaluating this plan.

Both engines loaded `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
on one Radeon 8060S (gfx1151, 40 CUs), Ryzen AI Max+ 395, Linux
7.0.14-28-pve, ROCm 10.1.0, HIP 7.16.26385 and Go 1.24.4. Both used a
Release HIP build, full GPU layer offload, F16 K/V, one request and automatic
power management. Yuuka requested one greedy output token. The standalone
throughput runs used the engines' respective benchmark token streams.

| Prompt tokens | yuuka tok/s | Upstream tok/s | Target tok/s at +5% | Current time → target |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1506.50 | 1865.68 | 1959.0 | 85.0 → 65.3 ms |
| 512 | 1765.00 | 2203.19 | 2313.3 | 290.1 → 221.3 ms |
| 2048 | 1726.59 | 2190.18 | 2299.7 | 1186 → 891 ms |
| 8192 | 1593.45 | 1991.68 | 2091.3 | 5141 → 3917 ms |

The first three rows used 512-token chunks on both sides, two yuuka warmups,
the built-in llama-bench warmup and five timed repetitions. The original 8K
row under that protocol was 1501.64 vs 1948.40 tok/s. The table uses the
best **observed** 8K configurations from a second session with three timed
repetitions. Yuuka chunk sizes 512/1024/2048/4096 gave
1504.42/1566.90/1593.45/1588.70 tok/s. Upstream gave 1991.68 at
`-b 512 -ub 512`. At `-b 8192`, microbatches 256/512/1024/2048 gave
1967.32/1969.14/1945.87/1870.65 tok/s.

## Attribution

Numbers are GPU milliseconds per **whole prompt**, median of three measured
traces, fixed chunk 512. Each cell is yuuka / upstream. Projection includes
the last FFN and, on the reference's Q6_K path, dequantization and BLAS
conversions. The recurrence group includes convolution, preparation and
gating. Both engines place their kernel boundaries differently.

| Family | 128 | 512 | 2048 | 8192 |
| --- | ---: | ---: | ---: | ---: |
| Weight projections | 69.4 / 53.8 | 227.2 / 164.5 | 926.9 / 707.3 | 3628.0 / 2798.8 |
| Complete recurrence | 12.5 / 6.7 | 53.9 / 35.1 | 220.2 / 132.6 | 874.6 / 588.2 |
| Attention with partial-result merge | 0.8 / 1.4 | 5.4 / 6.0 | 59.4 / 48.6 | 959.6 / 662.7 |
| All kernels | 88.8 / 68.9 | 306.6 / 230.6 | 1277.9 / 990.5 | 5733.7 / 4449.7 |

At pp512 yuuka already spends less time on activation quantization (2.8 vs
6.6 ms). At 8K its LM head costs 3.0 vs 35.9 ms. With this llama-bench
batch setting the reference trace contains one output projection per
512-token logical batch. Yuuka runs its LM head once per prompt. Keep that
distinction visible when evaluating final-layer or long-context changes.

The comparison host stored the raw runs under
`~/yuuka-bench-bfd36c0/`. The run root contains the unprofiled CLI logs
and reference JSON samples. The subdirectories `prefill-profile-mJNeAI/`
and `prefill-batch-sweep-AAgTHj/` contain the ROCm traces and the 8K sweep,
respectively.

## Reproduce and compare

1. Build both revisions in Release mode. Record the model hash, resolved
   commits, versions, GPU, kernel, power mode and temperatures alongside the
   raw output. Use the build instructions in the [benchmark report](../benchmarks.md#reproduction).
2. On the GPU host set `MODEL` to the verified GGUF, `YUUKA` to the HIP CLI
   built with `-tags hip`, and `LLAMA_BENCH` to the ROCm executable built
   from the upstream tag. Run the following from the repository root with
   the GPU available exclusively. The commands use a fresh output directory.

   ```bash
   out=$(mktemp -d)
   sha256sum "$MODEL"
   "$LLAMA_BENCH" --version
   for p in 128 512 2048 8192; do
     "$YUUKA" bench --backend hip --model "$MODEL" --kv-type f16 \
       --chunk-size 512 --prompt-tokens "$p" --decode-tokens 1 \
       --warmup 2 --reps 5 > "$out/yuuka-pp$p.txt" || exit 1
     "$LLAMA_BENCH" -m "$MODEL" -ngl 99 -ctk f16 -ctv f16 \
       -b 512 -ub 512 -p "$p" -n 0 -r 5 -o json > "$out/llama-pp$p.json" || exit 1
   done
   ```

3. Read yuuka's `prefill` p50 and take the median of llama-bench's
   `samples_ts`. Use `prompt_tokens / prefill_seconds` on both sides.
   Retain the fixed-chunk series for attribution. Repeat the 8K settings
   listed above to select each engine's best measured configuration. Repeat
   paired runs in reversed order if the spread approaches the claimed lead.
4. For kernel work run `TestGPUProfilePrefill` with `YUUKA_TEST_MODEL` and
   `YUUKA_TEST_KV=f16`. It measures one 512-row chunk at positions 0, 2048
   and 7680. Use `rocprofv3 --kernel-trace true -f rocpd` for native traces.
   Exclude warmup and model setup, retain casts and merge kernels, and report
   the positions and row counts. The Go whole-prompt traces above used a
   diagnostic launcher with a libc exit hook to flush the profiler. Use the
   normal CLI for throughput. [Existing profile test](../../engine/gpu/prefill_profile_test.go#L25)
5. Before declaring a win, add a benchmark-only comparison driver that feeds
   the same saved token IDs and positions to the two independent runtimes,
   requests logits only on the last prompt row, and synchronizes through
   availability of that row on the host. Time the interval from submitting
   IDs to receiving logits, with weights and plans warm and session state
   reset between samples. Keep tokenization, model loading and sampling
   outside that interval. Report this series alongside the standalone
   tools. The [iteration 8 record](acceptance/08-upstream-comparison.md)
   contains the driver build commands, inputs and output checks. Link the
   reference executable only into its own benchmark process.

For each stage run its targeted tests and the real-model greedy/long-context
checks. For integration use the [project test commands](../testing.md#94-current-test-commands),
including Go build, vet, tests, race tests, native CTest and the HIP-tagged
suite. Check for model or device skips. Record accepted dispatch changes in
[kernels.md](../kernels.md) and measured results in [benchmarks.md](../benchmarks.md).
Use a fresh paired upstream median for the final 1.05× gate at each length.

For a first implementation slice, the following commands exercise the
native operations, speculative output rows, real-model logits, decode
continuation and the position-dependent profile on the GPU host.

```bash
ctest --test-dir build -R 'hip_(prefill_correctness|paged_graph_correctness|plan_validation)' --output-on-failure
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu -count=1 -v \
  -run '^(TestTinyGPUPrefillLogits|TestGPUChunkedPrefillLongContext|TestGPUMatchesCPU|TestGPUDecodePathMatchesCPU|TestGPURealModelSpeculativeMatchesGreedy)$'
YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -tags hip ./engine/gpu \
  -run '^TestGPUProfilePrefill$' -count=3 -v
```
