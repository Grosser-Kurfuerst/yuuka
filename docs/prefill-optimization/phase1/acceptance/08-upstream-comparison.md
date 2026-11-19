# Iteration 8 upstream prefill comparison

## Runtime and input

The two engines ran on the [AI configuration](01-baseline.md#environment).
Yuuka is the Release gfx1151 build of `8ce3a62` with C ABI 8, built
with ROCm 10.1.0, HIP 7.16.26385 and Go 1.24.4. The official
[llama.cpp v1.0.0](https://github.com/ggml-org/llama.cpp/tree/b1b770d90d94cad9e9ea5b993bdbc383a522fb28)
tag and its checkout both resolved to
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`. The Release reference build
has HIP graphs enabled, targets gfx1151, and reported all 33 layers
offloaded to ROCm0.

Both loaded `Qwen3.5-4B-Q4_K_M.gguf`, SHA-256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
The token file was a single 8192-token stream derived from the fixed
repeated prompt. Its SHA-256 was
`e613216db780a72fdbe796b31c63b4afabf273199dbc67f799b5c082d18e3e30`.
Each run consumed its corresponding prefix of this file with positions
starting at zero, F16 K/V, one sequence and capacity requested as
the prompt length plus 64 tokens. The reference uses automatic Flash
Attention selection.

## Reproduce the matched test

Build the [Yuuka driver](../../../../bench/prefill-compare/main.go) from the
project root and build the [reference driver](../../../../bench/prefill-compare/llama/driver.cpp)
against the verified official checkout. Set `YUUKA_BUILD` and `LLAMA`
to the corresponding Release build directory and source checkout,
`MODEL` to the GGUF, and `out` to a directory for logs.

```bash
export LD_LIBRARY_PATH="$YUUKA_BUILD:$LLAMA/build/bin:${LD_LIBRARY_PATH:-}"
CGO_LDFLAGS="-L$YUUKA_BUILD" go build -tags hip \
  -o "$out/yuuka-compare" ./bench/prefill-compare
c++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  bench/prefill-compare/llama/driver.cpp -L"$LLAMA/build/bin" \
  -Wl,-rpath,"$LLAMA/build/bin" -lllama -lggml -lggml-base \
  -o "$out/llama-compare"
"$out/yuuka-compare" -model "$MODEL" -tokens "$out/tokens.i32" \
  -prompt 8192 -write-tokens

while read -r series n yuuka_chunk llama_chunk; do
  for run in yuuka-1 llama-1 llama-2 yuuka-2; do
    result="$out/$series-$n-$run"
    case "$run" in
      yuuka-*)
        "$out/yuuka-compare" -model "$MODEL" -tokens "$out/tokens.i32" \
          -prompt "$n" -chunk "$yuuka_chunk" -warmup 2 -reps 5 \
          -logits "$result.f32" > "$result.json" ;;
      llama-*)
        "$out/llama-compare" "$MODEL" "$out/tokens.i32" "$n" "$llama_chunk" \
          2 5 "$result.f32" > "$result.json" ;;
    esac
  done
done <<'SETTINGS'
fixed 128 512 512
fixed 512 512 512
fixed 2048 512 512
fixed 8192 512 512
tuned 128 128 128
tuned 512 512 512
tuned 2048 2048 512
tuned 8192 2048 256
SETTINGS
```

Both drivers reset session state before each repetition, prepare host
inputs outside the timer, and time from prompt submission through the
availability of the final host logit row. The earlier chunks request
zero output rows. The last chunk requests one. Both drivers write the
last logits outside the timed interval and report individual durations
in nanoseconds. Model loading, tokenization and sampling are excluded.
The reference executable is independent of the Yuuka runtime.

## Equal chunk settings

Each length ran in Yuuka, llama.cpp, llama.cpp, Yuuka order. Each
process used two warmups and five timed samples. The rates below
divide prompt length by the pooled median duration of ten samples
per engine. The maximum chunk was 512 for both engines, with
128 actual rows for the shortest input.

| Prompt tokens | Yuuka tok/s | llama.cpp tok/s | Change | +5% gate |
| ---: | ---: | ---: | ---: | --- |
| 128 | 1889.17 | 1891.49 | -0.12% | Miss |
| 512 | 2387.30 | 2214.98 | +7.78% | Pass |
| 2048 | 2303.03 | 2201.86 | +4.59% | Miss |
| 8192 | 2000.92 | 1944.71 | +2.89% | Miss |

## Measured chunk settings

The tuning run checked short chunks of 64 and 96 for pp128 and
reference chunks of 128 and 256 at pp512. At pp2048 and pp8192 it
checked Yuuka chunk 2048 and reference chunks 256, 1024 and 2048
against the equal-chunk baseline. The fastest observed settings
were replayed in a fresh interleaved comparison with the same
two warmups and five timed samples per process. The raw log gives
reference batch and microbatch settings equal to each other.

| Prompt | Chunks Yuuka / llama.cpp | Yuuka ms | llama.cpp ms | Yuuka tok/s | llama.cpp tok/s | Change | +5% gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 128 / 128 | 68.586 | 69.001 | 1866.28 | 1855.04 | +0.61% | Miss |
| 512 | 512 / 512 | 218.868 | 233.325 | 2339.31 | 2194.36 | +6.61% | Pass |
| 2048 | 2048 / 512 | 860.945 | 938.733 | 2378.78 | 2181.66 | +9.04% | Pass |
| 8192 | 2048 / 256 | 3921.183 | 4183.338 | 2089.17 | 1958.24 | +6.69% | Pass |

The process ordering and median reduction match the equal-chunk test.
The device used automatic power mode and starting readings of 42–66 °C
across the matched comparison. The pp128 target would require about
65.715 ms under the measured reference, a 2.871 ms reduction from
this Yuuka build.

Three diagnostic pp128 traces on the final build put 57.531 ms in
248 GEMM operations, 5.010 ms in the 24 recurrent operations,
2.870 ms in the LM head and 0.737 ms in all eight attention
operations. GEMM accounts for about 82% of the traced time. The
remaining 2.871 ms is about 5% of the traced GEMM budget.

## Output checks

Each engine returned 248,320 finite FP32 logits. Its output repeated
bit for bit across the two processes per cell. Both engines agreed on
top-1 for all lengths under both chunk settings. The tuned runs had
the following differences between engines.

| Prompt | Both top-1 token IDs | Max absolute logit difference | Mean absolute logit difference |
| ---: | ---: | ---: | ---: |
| 128 | 5388 | 1.07730 | 0.21194 |
| 512 | 13477 | 0.50055 | 0.11221 |
| 2048 | 5388 | 0.34167 | 0.06766 |
| 8192 | 13477 | 0.43106 | 0.04976 |

Both drivers rejected the same malformed token file. Regenerating
the valid file with the final Yuuka driver gave the same bytes.

## Standalone tools

The standalone Yuuka CLI and upstream `llama-bench` ran at the same
measured chunk settings. Yuuka used two warmups and five timed samples
with one greedy output. The reference used its built-in warmup and
`-n 0 -r 5`. Each standalone tool supplies its own prompt tokens.
The matched-input table governs the 5% gate.

| Prompt | Yuuka CLI tok/s | llama-bench tok/s |
| ---: | ---: | ---: |
| 128 | 1833.22 | 1822.18 |
| 512 | 2312.91 | 2161.68 |
| 2048 | 2329.17 | 2142.25 |
| 8192 | 2104.72 | 1923.42 |

## Verification

Both drivers built successfully on AI. The reference driver used
`-Wall -Wextra -Werror`. Release `ctest --test-dir build --output-on-failure -V`
passed 9/9. `go build ./...`, `go vet ./...`, `go vet -tags hip ./...`,
`go test ./...`, `go test -race ./...` and
`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip ./... -count=1 -v`
passed. The HIP suite included the split attention and full-row
oracles, CPU/GPU logits, separate decode, speculative greedy and
all-logits tests. One MoE test and two SentencePiece tests skipped
because their fixtures were absent.

Raw process samples, logits, environment and build logs are under
`/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/iteration-08-Bc1X8h/`
on AI.

## Review and decision

The matched benchmark is verified at all four lengths. The tuned
settings clear the 5% margin at 512, 2048 and 8192 tokens. The 128-token
case is below the margin. One read-only review returned
`NO_BLOCKING_ISSUES`. Relative documentation links and
`git diff --check` passed.
