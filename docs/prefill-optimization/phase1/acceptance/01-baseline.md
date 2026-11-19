# Iteration 1 baseline

## Scope

Reproduce the fixed-512 measurements and the long-context chunk sweep before
changing kernels. This iteration adds the [progress index](README.md), its
entry in the [implementation plan](../README.md), and this acceptance record.

## Environment

- Host `AI`, 2026-11-19, Ryzen AI Max+ 395 with Radeon 8060S, gfx1151 and
  Wave32. Linux `7.0.14-28-pve`, ROCm 10.1.0, HIP 7.16.26385, Go 1.24.4.
- Yuuka Release HIP build from `bfd36c0266ce1fd5d495e5e1d8e3bae3876f940e`,
  rebuilt from a clean export. The plan commits above that revision contain
  documentation only.
- Official llama.cpp tag `v1.0.0` at
  `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, existing ROCm Release build.
- Model `Qwen3.5-4B-Q4_K_M.gguf`, verified SHA-256
  `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
- F16 K/V, device offload, one request, automatic power profile. AMD SMI
  reported no active GPU process and the GPU temperature was 32 °C before
  the run. Yuuka used two warmups and greedy generation of one output token.
  llama-bench used its built-in warmup and `-n 0`.
- Raw logs on the GPU host under
  `/home/kurfuerst/yuuka-prefill-iterations-DmoXKP/`.

## Fixed chunk 512

Five timed samples per cell. The commands follow the [reproduction
protocol](../measurement.md#reproduce-and-compare). Yuuka uses the CLI prefill
p50 and llama-bench uses the median of `samples_ts`. Each standalone tool
generates its own input tokens.

| Prompt | Yuuka tok/s | llama.cpp tok/s | Yuuka / llama.cpp |
| ---: | ---: | ---: | ---: |
| 128 | 1509.74 | 1902.67 | 79.3% |
| 512 | 1779.76 | 2228.44 | 79.9% |
| 2048 | 1737.95 | 2214.65 | 78.5% |
| 8192 | 1510.55 | 1978.96 | 76.3% |

The pp512 medians correspond to 287.7 ms and 229.8 ms respectively. This
reproduces the roughly 80% throughput ratio that motivated the plan.

## 8K chunk sweep

Three timed samples per sweep cell. The reference's 512/512 entry is the
five-sample fixed-chunk control from the same session.

| Engine | Logical batch | Physical chunk | Median tok/s |
| --- | ---: | ---: | ---: |
| Yuuka | — | 512 | 1498.12 |
| Yuuka | — | 1024 | 1567.09 |
| Yuuka | — | 2048 | 1581.64 |
| Yuuka | — | 4096 | 1577.02 |
| llama.cpp | 512 | 512 | 1978.96 |
| llama.cpp | 8192 | 256 | 1966.23 |
| llama.cpp | 8192 | 512 | 1948.46 |
| llama.cpp | 8192 | 1024 | 1952.72 |
| llama.cpp | 8192 | 2048 | 1874.67 |

The best observed Yuuka chunk was 2048, about 5.6% ahead of the 512
control in the sweep. The small gap between 2048 and 4096 leaves both
worth retesting after the kernel changes. Temperature readings at
benchmark boundaries were 35–63 °C in automatic power mode.

## Current plan profile

`TestGPUProfilePrefill` passed three times with the real GGUF and F16
KV. The table gives medians in milliseconds for one 512-row plan.
The three positions are diagnostic samples. They do not represent
whole-prompt throughput or a replay of populated long-context KV.

| Absolute chunk position | GEMM | Recurrent chunk | Paged attention | Q8_1 quantization | LM head | All profiled operations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 217.739 | 50.491 | 5.060 | 3.226 | 2.969 | 294.361 |
| 2048 | 216.957 | 50.395 | 29.143 | 3.249 | 2.960 | 317.436 |
| 7680 | 216.218 | 50.329 | 120.714 | 3.308 | 2.952 | 407.447 |

The trace ran 611 plan operations per chunk, including 248 GEMM,
24 recurrent, 8 paged attention, 128 Q8_1 and 1 LM head operations.
The GEMM column aggregates the current routes by plan opcode. Later
MMQ comparisons will isolate the Q4_K gate/up shape.

## Verification

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1151`,
  `cmake --build build -j8`, and `go build -tags hip -o <run>/yuuka ./cmd/yuuka`
  passed on AI.
- `ctest --test-dir build --output-on-failure` passed all 9 native tests
  with zero skips.
- `go build ./...`, `go vet ./...`, `go test ./...` and
  `go test -race ./...` passed locally. The HTTP tests needed loopback
  access for the test listener.
- With `YUUKA_TEST_MODEL` set to the verified GGUF, the HIP-tagged
  `TestTinyGPUPrefillLogits`, `TestGPUChunkedPrefillLongContext`,
  `TestGPUMatchesCPU`, `TestGPUDecodePathMatchesCPU`, and
  `TestGPURealModelSpeculativeMatchesGreedy` all passed. The test commands
  are in [Measurement and acceptance](../measurement.md#reproduce-and-compare).

The final matched-input and matched-output comparison remains iteration 8.

## Review

One read-only review round. Final verdict: `NO_BLOCKING_ISSUES`.
