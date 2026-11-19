# Benchmark Report: Qwen3.5-4B Q4_K_M

This file is the current performance record for the repository: every table
below was measured on the code in the tree at the commit listed under
[Environment and protocol](#environment-and-protocol), with the stated
protocol. Superseded phase-by-phase reports are kept in git history and are not
repeated here. Numbers are observations on one machine, not guarantees; device
clock, power, and thermal state are not pinned.

## Environment and protocol

| Field | Value |
| --- | --- |
| Date | 2026-11-18 |
| Model | `Qwen3.5-4B-Q4_K_M.gguf` |
| Model SHA-256 | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Model size | 2,740,937,888 bytes |
| Architecture | `qwen35`, 32 layers (8 full attention, 24 gated delta-net) |
| Quantization | Q4_K_M (the GGUF carries mixed per-tensor dtypes; no requantization is applied) |
| Device | AMD Radeon 8060S Graphics, `gfx1151`, 40 CUs, 122,880 MiB reported VRAM |
| ROCm / HIP | ROCm 10.1 layout, HIP 7.16.26385 |
| Host | AMD Ryzen AI MAX+ 395 w/ Radeon 8060S, 32 threads, 120 GiB shared memory |
| Kernel | 7.0.14-28-pve |
| Power performance level | `auto` (amdgpu sysfs) |
| Go | go1.24.4, Release `cmake --build build` |
| Commit | `d56ad88` |
| Tokenizer / chat template | GGUF-embedded Qwen3.5 vocabulary and template |
| Sampling | Greedy, temperature 0 |
| Prefill chunk | 512 tokens (`--chunk-size 512`), except where a row states otherwise |
| Warmup / repetitions | 2 / 5 for prefill rows, 2 / 3 for decode and serving rows, median reported |

Tables re-measured at the commit above are the prompt-length prefill rows, the
correctness gates, the prefill LM head, and the kernel resource usage. The decode
matvec table was measured at `bfd36c0`, which differs from the commit above only
in documentation. The
decode, KV-format, attention, isolated GEMM, batched GEMV, MoE, gated delta-net,
serving, server, and llama.cpp comparison tables were measured at `161fbba` and
were not repeated: the LM head change is confined to the prefill tail and does
not touch the decode dispatch. `scripts/prefill_counters.sh` was rerun and still
reproduces the recorded MMQ GEMM instruction counters.

Protocol:

- `build/yuuka_prefill_sweep` measures isolated kernels (GEMM, GEMV) per dtype,
  shape, and row count; each cell is the median of five timed submissions after
  two warmups, and the K-quant `mmq` cell includes the Q8_1 activation pass.
- `yuuka bench` measures end to end through the engine. TTFT is measured from
  submission; decode throughput is `(decode_tokens - 1) / decode p50`, matching
  the CLI report.
- `bench/serving` measures the scheduler path (queue wait, TTFT, prefill,
  decode, sampling) and reports aggregate throughput over its wall interval.
- The server comparison drives the OpenAI-compatible HTTP surface of
  `yuuka-server` and `llama-server` with a stock streaming client on the same
  host, so TTFT includes HTTP and SSE overhead. Prompt caching is disabled on
  both sides (`--prefix-cache=false`, `"cache_prompt": false`).
- `scripts/prefill_counters.sh` reproduces the per-MMA prefill counters
  (instruction mix, bank conflicts, occupancy) recorded in
  [kernels.md](kernels.md#63-int8-mmq-gemm). It profiles the
  native sweep binary only: `rocprofv3` cannot write its database for a Go
  process, as recorded in [testing.md](testing.md#63-what-to-measure-separately).
- `scripts/decode_profile.sh` reproduces the [decode matvec](#decode-matvec)
  table in one command: it runs `build/yuuka_gemv_decode_sweep` unprofiled and
  under `rocprofv3`, runs its `--threshold` and `--int8` modes for the output
  projection crossover and the [int8 comparison](#int8-decode-matvec), then
  runs `TestGPUProfileDecode` for the per-operation attribution of one real
  decode step. The sweep keeps every matrix resident, so its GB/s figures are the
  hot-matrix upper bound of the real step.
- Only one benchmark process may use the GPU at a time; two processes sharing
  the device invalidate both results.
- The llama.cpp reference is `strix-llama.cpp` commit `c8912d7f5` (build 12216,
  version 0.6.0-dev), ROCm backend, `-ngl 99 -b 512 -ub 512`. `llama-bench`
  rejects an `f32` KV cache type, so its rows use llama.cpp's default `f16` KV;
  yuuka rows state their KV format.

## Correctness gates

Current tree, same commit:

| Check | Result |
| --- | --- |
| `ctest --test-dir build` (native ABI, quantized GEMV, operators, GEMM, prefill, paged KV/graph, embedding, MoE) | 9 / 9 pass |
| `TestGPUChunkedPrefillLongContext`, 1024 real-model tokens, chunks 128/512/1024 | top-1 `33075` in all three; maximum logit difference 0 against the chunk-128 reference |
| `TestGPUMatchesCPU` | first-step maximum logit difference `0.00004673` with [short-chunk FP32 streaming](prefill-optimization/phase2/acceptance/04a-streaming-short-tails.md), greedy tokens match |
| `tests/hip/prefill_test.cpp` MMQ checks | Q8_1 digits identical to the host quantizer; GEMM maximum absolute difference `6.9e-5` (Q4_K) and `1.5e-4` (Q5_K) against the CPU product |
| `tests/hip/prefill_test.cpp` LM head checks | every dtype within `1.3e-4` of the host dot products for 1/3/16 rows, bitwise reproducible |
| `TestPlanLMHeadMatchesDequant` | Q8_0 bitwise equal to the dequantized CPU dot; Q6_K within `2e-6` |
| `yuuka run` at 101 / 501 / 2001 prompt tokens | generated tokens identical before and after the LM head change |
| `go build ./...`, `go vet ./...`, `go test ./...`, `go test -tags hip ./...` | pass |

The Q8_1 prefill path moves logits against an FP16-activation reference
(measured earlier at 0.0031 maximum for the tiny prompt and 0.264 for a
2048-token prompt, top-1 preserved). Chunk sizes stay inside the 0.05 chunk
gate; the quantization grid itself is the source of that delta, and it is the
regime llama.cpp's K-quant prefill also runs in.

## Prefill: prompt length

`yuuka bench --backend hip --decode-tokens 1 --chunk-size 512`, F32 KV unless
stated, median of five repetitions:

| Prompt tokens | F32 KV p50 | F32 tok/s | F16 KV p50 | F16 tok/s | Q8_0 KV p50 | Q8_0 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.085 s | 1511.8 | 0.095 s | 1344.4 | 0.096 s | 1326.9 |
| 512 | 0.292 s | 1756.2 | 0.313 s | 1634.8 | 0.330 s | 1553.6 |
| 2048 | 1.201 s | 1705.6 | 1.248 s | 1641.5 | 1.524 s | 1343.7 |
| 8192 | 5.697 s | 1437.9 | 5.696 s | 1438.2 | 9.480 s | 864.1 |

Measured 2026-11-18 after the streaming prefill LM head
([kernels.md](kernels.md#65-streaming-lm-head)). The F32 column is a same-session
comparison: back-to-back runs at `aec0570` with the same warmup and repetitions
gave 1345.1 / 1626.2 / 1673.9 / 1409.2 tok/s at the four lengths, so the stage is
+12.4% / +8.0% / +1.9% / +2.0%. The gain follows the 8.4 ms the tail saves per
chunk, which is a larger share of the shorter prompts. F32 and F16 KV are within
4% of each other at every length, because both run the same attention kernel;
Q8_0 keeps the tiled kernel and diverges from 2048 tokens, where attention
starts to dominate. The F16 and Q8_0 rows are the pre-LM-head measurements and
were not repeated.

Before this stage the same table read 1380.1 / 1704.8 / 1715.2 / 1441.7 tok/s for
F32 KV at `161fbba`. That earlier row was measured after the FP16 MMA prefill
attention kernel ([kernels.md](kernels.md#75-fp16-mma-paged-prefill-attention)),
which had moved the same F32 KV lengths from 1609.1 / 1415.4 / 877.7 to
1704.8 / 1715.2 / 1441.7 at 512 / 2048 / 8192 tokens. Its values differ from the
current ones by 0.5-1.6% at the longer prompts, which is the run-to-run noise of
an unpinned clock on this device, so the same-session A/B above is the number to
compare the LM head stage against.

The [Q4_K FFN specialization acceptance](prefill-optimization/phase1/acceptance/02-q4-ffn.md)
records the 2026-11-19 gfx1151 prefill comparison using F16 KV,
including isolated MMQ, complete FFN and prompt-length medians.
The [final FFN acceptance](prefill-optimization/phase1/acceptance/03-final-ffn.md)
records the next F16 KV comparison, with one FFN row in the ordinary
prefill output tail.
The [MMQ shape expansion](prefill-optimization/phase1/acceptance/04a-mmq-shapes.md)
records projection shape medians and the short- and long-prompt A/B.
The [prepared GDN comparison](prefill-optimization/phase1/acceptance/04b-gdn.md)
records the recurrence timings and the model-state comparison by lane mapping.
The [FFN fusion comparison](prefill-optimization/phase1/acceptance/05-ffn-fusion.md)
records the isolated quantizer check, complete FFN time and prompt rates.
The [final query crop comparison](prefill-optimization/phase1/acceptance/06-final-query.md)
records K/V equivalence, output work per chunk and the resulting prompt rates.
The [long attention comparison](prefill-optimization/phase1/acceptance/07-long-attention.md)
records split-key numerics, merge costs, scratch and the 8K chunk sweep.
The [official v1.0.0 matched-input comparison](prefill-optimization/phase1/acceptance/08-upstream-comparison.md)
records host logit latency and the per-length gate. At measured chunk settings
the prefill leads were +0.61%, +6.61%, +9.04% and +6.69% for 128, 512,
2048 and 8192 tokens respectively.

## Decode: output length

`yuuka bench --backend hip --prompt-tokens 128 --decode-tokens 128|256`,
F32 KV, median of three repetitions:

| Decode tokens | Context | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 320 | 0.085 s | 0.085 s | 3.280 s | 38.70 |
| 256 | 448 | 0.162 s | 0.162 s | 7.429 s | 34.32 |

Decode is weight-bandwidth bound and flat in output length: the two rows differ by
11%, which is the prefill of the longer prompt plus the drift of an unpinned
clock, not a change of regime. Effective weight-read bandwidth is
$2.726\\,\\mathrm{GB} \\times 38.7\\,\\mathrm{s^{-1}} \\approx 105.5\\,\\mathrm{GB/s}$
against the 130 GB/s the profiled step measures for the family, so the family
itself is still the limiter ([isolated kernels](#isolated-kernels)). The 3 ms per
token between the two figures is the host side of the loop: the sampler reads
248,320 logits and the row is copied device-to-host once per token.

The decode rows measure the selected route, which is the FP32 streaming matvec
(the int8 route is measured slower and stays opt-in; see
[int8 decode matvec](#int8-decode-matvec)).

The table was re-measured four times: when the decode output projection moved to
the streaming LM head kernel, when the rest of the decode matvec family followed
it, when the gated delta-net moved to the fused per-token pair, and when the
residual add and the gate/up pair were folded into the matvec. Same flags, same
session, 128 decode tokens: 13.241 s, then 12.079 s, then 4.245 s, 3.341 s, then
3.280 s. The profiled step went 97.4, 89.4, 31.6, 24.3, 23.4 ms with the matvec
family at 87.6, 79.2, 21.6, 21.5, 21.1 ms, so the matvec share rose from 68% to
90% as everything else left the critical path. Prefill is unaffected across all
five: 0.085-0.088 s.

### Decode fusion

`TestGPUProfileDecode` with and without `YUUKA_TEST_DECODE_UNFUSED=1`, same
session, same plan input, one decode step after a real forward pass:

| Plan | Operations | Matvec | Everything else | Step |
| --- | ---: | ---: | ---: | ---: |
| fusions off | 507 | 21.452 ms | 3.01 ms | 24.461 ms |
| fusions on | 379 | 21.104 ms | 2.39 ms | **23.492 ms** |

The residual epilogue replaces 64 ADD operations, and the gate/up pair replaces
64 operations (two matvecs and a SILU_MUL) with 32. Both plans produce
bit-identical logits over eight greedy steps
(`TestGPUDecodeFusionsMatchUnfused`, maximum difference 0).

`TestGPUGraphReplayMatchesAndTimes`, 32 decode steps: 22.398 ms per step on the
plain submission path against 22.385 ms with the plan captured into a HIP Graph,
a ratio of 0.999 with identical tokens. Replay removes every per-operation host
submission, so the inter-operation gap is below the resolution of that comparison,
which is what the matvec fusion targeted.

Variants measured and rejected for the same kernel:

| Variant | Step | Matvec | Verdict |
| --- | ---: | ---: | --- |
| fused delta-net tree (before the fusion) | 24.26 ms | 21.45 ms | - |
| eight output channels per wave | 25.9 ms | 21.8 ms | 28% slower, spills |
| `kStepK` 512 | 24.2 ms | 21.8 ms | 7% slower, staging outweighs barriers |
| pair with a run-time dtype switch | 26.2 ms | 23.9 ms | 40% slower than the templated pair |

### Gated delta-net

`build/yuuka_deltanet_sweep`: one recurrent layer at the decode geometry (32
heads, 16 groups, 128-wide state), each candidate kernel timed with HIP events
around a single launch, five warmups and twenty-five submissions. The state is
resident, so this is the hot upper bound; the in-step figures below are lower. The
five-pass form reads the state three times and writes it twice, the fused forms
once each, which is the traffic the GB/s column uses.

| Form | ms/layer | State GB/s | x five-pass |
| --- | ---: | ---: | ---: |
| five-pass kernel (legacy) | 0.3023 | 34.7 | 1.00 |
| copy floor (read and write once) | 0.0177 | 236.8 | 17.1 |
| register, 128 threads/head | 0.0133 | 316.2 | 22.8 |
| register, 256 threads/head | 0.0117 | 357.2 | 25.8 |
| register, 512 threads/head (selected) | 0.0112 | 375.1 | 27.0 |
| register, 512 threads/head, no gated RMSNorm | 0.0108 | 387.6 | 27.9 |
| shared memory, 128 threads/head | 0.0334 | 125.5 | 9.0 |

The register form wins because the 64 KB state cannot share the workgroup budget
with the normalization scratch the shared form would need, and because 512
threads give the memory pipeline more independent loads per lane than the copy
kernel itself: the selected form is faster than a read-and-write copy of the same
bytes, which puts the limiter at memory-level parallelism rather than bandwidth.

The same layer inside a decode step (`TestGPUProfileDecode`), per token:

| Operation | Ops | ms | Share |
| --- | ---: | ---: | ---: |
| `recurrent_token` | 24 | 0.467 | 1.9% |
| `conv_l2norm` | 24 | 0.328 | 1.3% |
| delta-net unit | 48 | 0.790 | 3.3% |
| the five operators it replaced | 120 | 8.177 | 25.9% |

The recurrent state alone moves 100.7 MB per token at 217 GB/s inside the step,
against 34.7 GB/s for the five-pass form. The operation count of the whole decode
step fell from 579 to 507.

`build/yuuka_gemv_decode_sweep --recurrent` is the reproducible RECURRENT cell: it
builds the same layer from both plans and reports 0.3371 ms per layer for the five
operators against 0.0168 ms for the fused pair, that is 8.1 ms against 0.40 ms for
24 layers, at 249.5 GB/s of state traffic.

Numerics: `tests/hip/deltanet_test.cpp` bounds both the fused pair and the five
separate operators against an independent host reference over three tokens with a
decay of about 0.75; the worst difference is 3e-8, and the two device paths agree
with each other to 7e-9. On the real model the two plans produce identical greedy
tokens over eight steps with a 9.5e-6 maximum logit difference.

### Int8 decode matvec

`build/yuuka_gemv_decode_sweep --int8`, the same protocol as the decode matvec
table, one row per submission with the matrix resident. The int8 forms are the
int8 route: the pre-quantized form reads the Q8_1 digits of the activation
(`YUUKA_OP_QUANT_Q8_1`), the fused form quantizes the row inside the matvec block.

| Shape | dtype | FP32 stream | int8 pre-quantized | int8 fused |
| --- | --- | ---: | ---: | ---: |
| `ffn_gate` 2560 -> 9216 | Q4_K | 198.8 | 210.6 | 194.2 |
| `ffn_up` 2560 -> 9216 | Q4_K | 197.4 | 209.9 | 184.2 |
| `ffn_down` 9216 -> 2560 | Q6_K | 204.5 | 196.1 | 175.2 |
| `ffn_down` 9216 -> 2560 | Q4_K | 141.6 | 159.2 | 133.6 |
| `attn_q` 2560 -> 8192 | Q4_K | 177.7 | 190.3 | 168.7 |
| `attn_k` 2560 -> 1024 | Q4_K | 64.3 | 66.1 | 57.3 |
| `attn_v` 2560 -> 1024 | Q6_K | 95.8 | 97.2 | 86.8 |
| `attn_output` 4096 -> 2560 | Q4_K | 109.6 | 146.7 | 108.0 |
| `attn_gate` 2560 -> 4096 | Q4_K | 165.3 | 166.8 | 162.8 |
| `attn_qkv` 2560 -> 8192 | Q5_K | 183.5 | 224.2 | 199.0 |
| `ssm_out` 4096 -> 2560 | Q5_K | 126.8 | 133.8 | 130.3 |
| `lm_head` 2560 -> 248320 | Q6_K | 182.6 | 194.9 | 183.4 |

In the decode step (`TestGPUProfileDecode` with
`YUUKA_TEST_DECODE_INT8=fused|prequantized`), and end to end on the 128-token
prompt:

| Route | Matvec | Quantization | Profiled step | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| FP32 streaming, selected | 21.39 ms | - | 31.57 ms | 29.91 |
| int8 pre-quantized | 20.07 ms | 1.09 ms | 31.78 ms | 30.64 |
| int8 fused | 21.9 ms | - | 32.27 ms | 29.75 |

The int8 arithmetic is 6% faster in step, the quantization pass costs 5% back,
and the fused form costs more than the launch it removes. Decode stays on the
FP32 streaming kernel: the step reads 2,726 MB of weights at 127 GB/s against a
244 GB/s read ceiling, so it is not instruction bound, and the 31 tokens/s
target is missed. Numerics: identical tokens over eight greedy steps with
0.2656 maximum logit difference against the FP32 route, and 5.7e-4 relative
against the quantized host reference in `tests/hip/gemv_test.cpp`.

## Speculative decoding

Exact greedy prompt-lookup speculation, 512-token prompt, 128 generated tokens,
F32 KV, median of three repetitions:

| Path | Prefill p50 | TTFT p50 | Decode p50 | Decode tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain greedy | 0.396 s | 0.396 s | 14.050 s | 9.04 | 1.00x |
| Speculative (`--speculative`) | 0.396 s | 0.509 s | 3.500 s | 36.28 | 4.01x |

The generated tokens are identical to the plain greedy path. TTFT rises because
the first step drafts and verifies a block before emitting a token. Acceptance
is prompt-dependent: the repeated-prose benchmark prompt accepts draft blocks
almost fully, while a prompt with no repeated n-grams falls back to normal
decode and gains nothing.

## KV cache formats

Qwen3.5-4B has 8 full-attention layers, 4 KV heads, and head size 256, so one
KV row is 1024 values per layer:

| Format | Bytes per token (K+V) | Relative to F32 | 4096-token request |
| --- | ---: | ---: | ---: |
| F32 | 65,536 | 1.00x | 256 MiB |
| F16 | 32,768 | 0.50x | 128 MiB |
| Q8_0 | 17,408 | 0.266x | 68 MiB |

At a 128-token prompt and 32 generated tokens, median of three repetitions:

| KV format | Prefill p50 | Decode p50 | Decode tok/s |
| --- | ---: | ---: | ---: |
| F32 | 0.123 s | 3.203 s | 9.68 |
| F16 | 0.123 s | 3.189 s | 9.72 |
| Q8_0 | 0.123 s | 3.182 s | 9.74 |

The differences are inside run-to-run noise: KV format is a memory-capacity
feature at this context length, and the prefill effect appears only through the
attention kernel the format selects (see [prefill attention](#prefill-attention)).
The real-model
correctness test produced the same greedy tokens for all three formats, with
maximum logit difference from F32 of `1.12e-3` (F16) and `4.53e-2` (Q8_0).

## Prefill attention

Isolated paged prefill attention, one layer, F32 pool, Qwen3.5 geometry (16
query heads, 4 KV heads, head width 256), median of 20 launches after a warmup.
`tiled` is `attention_prefill_paged_f32_kernel`, `mma` is
`attention_prefill_paged_wmma_kernel` from
[kernels.md](kernels.md#75-fp16-mma-paged-prefill-attention). Op-level harness:
one `ATTENTION_PREFILL_PAGED` op, no other work in the plan.

| Rows | Position | Tiled | MMA | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 | 0.016 ms | 0.039 ms | 0.41x |
| 32 | 0 | 0.027 ms | 0.041 ms | 0.66x |
| 48 | 0 | 0.044 ms | 0.052 ms | 0.85x |
| 64 | 0 | 0.075 ms | 0.057 ms | 1.32x |
| 128 | 0 | 0.195 ms | 0.083 ms | 2.35x |
| 512 | 0 | 2.468 ms | 0.507 ms | 4.87x |
| 16 | 256 | 0.209 ms | 0.119 ms | 1.76x |
| 32 | 1024 | 0.844 ms | 0.311 ms | 2.71x |
| 16 | 2048 | 3.518 ms | 0.609 ms | 5.78x |
| 32 | 7680 | 14.651 ms | 3.266 ms | 4.49x |
| 512 | 2048 | 18.809 ms | 3.409 ms | 5.52x |
| 512 | 7680 | 70.243 ms | 14.815 ms | 4.74x |

The dispatch rule follows the table: the MMA kernel is selected from 64 rows or
from a 256-token prefix. Over the eight full-attention layers of a 512-token
chunk that is 4.1 ms of attention at position 0 and 117.6 ms at position 7680,
against 19.7 ms and 562 ms for the tiled kernel (the design targets are 6 ms and
150 ms).

Numerics, same harness: the MMA output differs from the FP32 reference by
9.3e-5 maximum absolute value at head width 256 and differs from the tiled
kernel by the same 9.3e-5, which is entirely the FP16 staging step. Model level,
`TestGPUAttentionPathsAgree` records 0.015 maximum logit difference at a 64-token
prompt (top-1 `33075` in both arms), against 0.011 for the F32-to-F16 KV storage
change on the tiled kernel over the same prompt. At 2048 tokens the same
comparison records 0.220 (MMA) and 0.220 (rocBLAS) against 0.235 for the
KV-format change, because the int8 prefill route amplifies any ~1e-4 hidden-state
change onto different Q8_1 grid points.

## Isolated kernels

### Prefill GEMM

`build/yuuka_prefill_sweep`, TFLOP/s per cell, five timed repetitions after two
warmups. `mmq` is the int8 Q8_1 route and includes its activation-quantization
pass; `wmma` is the custom rocWMMA kernel. K-quant dispatch selects `mmq` for
`M >= 128` and `N >= 64`.

| dtype | shape (K -> N) | M | wmma | mmq |
| --- | --- | ---: | ---: | ---: |
| Q4_K | 2560 -> 9216 | 128 | 8.6 | 17.2 |
| Q4_K | 2560 -> 9216 | 512 | 8.4 | 17.3 |
| Q4_K | 9216 -> 2560 | 128 | 9.8 | 18.4 |
| Q4_K | 9216 -> 2560 | 512 | 8.0 | 18.3 |
| Q4_K | 2560 -> 8192 | 128 | 7.8 | 15.6 |
| Q4_K | 2560 -> 8192 | 512 | 8.3 | 17.9 |
| Q5_K | 2560 -> 9216 | 128 | 8.5 | 17.2 |
| Q5_K | 2560 -> 9216 | 512 | 8.3 | 17.6 |
| Q5_K | 9216 -> 2560 | 128 | 8.6 | 17.4 |
| Q5_K | 9216 -> 2560 | 512 | 7.5 | 18.1 |
| Q5_K | 2560 -> 8192 | 128 | 7.9 | 14.8 |
| Q5_K | 2560 -> 8192 | 512 | 8.5 | 17.5 |
| Q6_K | 2560 -> 9216 | 128 | 7.3 | 14.0 |
| Q6_K | 2560 -> 9216 | 512 | 7.6 | 14.4 |
| Q6_K | 9216 -> 2560 | 128 | 7.1 | 14.0 |
| Q6_K | 9216 -> 2560 | 512 | 6.4 | 12.4 |
| Q6_K | 2560 -> 8192 | 128 | 6.8 | 12.9 |
| Q6_K | 2560 -> 8192 | 512 | 7.2 | 13.1 |

F16 and Q8_0 rows are not routed through `mmq`; for reference, `wmma` reaches
8.0-9.2 TFLOP/s on Q8_0 and 3.4-4.7 on F16, while dequantize-plus-rocBLAS
reaches 16.0-19.0 on F16 for the wide shapes (8.1 on the tall-`K` row) plus the
dequantize pass, and is rejected for K-quants on numeric grounds.

The int8 route profiles at 22.2 VALU and 4.8 LDS instructions per
4096-MAC WMMA on Q4_K `2560 -> 9216` at M=512, with `LDSBankConflict` at 0.0%
and two blocks resident per CU (`OccupancyPercent` 45.6%), against an MMA-only
ceiling of 48-52 TFLOP/s for the same tile geometry. The loop is VALU-issue
bound: 16 of the 22.2 VALU are the scale application, which is four operations
per accumulator element per 32-value block by construction, and 2.5 of the 4.8
LDS are the `ds_swizzle_b32` that rocWMMA uses to duplicate gfx11 wave32
operands. See [decisions and rejected routes](#decisions-and-rejected-routes)
and [kernels.md](kernels.md#63-int8-mmq-gemm).

### Batched GEMV

`build/yuuka_gemv_sweep` at `n_in = n_out = 2560`, two warmups and ten
repetitions per cell. Values are the batched kernel's speedup over one GEMV
submission per row:

| dtype | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 1.03 | 1.59 | 2.35 | 3.17 | 3.91 |
| Q5_K | 1.15 | 1.68 | 2.51 | 3.51 | 4.32 |
| Q6_K | 1.08 | 1.28 | 1.87 | 2.52 | 3.14 |
| Q8_0 | 0.74 | 0.95 | 1.28 | 1.64 | 1.73 |
| Q4_0 | 1.03 | 0.96 | 1.33 | 1.59 | 1.97 |
| Q5_0 | 0.73 | 0.96 | 1.25 | 1.81 | 2.07 |
| F16 | 0.71 | 0.89 | 1.17 | 1.31 | 1.49 |

K-quants win from batch 2, the other types from batch 4; the dispatch table uses
those thresholds. Single-request decode always runs batch 1, so this table is
prefill and batching evidence, not decode evidence.

### Prefill LM head

`LMHEAD` cells of `build/yuuka_prefill_sweep`: the streaming LM head against one
GEMV submission per row, `n_in = 2560`, median of five timed submissions after
two warmups, per-row GEMV inside one plan:

| dtype | N | M | per-row GEMV | GB/s | streaming LM head | GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 512 | 1 | 0.039 ms | 19 | 0.024 ms | 31 |
| Q4_K | 1024 | 1 | 0.056 ms | 26 | 0.024 ms | 62 |
| Q4_K | 2048 | 1 | 0.098 ms | 30 | 0.035 ms | 83 |
| Q4_K | 4096 | 1 | 0.173 ms | 34 | 0.035 ms | 167 |
| Q4_K | 32768 | 1 | 1.876 ms | 25 | 0.319 ms | 148 |
| Q4_K | 248320 | 1 | 15.008 ms | 24 | 2.356 ms | 152 |
| Q4_K | 248320 | 16 | 240.817 ms | 1.5 | 6.585 ms | 54 |
| Q6_K | 512 | 1 | 0.076 ms | 14 | 0.022 ms | 48 |
| Q6_K | 1024 | 1 | 0.047 ms | 46 | 0.023 ms | 93 |
| Q6_K | 2048 | 1 | 0.076 ms | 57 | 0.036 ms | 121 |
| Q6_K | 4096 | 1 | 0.136 ms | 63 | 0.039 ms | 220 |
| Q6_K | 32768 | 1 | 1.519 ms | 45 | 0.393 ms | 175 |
| Q6_K | 248320 | 1 | 11.937 ms | 44 | 2.896 ms | 180 |
| Q6_K | 248320 | 16 | 191.002 ms | 2.7 | 7.667 ms | 68 |
| Q8_0 | 512 | 1 | 0.072 ms | 20 | 0.092 ms | 15 |
| Q8_0 | 1024 | 1 | 0.025 ms | 111 | 0.025 ms | 110 |
| Q8_0 | 2048 | 1 | 0.045 ms | 125 | 0.040 ms | 140 |
| Q8_0 | 4096 | 1 | 0.089 ms | 125 | 0.036 ms | 310 |
| Q8_0 | 32768 | 1 | 0.925 ms | 96 | 0.442 ms | 202 |
| Q8_0 | 248320 | 1 | 7.215 ms | 94 | 3.113 ms | 217 |
| Q8_0 | 248320 | 16 | 115.507 ms | 5.8 | 7.135 ms | 95 |

The streaming form wins from 512 outputs for the K-quants and from 2048 for
Q8_0; the dispatch rule therefore stops at 1024 outputs, where Q8_0 is level.
The 521 MB Q6_K cell at 180 GB/s is 74% of the 244 GB/s a pure streaming read
reaches on this device with the same matrix size, and the row count is nearly
free: sixteen rows cost 2.6x one row instead of 16x. See
[kernels.md](kernels.md#65-streaming-lm-head) for the kernel contract.

### Decode matvec

`build/yuuka_gemv_decode_sweep`, the exact shapes one `qwen35` decode token
dispatches at the dtypes the GGUF carries, measured 2026-11-18 at commit
`bfd36c0` on the reference device with the matrix resident. Each cell runs five
warmups and twenty-five timed submissions of one plan operation; the value is
the median submission, timed on the host around the submit and the wait. The
traced run of the same binary supplies the kernel time and the instruction
counts. `scripts/decode_profile.sh` reproduces this table, and
[testing.md](testing.md#63-what-to-measure-separately) describes the
per-operation attribution it feeds.

`GEMV` is the per-row kernel the decode plan uses today, `batch 4` the batched
form, and `LM head` the streaming kernel that the prefill tail already uses.
`attn_k` and `attn_v` share the `2560 -> 1024` shape, so the `Q4_K` row covers
both; `ffn_gate` and `ffn_up` share `2560 -> 9216`.

| tensor | dtype | shape (K -> N) | GEMV ms | GB/s | batch 4 ms | GB/s | LM head ms | GB/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ffn_gate`, `ffn_up` | Q4_K | 2560 -> 9216 | 0.351 | 38 | 0.553 | 24 | 0.067 | 197 |
| `ffn_down` | Q6_K | 9216 -> 2560 | 0.255 | 76 | 0.494 | 39 | 0.097 | 199 |
| `ffn_down` | Q4_K | 9216 -> 2560 | 0.316 | 42 | 0.517 | 26 | 0.096 | 139 |
| `attn_q` | Q4_K | 2560 -> 8192 | 0.316 | 37 | 0.501 | 24 | 0.069 | 172 |
| `attn_k`, `attn_v` | Q4_K | 2560 -> 1024 | 0.059 | 25 | 0.094 | 16 | 0.023 | 64 |
| `attn_v` | Q6_K | 2560 -> 1024 | 0.055 | 39 | 0.094 | 23 | 0.022 | 97 |
| `attn_output` | Q4_K | 4096 -> 2560 | 0.146 | 40 | 0.230 | 26 | 0.054 | 109 |
| `attn_gate` | Q4_K | 2560 -> 4096 | 0.171 | 34 | 0.264 | 22 | 0.036 | 165 |
| `attn_qkv` | Q5_K | 2560 -> 8192 | 0.355 | 41 | 0.533 | 27 | 0.083 | 173 |
| `ssm_out` | Q5_K | 4096 -> 2560 | 0.185 | 39 | 0.249 | 29 | 0.058 | 124 |
| `ssm_beta` | Q8_0 | 2560 -> 32 | 0.009 | 10 | 0.016 | 6 | 0.024 | 4 |
| `lm_head` | Q6_K | 2560 -> 248320 | 11.218 | 46 | 17.138 | 30 | 2.780 | 188 |

Per-cell instruction counts from the traced run, as wave-level instructions
times 32 lanes per packed weight. `lds/w` counts shared-memory instructions, and
the per-row GEMV's scale staging is what puts it two orders of magnitude above
the streaming kernel:

| cell | form | waves | VALU per weight | LDS per weight |
| --- | --- | ---: | ---: | ---: |
| `ffn_gate` Q4_K 2560 -> 9216 | GEMV | 73,728 | 18.0 | 2.66 |
| `ffn_gate` Q4_K 2560 -> 9216 | batch 4 | 73,728 | 62.6 | 1.65 |
| `ffn_gate` Q4_K 2560 -> 9216 | LM head | 2,304 | 8.5 | 0.16 |
| `ffn_down` Q6_K 9216 -> 2560 | GEMV | 20,480 | 26.9 | 0.11 |
| `ffn_down` Q6_K 9216 -> 2560 | LM head | 640 | 10.9 | 0.11 |
| `attn_qkv` Q5_K 2560 -> 8192 | GEMV | 65,536 | 20.2 | 2.66 |
| `attn_qkv` Q5_K 2560 -> 8192 | LM head | 2,048 | 13.8 | 0.16 |
| `lm_head` Q6_K 2560 -> 248320 | GEMV | 1,986,560 | 29.1 | 0.41 |
| `lm_head` Q6_K 2560 -> 248320 | LM head | 62,080 | 11.3 | 0.16 |

Findings:

- The per-row GEMV is 4-6x slower than the streaming kernel on every shape the
  model uses except `ssm_beta` at 32 outputs, where it is 2.6x faster. That
  crossover is why the streaming route stays behind a shape rule rather than
  becoming the default.
- The instruction counts explain the shape of the loss, but not all of it: the
  per-row GEMV executes 18 to 29 VALU per weight against 8.5 to 13.8 for the
  streaming kernel, a factor of two, while the time differs by four to six. The
  rest is the per-row GEMV's block structure: one 256-thread block per output
  row, a block-wide tree reduction, and two barriers per 256-value K step.
- Batching four rows into one submission does not amortize the weight read. The
  batched kernel executes 3.5x the instructions of the single-row form because
  it repeats the K loop per row, so its per-row cost is 1.5-2x better and its
  per-submission cost is worse. Decode at batch 1 gains nothing from it.
- Wall time minus kernel time is the per-submission launch cost: 20-30
  microseconds for the small cells. The instrumented decode step pays that 579
  times, which is the 6.2% submission gap of the profiled decode step
  ([decode fusion](#decode-fusion)).
- Two consecutive runs of `scripts/decode_profile.sh` agree within 9% on every
  cell and within 0.3% on the profiled decode step, so the table is stable
  enough to judge a kernel change against.

`build/yuuka_gemv_decode_sweep --threshold` measures the rows = 1 crossover that
`lmHeadMinOutputs` encodes, at the decode width with five warmups and
twenty-five timed submissions per cell. "From" is the smallest output count at
which the streaming kernel wins and keeps winning:

| dtype | GEMV at 256 | GEMV at 1024 | streaming at 1024 | streaming wins from |
| --- | ---: | ---: | ---: | ---: |
| Q4_K | 20.2 GB/s | 24.8 GB/s | 65 GB/s | 512 outputs |
| Q5_K | 23.2 GB/s | 27.9 GB/s | 72 GB/s | 512 outputs |
| Q6_K | 34.4 GB/s | 39.7 GB/s | 99 GB/s | 512 outputs |
| Q8_0 | 58.1 GB/s | 116.8 GB/s | 115 GB/s | 1536 outputs |
| F16 | 30.5 GB/s | 233.6 GB/s | 150 GB/s | 1536 outputs |

The threshold stays at 1024: it is the smallest single width that does not
regress Q8_0 or F16, where the vectorized GEMV is still ahead at that width, and
no output projection sits between 512 and 1536 outputs. See
[kernels.md](kernels.md#65-streaming-lm-head) for the rule these cells encode.

The hot-matrix sweep and the real decode step do not measure the same thing,
because a single resident matrix can stay in the 32 MB MALL cache while a token
rotates through 2.7 GB of weights. Per-layer comparison, same run, sweep value
against the in-step value of the profiled cell:

| matrix | size | sweep GB/s | step GB/s | ratio |
| --- | ---: | ---: | ---: | ---: |
| `attn_gate` Q4_K 2560 -> 4096 | 5.9 MB | 34 | 26.3 | 1.29 |
| `ssm_out` Q5_K 4096 -> 2560 | 7.2 MB | 39 | 28.6 | 1.36 |
| `attn_q` Q4_K 2560 -> 8192 | 11.8 MB | 37 | 26.1 | 1.42 |
| `ffn_gate` Q4_K 2560 -> 9216 | 13.3 MB | 38 | 26.3 | 1.44 |
| `ffn_down` Q4_K 9216 -> 2560 | 13.3 MB | 42 | 29.7 | 1.41 |
| `attn_qkv` Q5_K 2560 -> 8192 | 14.4 MB | 41 | 27.1 | 1.51 |
| `ffn_down` Q6_K 9216 -> 2560 | 19.4 MB | 76 | 47.9 | 1.59 |
| `lm_head` Q6_K 2560 -> 248320 | 521.5 MB | 46 | 46.4 | 0.99 |

Every matrix below the cache size reads 1.3-1.6x faster in the sweep, and the
one matrix far above it agrees within 1%. Use the sweep for the kernel's own
ceiling and the step column for what a token costs.

### Grouped MoE pipeline

Synthetic single-layer shape: 128 experts, top-8, hidden 2048, expert FFN 768,
Q4_K weights. One route plus grouped gate/up, SwiGLU, grouped down, and weighted
reduce on one stream:

| Rows | Route ms | Gate/up ms | Down ms | Reduce ms | Total ms | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.220 | 0.389 | 0.401 | 0.015 | 1.023 | 978 |
| 2 | 0.222 | 0.992 | 0.761 | 0.016 | 1.989 | 1005 |
| 4 | 0.228 | 1.875 | 1.410 | 0.018 | 3.530 | 1133 |
| 8 | 0.223 | 3.559 | 2.603 | 0.025 | 6.418 | 1246 |
| 16 | 0.231 | 6.800 | 4.602 | 0.031 | 11.621 | 1377 |
| 32 | 0.231 | 12.631 | 8.838 | 0.036 | 21.607 | 1481 |

These are kernel-pipeline numbers. No supported model runtime executes the
grouped HIP kernels, so there is no end-to-end MoE throughput number.

## Kernel resource usage

`scripts/kernel_resources.sh` compiles the dispatched families with
`-Rpass-analysis=kernel-resource-usage`. Every listed kernel reports zero
scratch and zero VGPR spills. Dynamic LDS is set by the launcher and does not
appear in the compiler's static figure.

| Kernel | VGPR | SGPR | Static LDS |
| --- | ---: | ---: | ---: |
| `gemm_mmq_kernel<Q4_K>` / `<Q5_K>` / `<Q6_K>` | 176 / 177 / 162 | 48 / 48 / 52 | 0 (64512 B dynamic) |
| `lm_head_kernel<*,1>` | 45-67 | 26-32 | 0 (dynamic `rows * 1024 B`) |
| `lm_head_kernel<*,16>` | 97-127 | 42-44 | 0 (dynamic `rows * 1024 B`) |
| `quantize_q8_1_kernel` | 20 | 27 | 0 |
| `gemm_wmma_kernel<*,64,2>` | 84-89 | 28-48 | 16,384 B |
| `gemm_wmma_k32_kernel<*>` | 42-58 | 25-46 | 14,336 B |
| `gemm_wmma_swiglu_kernel<*,64,2>` | 85-138 | 28-50 | 24,576 B |
| `gemm_wmma_swiglu_kernel<*,32,1>` | 85-90 | 30-49 | 20,480 B |
| `gemv_f32_kernel` / `gemv_f16_kernel` | 11 / 11 | 17 / 17 | 1,024 B |
| `gemv_q8_0_kernel` / `gemv_q6_k_kernel` | 12 / 24 | 17 / 20 | 1,024 B |
| `gemv_packed_kernel<Q4_0>` / `<Q5_0>` | 13 / 15 | 17 / 17 | 1,024 B |
| `gemv_k_kernel<Q4_K/Q5_K>` / `<Q6_K>` | 20 / 24 | 18 / 18 | 1,088 B |
| `gemv_batch_kernel<*>` | 26-34 | 37 | 1,024 B |
| `attention_prefill_f32_kernel` | 49 | 48 | 0 (36,864 B dynamic) |
| `attention_prefill_paged_f32_kernel<*>` | 51-53 | 61-63 | 0 |
| `attention_prefill_paged_wmma_kernel<*,256,*>` | 256 | 30-34 | 0 (50,176 B dynamic) |
| `gdn_chunk_f32_kernel` / `gdn_chunk_splitreg_f32_kernel<128>` | 37 / 93 | 70 / 34 | 8 B / 1,064 B |
| `gdn_norm_gate_rows_f32_kernel` | 17 | 18 | 0 |
| `rmsnorm_batch_f32_kernel` / `rmsnorm_heads_batch_f32_kernel` | 9 / 9 | 18 / 20 | 1,024 B |
| `rope_batch_f32_kernel` | 19 | 18 | 0 |
| `kv_write_batch_f32_kernel` / `kv_write_paged_kv_kernel` | 14 / 30 | 34 / 36 | 0 |
| `embedding_batch_f32_kernel` | 21 | 19 | 0 |
| `conv1d_chunk_f32_kernel` / `conv1d_state_f32_kernel` | 20 / 7 | 40 / 17 | 0 |
| `cast_f32_to_f16_kernel` / `cast_f16_to_f32_kernel` | 7 / 7 | 18 / 18 | 0 |
| `moe_route_kernel` / `moe_gate_up_grouped_kernel` / `moe_down_grouped_kernel` / `moe_reduce_add_kernel` | 10 / 23 / 21 / 9 | 32 / 48 / 42 / 21 | 0 / 1,024 B / 1,024 B / 0 |

## Serving and server mode

### In-process scheduler

`bench/serving`, HIP backend, context 256, block size 16, 64-token prompt and 16
generated tokens, greedy, p50 of three measured requests:

| Configuration | Latency | Queue wait | TTFT | Prefill | Decode | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Single request, graph replay | 1.5605 s | 2 µs | 0.1064 s | 0.0935 s | 1.4505 s | 10.21 |
| Single request, plain stream | 1.5595 s | 8 µs | 0.1044 s | 0.0925 s | 1.4517 s | 10.21 |
| Concurrency 4, graph replay | 6.3596 s | 0.1051 s | 0.6142 s | 0.0934 s | 1.4896 s | 9.90 aggregate |

Graph replay is inside noise at this context, so the server default stays the
plain stream path. Concurrency 4 admits four requests and exposes queue wait and
scheduler overhead, but aggregate output throughput stays at the single-request
rate: per-request decode time moves by under 3% from the single-user row because
each request's decode step reads the full quantized weights on its own. Fused
mixed-shape multi-request decode batching is not implemented.

Prefix cache, three identical 64-token prompts, no warmup: the first request
paid 0.109 s TTFT and 0.0956 s prefill; the two subsequent requests reported
4.286 ms TTFT p50 with zero prefill time, and aggregate throughput over the run
was 10.70 output tokens/s against 10.21 without the cache.

### OpenAI server

`yuuka-server` with `--backend hip --ctx 4096 --kv-type f32 --max-concurrency 1`
(4 for the concurrency row) and `llama-server` on the same model and device,
driven by an identically configured streaming client on localhost, prompt cache
off, greedy:

| Request | Metric | llama.cpp | yuuka |
| --- | --- | ---: | ---: |
| 140-token prompt, ≤128 out | Prefill tok/s | 1028 | 765 |
| | TTFT | 119 ms | 193 ms |
| | Decode tok/s | 59.07 | 9.49 |
| 530-token prompt, ≤128 out | Prefill tok/s | 1630 | 1157 |
| | TTFT | 287 ms | 465 ms |
| | Decode tok/s | 56.59 | 8.95 |
| 2051-token prompt, ≤64 out | Prefill tok/s | 2148 | 1115 |
| | TTFT | 946 ms | 1848 ms |
| | Decode tok/s | 50.53 | 7.35 |
| 4 concurrent, 140-token prompt, ≤64 out | Aggregate decode tok/s | 149.5 | 9.29 |
| | Per-request decode tok/s | 39.17 | 2.41 |
| | TTFT median | 371 ms | 690 ms |

The yuuka decode column of this table was measured before the streaming matvec
and gated delta-net work and is kept as the recorded server comparison; the CLI rows above are the current decode
rate, and the server path has not been re-measured against llama.cpp since.

Both servers tokenize the benchmark prompt to the same length (140, 530, 2051),
so the prefill comparison is like for like. yuuka-server prefill throughput sits
5-28% below the CLI rows at comparable prompt lengths (28% at ~140 tokens, 13%
at ~530, 5% at ~2050); that span covers the chat template, the HTTP path, and
server scheduling, so CLI and server numbers are not interchangeable.

### Gated delta-net chunk

Per recurrent layer of a 512-row chunk, one `RECURRENT_CHUNK` op in a plan,
20 repetitions after 3 warmups, 24 layers reported as the total. The per-token
path is the selected one; the chunkwise form is measured for
[kernels.md](kernels.md#91-chunkwise-form).

| Path / kernel | Per layer | 24 layers |
| --- | ---: | ---: |
| Per-token `gdn_chunk_splitreg_f32_kernel`, whole op | 2.01 ms | 48.2 ms |
| Chunkwise, whole op | 3.10 ms | 74.5 ms |
| Chunkwise, `gdn_scan_f32_kernel` | 1.73 ms | 41.5 ms |
| Chunkwise, `gdn_prep_f32_kernel` | 0.85 ms | 20.4 ms |
| Chunkwise, convolution + gated RMSNorm + preparation | 0.36 ms | 8.5 ms |

The convolution, gated RMSNorm, and preparation kernels are shared by both
paths. The chunkwise scan moves 2x the arithmetic of the per-token kernel
through the FP32 FMA pipe and is LDS-latency bound, so it does not pay for the
extra work; see [kernels.md](kernels.md#91-chunkwise-form).

## llama.cpp comparison

`llama-bench -ngl 99 -b 512 -ub 512 -r 3`, F16 KV (llama.cpp's default; its
`llama-bench` rejects an `f32` KV type), against the yuuka rows from
[prefill](#prefill-prompt-length) and [decode](#decode-output-length):

| Test | llama.cpp t/s | yuuka t/s | yuuka/llama |
| --- | ---: | ---: | ---: |
| pp128 | 1743.28 ± 233.98 | 1344.4 (F16 KV) | 0.77 |
| pp512 | 2346.17 ± 80.87 | 1634.8 (F16 KV) | 0.70 |
| pp2048 | 2409.70 ± 5.39 | 1641.5 (F16 KV) | 0.68 |
| pp8192 | 2170.79 ± 4.19 | 1438.2 (F16 KV) | 0.66 |
| tg128 (prompt 128) | 62.29 ± 0.19 | 38.70 (F32 KV) | 0.62 |

The remaining gaps, in order of size:

1. Decode, 1.6x: llama.cpp streams the 2.726 GB of quantized weights at about
   170 GB/s, near the practical memory roofline; yuuka streams the same weights at
   129.7 GB/s inside a step after moving the whole matvec family to the
   channel-parallel kernel ([decode matvec](#decode-matvec)). What remains is 90%
   matvec at 129.7 GB/s against a 244 GB/s read ceiling, plus 0.79 ms of gated
   delta-net and 1.6 ms of other operations in a 23.4 ms step; the delta-net and
   the submission gaps left the critical path with those fusions
   ([gated delta-net](#gated-delta-net), [decode fusion](#decode-fusion)).
   The matvec's remaining loss is a per-launch ramp of about 22 us on the 5-19 MB
   matrices, which only merges of same-dtype, comparable-width projections
   amortize.
2. Prefill at short and mid prompt lengths, 1.3-1.7x: the quantized prefill GEMM
   runs at 17-18 TFLOP/s against llama.cpp's int8 MMQ kernel, measured at
   24.9 TFLOP/s on the same device in the 2026-11-16 attribution (see
   [isolated kernels](#isolated-kernels)); the gated delta-net chunk adds
   further time yuuka does not measure separately here.
3. Long-context prefill, 1.5x at 8192: the FP16 MMA attention kernel closed
   most of this gap, so what remains is the quantized GEMM plus the delta-net
   chunk, not attention. Q8_0 KV keeps the tiled kernel and stays at 0.40.
4. Multi-request decode, 16x aggregate: llama.cpp batches the resident requests
   into one weight pass; yuuka re-reads the weights per request because fused
   mixed-shape decode batching is not implemented.

## Decisions and rejected routes

Each of these is a measured decision that still holds in the current code:

- **FP32 accumulation is required.** FP16 compute with an FP16 destination
  moves the 512-token prefill logits by 0.305 against the 0.05 chunk gate, so
  every prefill matmul keeps an FP32 accumulator.
- **The dequantize-plus-rocBLAS prefill route is removed** for K-quants. It was
  1.6-3.3x faster per GEMM from 256 rows upward but numerically rejected on
  tall-K shapes and needed an FP16 copy of every weight matrix; the int8 MMQ
  route replaced it.
- **The generic MMQ path uses a 128-value K step.** It stages
  both operands in WMMA fragment order (0.0% bank conflicts from 35.9%),
  expands Q4_K/Q5_K digits with word-wide masks, and uses a 128-value K step so
  two blocks stay resident per CU: 13.8 -> 17.3-18.5 TFLOP/s on Q4_K
  `2560 -> 9216` at M=512. Rejected against it (see
  [decisions and rejected routes](#decisions-and-rejected-routes)):
  a 60 KiB double-buffered 256-value stage (one block per CU, 14.6 TFLOP/s),
  register prefetch of the next step's weights (192 VGPRs, 161-195 spills,
  6.4 TFLOP/s), a 64-value K step (16.4 TFLOP/s), and rebuilding the rocWMMA
  gfx11 operand duplication by storing duplicate operands in LDS. The
  [gfx1151 specialization](kernels.md#gfx1151-direct-mmq-specialization) uses
  LDS broadcast and vectorized activation staging on the measured Q4_K cell.
- **The chunkwise gated delta-net form is implemented but not selected.** It
  matches the per-token recurrence to `1.6e-7` and removes the per-token barrier
  chain, but takes 3.10 ms per recurrent layer against 2.01 ms for the
  per-token kernel, because its dense products run on the FP32 FMA pipe and the
  scan is LDS-latency bound (0.4 instructions per cycle per CU). One state
  column per thread (2.23 ms for the scan alone) and four columns per thread
  (192 VGPRs, 909 spill slots) were measured and rejected. Moving the dense
  products to the FP16 WMMA path is the recorded follow-up.
- **The MMQ scale epilogue is not reducible to two FMA per element.** The
  activation scale differs per accumulator tile, so the 32 multiplies of a
  32-value block cannot be hoisted, and `v_fma_mix_f32` does not convert an
  integer operand on gfx1151. The loop keeps four VALU per element per block,
  which is 16 of its 22.2 VALU per WMMA.
- **128-row WMMA tiles and a 64-wide K step are rejected** for the custom GEMM.
  The 64x64/K32 and 32x64 variants remain the dispatch choice for the paths they
  serve, and the wide K-quant shapes in this report come from the int8 MMQ route
  instead.
- **Prefill attention is settled on the FP16 MMA kernel** for F32 and F16 pools
  from 64 rows or a 256-token prefix; the tiled kernel stays for Q8_0, for
  shapes the MMA kernel does not cover, and below that crossover (measured
  1.2-2.4x faster there). Staging K transposed per key tile instead of Q once
  per workgroup, and keeping the accumulator at 192 registers instead of
  `__launch_bounds__(128, 2)`, were measured and rejected: the first costs 128
  scalar shared-memory loads per key tile, the second spilled 21 slots into the
  key loop for 1.5x. The earlier group-of-four tiled kernel and the rocBLAS
  path are no longer selected by default; rocBLAS stays reachable through
  `LegacyBlasAttention` for A/B measurement.
- **Disabled by default:** the FFN FP16 weight cache (removed with the
  dequantize route) and HIP Graph replay (no gain at the serving contexts
  measured). Speculative decoding is opt-in: exact and 4x faster on repetitive
  prompts, no gain where draft acceptance is low.
- **Known limits:** decode throughput is a single-request weight-bandwidth
  number until fused multi-request decode batching exists; NPU execution is a
  diagnostic probe only; MoE grouped kernels have no supported model runtime;
  the longest measured prompt is 8192 tokens at chunk 512.

## Reproduction

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
go build -tags hip -o /tmp/yuuka-hip ./cmd/yuuka
go build -tags hip -o /tmp/yuuka-serving-bench ./bench/serving
MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf

ctest --test-dir build --output-on-failure
YUUKA_TEST_MODEL="$MODEL" go test -tags hip ./engine/gpu \
  -run 'TestGPUChunkedPrefillLongContext|TestGPUMatchesCPU' -count=1 -v

./build/yuuka_prefill_sweep
./build/yuuka_gemv_sweep
./build/yuuka_moe_sweep
./build/yuuka_gemv_decode_sweep
scripts/kernel_resources.sh
YUUKA_TEST_MODEL="$MODEL" scripts/decode_profile.sh   # decode matvec table
YUUKA_TEST_MODEL="$MODEL" scripts/prefill_counters.sh # prefill counter table

for p in 128 512 1024 2048 4096 8192; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens "$p" --decode-tokens 1 --warmup 2 --reps 5 --kv-type f32
done
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 8192 --decode-tokens 1 --warmup 2 --reps 5 --kv-type f16
for d in 128 256; do
  /tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
    --prompt-tokens 128 --decode-tokens "$d" --warmup 2 --reps 3 --kv-type f32
done
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 512 --decode-tokens 128 --warmup 2 --reps 3 --kv-type f32
/tmp/yuuka-hip bench --model "$MODEL" --backend hip --chunk-size 512 \
  --prompt-tokens 512 --decode-tokens 128 --warmup 2 --reps 3 --kv-type f32 --speculative

/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 64 --concurrency 4 --requests 8 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 1 --graph
/tmp/yuuka-serving-bench --model "$MODEL" --backend hip --ctx 256 --block-size 16 \
  --kv-blocks 16 --concurrency 1 --requests 3 --prompt-tokens 64 --decode-tokens 16 \
  --warmup 0 --graph --prefix-cache

# llama.cpp reference, same model and device.
llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 \
  -p 128,512,2048,8192 -n 0 -r 3 -o md
llama-bench -m "$MODEL" -ngl 99 -b 512 -ub 512 -p 128 -n 128 -r 3 -o md
```

Server-mode comparison: start `yuuka-server --model "$MODEL" --backend hip
--ctx 4096 --kv-type f32 --prefill-chunk 512 --max-concurrency 1
--prefix-cache=false` and `llama-server -m "$MODEL" -ngl 99 -c 4096 -ctk f16
-ctv f16 -b 512 -ub 512 -np 1 --jinja`, then post greedy streaming requests to
`/v1/chat/completions` with `cache_prompt: false` on the llama.cpp side. One
client thread per request, requests alternating between servers, three
repetitions per cell.
