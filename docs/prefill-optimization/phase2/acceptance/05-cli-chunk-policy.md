# Step 1d CLI prefill chunk sweep

## Baseline and protocol

The sweep follows the [Step 1d plan](../stage1d-cli-chunk-policy.md).
The Release source was `0355fb1`, which has the same executable code as
`c7dc298`. The baseline includes Stage 1a `3caea06`, Step 1c `260094f`,
Step 1b-1 `ac659b7` and Step 1b-2 `c7dc298`.

The AI container ran one GPU process at a time on Radeon 8060S
(`gfx1151`), Linux `7.0.14-28-pve`, ROCm 10.1.0, HIP 7.16.26385,
and Go 1.24.4. The host CPU is Ryzen AI Max+ 395 with 117 GiB of
container-visible unified memory. The power profile was automatic. The GPU reported 69–85 °C during
sweep spot checks. One observed 4K run showed 2598 MHz graphics clock
and 119 W socket power. AMD SMI did not expose a throttle status on
this device. The [comparison driver](../../../../bench/prefill-compare/main.go)
uses F16 KV, capacity of prompt length plus 64 before page rounding,
and the following two verified fixtures.

| Input | SHA-256 |
| --- | --- |
| Qwen3.5-4B-Q4_K_M GGUF | `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4` |
| Generated 8192-token little-endian int32 file | `e613216db780a72fdbe796b31c63b4afabf273199dbc67f799b5c082d18e3e30` |

For each prompt prefix, runs used chunk settings `512, 1024, 1024, 512`
and `512, 2048, 2048, 512`. Each process had two warmups and five
timed samples. Each table cell pools ten samples and reports median
and full observed range in milliseconds. The timer starts immediately
before the call to `Session.Prefill` and ends with the final host
logit row. It includes host preparation inside the call. Greedy
continuation and the F32 logit dump are outside that timer.

The driver clamps the configured chunk to the prompt length. Each
sequence uses as many complete chunks as fit and one remainder when
nonzero. For example, 2066 rows at settings 512, 1024 and 2048 execute
`4×512 + 18`, `2×1024 + 18` and `2048 + 18` respectively.

## F16 matched-token results

Each comparison uses its own adjacent 512 control processes. The first
column of IDs contains the control top-1 and three further greedy IDs.
A max difference of zero compares the complete final FP32 logit row.

| Prompt | Control greedy IDs | 512 control (ms) | 1024 (ms) | Change | Max logit diff | Greedy | 512 control (ms) | 2048 (ms) | Change | Max logit diff | Greedy |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 127 | [15217, 5388, 13, 561] | 65.195 [64.941–67.720] | 64.974 [64.552–68.580] | -0.34% | 0.000000 | Match | 64.895 [64.733–65.122] | 64.834 [64.539–67.205] | -0.09% | 0.000000 | Match |
| 128 | [5388, 13, 561, 3841] | 67.620 [67.465–70.616] | 68.062 [67.803–71.689] | +0.65% | 0.000000 | Match | 68.073 [67.938–69.465] | 67.685 [67.415–68.327] | -0.57% | 0.000000 | Match |
| 129 | [13, 561, 3841, 13477] | 86.669 [86.245–89.461] | 86.544 [86.362–86.770] | -0.14% | 0.000000 | Match | 87.070 [86.419–87.849] | 87.025 [86.780–90.832] | -0.05% | 0.000000 | Match |
| 512 | [13477, 37550, 33075, 888] | 213.885 [212.813–217.457] | 214.712 [212.712–220.198] | +0.39% | 0.000000 | Match | 214.195 [212.889–221.728] | 215.225 [213.523–217.858] | +0.48% | 0.000000 | Match |
| 513 | [37550, 33075, 888, 279] | 238.865 [236.545–242.187] | 242.223 [239.888–251.444] | +1.41% | 1.066872 | Match | 236.895 [235.737–246.271] | 244.597 [240.471–253.804] | +3.25% | 1.066872 | Match |
| 530 | [561, 3841, 13477, 37550] | 251.935 [249.247–256.391] | 243.885 [242.260–247.919] | -3.20% | 0.308279 | Match | 252.107 [250.425–257.076] | 244.713 [243.503–248.708] | -2.93% | 0.308279 | Match |
| 600 | [561, 3841, 13477, 37550] | 273.998 [271.814–279.717] | 264.188 [262.983–269.294] | -3.58% | 0.301544 | Match | 272.782 [271.268–278.888] | 265.087 [263.066–272.624] | -2.82% | 0.301544 | Match |
| 1024 | [33075, 888, 279, 15217] | 439.891 [436.966–446.057] | 422.371 [418.304–430.199] | -3.98% | 0.000000 | Match | 441.645 [438.081–449.352] | 420.681 [417.960–430.026] | -4.75% | 0.000000 | Match |
| 1025 | [888, 279, 15217, 5388] | 462.491 [460.547–466.011] | 447.787 [443.408–453.443] | -3.18% | 0.000000 | Match | 463.043 [459.394–472.280] | 450.798 [446.554–455.446] | -2.64% | 0.105489 | Match |
| 1042 | [13477, 37550, 33075, 888] | 476.872 [473.554–481.160] | 456.828 [454.143–460.696] | -4.20% | 0.000000 | Match | 477.508 [471.976–484.965] | 452.583 [447.334–457.758] | -5.22% | 0.163726 | Match |
| 2048 | [5388, 13, 561, 3841] | 902.857 [893.370–908.273] | 862.600 [853.787–875.475] | -4.46% | 0.000000 | Match | 904.935 [897.828–926.771] | 870.109 [863.372–882.363] | -3.85% | 0.000000 | Match |
| 2049 | [13, 561, 3841, 13477] | 928.878 [922.247–936.531] | 890.705 [885.263–905.389] | -4.11% | 0.000000 | Match | 932.385 [923.182–938.935] | 900.068 [887.980–910.517] | -3.47% | 0.000000 | Match |
| 2066 | [279, 15217, 5388, 13] | 940.386 [935.129–975.304] | 898.925 [895.226–903.388] | -4.41% | 0.000000 | Match | 938.984 [933.545–956.823] | 913.015 [907.965–918.835] | -2.77% | 0.000000 | Match |
| 4096 | [279, 15217, 5388, 13] | 1888.618 [1878.125–1913.558] | 1798.080 [1789.399–1835.373] | -4.79% | 0.000000 | Match | 1894.294 [1888.657–1908.095] | 1812.867 [1805.002–1825.947] | -4.30% | 0.000000 | Match |
| 4097 | [15217, 5388, 13, 561] | 1926.930 [1918.750–1944.700] | 1833.988 [1824.477–1867.910] | -4.82% | 0.000000 | Match | 1928.573 [1919.680–1937.925] | 1842.967 [1828.406–1859.896] | -4.44% | 0.000000 | Match |
| 4114 | [33075, 888, 279, 15217] | 1943.223 [1932.473–1964.738] | 1847.427 [1840.142–1860.748] | -4.93% | 0.000000 | Match | 1931.417 [1922.229–1940.149] | 1864.752 [1846.997–1938.167] | -3.45% | 0.000000 | Match |
| 8192 | [13477, 37550, 33075, 888] | 4114.993 [4105.875–4129.707] | 3909.015 [3889.650–3935.964] | -5.01% | 0.000000 | Match | 4114.236 [4106.021–4134.927] | 3923.112 [3898.357–3967.621] | -4.65% | 0.000000 | Match |

At 513 tokens, the larger settings remove the one-token tail but
increase latency by 1.41–3.25%. Keep the 512 route for that part of the
range. At every measured length from 1024 to 8192, chunk 1024 wins by
3.18–5.01%, has zero maximum logit difference, and preserves all four
greedy IDs. Chunk 2048 provides comparable speed with twice the
matching 1024 workspace reservation. The equal-weight ratio across
all 17 prompt lengths is 0.97418 for the measured-band
1024 policy against fixed 512.

## CLI timing and startup allocations

The raw inputs are 204 or 409 repetitions of
`The quick brown fox jumps over the lazy dog. `. Chat runs use the
same text with `--chat`, no system message and thinking disabled.
Each run sets `--ctx 8320 --backend hip --max-tokens 4`, samples
greedily, and starts a fresh process. The table pools two processes
for each arm in each 512/candidate/candidate/512 pair. First stdout
includes process startup, tokenization and setup. Engine TTFT is the
CLI's logged measurement. The sample start temperatures were
59–61 °C.

| Prompt mode | KV | Actual tokens | Candidate chunk | Prefill (ms) | Engine TTFT (ms) | Start to first stdout (ms) | Generated text |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| raw | F32 (default) | 2041 | 1024 | 906.00 → 874.50 | 906.00 → 875.00 | 1448.65 → 1430.99 | Match |
| raw | F32 (default) | 2041 | 2048 | 913.50 → 883.50 | 914.00 → 883.50 | 1458.08 → 1452.69 | Match |
| chat | F32 (default) | 2052 | 1024 | 953.00 → 912.00 | 953.00 → 912.00 | 1490.71 → 1473.00 | Match |
| chat | F32 (default) | 2052 | 2048 | 959.00 → 927.00 | 959.50 → 927.00 | 1496.28 → 1490.96 | Match |
| raw | f16 | 4091 | 1024 | 1892.50 → 1803.50 | 1892.50 → 1804.00 | 2433.96 → 2366.64 | Match |
| raw | f16 | 4091 | 2048 | 1920.00 → 1802.50 | 1920.00 → 1802.50 | 2460.40 → 2367.71 | Match |
| chat | f16 | 4102 | 1024 | 1944.50 → 1854.00 | 1944.50 → 1854.00 | 2477.32 → 2414.60 | Match |
| chat | f16 | 4102 | 2048 | 1933.50 → 1852.00 | 1934.50 → 1852.50 | 2465.56 → 2418.52 | Match |
| raw | q8_0 | 2041 | 1024 | 1119.50 → 1096.00 | 1119.50 → 1096.00 | 1641.67 → 1640.00 | Match |
| raw | q8_0 | 2041 | 2048 | 1126.00 → 1099.00 | 1126.00 → 1099.50 | 1653.24 → 1648.95 | Match |
| chat | q8_0 | 2052 | 1024 | 1162.50 → 1139.00 | 1162.50 → 1139.50 | 1693.41 → 1679.51 | Match |
| chat | q8_0 | 2052 | 2048 | 1169.50 → 1150.50 | 1169.50 → 1150.50 | 1700.64 → 1707.82 | Match |

Each latency entry is the paired 512 control followed by the candidate.
The separate F32 `yuuka bench` 2049-token smoke used
`--decode-tokens 1 --ctx 8320 --warmup 2 --reps 5`. Prefill p50
was 930 ms with the omitted flag and 932/896/898 ms with explicit
512/1024/2048, respectively.

The live prefill bytes are the sum of [configurePrefill](../../../../engine/gpu/model.go)
entries for the verified 4B model at the same `--ctx 8320`. The
base workspace is 308,740 bytes per row and F16 adds 66,112 bytes
per row for split-key attention scratch. These default runs do
not enable the legacy attention BLAS workspace. Peak requested
device bytes were measured separately by tracing successful
`hipMalloc` and `hipFree` around the same CLI, with one 128-token
prefill and the indicated explicit setting. The allocation trace
from `rocprofv3` produced no files for this Go executable, so the
measurement used a temporary preload wrapper around the HIP APIs.
The peak deltas match the prefill buffer size deltas exactly.

| KV | Chunk | Live prefill bytes | Increase vs 512 (MiB) | Peak live HIP bytes |
| --- | ---: | ---: | ---: | ---: |
| F16 | 512 | 191,924,224 | 0.000 | 3,265,250,856 |
| F16 | 1024 | 383,848,448 | 183.033 | 3,457,175,080 |
| F16 | 2048 | 767,696,896 | 549.100 | 3,841,023,528 |
| F32 | 512 | 158,074,880 | 0.000 | 3,504,031,272 |
| F32 | 1024 | 316,149,760 | 150.752 | 3,662,106,152 |
| F32 | 2048 | 632,299,520 | 452.256 | 3,978,255,912 |
| Q8_0 | 512 | 158,074,880 | 0.000 | 3,103,606,312 |
| Q8_0 | 1024 | 316,149,760 | 150.752 | 3,261,681,192 |
| Q8_0 | 2048 | 632,299,520 | 452.256 | 3,577,830,952 |

## Sweep decision and verification

The evidence supports validating a single 1024-token setting for
F16 KV on the reference gfx1151 4B Q4_K_M configuration at prompt
lengths 1024–8192. Both CLI first-output cases in this format
improved. F32 and Q8_0 each have only a pair of raw/chat lengths
near 2048, so the full-band evidence applies to F16 only. All ten
measured F16 1024–8192 comparisons also passed binary `cmp` on
the full prefill logit dump.

The AI Release CMake build and `ctest --test-dir build --output-on-failure`
passed (10/10). `go build ./...`, `go vet ./...`, `go test ./...`,
`go test -race ./...`, HIP-tagged build and vet, and the serial
HIP suite passed. The CPU race run used the regular fixtures after
stopping an initial run that also enabled the expensive real-model
CPU test. The normal and HIP-tagged runs exercised that real-model
test. The HIP suite used `YUUKA_TEST_MODEL`, `YUUKA_TEST_KV=f16` and
`YUUKA_TEST_ATTENTION_LONG=1`. Both the 1024-token chunk invariance
and the 8192-token F16 split/direct test ran and passed. Two earlier
short-chunk cross-build tests skipped because their optional
control executables were absent. The optional MoE and two
SentencePiece fixture tests also skipped.

The raw samples and temporary allocation probe are in
`/tmp/yuuka-phase2-5-11.3wZQZG/1d` on AI.

The measurement iteration received `NO_BLOCKING_ISSUES` after one
read-only review. The measured interval description was clarified
and the relative links and `git diff --check` passed.
