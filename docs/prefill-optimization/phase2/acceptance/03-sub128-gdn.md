# Stage 1c short-chunk GDN

## Instrumentation and control

The control was `c09cd06` on AI, gfx1151 (Ryzen AI Max+ 395 / Radeon
8060S, ROCm 10.1.0, HIP 7.16.26385, Go 1.24.4). The Qwen3.5-4B Q4_K_M
GGUF matched SHA-256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
Real-model runs used F16 KV, one active GPU process and automatic
power mode.

- [Native sweep](../../../../bench/hip/prefill_sweep.cpp).
  `yuuka_prefill_sweep --sub128` covers ten projection shapes at
  twelve row counts, including streaming where it is supported.
  The same run profiles each recurrent route as one plan operation.
  `--direct-mmq --sub128` times MMQ separately with prequantized
  activations, including the generic Q6_K cells.
- [Model profile](../../../../engine/gpu/prefill_profile_test.go).
  `YUUKA_TEST_PROFILE_ROWS=1,2,18,64,96,127,128` selects rows in
  `TestGPUProfilePrefill` with HIP events around each operation.
- [Serving benchmark](../../../../bench/serving/main.go).
  `-kv-type f16 -ctx 640 -kv-blocks 72 -prompt-tokens 512 -decode-tokens 2
  -concurrency 1 -warmup 3 -requests 7 -prefix-cache -append-tokens 2`
  requires every extension to reuse exactly the warmup prefix. The
  corresponding 18-token case used `-append-tokens 18`.

All 120 projection cells per sweep ran on the control. Both model
profiles and serving runs exercised the normal HIP plan. The cache
reported one miss and nine hits for each suffix size, with prefill
limited to two or 18 rows. Control TTFT p50 was 91.514 and 68.657 ms
respectively.

### Isolated recurrent operation

Each row cell alternated inline and prepared dispatch, with two
warmups and 15 measured samples per route and process. State was reset
before each timed operation. The values below are medians of three
process medians.

| Rows | Inline (ms) | Prepared (ms) |
| ---: | ---: | ---: |
| 1 | 0.02473 | 0.02577 |
| 2 | 0.02825 | 0.02661 |
| 4 | 0.03450 | 0.02821 |
| 8 | 0.04845 | 0.03238 |
| 16 | 0.07546 | 0.04027 |
| 18 | 0.08231 | 0.04236 |
| 32 | 0.13437 | 0.05626 |
| 48 | 0.18991 | 0.07430 |
| 64 | 0.24858 | 0.09233 |
| 88 | 0.33125 | 0.11942 |
| 96 | 0.36035 | 0.12323 |
| 127 | 0.47640 | 0.16186 |
| 128 | 0.46947 | 0.16366 |

<details>
<summary>Projection control timings by route</summary>

Per-cell times are in microseconds. The first six timings are medians of
three process means. Each process uses two warmups and five timed calls.
Pure MMQ excludes quantization and comes from one process with two
warmups and nine timed calls. A dash marks a route without a measurement.

| Weights | K | N | M | Per-row GEMV | Batched GEMV | Streaming | WMMA | BLAS with dequantization | MMQ with quantization | Pure MMQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2560 | 1024 | 1 | 52.0 | 58.0 | 24.0 | 84.0 | 91.0 | 74.0 | 65.9 |
| Q4_K | 2560 | 1024 | 2 | 90.0 | 66.0 | 36.0 | 93.0 | 89.0 | 75.0 | 67.0 |
| Q4_K | 2560 | 1024 | 4 | 156.0 | 92.0 | 45.0 | 91.0 | 93.0 | 79.0 | 157.2 |
| Q4_K | 2560 | 1024 | 8 | 286.0 | 125.0 | 53.0 | 92.0 | 95.0 | 77.0 | 170.9 |
| Q4_K | 2560 | 1024 | 16 | 561.0 | 195.0 | 94.0 | 106.0 | 95.0 | 79.0 | 88.8 |
| Q4_K | 2560 | 1024 | 18 | 651.0 | — | — | 117.0 | 96.0 | 79.0 | 137.0 |
| Q4_K | 2560 | 1024 | 32 | 1081.0 | — | — | 132.0 | 95.0 | 80.0 | 167.2 |
| Q4_K | 2560 | 1024 | 48 | 1635.0 | — | — | 129.0 | 94.0 | 83.0 | 169.0 |
| Q4_K | 2560 | 1024 | 64 | 2154.0 | — | — | 134.0 | 97.0 | 83.0 | 176.5 |
| Q4_K | 2560 | 1024 | 88 | 3006.0 | — | — | 134.0 | 95.0 | 87.0 | 170.5 |
| Q4_K | 2560 | 1024 | 96 | 3170.0 | — | — | 133.0 | 94.0 | 84.0 | 176.0 |
| Q4_K | 2560 | 1024 | 127 | 4220.0 | — | — | 142.0 | 96.0 | 91.0 | 174.4 |
| Q4_K | 2560 | 4096 | 1 | 174.0 | 177.0 | 37.0 | 113.0 | 148.0 | 160.0 | 154.5 |
| Q4_K | 2560 | 4096 | 2 | 331.0 | 206.0 | 66.0 | 113.0 | 162.0 | 158.0 | 156.1 |
| Q4_K | 2560 | 4096 | 4 | 644.0 | 266.0 | 79.0 | 115.0 | 164.0 | 161.0 | 155.0 |
| Q4_K | 2560 | 4096 | 8 | 1267.0 | 382.0 | 115.0 | 115.0 | 167.0 | 159.0 | 155.6 |
| Q4_K | 2560 | 4096 | 16 | 2512.0 | 618.0 | 169.0 | 125.0 | 170.0 | 162.0 | 157.0 |
| Q4_K | 2560 | 4096 | 18 | 2824.0 | — | — | 134.0 | 162.0 | 164.0 | 155.0 |
| Q4_K | 2560 | 4096 | 32 | 5004.0 | — | — | 153.0 | 165.0 | 166.0 | 159.1 |
| Q4_K | 2560 | 4096 | 48 | 7487.0 | — | — | 170.0 | 170.0 | 165.0 | 160.4 |
| Q4_K | 2560 | 4096 | 64 | 9983.0 | — | — | 247.0 | 166.0 | 170.0 | 156.6 |
| Q4_K | 2560 | 4096 | 88 | 13716.0 | — | — | 290.0 | 186.0 | 173.0 | 165.5 |
| Q4_K | 2560 | 4096 | 96 | 14965.0 | — | — | 288.0 | 181.0 | 173.0 | 166.7 |
| Q4_K | 2560 | 4096 | 127 | 19795.0 | — | — | 298.0 | 175.0 | 179.0 | 171.2 |
| Q4_K | 2560 | 8192 | 1 | 324.0 | 323.0 | 71.0 | 149.0 | 548.0 | 297.0 | 285.1 |
| Q4_K | 2560 | 8192 | 2 | 626.0 | 382.0 | 101.0 | 149.0 | 530.0 | 292.0 | 285.2 |
| Q4_K | 2560 | 8192 | 4 | 1233.0 | 498.0 | 120.0 | 149.0 | 523.0 | 294.0 | 287.0 |
| Q4_K | 2560 | 8192 | 8 | 2447.0 | 730.0 | 156.0 | 154.0 | 528.0 | 291.0 | 285.2 |
| Q4_K | 2560 | 8192 | 16 | 4879.0 | 1199.0 | 255.0 | 164.0 | 528.0 | 294.0 | 289.8 |
| Q4_K | 2560 | 8192 | 18 | 5484.0 | — | — | 166.0 | 535.0 | 298.0 | 293.4 |
| Q4_K | 2560 | 8192 | 32 | 9735.0 | — | — | 182.0 | 538.0 | 311.0 | 295.9 |
| Q4_K | 2560 | 8192 | 48 | 14593.0 | — | — | 313.0 | 553.0 | 309.0 | 302.5 |
| Q4_K | 2560 | 8192 | 64 | 19472.0 | — | — | 308.0 | 581.0 | 314.0 | 306.6 |
| Q4_K | 2560 | 8192 | 88 | 26775.0 | — | — | 568.0 | 684.0 | 325.0 | 320.3 |
| Q4_K | 2560 | 8192 | 96 | 29209.0 | — | — | 584.0 | 680.0 | 331.0 | 315.4 |
| Q4_K | 2560 | 8192 | 127 | 38651.0 | — | — | 671.0 | 694.0 | 339.0 | 327.7 |
| Q4_K | 2560 | 9216 | 1 | 360.0 | 356.0 | 66.0 | 165.0 | 610.0 | 295.0 | 285.5 |
| Q4_K | 2560 | 9216 | 2 | 701.0 | 423.0 | 103.0 | 160.0 | 590.0 | 295.0 | 284.5 |
| Q4_K | 2560 | 9216 | 4 | 1387.0 | 554.0 | 120.0 | 163.0 | 587.0 | 294.0 | 287.1 |
| Q4_K | 2560 | 9216 | 8 | 2748.0 | 809.0 | 159.0 | 166.0 | 595.0 | 293.0 | 287.1 |
| Q4_K | 2560 | 9216 | 16 | 5471.0 | 1335.0 | 259.0 | 175.0 | 592.0 | 298.0 | 289.2 |
| Q4_K | 2560 | 9216 | 18 | 6156.0 | — | — | 178.0 | 601.0 | 296.0 | 293.5 |
| Q4_K | 2560 | 9216 | 32 | 10928.0 | — | — | 194.0 | 601.0 | 302.0 | 296.8 |
| Q4_K | 2560 | 9216 | 48 | 16394.0 | — | — | 332.0 | 616.0 | 310.0 | 306.9 |
| Q4_K | 2560 | 9216 | 64 | 21828.0 | — | — | 360.0 | 641.0 | 315.0 | 300.1 |
| Q4_K | 2560 | 9216 | 88 | 30031.0 | — | — | 611.0 | 669.0 | 327.0 | 325.4 |
| Q4_K | 2560 | 9216 | 96 | 32770.0 | — | — | 627.0 | 773.0 | 332.0 | 327.0 |
| Q4_K | 2560 | 9216 | 127 | 43722.0 | — | — | 681.0 | 772.0 | 342.0 | 340.0 |
| Q4_K | 4096 | 2560 | 1 | 144.0 | 156.0 | 47.0 | 140.0 | 141.0 | 148.0 | 143.3 |
| Q4_K | 4096 | 2560 | 2 | 276.0 | 177.0 | 53.0 | 140.0 | 170.0 | 148.0 | 141.8 |
| Q4_K | 4096 | 2560 | 4 | 533.0 | 226.0 | 69.0 | 140.0 | 171.0 | 146.0 | 142.3 |
| Q4_K | 4096 | 2560 | 8 | 1045.0 | 325.0 | 93.0 | 146.0 | 170.0 | 150.0 | 143.7 |
| Q4_K | 4096 | 2560 | 16 | 2070.0 | 637.0 | 138.0 | 172.0 | 168.0 | 152.0 | 144.6 |
| Q4_K | 4096 | 2560 | 18 | 2327.0 | — | — | 181.0 | 168.0 | 151.0 | 146.0 |
| Q4_K | 4096 | 2560 | 32 | 4118.0 | — | — | 214.0 | 170.0 | 152.0 | 147.3 |
| Q4_K | 4096 | 2560 | 48 | 6170.0 | — | — | 227.0 | 177.0 | 158.0 | 152.6 |
| Q4_K | 4096 | 2560 | 64 | 8218.0 | — | — | 222.0 | 175.0 | 162.0 | 153.9 |
| Q4_K | 4096 | 2560 | 88 | 11286.0 | — | — | 237.0 | 188.0 | 168.0 | 160.4 |
| Q4_K | 4096 | 2560 | 96 | 12311.0 | — | — | 238.0 | 229.0 | 170.0 | 161.7 |
| Q4_K | 4096 | 2560 | 127 | 16288.0 | — | — | 275.0 | 256.0 | 178.0 | 168.0 |
| Q4_K | 9216 | 2560 | 1 | 324.0 | 334.0 | 95.0 | 280.0 | 620.0 | 293.0 | 287.0 |
| Q4_K | 9216 | 2560 | 2 | 628.0 | 372.0 | 110.0 | 285.0 | 597.0 | 295.0 | 289.0 |
| Q4_K | 9216 | 2560 | 4 | 1233.0 | 504.0 | 126.0 | 290.0 | 593.0 | 296.0 | 288.7 |
| Q4_K | 9216 | 2560 | 8 | 2445.0 | 758.0 | 187.0 | 306.0 | 598.0 | 297.0 | 297.0 |
| Q4_K | 9216 | 2560 | 16 | 4864.0 | 1248.0 | 261.0 | 361.0 | 603.0 | 301.0 | 293.2 |
| Q4_K | 9216 | 2560 | 18 | 5486.0 | — | — | 385.0 | 609.0 | 297.0 | 295.1 |
| Q4_K | 9216 | 2560 | 32 | 9709.0 | — | — | 459.0 | 631.0 | 309.0 | 297.2 |
| Q4_K | 9216 | 2560 | 48 | 14578.0 | — | — | 494.0 | 634.0 | 312.0 | 300.3 |
| Q4_K | 9216 | 2560 | 64 | 19411.0 | — | — | 465.0 | 635.0 | 315.0 | 304.7 |
| Q4_K | 9216 | 2560 | 88 | 26700.0 | — | — | 515.0 | 686.0 | 325.0 | 309.8 |
| Q4_K | 9216 | 2560 | 96 | 29146.0 | — | — | 503.0 | 898.0 | 335.0 | 309.5 |
| Q4_K | 9216 | 2560 | 127 | 38599.0 | — | — | 601.0 | 1216.0 | 342.0 | 329.1 |
| Q5_K | 2560 | 8192 | 1 | 366.0 | 348.0 | 80.0 | 157.0 | 557.0 | 305.0 | 294.0 |
| Q5_K | 2560 | 8192 | 2 | 709.0 | 402.0 | 117.0 | 155.0 | 542.0 | 304.0 | 296.0 |
| Q5_K | 2560 | 8192 | 4 | 1402.0 | 514.0 | 136.0 | 153.0 | 540.0 | 304.0 | 293.5 |
| Q5_K | 2560 | 8192 | 8 | 2785.0 | 740.0 | 173.0 | 162.0 | 540.0 | 303.0 | 296.1 |
| Q5_K | 2560 | 8192 | 16 | 5555.0 | 1224.0 | 262.0 | 172.0 | 547.0 | 305.0 | 297.4 |
| Q5_K | 2560 | 8192 | 18 | 6244.0 | — | — | 171.0 | 546.0 | 308.0 | 299.7 |
| Q5_K | 2560 | 8192 | 32 | 11083.0 | — | — | 193.0 | 554.0 | 310.0 | 306.5 |
| Q5_K | 2560 | 8192 | 48 | 16616.0 | — | — | 351.0 | 559.0 | 319.0 | 314.8 |
| Q5_K | 2560 | 8192 | 64 | 22172.0 | — | — | 317.0 | 588.0 | 320.0 | 318.3 |
| Q5_K | 2560 | 8192 | 88 | 30473.0 | — | — | 597.0 | 656.0 | 342.0 | 328.9 |
| Q5_K | 2560 | 8192 | 96 | 33252.0 | — | — | 613.0 | 704.0 | 341.0 | 334.2 |
| Q5_K | 2560 | 8192 | 127 | 43994.0 | — | — | 676.0 | 709.0 | 353.0 | 339.9 |
| Q5_K | 4096 | 2560 | 1 | 188.0 | 179.0 | 51.0 | 149.0 | 141.0 | 153.0 | 143.7 |
| Q5_K | 4096 | 2560 | 2 | 359.0 | 195.0 | 68.0 | 153.0 | 169.0 | 153.0 | 149.3 |
| Q5_K | 4096 | 2560 | 4 | 700.0 | 240.0 | 77.0 | 154.0 | 170.0 | 153.0 | 147.5 |
| Q5_K | 4096 | 2560 | 8 | 1382.0 | 344.0 | 102.0 | 158.0 | 171.0 | 155.0 | 147.9 |
| Q5_K | 4096 | 2560 | 16 | 2745.0 | 647.0 | 144.0 | 184.0 | 171.0 | 155.0 | 148.9 |
| Q5_K | 4096 | 2560 | 18 | 3084.0 | — | — | 195.0 | 173.0 | 154.0 | 149.6 |
| Q5_K | 4096 | 2560 | 32 | 5467.0 | — | — | 229.0 | 175.0 | 160.0 | 151.8 |
| Q5_K | 4096 | 2560 | 48 | 8191.0 | — | — | 241.0 | 171.0 | 163.0 | 156.9 |
| Q5_K | 4096 | 2560 | 64 | 10915.0 | — | — | 230.0 | 175.0 | 167.0 | 158.7 |
| Q5_K | 4096 | 2560 | 88 | 15016.0 | — | — | 250.0 | 188.0 | 173.0 | 163.9 |
| Q5_K | 4096 | 2560 | 96 | 16364.0 | — | — | 253.0 | 234.0 | 173.0 | 164.7 |
| Q5_K | 4096 | 2560 | 127 | 21645.0 | — | — | 294.0 | 255.0 | 179.0 | 170.2 |
| Q6_K | 9216 | 2560 | 1 | 261.0 | 320.0 | 93.0 | 347.0 | 881.0 | 376.0 | 371.0 |
| Q6_K | 9216 | 2560 | 2 | 507.0 | 376.0 | 107.0 | 351.0 | 848.0 | 377.0 | 367.9 |
| Q6_K | 9216 | 2560 | 4 | 1016.0 | 495.0 | 128.0 | 363.0 | 837.0 | 380.0 | 377.4 |
| Q6_K | 9216 | 2560 | 8 | 2012.0 | 728.0 | 167.0 | 380.0 | 856.0 | 379.0 | 367.1 |
| Q6_K | 9216 | 2560 | 16 | 3996.0 | 1231.0 | 255.0 | 424.0 | 850.0 | 383.0 | 371.7 |
| Q6_K | 9216 | 2560 | 18 | 4502.0 | — | — | 456.0 | 854.0 | 391.0 | 380.3 |
| Q6_K | 9216 | 2560 | 32 | 8302.0 | — | — | 529.0 | 874.0 | 392.0 | 391.0 |
| Q6_K | 9216 | 2560 | 48 | 12016.0 | — | — | 604.0 | 879.0 | 404.0 | 386.5 |
| Q6_K | 9216 | 2560 | 64 | 16008.0 | — | — | 556.0 | 892.0 | 408.0 | 395.5 |
| Q6_K | 9216 | 2560 | 88 | 22408.0 | — | — | 699.0 | 955.0 | 421.0 | 397.1 |
| Q6_K | 9216 | 2560 | 96 | 24200.0 | — | — | 654.0 | 1345.0 | 410.0 | 398.1 |
| Q6_K | 9216 | 2560 | 127 | 31992.0 | — | — | 791.0 | 1388.0 | 434.0 | 413.0 |
| Q6_K | 2560 | 1024 | 1 | 166.0 | 57.0 | 23.0 | 102.0 | 94.0 | 98.0 | 95.5 |
| Q6_K | 2560 | 1024 | 2 | 79.0 | 66.0 | 34.0 | 103.0 | 123.0 | 99.0 | 102.4 |
| Q6_K | 2560 | 1024 | 4 | 142.0 | 91.0 | 38.0 | 104.0 | 126.0 | 101.0 | 96.8 |
| Q6_K | 2560 | 1024 | 8 | 266.0 | 122.0 | 53.0 | 106.0 | 127.0 | 95.0 | 106.2 |
| Q6_K | 2560 | 1024 | 16 | 516.0 | 197.0 | 89.0 | 118.0 | 131.0 | 99.0 | 101.3 |
| Q6_K | 2560 | 1024 | 18 | 576.0 | — | — | 128.0 | 124.0 | 101.0 | 111.3 |
| Q6_K | 2560 | 1024 | 32 | 1012.0 | — | — | 146.0 | 132.0 | 102.0 | 93.8 |
| Q6_K | 2560 | 1024 | 48 | 1508.0 | — | — | 140.0 | 126.0 | 102.0 | 94.6 |
| Q6_K | 2560 | 1024 | 64 | 2017.0 | — | — | 147.0 | 130.0 | 106.0 | 94.8 |
| Q6_K | 2560 | 1024 | 88 | 2787.0 | — | — | 152.0 | 130.0 | 108.0 | 98.1 |
| Q6_K | 2560 | 1024 | 96 | 3067.0 | — | — | 156.0 | 129.0 | 110.0 | 98.3 |
| Q6_K | 2560 | 1024 | 127 | 4033.0 | — | — | 160.0 | 134.0 | 110.0 | 101.0 |

</details>

### Model profile control

Medians of three runs at the synthetic dispatch positions below. Each
cell is the sum of timed GPU plan operations in milliseconds. The 512-token
cache-increment benchmark above supplies the populated-prefix timings.

| Rows | Position 0 | Position 2048 | Position 7680 |
| ---: | ---: | ---: | ---: |
| 1 | 78.558 | 86.406 | 107.098 |
| 2 | 87.014 | 94.381 | 115.009 |
| 18 | 63.166 | 71.692 | 92.019 |
| 64 | 86.902 | 94.333 | 115.114 |
| 96 | 114.740 | 121.656 | 145.670 |
| 127 | 132.354 | 136.359 | 159.177 |
| 128 | 73.111 | 79.381 | 98.358 |

Three read-only instrumentation review rounds ended in
`NO_BLOCKING_ISSUES`. The sweep now propagates submission and wait
errors. The profiler synchronizes the plan cache after resizing its
buffers. The real-model run with row order `2,18,2,1,64,96,127,128`
passed. On the CPU fixture, the append benchmark passed under the
race detector at concurrency four, with nine hits and one miss.

## Stage 1c prepared recurrence

The isolated crossover selects two rows. At one row, inline took
0.02473 ms versus 0.02577 ms prepared. Prepared was faster in
every tested cell starting at two rows in all three sweeps. The
[planner](../../../../engine/gpu/plan.go) now uses that threshold
for gfx1151, state width 128, and a sufficiently large prepared
span. The recurrence kernels and projection routes are the same
as the control.

In the real-model 127-row profile at position zero, the median
sum of the 24 recurrent operations went from 12.596 to 4.668 ms.
The candidate run sampled 4.641, 4.722 and 4.668 ms. This meets
the 5.5 ms target. At 128 or more rows the plan still binds the
same prepared route.

The matched-input comparison used the
[prefill comparison driver](../../../../bench/prefill-compare/main.go)
with F16 KV and the Phase 1 token file. It dumped the final FP32
logits and hashed `Session.Snapshot` after each prefill, outside
the timed interval. The latter includes the convolution history
and matrix state for all recurrent layers. The complete logit
files and state digests matched for 1, 2, 17, 64, 127 and 128
tokens and for `512+1` and `512+18` at chunk 512.

Run each Release binary with `-model "$MODEL" -tokens "$TOKENS" -prompt "$N"
-chunk 512 -warmup 0 -reps 1 -logits "$OUT"`. The diagnostic adds the
following code before writing the logit file. Both output files and
the diagnostic stderr lines were compared with `cmp`.

```go
state, err := session.Snapshot()
if err != nil {
    return err
}
fmt.Fprintf(os.Stderr, "state=(%d %x)\n", session.Position(), sha256.Sum256(state))
```

The end-to-end TTFT gates for short prompts, tails and cache increments
are evaluated with the projection changes in Step 1b.

`go build ./...`, `go vet ./...`, `go vet -tags hip ./...`,
`go test -race ./...`, the real-model HIP suite
(`YUUKA_TEST_MODEL="$MODEL" YUUKA_TEST_KV=f16 go test -p 1 -tags hip -v ./...`),
and all ten CTest suites passed on AI. One MoE and two SentencePiece
tests skipped because their separate fixtures were absent.

Two read-only review rounds ended in `NO_BLOCKING_ISSUES`. The roadmap
now assigns the joint end-to-end gates to Step 1b, consistent with the
stage plan.
