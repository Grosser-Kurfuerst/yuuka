#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

// Int8 MMQ prefill GEMM and its Q8_1 activation quantizer.
//
// The design follows llama.cpp's MMQ approach (MIT, commit b1b770d90) at the
// algorithmic level: activations are quantized to int8 with one FP32 scale and
// one FP32 block sum per 32 values, packed K-quant weights stay quantized and
// are decoded to int8, and the 16x16x16 int8 WMMA accumulates exact int32
// products. The 32-value weight scale is applied after each pair of 16-wide
// slices, which is exact because the int32 partial sum is exact and the scale
// is constant across the block. The Q4_K/Q5_K min term becomes a per-row
// weight minimum dotted with the activation block sum.
//
// The implementation here is independent: staging, tile shape, scale layout,
// and epilogue were written for this repository and validated against the CPU
// reference in tests/hip/prefill_test.cpp.

namespace {

constexpr int kTileI = 64;   // output channels (weight rows)
constexpr int kTileJ = 128;  // tokens
constexpr int kStepK = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kSpan = 8;
constexpr int kSpansPerRow = kStepK / kSpan;
constexpr int kBlock32 = kStepK / 32;
// Row stride of the staged operand tiles. A 256-byte stride makes every tile
// row start at the same shared-memory bank, and the 16x16 int8 fragment loads
// then conflict 16 ways: measured 12.5 TF/s against 48.1 TF/s at 272 bytes. The
// padding adds 1 KiB per tile and keeps the stride a multiple of the 16-byte
// fragment granularity.
constexpr int kTileStride = 272;
// Epilogue row stride. Padding the token-major staging tile moves the lane
// stores in the accumulator epilogue off a single shared-memory bank.
constexpr int kEpilogueStride = kTileI + 1;

// One dynamic shared-memory allocation, sliced by offset. A and B hold the
// staged int8 operands for the current K step, and the scale tiles carry the
// weight and activation scales for the same step.
constexpr int kABytes = kTileI * kTileStride;
constexpr int kBBytes = kTileJ * kTileStride;
constexpr int kAScaleOffset = kABytes + kBBytes;
constexpr int kAScaleBytes = 16 * kTileI * sizeof(float);
constexpr int kBScaleOffset = kAScaleOffset + kAScaleBytes;
constexpr int kBScaleBytes = kBlock32 * kTileJ * sizeof(float2);
constexpr int kSharedBytes = kBScaleOffset + kBScaleBytes;
static_assert(kSharedBytes == 64512, "unexpected shared memory budget");
static_assert(kSharedBytes <= 64 * 1024, "exceeds the gfx1151 LDS budget");
static_assert(kTileStride % 16 == 0, "fragment loads need 16-byte rows");
static_assert(kTileJ * kEpilogueStride * int(sizeof(float)) <= kABytes + kBBytes,
              "epilogue tile must fit the operand staging area");

// pack4 packs four values in [-128, 127] into one 32-bit word in little-endian
// byte order, which is the int8 order of the staged operand tile.
__device__ __forceinline__ int32_t pack4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// q6k_span8 decodes eight consecutive Q6_K values starting at a multiple of 8
// inside one 256-value block. The eight values lie inside one 32-value region,
// so the ql nibble and qh bit positions are hoisted once per span and each
// 16-bit load covers two values. Block starts are only 2-byte aligned, so the
// words are assembled from 16-bit halves.
__device__ __forceinline__ void q6k_span8(const uint8_t * blk, int j, int8_t * dst) {
    const int n = j / 128;
    const int l = j & 127;
    const uint8_t * ql = blk + n * 64;
    const uint8_t * qh = blk + 128 + n * 32;
    const uint8_t * ql_src = ql;
    const uint8_t * qh_src = qh;
    int ql_shift = 0;
    int qh_shift = 0;
    if (l < 32) {
        ql_src = ql + l;
        qh_src = qh + l;
    } else if (l < 64) {
        ql_src = ql + l;
        qh_src = qh + (l - 32);
        qh_shift = 2;
    } else if (l < 96) {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 64);
        qh_shift = 4;
    } else {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 96);
        qh_shift = 6;
    }
    int32_t lo = 0;
    int32_t hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t ql_word =
            *reinterpret_cast<const uint16_t *>(ql_src + 2 * i);
        const uint32_t qh_word =
            *reinterpret_cast<const uint16_t *>(qh_src + 2 * i);
        const int q0 = static_cast<int>((ql_word >> ql_shift) & 0xF) |
                       (static_cast<int>((qh_word >> qh_shift) & 3) << 4);
        const int q1 = static_cast<int>((ql_word >> (8 + ql_shift)) & 0xF) |
                       (static_cast<int>((qh_word >> (8 + qh_shift)) & 3) << 4);
        const int32_t word = pack4(q0 - 32, q1 - 32, 0, 0);
        if (i < 2) {
            lo |= word << (16 * i);
        } else {
            hi |= word << (16 * (i - 2));
        }
    }
    *reinterpret_cast<int32_t *>(dst) = lo;
    *reinterpret_cast<int32_t *>(dst + 4) = hi;
}

// stage_weight_span decodes eight consecutive weights of one packed row into
// int8. The A tile keeps the raw quantized values, so the integer WMMA sees the
// same digits the dequantized reference path sees; the block scale and the
// Q4_K/Q5_K minimum are staged separately.
template <uint32_t DType>
__device__ __forceinline__ void stage_weight_span(int8_t * dst, const uint8_t * row,
                                                  int64_t k, float * scale_a,
                                                  int row_index) {
    if (DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const int block_bytes = (DType == YUUKA_DTYPE_Q4_K) ? 144 : 176;
        const uint8_t * blk = row + (k / 256) * block_bytes;
        const uint8_t * ql_base = blk + (DType == YUUKA_DTYPE_Q4_K ? 16 : 48);
        const uint8_t * ql = ql_base + (sub / 2) * 32 + within;
        const bool high = (sub & 1) != 0;
        const int shift = high ? 4 : 0;
        const uint2 packed_ql = *reinterpret_cast<const uint2 *>(ql);
        int q[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = static_cast<int>((packed_ql.x >> (8 * i + shift)) & 0xF);
            q[i + 4] = static_cast<int>((packed_ql.y >> (8 * i + shift)) & 0xF);
        }
        if (DType == YUUKA_DTYPE_Q5_K) {
            const uint2 packed_qh =
                *reinterpret_cast<const uint2 *>(blk + 16 + within);
            const uint32_t mask = 1u << sub;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                if ((packed_qh.x >> (8 * i)) & mask) {
                    q[i] += 16;
                }
                if ((packed_qh.y >> (8 * i)) & mask) {
                    q[i + 4] += 16;
                }
            }
        }
        *reinterpret_cast<int32_t *>(dst) = pack4(q[0], q[1], q[2], q[3]);
        *reinterpret_cast<int32_t *>(dst + 4) = pack4(q[4], q[5], q[6], q[7]);
        if (within == 0) {
            uint8_t sc = 0;
            uint8_t m = 0;
            get_scale_min_k4(sub, blk + 4, sc, m);
            const float d = load_f16_device(blk);
            const float dm = load_f16_device(blk + 2);
            scale_a[(2 * sub) * kTileI + row_index] = d * static_cast<float>(sc);
            scale_a[(2 * sub + 1) * kTileI + row_index] =
                -(dm * static_cast<float>(m));
        }
        return;
    }
    // Q6_K: one scale per 16 values and no minimum.
    const int j = static_cast<int>(k % 256);
    const uint8_t * blk = row + (k / 256) * 210;
    q6k_span8(blk, j, dst);
    if ((j & 15) == 0) {
        const int n = j / 128;
        const int l = j & 127;
        int sidx = 0;
        if (l < 32) {
            sidx = l / 16;
        } else if (l < 64) {
            sidx = (l - 32) / 16 + 2;
        } else if (l < 96) {
            sidx = (l - 64) / 16 + 4;
        } else {
            sidx = (l - 96) / 16 + 6;
        }
        const int8_t sc = reinterpret_cast<const int8_t *>(blk + 192)[n * 8 + sidx];
        scale_a[(j / 16) * kTileI + row_index] =
            load_f16_device(blk + 208) * static_cast<float>(sc);
    }
}

// stage_weights decodes the kTileI x kStepK weight tile. Thread t owns spans
// t, t + kThreads, ... so each thread writes eight spans of eight values.
template <uint32_t DType>
__device__ __forceinline__ void stage_weights(int8_t * tile_a, float * scale_a,
                                              const uint8_t * w, uint64_t row_bytes,
                                              int64_t n0, int64_t n_out, int64_t k0) {
    constexpr int kSpansPerThread = (kTileI * kSpansPerRow) / kThreads;
    #pragma unroll
    for (int i = 0; i < kSpansPerThread; ++i) {
        const int span_index = threadIdx.x + i * kThreads;
        const int row = span_index / kSpansPerRow;
        const int span = span_index % kSpansPerRow;
        int8_t * dst = tile_a + row * kTileStride + span * kSpan;
        const int64_t gn = n0 + row;
        if (gn >= n_out) {
            *reinterpret_cast<int32_t *>(dst) = 0;
            *reinterpret_cast<int32_t *>(dst + 4) = 0;
            continue;
        }
        stage_weight_span<DType>(dst, w + gn * row_bytes, k0 + span * kSpan,
                                 scale_a, row);
    }
}

// stage_activations copies the Q8_1 activation tile and its block scales. Rows
// beyond the batch are zeroed, which also zeroes their scale and sum, so the
// epilogue masks them instead of biasing the tile.
__device__ __forceinline__ void stage_activations(int8_t * tile_b, float2 * scale_b,
                                                  const int8_t * xs, const float2 * xds,
                                                  int64_t m0, int64_t rows,
                                                  int64_t n_in, int64_t k0) {
    constexpr int kWordsPerRow = kStepK / 16;
    constexpr int kWords = kTileJ * kWordsPerRow;
    for (int idx = threadIdx.x; idx < kWords; idx += kThreads) {
        const int row = idx / kWordsPerRow;
        const int col = (idx % kWordsPerRow) * 16;
        const int64_t gm = m0 + row;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (gm < rows) {
            value = *reinterpret_cast<const uint4 *>(xs + gm * n_in + k0 + col);
        }
        *reinterpret_cast<uint4 *>(tile_b + row * kTileStride + col) = value;
    }
    constexpr int kScaleCount = kTileJ * kBlock32;
    for (int idx = threadIdx.x; idx < kScaleCount; idx += kThreads) {
        const int row = idx / kBlock32;
        const int block = idx % kBlock32;
        const int64_t gm = m0 + row;
        float2 value = make_float2(0.0f, 0.0f);
        if (gm < rows) {
            value = xds[gm * (n_in / 32) + k0 / 32 + block];
        }
        scale_b[block * kTileJ + row] = value;
    }
}

// quantize_q8_1_kernel writes Q8_1 activations: one warp per 128 contiguous
// values, four values per lane. Each 32-value block keeps its amax-derived FP32
// scale and the scaled integer sum, which is what the Q4_K/Q5_K minimum term
// needs. n is a multiple of 128, so a warp never crosses a row.
__global__ void quantize_q8_1_kernel(int8_t * qs, float2 * ds, const float * x,
                                     int64_t n, int64_t warps) {
    const int64_t warp =
        static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
    if (warp >= warps) {
        return;
    }
    const int lane = threadIdx.x % 32;
    const int64_t warps_per_row = n / 128;
    const int64_t row = warp / warps_per_row;
    const int64_t col0 = (warp % warps_per_row) * 128 + lane * 4;
    const float4 value = *reinterpret_cast<const float4 *>(x + row * n + col0);
    float amax = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                       fmaxf(fabsf(value.z), fabsf(value.w)));
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, offset));
    }
    const float d = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.x * inv))));
    const int q1 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.y * inv))));
    const int q2 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.z * inv))));
    const int q3 = static_cast<int>(fmaxf(-127.0f, fminf(127.0f, rintf(value.w * inv))));
    *reinterpret_cast<int32_t *>(qs + row * n + col0) = pack4(q0, q1, q2, q3);
    float sum = static_cast<float>(q0 + q1 + q2 + q3);
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        sum += __shfl_xor_sync(0xffffffffffffffffull, sum, offset);
    }
    if ((lane & 7) == 0) {
        ds[row * (n / 32) + col0 / 32] = make_float2(d, d * sum);
    }
}

// gemm_mmq_kernel computes a kTileI x kTileJ output tile with eight waves. Each
// wave owns one 16-row weight fragment and four 16-column activation fragments,
// so a wave issues four 16x16x16 int8 MMAs per 16-value K slice. The int32
// accumulator mapping was measured on gfx1151 against the CPU product:
// lane l element e holds output row 2*e + l/16 of column l%16 inside the
// 16x16 tile. Consequently one activation scale per lane (its column) and eight
// weight scales per lane (its rows) are applied per block.
template <uint32_t DType>
__global__ void gemm_mmq_kernel(float * out, const int8_t * xs, const float2 * xds,
                                const uint8_t * w, int64_t n_in, int64_t n_out,
                                int64_t rows, uint64_t row_bytes) {
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t * tile_a = reinterpret_cast<int8_t *>(smem);
    int8_t * tile_b = reinterpret_cast<int8_t *>(smem + kABytes);
    float * scale_a = reinterpret_cast<float *>(smem + kAScaleOffset);
    float2 * scale_b = reinterpret_cast<float2 *>(smem + kBScaleOffset);
    float * epilogue = reinterpret_cast<float *>(smem);

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int row_base = (warp & 3) * 16;
    const int col_base = (warp / 4) * 64;

    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kTileI;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kTileJ;

    // Q6_K carries one scale per 16 values and Q4_K/Q5_K one per 32 values, so
    // the slice group that shares a scale differs by dtype.
    constexpr bool kScalePer16 = (DType == YUUKA_DTYPE_Q6_K);
    constexpr int kSlicesPerBlock = kScalePer16 ? 1 : 2;
    constexpr int kBlocksPerStep = kStepK / (kScalePer16 ? 16 : 32);

    float sum[4][8];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            sum[t][e] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < n_in; k0 += kStepK) {
        stage_weights<DType>(tile_a, scale_a, w, row_bytes, n0, n_out, k0);
        stage_activations(tile_b, scale_b, xs, xds, m0, rows, n_in, k0);
        __syncthreads();
        #pragma unroll
        for (int block = 0; block < kBlocksPerStep; ++block) {
            rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t> c[4];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                rocwmma::fill_fragment(c[t], 0);
            }
            #pragma unroll
            for (int slice = 0; slice < kSlicesPerBlock; ++slice) {
                const int step = (block * kSlicesPerBlock + slice) * 16;
                rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t,
                                  rocwmma::row_major> a;
                rocwmma::load_matrix_sync(a, tile_a + row_base * kTileStride + step,
                                          kTileStride);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t,
                                      rocwmma::col_major> b;
                    rocwmma::load_matrix_sync(
                        b, tile_b + (col_base + t * 16) * kTileStride + step,
                        kTileStride);
                    rocwmma::mma_sync(c[t], a, b, c[t]);
                }
            }
            // The weight scales belong to this lane's eight rows, so they are
            // loaded once per block and reused by all four accumulator tiles.
            // Q4_K/Q5_K additionally carry a per-row minimum that the activation
            // block sum multiplies; Q6_K has no minimum.
            const int b32 = kScalePer16 ? (block >> 1) : block;
            const int slot0 = kScalePer16 ? block : 2 * block;
            float wd[8];
            float wm[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = row_base + 2 * e + (lane >> 4);
                wd[e] = scale_a[slot0 * kTileI + row];
                wm[e] = kScalePer16 ? 0.0f : scale_a[(slot0 + 1) * kTileI + row];
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const float2 dsb =
                    scale_b[b32 * kTileJ + col_base + t * 16 + (lane & 15)];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sum[t][e] = fmaf(wd[e] * dsb.x, static_cast<float>(c[t].x[e]),
                                     fmaf(wm[e], dsb.y, sum[t][e]));
                }
            }
        }
        __syncthreads();
    }

    // The epilogue aliases the operand staging area: stage token-major, then
    // write the tile back coalesced over the output channel.
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = row_base + 2 * e + (lane >> 4);
            const int col = col_base + t * 16 + (lane & 15);
            epilogue[col * kEpilogueStride + row] = sum[t][e];
        }
    }
    __syncthreads();
    for (int idx = tid; idx < kTileJ * kTileI; idx += kThreads) {
        const int col = idx / kTileI;
        const int row = idx % kTileI;
        const int64_t gm = m0 + col;
        const int64_t gn = n0 + row;
        if (gm < rows && gn < n_out) {
            out[gm * n_out + gn] = epilogue[col * kEpilogueStride + row];
        }
    }
}

template <uint32_t DType>
void launch_mmq_tile(float * out, const int8_t * xs, const float2 * xds,
                     const uint8_t * w, int64_t n_in, int64_t n_out, int64_t rows,
                     uint64_t row_bytes, hipStream_t stream) {
    static const hipError_t attribute = hipFuncSetAttribute(
        reinterpret_cast<const void *>(&gemm_mmq_kernel<DType>),
        hipFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    // A rejected attribute makes the launch below fail with a clear device
    // error through the caller's hipGetLastError check; nothing runs silently.
    (void)attribute;
    const dim3 grid(static_cast<unsigned>((n_out + kTileI - 1) / kTileI),
                    static_cast<unsigned>((rows + kTileJ - 1) / kTileJ));
    gemm_mmq_kernel<DType><<<grid, kThreads, kSharedBytes, stream>>>(
        out, xs, xds, w, n_in, n_out, rows, row_bytes);
}

} // namespace

namespace yuuka {

void launch_quantize_q8_1(void * qs, void * ds, const float * x, uint64_t n,
                          uint64_t batch, hipStream_t stream) {
    if (n == 0 || batch == 0 || n % 128 != 0) {
        return;
    }
    const int64_t warps = static_cast<int64_t>((n * batch) / 128);
    const int64_t blocks = (warps + kWarps - 1) / kWarps;
    quantize_q8_1_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        static_cast<int8_t *>(qs), static_cast<float2 *>(ds), x,
        static_cast<int64_t>(n), warps);
}

void launch_gemm_mmq_f32out(float * out, const void * qs, const void * ds,
                            const void * w, uint64_t n_in, uint64_t n_out,
                            uint64_t batch, uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0 || n_in % kStepK != 0) {
        return;
    }
    const uint64_t block_bytes =
        dtype == YUUKA_DTYPE_Q4_K    ? 144u
        : dtype == YUUKA_DTYPE_Q5_K  ? 176u
        : dtype == YUUKA_DTYPE_Q6_K  ? 210u
                                     : 0u;
    if (block_bytes == 0) {
        return;
    }
    const uint64_t row_bytes = (n_in / 256) * block_bytes;
    const auto * activations = static_cast<const int8_t *>(qs);
    const auto * scales = static_cast<const float2 *>(ds);
    const auto * weights = static_cast<const uint8_t *>(w);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    const int64_t m = static_cast<int64_t>(batch);
    switch (dtype) {
        case YUUKA_DTYPE_Q4_K:
            launch_mmq_tile<YUUKA_DTYPE_Q4_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_mmq_tile<YUUKA_DTYPE_Q5_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_mmq_tile<YUUKA_DTYPE_Q6_K>(out, activations, scales, weights, in,
                                             on, m, row_bytes, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
