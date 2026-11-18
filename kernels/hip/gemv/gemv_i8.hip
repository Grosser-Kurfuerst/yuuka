#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

// Int8 decode matvec.
//
// The streaming decode kernel of lm_head.hip reads 16 bytes of packed weights
// per lane and step and turns them into FP32 with decode_span4, which costs a
// nibble extract, a convert, and an FMA per weight. This kernel keeps the same
// mapping but dots the four packed digits of a lane against the Q8_1 activation
// word with one v_dot4 instruction, and applies the block scale and the Q4_K /
// Q5_K minimum once per sub-block instead of once per weight.
//
// Shape: out[n] = sum_k W[n][k] * x[k], with x quantized to Q8_1 by
// YUUKA_OP_QUANT_Q8_1. One row only: speculative verification keeps the FP32
// streaming kernel, which stores sixteen activation rows in registers per lane,
// and the int8 route pays for itself only where the weight read dominates.
//
// Mapping (identical to lm_head_kernel, minus the shared-memory staging):
//   - one wave owns four output channels and walks the K axis;
//   - a lane owns four consecutive K values of each of its four channels, which
//     lie inside one 32-value scale block, so the block scale and minimum are
//     uniform for the span and one v_dot4 retires four weights;
//   - the packed int8 activation word of a lane is read once per 128-value half
//     and reused for all four channels, so the activation is a 4-byte load per
//     lane per half and needs no staging;
//   - the Q4_K/Q5_K minimum term multiplies the lane's own activation digit sum,
//     which the wave reduction then completes across the whole block;
//   - the Q8_1 block scale of the activation is applied per 32-value sub-block,
//     because the digits are integers and the block already has its own amax.
//
// Arithmetic. value_i = scale * q_i + bias is the same per-weight value
// decode_span4 produces, and the int8 dot is exact in int32, so the only
// difference from the FP32 route is where the scale is applied (per sub-block
// instead of per value) and that the activation is Q8_1 instead of FP32.

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kChansPerWarp = 4;
constexpr int kThreads = 32 * kWarpsPerBlock;
constexpr int kChansPerBlock = kWarpsPerBlock * kChansPerWarp;
constexpr int kHalfK = 128;   // values per half step, 32 lanes of four
constexpr int kValuesPerLane = 4;

// dot4 is the signed int8 dot product of the four digits packed in each
// operand. v_dot4 reads both operands as unsigned by default and each negation
// flag selects the signed reading of its operand: measured on gfx1151,
// (0x01010101, 0xFFFFFFFF) returns 1020 with no flag, -4 with the second flag,
// and -4 with the first flag on (0xFFFFFFFF, 0x01010101). Both digits of a
// Q6_K or Q8_0 row and the activations are signed, so both flags are set.
__device__ __forceinline__ int dot4(int32_t a, int32_t b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, 0, false);
}

// digit_sum returns the sum of the four signed int8 digits packed in a.
__device__ __forceinline__ int digit_sum(int32_t a) {
    return dot4(a, 0x01010101);
}

template <uint32_t DType>
__global__ void gemv_i8_kernel(float * out, const int8_t * qs, const float2 * ds,
                               const uint8_t * w, int64_t n_in, int64_t n_out,
                               int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t ch0 = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) *
                        kChansPerWarp;
    // A wave past the output width still runs the reduction-free path, so the
    // guard is on the work, not on the launch.
    const bool active = ch0 < n_out;
    const int j0 = kValuesPerLane * lane;
    const int64_t steps = n_in / 256;
    const uint8_t * base = active ? w + ch0 * row_bytes : w;
    constexpr bool kHasBias = DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K;

    float acc[kChansPerWarp];
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        acc[c] = 0.0f;
    }

    for (int64_t step = 0; step < steps; ++step) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int64_t k = step * 256 + half * kHalfK + j0;
            int32_t a4 = 0;
            __builtin_memcpy(&a4, qs + k, 4);
            const int a_sum = digit_sum(a4);
            // Every lane's four values lie in one 32-value block, so the block
            // scale of the activation is uniform for the span.
            const float a_scale = ds[k / 32].x;
            if (active) {
                #pragma unroll
                for (int c = 0; c < kChansPerWarp; ++c) {
                    if (ch0 + c >= n_out) {
                        break;
                    }
                    int32_t q = 0;
                    float scale = 0.0f;
                    float bias = 0.0f;
                    decode_span4_i8<DType>(base + c * row_bytes, k, q, scale, bias);
                    acc[c] = fmaf(a_scale * scale, static_cast<float>(dot4(q, a4)), acc[c]);
                    if (kHasBias && bias != 0.0f) {
                        acc[c] = fmaf(a_scale * bias, static_cast<float>(a_sum), acc[c]);
                    }
                }
            }
        }
    }

    if (!active) {
        return;
    }
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        if (ch0 + c >= n_out) {
            break;
        }
        const float total = warp_reduce_sum(acc[c]);
        if (lane == 0) {
            out[ch0 + c] = total;
        }
    }
}

// gemv_i8_fused_kernel quantizes the activation row inside the block instead of
// reading a separately quantized copy. The digits and their block scales are
// staged in shared memory once per block (one byte per value plus one float per
// 32 values), which removes the separate quantization launch the caller would
// otherwise submit for every activation row. The block re-quantizes the row, so
// the work is repeated once per block rather than once per row: measured, the
// repetition is cheaper than the launch it replaces for every shape the decode
// graph dispatches (docs/benchmarks.md).
template <uint32_t DType>
__global__ void gemv_i8_fused_kernel(float * out, const float * x, const uint8_t * w,
                                     int64_t n_in, int64_t n_out, int64_t row_bytes) {
    extern __shared__ __align__(16) uint8_t smem[];
    int8_t * xs8 = reinterpret_cast<int8_t *>(smem);
    float * xs_d = reinterpret_cast<float *>(smem + ((n_in + 15) & ~int64_t(15)));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Stage the activation: one value per lane per 32-value block, so the block
    // amax is one wave reduction and the digits land in value order.
    const int64_t blocks32 = n_in / 32;
    for (int64_t b = warp; b < blocks32; b += kWarpsPerBlock) {
        const int64_t idx = b * 32 + lane;
        const float value = x[idx];
        float amax = fabsf(value);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, offset));
        }
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        const int q = static_cast<int>(
            fmaxf(-127.0f, fminf(127.0f, rintf(value * inv))));
        xs8[idx] = static_cast<int8_t>(q);
        if (lane == 0) {
            xs_d[b] = amax / 127.0f;
        }
    }
    __syncthreads();

    const int64_t ch0 = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) *
                        kChansPerWarp;
    const bool active = ch0 < n_out;
    const int j0 = kValuesPerLane * lane;
    const int64_t steps = n_in / 256;
    const uint8_t * base = active ? w + ch0 * row_bytes : w;
    constexpr bool kHasBias = DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K;

    float acc[kChansPerWarp];
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        acc[c] = 0.0f;
    }

    for (int64_t step = 0; step < steps; ++step) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int64_t k = step * 256 + half * kHalfK + j0;
            int32_t a4 = 0;
            __builtin_memcpy(&a4, xs8 + k, 4);
            const int a_sum = digit_sum(a4);
            const float a_scale = xs_d[k / 32];
            if (active) {
                #pragma unroll
                for (int c = 0; c < kChansPerWarp; ++c) {
                    if (ch0 + c >= n_out) {
                        break;
                    }
                    int32_t q = 0;
                    float scale = 0.0f;
                    float bias = 0.0f;
                    decode_span4_i8<DType>(base + c * row_bytes, k, q, scale, bias);
                    acc[c] = fmaf(a_scale * scale, static_cast<float>(dot4(q, a4)), acc[c]);
                    if (kHasBias && bias != 0.0f) {
                        acc[c] = fmaf(a_scale * bias, static_cast<float>(a_sum), acc[c]);
                    }
                }
            }
        }
    }

    if (!active) {
        return;
    }
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        if (ch0 + c >= n_out) {
            break;
        }
        const float total = warp_reduce_sum(acc[c]);
        if (lane == 0) {
            out[ch0 + c] = total;
        }
    }
}

// row_bytes_for returns the packed row width of one dtype, or zero when the
// dtype has no int8 decode row in this kernel.
uint64_t row_bytes_for(uint32_t dtype, uint64_t n_in) {
    switch (dtype) {
        case YUUKA_DTYPE_Q8_0: return (n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return (n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return (n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return (n_in / 256) * 210;
        default: return 0;
    }
}

template <uint32_t DType>
void launch_gemv_i8_tile(float * out, const int8_t * qs, const float2 * ds,
                         const uint8_t * w, int64_t n_in, int64_t n_out,
                         int64_t row_bytes, hipStream_t stream) {
    const unsigned blocks =
        static_cast<unsigned>((n_out + kChansPerBlock - 1) / kChansPerBlock);
    gemv_i8_kernel<DType><<<blocks, kThreads, 0, stream>>>(out, qs, ds, w, n_in,
                                                           n_out, row_bytes);
}

} // namespace

namespace yuuka {

void launch_gemv_i8_fused_f32out(float * out, const float * x, const void * w,
                                 uint64_t n_in, uint64_t n_out, uint32_t dtype,
                                 hipStream_t stream) {
    if (out == nullptr || x == nullptr || w == nullptr) {
        return;
    }
    if (n_in == 0 || n_out == 0 || n_in % 256 != 0) {
        return;
    }
    const uint64_t row_bytes = row_bytes_for(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * weights = static_cast<const uint8_t *>(w);
    const unsigned blocks =
        static_cast<unsigned>((n_out + kChansPerBlock - 1) / kChansPerBlock);
    const size_t shared =
        static_cast<size_t>((n_in + 15) & ~uint64_t(15)) +
        static_cast<size_t>(n_in / 32) * sizeof(float);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto rb = static_cast<int64_t>(row_bytes);
    switch (dtype) {
        case YUUKA_DTYPE_Q8_0:
            gemv_i8_fused_kernel<YUUKA_DTYPE_Q8_0><<<blocks, kThreads, shared, stream>>>(
                out, x, weights, in, on, rb);
            break;
        case YUUKA_DTYPE_Q4_K:
            gemv_i8_fused_kernel<YUUKA_DTYPE_Q4_K><<<blocks, kThreads, shared, stream>>>(
                out, x, weights, in, on, rb);
            break;
        case YUUKA_DTYPE_Q5_K:
            gemv_i8_fused_kernel<YUUKA_DTYPE_Q5_K><<<blocks, kThreads, shared, stream>>>(
                out, x, weights, in, on, rb);
            break;
        case YUUKA_DTYPE_Q6_K:
            gemv_i8_fused_kernel<YUUKA_DTYPE_Q6_K><<<blocks, kThreads, shared, stream>>>(
                out, x, weights, in, on, rb);
            break;
        default:
            break;
    }
}

void launch_gemv_i8_f32out(float * out, const void * qs, const void * ds,
                           const void * w, uint64_t n_in, uint64_t n_out,
                           uint32_t dtype, hipStream_t stream) {
    if (out == nullptr || qs == nullptr || ds == nullptr || w == nullptr) {
        return;
    }
    if (n_in == 0 || n_out == 0 || n_in % 256 != 0) {
        return;
    }
    const uint64_t row_bytes = row_bytes_for(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto * quantized = static_cast<const int8_t *>(qs);
    const auto * scales = static_cast<const float2 *>(ds);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto rb = static_cast<int64_t>(row_bytes);
    switch (dtype) {
        case YUUKA_DTYPE_Q8_0:
            launch_gemv_i8_tile<YUUKA_DTYPE_Q8_0>(out, quantized, scales, weights, in, on, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_gemv_i8_tile<YUUKA_DTYPE_Q4_K>(out, quantized, scales, weights, in, on, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_gemv_i8_tile<YUUKA_DTYPE_Q5_K>(out, quantized, scales, weights, in, on, rb, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_gemv_i8_tile<YUUKA_DTYPE_Q6_K>(out, quantized, scales, weights, in, on, rb, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
