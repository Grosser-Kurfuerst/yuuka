#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

// Streaming LM head for the prefill graph.
//
// The prefill graph ends with one RMSNorm row against the output matrix, which
// is the widest matrix in the model (vocab x hidden) and is read exactly once
// per prompt chunk. The operator is therefore pure weight bandwidth: the
// single-row GEMV reaches 46 GB/s on gfx1151 because it launches one small
// block per output channel and pays a block reduction per row, while the
// streaming form below keeps every lane busy on 16 bytes of packed weights per
// step and reached 226 GB/s on the same matrix.
//
// Shape: out[m][n] = sum_k W[n][k] * x[m][k] with n in [0, n_out) and m in
// [0, rows). rows is small by construction: the chunk tail needs the last row
// only, and speculative verification needs at most kMaxRows candidate rows.
// Every extra row is nearly free because the weight row is already in
// registers, so the kernel is written for the whole row block instead of one
// row at a time; sixteen rows cost 5.9 ms against 2.3 ms for one on the
// 2560 -> 248320 Q6_K matrix, where sixteen separate GEMV submissions cost
// 181 ms.
//
// Mapping, chosen so that every load instruction consumes a whole 32-byte
// sector of the four-byte lanes it reads:
//   - one wave owns four output channels and walks the K axis itself;
//   - a lane owns four consecutive K values of each of its four channels, so a
//     quantization block is covered by 32 lanes without overlap and the packed
//     reads of the 32 lanes are contiguous;
//   - the activation row block for the current 256-value K step is staged in
//     shared memory once per block, so the activation is read from L2 once per
//     block instead of once per output channel;
//   - a lane accumulates the four values of every channel for every row in
//     registers, and the final wave reduction is one shfl tree per channel.
//
// The kernel keeps no intermediate tile and therefore needs no K-quant
// dequantization buffer: decode_span4 in quant.cuh decodes the four values of a
// lane straight into registers.

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kChansPerWarp = 4;
constexpr int kThreads = 32 * kWarpsPerBlock;
constexpr int kStepK = 256;  // values staged per step, one Q4_K/Q5_K/Q6_K block
constexpr int kMaxRows = 16;  // matches the speculative verification cap

// lm_head_kernel dots KStep-wide K steps of kChansPerWarp packed weight rows
// against up to MaxRows activation rows. MaxRows is a compile-time bound
// because the accumulators must stay in registers: the single-row instantiation
// uses 69 VGPRs and the sixteen-row instantiation 127, where a runtime bound
// would index the accumulator array dynamically and spill it to scratch.
template <uint32_t DType, int MaxRows>
__global__ void lm_head_kernel(float * out, const float * x, const uint8_t * w,
                               int64_t n_in, int64_t n_out, int64_t rows,
                               int64_t row_bytes) {
    extern __shared__ __align__(16) float xs[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t ch0 = (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) *
                        kChansPerWarp;
    // A warp whose channels are all past the output width must still reach the
    // barriers that stage the activation tile, so it skips its own work
    // instead of returning.
    const bool active = ch0 < n_out;
    const int j0 = 4 * lane;
    const int64_t steps = n_in / kStepK;
    // An inactive wave keeps a valid pointer so the guarded decode below never
    // forms one past the weight buffer.
    const uint8_t * base = active ? w + ch0 * row_bytes : w;

    float acc[kChansPerWarp][MaxRows];
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        #pragma unroll
        for (int m = 0; m < MaxRows; ++m) {
            acc[c][m] = 0.0f;
        }
    }

    for (int64_t step = 0; step < steps; ++step) {
        for (int64_t m = 0; m < rows; ++m) {
            const float * src = x + m * n_in + step * kStepK;
            for (int idx = static_cast<int>(threadIdx.x); idx < kStepK;
                 idx += kThreads) {
                xs[m * kStepK + idx] = src[idx];
            }
        }
        __syncthreads();
        // Two 128-value halves per step: a lane covers four consecutive values
        // in each, which is the layout decode_span4 expects.
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int base_j = half * 128 + j0;
            if (active) {
                float wv[kChansPerWarp][4];
                #pragma unroll
                for (int c = 0; c < kChansPerWarp; ++c) {
                    if (ch0 + c >= n_out) {
                        break;
                    }
                    decode_span4<DType>(base + c * row_bytes,
                                        step * kStepK + base_j, wv[c]);
                }
                #pragma unroll
                for (int m = 0; m < MaxRows; ++m) {
                    if (m >= rows) {
                        continue;
                    }
                    const float4 xv = *reinterpret_cast<const float4 *>(
                        xs + m * kStepK + base_j);
                    #pragma unroll
                    for (int c = 0; c < kChansPerWarp; ++c) {
                        if (ch0 + c >= n_out) {
                            break;
                        }
                        acc[c][m] = fmaf(wv[c][0], xv.x, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][1], xv.y, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][2], xv.z, acc[c][m]);
                        acc[c][m] = fmaf(wv[c][3], xv.w, acc[c][m]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!active) {
        return;
    }
    #pragma unroll
    for (int c = 0; c < kChansPerWarp; ++c) {
        if (ch0 + c >= n_out) {
            break;
        }
        #pragma unroll
        for (int m = 0; m < MaxRows; ++m) {
            if (m >= rows) {
                continue;
            }
            const float total = warp_reduce_sum(acc[c][m]);
            if (lane == 0) {
                out[m * n_out + ch0 + c] = total;
            }
        }
    }
}

template <uint32_t DType>
void launch_lm_head_tile(float * out, const float * x, const uint8_t * w,
                         int64_t n_in, int64_t n_out, int64_t rows,
                         int64_t row_bytes, hipStream_t stream) {
    constexpr int kChansPerBlock = kWarpsPerBlock * kChansPerWarp;
    const unsigned blocks =
        static_cast<unsigned>((n_out + kChansPerBlock - 1) / kChansPerBlock);
    const size_t shared = static_cast<size_t>(rows) * kStepK * sizeof(float);
    if (rows == 1) {
        lm_head_kernel<DType, 1><<<blocks, kThreads, shared, stream>>>(
            out, x, w, n_in, n_out, rows, row_bytes);
    } else {
        lm_head_kernel<DType, kMaxRows><<<blocks, kThreads, shared, stream>>>(
            out, x, w, n_in, n_out, rows, row_bytes);
    }
}

// row_bytes_for returns the packed row width of one dtype, or zero when the
// dtype has no packed row in this kernel.
uint64_t row_bytes_for(uint32_t dtype, uint64_t n_in) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return n_in * 4;
        case YUUKA_DTYPE_F16: return n_in * 2;
        case YUUKA_DTYPE_Q4_0: return (n_in / 32) * 18;
        case YUUKA_DTYPE_Q5_0: return (n_in / 32) * 22;
        case YUUKA_DTYPE_Q8_0: return (n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return (n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return (n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return (n_in / 256) * 210;
        default: return 0;
    }
}

} // namespace

namespace yuuka {

void launch_lm_head_f32out(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t rows,
                           uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || rows == 0 || rows > kMaxRows ||
        n_in % kStepK != 0) {
        return;
    }
    const uint64_t row_bytes = row_bytes_for(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto m = static_cast<int64_t>(rows);
    const auto rb = static_cast<int64_t>(row_bytes);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_lm_head_tile<YUUKA_DTYPE_F32>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_lm_head_tile<YUUKA_DTYPE_F16>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q4_0>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q5_0>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_lm_head_tile<YUUKA_DTYPE_Q8_0>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q4_K>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q5_K>(out, x, weights, in, on, m, rb, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_lm_head_tile<YUUKA_DTYPE_Q6_K>(out, x, weights, in, on, m, rb, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
