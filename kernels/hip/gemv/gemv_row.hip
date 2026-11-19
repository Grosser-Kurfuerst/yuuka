#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

// Remapped decode matvec: one wave per output row, sixteen K values per lane.
//
// The previous streaming kernel tied the grid to the output width in blocks of
// 32 channels and read two bytes of Q4_K weights per lane, so a narrow
// projection launched too few blocks to cover DRAM latency and every lane had
// little data in flight. This kernel keeps the weights packed and changes the
// shape of the mapping instead:
//   - grid = ceil(n_out / (kRowWarps * kRows)): one block covers sixteen output
//     rows, so a 1024-output projection launches 64 blocks instead of 32;
//   - a lane owns sixteen consecutive K values and decodes them from one
//     eight- to sixteen-byte span of the packed row, one whole 32-value
//     sub-block of a K-quant or half of a Q8_0/Q4_0/Q5_0 block;
//   - the lane's sixteen activation values are read once per K step from the
//     activation row and reused for every row the wave handles, so the
//     activation is read from L1 once per warp step and never staged in shared
//     memory, which removes the per-K-step barrier entirely;
//   - the only reduction is one wave shuffle tree per row in the epilogue.
//
// kRows is a compile-time row count per wave so the accumulators stay in
// registers: the block covers kRowWarps * kRows output rows and walks K once.
// The launcher only selects this kernel for a 512-value K step, which every
// decode projection in the supported models satisfies, and for one row block.

namespace {

constexpr int kRowWarps = 4;
constexpr int kRowThreads = 32 * kRowWarps;
constexpr int kSpanValues = 16;              // K values one lane owns per step
constexpr int kStepValues = 32 * kSpanValues; // 512 K values per wave step
constexpr int kRowsPerWave = 2;
constexpr int kRowsPerBlock = kRowWarps * kRowsPerWave;

// load_activation_span reads the lane's sixteen activation values. The
// activation is read once per warp per K step and reused for every row the wave
// handles; it is small enough to stay in L1, which is why the kernel stages
// nothing in shared memory and needs no barrier inside K.
__device__ __forceinline__ void load_activation_span(const float * x, int64_t k,
                                                    float (&xv)[kSpanValues]) {
    #pragma unroll
    for (int i = 0; i < kSpanValues / 4; ++i) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(x + k + 4 * i));
        xv[4 * i + 0] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
    }
}

// gemv_row_kernel is out[row] = dot(W[row], x) + res[row], one wave per row and
// kRows rows per wave.
template <uint32_t DType, int kRows>
__global__ void gemv_row_kernel(float * out, const float * x, const uint8_t * w,
                                const float * res, int64_t n_in, int64_t n_out,
                                int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t steps = n_in / kStepValues;

    float acc[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc[r] = 0.0f;
    }
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k = step * kStepValues + lane * kSpanValues;
        float xv[kSpanValues];
        load_activation_span(x, k, xv);
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int64_t row = first + r * kRowWarps;
            if (row >= n_out) {
                continue;
            }
            float wv[kSpanValues];
            decode_span16<DType>(w + row * row_bytes, k, wv);
            #pragma unroll
            for (int i = 0; i < kSpanValues; ++i) {
                acc[r] = fmaf(wv[i], xv[i], acc[r]);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t row = first + r * kRowWarps;
        const float total = warp_reduce_sum(acc[r]);
        if (lane == 0 && row < n_out) {
            out[row] = res != nullptr ? total + res[row] : total;
        }
    }
}

// gemv_row_pair_kernel walks one activation row against two packed matrices, so
// the gate/up projection reads the activation once. Both matrices must share
// the dtype and row stride; the output widths may differ.
template <uint32_t DType, int kRows, bool kSwiglu>
__global__ void gemv_row_pair_kernel(float * out0, float * out1, const float * x,
                                     const uint8_t * w0, const uint8_t * w1,
                                     int64_t n_in, int64_t n_out0, int64_t n_out1,
                                     int64_t row_bytes) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    const int64_t steps = n_in / kStepValues;

    float acc0[kRows];
    float acc1[kRows];
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
    }
    #pragma unroll 2
    for (int64_t step = 0; step < steps; ++step) {
        const int64_t k = step * kStepValues + lane * kSpanValues;
        float xv[kSpanValues];
        load_activation_span(x, k, xv);
        #pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int64_t row = first + r * kRowWarps;
            if (row < n_out0) {
                float wv[kSpanValues];
                decode_span16<DType>(w0 + row * row_bytes, k, wv);
                #pragma unroll
                for (int i = 0; i < kSpanValues; ++i) {
                    acc0[r] = fmaf(wv[i], xv[i], acc0[r]);
                }
            }
            if (row < n_out1) {
                float wv[kSpanValues];
                decode_span16<DType>(w1 + row * row_bytes, k, wv);
                #pragma unroll
                for (int i = 0; i < kSpanValues; ++i) {
                    acc1[r] = fmaf(wv[i], xv[i], acc1[r]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int64_t row = first + r * kRowWarps;
        const float total0 = warp_reduce_sum(acc0[r]);
        const float total1 = warp_reduce_sum(acc1[r]);
        if (lane != 0) {
            continue;
        }
        if (row < n_out0) {
            out0[row] = kSwiglu
                            ? (total0 / (1.0f + expf(-total0))) * total1
                            : total0;
        }
        if (!kSwiglu && row < n_out1) {
            out1[row] = total1;
        }
    }
}

template <uint32_t DType>
void launch_gemv_row_dtype(float * out, const float * x, const uint8_t * w,
                           const float * res, int64_t n_in, int64_t n_out,
                           int64_t row_bytes, hipStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(
        (n_out + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_row_kernel<DType, kRowsPerWave><<<blocks, kRowThreads, 0, stream>>>(
        out, x, w, res, n_in, n_out, row_bytes);
}

template <uint32_t DType, bool kSwiglu>
void launch_gemv_row_pair_dtype(float * out0, float * out1, const float * x,
                                const uint8_t * w0, const uint8_t * w1, int64_t n_in,
                                int64_t n_out0, int64_t n_out1, int64_t row_bytes,
                                hipStream_t stream) {
    const int64_t widest = n_out0 > n_out1 ? n_out0 : n_out1;
    const unsigned blocks = static_cast<unsigned>(
        (widest + kRowsPerBlock - 1) / kRowsPerBlock);
    gemv_row_pair_kernel<DType, kRowsPerWave, kSwiglu>
        <<<blocks, kRowThreads, 0, stream>>>(out0, out1, x, w0, w1, n_in, n_out0,
                                             n_out1, row_bytes);
}

bool gemv_row_supported(uint32_t dtype, uint64_t n_in) {
    // The kernel walks K in 512-value steps; every decode projection of the
    // supported models satisfies this, and any other width keeps the previous
    // streaming kernel.
    if (n_in % kStepValues != 0) {
        return false;
    }
    switch (dtype) {
        case YUUKA_DTYPE_F32:
        case YUUKA_DTYPE_F16:
        case YUUKA_DTYPE_Q4_0:
        case YUUKA_DTYPE_Q5_0:
        case YUUKA_DTYPE_Q8_0:
        case YUUKA_DTYPE_Q4_K:
        case YUUKA_DTYPE_Q5_K:
        case YUUKA_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

} // namespace

namespace yuuka {

bool gemv_row_available(uint32_t dtype, uint64_t n_in) {
    return gemv_row_supported(dtype, n_in);
}

void launch_gemv_row_f32out(float * out, const float * x, const void * w,
                            const float * res, uint64_t n_in, uint64_t n_out,
                            uint32_t dtype, hipStream_t stream) {
    const auto * weights = static_cast<const uint8_t *>(w);
    const auto in = static_cast<int64_t>(n_in);
    const auto on = static_cast<int64_t>(n_out);
    const auto rb = static_cast<int64_t>(yuuka_packed_row_bytes(dtype, n_in));
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_gemv_row_dtype<YUUKA_DTYPE_F32>(out, x, weights, res, in, on, rb,
                                                   stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_gemv_row_dtype<YUUKA_DTYPE_F16>(out, x, weights, res, in, on, rb,
                                                   stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q4_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q5_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q8_0>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q4_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q5_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
        default:
            launch_gemv_row_dtype<YUUKA_DTYPE_Q6_K>(out, x, weights, res, in, on, rb,
                                                    stream);
            break;
    }
}

void launch_gemv_row_pair_f32out(float * out0, float * out1, const float * x,
                                 const void * w0, const void * w1, uint64_t n_in,
                                 uint64_t n_out0, uint64_t n_out1, uint32_t dtype,
                                 int swiglu, hipStream_t stream) {
    const auto * w0b = static_cast<const uint8_t *>(w0);
    const auto * w1b = static_cast<const uint8_t *>(w1);
    const auto in = static_cast<int64_t>(n_in);
    const auto o0 = static_cast<int64_t>(n_out0);
    const auto o1 = static_cast<int64_t>(n_out1);
    const auto rb = static_cast<int64_t>(yuuka_packed_row_bytes(dtype, n_in));
    const bool sigmoid = swiglu != 0;
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_F32, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_F32, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_F16:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_F16, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_F16, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q8_0, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q8_0, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q4_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q5_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
        default:
            sigmoid ? launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q6_K, true>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream)
                    : launch_gemv_row_pair_dtype<YUUKA_DTYPE_Q6_K, false>(
                          out0, out1, x, w0b, w1b, in, o0, o1, rb, stream);
            break;
    }
}

} // namespace yuuka
