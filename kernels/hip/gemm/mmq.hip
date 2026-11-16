#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kBlockM = 16;
constexpr int kBlockN = 16;
constexpr int kThreads = kBlockM * kBlockN;

// mmq_weight_i8 returns the signed int8 weight value of one packed element. The
// packed quants are already the integer values the matmul multiplies: 0..15 for
// Q4_K, 0..31 for Q5_K, and -32..31 for Q6_K. No rounding happens here, which is
// why the integer product is exact.
__device__ __forceinline__ int mmq_weight_i8(uint32_t dtype, const uint8_t * row,
                                             int64_t k) {
    if (dtype == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (k / 256) * 144;
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t byte = blk[16 + (sub / 2) * 32 + within];
        return (sub & 1) ? (byte >> 4) : (byte & 0xF);
    }
    if (dtype == YUUKA_DTYPE_Q5_K) {
        const uint8_t * blk = row + (k / 256) * 176;
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t byte = blk[48 + (sub / 2) * 32 + within];
        int q = (sub & 1) ? (byte >> 4) : (byte & 0xF);
        if (blk[16 + within] & static_cast<uint8_t>(1u << sub)) {
            q += 16;
        }
        return q;
    }
    const uint8_t * blk = row + (k / 256) * 210;
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int n = static_cast<int>((k % 256) / 128);
    const int l = static_cast<int>(k & 127);
    const uint8_t * qlN = ql + n * 64;
    const uint8_t * qhN = qh + n * 32;
    int q = 0;
    if (l < 32) {
        q = (qlN[l] & 0xF) | (((qhN[l] >> 0) & 3) << 4);
    } else if (l < 64) {
        q = (qlN[l] & 0xF) | (((qhN[l - 32] >> 2) & 3) << 4);
    } else if (l < 96) {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 64] >> 4) & 3) << 4);
    } else {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 96] >> 6) & 3) << 4);
    }
    return q - 32;
}

// mmq_weight_scale returns the scale and the minimum for the step that starts at
// k. Q4_K and Q5_K carry both per 32-value sub-block; Q6_K carries a scale per
// 16 values and no minimum, so its step is 16 values.
__device__ __forceinline__ void mmq_weight_scale(uint32_t dtype, const uint8_t * row,
                                                 int64_t k, float & scale, float & min) {
    if (dtype == YUUKA_DTYPE_Q6_K) {
        const uint8_t * blk = row + (k / 256) * 210;
        const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
        const float d = load_f16_device(blk + 208);
        scale = d * static_cast<float>(sc[(k % 256) / 16]);
        min = 0.0f;
        return;
    }
    uint8_t s = 0;
    uint8_t m = 0;
    if (dtype == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (k / 256) * 144;
        get_scale_min_k4(static_cast<int>((k % 256) / 32), blk + 4, s, m);
        scale = load_f16_device(blk) * static_cast<float>(s);
        min = load_f16_device(blk + 2) * static_cast<float>(m);
        return;
    }
    const uint8_t * blk = row + (k / 256) * 176;
    get_scale_min_k4(static_cast<int>((k % 256) / 32), blk + 4, s, m);
    scale = load_f16_device(blk) * static_cast<float>(s);
    min = load_f16_device(blk + 2) * static_cast<float>(m);
}

// mmq_kernel computes a 16x16 output tile from two-pass int8 activations and
// packed int8 weights. The integer products accumulate exactly in int32 per
// step, and the step's weight scale, weight minimum, activation scales, and
// activation sums are applied in FP32 afterwards, so the accumulator never
// rounds. Only the activation pair's own reconstruction error remains.
template <uint32_t DType>
__global__ void mmq_kernel(float * out, const int8_t * q1, const int8_t * q2,
                           const float * ascales, const int32_t * asums,
                           const uint8_t * w, int64_t n_in, int64_t n_out,
                           int64_t rows, int64_t act_block_stride,
                           uint64_t row_bytes) {
    const int step = (DType == YUUKA_DTYPE_Q6_K) ? 16 : 32;
    const int tr = threadIdx.x / kBlockN;
    const int tc = threadIdx.x % kBlockN;
    const int64_t row0 = int64_t(blockIdx.y) * kBlockM;
    const int64_t col0 = int64_t(blockIdx.x) * kBlockN;

    __shared__ __align__(16) int8_t a1[kBlockM][32];
    __shared__ __align__(16) int8_t a2[kBlockM][32];
    __shared__ __align__(16) int8_t wq[kBlockN][32];
    __shared__ float wscale[kBlockN];
    __shared__ float wmin[kBlockN];
    __shared__ float as1[kBlockM];
    __shared__ float as2[kBlockM];
    __shared__ int32_t asum1[kBlockM];
    __shared__ int32_t asum2[kBlockM];

    float acc = 0.0f;
    for (int64_t k0 = 0; k0 < n_in; k0 += step) {
        const int64_t blk = k0 / 32;
        for (int idx = threadIdx.x; idx < kBlockM * step; idx += kThreads) {
            const int r = idx / step;
            const int k = idx % step;
            const int64_t gr = row0 + r;
            const int64_t gk = k0 + k;
            if (gr < rows) {
                const int64_t pos = gr * n_in + gk;
                a1[r][k] = q1[pos];
                a2[r][k] = q2[pos];
            } else {
                a1[r][k] = 0;
                a2[r][k] = 0;
            }
        }
        for (int idx = threadIdx.x; idx < kBlockN * step; idx += kThreads) {
            const int c = idx / step;
            const int k = idx % step;
            const int64_t gc = col0 + c;
            wq[c][k] = gc < n_out
                           ? static_cast<int8_t>(mmq_weight_i8(DType, w + gc * row_bytes,
                                                               k0 + k))
                           : static_cast<int8_t>(0);
        }
        if (threadIdx.x < kBlockN) {
            const int64_t gc = col0 + threadIdx.x;
            float scale = 0.0f;
            float min = 0.0f;
            if (gc < n_out) {
                mmq_weight_scale(DType, w + gc * row_bytes, k0, scale, min);
            }
            wscale[threadIdx.x] = scale;
            wmin[threadIdx.x] = min;
        }
        if (threadIdx.x < kBlockM) {
            const int64_t gr = row0 + threadIdx.x;
            const int64_t pos = gr * act_block_stride + blk;
            if (gr < rows) {
                as1[threadIdx.x] = ascales[pos * 2];
                as2[threadIdx.x] = ascales[pos * 2 + 1];
            } else {
                as1[threadIdx.x] = 0.0f;
                as2[threadIdx.x] = 0.0f;
            }
            if (DType != YUUKA_DTYPE_Q6_K) {
                if (gr < rows) {
                    asum1[threadIdx.x] = asums[pos * 2];
                    asum2[threadIdx.x] = asums[pos * 2 + 1];
                } else {
                    asum1[threadIdx.x] = 0;
                    asum2[threadIdx.x] = 0;
                }
            }
        }
        __syncthreads();
        // gfx1151 has no int8 dot instruction (v_dot4_i32_i8 needs the CDNA
        // dot1-insts feature), so this widens to int32 and multiplies. That is
        // why the int8 route loses on speed even though its accumulation is
        // exact; docs/benchmarks.md records the measurement.
        int p1 = 0;
        int p2 = 0;
        #pragma unroll 4
        for (int k = 0; k < step; ++k) {
            const int wv = static_cast<int>(wq[tc][k]);
            p1 += wv * static_cast<int>(a1[tr][k]);
            p2 += wv * static_cast<int>(a2[tr][k]);
        }
        acc += wscale[tc] * (as1[tr] * static_cast<float>(p1) +
                             as2[tr] * static_cast<float>(p2));
        if (DType != YUUKA_DTYPE_Q6_K) {
            acc -= wmin[tc] * (as1[tr] * static_cast<float>(asum1[tr]) +
                               as2[tr] * static_cast<float>(asum2[tr]));
        }
        __syncthreads();
    }
    const int64_t gr = row0 + tr;
    const int64_t gc = col0 + tc;
    if (gr < rows && gc < n_out) {
        out[gr * n_out + gc] = acc;
    }
}

} // namespace

namespace yuuka {

void launch_mmq_gemm_f32out(float * out, const int8_t * q1, const int8_t * q2,
                            const float * ascales, const int32_t * asums,
                            const uint8_t * w, int64_t n_in, int64_t n_out,
                            int64_t rows, uint64_t row_bytes, uint32_t dtype,
                            hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || rows == 0 || n_in % 32 != 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + kBlockM - 1) / kBlockM));
    const int64_t act_block_stride = n_in / 32;
    switch (dtype) {
        case YUUKA_DTYPE_Q4_K:
            mmq_kernel<YUUKA_DTYPE_Q4_K><<<grid, kThreads, 0, stream>>>(
                out, q1, q2, ascales, asums, w, n_in, n_out, rows,
                act_block_stride, row_bytes);
            break;
        case YUUKA_DTYPE_Q5_K:
            mmq_kernel<YUUKA_DTYPE_Q5_K><<<grid, kThreads, 0, stream>>>(
                out, q1, q2, ascales, asums, w, n_in, n_out, rows,
                act_block_stride, row_bytes);
            break;
        case YUUKA_DTYPE_Q6_K:
            mmq_kernel<YUUKA_DTYPE_Q6_K><<<grid, kThreads, 0, stream>>>(
                out, q1, q2, ascales, asums, w, n_in, n_out, rows,
                act_block_stride, row_bytes);
            break;
        default:
            break;
    }
}

} // namespace yuuka
