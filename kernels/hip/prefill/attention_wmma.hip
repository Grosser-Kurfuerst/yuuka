#include "../common.cuh"
#include "../kv.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

namespace {

// The WMMA prefill attention kernel tiles a 16-row query block against 16-key
// KV tiles. One workgroup is four waves, one wave per query head of the group
// that shares a KV head, so the staged K/V tile serves all four heads. Each
// wave keeps the whole 16 x head_dim output block in accumulator registers and
// runs the online softmax in the accumulator fragment layout, which keeps the
// per-row max, the rescale factor, and the probability values on the lane that
// owns them.
constexpr int kWmmaHeads = 4;
constexpr int kWmmaQTile = 16;
constexpr int kWmmaKTile = 16;
constexpr int kWmmaThreads = kWmmaHeads * 32;
// A short chunk leaves too few 16-row query tiles to fill the device, so the
// tiled kernel wins below this many rows and at short context. The crossover
// and its measurements are recorded in kernels.md.
constexpr int kWmmaMinRows = 64;
constexpr int kWmmaMinPrefix = 256;
// A 16x16 FP32 accumulator spreads 256 values over one wave, eight per lane.
// Lane l holds element e at output row 2*e + (l >> 4), column l & 15.
constexpr int kWmmaLaneElems = 8;

using frag_a = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half,
                                 rocwmma::row_major>;
using frag_b_col = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::col_major>;
using frag_b_row = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half,
                                     rocwmma::row_major>;
using frag_acc = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float>;

// wmma_paged_row maps a logical KV position to its pool row through the block
// table.
__device__ __forceinline__ int64_t wmma_paged_row(const int32_t * block_table,
                                                 int block_size, int pos) {
    const int block = block_table[pos / block_size];
    return static_cast<int64_t>(block) * block_size + (pos % block_size);
}

// stage_kv_tile gathers one 16-key tile of K and V through the block table and
// writes it as FP16. An F16 pool is copied, an F32 pool is converted, and keys
// outside the tile's causal bound are zeroed. Each key row segment is
// contiguous in the pool, so the copy moves four values per step.
template <uint32_t DType, int HD>
__device__ __forceinline__ void stage_kv_tile(
    __half * kh, __half * vh, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_dim, int kvh, int k0,
    int valid) {
    constexpr int kVec = 4;
    constexpr int kVecs = HD / kVec;
    const int tid = threadIdx.x;
    for (int idx = tid; idx < kWmmaKTile * kVecs; idx += kWmmaThreads) {
        const int kk = idx / kVecs;
        const int d = (idx % kVecs) * kVec;
        __half * kdst = kh + kk * HD + d;
        __half * vdst = vh + kk * HD + d;
        if (kk >= valid) {
            *reinterpret_cast<uint2 *>(kdst) = make_uint2(0u, 0u);
            *reinterpret_cast<uint2 *>(vdst) = make_uint2(0u, 0u);
            continue;
        }
        const int64_t row = wmma_paged_row(block_table, block_size, k0 + kk);
        const int64_t col = static_cast<int64_t>(kvh) * HD + d;
        if constexpr (DType == YUUKA_DTYPE_F16) {
            const __half * ksrc =
                static_cast<const __half *>(k_pool) + row * kv_dim + col;
            const __half * vsrc =
                static_cast<const __half *>(v_pool) + row * kv_dim + col;
            *reinterpret_cast<uint2 *>(kdst) =
                *reinterpret_cast<const uint2 *>(ksrc);
            *reinterpret_cast<uint2 *>(vdst) =
                *reinterpret_cast<const uint2 *>(vsrc);
        } else {
            const float * ksrc =
                static_cast<const float *>(k_pool) + row * kv_dim + col;
            const float * vsrc =
                static_cast<const float *>(v_pool) + row * kv_dim + col;
            const float4 k4 = *reinterpret_cast<const float4 *>(ksrc);
            const float4 v4 = *reinterpret_cast<const float4 *>(vsrc);
            __half2 * kout = reinterpret_cast<__half2 *>(kdst);
            __half2 * vout = reinterpret_cast<__half2 *>(vdst);
            kout[0] = __floats2half2_rn(k4.x, k4.y);
            kout[1] = __floats2half2_rn(k4.z, k4.w);
            vout[0] = __floats2half2_rn(v4.x, v4.y);
            vout[1] = __floats2half2_rn(v4.z, v4.w);
        }
    }
}

// attention_prefill_paged_wmma_kernel is the FP16 MMA paged prefill attention
// path. Q, K, and V are staged as FP16, both products accumulate in FP32, and
// the causal mask is applied inside the tile before the online softmax update.
//
// The score product runs as S^T = K Q^T so that both operands are row major in
// shared memory: K keeps the key-row layout it is gathered in, and Q is stored
// transposed once per workgroup instead of once per key tile. A score lane then
// owns one query row and eight keys, so a row reduction is eight register
// operations and one cross-lane step instead of a four-step butterfly, and the
// running maximum and sum are scalars.
template <uint32_t DType, int HD, bool Dense>
__launch_bounds__(kWmmaThreads, 2)
__global__ void attention_prefill_paged_wmma_kernel(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int heads, int kv_dim,
    int rows, int position, float scale) {
    constexpr int kOutFrags = HD / 16;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int kvh = blockIdx.x;
    const int head = kvh * kWmmaHeads + wave;
    const int q_base = blockIdx.y * kWmmaQTile;
    const int head_stride = Dense ? HD : 2 * HD;
    const int row_stride = heads * head_stride;

    extern __shared__ __align__(16) unsigned char smem[];
    // qh holds Q transposed, [dim][query row] per head, which is the matrix_b
    // row-major layout of the score product.
    __half * qh = reinterpret_cast<__half *>(smem);
    __half * kh = qh + kWmmaHeads * kWmmaQTile * HD;
    __half * vh = kh + kWmmaKTile * HD;
    __half * ph = vh + kWmmaKTile * HD;
    const int ph_stride = kWmmaQTile * kWmmaKTile;

    for (int idx = tid; idx < kWmmaHeads * kWmmaQTile * HD;
         idx += kWmmaThreads) {
        const int h = idx / (kWmmaQTile * HD);
        const int d = (idx / kWmmaQTile) % HD;
        const int r = idx % kWmmaQTile;
        const int q_row = q_base + r;
        float value = 0.0f;
        if (q_row < rows) {
            value = q[static_cast<int64_t>(q_row) * row_stride +
                      (kvh * kWmmaHeads + h) * head_stride + d];
        }
        qh[h * kWmmaQTile * HD + d * kWmmaQTile + r] = __float2half(value);
    }
    __syncthreads();

    frag_acc acc[kOutFrags];
    #pragma unroll
    for (int n = 0; n < kOutFrags; ++n) {
        rocwmma::fill_fragment(acc[n], 0.0f);
    }

    const int lane_key = lane >> 4;
    const int lane_row = lane & 15;
    const int q_last = min(q_base + kWmmaQTile, rows) - 1;
    const int q_pos = position + min(q_base + lane_row, q_last);
    const int max_pos = position + q_last;
    const __half * q_head_ptr = qh + wave * kWmmaQTile * HD;
    __half * p_tile = ph + wave * ph_stride;
    // The softmax runs in base-2 units so an exponent is a single scaled exp.
    const float log2_scale = scale * 1.4426950408889634f;

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int k0 = 0; k0 <= max_pos; k0 += kWmmaKTile) {
        const int valid = min(kWmmaKTile, max_pos + 1 - k0);
        stage_kv_tile<DType, HD>(kh, vh, k_pool, v_pool, block_table, block_size,
                                 kv_dim, kvh, k0, valid);
        __syncthreads();

        // S^T: rows are keys, columns are query rows.
        frag_acc score;
        rocwmma::fill_fragment(score, 0.0f);
        frag_a k_frag;
        frag_b_row q_frag;
        #pragma unroll
        for (int d = 0; d < HD; d += 16) {
            rocwmma::load_matrix_sync(k_frag, kh + d, HD);
            rocwmma::load_matrix_sync(q_frag, q_head_ptr + d * kWmmaQTile,
                                      kWmmaQTile);
            rocwmma::mma_sync(score, k_frag, q_frag, score);
        }

        float prob[kWmmaLaneElems];
        float mx = -INFINITY;
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int key = 2 * e + lane_key;
            float value = score.x[e] * log2_scale;
            if (key >= valid || k0 + key > q_pos) {
                value = -INFINITY;
            }
            prob[e] = value;
            mx = fmaxf(mx, value);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float updated = fmaxf(row_max, mx);
        const float rescale = exp2f(row_max - updated);
        row_max = updated;
        float sum = 0.0f;
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int key = 2 * e + lane_key;
            const float p = exp2f(prob[e] - updated);
            sum += p;
            p_tile[lane_row * kWmmaKTile + key] = __float2half(p);
        }
        sum += __shfl_xor(sum, 16);
        row_sum = row_sum * rescale + sum;
        // The accumulator holds query rows in its own lane mapping, so the
        // rescale factor is fetched from the lane that owns each of those rows.
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const float alpha = __shfl(rescale, 2 * e + lane_key);
            #pragma unroll
            for (int n = 0; n < kOutFrags; ++n) {
                acc[n].x[e] *= alpha;
            }
        }
        __syncwarp();

        frag_a p_frag;
        rocwmma::load_matrix_sync(p_frag, p_tile, kWmmaKTile);
        frag_b_row v_frag;
        #pragma unroll
        for (int n = 0; n < kOutFrags; ++n) {
            rocwmma::load_matrix_sync(v_frag, vh + n * 16, HD);
            rocwmma::mma_sync(acc[n], p_frag, v_frag, acc[n]);
        }
        __syncthreads();
    }

    const float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int n = 0; n < kOutFrags; ++n) {
        #pragma unroll
        for (int e = 0; e < kWmmaLaneElems; ++e) {
            const int row = 2 * e + lane_key;
            const int q_row = q_base + row;
            if (q_row >= rows) {
                continue;
            }
            const int col = n * 16 + lane_row;
            const int64_t out_index = static_cast<int64_t>(q_row) * heads * HD +
                                      static_cast<int64_t>(head) * HD + col;
            float value = acc[n].x[e] * __shfl(inv_sum, row);
            if constexpr (!Dense) {
                const float gate =
                    q[static_cast<int64_t>(q_row) * row_stride +
                      static_cast<int64_t>(head) * head_stride + HD + col];
                value = value / (1.0f + expf(-gate));
            }
            out[out_index] = value;
        }
    }
}

template <uint32_t DType, int HD>
void launch_attention_wmma_dtype(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int heads, int kv_dim,
    int rows, int position, float scale, int dense, dim3 grid, size_t shared,
    hipStream_t stream) {
    if (dense) {
        static const hipError_t attribute = hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                &attention_prefill_paged_wmma_kernel<DType, HD, true>),
            hipFuncAttributeMaxDynamicSharedMemorySize, shared);
        (void)attribute;
        attention_prefill_paged_wmma_kernel<DType, HD, true>
            <<<grid, kWmmaThreads, shared, stream>>>(out, q, k_pool, v_pool,
                                                     block_table, block_size,
                                                     heads, kv_dim, rows,
                                                     position, scale);
    } else {
        static const hipError_t attribute = hipFuncSetAttribute(
            reinterpret_cast<const void *>(
                &attention_prefill_paged_wmma_kernel<DType, HD, false>),
            hipFuncAttributeMaxDynamicSharedMemorySize, shared);
        (void)attribute;
        attention_prefill_paged_wmma_kernel<DType, HD, false>
            <<<grid, kWmmaThreads, shared, stream>>>(out, q, k_pool, v_pool,
                                                     block_table, block_size,
                                                     heads, kv_dim, rows,
                                                     position, scale);
    }
}

}  // namespace

namespace yuuka {

size_t attention_wmma_shared_bytes(int head_dim) {
    const size_t hd = static_cast<size_t>(head_dim);
    return (kWmmaHeads * kWmmaQTile * hd + 2 * kWmmaKTile * hd) * sizeof(__half) +
           kWmmaHeads * kWmmaQTile * kWmmaKTile * sizeof(__half);
}

bool attention_wmma_preferred(int rows, int position) {
    return rows >= kWmmaMinRows || position + rows >= kWmmaMinPrefix;
}

bool attention_wmma_supported(int heads, int kv_heads, int head_dim,
                              int block_size, uint32_t dtype) {
    if (block_size <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return false;
    }
    if (dtype != YUUKA_DTYPE_F32 && dtype != YUUKA_DTYPE_F16) {
        return false;
    }
    if (heads != kv_heads * kWmmaHeads) {
        return false;
    }
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) {
        return false;
    }
    return attention_wmma_shared_bytes(head_dim) <= 64 * 1024;
}

void launch_attention_prefill_paged_wmma_f32(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * block_table, int block_size, int kv_heads, int heads,
    int head_dim, int kv_dim, int rows, int position, float scale, uint32_t dtype,
    int dense, hipStream_t stream) {
    const size_t shared = attention_wmma_shared_bytes(head_dim);
    const dim3 grid(static_cast<unsigned>(kv_heads),
                    static_cast<unsigned>((rows + kWmmaQTile - 1) / kWmmaQTile));
    if (dtype == YUUKA_DTYPE_F16) {
        if (head_dim == 64) {
            launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 64>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        } else if (head_dim == 128) {
            launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 128>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        } else {
            launch_attention_wmma_dtype<YUUKA_DTYPE_F16, 256>(
                out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim,
                rows, position, scale, dense, grid, shared, stream);
        }
        return;
    }
    if (head_dim == 64) {
        launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 64>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, dense, grid, shared, stream);
    } else if (head_dim == 128) {
        launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 128>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, dense, grid, shared, stream);
    } else {
        launch_attention_wmma_dtype<YUUKA_DTYPE_F32, 256>(
            out, q, k_pool, v_pool, block_table, block_size, heads, kv_dim, rows,
            position, scale, dense, grid, shared, stream);
    }
}

}  // namespace yuuka
