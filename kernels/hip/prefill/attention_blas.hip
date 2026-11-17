#include "../kv.cuh"
#include "../yuuka_kernels.h"

#include <rocblas/rocblas.h>
#include <mutex>

namespace {

constexpr int kThreads = 256;

rocblas_handle attention_blas_handle() {
    static rocblas_handle handle = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        if (rocblas_create_handle(&handle) != rocblas_status_success) handle = nullptr;
    });
    return handle;
}

__global__ void pack_query_f16(__half * dst, const float * q,
                               int rows, int heads, int head_dim) {
    const int64_t total = static_cast<int64_t>(rows) * heads * head_dim;
    for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
         idx < total; idx += gridDim.x * int64_t(blockDim.x)) {
        const int d = idx % head_dim;
        const int row = (idx / head_dim) % rows;
        const int head = idx / (int64_t(rows) * head_dim);
        dst[idx] = __float2half(q[(int64_t(row) * heads + head) * (2 * head_dim) + d]);
    }
}

__global__ void gather_kv_f16(__half * k_dst, __half * v_dst,
                              const __half * k_pool, const __half * v_pool,
                              const int32_t * table, int block_size, int ctx,
                              int kvheads, int head_dim) {
    const int kv_dim = kvheads * head_dim;
    const int64_t total = static_cast<int64_t>(ctx) * kv_dim;
    for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
         idx < total; idx += gridDim.x * int64_t(blockDim.x)) {
        const int d = idx % head_dim;
        const int p = (idx / head_dim) % ctx;
        const int head = idx / (int64_t(ctx) * head_dim);
        const int64_t row = int64_t(table[p / block_size]) * block_size + p % block_size;
        const int64_t src = row * kv_dim + head * head_dim + d;
        k_dst[idx] = k_pool[src];
        v_dst[idx] = v_pool[src];
    }
}

// One full query row is reduced by one workgroup. Scores are scaled and
// causally masked, then normalized in FP32 before being stored as FP16.
__global__ void masked_softmax_f16(const float * scores, __half * probabilities, int rows, int heads,
                                   int ctx, int position, float scale) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    (void)heads;
    const int causal_end = position + r;
    __shared__ float reduced[kThreads];
    const float * score = scores + (int64_t(h) * rows + r) * ctx;
    __half * prob = probabilities + (int64_t(h) * rows + r) * ctx;
    float mx = -INFINITY;
    for (int p = tid; p < ctx; p += kThreads) {
        if (p <= causal_end) mx = fmaxf(mx, score[p] * scale);
    }
    reduced[tid] = mx;
    __syncthreads();
    for (int half = kThreads / 2; half > 0; half >>= 1) {
        if (tid < half) reduced[tid] = fmaxf(reduced[tid], reduced[tid + half]);
        __syncthreads();
    }
    const float max_value = reduced[0];
    float total = 0.0f;
    for (int p = tid; p < ctx; p += kThreads) {
        if (p <= causal_end) total += expf(score[p] * scale - max_value);
    }
    reduced[tid] = total;
    __syncthreads();
    for (int half = kThreads / 2; half > 0; half >>= 1) {
        if (tid < half) reduced[tid] += reduced[tid + half];
        __syncthreads();
    }
    const float inv = 1.0f / reduced[0];
    for (int p = tid; p < ctx; p += kThreads) {
        const float probability = p <= causal_end ?
            expf(score[p] * scale - max_value) * inv : 0.0f;
        prob[p] = __float2half(probability);
    }
}

__global__ void unpack_attn_f16(float * out, const float * src, const float * q,
                                int rows, int heads, int head_dim) {
    const int64_t total = int64_t(rows) * heads * head_dim;
    for (int64_t idx = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
         idx < total; idx += gridDim.x * int64_t(blockDim.x)) {
        const int d = idx % head_dim;
        const int h = (idx / head_dim) % heads;
        const int r = idx / (int64_t(heads) * head_dim);
        const int64_t s = (int64_t(h) * rows + r) * head_dim + d;
        const float gate = q[(int64_t(r) * heads + h) * (2 * head_dim) + head_dim + d];
        out[idx] = src[s] / (1.0f + expf(-gate));
    }
}

}  // namespace

namespace yuuka {

int launch_attention_prefill_paged_blas_f16(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * table, int block_size, int kvheads, int heads, int head_dim,
    int rows, int position, float scale, void * workspace, hipStream_t stream) {
    if (heads <= 0 || kvheads <= 0 || rows <= 0 || head_dim <= 0 ||
        heads % kvheads != 0 || workspace == nullptr) return -1;
    const int ctx = position + rows;
    const int64_t query_elems = int64_t(heads) * rows * head_dim;
    const int64_t kv_elems = int64_t(kvheads) * ctx * head_dim;
    const int64_t score_elems = int64_t(heads) * rows * ctx;
    auto * base = static_cast<__half *>(workspace);
    __half * q16 = base;
    __half * k16 = q16 + query_elems;
    __half * v16 = k16 + kv_elems;
    float * score = reinterpret_cast<float *>(v16 + kv_elems);
    __half * probability = reinterpret_cast<__half *>(score + score_elems);
    float * result = reinterpret_cast<float *>(probability + score_elems);
    pack_query_f16<<<(query_elems + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        q16, q, rows, heads, head_dim);
    gather_kv_f16<<<(int64_t(ctx) * kvheads * head_dim + kThreads - 1) / kThreads,
                   kThreads, 0, stream>>>(
        k16, v16, static_cast<const __half *>(k_pool),
        static_cast<const __half *>(v_pool), table, block_size, ctx, kvheads, head_dim);
    rocblas_handle handle = attention_blas_handle();
    if (handle == nullptr) return -2;
    if (rocblas_set_stream(handle, stream) != rocblas_status_success) return -3;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int h = 0; h < heads; ++h) {
        const int kvh = h / (heads / kvheads);
        rocblas_status status = rocblas_gemm_ex(
            handle, rocblas_operation_transpose, rocblas_operation_none,
            ctx, rows, head_dim, &alpha,
            k16 + int64_t(kvh) * ctx * head_dim, rocblas_datatype_f16_r, head_dim,
            q16 + int64_t(h) * rows * head_dim, rocblas_datatype_f16_r, head_dim,
            &beta, score + int64_t(h) * rows * ctx, rocblas_datatype_f32_r, ctx,
            score + int64_t(h) * rows * ctx, rocblas_datatype_f32_r, ctx,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        if (status != rocblas_status_success) return -100 - static_cast<int>(status);
    }
    const dim3 softmax_grid(static_cast<unsigned>(rows), static_cast<unsigned>(heads));
    masked_softmax_f16<<<softmax_grid, kThreads, 0, stream>>>(
        score, probability, rows, heads, ctx, position, scale);
    for (int h = 0; h < heads; ++h) {
        const int kvh = h / (heads / kvheads);
        rocblas_status status = rocblas_gemm_ex(
            handle, rocblas_operation_none, rocblas_operation_none,
            head_dim, rows, ctx, &alpha,
            v16 + int64_t(kvh) * ctx * head_dim, rocblas_datatype_f16_r, head_dim,
            probability + int64_t(h) * rows * ctx, rocblas_datatype_f16_r, ctx,
            &beta, result + int64_t(h) * rows * head_dim, rocblas_datatype_f32_r, head_dim,
            result + int64_t(h) * rows * head_dim, rocblas_datatype_f32_r, head_dim,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        if (status != rocblas_status_success) return -200 - static_cast<int>(status);
    }
    unpack_attn_f16<<<(int64_t(rows) * heads * head_dim + kThreads - 1) / kThreads,
                      kThreads, 0, stream>>>(out, result, q, rows, heads, head_dim);
    return hipGetLastError() == hipSuccess ? 0 : -300;
}

}  // namespace yuuka
