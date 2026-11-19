#include "../common.cuh"
#include "../yuuka_kernels.h"

namespace {

constexpr int kThreads = 256;

// rmsnorm_reduce returns the block-wide sum of value. One wave shuffle tree per
// wave plus one cross-wave add replaces the eight-barrier tree the first version
// used: the reduction runs on every rmsnorm launch of the decode step, where the
// launch cost is the floor and the barriers were most of the time.
__device__ __forceinline__ float rmsnorm_reduce(float value, float * scratch,
                                                int warps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < warps ? scratch[lane] : 0.0f;
        total = warp_reduce_sum(total);
        if (lane == 0) {
            scratch[0] = total;
        }
    }
    __syncthreads();
    return scratch[0];
}

__global__ void rmsnorm_f32_kernel(float * dst, const float * src, const float * weight,
                                   int64_t n, float eps) {
    __shared__ float scratch[kThreads / 32];
    const int warps = blockDim.x / 32;
    float sum = 0.0f;
    if ((n & 3) == 0) {
        const float4 * src4 = reinterpret_cast<const float4 *>(src);
        const int64_t n4 = n / 4;
        for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = src4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            sum += src[i] * src[i];
        }
    }
    sum = rmsnorm_reduce(sum, scratch, warps);
    const float scale = rsqrtf(sum / static_cast<float>(n) + eps);
    if ((n & 3) == 0) {
        float4 * dst4 = reinterpret_cast<float4 *>(dst);
        const float4 * src4 = reinterpret_cast<const float4 *>(src);
        const float4 * weight4 = reinterpret_cast<const float4 *>(weight);
        const int64_t n4 = n / 4;
        for (int64_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = src4[i];
            const float4 w = weight4[i];
            dst4[i] = make_float4(v.x * scale * w.x, v.y * scale * w.y,
                                  v.z * scale * w.z, v.w * scale * w.w);
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            dst[i] = src[i] * scale * weight[i];
        }
    }
}

// rmsnorm_heads_f32_kernel runs one wave per head, so a 256-wide head norm is a
// single shuffle tree and no barrier: the scale is a per-head scalar that every
// lane of its wave computes from the reduced sum.
__global__ void rmsnorm_heads_f32_kernel(float * dst, const float * src, const float * weight,
                                         int64_t heads, int64_t n, int64_t stride, float eps) {
    const int64_t head = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) +
                         (threadIdx.x >> 5);
    if (head >= heads) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const float * head_src = src + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    if ((n & 3) == 0) {
        const float4 * src4 = reinterpret_cast<const float4 *>(head_src);
        const int64_t n4 = n / 4;
        for (int64_t i = lane; i < n4; i += 32) {
            const float4 v = src4[i];
            sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int64_t i = lane; i < n; i += 32) {
            sum += head_src[i] * head_src[i];
        }
    }
    const float scale = rsqrtf(warp_reduce_sum(sum) / static_cast<float>(n) + eps);
    if ((n & 3) == 0) {
        float4 * dst4 = reinterpret_cast<float4 *>(head_dst);
        const float4 * src4 = reinterpret_cast<const float4 *>(head_src);
        const float4 * weight4 = reinterpret_cast<const float4 *>(weight);
        const int64_t n4 = n / 4;
        for (int64_t i = lane; i < n4; i += 32) {
            const float4 v = src4[i];
            const float4 w = weight4[i];
            dst4[i] = make_float4(v.x * scale * w.x, v.y * scale * w.y,
                                  v.z * scale * w.z, v.w * scale * w.w);
        }
    } else {
        for (int64_t i = lane; i < n; i += 32) {
            head_dst[i] = head_src[i] * scale * weight[i];
        }
    }
}

__global__ void l2norm_heads_f32_kernel(float * dst, const float * src, int64_t heads,
                                        int64_t n, int64_t stride, float eps, float extra_scale) {
    const int64_t head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float scratch[kThreads];
    const float * head_src = src + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += head_src[i] * head_src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum + eps) * extra_scale;
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        head_dst[i] = head_src[i] * scale;
    }
}

__global__ void gated_rmsnorm_heads_f32_kernel(float * dst, const float * src, const float * weight,
                                               const float * gate, int64_t heads, int64_t n,
                                               int64_t stride, float eps) {
    const int64_t head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    __shared__ float scratch[kThreads];
    const float * head_src = src + head * stride;
    const float * head_gate = gate + head * stride;
    float * head_dst = dst + head * stride;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        sum += head_src[i] * head_src[i];
    }
    sum = block_reduce_sum<kThreads>(sum, scratch);
    if (threadIdx.x == 0) {
        scratch[0] = rsqrtf(sum / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float scale = scratch[0];
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float g = head_gate[i];
        head_dst[i] = head_src[i] * scale * weight[i] * (g / (1.0f + expf(-g)));
    }
}

__global__ void silu_mul_f32_kernel(float * out, const float * gate, const float * up, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

__global__ void add_f32_kernel_device(float * x, const float * delta, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] += delta[i];
    }
}

__global__ void copy_f32_kernel(float * dst, const float * src, int64_t n) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void scale_f32_kernel(float * x, int64_t n, float scale) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] *= scale;
    }
}

} // namespace

namespace yuuka {

void launch_rmsnorm_f32(float * dst, const float * src, const float * weight,
                        int64_t n, float eps, hipStream_t stream) {
    rmsnorm_f32_kernel<<<1, kThreads, 0, stream>>>(dst, src, weight, n, eps);
}

void launch_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                              int heads, int64_t n, int64_t stride, float eps,
                              hipStream_t stream) {
    const int warps = kThreads / 32;
    const int blocks = (heads + warps - 1) / warps;
    rmsnorm_heads_f32_kernel<<<blocks, kThreads, 0, stream>>>(
        dst, src, weight, heads, n, stride, eps);
}

void launch_l2norm_heads_f32(float * dst, const float * src, int heads, int64_t n,
                             int64_t stride, float eps, float extra_scale,
                             hipStream_t stream) {
    l2norm_heads_f32_kernel<<<heads, kThreads, 0, stream>>>(
        dst, src, heads, n, stride, eps, extra_scale);
}

void launch_gated_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                                    const float * gate, int heads, int64_t n,
                                    int64_t stride, float eps, hipStream_t stream) {
    gated_rmsnorm_heads_f32_kernel<<<heads, kThreads, 0, stream>>>(
        dst, src, weight, gate, heads, n, stride, eps);
}

void launch_silu_mul_f32(float * out, const float * gate, const float * up,
                         int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    silu_mul_f32_kernel<<<blocks, kThreads, 0, stream>>>(out, gate, up, n);
}

void launch_add_f32_buffers(float * x, const float * delta, int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    add_f32_kernel_device<<<blocks, kThreads, 0, stream>>>(x, delta, n);
}

void launch_copy_f32(float * dst, const float * src, int64_t n, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    copy_f32_kernel<<<blocks, kThreads, 0, stream>>>(dst, src, n);
}

void launch_scale_f32(float * x, int64_t n, float scale, hipStream_t stream) {
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    scale_f32_kernel<<<blocks, kThreads, 0, stream>>>(x, n, scale);
}

} // namespace yuuka
