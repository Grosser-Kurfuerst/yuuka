#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float load_f16_device(const void * ptr) {
    return __half2float(*reinterpret_cast<const __half *>(ptr));
}

// warp_reduce_sum reduces a value across a full 32-lane wave. Every lane
// receives the total, so callers can use the result without a block barrier.
__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffffffffffull, value, offset);
    }
    return value;
}

template <int kThreads>
__device__ __forceinline__ float block_reduce_sum(float value, float * scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}
