// Phase 3 instruction-throughput probe for gfx1151: compares the scalar FP32
// FMA, the packed FP16 dot2, the packed int8 dot4, and the int8 rocWMMA tile.
// Each kernel keeps its operands in registers and repeats a fixed eight-way
// independent accumulation, so the printed rate is instruction throughput, not
// memory bandwidth. The synthetic buffers only seed the operands.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// YUUKA_PROBE_DOT1 is set by CMake: 0 on the RDNA 3.5 parts, which have no
// dot1-insts and therefore cannot compile the int8 dot4 builtin.
#ifndef YUUKA_PROBE_DOT1
#define YUUKA_PROBE_DOT1 1
#endif

namespace {

constexpr uint64_t kBufferBytes = 256ull << 20;
constexpr int kThreads = 256;
constexpr int kBlocks = 4096;
constexpr int kIters = 1024;
constexpr int kReps = 20;

void check(hipError_t err, const char * what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

double timed_ms(void (*launch)(const void *, void *, hipStream_t), const void * in,
                void * out, hipStream_t stream) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check(hipEventCreate(&start), "event create");
    check(hipEventCreate(&stop), "event create");
    for (int i = 0; i < 3; ++i) launch(in, out, stream);
    check(hipEventRecord(start, stream), "event record");
    for (int i = 0; i < kReps; ++i) launch(in, out, stream);
    check(hipEventRecord(stop, stream), "event record");
    check(hipEventSynchronize(stop), "sync");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed");
    check(hipEventDestroy(stop), "destroy");
    check(hipEventDestroy(start), "destroy");
    return double(ms) / kReps;
}

void report(const char * name, double macs_per_thread, int macs_per_op, double ms) {
    const double threads = double(kBlocks) * kThreads;
    const double macs = threads * macs_per_thread;
    const double tflops = 2.0 * macs / (ms / 1000.0) / 1e12;
    std::printf("%-12s %10.3f ms  %6.2f TFLOP/s-equivalent  (%d MACs/op)\n", name, ms,
                tflops, macs_per_op);
}

__global__ void fma_kernel(const void * in, void * out) {
    const float * src = static_cast<const float *>(in);
    float * dst = static_cast<float *>(out);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float seed = src[(idx * 8) & ((kBufferBytes / 4) - 1)];
    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = seed + float(j);
    }
    const float m = 1.000001f;
    const float c = 1e-7f;
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = fmaf(acc[j], m, c);
        }
    }
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sum += acc[j];
    }
    dst[idx & 1023] = sum;
}

__global__ void fdot2_kernel(const void * in, void * out) {
    const __half * src = static_cast<const __half *>(in);
    float * dst = static_cast<float *>(out);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const __half2 seed = *reinterpret_cast<const __half2 *>(
        src + 2 * ((idx * 16) & ((kBufferBytes / 2) - 1)));
    __half2 v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = seed;
    }
    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0.0f;
    }
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = __builtin_amdgcn_fdot2(v[j], seed, acc[j], false);
        }
    }
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sum += acc[j];
    }
    dst[idx & 1023] = sum;
}

#if YUUKA_PROBE_DOT1
__global__ void sdot4_kernel(const void * in, void * out) {
    const int * src = static_cast<const int *>(in);
    float * dst = static_cast<float *>(out);
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int seed = src[(idx * 4) & ((kBufferBytes / 4) - 1)];
    int acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc[j] = 0;
    }
    for (int i = 0; i < kIters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[j] = __builtin_amdgcn_sdot4(seed, seed, acc[j], false);
        }
    }
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sum += acc[j];
    }
    dst[idx & 1023] = float(sum);
}
#endif

__global__ void wmma_int8_kernel(const void * in, void * out) {
    using namespace rocwmma;
    const int8_t * src = static_cast<const int8_t *>(in);
    float * dst = static_cast<float *>(out);
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;
    fill_fragment(c_frag, 0);
    const int64_t base = (static_cast<int64_t>(blockIdx.x) * 256) % (kBufferBytes - 512);
    load_matrix_sync(a_frag, src + base, 16);
    load_matrix_sync(b_frag, src + base + 256, 16);
    for (int i = 0; i < kIters; ++i) {
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    if (threadIdx.x == 0) {
        dst[blockIdx.x & 1023] = float(c_frag.x[0]);
    }
}

void launch_fma(const void * in, void * out, hipStream_t stream) {
    fma_kernel<<<kBlocks, kThreads, 0, stream>>>(in, out);
}

void launch_fdot2(const void * in, void * out, hipStream_t stream) {
    fdot2_kernel<<<kBlocks, kThreads, 0, stream>>>(in, out);
}

#if YUUKA_PROBE_DOT1
void launch_sdot4(const void * in, void * out, hipStream_t stream) {
    sdot4_kernel<<<kBlocks, kThreads, 0, stream>>>(in, out);
}
#endif

void launch_wmma(const void * in, void * out, hipStream_t stream) {
    wmma_int8_kernel<<<kBlocks, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main() {
    int device = 0;
    check(hipGetDevice(&device), "get device");
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, device), "device properties");
    std::printf("dot probe: %s (%llu MiB, %d blocks x %d threads, %d repeats)\n",
                prop.gcnArchName,
                static_cast<unsigned long long>(kBufferBytes >> 20), kBlocks,
                kThreads, kReps);

    hipStream_t stream = nullptr;
    check(hipStreamCreate(&stream), "stream");
    void * in = nullptr;
    void * out = nullptr;
    check(hipMalloc(&in, kBufferBytes), "malloc in");
    check(hipMalloc(&out, 4096), "malloc out");
    check(hipMemset(in, 0x21, kBufferBytes), "memset in");

    const double fma_ms = timed_ms(launch_fma, in, out, stream);
    report("fp32-fma", double(kIters) * 8.0, 1, fma_ms);
    const double fdot2_ms = timed_ms(launch_fdot2, in, out, stream);
    report("fdot2", double(kIters) * 16.0, 2, fdot2_ms);
#if YUUKA_PROBE_DOT1
    const double sdot4_ms = timed_ms(launch_sdot4, in, out, stream);
    report("sdot4", double(kIters) * 32.0, 4, sdot4_ms);
#else
    std::printf("%-12s %s\n", "sdot4",
                "unavailable: this target has no dot1-insts");
#endif
    const double wmma_ms = timed_ms(launch_wmma, in, out, stream);
    // One 16x16x16 tile per wave instruction: 4096 MACs per call and wave.
    const double wmma_macs_per_thread = double(kIters) * 4096.0 / 32.0;
    report("rocwmma-int8", wmma_macs_per_thread, 4096, wmma_ms);

    check(hipFree(out), "free out");
    check(hipFree(in), "free in");
    check(hipStreamDestroy(stream), "stream destroy");
    return 0;
}
