#include "yuuka_kernels.h"

#include <rocblas/rocblas.h>

#include <mutex>

namespace {

// The baseline handle is process-lifetime state. The current executor submits
// on one stream from one thread, so a shared handle with an explicit stream is
// sufficient. Concurrent use would need one handle per stream.
rocblas_handle cached_handle() {
    static rocblas_handle handle = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        if (rocblas_create_handle(&handle) != rocblas_status_success) {
            handle = nullptr;
        }
    });
    return handle;
}

} // namespace

namespace yuuka {

int launch_gemm_blas_f16(float * out, void * out16, const void * x16,
                         const void * w, uint64_t n_in, uint64_t n_out,
                         uint64_t batch, hipStream_t stream) {
    rocblas_handle handle = cached_handle();
    if (handle == nullptr) {
        return -1;
    }
    rocblas_status stream_status = rocblas_set_stream(handle, stream);
    if (stream_status != rocblas_status_success) {
        return static_cast<int>(stream_status);
    }
    // The alpha and beta scalars match the compute type.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // out is row-major [batch, n_out], which is column-major [n_out, batch]
    // with leading dimension n_out. W is [n_out, n_in] row-major, so it is the
    // transposed operand of a column-major [n_in, n_out] matrix.
    //
    // FP32 compute with an FP16 destination keeps the FP32 accumulator, so
    // rocBLAS rounds each output once instead of rounding the running sum after
    // every K step. That FP16 form of the result lands in the caller-owned
    // out16 scratch, and one cast kernel converts it to the FP32 result. FP16
    // compute measured 1.6x faster and is rejected: it moved the prefill logits
    // by 0.305 against the 0.05 gate (docs/benchmarks.md).
    rocblas_status status = rocblas_gemm_ex(
        handle, rocblas_operation_transpose, rocblas_operation_none,
        static_cast<rocblas_int>(n_out), static_cast<rocblas_int>(batch),
        static_cast<rocblas_int>(n_in), &alpha,
        w, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_in),
        x16, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_in), &beta,
        out16, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_out),
        out16, rocblas_datatype_f16_r, static_cast<rocblas_int>(n_out),
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    if (status != rocblas_status_success) {
        return static_cast<int>(status);
    }
    launch_cast_f16_to_f32(out, out16, n_out * batch, stream);
    return 0;
}

} // namespace yuuka
