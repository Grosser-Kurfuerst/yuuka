#ifndef YUUKA_KERNELS_H
#define YUUKA_KERNELS_H

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace yuuka {

void launch_add_f32(float * dst, const float * a, const float * b, uint64_t n,
                    hipStream_t stream);

void launch_gemv_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q4_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q5_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q8_0(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q4_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q5_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);
void launch_gemv_q6_k(float * out, const float * x, const void * w,
                      uint64_t n_in, uint64_t n_out, hipStream_t stream);

void launch_gemv_batch_f32(float * out, const float * x, const float * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream);
void launch_gemv_batch_f16(float * out, const float * x, const void * w,
                           uint64_t n_in, uint64_t n_out, uint64_t batch,
                           hipStream_t stream);
void launch_gemv_batch_q4_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q5_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q8_0(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q4_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q5_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);
void launch_gemv_batch_q6_k(float * out, const float * x, const void * w,
                            uint64_t n_in, uint64_t n_out, uint64_t batch,
                            hipStream_t stream);

void launch_gemm_f32(float * out, const float * x, const float * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);
void launch_gemm_f16(float * out, const float * x, const void * w,
                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                     hipStream_t stream);

/* Prefill GEMM family. launch_gemm_wmma_f32out accepts every supported weight
 * dtype and FP32 activations. launch_gemm_blas_f16 is the rocBLAS route: FP16
 * weights, FP16 activations, FP32 compute, and an FP16 output scratch that this
 * call casts to the FP32 result. rocBLAS therefore accumulates in FP32 and
 * rounds only the result. It returns 0 on success or a rocBLAS status. */
void launch_gemm_wmma_f32out(float * out, const float * x, const void * w,
                             uint64_t n_in, uint64_t n_out, uint64_t batch,
                             uint32_t dtype, hipStream_t stream);

/* Fused gate/up WMMA GEMM. Both weights have the same shape and dtype and the
 * kernel computes out = silu(W_gate * x) * (W_up * x) without writing the two
 * GEMM intermediates. */
void launch_gemm_wmma_swiglu_f32out(float * out, const float * x,
                                    const void * w_gate, const void * w_up,
                                    uint64_t n_in, uint64_t n_out, uint64_t batch,
                                    uint32_t dtype, hipStream_t stream);
int launch_gemm_blas_f16(float * out, void * out16, const void * x16, const void * w,
                         uint64_t n_in, uint64_t n_out, uint64_t batch,
                         hipStream_t stream);
void launch_cast_f32_to_f16(void * dst, const float * src, uint64_t n,
                            hipStream_t stream);
void launch_cast_f16_to_f32(void * dst, const void * src, uint64_t n,
                            hipStream_t stream);

/* Int8 MMQ prefill family.
 *
 * launch_quantize_q8_1 converts FP32 activations into Q8_1: qs holds batch*n
 * int8 values in [batch][n] row-major order, ds holds batch*(n/32) float2
 * entries where x is the 32-value block scale and y is the scale times the
 * integer block sum. n must be a multiple of 128.
 *
 * launch_gemm_mmq_f32out computes out[batch, n_out] = W[n_out, n_in] * x with
 * the Q8_1 activations above, int8-decoded Q4_K/Q5_K/Q6_K weights, exact int32
 * WMMA accumulation, and FP32 scale application. n_in must be a multiple of
 * 256, which every K-quant row layout already requires. */
void launch_quantize_q8_1(void * qs, void * ds, const float * x, uint64_t n,
                          uint64_t batch, hipStream_t stream);
void launch_gemm_mmq_f32out(float * out, const void * qs, const void * ds,
                            const void * w, uint64_t n_in, uint64_t n_out,
                            uint64_t batch, uint32_t dtype, hipStream_t stream);

/* Dequantizes a packed K-quant weight matrix into FP16.
 *
 * dst holds n_in*n_out FP16 elements. src holds the packed [n_out, n_in] matrix
 * with the dtype's row stride. The decoded values are bit-identical to the
 * values the rocWMMA path stages for the same tile, so only the accumulation
 * differs. Only Q4_K, Q5_K, and Q6_K are supported; an unknown dtype writes
 * nothing. */
void launch_dequant_f16(void * dst, const void * w, uint64_t n_in, uint64_t n_out,
                        uint32_t dtype, hipStream_t stream);

void launch_rmsnorm_f32(float * dst, const float * src, const float * weight,
                        int64_t n, float eps, hipStream_t stream);
void launch_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                              int heads, int64_t n, int64_t stride, float eps,
                              hipStream_t stream);
void launch_l2norm_heads_f32(float * dst, const float * src, int heads, int64_t n,
                             int64_t stride, float eps, float extra_scale,
                             hipStream_t stream);
void launch_gated_rmsnorm_heads_f32(float * dst, const float * src, const float * weight,
                                    const float * gate, int heads, int64_t n,
                                    int64_t stride, float eps, hipStream_t stream);
void launch_silu_mul_f32(float * out, const float * gate, const float * up,
                         int64_t n, hipStream_t stream);
void launch_add_f32_buffers(float * x, const float * delta, int64_t n, hipStream_t stream);
void launch_copy_f32(float * dst, const float * src, int64_t n, hipStream_t stream);
void launch_scale_f32(float * x, int64_t n, float scale, hipStream_t stream);
void launch_rope_f32(float * data, int heads, int stride, int head_dim, int rot_dim,
                     int pos, float base, hipStream_t stream);
void launch_rope_dynamic_f32(float * data, int heads, int stride, int head_dim,
                             int rot_dim, const int32_t * dynamic, float base,
                             hipStream_t stream);
void launch_attention_f32(float * out, const float * q, const float * k_cache,
                          const float * v_cache, float * scores,
                          int kv_heads, int heads, int head_dim, int kv_dim,
                          int position, int capacity, float scale, hipStream_t stream);
void launch_attention_paged_f32(float * out, const float * q, const void * k_pool,
                                const void * v_pool, float * scores,
                                const int32_t * block_table, int block_size,
                                int kv_heads, int heads, int head_dim, int kv_dim,
                                int position, int score_stride, float scale,
                                uint32_t dtype, int dense, hipStream_t stream);
void launch_attention_paged_dynamic_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        float * scores, const int32_t * block_table,
                                        const int32_t * dynamic, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int score_stride, float scale,
                                        uint32_t dtype, int dense, hipStream_t stream);
void launch_conv1d_f32(float * state, const float * weights, const float * qkv,
                       float * out, int channels, int kernel, hipStream_t stream);
void launch_gdn_f32(float * state, const float * q, const float * k, const float * v,
                    const float * alpha, const float * beta, const float * dt,
                    const float * a, float * out, int groups, int heads,
                    int state_size, hipStream_t stream);
void launch_embedding_f32(float * out, const void * weights, int32_t token,
                          uint64_t row_bytes, int64_t cols, uint32_t dtype,
                          hipStream_t stream);
void launch_embedding_dynamic_f32(float * out, const void * weights,
                                  const int32_t * dynamic, uint64_t row_bytes,
                                  int64_t cols, uint32_t dtype, hipStream_t stream);

/* Chunked prefill operators. Every row-pointer argument is a contiguous
 * [rows, width] buffer in row-major order. position is the token position of
 * row zero. */
void launch_rmsnorm_batch_f32(float * dst, const float * src, const float * weight,
                              int rows, int cols, int src_stride, int dst_stride,
                              float eps, hipStream_t stream);
void launch_rmsnorm_heads_batch_f32(float * dst, const float * src,
                                    const float * weight, int rows, int heads,
                                    int size, int stride, float eps,
                                    hipStream_t stream);
void launch_rope_batch_f32(float * data, int rows, int heads, int stride,
                           int head_dim, int rot_dim, int pos, float base,
                           int row_stride, hipStream_t stream);
void launch_kv_write_batch_f32(float * k_cache, float * v_cache, const float * k,
                               const float * v, int rows, int kv_dim, int position,
                               hipStream_t stream);
void launch_kv_write_paged_f32(void * k_pool, void * v_pool, const float * k,
                               const float * v, int rows, int kv_dim,
                               const int32_t * block_table, int block_size,
                               int position, uint32_t dtype, hipStream_t stream);
void launch_kv_write_paged_dynamic_f32(void * k_pool, void * v_pool,
                                       const float * k, const float * v,
                                       int rows, int kv_dim,
                                       const int32_t * block_table,
                                       const int32_t * dynamic, int block_size,
                                       uint32_t dtype, hipStream_t stream);
void launch_embedding_batch_f32(float * out, const void * weights,
                                const int32_t * tokens, int rows, uint64_t row_bytes,
                                int64_t cols, uint32_t dtype, hipStream_t stream);
void launch_attention_prefill_f32(float * out, const float * q,
                                  const float * k_cache, const float * v_cache,
                                  int kv_heads, int heads, int head_dim, int kv_dim,
                                  int rows, int position, int capacity, float scale,
                                  hipStream_t stream);
void launch_attention_prefill_paged_f32(float * out, const float * q,
                                        const void * k_pool, const void * v_pool,
                                        const int32_t * block_table, int block_size,
                                        int kv_heads, int heads, int head_dim,
                                        int kv_dim, int rows, int position,
                                        float scale, uint32_t dtype, int dense,
                                        hipStream_t stream);
// Experimental GEMM-based attention with gathered FP16 KV, batched QK/PV
// matrix multiplication, causal softmax, and gated final FP32 output.
int launch_attention_prefill_paged_blas_f16(
    float * out, const float * q, const void * k_pool, const void * v_pool,
    const int32_t * table, int block_size, int kvheads, int heads, int head_dim,
    int rows, int position, float scale, void * workspace, hipStream_t stream);

/* MoE routing and grouped expert kernels. ids/weights are [rows, used], counts
 * and assignment buckets are [experts], and each bucket holds up to rows*used
 * assignments. */
int launch_moe_route(const float * logits, int rows, int experts, int used,
                     int bucket_capacity, int32_t * ids, float * weights,
                     int32_t * counts, int32_t * assign_rows,
                     int32_t * assign_slots, float * assign_weights,
                     hipStream_t stream);
void launch_moe_gate_up_grouped(const float * norm, const void * gate_exps,
                                const void * up_exps, const int32_t * counts,
                                const int32_t * assign_rows,
                                const int32_t * assign_slots, float * gate_out,
                                float * up_out, int rows, int experts, int used,
                                int hidden, int ffn, uint64_t hidden_row_bytes,
                                uint32_t dtype, hipStream_t stream);
void launch_moe_down_grouped(const float * act, const void * down_exps,
                             const int32_t * counts, const int32_t * assign_rows,
                             const int32_t * assign_slots, float * out, int rows,
                             int experts, int used, int hidden, int ffn,
                             uint64_t ffn_row_bytes, uint32_t dtype,
                             hipStream_t stream);
void launch_moe_reduce_add(float * x, const float * expert_out,
                           const float * weights, int rows, int used, int hidden,
                           hipStream_t stream);

void launch_recurrent_chunk_f32(float * conv_state, float * rec_state,
                                const float * qkv, const float * z,
                                const float * alpha, const float * beta,
                                const float * conv_weights, const float * dt,
                                const float * a, const float * norm, float * conv,
                                float * out, int rows, int groups, int heads,
                                int state_size, int conv_dim, int inner,
                                int conv_kernel, float eps, float q_scale,
                                hipStream_t stream);

} // namespace yuuka

#endif
