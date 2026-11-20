#pragma once

#include "common.cuh"

#include "yuuka_hip.h"

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * q,
                                                 uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__device__ __forceinline__ float q4k_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    const float dm = load_f16_device(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t * qs = blk + 16;
    const int group = j / 64;
    const int within = j % 64;
    const int sub = group * 2 + (within >= 32 ? 1 : 0);
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min_k4(sub, scales, sc, m);
    const int lane = within % 32;
    const uint8_t q = within < 32 ? (qs[group * 32 + lane] & 0xF)
                                  : (qs[group * 32 + lane] >> 4);
    return d * sc * q - dm * m;
}

__device__ __forceinline__ float q5k_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    const float dm = load_f16_device(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t * qh = blk + 16;
    const uint8_t * ql = blk + 48;
    const int group = j / 64;
    const int within = j % 64;
    const int sub = group * 2 + (within >= 32 ? 1 : 0);
    uint8_t sc = 0;
    uint8_t m = 0;
    get_scale_min_k4(sub, scales, sc, m);
    const int lane = within % 32;
    int q = within < 32 ? (ql[group * 32 + lane] & 0xF)
                        : (ql[group * 32 + lane] >> 4);
    const uint8_t mask = static_cast<uint8_t>(1u << (2 * group + (within < 32 ? 0 : 1)));
    if (qh[lane] & mask) {
        q += 16;
    }
    return d * sc * q - dm * m;
}

__device__ __forceinline__ float q6k_value(const uint8_t * blk, int j) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const float d = load_f16_device(blk + 208);
    const int n = j / 128;
    const int l = j % 128;
    const uint8_t * qlN = ql + n * 64;
    const uint8_t * qhN = qh + n * 32;
    const int8_t * scN = sc + n * 8;
    int q = 0;
    int sidx = 0;
    if (l < 32) {
        q = (qlN[l] & 0xF) | (((qhN[l] >> 0) & 3) << 4);
        sidx = l / 16 + 0;
    } else if (l < 64) {
        q = (qlN[l] & 0xF) | (((qhN[l - 32] >> 2) & 3) << 4);
        sidx = (l - 32) / 16 + 2;
    } else if (l < 96) {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 64] >> 4) & 3) << 4);
        sidx = (l - 64) / 16 + 4;
    } else {
        q = (qlN[l - 64] >> 4) | (((qhN[l - 96] >> 6) & 3) << 4);
        sidx = (l - 96) / 16 + 6;
    }
    return d * scN[sidx] * static_cast<float>(q - 32);
}

__device__ __forceinline__ float q8_value(const uint8_t * blk, int j) {
    const float d = load_f16_device(blk);
    return d * static_cast<float>(static_cast<int8_t>(blk[2 + j]));
}

// load_u32_pair assembles one 32-bit word from two 16-bit loads. Packed GGUF
// rows are only two-byte aligned at several row strides (210 bytes for Q6_K,
// 34 for Q8_0, 18 for Q4_0, 22 for Q5_0), so the four packed values of one
// lane cannot be read as one aligned word.
__device__ __forceinline__ uint32_t load_u32_pair(const uint8_t * p) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    __builtin_memcpy(&lo, p, 2);
    __builtin_memcpy(&hi, p + 2, 2);
    return lo | (hi << 16);
}

// decode_span4 writes the four consecutive values of one packed weight row that
// start at index k into w, in order. It is the vectorized form of the per-value
// decoders above for callers that consume four neighbouring values at a time:
// the block base, the nibble or high-bit selection, and the K-quant scale and
// minimum are hoisted out of the four values. The contract is that k is a
// multiple of four and that the span lies inside one quantization block, which
// holds for every lane mapping that lays four consecutive values per lane
// inside a 32-value group.
//
// Every decoder restates the arithmetic of its per-value counterpart, including
// the order of the scale and minimum products, so both decode the same digits.
// tests/hip/prefill_test.cpp compares the LM head kernel that uses this helper
// against the independent host references for every dtype.
template <uint32_t DType>
__device__ __forceinline__ void decode_span4(const uint8_t * row, int64_t k,
                                            float w[4]) {
    if constexpr (DType == YUUKA_DTYPE_F32) {
        const float4 v = *reinterpret_cast<const float4 *>(row + k * 4);
        w[0] = v.x;
        w[1] = v.y;
        w[2] = v.z;
        w[3] = v.w;
    } else if constexpr (DType == YUUKA_DTYPE_F16) {
        const float2 lo = __half22float2(*reinterpret_cast<const __half2 *>(row + k * 2));
        const float2 hi =
            __half22float2(*reinterpret_cast<const __half2 *>(row + k * 2 + 4));
        w[0] = lo.x;
        w[1] = lo.y;
        w[2] = hi.x;
        w[3] = hi.y;
    } else if constexpr (DType == YUUKA_DTYPE_Q8_0) {
        const uint8_t * blk = row + (k / 32) * 34;
        const uint32_t packed = load_u32_pair(blk + 2 + (k & 31));
        const float d = load_f16_device(blk);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = d * static_cast<float>(
                           static_cast<int8_t>((packed >> (8 * i)) & 0xFF));
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q4_0) {
        const uint8_t * blk = row + (k / 32) * 18;
        const int j = static_cast<int>(k & 31);
        const uint32_t packed = load_u32_pair(blk + 2 + (j & 15));
        const float d = load_f16_device(blk);
        const int shift = j < 16 ? 0 : 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = static_cast<int>((packed >> (8 * i + shift)) & 0xF);
            w[i] = d * static_cast<float>(q - 8);
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q5_0) {
        const uint8_t * blk = row + (k / 32) * 22;
        const int j = static_cast<int>(k & 31);
        const uint32_t high = load_u32_pair(blk + 2);
        const uint32_t packed = load_u32_pair(blk + 6 + (j & 15));
        const float d = load_f16_device(blk);
        const int shift = j < 16 ? 0 : 4;
        const uint32_t bits = (high >> j) & 0xF;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = static_cast<int>((packed >> (8 * i + shift)) & 0xF) |
                          (static_cast<int>((bits >> i) & 1) << 4);
            w[i] = d * static_cast<float>(q - 16);
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (k / 256) * 144;
        const int j = static_cast<int>(k & 255);
        const int sub = j / 32;
        const int within = j & 31;
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float ds = load_f16_device(blk) * static_cast<float>(sc);
        const float dm = load_f16_device(blk + 2) * static_cast<float>(m);
        const uint8_t * qs = blk + 16 + (sub / 2) * 32 + within;
        uint32_t packed = 0;
        __builtin_memcpy(&packed, qs, 4);
        const int shift = (sub & 1) ? 4 : 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = static_cast<int>((packed >> (8 * i + shift)) & 0xF);
            w[i] = ds * static_cast<float>(q) - dm;
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q5_K) {
        const uint8_t * blk = row + (k / 256) * 176;
        const int j = static_cast<int>(k & 255);
        const int sub = j / 32;
        const int within = j & 31;
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float ds = load_f16_device(blk) * static_cast<float>(sc);
        const float dm = load_f16_device(blk + 2) * static_cast<float>(m);
        const uint8_t * ql = blk + 48 + (sub / 2) * 32 + within;
        uint32_t packed = 0;
        __builtin_memcpy(&packed, ql, 4);
        const uint32_t high = load_u32_pair(blk + 16 + within);
        const int shift = (sub & 1) ? 4 : 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = static_cast<int>((packed >> (8 * i + shift)) & 0xF);
            if ((high >> (8 * i)) & (1u << sub)) {
                q += 16;
            }
            w[i] = ds * static_cast<float>(q) - dm;
        }
    } else {
        static_assert(DType == YUUKA_DTYPE_Q6_K, "unsupported span dtype");
        const uint8_t * blk = row + (k / 256) * 210;
        const int j = static_cast<int>(k & 255);
        const int half = j / 128;
        const int l = j & 127;
        const int seg = l / 32;
        const int within = l & 31;
        const uint32_t ql = load_u32_pair(blk + half * 64 + 32 * (seg & 1) + within);
        const uint32_t qh = load_u32_pair(blk + 128 + half * 32 + within);
        const int sc = static_cast<int>(reinterpret_cast<const int8_t *>(
            blk + 192 + half * 8)[(within / 16) + 2 * seg]);
        const float d = load_f16_device(blk + 208) * static_cast<float>(sc);
        const int nibble = (seg & 2) ? 4 : 0;
        const int bits = 2 * seg;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = static_cast<int>((ql >> (8 * i + nibble)) & 0xF) |
                          (static_cast<int>((qh >> (8 * i + bits)) & 3) << 4);
            w[i] = d * static_cast<float>(q - 32);
        }
    }
}

// packed_value decodes element index of one packed GGUF weight row. The caller
// selects dtype at runtime; the switch mirrors the host capability checks.
__device__ __forceinline__ float packed_value(uint32_t dtype, const uint8_t * row,
                                              int64_t index) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return reinterpret_cast<const float *>(row)[index];
        case YUUKA_DTYPE_F16:
            return load_f16_device(row + index * 2);
        case YUUKA_DTYPE_Q4_0: {
            const uint8_t * blk = row + (index / 32) * 18;
            const int64_t j = index % 32;
            const uint8_t packed = blk[2 + (j & 15)];
            const int q = (j < 16) ? (packed & 0xF) : (packed >> 4);
            return load_f16_device(blk) * static_cast<float>(q - 8);
        }
        case YUUKA_DTYPE_Q5_0: {
            const uint8_t * blk = row + (index / 32) * 22;
            const int64_t j = index % 32;
            const uint8_t packed = blk[6 + (j & 15)];
            const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                                (static_cast<uint32_t>(blk[3]) << 8) |
                                (static_cast<uint32_t>(blk[4]) << 16) |
                                (static_cast<uint32_t>(blk[5]) << 24);
            const int high = (j < 16)
                                 ? static_cast<int>(((qh >> j) << 4) & 0x10)
                                 : static_cast<int>((qh >> ((j & 15) + 12)) & 0x10);
            const int q = (j < 16 ? (packed & 0xF) : (packed >> 4)) | high;
            return load_f16_device(blk) * static_cast<float>(q - 16);
        }
        case YUUKA_DTYPE_Q8_0:
            return q8_value(row + (index / 32) * 34, static_cast<int>(index % 32));
        case YUUKA_DTYPE_Q4_K:
            return q4k_value(row + (index / 256) * 144, static_cast<int>(index % 256));
        case YUUKA_DTYPE_Q5_K:
            return q5k_value(row + (index / 256) * 176, static_cast<int>(index % 256));
        case YUUKA_DTYPE_Q6_K:
            return q6k_value(row + (index / 256) * 210, static_cast<int>(index % 256));
        default:
            return 0.0f;
    }
}

// pack4_i8 packs four values in -128..127 into one 32-bit word in little-endian
// byte order, which is the int8 order dp4a consumes.
__device__ __forceinline__ int32_t pack4_i8(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// decode_span4_i8 decodes the four consecutive values of one packed weight row
// that start at k into the int8 form dp4a consumes, and returns the affine
// (scale, bias) of the sub-block that contains them, so that the four values are
// scale * q + bias with q the digit the kernel dots against the activations.
// The digit is the unsigned nibble for Q4_K (0..15), Q5_K (0..31), and Q4_0
// (0..15), the signed 6-bit value for Q6_K (-32..31), and int8 for Q8_0.
//
// The contract matches decode_span4: k is a multiple of four and the span lies
// inside one scale block, which every 32-value block layout and the four-values-
// per-lane mapping guarantee. The per-value arithmetic is the same as q4k_value,
// q5k_value, q6k_value, and q8_value; the reference tests check that the int8
// route and the FP32 route agree to the recorded tolerance.
template <uint32_t DType>
__device__ __forceinline__ void decode_span4_i8(const uint8_t * row, int64_t k,
                                               int32_t & q, float & scale,
                                               float & bias) {
    if constexpr (DType == YUUKA_DTYPE_Q4_K || DType == YUUKA_DTYPE_Q5_K) {
        constexpr int kBlockBytes = (DType == YUUKA_DTYPE_Q4_K) ? 144 : 176;
        const uint8_t * blk = row + (k / 256) * kBlockBytes;
        const int j = static_cast<int>(k % 256);
        const int sub = j / 32;
        const int within = j & 31;
        const uint8_t * ql = blk + (DType == YUUKA_DTYPE_Q4_K ? 16 : 48) +
                             (sub / 2) * 32 + within;
        // A Q4_K/Q5_K row stride is 144 or 176 bytes, so the four-byte read is
        // naturally aligned.
        uint32_t packed = 0;
        __builtin_memcpy(&packed, ql, 4);
        packed = (packed >> ((sub & 1) ? 4 : 0)) & 0x0F0F0F0Fu;
        if constexpr (DType == YUUKA_DTYPE_Q5_K) {
            // The fifth bit of each digit is one bit of four consecutive qh
            // bytes, and it belongs at bit 4 of the matching output byte.
            uint32_t high = 0;
            __builtin_memcpy(&high, blk + 16 + within, 4);
            packed |= ((high >> sub) & 0x01010101u) << 4;
        }
        q = static_cast<int32_t>(packed);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        scale = load_f16_device(blk) * static_cast<float>(sc);
        bias = -(load_f16_device(blk + 2) * static_cast<float>(m));
    } else if constexpr (DType == YUUKA_DTYPE_Q6_K) {
        const uint8_t * blk = row + (k / 256) * 210;
        const int j = static_cast<int>(k % 256);
        const int n = j / 128;
        const int l = j % 128;
        const uint8_t * ql = blk + n * 64;
        const uint8_t * qh = blk + 128 + n * 32;
        int qlbase = 0;
        int qhbase = 0;
        int shift = 0;
        int sidx = 0;
        if (l < 32) {
            qlbase = l;
            qhbase = l;
            shift = 0;
            sidx = l / 16;
        } else if (l < 64) {
            // The second 32-value group takes the low nibble of the second 32
            // ql bytes, and its high bits are the same qh bytes as the first.
            qlbase = l;
            qhbase = l - 32;
            shift = 2;
            sidx = (l - 32) / 16 + 2;
        } else if (l < 96) {
            qlbase = l - 64;
            qhbase = l - 64;
            shift = 4;
            sidx = (l - 64) / 16 + 4;
        } else {
            qlbase = l - 64;
            qhbase = l - 96;
            shift = 6;
            sidx = (l - 96) / 16 + 6;
        }
        // A Q6_K row stride is 210 bytes, so a block base is only two-byte
        // aligned and the two four-byte reads go through load_u32_pair.
        const uint32_t low = load_u32_pair(ql + qlbase);
        const uint32_t high = load_u32_pair(qh + qhbase);
        // The low two 32-value groups of a 128-value slice take the low nibble
        // of ql and the upper two take the high nibble.
        const uint32_t nibbles = ((l < 64) ? low : (low >> 4)) & 0x0F0F0F0Fu;
        const uint32_t top = ((high >> shift) & 0x03030303u) << 4;
        const uint32_t span = nibbles | top;
        // Each 6-bit digit is signed by subtracting 32; subtracting the packed
        // constant would borrow across bytes, so each byte is converted alone.
        q = pack4_i8(static_cast<int>(span & 0xFF) - 32,
                     static_cast<int>((span >> 8) & 0xFF) - 32,
                     static_cast<int>((span >> 16) & 0xFF) - 32,
                     static_cast<int>(span >> 24) - 32);
        const auto * scales = reinterpret_cast<const int8_t *>(blk + 192);
        scale = load_f16_device(blk + 208) *
                static_cast<float>(scales[n * 8 + sidx]);
        bias = 0.0f;
    } else if constexpr (DType == YUUKA_DTYPE_Q8_0) {
        const uint8_t * blk = row + (k / 32) * 34;
        const int within = static_cast<int>(k & 31);
        // A Q8_0 row stride is 34 bytes, so the read is two-byte aligned.
        q = static_cast<int32_t>(load_u32_pair(blk + 2 + within));
        scale = load_f16_device(blk);
        bias = 0.0f;
    } else {
        q = 0;
        scale = 0.0f;
        bias = 0.0f;
    }
}

// yuuka_packed_row_bytes returns the byte width of one packed weight row, or zero for
// a dtype this file does not decode. The kernel launchers and the Go planner
// share the same table.
inline uint64_t yuuka_packed_row_bytes(uint32_t dtype, uint64_t n_in) {
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

// load_span16 loads sixteen contiguous packed bytes into four words. Q6_K,
// Q8_0, Q4_0, and Q5_0 rows have strides that are not a multiple of sixteen
// bytes, so their span base is only two-byte aligned and the bytes are copied
// rather than read as one vector; Q4_K and Q5_K are sixteen-byte aligned and use
// a vector load directly.
__device__ __forceinline__ void load_span16(const uint8_t * p, uint32_t words[4]) {
    __builtin_memcpy(words, p, 16);
}

// decode_span16 writes the sixteen consecutive values of one packed weight row
// that start at k. It is the wide form of decode_span4 for the remapped decode
// matvec: a lane owns sixteen K values, so a warp covers 512 values with one
// eight- to sixteen-byte load per lane, and the kernel never stages the
// activation row. Every dtype reads the bytes of its quantization block that
// cover the span, hoists the block scale and minimum out of the sixteen values,
// and restates the same arithmetic as decode_span4.
//
// The contract is that k is a multiple of sixteen and that the span lies inside
// one scale block, which a 512-value K step and a sixteen-values-per-lane
// mapping guarantee for the dtypes served here.
template <uint32_t DType>
__device__ __forceinline__ void decode_span16(const uint8_t * row, int64_t k,
                                              float w[16]) {
    if constexpr (DType == YUUKA_DTYPE_F32) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 v = *reinterpret_cast<const float4 *>(row + (k + 4 * i) * 4);
            w[4 * i + 0] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else if constexpr (DType == YUUKA_DTYPE_F16) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __half2 h = *reinterpret_cast<const __half2 *>(row + (k + 2 * i) * 2);
            const float2 v = __half22float2(h);
            w[2 * i + 0] = v.x;
            w[2 * i + 1] = v.y;
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q8_0) {
        const uint8_t * blk = row + (k / 32) * 34;
        const int within = static_cast<int>(k & 31);
        const float d = load_f16_device(blk);
        uint32_t words[4];
        load_span16(blk + 2 + within, words);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint32_t byte = (words[i / 4] >> (8 * (i % 4))) & 0xFF;
            w[i] = d * static_cast<float>(static_cast<int8_t>(byte));
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q4_0) {
        const uint8_t * blk = row + (k / 32) * 18;
        const int within = static_cast<int>(k & 31);
        const float d = load_f16_device(blk);
        uint32_t words[4];
        load_span16(blk + 2, words);
        const int shift = within < 16 ? 0 : 4;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = static_cast<int>((words[i / 4] >> (8 * (i % 4) + shift)) & 0xF);
            w[i] = d * static_cast<float>(q - 8);
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q5_0) {
        const uint8_t * blk = row + (k / 32) * 22;
        const int within = static_cast<int>(k & 31);
        const float d = load_f16_device(blk);
        uint32_t words[4];
        load_span16(blk + 6, words);
        uint32_t high = 0;
        __builtin_memcpy(&high, blk + 2, 4);
        high >>= within;
        const int shift = within < 16 ? 0 : 4;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = static_cast<int>((words[i / 4] >> (8 * (i % 4) + shift)) & 0xF) |
                          (static_cast<int>((high >> i) & 1) << 4);
            w[i] = d * static_cast<float>(q - 16);
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q4_K) {
        const uint8_t * blk = row + (k / 256) * 144;
        const int j = static_cast<int>(k & 255);
        const int sub = j / 32;
        const int within = j & 31;
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float ds = load_f16_device(blk) * static_cast<float>(sc);
        const float dm = load_f16_device(blk + 2) * static_cast<float>(m);
        const uint8_t * qs = blk + 16 + (sub / 2) * 32 + within;
        // A Q4_K row stride is a multiple of 144 bytes and within is 0 or 16,
        // so the span is sixteen-byte aligned.
        const uint4 raw = *reinterpret_cast<const uint4 *>(qs);
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
        const int shift = (sub & 1) ? 4 : 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = static_cast<int>((words[i / 4] >> (8 * (i % 4) + shift)) & 0xF);
            w[i] = ds * static_cast<float>(q) - dm;
        }
    } else if constexpr (DType == YUUKA_DTYPE_Q5_K) {
        const uint8_t * blk = row + (k / 256) * 176;
        const int j = static_cast<int>(k & 255);
        const int sub = j / 32;
        const int within = j & 31;
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float ds = load_f16_device(blk) * static_cast<float>(sc);
        const float dm = load_f16_device(blk + 2) * static_cast<float>(m);
        const uint8_t * ql = blk + 48 + (sub / 2) * 32 + within;
        const uint8_t * qh = blk + 16 + within;
        // A Q5_K row stride is a multiple of 176 bytes and within is 0 or 16,
        // so both spans are sixteen-byte aligned.
        const uint4 raw = *reinterpret_cast<const uint4 *>(ql);
        const uint4 hi = *reinterpret_cast<const uint4 *>(qh);
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
        const uint32_t highs[4] = {hi.x, hi.y, hi.z, hi.w};
        const int shift = (sub & 1) ? 4 : 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int q = static_cast<int>((words[i / 4] >> (8 * (i % 4) + shift)) & 0xF);
            if ((highs[i / 4] >> (8 * (i % 4))) & (1u << sub)) {
                q += 16;
            }
            w[i] = ds * static_cast<float>(q) - dm;
        }
    } else {
        static_assert(DType == YUUKA_DTYPE_Q6_K, "unsupported span dtype");
        const uint8_t * blk = row + (k / 256) * 210;
        const int j = static_cast<int>(k & 255);
        const int half = j / 128;
        const int l = j & 127;
        const int seg = l / 32;
        const int within = l & 31;
        const uint8_t * ql = blk + half * 64 + 32 * (seg & 1) + within;
        const uint8_t * qh = blk + 128 + half * 32 + within;
        const int sc = static_cast<int>(reinterpret_cast<const int8_t *>(
            blk + 192 + half * 8)[2 * seg + within / 16]);
        const float d = load_f16_device(blk + 208) * static_cast<float>(sc);
        uint32_t qlw[4];
        uint32_t qhw[4];
        load_span16(ql, qlw);
        load_span16(qh, qhw);
        const int nibble = (seg & 2) ? 4 : 0;
        const int bits = 2 * seg;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int q = static_cast<int>((qlw[i / 4] >> (8 * (i % 4) + nibble)) & 0xF) |
                          (static_cast<int>((qhw[i / 4] >> (8 * (i % 4) + bits)) & 3) << 4);
            w[i] = d * static_cast<float>(q - 32);
        }
    }
}
