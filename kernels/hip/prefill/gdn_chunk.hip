#include "../common.cuh"
#include "../yuuka_kernels.h"

// Chunkwise gated delta rule for the recurrent layers of qwen35.
//
// The per-token recurrence is
//
//   S_r = gamma_r (I - beta_r k_r k_r^T) S_{r-1} + beta_r k_r v_r^T
//
// which serializes every token behind two block barriers and a full state pass.
// Over a chunk of n tokens with p_r = gamma_1 .. gamma_r and g_r = log p_r the
// same recurrence is
//
//   S_C = exp(g_c) S_0 + K^T (M^-1 diag(beta) (exp(g_c - g) V - exp(g_c) K S_0))
//   o_r = exp(g_r) (q_r S_0 + Z (diag(beta exp(-g)) V - K S_0))_r
//   Z   = (Q K^T on the lower triangle) M^-1
//   M   = I + B,   B[r][s] = beta_r (k_r . k_s)   for s < r,
//
// where K/Q/V are the normalized chunk rows and g_c the last cumulative decay.
// The decay of a row is a scalar for the whole state matrix, so it factors out
// of the triangular solve and only rows inside the chunk interact. Every
// product above is dense, so each chunk becomes matrix work instead of n
// serialized rank-one updates.
//
// Three kernels run per prompt chunk:
//   - gdn_prepare_rows_f32_kernel normalizes q/k and applies the gate
//     activations in place, so the later kernels read one layout.
//   - gdn_prep_f32_kernel runs once per (chunk, head) and writes M^-1 and Z.
//   - gdn_scan_f32_kernel walks the chunks of one head in order with one state
//     column block per workgroup, so the sequential part is a short chain of
//     dense products and no barrier scales with the chunk length.
//
// The chunk length is fixed at 32 so the triangular system fits one warp-wide
// inversion and the state projection stays a 32-row tile.

namespace {

constexpr int kGdnChunk = 32;
constexpr int kGdnState = 128;
constexpr int kGdnCols = 32;                     // state columns per scan block
constexpr int kGdnParts = 16;                    // state row parts per column group
constexpr int kGdnPartRows = kGdnState / kGdnParts;
constexpr int kGdnThreads = 256;                 // 16 column groups x 16 row parts
// A thread keeps two state columns in registers, so one LDS float4 of a staged
// tile feeds eight multiply-adds. With one column per thread the tile was read
// once per column and the sequential chain dominated.
constexpr int kGdnColsPerThread = 2;
constexpr int kGdnRowsPerThread = kGdnChunk / kGdnParts;
constexpr int kGdnStride = 36;                   // padded LDS stride, 16B aligned
constexpr int kGdnMatrix = kGdnChunk * kGdnStride;
constexpr int kGdnTile = kGdnState * kGdnStride;

// Pair indices of the two triangular enumerations used by the prep kernel.
__device__ __forceinline__ void decodeStrictLower(int pair, int & r, int & s) {
    int rem = pair;
    r = 1;
    while (rem >= r) {
        rem -= r;
        ++r;
    }
    s = rem;
}

__device__ __forceinline__ void decodeLower(int pair, int & r, int & s) {
    int rem = pair;
    r = 0;
    while (rem >= r + 1) {
        rem -= r + 1;
        ++r;
    }
    s = rem;
}

// The scan stages a chunk row tile as krt[(i & 15) * kGdnParts + (i >> 4)][r]:
// the eight row parts of a column then read LDS windows that differ by
// kGdnStride floats and therefore land on different banks, while the 32 chunk
// rows of one state row stay contiguous for float4 loads.
__device__ __forceinline__ int krtOffset(int row_part, int row_in_part) {
    return (row_in_part * kGdnParts + row_part) * kGdnStride;
}


// gdn_prepare_rows_f32_kernel normalizes the q and k segments of every row in
// place and replaces the raw alpha and beta projections with the log decay and
// the sigmoid gate.
__global__ void gdn_prepare_rows_f32_kernel(float * conv, float * alpha,
                                            float * beta, const float * dt,
                                            const float * a, int rows,
                                            int groups, int heads, int state,
                                            int conv_dim, float eps,
                                            float q_scale) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps = blockDim.x >> 5;
    const int key_dim = groups * state;
    float * line = conv + static_cast<int64_t>(row) * conv_dim;
    // One wave normalizes one segment: state values, four per lane.
    for (int task = warp; task < 2 * groups; task += warps) {
        const int which = task / groups;
        const int group = task % groups;
        float * seg = line + (which == 0 ? 0 : key_dim) + group * state;
        float sum = 0.0f;
        for (int i = lane; i < state; i += 32) {
            sum = fmaf(seg[i], seg[i], sum);
        }
        sum = warp_reduce_sum(sum);
        const float scale = rsqrtf(sum + eps) * (which == 0 ? q_scale : 1.0f);
        for (int i = lane; i < state; i += 32) {
            seg[i] *= scale;
        }
    }
    // The gates are per time-step head, not per group.
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const int64_t idx = static_cast<int64_t>(row) * heads + h;
        const float raw = alpha[idx] + dt[h];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        alpha[idx] = softplus * a[h];
        beta[idx] = 1.0f / (1.0f + expf(-beta[idx]));
    }
}

// gdn_prep_f32_kernel builds the triangular system of one (chunk, head) and
// writes its inverse and Z. Rows past the end of the chunk stay zero, so a
// partial last chunk needs no separate path.
__global__ void gdn_prep_f32_kernel(const float * conv,
                                    const float * beta, float * scratch,
                                    int rows, int groups, int heads, int state,
                                    int conv_dim) {
    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    const int c0 = chunk * kGdnChunk;
    const int n = min(kGdnChunk, rows - c0);
    const int tid = threadIdx.x;
    const int key_dim = groups * state;
    const int kh = head % groups;
    const int row_stride = state + 1;   // odd stride keeps the dot products conflict free

    extern __shared__ float lds[];
    float * kds = lds;                              // [chunk][state], K rows
    float * qds = kds + kGdnChunk * row_stride;      // [chunk][state], Q rows
    float * bds = qds + kGdnChunk * row_stride;      // B, then P
    float * mds = bds + kGdnMatrix;                  // M^-1
    float * zds = mds + kGdnMatrix;                  // Z

    for (int r = 0; r < n; ++r) {
        const float * line = conv + static_cast<int64_t>(c0 + r) * conv_dim;
        float * krow = kds + r * row_stride;
        float * qrow = qds + r * row_stride;
        for (int i = tid; i < state; i += blockDim.x) {
            krow[i] = line[key_dim + kh * state + i];
            qrow[i] = line[kh * state + i];
        }
    }
    for (int idx = tid; idx < kGdnMatrix; idx += blockDim.x) {
        if (idx % kGdnStride < kGdnChunk) {
            bds[idx] = 0.0f;
            mds[idx] = 0.0f;
            zds[idx] = 0.0f;
        }
    }
    __syncthreads();

    // B[r][s] = beta_r (k_r . k_s), strictly lower.
    for (int pair = tid; pair < kGdnChunk * (kGdnChunk - 1) / 2; pair += blockDim.x) {
        int r = 0;
        int s = 0;
        decodeStrictLower(pair, r, s);
        if (r >= n) {
            continue;
        }
        const float * kr = kds + r * row_stride;
        const float * ks = kds + s * row_stride;
        float sum = 0.0f;
        for (int i = 0; i < state; ++i) {
            sum = fmaf(kr[i], ks[i], sum);
        }
        bds[r * kGdnStride + s] = beta[static_cast<int64_t>(c0 + r) * heads + head] * sum;
    }
    __syncthreads();

    // M^-1 by forward substitution: one wave owns one column of the inverse and
    // keeps the rows it has resolved in registers.
    if (tid < kGdnChunk) {
        const int col = tid;
        float column[kGdnChunk];
        #pragma unroll
        for (int r = 0; r < kGdnChunk; ++r) {
            column[r] = 0.0f;
        }
        column[col] = 1.0f;
        for (int r = col + 1; r < n; ++r) {
            float acc = 0.0f;
            for (int u = col; u < r; ++u) {
                acc = fmaf(bds[r * kGdnStride + u], column[u], acc);
            }
            column[r] = -acc;
        }
        for (int r = col; r < n; ++r) {
            mds[r * kGdnStride + col] = column[r];
        }
    }
    __syncthreads();

    // P[r][s] = q_r . k_s for s <= r, over the B block.
    for (int pair = tid; pair < kGdnChunk * (kGdnChunk + 1) / 2; pair += blockDim.x) {
        int r = 0;
        int s = 0;
        decodeLower(pair, r, s);
        if (r >= n) {
            continue;
        }
        const float * qr = qds + r * row_stride;
        const float * ks = kds + s * row_stride;
        float sum = 0.0f;
        for (int i = 0; i < state; ++i) {
            sum = fmaf(qr[i], ks[i], sum);
        }
        bds[r * kGdnStride + s] = sum;
    }
    __syncthreads();

    // Z = (P on the lower triangle) M^-1, lower triangular as well.
    for (int pair = tid; pair < kGdnChunk * (kGdnChunk + 1) / 2; pair += blockDim.x) {
        int r = 0;
        int s = 0;
        decodeLower(pair, r, s);
        if (r >= n) {
            continue;
        }
        float sum = 0.0f;
        for (int u = s; u <= r; ++u) {
            sum = fmaf(bds[r * kGdnStride + u], mds[u * kGdnStride + s], sum);
        }
        zds[r * kGdnStride + s] = sum;
    }
    __syncthreads();

    float * out = scratch + (static_cast<int64_t>(chunk) * heads + head) * 2 *
                                   kGdnChunk * kGdnChunk;
    for (int idx = tid; idx < kGdnChunk * kGdnChunk; idx += blockDim.x) {
        const int r = idx / kGdnChunk;
        const int s = idx % kGdnChunk;
        const bool live = r < n && s <= r;
        out[idx] = live ? mds[r * kGdnStride + s] : 0.0f;
        out[kGdnChunk * kGdnChunk + idx] = live ? zds[r * kGdnStride + s] : 0.0f;
    }
}

// projectState forms X S_0 for one staged chunk tile and writes it to
// dst[r][column .. column + kGdnColsPerThread) on the part-zero lanes. Four
// chunk rows are accumulated at a time so only four registers per column are
// live, and one float4 of the tile feeds one multiply-add per column.
__device__ __forceinline__ void projectState(const float * tile, const float * cell,
                                             int part, int column, float * dst) {
    #pragma unroll
    for (int q = 0; q < kGdnChunk / 4; ++q) {
        float a[kGdnColsPerThread][4];
        #pragma unroll
        for (int c = 0; c < kGdnColsPerThread; ++c) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                a[c][t] = 0.0f;
            }
        }
        #pragma unroll
        for (int j = 0; j < kGdnPartRows; ++j) {
            const float4 k = *reinterpret_cast<const float4 *>(tile + krtOffset(part, j) + 4 * q);
            #pragma unroll
            for (int c = 0; c < kGdnColsPerThread; ++c) {
                const float s = cell[c * kGdnPartRows + j];
                a[c][0] = fmaf(k.x, s, a[c][0]);
                a[c][1] = fmaf(k.y, s, a[c][1]);
                a[c][2] = fmaf(k.z, s, a[c][2]);
                a[c][3] = fmaf(k.w, s, a[c][3]);
            }
        }
        #pragma unroll
        for (int c = 0; c < kGdnColsPerThread; ++c) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                #pragma unroll
                for (int mask = 1; mask < kGdnParts; mask <<= 1) {
                    a[c][t] += __shfl_xor_sync(0xffffffffffffffffull, a[c][t], mask);
                }
            }
        }
        if (part == 0) {
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                #pragma unroll
                for (int c = 0; c < kGdnColsPerThread; ++c) {
                    dst[(4 * q + t) * kGdnStride + column + c] = a[c][t];
                }
            }
        }
    }
}

// stageTile copies the chunk rows of one q or k head into the LDS tile in the
// layout krtOffset expects.
__device__ __forceinline__ void stageTile(float * tile, const float * conv, int c0,
                                          int n, int offset, int row_stride) {
    for (int idx = threadIdx.x; idx < kGdnChunk * kGdnState; idx += blockDim.x) {
        const int r = idx / kGdnState;
        const int i = idx % kGdnState;
        float value = 0.0f;
        if (r < n) {
            value = conv[static_cast<int64_t>(c0 + r) * row_stride + offset + i];
        }
        tile[krtOffset(i / kGdnPartRows, i % kGdnPartRows) + r] = value;
    }
}

// gdn_scan_f32_kernel walks the chunks of one head in order. A thread owns one
// state column and one of eight row parts, so the state slice stays in
// registers and the state projections only reduce across the eight parts of its
// column, which are lanes of the same wave.
__global__ void gdn_scan_f32_kernel(float * rec, const float * conv,
                                    const float * alpha, const float * beta,
                                    const float * scratch, float * out, int rows,
                                    int groups, int heads, int state,
                                    int conv_dim, int inner) {
    const int head = blockIdx.x;
    const int col_base = blockIdx.y * kGdnCols;
    const int tid = threadIdx.x;
    const int colgroup = tid / kGdnParts;
    const int part = tid % kGdnParts;
    const int col = colgroup * kGdnColsPerThread;
    const int state_col = col_base + col;
    const int first_row = part * kGdnPartRows;
    const int key_dim = groups * state;
    const int kh = head % groups;
    const int chunks = (rows + kGdnChunk - 1) / kGdnChunk;

    extern __shared__ float lds[];
    float * kds = lds;                              // [state][chunk], K then Q
    float * mds = kds + kGdnTile;                   // M^-1
    float * zds = mds + kGdnMatrix;                 // Z
    float * proj = zds + kGdnMatrix;                // K S_0
    float * xtmp = proj + kGdnMatrix;               // triangular solve operand
    float * valds = xtmp + kGdnMatrix;              // output operand
    float * vnt = valds + kGdnMatrix;               // corrected values by column
    float * qds = vnt + kGdnMatrix;                 // Q S_0, saved across the update
    __shared__ float decay_log[kGdnChunk];

    float * matrix = rec + static_cast<int64_t>(head) * state * state;
    // cell[c][j] holds state row first_row + j of column state_col + c.
    float cell[kGdnColsPerThread][kGdnPartRows];
    #pragma unroll
    for (int c = 0; c < kGdnColsPerThread; ++c) {
        #pragma unroll
        for (int j = 0; j < kGdnPartRows; ++j) {
            cell[c][j] = matrix[(first_row + j) * state + state_col + c];
        }
    }

    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int c0 = chunk * kGdnChunk;
        const int n = min(kGdnChunk, rows - c0);
        if (tid == 0) {
            float acc = 0.0f;
            for (int r = 0; r < n; ++r) {
                acc += alpha[static_cast<int64_t>(c0 + r) * heads + head];
                decay_log[r] = acc;
            }
        }
        for (int idx = tid; idx < kGdnMatrix; idx += blockDim.x) {
            const int r = idx / kGdnStride;
            const int s = idx % kGdnStride;
            const float * src = scratch +
                (static_cast<int64_t>(chunk) * heads + head) * 2 * kGdnChunk * kGdnChunk;
            mds[idx] = s < kGdnChunk ? src[r * kGdnChunk + s] : 0.0f;
            zds[idx] = s < kGdnChunk ? src[kGdnChunk * kGdnChunk + r * kGdnChunk + s] : 0.0f;
        }
        stageTile(kds, conv, c0, n, kh * state, conv_dim);
        __syncthreads();
        const float gc = decay_log[n - 1];
        const float decay = expf(gc);

        // Q S_0 comes first: the output rows refer to the chunk-start state and
        // the state update below overwrites it.
        projectState(kds, &cell[0][0], part, col, qds);


        __syncthreads();
        stageTile(kds, conv, c0, n, key_dim + kh * state, conv_dim);
        __syncthreads();
        projectState(kds, &cell[0][0], part, col, proj);


        // xi = beta_r (exp(g_c - g_r) v_r - exp(g_c) (K S_0)_r) and the output
        // operand beta_r (exp(-g_r) v_r - (K S_0)_r), four rows per thread.
        for (int j = 0; j < kGdnRowsPerThread; ++j) {
            const int r = part * kGdnRowsPerThread + j;
            for (int c = 0; c < kGdnColsPerThread; ++c) {
                float xi = 0.0f;
                float operand = 0.0f;
                if (r < n) {
                    const float value = conv[static_cast<int64_t>(c0 + r) * conv_dim +
                                             2 * key_dim + head * state + state_col + c];
                    const float base = proj[r * kGdnStride + col + c];
                    const float br = beta[static_cast<int64_t>(c0 + r) * heads + head];
                    const float g = decay_log[r];
                    xi = br * (expf(gc - g) * value - decay * base);
                    operand = br * (expf(-g) * value - base);
                }
                xtmp[r * kGdnStride + col + c] = xi;
                valds[r * kGdnStride + col + c] = operand;
            }
        }
        __syncthreads();

        // Corrected values M^-1 xi, written by column for the state update.
        for (int j = 0; j < kGdnRowsPerThread; ++j) {
            const int r = part * kGdnRowsPerThread + j;
            const float * mrow = mds + r * kGdnStride;
            #pragma unroll
            for (int c = 0; c < kGdnColsPerThread; ++c) {
                const float * xcol = xtmp + col + c;
                // Four partial sums break the triangular chain every four rows:
                // one long chain leaves every load on the critical path.
                float p0 = 0.0f;
                float p1 = 0.0f;
                float p2 = 0.0f;
                float p3 = 0.0f;
                #pragma unroll
                for (int s = 0; s < kGdnChunk; s += 4) {
                    p0 = fmaf(s + 0 <= r ? mrow[s + 0] : 0.0f, xcol[(s + 0) * kGdnStride], p0);
                    p1 = fmaf(s + 1 <= r ? mrow[s + 1] : 0.0f, xcol[(s + 1) * kGdnStride], p1);
                    p2 = fmaf(s + 2 <= r ? mrow[s + 2] : 0.0f, xcol[(s + 2) * kGdnStride], p2);
                    p3 = fmaf(s + 3 <= r ? mrow[s + 3] : 0.0f, xcol[(s + 3) * kGdnStride], p3);
                }
                vnt[(col + c) * kGdnStride + r] = (p0 + p1) + (p2 + p3);
            }
        }
        __syncthreads();

        // S = exp(g_c) S_0 + sum_r k_r val_r^T.
        #pragma unroll
        for (int c = 0; c < kGdnColsPerThread; ++c) {
            #pragma unroll
            for (int j = 0; j < kGdnPartRows; ++j) {
                cell[c][j] *= decay;
            }
        }
        // The chunk-row loop stays rolled: unrolling all 128 float4 loads of
        // the update pushes the kernel past the register budget and spills.
        #pragma unroll
        for (int j = 0; j < kGdnPartRows; ++j) {
            const float4 * krow = reinterpret_cast<const float4 *>(kds + krtOffset(part, j));
            #pragma unroll 1
            for (int q = 0; q < kGdnChunk / 4; ++q) {
                const float4 k = krow[q];
                #pragma unroll
                for (int c = 0; c < kGdnColsPerThread; ++c) {
                    const float4 v = *reinterpret_cast<const float4 *>(
                        vnt + (col + c) * kGdnStride + 4 * q);
                    cell[c][j] = fmaf(k.x, v.x, cell[c][j]);
                    cell[c][j] = fmaf(k.y, v.y, cell[c][j]);
                    cell[c][j] = fmaf(k.z, v.z, cell[c][j]);
                    cell[c][j] = fmaf(k.w, v.w, cell[c][j]);
                }
            }
        }

        // o_r = exp(g_r) (q_r S_0 + Z (diag(beta exp(-g)) V - K S_0)).
        for (int j = 0; j < kGdnRowsPerThread; ++j) {
            const int r = part * kGdnRowsPerThread + j;
            if (r >= n) {
                continue;
            }
            const float * zrow = zds + r * kGdnStride;
            #pragma unroll
            for (int c = 0; c < kGdnColsPerThread; ++c) {
                const float * vcol = valds + col + c;
                float p0 = 0.0f;
                float p1 = 0.0f;
                float p2 = 0.0f;
                float p3 = 0.0f;
                #pragma unroll
                for (int s = 0; s < kGdnChunk; s += 4) {
                    p0 = fmaf(s + 0 <= r ? zrow[s + 0] : 0.0f, vcol[(s + 0) * kGdnStride], p0);
                    p1 = fmaf(s + 1 <= r ? zrow[s + 1] : 0.0f, vcol[(s + 1) * kGdnStride], p1);
                    p2 = fmaf(s + 2 <= r ? zrow[s + 2] : 0.0f, vcol[(s + 2) * kGdnStride], p2);
                    p3 = fmaf(s + 3 <= r ? zrow[s + 3] : 0.0f, vcol[(s + 3) * kGdnStride], p3);
                }
                const float sum = (p0 + p1) + (p2 + p3);
                out[static_cast<int64_t>(c0 + r) * inner + head * state + state_col + c] =
                    expf(decay_log[r]) * (qds[r * kGdnStride + col + c] + sum);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < kGdnColsPerThread; ++c) {
        #pragma unroll
        for (int j = 0; j < kGdnPartRows; ++j) {
            matrix[(first_row + j) * state + state_col + c] = cell[c][j];
        }
    }
}

} // namespace

namespace yuuka {

// launch_gdn_chunked_f32 runs the chunkwise gated delta rule for one recurrent
// layer. scratch must hold two 32x32 matrices per (chunk, head) of triangular
// intermediate data and is owned by the caller.
void launch_gdn_chunked_f32(float * rec, float * conv, float * alpha,
                            float * beta, const float * dt, const float * a,
                            float * out, float * scratch, int rows, int groups,
                            int heads, int state, int conv_dim, int inner,
                            float eps, float q_scale, hipStream_t stream) {
    const int chunks = (rows + kGdnChunk - 1) / kGdnChunk;
    gdn_prepare_rows_f32_kernel<<<rows, kGdnThreads, 0, stream>>>(
        conv, alpha, beta, dt, a, rows, groups, heads, state, conv_dim, eps,
        q_scale);
    const size_t prep_shared =
        size_t(2 * kGdnChunk * (state + 1) + 3 * kGdnMatrix) * sizeof(float);
    gdn_prep_f32_kernel<<<dim3(static_cast<unsigned>(chunks),
                               static_cast<unsigned>(heads)),
                          kGdnThreads, prep_shared, stream>>>(
        conv, beta, scratch, rows, groups, heads, state, conv_dim);
    // The scan keeps the state slice in registers, so its dynamic shared memory
    // only holds the staged chunk tiles and the triangular matrices.
    const size_t scan_shared = size_t(kGdnTile + 7 * kGdnMatrix) * sizeof(float);
    gdn_scan_f32_kernel<<<dim3(static_cast<unsigned>(heads),
                               static_cast<unsigned>(state / kGdnCols)),
                          kGdnThreads, scan_shared, stream>>>(
        rec, conv, alpha, beta, scratch, out, rows, groups, heads, state, conv_dim,
        inner);
}

} // namespace yuuka
