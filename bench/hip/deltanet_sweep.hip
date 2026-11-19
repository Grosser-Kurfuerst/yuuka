// deltanet_sweep measures the gated delta-net per-token kernels at the exact
// shape one Qwen3.5-4B Q4_K_M recurrent layer dispatches
// (32 heads, 16 groups, 128-wide state), so the choice between the register
// resident and shared-memory resident forms is made on measured time and not on
// the layout argument alone.
//
// Every form is a standalone copy of the kernel it measures, so the sweep does
// not depend on an implementation being shipped in the library, and the kernels
// are timed with HIP events around a single launch rather than from the host, so
// the printed figure is kernel time. The copy cell is the floor of the
// read-once/write-once pattern every fused form has to reach.
//
// Variants:
//   legacy   the five-pass kernel of kernels/hip/ssm/ssm.hip: decay, S^T k,
//            rank-1 update, S^T q over global memory.
//   copy     read and write the state once, nothing else: the floor.
//   regN     one column of the state per thread split across N thread groups,
//            register resident, decay and update fused in one pass, with the
//            gated RMSNorm of the head folded in.
//   regN-nn  the same without the gated RMSNorm, to price that fusion.
//   lds      the whole 128x128 state in shared memory, one column per thread,
//            five traversals of shared memory, no gated RMSNorm (the 64 KB
//            state fills the workgroup budget).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int kStateSize = 128;
constexpr int kHeads = 32;
constexpr int kGroups = 16;
constexpr int kThreads = 256;
constexpr int kWarmup = 5;
constexpr int kTimed = 25;
constexpr int kLayers = 24;

void check(hipError_t err, const char * what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(err));
        std::exit(1);
    }
}

__device__ __forceinline__ float wave_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffffffffffull, value, offset);
    }
    return value;
}

__device__ __forceinline__ void stage_decay(float decay, float beta, float * hdr) {
    hdr[0] = decay;
    hdr[1] = beta;
}

// legacy_kernel is the five-pass kernel the decode plan used before the fusion.
__global__ void legacy_kernel(float * state, const float * q, const float * k,
                              const float * v, const float * alpha, const float * beta_raw,
                              const float * dt, const float * a, float * out,
                              int groups, int heads, int state_size) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    extern __shared__ float smem[];
    float * qs = smem;
    float * ks = smem + state_size;
    float * ds = smem + 2 * state_size;
    __shared__ float decay_shared;
    __shared__ float beta_shared;
    const int tid = threadIdx.x;
    const int kh = head % groups;
    qs[tid] = q[kh * state_size + tid];
    ks[tid] = k[kh * state_size + tid];
    const float value = v[head * state_size + tid];
    if (tid == 0) {
        const float raw = alpha[head] + dt[head];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        decay_shared = expf(softplus * a[head]);
        beta_shared = 1.0f / (1.0f + expf(-beta_raw[head]));
    }
    __syncthreads();
    const float decay = decay_shared;
    const float beta = beta_shared;
    float * matrix = state + head * state_size * state_size;
    for (int j = 0; j < state_size; ++j) {
        matrix[tid * state_size + j] *= decay;
    }
    __syncthreads();
    float sk = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        sk += matrix[i * state_size + tid] * ks[i];
    }
    ds[tid] = (value - sk) * beta;
    __syncthreads();
    const float ki = ks[tid];
    for (int j = 0; j < state_size; ++j) {
        matrix[tid * state_size + j] += ki * ds[j];
    }
    __syncthreads();
    float result = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        result += matrix[i * state_size + tid] * qs[i];
    }
    out[head * state_size + tid] = result;
}

// copy_kernel reads and writes each state element once: the floor of the
// read-once/write-once pattern.
__global__ void copy_kernel(float * state, int heads) {
    const int head = blockIdx.x;
    if (head >= heads) return;
    float * matrix = state + head * kStateSize * kStateSize;
    for (int i = threadIdx.x; i < kStateSize * kStateSize; i += blockDim.x) {
        const float value = matrix[i];
        matrix[i] = value * 1.0000001f;
    }
}

// reg_kernel keeps one column of the head's state in registers. kSplits thread
// groups share a column, each holding kStateSize/kSplits rows of it, so the
// decay, the two S^T products, and the rank-1 update are one pass over the
// registers with a single state read and a single state write.
template <int kSplits, bool kFuseNorm>
__global__ void reg_kernel(float * state, const float * q, const float * k,
                           const float * v, const float * alpha, const float * beta_raw,
                           const float * dt, const float * a, float * out,
                           const float * norm_weight, const float * gate,
                           int groups, int heads, float eps) {
    constexpr int kRows = kStateSize / kSplits;
    constexpr int kThreads = kStateSize * kSplits;
    extern __shared__ float smem[];
    float * ksh = smem;
    float * qsh = smem + kStateSize;
    float * part = qsh + kStateSize;
    float * res_part = part + kThreads;
    float * dsh = res_part + kThreads;
    float * red = dsh + kStateSize;
    float * nrm = red + kThreads / 32;

    const int head = blockIdx.x;
    if (head >= heads) return;
    const int tid = threadIdx.x;
    const int col = tid % kStateSize;
    const int slice = tid / kStateSize;
    const int i0 = slice * kRows;
    const int kh = head % groups;
    const float * qh = q + kh * kStateSize;
    const float * khs = k + kh * kStateSize;
    float * m = state + size_t(head) * kStateSize * kStateSize;

    float s[kRows];
    #pragma unroll
    for (int j = 0; j < kRows; ++j) s[j] = m[(i0 + j) * kStateSize + col];
    if (tid < kStateSize) {
        ksh[tid] = khs[tid];
        qsh[tid] = qh[tid];
    }
    __shared__ float hdr[2];
    if (tid == 0) {
        const float raw = alpha[head] + dt[head];
        const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
        stage_decay(expf(softplus * a[head]), 1.0f / (1.0f + expf(-beta_raw[head])), hdr);
    }
    __syncthreads();
    const float decay = hdr[0];
    const float beta = hdr[1];

    // S^T k over the decayed state: the decay is folded into the total instead
    // of into every element, which is the same product with one rounding.
    float sk = 0.0f;
    #pragma unroll
    for (int j = 0; j < kRows; ++j) sk += s[j] * ksh[i0 + j];
    part[tid] = sk;
    __syncthreads();
    #pragma unroll
    for (int split = 1; split < kSplits; ++split) sk += part[tid ^ (split * kStateSize)];
    sk *= decay;
    if (slice == 0) dsh[col] = (v[head * kStateSize + col] - sk) * beta;
    __syncthreads();
    const float dc = dsh[col];
    float res = 0.0f;
    #pragma unroll
    for (int j = 0; j < kRows; ++j) {
        const float nv = s[j] * decay + ksh[i0 + j] * dc;
        s[j] = nv;
        res += nv * qsh[i0 + j];
    }
    #pragma unroll
    for (int j = 0; j < kRows; ++j) m[(i0 + j) * kStateSize + col] = s[j];
    res_part[tid] = res;
    __syncthreads();
    #pragma unroll
    for (int split = 1; split < kSplits; ++split) res += res_part[tid ^ (split * kStateSize)];

    if (!kFuseNorm) {
        if (slice == 0) out[head * kStateSize + col] = res;
        return;
    }
    // The gated RMSNorm of a head is over that head's own stateSize values, so
    // the block that owns the head also owns the whole reduction.
    if (slice == 0) {
        float sq = wave_sum(res * res);
        if ((tid & 31) == 0) red[tid >> 5] = sq;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < kStateSize / 32; ++w) total += red[w];
        nrm[0] = rsqrtf(total / static_cast<float>(kStateSize) + eps);
    }
    __syncthreads();
    if (slice == 0) {
        const float g = gate[head * kStateSize + col];
        out[head * kStateSize + col] =
            res * nrm[0] * norm_weight[col] * (g / (1.0f + expf(-g)));
    }
}

// lds_kernel holds the whole 128x128 state of the head in shared memory: one
// column per thread, five traversals of shared memory instead of global memory.
// The 64 KB state fills the workgroup budget, so the gated RMSNorm stays a
// separate pass and this form writes the plain delta-net output. q and k are
// read from global memory because there is no shared memory left to stage them.
__global__ void lds_kernel(float * state, const float * q, const float * k,
                           const float * v, const float * alpha, const float * beta_raw,
                           const float * dt, const float * a, float * out,
                           int groups, int heads) {
    extern __shared__ float m[];
    const int head = blockIdx.x;
    if (head >= heads) return;
    const int tid = threadIdx.x;
    float * src = state + size_t(head) * kStateSize * kStateSize;
    for (int i = tid; i < kStateSize * kStateSize; i += blockDim.x) m[i] = src[i];
    const int kh = head % groups;
    const float * qh = q + kh * kStateSize;
    const float * khs = k + kh * kStateSize;
    // The state fills the whole workgroup shared memory budget, so the two
    // scalars are computed redundantly by every thread instead of staged.
    const float raw = alpha[head] + dt[head];
    const float softplus = raw > 20.0f ? raw : log1pf(expf(raw));
    const float decay = expf(softplus * a[head]);
    const float beta = 1.0f / (1.0f + expf(-beta_raw[head]));
    __syncthreads();
    const float value = v[head * kStateSize + tid];
    for (int i = 0; i < kStateSize; ++i) m[i * kStateSize + tid] *= decay;
    float sk = 0.0f;
    for (int i = 0; i < kStateSize; ++i) sk += m[i * kStateSize + tid] * khs[i];
    const float d = (value - sk) * beta;
    for (int i = 0; i < kStateSize; ++i) m[i * kStateSize + tid] += khs[i] * d;
    float result = 0.0f;
    for (int i = 0; i < kStateSize; ++i) result += m[i * kStateSize + tid] * qh[i];
    out[head * kStateSize + tid] = result;
    // Each thread owns its column, so the passes above need no barrier; the
    // coalesced store back does, because it reads other threads' columns.
    __syncthreads();
    for (int i = tid; i < kStateSize * kStateSize; i += blockDim.x) src[i] = m[i];
}

struct Variant {
    const char * name;
    int kind;  // 0 legacy, 1 copy, 2 register with norm, 3 lds, 4 register without norm
    int splits;
};

size_t shared_for(int splits) {
    const int threads = kStateSize * splits;
    return size_t(2 * kStateSize + 2 * threads + kStateSize + threads / 32 + 1) * sizeof(float);
}

void launch(const Variant & variant, float * state, const float * q, const float * k,
            const float * v, const float * alpha, const float * beta, const float * dt,
            const float * a, float * out, const float * norm_weight, const float * gate,
            float eps) {
    const bool fuse = variant.kind != 4;
    switch (variant.splits) {
        case 1:
            if (variant.kind == 0) {
                legacy_kernel<<<kHeads, kStateSize, size_t(3 * kStateSize) * sizeof(float)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, kGroups, kHeads, kStateSize);
            } else if (variant.kind == 1) {
                copy_kernel<<<kHeads, kThreads>>>(state, kHeads);
            } else if (variant.kind == 3) {
                check(hipFuncSetAttribute(reinterpret_cast<const void *>(lds_kernel),
                                          hipFuncAttributeMaxDynamicSharedMemorySize,
                                          kStateSize * kStateSize * int(sizeof(float))),
                      "lds attribute");
                lds_kernel<<<kHeads, kStateSize,
                             size_t(kStateSize * kStateSize) * sizeof(float)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, kGroups, kHeads);
            } else if (fuse) {
                reg_kernel<1, true><<<kHeads, kStateSize, shared_for(1)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            } else {
                reg_kernel<1, false><<<kHeads, kStateSize, shared_for(1)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            }
            break;
        case 2:
            if (fuse) {
                reg_kernel<2, true><<<kHeads, 2 * kStateSize, shared_for(2)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            } else {
                reg_kernel<2, false><<<kHeads, 2 * kStateSize, shared_for(2)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            }
            break;
        default:
            if (fuse) {
                reg_kernel<4, true><<<kHeads, 4 * kStateSize, shared_for(4)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            } else {
                reg_kernel<4, false><<<kHeads, 4 * kStateSize, shared_for(4)>>>(
                    state, q, k, v, alpha, beta, dt, a, out, norm_weight, gate, kGroups,
                    kHeads, eps);
            }
            break;
    }
}

// State bytes one token moves through the kernel: the five-pass form reads the
// state three times and writes it twice, the fused forms once each.
double state_bytes(const Variant & variant) {
    const double per_head = double(kStateSize) * kStateSize * sizeof(float);
    if (variant.kind == 0) return 5.0 * per_head * kHeads;
    if (variant.kind == 3) return 2.0 * per_head * kHeads;  // global traffic only
    return 2.0 * per_head * kHeads;
}

}  // namespace

int main() {
    hipDeviceProp_t prop{};
    check(hipGetDeviceProperties(&prop, 0), "device properties");
    std::printf("# device %s, %d CU, %.2f GHz, %zu KB shared per block (%zu KB opt-in)\n",
                prop.gcnArchName, prop.multiProcessorCount, prop.clockRate / 1.0e6,
                size_t(prop.sharedMemPerBlock) >> 10, size_t(prop.sharedMemPerBlockOptin) >> 10);
    std::printf("# shape: %d heads, %d groups, %d-wide state, %d layers\n", kHeads, kGroups,
                kStateSize, kLayers);
    std::printf("# legacy traffic counts five state traversals, the fused forms two\n");

    const size_t state_elems = size_t(kHeads) * kStateSize * kStateSize;
    const size_t head_elems = size_t(kHeads) * kStateSize;
    const size_t group_elems = size_t(kGroups) * kStateSize;
    std::vector<float> host_state(state_elems), host_q(group_elems), host_k(group_elems),
        host_v(head_elems), host_gate(head_elems), host_norm(kStateSize);
    std::mt19937 rng(0x5eed);
    std::uniform_real_distribution<float> small(-0.05f, 0.05f);
    for (float & value : host_state) value = small(rng);
    for (float & value : host_q) value = small(rng);
    for (float & value : host_k) value = small(rng);
    for (float & value : host_v) value = small(rng);
    for (float & value : host_gate) value = small(rng);
    for (float & value : host_norm) value = 1.0f + small(rng) * 0.1f;

    float * dev_state = nullptr;
    float * dev_q = nullptr;
    float * dev_k = nullptr;
    float * dev_v = nullptr;
    float * dev_out = nullptr;
    float * dev_gate = nullptr;
    float * dev_norm = nullptr;
    check(hipMalloc(&dev_state, state_elems * sizeof(float)), "alloc state");
    check(hipMalloc(&dev_q, group_elems * sizeof(float)), "alloc q");
    check(hipMalloc(&dev_k, group_elems * sizeof(float)), "alloc k");
    check(hipMalloc(&dev_v, head_elems * sizeof(float)), "alloc v");
    check(hipMalloc(&dev_out, head_elems * sizeof(float)), "alloc out");
    check(hipMalloc(&dev_gate, head_elems * sizeof(float)), "alloc gate");
    check(hipMalloc(&dev_norm, kStateSize * sizeof(float)), "alloc norm");
    check(hipMemcpy(dev_q, host_q.data(), group_elems * sizeof(float), hipMemcpyHostToDevice), "h2d q");
    check(hipMemcpy(dev_k, host_k.data(), group_elems * sizeof(float), hipMemcpyHostToDevice), "h2d k");
    check(hipMemcpy(dev_v, host_v.data(), head_elems * sizeof(float), hipMemcpyHostToDevice), "h2d v");
    check(hipMemcpy(dev_gate, host_gate.data(), head_elems * sizeof(float), hipMemcpyHostToDevice), "h2d gate");
    check(hipMemcpy(dev_norm, host_norm.data(), kStateSize * sizeof(float), hipMemcpyHostToDevice), "h2d norm");

    // The alpha/beta/dt/a tensors hold one value per head. Small values keep the
    // decay inside (0, 1) and the state bounded over repeated submissions.
    std::vector<float> host_alpha(kHeads), host_beta(kHeads), host_dt(kHeads), host_a(kHeads);
    for (int h = 0; h < kHeads; ++h) {
        host_alpha[h] = 0.5f + 0.01f * float(h % 7);
        host_beta[h] = 0.5f + 0.01f * float(h % 5);
        host_dt[h] = 0.05f;
        host_a[h] = -0.3f;
    }
    float * dev_alpha = nullptr;
    float * dev_beta = nullptr;
    float * dev_dt = nullptr;
    float * dev_a = nullptr;
    check(hipMalloc(&dev_alpha, kHeads * sizeof(float)), "alloc alpha");
    check(hipMalloc(&dev_beta, kHeads * sizeof(float)), "alloc beta");
    check(hipMalloc(&dev_dt, kHeads * sizeof(float)), "alloc dt");
    check(hipMalloc(&dev_a, kHeads * sizeof(float)), "alloc a");
    check(hipMemcpy(dev_alpha, host_alpha.data(), kHeads * sizeof(float), hipMemcpyHostToDevice), "h2d alpha");
    check(hipMemcpy(dev_beta, host_beta.data(), kHeads * sizeof(float), hipMemcpyHostToDevice), "h2d beta");
    check(hipMemcpy(dev_dt, host_dt.data(), kHeads * sizeof(float), hipMemcpyHostToDevice), "h2d dt");
    check(hipMemcpy(dev_a, host_a.data(), kHeads * sizeof(float), hipMemcpyHostToDevice), "h2d a");

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check(hipEventCreate(&start), "event create");
    check(hipEventCreate(&stop), "event create");

    const Variant variants[] = {
        {"legacy", 0, 1},
        {"copy", 1, 1},
        {"reg1", 2, 1},
        {"reg2", 2, 2},
        {"reg4", 2, 4},
        {"reg2-nn", 4, 2},
        {"reg4-nn", 4, 4},
        {"lds", 3, 1},
    };
    const float eps = 1e-6f;
    double legacy_ms = 0.0;
    std::printf("%-8s %11s %9s %9s %12s\n", "variant", "ms/layer", "GB/s", "x legacy",
                "us/token 24L");
    for (const Variant & variant : variants) {
        check(hipMemcpy(dev_state, host_state.data(), state_elems * sizeof(float),
                        hipMemcpyHostToDevice),
              "h2d state");
        for (int i = 0; i < kWarmup; ++i) {
            launch(variant, dev_state, dev_q, dev_k, dev_v, dev_alpha, dev_beta, dev_dt,
                   dev_a, dev_out, dev_norm, dev_gate, eps);
        }
        check(hipDeviceSynchronize(), "warmup sync");
        std::vector<double> samples;
        for (int i = 0; i < kTimed; ++i) {
            check(hipEventRecord(start), "record");
            launch(variant, dev_state, dev_q, dev_k, dev_v, dev_alpha, dev_beta, dev_dt,
                   dev_a, dev_out, dev_norm, dev_gate, eps);
            check(hipEventRecord(stop), "record");
            check(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            check(hipEventElapsedTime(&ms, start, stop), "elapsed");
            samples.push_back(ms);
        }
        std::sort(samples.begin(), samples.end());
        const double ms = samples[samples.size() / 2];
        if (variant.kind == 0) legacy_ms = ms;
        std::printf("%-8s %11.4f %9.1f %9.2f %12.1f\n", variant.name, ms,
                    state_bytes(variant) / (ms * 1.0e6), legacy_ms > 0.0 ? legacy_ms / ms : 0.0,
                    ms * kLayers * 1000.0);
    }

    // Agreement: the fused forms must reproduce the legacy kernel's output and
    // the state it leaves behind. Each form starts from the same input state.
    auto run_once = [&](const Variant & variant, std::vector<float> & out,
                        std::vector<float> & state_out) {
        check(hipMemcpy(dev_state, host_state.data(), state_elems * sizeof(float),
                        hipMemcpyHostToDevice),
              "h2d state");
        std::vector<float> zero(head_elems, 0.0f);
        check(hipMemcpy(dev_out, zero.data(), head_elems * sizeof(float), hipMemcpyHostToDevice),
              "h2d out");
        launch(variant, dev_state, dev_q, dev_k, dev_v, dev_alpha, dev_beta, dev_dt,
               dev_a, dev_out, dev_norm, dev_gate, eps);
        check(hipDeviceSynchronize(), "ref sync");
        out.assign(head_elems, 0.0f);
        state_out.assign(state_elems, 0.0f);
        check(hipMemcpy(out.data(), dev_out, head_elems * sizeof(float),
                        hipMemcpyDeviceToHost),
              "d2h out");
        check(hipMemcpy(state_out.data(), dev_state, state_elems * sizeof(float),
                        hipMemcpyDeviceToHost),
              "d2h state");
    };
    std::vector<float> ref_out;
    std::vector<float> ref_state;
    std::vector<float> got_out;
    std::vector<float> got_state;
    run_once(variants[0], ref_out, ref_state);
    // The norm-fused forms write the gated RMSNorm of the delta-net output, so
    // their expected value is that transform of the legacy output computed here
    // from the same values.
    std::vector<float> ref_normed(ref_out.size());
    for (int head = 0; head < kHeads; ++head) {
        double total = 0.0;
        for (int c = 0; c < kStateSize; ++c) {
            const double v = ref_out[head * kStateSize + c];
            total += v * v;
        }
        const double scale = 1.0 / std::sqrt(total / kStateSize + double(eps));
        for (int c = 0; c < kStateSize; ++c) {
            const double v = ref_out[head * kStateSize + c];
            const double g = host_gate[head * kStateSize + c];
            ref_normed[head * kStateSize + c] =
                float(v * scale * host_norm[c] * (g / (1.0 + std::exp(-g))));
        }
    }
    std::printf("# agreement against the legacy kernel: max absolute difference over %zu "
                "output and %zu state values\n", ref_out.size(), ref_state.size());
    for (const Variant & variant : {variants[3], variants[4], variants[6], variants[7]}) {
        run_once(variant, got_out, got_state);
        const bool normed = variant.kind == 2;
        double worst_out = 0.0;
        double worst_state = 0.0;
        for (size_t i = 0; i < ref_out.size(); ++i) {
            worst_out = std::max(worst_out, double(std::fabs((normed ? ref_normed[i] : ref_out[i]) -
                                                             got_out[i])));
        }
        for (size_t i = 0; i < ref_state.size(); ++i) {
            worst_state = std::max(worst_state, double(std::fabs(ref_state[i] - got_state[i])));
        }
        std::printf("# %-8s out %.3e state %.3e (expected value: %s)\n", variant.name, worst_out,
                    worst_state, normed ? "gated RMSNorm of the legacy output" : "legacy output");
    }

    check(hipEventDestroy(start), "event destroy");
    check(hipEventDestroy(stop), "event destroy");
    (void)hipFree(dev_state);
    (void)hipFree(dev_q);
    (void)hipFree(dev_k);
    (void)hipFree(dev_v);
    (void)hipFree(dev_out);
    (void)hipFree(dev_gate);
    (void)hipFree(dev_norm);
    (void)hipFree(dev_alpha);
    (void)hipFree(dev_beta);
    (void)hipFree(dev_dt);
    (void)hipFree(dev_a);
    return 0;
}
