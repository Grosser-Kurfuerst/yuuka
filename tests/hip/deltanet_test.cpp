// Compares the fused gated delta-net operators of roadmap4 D4 against the
// separate operators they replace and against a host reference, on the geometry
// one Qwen3.5-4B recurrent layer dispatches: 8192 qkv channels, a 4-tap
// convolution, 16 groups of a 128-wide state, and 32 heads.
//
// The two plans run on identical buffers, so the comparison covers the
// convolution ring state, the recurrent state, and the normalized output after
// each token. Three tokens are submitted because the states are path dependent:
// a layout or ordering mistake in either kernel shows up as a growing
// difference, not as a constant offset. The host reference is the documented
// order of operations, so a mistake shared by both device paths still fails.
//
// The decay is chosen far from one (about 0.75) on purpose: a great many
// mistakes in this kernel are invisible when the state barely decays, and the
// first version of the fused kernel was one of them.
#include "yuuka_hip.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int64_t kGroups = 16;
constexpr int64_t kHeads = 32;
constexpr int64_t kState = 128;
constexpr int64_t kKernel = 4;
constexpr int64_t kInner = kHeads * kState;
constexpr int64_t kKey = kGroups * kState;
constexpr int64_t kConvDim = 2 * kKey + kInner;
constexpr int kTokens = 3;
constexpr float kEps = 1e-6f;

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

bool upload(yuuka_device * device, const std::vector<float> & host, yuuka_buffer ** out) {
    const uint64_t bytes = uint64_t(host.size()) * sizeof(float);
    return check(yuuka_buffer_alloc(device, bytes, out), "alloc") &&
           check(yuuka_memcpy_h2d(*out, host.data(), bytes), "h2d");
}

bool download(yuuka_buffer * buf, std::vector<float> & host) {
    return check(yuuka_memcpy_d2h(host.data(), buf, uint64_t(host.size()) * sizeof(float)),
                 "d2h");
}

// Fixture holds the buffers both plans share. The inputs are immutable and each
// plan gets its own copies of the two states and of the outputs, so one run
// cannot contaminate the other.
struct Fixture {
    yuuka_buffer * conv_weights = nullptr;
    yuuka_buffer * qkv = nullptr;
    yuuka_buffer * alpha = nullptr;
    yuuka_buffer * beta = nullptr;
    yuuka_buffer * dt = nullptr;
    yuuka_buffer * a = nullptr;
    yuuka_buffer * gate = nullptr;
    yuuka_buffer * norm = nullptr;
    yuuka_buffer * conv_state_legacy = nullptr;
    yuuka_buffer * conv_out_legacy = nullptr;
    yuuka_buffer * rec_state_legacy = nullptr;
    yuuka_buffer * gdn_legacy = nullptr;
    yuuka_buffer * norm_legacy = nullptr;
    yuuka_buffer * conv_state_fused = nullptr;
    yuuka_buffer * conv_out_fused = nullptr;
    yuuka_buffer * rec_state_fused = nullptr;
    yuuka_buffer * norm_fused = nullptr;
};

yuuka_op basis(uint32_t code) {
    yuuka_op op{};
    op.code = code;
    return op;
}

// build_legacy is the unfused plan: CONV1D, the two L2 norms, GDN, and the
// gated RMSNorm.
bool build_legacy(yuuka_plan * plan, const Fixture & f) {
    yuuka_op conv = basis(YUUKA_OP_CONV1D);
    conv.buf[0] = f.conv_state_legacy;
    conv.buf[1] = f.conv_weights;
    conv.buf[2] = f.qkv;
    conv.buf[3] = f.conv_out_legacy;
    conv.n[0] = uint64_t(kConvDim);
    conv.n[1] = uint64_t(kKernel);
    bool ok = check(yuuka_plan_add(plan, &conv), "legacy conv1d");
    for (int pass = 0; pass < 2; ++pass) {
        yuuka_op norm = basis(YUUKA_OP_L2NORM);
        norm.buf[0] = f.conv_out_legacy;
        norm.buf[2] = f.conv_out_legacy;
        norm.off[0] = uint64_t(pass == 0 ? 0 : kKey) * sizeof(float);
        norm.off[2] = norm.off[0];
        norm.n[0] = uint64_t(kGroups);
        norm.n[1] = uint64_t(kState);
        norm.n[2] = uint64_t(kState);
        norm.f[0] = kEps;
        norm.f[1] = pass == 0 ? float(1.0 / std::sqrt(double(kState))) : 1.0f;
        ok = check(yuuka_plan_add(plan, &norm), "legacy l2norm") && ok;
    }
    yuuka_op gdn = basis(YUUKA_OP_GDN);
    gdn.buf[0] = f.rec_state_legacy;
    gdn.buf[1] = f.conv_out_legacy;
    gdn.buf[2] = f.conv_out_legacy;
    gdn.buf[3] = f.conv_out_legacy;
    gdn.buf[4] = f.alpha;
    gdn.buf[5] = f.beta;
    gdn.buf[6] = f.dt;
    gdn.buf[7] = f.a;
    gdn.buf[8] = f.gdn_legacy;
    gdn.off[2] = uint64_t(kKey) * sizeof(float);
    gdn.off[3] = uint64_t(2 * kKey) * sizeof(float);
    gdn.n[0] = uint64_t(kGroups);
    gdn.n[1] = uint64_t(kHeads);
    gdn.n[2] = uint64_t(kState);
    ok = check(yuuka_plan_add(plan, &gdn), "legacy gdn") && ok;
    yuuka_op gnorm = basis(YUUKA_OP_GATED_RMSNORM);
    gnorm.buf[0] = f.gdn_legacy;
    gnorm.buf[1] = f.norm;
    gnorm.buf[2] = f.gate;
    gnorm.buf[3] = f.norm_legacy;
    gnorm.n[0] = uint64_t(kState);
    gnorm.n[1] = uint64_t(kHeads);
    gnorm.n[2] = uint64_t(kState);
    gnorm.f[0] = kEps;
    ok = check(yuuka_plan_add(plan, &gnorm), "legacy gated rmsnorm") && ok;
    return ok;
}

// build_fused is the two-operator plan: CONV_L2NORM and RECURRENT_TOKEN.
bool build_fused(yuuka_plan * plan, const Fixture & f) {
    yuuka_op conv = basis(YUUKA_OP_CONV_L2NORM);
    conv.buf[0] = f.conv_state_fused;
    conv.buf[1] = f.conv_weights;
    conv.buf[2] = f.qkv;
    conv.buf[3] = f.conv_out_fused;
    conv.n[0] = uint64_t(kConvDim);
    conv.n[1] = uint64_t(kKernel);
    conv.n[2] = uint64_t(2 * kKey);
    conv.n[3] = uint64_t(kKey);
    conv.n[4] = uint64_t(kState);
    conv.f[0] = kEps;
    conv.f[1] = float(1.0 / std::sqrt(double(kState)));
    bool ok = check(yuuka_plan_add(plan, &conv), "fused conv_l2norm");
    yuuka_op token = basis(YUUKA_OP_RECURRENT_TOKEN);
    token.buf[0] = f.rec_state_fused;
    token.buf[1] = f.conv_out_fused;
    token.buf[2] = f.conv_out_fused;
    token.buf[3] = f.conv_out_fused;
    token.buf[4] = f.alpha;
    token.buf[5] = f.beta;
    token.buf[6] = f.dt;
    token.buf[7] = f.a;
    token.buf[8] = f.norm_fused;
    token.buf[9] = f.norm;
    token.buf[10] = f.gate;
    token.off[2] = uint64_t(kKey) * sizeof(float);
    token.off[3] = uint64_t(2 * kKey) * sizeof(float);
    token.n[0] = uint64_t(kGroups);
    token.n[1] = uint64_t(kHeads);
    token.n[2] = uint64_t(kState);
    token.f[0] = kEps;
    ok = check(yuuka_plan_add(plan, &token), "fused recurrent token") && ok;
    return ok;
}

// reference_token recomputes one token of the gated delta-net on the host from
// the convolution output the two plans agree on, in the documented order:
// decay the state, form the two S^T products around the rank-1 update, then the
// gated RMSNorm of the head.
void reference_token(const std::vector<float> & conv, const std::vector<float> & alpha,
                     const std::vector<float> & beta, const std::vector<float> & dt,
                     const std::vector<float> & a, const std::vector<float> & gate,
                     const std::vector<float> & weight, std::vector<float> & state,
                     std::vector<float> & out) {
    std::vector<float> q(static_cast<size_t>(kKey));
    std::vector<float> k(static_cast<size_t>(kKey));
    for (size_t i = 0; i < q.size(); ++i) q[i] = conv[i];
    for (size_t i = 0; i < k.size(); ++i) k[i] = conv[static_cast<size_t>(kKey) + i];
    std::vector<float> driven(static_cast<size_t>(kState));
    std::vector<float> row(static_cast<size_t>(kState));
    for (int64_t head = 0; head < kHeads; ++head) {
        const int64_t group = head % kGroups;
        const float raw = alpha[size_t(head)] + dt[size_t(head)];
        const float softplus = raw > 20.0f ? raw : float(std::log1p(std::exp(double(raw))));
        const float decay = float(std::exp(double(softplus) * double(a[size_t(head)])));
        const float sigmoid = 1.0f / (1.0f + float(std::exp(-double(beta[size_t(head)]))));
        float * m = state.data() + size_t(head) * size_t(kState) * size_t(kState);
        for (int64_t i = 0; i < kState * kState; ++i) m[i] *= decay;
        for (int64_t c = 0; c < kState; ++c) {
            float sum = 0.0f;
            for (int64_t i = 0; i < kState; ++i) {
                sum += m[i * kState + c] * k[size_t(group * kState + i)];
            }
            driven[size_t(c)] = (conv[size_t(2 * kKey + head * kState + c)] - sum) * sigmoid;
        }
        for (int64_t i = 0; i < kState; ++i) {
            const float ki = k[size_t(group * kState + i)];
            for (int64_t c = 0; c < kState; ++c) m[i * kState + c] += ki * driven[size_t(c)];
        }
        float sum_sq = 0.0f;
        for (int64_t c = 0; c < kState; ++c) {
            float sum = 0.0f;
            for (int64_t i = 0; i < kState; ++i) {
                sum += m[i * kState + c] * q[size_t(group * kState + i)];
            }
            row[size_t(c)] = sum;
            sum_sq += sum * sum;
        }
        const float scale = 1.0f / std::sqrt(sum_sq / float(kState) + kEps);
        for (int64_t c = 0; c < kState; ++c) {
            const float g = gate[size_t(head * kState + c)];
            out[size_t(head * kState + c)] =
                row[size_t(c)] * scale * weight[size_t(c)] * (g / (1.0f + std::exp(-g)));
        }
    }
}

// worst returns the largest absolute difference and reports the magnitude it was
// measured against.
double worst(const std::vector<float> & want, const std::vector<float> & got, double * scale) {
    double worst_diff = 0.0;
    double largest = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        worst_diff = std::max(worst_diff, double(std::fabs(double(want[i]) - double(got[i]))));
        largest = std::max(largest, double(std::fabs(double(want[i]))));
    }
    *scale = largest;
    return worst_diff;
}

} // namespace

int main() {
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * completion = nullptr;
    if (!check(yuuka_stream_create(device, &stream), "stream") ||
        !check(yuuka_event_create(device, &completion), "event")) {
        return 1;
    }

    std::mt19937 rng(20261118);
    std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
    std::vector<float> host_conv_weights(size_t(kConvDim) * kKernel);
    for (float & value : host_conv_weights) value = 0.3f + uniform(rng);
    std::vector<float> host_qkv(kConvDim);
    for (float & value : host_qkv) value = uniform(rng);
    // alpha, dt, and a are chosen so the decay is about 0.75: the decayed and
    // undecayed state differ enough that a missing factor cannot hide.
    std::vector<float> host_alpha(kHeads), host_beta(kHeads), host_dt(kHeads), host_a(kHeads);
    for (int64_t h = 0; h < kHeads; ++h) {
        host_alpha[size_t(h)] = 0.5f + 0.01f * float(h % 7);
        host_beta[size_t(h)] = 0.6f + 0.01f * float(h % 5);
        host_dt[size_t(h)] = 0.05f;
        host_a[size_t(h)] = -0.3f;
    }
    std::vector<float> host_gate(kInner);
    for (float & value : host_gate) value = uniform(rng);
    std::vector<float> host_norm(kState);
    for (float & value : host_norm) value = 1.0f + uniform(rng) * 0.1f;
    std::vector<float> host_conv_state(size_t(kConvDim) * (kKernel - 1));
    for (float & value : host_conv_state) value = uniform(rng);
    std::vector<float> host_rec_state(size_t(kHeads) * kState * kState);
    for (float & value : host_rec_state) value = uniform(rng);

    Fixture f;
    bool ok = upload(device, host_conv_weights, &f.conv_weights) &&
              upload(device, host_qkv, &f.qkv) && upload(device, host_alpha, &f.alpha) &&
              upload(device, host_beta, &f.beta) && upload(device, host_dt, &f.dt) &&
              upload(device, host_a, &f.a) && upload(device, host_gate, &f.gate) &&
              upload(device, host_norm, &f.norm) &&
              upload(device, host_conv_state, &f.conv_state_legacy) &&
              upload(device, host_conv_state, &f.conv_state_fused) &&
              upload(device, host_rec_state, &f.rec_state_legacy) &&
              upload(device, host_rec_state, &f.rec_state_fused);
    const std::vector<float> zeros_conv(size_t(kConvDim), 0.0f);
    const std::vector<float> zeros_inner(size_t(kInner), 0.0f);
    ok = ok && upload(device, zeros_conv, &f.conv_out_legacy) &&
         upload(device, zeros_conv, &f.conv_out_fused) &&
         upload(device, zeros_inner, &f.gdn_legacy) &&
         upload(device, zeros_inner, &f.norm_legacy) &&
         upload(device, zeros_inner, &f.norm_fused);
    if (!ok) {
        return 1;
    }

    yuuka_plan * legacy = nullptr;
    yuuka_plan * fused = nullptr;
    if (!check(yuuka_plan_create(device, &legacy), "legacy plan") ||
        !check(yuuka_plan_create(device, &fused), "fused plan") ||
        !build_legacy(legacy, f) || !build_fused(fused, f)) {
        return 1;
    }

    const size_t conv_size = size_t(kConvDim);
    const size_t rec_size = size_t(kHeads) * size_t(kState) * size_t(kState);
    const size_t inner_size = size_t(kInner);
    const size_t ring_size = size_t(kConvDim) * size_t(kKernel - 1);
    std::vector<float> conv_legacy(conv_size);
    std::vector<float> conv_fused(conv_size);
    std::vector<float> rec_legacy(rec_size);
    std::vector<float> rec_fused(rec_size);
    std::vector<float> norm_legacy(inner_size);
    std::vector<float> norm_fused(inner_size);
    std::vector<float> ring_legacy(ring_size);
    std::vector<float> ring_fused(ring_size);
    // The host reference advances its own copies of the two states.
    std::vector<float> host_conv_mirror = host_conv_state;
    std::vector<float> host_state = host_rec_state;
    std::vector<float> host_normed(inner_size);

    bool match = true;
    double worst_conv = 0.0;
    double worst_state = 0.0;
    double worst_ring = 0.0;
    double worst_norm = 0.0;
    double worst_host = 0.0;
    for (int token = 0; token < kTokens; ++token) {
        for (yuuka_plan * plan : {legacy, fused}) {
            if (!check(yuuka_plan_execute(plan, stream, 0, 0, 1, completion), "execute") ||
                !check(yuuka_event_wait(completion), "wait")) {
                match = false;
                break;
            }
        }
        if (!match) {
            break;
        }
        ok = download(f.conv_out_legacy, conv_legacy) &&
             download(f.conv_out_fused, conv_fused) &&
             download(f.rec_state_legacy, rec_legacy) &&
             download(f.rec_state_fused, rec_fused) &&
             download(f.norm_legacy, norm_legacy) &&
             download(f.norm_fused, norm_fused) &&
             download(f.conv_state_legacy, ring_legacy) &&
             download(f.conv_state_fused, ring_fused);
        if (!ok) {
            match = false;
            break;
        }
        // The convolution result is the input of the recurrent step, so it is
        // compared first and then used as the reference input.
        double scale = 0.0;
        const double conv_diff = worst(conv_legacy, conv_fused, &scale);
        reference_token(conv_legacy, host_alpha, host_beta, host_dt, host_a, host_gate,
                        host_norm, host_state, host_normed);
        // The host mirror of the convolution ring state is advanced the same way
        // the two kernels advance theirs, so the ring comparison stays a check of
        // the kernels and not of the reference.
        for (size_t i = 0; i < host_conv_mirror.size(); ++i) {
            host_conv_mirror[i] = ring_legacy[i];
        }
        const double state_diff = worst(rec_legacy, rec_fused, &scale);
        const double ring_diff = worst(ring_legacy, ring_fused, &scale);
        const double norm_diff = worst(norm_legacy, norm_fused, &scale);
        const double legacy_host =
            std::max(worst(host_state, rec_legacy, &scale), worst(host_normed, norm_legacy, &scale));
        const double fused_host =
            std::max(worst(host_state, rec_fused, &scale), worst(host_normed, norm_fused, &scale));
        worst_conv = std::max(worst_conv, conv_diff);
        worst_state = std::max(worst_state, state_diff);
        worst_ring = std::max(worst_ring, ring_diff);
        worst_norm = std::max(worst_norm, norm_diff);
        worst_host = std::max(worst_host, std::max(legacy_host, fused_host));
        std::printf("token %d: conv %.3e state %.3e ring %.3e normalized %.3e | against the"
                    " host reference: legacy %.3e fused %.3e\n", token, conv_diff, state_diff,
                    ring_diff, norm_diff, legacy_host, fused_host);
    }
    if (!match) {
        return 1;
    }
    // Both device paths reassociate some of the column sums against the
    // reference, so the bound is an absolute difference, not an exact match. It
    // is tight enough that the missing decay factor of the first version of the
    // fused kernel (2.6e-04) fails it.
    const double bound = 1e-5;
    const bool agrees = worst_conv <= bound && worst_state <= bound && worst_ring <= bound &&
                        worst_norm <= bound && worst_host <= bound;
    std::printf("%s: worst conv %.3e state %.3e ring %.3e normalized %.3e host %.3e"
                " (bound %.0e)\n", agrees ? "agree" : "DIFFER", worst_conv, worst_state,
                worst_ring, worst_norm, worst_host, bound);

    // The fused kernels are compiled for the supported geometry, so everything
    // else must be rejected instead of silently taking another path.
    yuuka_op narrow = basis(YUUKA_OP_RECURRENT_TOKEN);
    narrow.buf[0] = f.rec_state_fused;
    narrow.buf[1] = f.conv_out_fused;
    narrow.buf[2] = f.conv_out_fused;
    narrow.buf[3] = f.conv_out_fused;
    narrow.buf[4] = f.alpha;
    narrow.buf[5] = f.beta;
    narrow.buf[6] = f.dt;
    narrow.buf[7] = f.a;
    narrow.buf[8] = f.norm_fused;
    narrow.buf[9] = f.norm;
    narrow.buf[10] = f.gate;
    narrow.n[0] = uint64_t(kGroups);
    narrow.n[1] = uint64_t(kHeads);
    narrow.n[2] = 64;
    const yuuka_status narrow_status = yuuka_plan_add(fused, &narrow);
    std::printf("state 64: %s\n", narrow_status == YUUKA_OK ? "accepted" : yuuka_last_error());
    const bool rejects_narrow = narrow_status == YUUKA_ERR_UNSUPPORTED;
    yuuka_op unaligned = basis(YUUKA_OP_CONV_L2NORM);
    unaligned.buf[0] = f.conv_state_fused;
    unaligned.buf[1] = f.conv_weights;
    unaligned.buf[2] = f.qkv;
    unaligned.buf[3] = f.conv_out_fused;
    unaligned.n[0] = uint64_t(kConvDim) - 128;
    unaligned.n[1] = uint64_t(kKernel);
    unaligned.n[2] = uint64_t(2 * kKey);
    unaligned.n[3] = uint64_t(kKey);
    unaligned.n[4] = uint64_t(kState);
    unaligned.f[0] = kEps;
    unaligned.f[1] = float(1.0 / std::sqrt(double(kState)));
    const yuuka_status unaligned_status = yuuka_plan_add(fused, &unaligned);
    std::printf("channels %lld: %s\n", static_cast<long long>(kConvDim - 128),
                unaligned_status == YUUKA_OK ? "accepted" : yuuka_last_error());
    const bool rejects_unaligned = unaligned_status == YUUKA_ERR_UNSUPPORTED;

    yuuka_plan_destroy(legacy);
    yuuka_plan_destroy(fused);
    yuuka_event_destroy(completion);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return agrees && rejects_narrow && rejects_unaligned ? 0 : 1;
}
