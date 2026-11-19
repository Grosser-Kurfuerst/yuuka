// prefill_sweep compares prefill matrix implementations on gfx1151:
// per-row GEMV, batched GEMV, custom rocWMMA GEMM, the rocBLAS route, and the
// dequant-plus-rocBLAS route for K-quant weights. Use --direct-mmq to
// isolate MMQ at all model shapes and representative row boundaries.

#include "yuuka_hip.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Buffer {
    yuuka_buffer * buf = nullptr;
    uint64_t bytes = 0;
};

bool check(yuuka_status status, const char * what) {
    if (status == YUUKA_OK) return true;
    std::fprintf(stderr, "%s: %s\n", what, yuuka_last_error());
    return false;
}

bool alloc(yuuka_device * device, Buffer & b, uint64_t bytes) {
    b.bytes = bytes;
    return check(yuuka_buffer_alloc(device, bytes, &b.buf), "alloc");
}

void release(Buffer & b) {
    if (b.buf != nullptr) {
        yuuka_buffer_free(b.buf);
        b.buf = nullptr;
    }
}

uint64_t row_bytes(yuuka_dtype dtype, uint64_t n) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return n * 4;
        case YUUKA_DTYPE_F16: return n * 2;
        case YUUKA_DTYPE_Q8_0: return (n / 32) * 34;
        case YUUKA_DTYPE_Q4_K: return (n / 256) * 144;
        case YUUKA_DTYPE_Q5_K: return (n / 256) * 176;
        case YUUKA_DTYPE_Q6_K: return (n / 256) * 210;
        default: return 0;
    }
}

const char * dtype_name(yuuka_dtype dtype) {
    switch (dtype) {
        case YUUKA_DTYPE_F32: return "F32";
        case YUUKA_DTYPE_F16: return "F16";
        case YUUKA_DTYPE_Q8_0: return "Q8_0";
        case YUUKA_DTYPE_Q4_K: return "Q4_K";
        case YUUKA_DTYPE_Q5_K: return "Q5_K";
        case YUUKA_DTYPE_Q6_K: return "Q6_K";
        default: return "?";
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct Result {
    const char * name;
    double ms;
    bool ok;
};

Result bench_rows(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                  yuuka_plan * plan, const char * name, int iterations,
                  std::function<void()> submit, std::function<bool()> wait) {
    for (int i = 0; i < 2; ++i) {
        submit();
        if (!wait()) return {name, 0, false};
    }
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        submit();
        if (!wait()) return {name, 0, false};
    }
    const auto stop = Clock::now();
    (void)device;
    (void)stream;
    (void)event;
    (void)plan;
    return {name, elapsed_ms(start, stop) / iterations, true};
}

void run_shape(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
               yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    const uint64_t rb = row_bytes(dtype, uint64_t(n_in));
    std::mt19937 rng(12345);
    std::vector<uint8_t> weights(rb * uint64_t(n_out));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<uint8_t>(rng() & 0xff);
    }
    // Keep the scale halfwords in a sane range for all K-quants.
    if (dtype == YUUKA_DTYPE_Q4_K || dtype == YUUKA_DTYPE_Q5_K) {
        for (int64_t row = 0; row < n_out; ++row) {
            weights[row * rb + 0] = 0x11;
            weights[row * rb + 1] = 0x1c;
        }
    }
    std::vector<float> x(size_t(n_in * rows));
    for (float & v : x) v = 0.25f;

    const bool kquant = dtype == YUUKA_DTYPE_Q4_K || dtype == YUUKA_DTYPE_Q5_K ||
                        dtype == YUUKA_DTYPE_Q6_K;
    Buffer wb, xb, ob, x16, o16, w16, rowbuf, qsb, dsb;
    bool ok = alloc(device, wb, weights.size()) &&
              alloc(device, xb, x.size() * 4) && alloc(device, ob, size_t(n_out * rows) * 4) &&
              alloc(device, x16, x.size() * 2) &&
              alloc(device, o16, size_t(n_out * rows) * 2) &&
              alloc(device, rowbuf, size_t(n_in) * 4);
    if (ok && kquant) {
        ok = alloc(device, w16, size_t(n_in * n_out) * 2) &&
             alloc(device, qsb, uint64_t(n_in) * uint64_t(rows)) &&
             alloc(device, dsb, uint64_t(rows) * uint64_t(n_in / 32) * 8);
    }
    if (ok) {
        ok = check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()), "h2d w") &&
             check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4), "h2d x");
    }

    yuuka_plan * plan = nullptr;
    if (ok) {
        ok = check(yuuka_plan_create(device, &plan), "plan create");
    }
    if (ok) {
        for (int64_t row = 0; row < rows; ++row) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMV;
            op.buf[0] = wb.buf;
            op.buf[1] = xb.buf;
            op.buf[2] = ob.buf;
            op.off[1] = uint64_t(row * n_in) * 4;
            op.off[2] = uint64_t(row * n_out) * 4;
            op.n[0] = uint64_t(n_in);
            op.n[1] = uint64_t(n_out);
            op.u[0] = dtype;
            if (!check(yuuka_plan_add(plan, &op), "add per-row gemv")) {
                ok = false;
                break;
            }
        }
    }

    std::vector<Result> results;
    auto wait_event = [&]() { return check(yuuka_event_wait(event), "wait"); };
    if (ok) {
        results.push_back(bench_rows(device, stream, event, plan, "row-gemv", 5, [&]() {
            (void)yuuka_plan_execute(plan, stream, 0, 0, 1, event);
        }, wait_event));
    }
    if (ok && rows <= 16) {
        results.push_back(bench_rows(device, stream, event, plan, "batch-gemv", 5, [&]() {
            (void)yuuka_gemv_batch(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                   dtype, event);
        }, wait_event));
    }
    if (ok) {
        results.push_back(bench_rows(device, stream, event, plan, "wmma", 5, [&]() {
            (void)yuuka_gemm_wmma(stream, ob.buf, xb.buf, wb.buf, n_in, n_out, rows,
                                  dtype, event);
        }, wait_event));
    }
    if (ok && dtype == YUUKA_DTYPE_F16) {
        results.push_back(bench_rows(device, stream, event, plan, "rocblas", 5, [&]() {
            (void)yuuka_cast_f32_to_f16(stream, x16.buf, xb.buf, n_in * rows, event);
            (void)yuuka_gemm_blas(stream, ob.buf, o16.buf, x16.buf, wb.buf, n_in, n_out,
                                  rows, event);
        }, wait_event));
    }
    // The K-quant route expands the packed matrix into FP16 on every call, so
    // the dequantization cost is part of the measured time.
    if (ok && kquant) {
        results.push_back(bench_rows(device, stream, event, plan, "blasq", 5, [&]() {
            (void)yuuka_dequant_f16(stream, w16.buf, wb.buf, n_in, n_out, dtype, event);
            (void)yuuka_cast_f32_to_f16(stream, x16.buf, xb.buf, n_in * rows, event);
            (void)yuuka_gemm_blas(stream, ob.buf, o16.buf, x16.buf, w16.buf, n_in, n_out,
                                  rows, event);
        }, wait_event));
    }
    // The MMQ route quantizes the activations on every call, so the Q8_1 pass is
    // part of the measured time.
    if (ok && kquant) {
        results.push_back(bench_rows(device, stream, event, plan, "mmq", 5, [&]() {
            (void)yuuka_quantize_q8_1(stream, qsb.buf, dsb.buf, xb.buf, n_in, rows,
                                      event);
            (void)yuuka_gemm_mmq(stream, ob.buf, qsb.buf, dsb.buf, wb.buf, n_in, n_out,
                                 rows, dtype, event);
        }, wait_event));
    }
    if (ok) {
        const double weight_gb = double(weights.size()) / 1e9;
        const double flops = 2.0 * double(n_in) * double(n_out) * double(rows);
        std::printf("%s M=%4lld N=%5lld K=%5lld", dtype_name(dtype),
                    static_cast<long long>(rows), static_cast<long long>(n_out),
                    static_cast<long long>(n_in));
        for (const Result & r : results) {
            if (!r.ok) {
                std::printf("  %s=FAIL", r.name);
                continue;
            }
            std::printf("  %s=%.3fms %.1fGB/s %.1fTF/s", r.name, r.ms,
                        weight_gb / (r.ms / 1000.0),
                        flops / (r.ms / 1000.0) / 1e12);
        }
        std::printf("\n");
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(rowbuf);
    release(dsb);
    release(qsb);
    release(w16);
    release(o16);
    release(x16);
    release(ob);
    release(xb);
    release(wb);
}

bool run_direct_mmq(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                    yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    const uint64_t rb = row_bytes(dtype, n_in);
    const int block_bytes = dtype == YUUKA_DTYPE_Q4_K ? 144 : 176;
    std::mt19937 rng(12345);
    std::vector<uint8_t> weights(rb * n_out);
    for (auto & value : weights) value = static_cast<uint8_t>(rng());
    for (size_t i = 0; i < weights.size(); i += block_bytes) {
        weights[i] = weights[i + 2] = 0x11;
        weights[i + 1] = weights[i + 3] = 0x1c;
    }
    const std::vector<float> x(n_in * rows, 0.25f);
    Buffer wb, xb, ob, qs, ds;
    bool ok = alloc(device, wb, weights.size()) && alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, n_out * rows * 4) && alloc(device, qs, x.size()) &&
              alloc(device, ds, x.size() / 32 * 8) &&
              check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()), "h2d w") &&
              check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4), "h2d x") &&
              check(yuuka_quantize_q8_1(stream, qs.buf, ds.buf, xb.buf, n_in, rows, event),
                    "quantize") &&
              check(yuuka_event_wait(event), "quantize wait");
    if (ok) {
        const auto result = bench_rows(device, stream, event, nullptr, "mmq", 9, [&]() {
            ok = check(yuuka_gemm_mmq(stream, ob.buf, qs.buf, ds.buf, wb.buf, n_in,
                                      n_out, rows, dtype, event), "mmq");
        }, [&]() { return ok && check(yuuka_event_wait(event), "mmq wait"); });
        ok = result.ok;
        if (ok) {
            std::printf("%s M=%lld N=%lld K=%lld mmq=%.6fms\n", dtype_name(dtype),
                        static_cast<long long>(rows), static_cast<long long>(n_out),
                        static_cast<long long>(n_in), result.ms);
        }
    }
    release(ds);
    release(qs);
    release(ob);
    release(xb);
    release(wb);
    return ok;
}

// run_lm_head compares the streaming LM head against one GEMV submission per
// row for the shapes the prefill tail actually asks for. The output matrix is
// the widest in the model and is read once, so the crossover between the two
// routes is a bandwidth question and is measured per dtype here rather than
// assumed.
void run_lm_head(yuuka_device * device, yuuka_stream * stream, yuuka_event * event,
                 yuuka_dtype dtype, int64_t n_in, int64_t n_out, int64_t rows) {
    const uint64_t rb = row_bytes(dtype, uint64_t(n_in));
    std::vector<uint8_t> weights(rb * uint64_t(n_out), 0x31);
    std::vector<float> x(size_t(n_in * rows), 0.25f);
    Buffer wb, xb, ob;
    bool ok = alloc(device, wb, weights.size()) &&
              alloc(device, xb, x.size() * 4) &&
              alloc(device, ob, size_t(n_out * rows) * 4) &&
              check(yuuka_memcpy_h2d(wb.buf, weights.data(), weights.size()), "h2d w") &&
              check(yuuka_memcpy_h2d(xb.buf, x.data(), x.size() * 4), "h2d x");
    yuuka_plan * plan = nullptr;
    if (ok) {
        ok = check(yuuka_plan_create(device, &plan), "plan create");
    }
    if (ok) {
        for (int64_t row = 0; row < rows; ++row) {
            yuuka_op op{};
            op.code = YUUKA_OP_GEMV;
            op.buf[0] = wb.buf;
            op.buf[1] = xb.buf;
            op.buf[2] = ob.buf;
            op.off[1] = uint64_t(row * n_in) * 4;
            op.off[2] = uint64_t(row * n_out) * 4;
            op.n[0] = uint64_t(n_in);
            op.n[1] = uint64_t(n_out);
            op.u[0] = dtype;
            if (!check(yuuka_plan_add(plan, &op), "add per-row gemv")) {
                ok = false;
                break;
            }
        }
    }
    auto wait_event = [&]() { return check(yuuka_event_wait(event), "wait"); };
    const Result row_gemv = ok
        ? bench_rows(device, stream, event, plan, "row-gemv", 5, [&]() {
              (void)yuuka_plan_execute(plan, stream, 0, 0, 1, event);
          }, wait_event)
        : Result{"row-gemv", 0, false};
    const Result lm_head = ok
        ? bench_rows(device, stream, event, plan, "lmhead", 5, [&]() {
              (void)yuuka_lm_head(stream, ob.buf, xb.buf, wb.buf, n_in, n_out,
                                  rows, dtype, event);
          }, wait_event)
        : Result{"lmhead", 0, false};
    if (ok) {
        const double weight_gb = double(weights.size()) / 1e9;
        const double row_rate = row_gemv.ok ? weight_gb / (row_gemv.ms / 1000.0) : 0.0;
        const double head_rate = lm_head.ok ? weight_gb / (lm_head.ms / 1000.0) : 0.0;
        std::printf("LMHEAD %s K=%lld N=%lld M=%lld  row-gemv=%.3fms %.1fGB/s"
                    "  lmhead=%.3fms %.1fGB/s\n",
                    dtype_name(dtype), static_cast<long long>(n_in),
                    static_cast<long long>(n_out), static_cast<long long>(rows),
                    row_gemv.ms, row_rate, lm_head.ms, head_rate);
    }
    if (plan != nullptr) yuuka_plan_destroy(plan);
    release(ob);
    release(xb);
    release(wb);
}

} // namespace

int main(int argc, char ** argv) {
    const bool direct = argc == 2 && std::strcmp(argv[1], "--direct-mmq") == 0;
    if (argc != 1 && !direct) {
        std::fprintf(stderr, "usage: %s [--direct-mmq]\n", argv[0]);
        return 2;
    }
    yuuka_device * device = nullptr;
    if (yuuka_device_open(0, &device) != YUUKA_OK) {
        std::fprintf(stderr, "SKIP: no HIP device: %s\n", yuuka_last_error());
        return 77;
    }
    yuuka_stream * stream = nullptr;
    yuuka_event * event = nullptr;
    if (!check(yuuka_stream_create(device, &stream), "stream") ||
        !check(yuuka_event_create(device, &event), "event")) {
        return 1;
    }
    if (direct) {
        constexpr auto q4 = YUUKA_DTYPE_Q4_K;
        constexpr auto q5 = YUUKA_DTYPE_Q5_K;
        const struct {
            yuuka_dtype dtype;
            int64_t n_in;
            int64_t n_out;
        } cells[] = {
            {q4, 2560, 1024}, {q4, 2560, 4096}, {q4, 2560, 8192}, {q4, 2560, 9216},
            {q4, 4096, 2560}, {q4, 9216, 2560}, {q5, 2560, 8192}, {q5, 4096, 2560},
        };
        bool ok = true;
        for (const auto & cell : cells) {
            for (int64_t m : {128, 129, 192, 511, 512, 1024, 2048, 4096}) {
                ok = run_direct_mmq(device, stream, event, cell.dtype, cell.n_in,
                                    cell.n_out, m) && ok;
            }
        }
        yuuka_event_destroy(event);
        yuuka_stream_destroy(stream);
        yuuka_device_close(device);
        return ok ? 0 : 1;
    }
    const yuuka_dtype dtypes[] = {YUUKA_DTYPE_F16, YUUKA_DTYPE_Q8_0, YUUKA_DTYPE_Q4_K,
                                  YUUKA_DTYPE_Q5_K, YUUKA_DTYPE_Q6_K};
    const int64_t rows[] = {1, 4, 8, 16, 32, 64, 128, 256, 512};
    const struct {
        int64_t n_in;
        int64_t n_out;
    } shapes[] = {
        {2560, 9216}, {9216, 2560}, {2560, 8192},
    };
    for (yuuka_dtype dtype : dtypes) {
        for (const auto & shape : shapes) {
            for (int64_t m : rows) {
                run_shape(device, stream, event, dtype, shape.n_in, shape.n_out, m);
            }
        }
    }
    // Prefill tail shapes: every supported output-matrix dtype at the real
    // vocabulary width, plus two smaller widths that show the crossover.
    const yuuka_dtype lm_dtypes[] = {YUUKA_DTYPE_Q4_K, YUUKA_DTYPE_Q6_K,
                                     YUUKA_DTYPE_Q8_0};
    const int64_t lm_out[] = {512, 1024, 2048, 4096, 32768, 248320};
    for (yuuka_dtype dtype : lm_dtypes) {
        for (int64_t n_out : lm_out) {
            for (int64_t m : {1, 16}) {
                run_lm_head(device, stream, event, dtype, 2560, n_out, m);
            }
        }
    }
    yuuka_event_destroy(event);
    yuuka_stream_destroy(stream);
    yuuka_device_close(device);
    return 0;
}
