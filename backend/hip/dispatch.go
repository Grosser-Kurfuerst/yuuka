//go:build hip && linux

package hip

import (
	"github.com/yuuka-ai/yuuka/tensor"
)

// PrefillPath is the selected prefill matmul implementation.
type PrefillPath int

const (
	// PrefillBatchedGemv reuses each weight element across a small batch.
	PrefillBatchedGemv PrefillPath = iota
	// PrefillWMMA is the custom rocWMMA GEMM with FP32 activations.
	PrefillWMMA
	// PrefillBLAS is the rocBLAS route for F16 weights.
	PrefillBLAS
	// PrefillMMQ is the int8 MMQ route: Q8_1 activations, int8-decoded
	// K-quant weights, exact int32 accumulation, and FP32 weight scales.
	PrefillMMQ
)

// String returns the diagnostic name of a prefill path.
func (p PrefillPath) String() string {
	switch p {
	case PrefillBatchedGemv:
		return "batched-gemv"
	case PrefillWMMA:
		return "wmma"
	case PrefillBLAS:
		return "rocblas"
	case PrefillMMQ:
		return "mmq"
	default:
		return "unknown"
	}
}

// SelectPrefillPath returns the measured prefill matmul choice for a dtype,
// row count, and shape. Measurements are in docs/benchmarks.md. Batched GEMV
// is selected below four rows and for the tall-K projections at the smallest
// WMMA batch; at one row the sweep shows per-row submission is faster for F16,
// Q8_0, and Q6_K, which remains a small-row follow-up. Above those points the
// custom WMMA GEMM beats per-row submission; the rocBLAS FP16 baseline wins for
// F16 only in the measured shape classes below.
//
// K-quant weights use the int8 MMQ route at 128 or more rows and a multiple-of
// 256 input width, which is every K-quant block width. The route keeps the
// weights packed, quantizes the activations to Q8_1, accumulates exact int32
// products in the 16x16x16 int8 WMMA, and applies the FP32 weight scales in the
// epilogue, so it has none of the FP16-destination reduction error that keeps
// rocBLAS off the tall projections. The measured sweep puts it at 10.4-14.8
// TF/s against 6.3-9.7 TF/s for WMMA across the 128-512 row classes, and 7-60%
// above the dequant-plus-rocBLAS route at 128-256 rows; at 512 rows that route
// is 2-17% faster on wide shapes, which the blanket rule intentionally trades
// for halving the tall-projection time and dropping the FP16 weight scratch.
// Below 128 rows the Q8_1 pass and the 128-token tile waste more work than they
// save, so WMMA stays.
func SelectPrefillPath(dtype tensor.DType, rows, nIn, nOut int64) PrefillPath {
	if rows <= 0 {
		return PrefillBatchedGemv
	}
	if dtype == tensor.F16 {
		if rows < 4 {
			return PrefillBatchedGemv
		}
		if nIn > nOut {
			if rows <= 64 {
				return PrefillBLAS
			}
			return PrefillWMMA
		}
		// Wide F16 projections keep batched GEMV below M=8: WMMA is 17%
		// slower at 2560->9216 and only within noise at 2560->8192.
		if rows < 8 {
			return PrefillBatchedGemv
		}
		if rows >= 128 {
			return PrefillBLAS
		}
		return PrefillWMMA
	}
	if rows < 4 {
		return PrefillBatchedGemv
	}
	if nIn > nOut && rows < 8 {
		return PrefillBatchedGemv
	}
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		if nIn%256 == 0 && nOut >= 64 && rows >= 128 {
			return PrefillMMQ
		}
	}
	return PrefillWMMA
}

// SelectLMHeadPath reports whether the streaming LM head kernel is selected for
// an output projection. The kernel serves 1..16 rows and streams every packed
// weight row once, which is what makes both output tails bandwidth bound
// instead of latency bound; it is measured against the per-row GEMV in
// docs/benchmarks.md. The prefill tail and the decode tail both use this rule,
// and any shape it does not cover keeps the per-row GEMV, which stays the
// documented fallback rather than a silent substitution.
func SelectLMHeadPath(dtype tensor.DType, rows, nIn, nOut int64) bool {
	if rows <= 0 || rows > MaxLMHeadRows {
		return false
	}
	if nIn <= 0 || nOut < lmHeadMinOutputs || nIn%256 != 0 {
		return false
	}
	switch dtype {
	case tensor.F32, tensor.F16, tensor.Q4_0, tensor.Q5_0, tensor.Q8_0,
		tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		return true
	default:
		return false
	}
}

// lmHeadMinOutputs is the measured crossover below which the per-row GEMV is
// still the better route. At 2560 input values the streaming kernel wins from
// 512 outputs upward for Q4_K, Q5_K, and Q6_K and from 1536 upward for Q8_0 and
// F16, where the vectorized GEMV is still ahead at 1024; 1024 is the smallest
// width that covers every dtype without regressing one, and no output
// projection sits between 512 and 1536. Sweeps: build/yuuka_prefill_sweep
// (LMHEAD cells) and build/yuuka_gemv_decode_sweep --threshold.
const lmHeadMinOutputs = 1024

// PrefillAttentionMMA reports whether the FP16 MMA paged prefill attention
// kernel is selected for a shape. It mirrors the C++ rule in
// kernels/hip/prefill/attention_wmma.hip: the kernel serves the four query
// heads that share a KV head, needs an FP16-storable pool and a head width of
// 64, 128, or 256, and only wins once the chunk or the prefix is long enough to
// fill the device. A shape it rejects keeps the tiled kernel, which is correct
// for every format.
func PrefillAttentionMMA(heads, kvHeads, headDim, rows, position int, dtype tensor.DType) bool {
	if heads != 4*kvHeads || headDim <= 0 || rows <= 0 {
		return false
	}
	if dtype != tensor.F32 && dtype != tensor.F16 {
		return false
	}
	if headDim != 64 && headDim != 128 && headDim != 256 {
		return false
	}
	return rows >= 64 || position+rows >= 256
}
