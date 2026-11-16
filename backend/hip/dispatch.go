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
	// PrefillDequantBLAS dequantizes the packed K-quant weights into an FP16
	// scratch and runs the rocBLAS route.
	PrefillDequantBLAS
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
	case PrefillDequantBLAS:
		return "blas-dequant"
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
// F16 only in the measured shape classes below. K-quant weights use the rocBLAS
// route through an FP16 dequant scratch for the wide class (`K <= N`) from 128
// rows upward. The sweep measures the crossover at 128 rows, where the route is
// neutral (0.95-1.16x) and larger batches reach 1.24-2.5x; below 128 rows the
// expansion dominates (0.31-0.64x) and WMMA stays. The 128-row floor also keeps
// the route independent of the usual prefill chunk sizes: chunk 128 and chunk
// 512 use one kernel, and the chunk-invariance test measures 0.042 logits,
// while a route that changes with the chunk size measures 0.058.
//
// Tall-`K` projections stay on WMMA at every batch size for a measured numeric
// reason: rocBLAS splits `K` on those shapes and the FP16 destination forces
// the split reduction through FP16, which measured 1.0-1.5e-3 relative error
// against 4.0-4.9e-4 on the wide shapes. The WMMA path selects its 64x64 or
// 32x64 tile internally from the same measured row and shape class; that choice
// is not exposed as a separate Go path.
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
		if nIn%256 == 0 && nIn <= nOut && rows >= 128 {
			return PrefillDequantBLAS
		}
	}
	return PrefillWMMA
}
