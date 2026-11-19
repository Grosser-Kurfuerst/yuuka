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
	// PrefillStreaming streams packed weights across a small FP32 row block.
	PrefillStreaming
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
	case PrefillStreaming:
		return "streaming"
	default:
		return "unknown"
	}
}

// SelectPrefillPath returns the measured route for a dtype, row count and
// shape. Small K-quant row blocks can use FP32 streaming. Wider blocks
// retain FP16-operand WMMA until the Q8_1 MMQ threshold at 128 rows.
// F16 weights also use rocBLAS for the shape classes measured in
// docs/benchmarks.md.
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
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K:
		if SelectLMHeadPath(dtype, rows, nIn, nOut) {
			return PrefillStreaming
		}
		if nIn%256 == 0 && nOut >= 64 && rows >= 128 {
			return PrefillMMQ
		}
	}
	if rows < 4 || (nIn > nOut && rows < 8) {
		return PrefillBatchedGemv
	}
	return PrefillWMMA
}

// SelectLMHeadPath reports whether the streaming LM head kernel is selected for
// an output projection. The kernel serves 1..16 rows and streams every packed
// weight row once, which is what makes both output tails bandwidth bound
// instead of latency bound; it is measured against the per-row GEMV in
// docs/benchmarks.md. The prefill tail and the decode tail use it directly, and
// SelectDecodeGemvPath uses it for the whole decode matvec family; any shape it
// does not cover keeps the per-row GEMV, which stays the documented fallback
// rather than a silent substitution.
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

// DecodeGemvPath is the selected decode matvec implementation.
type DecodeGemvPath int

const (
	// DecodeGemvPerRow is one 256-thread block per output row with a block-wide
	// tree reduction and, for Q4_K/Q5_K, two barriers per 256-value K step. It
	// is the fallback for shapes the streaming kernel does not cover and the
	// reference the kernel test compares against.
	DecodeGemvPerRow DecodeGemvPath = iota
	// DecodeGemvStreaming is the channel-parallel streaming kernel: one wave
	// owns four output channels and walks the K axis itself, every lane reads
	// 16 bytes of packed weights per step, the activation row is staged in
	// shared memory once per block, and the only reduction is one wave shuffle
	// per channel.
	DecodeGemvStreaming
	// DecodeGemvInt8 is the same mapping over an int8 activation: the four
	// packed digits of a lane are dotted against the Q8_1 activation word with
	// one v_dot4 instead of being decoded to FP32 first, and the block scale
	// and minimum are applied once per sub-block.
	DecodeGemvInt8
)

// IsInt8DecodeDType reports whether the int8 decode matvec serves a weight
// dtype. The route decodes the packed digits straight to int8, so it exists
// only for the dtypes whose per-value form is an integer plus an affine block
// scale: the K-quants and Q8_0. F32, F16, Q4_0, and Q5_0 keep the FP32
// streaming kernel.
func IsInt8DecodeDType(dtype tensor.DType) bool {
	switch dtype {
	case tensor.Q4_K, tensor.Q5_K, tensor.Q6_K, tensor.Q8_0:
		return true
	default:
		return false
	}
}

// String returns the diagnostic name of a decode matvec path.
func (p DecodeGemvPath) String() string {
	switch p {
	case DecodeGemvPerRow:
		return "per-row"
	case DecodeGemvStreaming:
		return "streaming"
	case DecodeGemvInt8:
		return "int8"
	default:
		return "unknown"
	}
}

// SelectDecodeGemvPath returns the measured decode matvec choice for a weight
// dtype and shape. It shares the output projection's shape predicate: the
// streaming mapping is bandwidth bound on every shape the decode graph
// dispatches (measured 64-199 GB/s against 25-76 GB/s for the per-row form), and
// the per-row GEMV stays below the crossover and for the shapes it cannot serve
// (a K step that is not 256 values, or an unsupported dtype). Sharing one
// predicate keeps the two tails and the layer matvecs from drifting apart; the
// crossover measurement is in docs/benchmarks.md.
//
// The FP32 streaming kernel is the selected route for every dtype. The int8
// route is implemented and reachable through int8Matvec, and it is measured
// slower end to end on gfx1151: the decode step reads one row of weights per
// matvec and is not instruction bound, so the ALU saving does not pay for the
// activation quantization the route needs (docs/benchmarks.md).
// It stays available so that measurement is reproducible and so a device or a
// later fusion can be re-evaluated against it.
//
// The narrow recurrent projections (ssm_alpha and ssm_beta, 32 outputs) stay on
// the per-row form because the streaming kernel needs 32 blocks to fill the
// device: at 32 outputs it runs one block and measures 3.7 GB/s against 9.5.
// Merging those launches is a launch-count change that is not implemented.
func SelectDecodeGemvPath(dtype tensor.DType, nIn, nOut int64, int8Matvec bool) DecodeGemvPath {
	if !SelectLMHeadPath(dtype, 1, nIn, nOut) {
		return DecodeGemvPerRow
	}
	if int8Matvec && IsInt8DecodeDType(dtype) {
		return DecodeGemvInt8
	}
	return DecodeGemvStreaming
}

// PrefillAttentionMMA reports whether the FP16 MMA paged prefill attention
// kernel is selected for a shape. It mirrors the C++ rule in
// kernels/hip/prefill/attention_wmma.hip: the kernel serves the four query
// heads that share a KV head, reads F32, F16 or Q8_0 KV at head widths
// 64, 128 or 256. The short-prefix policy keeps Q8_0 tiled until position
// 512, or 2048 for a 1024-row parent. Other formats use the shared crossover.
func PrefillAttentionMMA(heads, kvHeads, headDim, rows, position int, dtype tensor.DType) bool {
	if kvHeads <= 0 || heads != 4*kvHeads || headDim <= 0 || rows <= 0 {
		return false
	}
	if dtype != tensor.F32 && dtype != tensor.F16 && dtype != tensor.Q8_0 {
		return false
	}
	if headDim != 64 && headDim != 128 && headDim != 256 {
		return false
	}
	if dtype == tensor.Q8_0 {
		if rows == 1024 {
			return position >= 2048
		}
		return position >= 512
	}
	return rows >= 64 || position+rows >= 256
}
