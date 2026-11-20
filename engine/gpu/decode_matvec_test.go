//go:build hip && linux

package gpu

import (
	"math"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// TestGPUStreamingDecodeMatchesGemv compares the two decode matvec mappings on
// the real model. Both sessions run FP32 arithmetic, so the measured difference
// is the mapping change alone: the channel-parallel streaming kernel with
// DisableInt8Matvec against the per-row GEMV the original tree used. The int8
// arithmetic of the int8 route is bounded separately by
// TestGPUInt8MatvecMatchesFp32.
//
// The tolerance is absolute because the comparison is over the logits of one
// vocabulary: the streaming kernel sums four K values per lane and reduces
// across the wave, while the GEMV sums a strided set of K values per thread and
// reduces over the block. Both accumulate in FP32, so the difference is that
// reassociation and nothing else.
func TestGPUStreamingDecodeMatchesGemv(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	streaming, err := newRuntime(reader, model, 64, Options{})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer streaming.Close()
	perRow, err := newRuntime(reader, model, 64, Options{ForceDecodeGemv: true})
	if err != nil {
		t.Fatal(err)
	}
	defer perRow.Close()

	// The two plans must actually differ, and every matvec in the default plan
	// must follow the dispatch rule rather than a per-call choice.
	defaultOps := streaming.session.plan.Ops()
	perRowOps := perRow.session.plan.Ops()
	checkDecodeRule(t, defaultOps, false)
	defaultCounts := countMatvec(defaultOps)
	perRowCounts := countMatvec(perRowOps)
	t.Logf("default plan: %d int8, %d fp32 streaming, %d per-row matvecs;"+
		" per-row plan: %d, %d, %d", defaultCounts.int8, defaultCounts.streaming,
		defaultCounts.perRow, perRowCounts.int8, perRowCounts.streaming,
		perRowCounts.perRow)
	if perRowCounts.int8 != 0 || perRowCounts.streaming != 0 || perRowCounts.perRow == 0 {
		t.Fatalf("the forced plan runs %d int8, %d streaming, %d per-row matvecs,"+
			" want only per-row", perRowCounts.int8, perRowCounts.streaming,
			perRowCounts.perRow)
	}
	if defaultCounts.streaming <= model.Config.BlockCount {
		t.Fatalf("the fp32 streaming plan runs %d streaming matvecs, want at least one"+
			" per layer plus the output projection", defaultCounts.streaming)
	}

	streamLogits, err := streaming.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	perRowLogits, err := perRow.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	prefillDiff, _ := logitDiff(streamLogits, perRowLogits)
	t.Logf("prefill tail: max abs logit diff %.6g (streaming kernel on both routes)",
		prefillDiff)

	// Every logit compared below comes from the decode plan, where the two
	// sessions differ.
	const steps = 8
	streamTokens := make([]int32, 0, steps)
	perRowTokens := make([]int32, 0, steps)
	maxDiff := 0.0
	for range steps {
		streamToken, perRowToken := argmax(streamLogits), argmax(perRowLogits)
		streamTokens = append(streamTokens, streamToken)
		perRowTokens = append(perRowTokens, perRowToken)
		if streamLogits, err = streaming.Step(streamToken); err != nil {
			t.Fatal(err)
		}
		if perRowLogits, err = perRow.Step(perRowToken); err != nil {
			t.Fatal(err)
		}
		if diff, _ := logitDiff(streamLogits, perRowLogits); diff > maxDiff {
			maxDiff = diff
		}
	}
	for i := range streamTokens {
		if streamTokens[i] != perRowTokens[i] {
			t.Fatalf("token %d: streaming route %d, per-row route %d", i,
				streamTokens[i], perRowTokens[i])
		}
	}
	// The recorded maximum over the eight decode steps of this prompt. The
	// per-element kernel gate is 2e-4 relative in the native test; logits reach
	// about 30 in magnitude, so the absolute bound is wider.
	const tolerance = 0.02
	if maxDiff > tolerance {
		t.Fatalf("the two routes differ by %.6g, tolerance %.6g", maxDiff, tolerance)
	}
	t.Logf("tokens %v, max abs logit diff over %d steps %.6g (tolerance %.6g)",
		streamTokens, steps, maxDiff, tolerance)
}

// matvecCounts is the number of matvecs a decode plan holds per route.
type matvecCounts struct {
	int8      int
	streaming int
	perRow    int
}

// countMatvec counts the matvec operations of a decode plan by route.
func countMatvec(ops []hip.OpInfo) matvecCounts {
	var counts matvecCounts
	for _, op := range ops {
		switch op.Code {
		case hip.OpGemvQ8_1:
			counts.int8++
		case hip.OpLMHead:
			counts.streaming++
		case hip.OpGemv:
			counts.perRow++
		}
	}
	return counts
}

// opMatvecPath maps a matvec op code to the route it runs, and reports false for
// the non-matvec operations.
func opMatvecPath(op hip.OpInfo) (hip.DecodeGemvPath, bool) {
	switch op.Code {
	case hip.OpGemvQ8_1:
		return hip.DecodeGemvInt8, true
	case hip.OpLMHead:
		return hip.DecodeGemvStreaming, true
	case hip.OpGemv:
		return hip.DecodeGemvPerRow, true
	default:
		return hip.DecodeGemvPerRow, false
	}
}

// checkDecodeRule verifies that a default decode plan follows the dispatch rule
// operation by operation: every matvec the rule covers runs the route it
// selects, and the per-row operations are exactly the shapes it does not cover.
func checkDecodeRule(t *testing.T, ops []hip.OpInfo, fp32Streaming bool) {
	t.Helper()
	for _, op := range ops {
		got, isMatvec := opMatvecPath(op)
		if !isMatvec {
			continue
		}
		want := hip.SelectDecodeGemvPath(opDType(op), op.N[0], op.N[1], fp32Streaming)
		if got != want {
			t.Fatalf("plan runs %s/%dx%d on the %s route, but the rule selects %s",
				op.Name(), op.N[0], op.N[1], got, want)
		}
	}
}

// logitDiff returns the maximum absolute difference and the distance between the
// two argmax indices.
func logitDiff(a, b []float32) (float64, int) {
	if len(a) != len(b) {
		return math.Inf(1), -1
	}
	maxDiff := 0.0
	for i := range a {
		if diff := math.Abs(float64(a[i] - b[i])); diff > maxDiff {
			maxDiff = diff
		}
	}
	ia, ib := argmax(a), argmax(b)
	if ia > ib {
		return maxDiff, int(ia - ib)
	}
	return maxDiff, int(ib - ia)
}

// opDType returns the weight dtype an op carries in u[0].
func opDType(op hip.OpInfo) tensor.DType { return tensor.DType(op.U[0]) }

// TestGPUInt8MatvecMatchesFp32 isolates the arithmetic change of the int8 route.
// The int8 session dots the Q8_1 activation with the int8 digits of each packed
// weight; the default session runs the FP32 streaming kernel on the same shape,
// so both plans have the same operations in the same order and differ only in
// how each product is formed. The int8 route is not the default because it is
// measured slower end to end; this test is what pins its
// arithmetic while it stays available for measurement.
//
// The recorded bound is wider than the FP32 route's own agreement because the
// activation is quantized to 8 bits: each digit carries up to 1/254 of the
// block's amax, which is the error the prefill int8 route already carries.
func TestGPUInt8MatvecMatchesFp32(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	fp32RT, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer fp32RT.Close()
	if counts := countMatvec(fp32RT.session.plan.Ops()); counts.int8 != 0 {
		t.Fatalf("the fp32 reference plan runs %d int8 matvecs", counts.int8)
	}
	refLogits, err := fp32RT.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	const steps = 8
	refTokens := make([]int32, 0, steps)
	refSteps := make([][]float32, 0, steps)
	for range steps {
		refToken := argmax(refLogits)
		refTokens = append(refTokens, refToken)
		refSteps = append(refSteps, append([]float32(nil), refLogits...))
		if refLogits, err = fp32RT.Step(refToken); err != nil {
			t.Fatal(err)
		}
	}

	// Recorded for this prompt: 0.2656 maximum absolute logit difference over
	// eight greedy decode steps with identical tokens, against logits that reach
	// about 30. The prefill int8 route records 0.0031 at a short prompt and
	// 0.264 at 2048 tokens, so the decode route sits at the top of that band:
	// both quantize the activation to 8 bits per 32 values, and a decode step
	// feeds each step's logits into the next token, so the difference can
	// accumulate. The gate is the recorded value with margin; the invariant the
	// stage requires is the identical token sequence.
	const tolerance = 0.6
	for _, tc := range []struct {
		name string
		opts Options
	}{
		{name: "fused", opts: Options{Int8Matvec: true}},
		{name: "prequantized", opts: Options{Int8Matvec: true, PrequantizedInt8: true}},
	} {
		int8RT, err := newRuntime(reader, model, 64, tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		int8Ops := int8RT.session.plan.Ops()
		if counts := countMatvec(int8Ops); counts.int8 == 0 {
			int8RT.Close()
			t.Fatalf("%s: the plan runs no int8 matvec", tc.name)
		}
		checkDecodeRule(t, int8Ops, true)
		logits, err := int8RT.Forward(prompt)
		if err != nil {
			int8RT.Close()
			t.Fatal(err)
		}
		maxDiff := 0.0
		for step := range steps {
			token := argmax(logits)
			if token != refTokens[step] {
				int8RT.Close()
				t.Fatalf("%s token %d: int8 route %d, fp32 route %d", tc.name, step,
					token, refTokens[step])
			}
			diff, _ := logitDiff(logits, refSteps[step])
			if diff > maxDiff {
				maxDiff = diff
			}
			if step+1 < steps {
				if logits, err = int8RT.Step(token); err != nil {
					int8RT.Close()
					t.Fatal(err)
				}
			}
		}
		int8RT.Close()
		if maxDiff > tolerance {
			t.Fatalf("%s: the int8 and fp32 routes differ by %.6g, tolerance %.6g",
				tc.name, maxDiff, tolerance)
		}
		t.Logf("%s: %d steps, tokens %v, max abs logit diff %.6g (tolerance %.6g)",
			tc.name, steps, refTokens, maxDiff, tolerance)
	}
}
