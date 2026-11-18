//go:build hip && linux

package gpu

import (
	"math"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// TestGPUStreamingDecodeMatchesGemv compares the two decode matvec families on
// the real model. The default session runs the streaming channel-parallel kernel
// for every matvec the dispatch rule covers (roadmap4 D2); a session opened with
// ForceDecodeGemv runs the per-row GEMV the earlier tree used for all of them.
// Both plans read the same weights and the same activations, so the only
// difference is the summation order inside each projection.
//
// The tolerance is absolute because the comparison is over the logits of one
// vocabulary: the streaming kernel sums four K values per lane and reduces
// across the wave, while the GEMV sums a strided set of K values per thread and
// reduces over the block. Both accumulate in FP32, so the difference is that
// reassociation and nothing else.
func TestGPUStreamingDecodeMatchesGemv(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	streaming, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer streaming.Close()
	perRow, err := newRuntime(reader, model, 64, Options{ForceDecodeGemv: true})
	if err != nil {
		t.Fatal(err)
	}
	defer perRow.Close()

	// The two plans must actually differ, and every matvec in each must follow
	// the dispatch rule rather than a per-call choice.
	streamingOps := streaming.session.plan.Ops()
	perRowOps := perRow.session.plan.Ops()
	streamingLMHead, streamingGemv := countMatvec(streamingOps)
	perRowLMHead, perRowGemv := countMatvec(perRowOps)
	checkDecodeRule(t, streamingOps)
	t.Logf("streaming plan: %d streaming, %d per-row matvecs; per-row plan: %d, %d",
		streamingLMHead, streamingGemv, perRowLMHead, perRowGemv)
	if perRowLMHead != 0 || perRowGemv == 0 {
		t.Fatalf("the forced plan runs %d streaming and %d per-row matvecs, want only"+
			" per-row", perRowLMHead, perRowGemv)
	}
	if streamingLMHead <= model.Config.BlockCount {
		t.Fatalf("the default plan runs %d streaming matvecs, want at least one per"+
			" layer plus the output projection", streamingLMHead)
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

// countMatvec returns how many streaming and how many per-row matvecs a decode
// plan holds.
func countMatvec(ops []hip.OpInfo) (streaming, perRow int) {
	for _, op := range ops {
		switch op.Code {
		case hip.OpLMHead:
			streaming++
		case hip.OpGemv:
			perRow++
		}
	}
	return streaming, perRow
}

// checkDecodeRule verifies that a default decode plan follows the dispatch rule
// operation by operation: every matvec the rule covers streams, and the only
// per-row matvecs left are the shapes it does not cover.
func checkDecodeRule(t *testing.T, ops []hip.OpInfo) {
	t.Helper()
	for _, op := range ops {
		switch op.Code {
		case hip.OpLMHead:
			if got := hip.SelectDecodeGemvPath(opDType(op), op.N[0], op.N[1]); got != hip.DecodeGemvStreaming {
				t.Fatalf("plan streams %s/%dx%d, but the rule says %s", op.Name(),
					op.N[0], op.N[1], got)
			}
		case hip.OpGemv:
			if got := hip.SelectDecodeGemvPath(opDType(op), op.N[0], op.N[1]); got != hip.DecodeGemvPerRow {
				t.Fatalf("plan runs %s/%dx%d per-row, but the rule says %s", op.Name(),
					op.N[0], op.N[1], got)
			}
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
