//go:build hip && linux

package gpu

import (
	"math"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

// TestGPUStreamingTailMatchesGemv compares the two decode output-projection
// routes on the real model. The default session runs the streaming LM head
// kernel for the tail (roadmap4 D1); a session opened with ForceGemvTail runs
// the per-row GEMV the earlier tree used. Both plans read the same weights and
// the same normalized hidden state, so the only difference is the summation
// order inside the projection.
//
// The tolerance is absolute because the comparison is over the logits of one
// vocabulary: the streaming kernel sums four K values per lane and reduces
// across the wave, while the GEMV sums a strided set of K values per thread and
// reduces over the block. Both accumulate in FP32, so the difference is that
// reassociation and nothing else.
func TestGPUStreamingTailMatchesGemv(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	streaming, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer streaming.Close()
	perRow, err := newRuntime(reader, model, 64, Options{ForceGemvTail: true})
	if err != nil {
		t.Fatal(err)
	}
	defer perRow.Close()

	// The two plans must actually differ: a change that routed both through the
	// same kernel would make the comparison vacuous.
	if got := tailOpCode(t, streaming); got != hip.OpLMHead {
		t.Fatalf("default decode tail runs %s, want the streaming LM head",
			hip.OpName(got))
	}
	if got := tailOpCode(t, perRow); got != hip.OpGemv {
		t.Fatalf("forced decode tail runs %s, want the per-row GEMV",
			hip.OpName(got))
	}

	streamLogits, err := streaming.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	perRowLogits, err := perRow.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	// The prefill tail already ran the streaming kernel on both routes, so this
	// pair of logits is not the comparison; it only confirms the two sessions
	// saw the same prompt.
	prefillDiff, _ := logitDiff(streamLogits, perRowLogits)
	t.Logf("prefill tail: max abs logit diff %.6g (same kernel on both routes)",
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
	// The recorded maximum over the eight decode steps of this prompt. The per-element gate is 2e-4 relative in the kernel test;
	// logits reach about 30 in magnitude, so the absolute bound is wider.
	const tolerance = 0.02
	if maxDiff > tolerance {
		t.Fatalf("the two routes differ by %.6g, tolerance %.6g", maxDiff, tolerance)
	}
	t.Logf("tokens %v, max abs logit diff over %d steps %.6g (tolerance %.6g)",
		streamTokens, steps, maxDiff, tolerance)
}

// tailOpCode returns the op code of the last planned operation, which is the
// output projection for a decode plan.
func tailOpCode(t *testing.T, rt *Runtime) uint32 {
	t.Helper()
	ops := rt.session.plan.Ops()
	if len(ops) == 0 {
		t.Fatal("decode plan holds no operations")
	}
	return ops[len(ops)-1].Code
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
