//go:build hip && linux

package gpu

import (
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

// TestGPUFusedDeltaNetMatchesLegacy compares the two ways roadmap4 D4 can build
// a recurrent layer on the real model: the fused CONV_L2NORM and
// RECURRENT_TOKEN pair, which is the default, and the five separate operators it
// replaced (CONV1D, two L2NORM, GDN, GATED_RMSNORM). Both sessions run the same
// matvec route, so the measured difference is the delta-net change alone.
//
// The fused kernel folds the decay into the rank-1 update and reassociates the
// column sums, so the difference is that reassociation and the single rounding
// of the fused multiply-add, not a change of arithmetic class. The native test
// tests/hip/deltanet_test.cpp holds the same comparison at the operator level,
// including a host reference, and a decay far enough from one that a missing
// factor cannot hide.
func TestGPUFusedDeltaNetMatchesLegacy(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	fused, err := newRuntime(reader, model, 64, Options{})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer fused.Close()
	legacy, err := newRuntime(reader, model, 64, Options{LegacyDeltaNet: true})
	if err != nil {
		t.Fatal(err)
	}
	defer legacy.Close()

	// The two plans must actually differ: the fused pair on one side, the five
	// separate operators on the other.
	type deltaNetCounts struct{ fused, separate, layers int }
	count := func(ops []hip.OpInfo) deltaNetCounts {
		var counts deltaNetCounts
		for _, op := range ops {
			switch op.Code {
			case hip.OpConvL2Norm, hip.OpRecurrentToken:
				counts.fused++
			case hip.OpConv1D, hip.OpGDN, hip.OpGatedRMSNorm:
				counts.separate++
			}
			if op.Code == hip.OpRecurrentToken {
				counts.layers++
			}
		}
		return counts
	}
	fusedCounts, legacyCounts := count(fused.session.plan.Ops()), count(legacy.session.plan.Ops())
	t.Logf("fused plan: %d fused and %d separate delta-net operators over %d layers;"+
		" legacy plan: %d and %d", fusedCounts.fused, fusedCounts.separate,
		fusedCounts.layers, legacyCounts.fused, legacyCounts.separate)
	if fusedCounts.separate != 0 || fusedCounts.fused != 2*fusedCounts.layers ||
		fusedCounts.layers == 0 {
		t.Fatalf("the default plan runs %d fused and %d separate delta-net operators,"+
			" want one convolution and one recurrent operator per layer",
			fusedCounts.fused, fusedCounts.separate)
	}
	if legacyCounts.fused != 0 || legacyCounts.separate == 0 {
		t.Fatalf("the forced plan runs %d fused and %d separate delta-net operators,"+
			" want the separate operators only", legacyCounts.fused, legacyCounts.separate)
	}

	fusedLogits, err := fused.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	legacyLogits, err := legacy.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	// Prefill runs the chunked path, which D4 does not change, so it must agree
	// exactly up to the plan-independent arithmetic.
	prefillDiff, _ := logitDiff(fusedLogits, legacyLogits)
	t.Logf("prefill tail: max abs logit diff %.8g", prefillDiff)
	if prefillDiff > 1e-4 {
		t.Fatalf("the two plans disagree after prefill by %.8g, but the chunked"+
			" prefill path is shared", prefillDiff)
	}

	// The decode steps are where the two plans diverge, and the delta-net state
	// is path dependent, so the difference is measured over the greedy run.
	const steps = 8
	tokens := make([]int32, 0, steps)
	worst := 0.0
	for range steps {
		fusedToken, legacyToken := argmax(fusedLogits), argmax(legacyLogits)
		if fusedToken != legacyToken {
			t.Fatalf("step %d: the plans pick different tokens %d and %d", len(tokens),
				fusedToken, legacyToken)
		}
		tokens = append(tokens, fusedToken)
		if fusedLogits, err = fused.Step(fusedToken); err != nil {
			t.Fatal(err)
		}
		if legacyLogits, err = legacy.Step(legacyToken); err != nil {
			t.Fatal(err)
		}
		if diff, _ := logitDiff(fusedLogits, legacyLogits); diff > worst {
			worst = diff
		}
	}
	const tolerance = 0.02
	if worst > tolerance {
		t.Fatalf("the fused and separate delta-net plans differ by %.8g over eight"+
			" greedy steps, tolerance %.2f", worst, tolerance)
	}
	t.Logf("tokens %v, max abs logit diff over %d steps %.8g (tolerance %.2f)", tokens,
		steps, worst, tolerance)
}
