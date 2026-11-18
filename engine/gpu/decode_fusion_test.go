//go:build hip && linux

package gpu

import (
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

// TestGPUDecodeFusionsMatchUnfused builds the decode plan twice: with the two D5
// fusions (the residual epilogue on the projection that ends a sub-layer and the
// gate/up pair that forms both dots and their SwiGLU product in one launch), and
// with Options{NoResidualFusion, NoPairFusion} which keeps the separate ADD and
// SILU_MUL. Both forms perform the same arithmetic in the same order, so the
// logits must agree to rounding; the projection pair reassociates nothing,
// because each dot is still a single wave reduction over the same K order.
func TestGPUDecodeFusionsMatchUnfused(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	fused, err := newRuntime(reader, model, 64, Options{})
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer fused.Close()
	plain, err := newRuntime(reader, model, 64, Options{NoResidualFusion: true, NoPairFusion: true})
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()

	// The two plans must actually differ: the fused one has no ADD or SILU_MUL
	// and adds GEMV_RESIDUAL and GEMV_PAIR instead.
	count := func(ops []hip.OpInfo) map[uint32]int {
		counts := make(map[uint32]int)
		for _, op := range ops {
			counts[op.Code]++
		}
		return counts
	}
	fusedCounts, plainCounts := count(fused.session.plan.Ops()), count(plain.session.plan.Ops())
	t.Logf("fused plan: %d residual, %d pair, %d add, %d silu_mul, %d operations",
		fusedCounts[hip.OpGemvResidual], fusedCounts[hip.OpGemvPair], fusedCounts[hip.OpAdd],
		fusedCounts[hip.OpSiluMul], len(fused.session.plan.Ops()))
	t.Logf("unfused plan: %d residual, %d pair, %d add, %d silu_mul, %d operations",
		plainCounts[hip.OpGemvResidual], plainCounts[hip.OpGemvPair], plainCounts[hip.OpAdd],
		plainCounts[hip.OpSiluMul], len(plain.session.plan.Ops()))
	if fusedCounts[hip.OpGemvResidual] == 0 || fusedCounts[hip.OpGemvPair] == 0 {
		t.Fatalf("the default plan runs %d residual and %d pair operations, want both",
			fusedCounts[hip.OpGemvResidual], fusedCounts[hip.OpGemvPair])
	}
	if plainCounts[hip.OpGemvResidual] != 0 || plainCounts[hip.OpGemvPair] != 0 {
		t.Fatalf("the unfused plan still runs %d residual and %d pair operations",
			plainCounts[hip.OpGemvResidual], plainCounts[hip.OpGemvPair])
	}
	if payload := plainCounts[hip.OpAdd] + plainCounts[hip.OpSiluMul]; payload == 0 {
		t.Fatal("the unfused plan runs no ADD or SILU_MUL to compare against")
	}

	fusedLogits, err := fused.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	plainLogits, err := plain.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	const steps = 8
	tokens := make([]int32, 0, steps)
	worst := 0.0
	for range steps {
		fusedToken, plainToken := argmax(fusedLogits), argmax(plainLogits)
		if fusedToken != plainToken {
			t.Fatalf("step %d: fused picks %d, unfused picks %d", len(tokens), fusedToken,
				plainToken)
		}
		tokens = append(tokens, fusedToken)
		if fusedLogits, err = fused.Step(fusedToken); err != nil {
			t.Fatal(err)
		}
		if plainLogits, err = plain.Step(plainToken); err != nil {
			t.Fatal(err)
		}
		if diff, _ := logitDiff(fusedLogits, plainLogits); diff > worst {
			worst = diff
		}
	}
	// The fusions move work, not arithmetic, so the bound is the float
	// reassociation of the SwiGLU product (out = silu(g)*u in one instruction
	// stream against a separate pass over the same values).
	const tolerance = 1e-3
	if worst > tolerance {
		t.Fatalf("the fused and unfused plans differ by %.8g over eight greedy steps,"+
			" tolerance %.0e", worst, tolerance)
	}
	t.Logf("tokens %v, max abs logit diff over %d steps %.8g (tolerance %.0e)", tokens,
		steps, worst, tolerance)
}
