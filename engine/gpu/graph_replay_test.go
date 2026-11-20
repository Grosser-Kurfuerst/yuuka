//go:build hip && linux

package gpu

import (
	"testing"
	"time"

	"github.com/yuuka-ai/yuuka/tensor"
)

// TestGPUGraphReplayMatchesAndTimes measures the opt-in HIP Graph replay of the
// decode plan that the residual and gate/up fusions re-evaluated: with four hundred dependent
// operations the submission side could still be a gap, and a captured plan
// removes every per-operation host submission. The test requires the replayed
// session to produce the same greedy tokens as the plain submission path and
// reports both step times, because the answer (the gap is already hidden behind
// the device work) is the measurement, not an assertion. serving.md owns the
// replay contract.
func TestGPUGraphReplayMatchesAndTimes(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	const steps = 32
	run := func(opts Options) ([]int32, float64) {
		rt, err := newRuntime(reader, model, 64, opts)
		if err != nil {
			t.Skipf("HIP runtime unavailable: %v", err)
		}
		defer rt.Close()
		logits, err := rt.Forward(prompt)
		if err != nil {
			t.Fatal(err)
		}
		tokens := make([]int32, 0, steps)
		started := time.Now()
		for range steps {
			token := int32(argmax(logits))
			tokens = append(tokens, token)
			if logits, err = rt.Step(token); err != nil {
				t.Fatal(err)
			}
		}
		ms := float64(time.Since(started).Microseconds()) / 1000 / steps
		return tokens, ms
	}
	// Every KV pool format takes the same split-key attention kernel, and the
	// dynamic variant reads the position from device memory, so each format's
	// replayed graph must reproduce its own plain token stream and all formats
	// must agree with each other.
	var reference []int32
	for _, kvType := range []tensor.DType{tensor.F32, tensor.F16, tensor.Q8_0} {
		plainTokens, plainMS := run(Options{KVType: kvType, KVTypeSet: true})
		graphTokens, graphMS := run(Options{KVType: kvType, KVTypeSet: true, EnableGraph: true,
			GraphMinSteps: 2})
		if len(plainTokens) != len(graphTokens) {
			t.Fatalf("%s: plain path produced %d tokens, replayed path %d", kvType,
				len(plainTokens), len(graphTokens))
		}
		for i := range plainTokens {
			if plainTokens[i] != graphTokens[i] {
				t.Fatalf("%s token %d: plain %d, replayed %d", kvType, i,
					plainTokens[i], graphTokens[i])
			}
		}
		if reference == nil {
			reference = plainTokens
		} else {
			for i := range reference {
				if plainTokens[i] != reference[i] {
					t.Fatalf("%s token %d: %d, F32 %d", kvType, i, plainTokens[i],
						reference[i])
				}
			}
		}
		t.Logf("%s %d decode steps: plain %.3f ms/step (%.2f tok/s), replayed %.3f"+
			" ms/step (%.2f tok/s), ratio %.3f", kvType, steps, plainMS, 1000/plainMS,
			graphMS, 1000/graphMS, graphMS/plainMS)
	}
}
