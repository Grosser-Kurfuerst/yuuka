//go:build hip && linux

package gpu

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tensor"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

// TestGPUAttentionPathsAgree compares identical full-model prompts across the
// three prefill attention kernels: the tiled kernel (forced), the FP16 MMA
// kernel (default), and the legacy rocBLAS path. The KV pool is F16 so every
// arm reads the same stored values, and only the kernel changes.
func TestGPUAttentionPathsAgree(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MODEL unset")
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	// A 64-token prompt is the shortest chunk the FP16 MMA kernel accepts, and
	// it stays on the FP16 prefill route. The long prompt switches the prefill
	// GEMMs to the int8 Q8_1 route, which lands the two arms on different
	// activation grid points.
	n := 64
	repetitions := 7
	gate := 0.05
	if os.Getenv("YUUKA_TEST_ATTENTION_LONG") == "1" {
		n = 2048
		repetitions = 450
		gate = 0.3
	}
	ids, err := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", repetitions))
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) < n {
		t.Skipf("only %d tokens, need %d", len(ids), n)
	}
	ids = ids[:n]
	capacity := n + 64
	type arm struct {
		name string
		opts Options
		// mma and blas report the dispatch flag the model resolved to.
		mma  bool
		blas bool
	}
	arms := []arm{
		{name: "tiled", opts: Options{KVType: tensor.F16, ForceTiledAttention: true}},
		{name: "mma", opts: Options{KVType: tensor.F16}, mma: true},
		{name: "blas", opts: Options{KVType: tensor.F16, LegacyBlasAttention: true}, blas: true},
	}
	logits := make(map[string][]float32, len(arms))
	for _, a := range arms {
		rt, err := newRuntime(reader, model, capacity, a.opts)
		if err != nil {
			t.Fatal(err)
		}
		if err := rt.SetPrefillChunk(512); err != nil {
			rt.Close()
			t.Fatal(err)
		}
		if got, want := rt.model.attentionBlasEnabled, a.blas; got != want {
			rt.Close()
			t.Fatalf("%s: BLAS attention enabled=%v, want %v", a.name, got, want)
		}
		got, err := rt.Forward(ids)
		rt.Close()
		if err != nil {
			t.Fatalf("%s: %v", a.name, err)
		}
		logits[a.name] = got
	}
	baseline := logits["tiled"]
	for _, a := range arms[1:] {
		got := logits[a.name]
		var maxDiff float64
		for i, v := range got {
			delta := math.Abs(float64(v - baseline[i]))
			if delta > maxDiff {
				maxDiff = delta
			}
		}
		t.Logf("model %s attention vs tiled: top1=%d/%d max_diff=%.6g", a.name,
			top1(got), top1(baseline), maxDiff)
		// The arms share everything but the attention kernel. The short-prompt
		// gate matches the tolerance the KV-format test uses; the long-prompt
		// gate matches the Q8_1 route's measured sensitivity, which moves logits
		// by the same amount as changing the KV storage format on one kernel.
		// The measurements are recorded in docs/benchmarks.md.
		if top1(got) != top1(baseline) || maxDiff > gate {
			t.Fatalf("%s attention exceeds the %g model-level logit gate (%.6g)",
				a.name, gate, maxDiff)
		}
	}
}
