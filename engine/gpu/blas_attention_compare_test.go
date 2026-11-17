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

// TestGPUBlasAttentionMatchesTiled compares identical full-model prompts using
// F16 KV, changing only the prefill attention dispatch: the tiled kernel
// against the default matrixized kernel.
func TestGPUBlasAttentionMatchesTiled(t *testing.T) {
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
	n := 2048
	repetitions := 450
	if os.Getenv("YUUKA_TEST_BLAS_LONG") == "1" {
		n = 8192
		repetitions = 1800
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
	var baseline []float32
	for _, disable := range []bool{true, false} {
		rt, err := newRuntime(reader, model, capacity, Options{
			KVType:               tensor.F16,
			DisableBlasAttention: disable,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := rt.SetPrefillChunk(512); err != nil {
			rt.Close()
			t.Fatal(err)
		}
		if got, want := rt.model.attentionBlasEnabled, !disable; got != want {
			rt.Close()
			t.Fatalf("BLAS attention enabled=%v, want %v", got, want)
		}
		got, err := rt.Forward(ids)
		rt.Close()
		if err != nil {
			t.Fatalf("BLAS=%v: %v", !disable, err)
		}
		if disable {
			baseline = append([]float32(nil), got...)
			continue
		}
		var maxDiff float64
		for i, v := range got {
			delta := math.Abs(float64(v - baseline[i]))
			if delta > maxDiff {
				maxDiff = delta
			}
		}
		t.Logf("model BLAS attention vs tiled: top1=%d/%d max_diff=%.6g", top1(got), top1(baseline), maxDiff)
		if top1(got) != top1(baseline) || maxDiff > 0.05 {
			t.Fatalf("BLAS attention exceeds model-level 0.05 logit gate")
		}
	}
}
