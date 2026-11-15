package qwen35

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/tests/fixture"
	"github.com/yuuka-ai/yuuka/tokenizer"
)

func openTinyMoEFixture(t *testing.T) (*gguf.Reader, *tokenizer.Tokenizer, *Model) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny-qwen35moe.gguf")
	if err := fixture.WriteTinyQwen35MoE(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	return reader, tok, model
}

func TestTinyMoEFixtureLoad(t *testing.T) {
	_, _, model := openTinyMoEFixture(t)
	cfg := model.Config
	if cfg.Architecture != MoEArchitecture {
		t.Fatalf("architecture %q, want %q", cfg.Architecture, MoEArchitecture)
	}
	if !cfg.IsMoE() {
		t.Fatal("config did not detect MoE")
	}
	if cfg.ExpertCount != 4 || cfg.ExpertUsedCount != 2 {
		t.Fatalf("experts count=%d used=%d", cfg.ExpertCount, cfg.ExpertUsedCount)
	}
	if cfg.ExpertFFNSize != 16 || cfg.SharedFFNSize != 16 {
		t.Fatalf("expert widths routed=%d shared=%d", cfg.ExpertFFNSize, cfg.SharedFFNSize)
	}
	for i := range model.Layers {
		layer := &model.Layers[i]
		if layer.MoE == nil {
			t.Fatalf("layer %d has no MoE FFN", i)
		}
		if layer.FFN.Gate.Rows != 0 || layer.FFN.Up.Rows != 0 || layer.FFN.Down.Rows != 0 {
			t.Fatalf("layer %d loaded a dense FFN", i)
		}
		if layer.MoE.Router.Rows != int64(cfg.ExpertCount) ||
			layer.MoE.Router.Cols != int64(cfg.HiddenSize) {
			t.Fatalf("layer %d router shape %dx%d", i, layer.MoE.Router.Rows, layer.MoE.Router.Cols)
		}
		if len(layer.MoE.SharedGate) != cfg.HiddenSize {
			t.Fatalf("layer %d shared gate width %d", i, len(layer.MoE.SharedGate))
		}
	}
}

func TestTinyMoEFixtureForward(t *testing.T) {
	_, tok, model := openTinyMoEFixture(t)
	ids, err := tok.Encode("Hello")
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewState(model.Config, 16)
	if err != nil {
		t.Fatal(err)
	}
	first, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range first {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logit %d is not finite: %v", i, value)
		}
	}
	firstCopy := append([]float32(nil), first...)
	if _, err := model.Forward(ids, state); err != nil {
		t.Fatal(err)
	}
	state.Reset()
	second, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(firstCopy, second) {
		t.Fatal("MoE fixture forward is not deterministic after reset")
	}
}

func TestExpertSliceDoesNotOverlap(t *testing.T) {
	_, _, model := openTinyMoEFixture(t)
	layer := &model.Layers[0]
	experts := layer.MoE.GateExps
	first, err := expertSlice(experts, 0, model.Config.ExpertFFNSize)
	if err != nil {
		t.Fatal(err)
	}
	second, err := expertSlice(experts, 1, model.Config.ExpertFFNSize)
	if err != nil {
		t.Fatal(err)
	}
	if first.Rows != int64(model.Config.ExpertFFNSize) || first.Cols != int64(model.Config.HiddenSize) {
		t.Fatalf("expert shape %dx%d", first.Rows, first.Cols)
	}
	if len(first.Data) != len(second.Data) {
		t.Fatal("expert slices have different sizes")
	}
	if &first.Data[0] == &second.Data[0] {
		t.Fatal("expert slices share storage")
	}
	if _, err := expertSlice(experts, model.Config.ExpertCount, model.Config.ExpertFFNSize); err == nil {
		t.Fatal("out-of-range expert was accepted")
	}
}

func TestSelectExperts(t *testing.T) {
	logits := []float32{0.1, 0.9, 0.5, 0.3}
	ids := make([]int32, 2)
	weights := make([]float32, 2)
	selectExperts(logits, ids, weights)
	if !reflect.DeepEqual(ids, []int32{1, 2}) {
		t.Fatalf("selected experts %v", ids)
	}
	if math.Abs(float64(weights[0]+weights[1]-1)) > 1e-6 {
		t.Fatalf("selected weights %v do not sum to one", weights)
	}
	if weights[0] <= weights[1] {
		t.Fatalf("router weights %v are not ordered", weights)
	}
}

// TestQwen35MoEGoldenGeneration is opt-in through YUUKA_TEST_MOE_MODEL. The
// expected tokens are llama.cpp's greedy continuation for the same GGUF.
func TestQwen35MoEGoldenGeneration(t *testing.T) {
	path := os.Getenv("YUUKA_TEST_MOE_MODEL")
	if path == "" {
		t.Skip("YUUKA_TEST_MOE_MODEL is not set")
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	tok, err := tokenizer.NewFromGGUF(reader)
	if err != nil {
		t.Fatal(err)
	}
	model, err := Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	if model.Config.Architecture != MoEArchitecture {
		t.Skipf("model architecture %q is not %q", model.Config.Architecture, MoEArchitecture)
	}
	state, err := NewState(model.Config, 64)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tok.Encode("The capital of France is")
	if err != nil {
		t.Fatal(err)
	}
	logits, err := model.Forward(ids, state)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 0, 4)
	for i := 0; i < 4; i++ {
		best := int32(0)
		for id, value := range logits {
			if value > logits[best] {
				best = int32(id)
			}
		}
		got = append(got, best)
		if i+1 < 4 {
			logits, err = model.Step(best, state)
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	want := []int32{11751, 11, 264, 3177} // " Paris, a city"
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens %v, want %v", got, want)
	}
}
