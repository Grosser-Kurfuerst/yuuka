package serving

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/gguf"
	"github.com/yuuka-ai/yuuka/memory"
	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func TestQwen35MoEBackendPrefill(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35moe.gguf")
	if err := fixture.WriteTinyQwen35MoE(path); err != nil {
		t.Fatal(err)
	}
	reader, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	model, err := qwen35.Load(reader)
	if err != nil {
		t.Fatal(err)
	}
	if !model.Config.IsMoE() {
		t.Fatal("fixture did not load as a MoE model")
	}
	backend, err := NewQwen35BackendWithFormat(model, "tiny-qwen35moe", 8, 8, qwen35.KVF32)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = backend.Close() })
	table := memory.NewTable(backend.Allocator())
	t.Cleanup(table.Release)
	if err := table.Cover(8); err != nil {
		t.Fatal(err)
	}
	seq, err := backend.NewSession(table, 8, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = seq.Close() })
	logits, err := seq.Prefill([]int32{1, 2, 3}, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logit %d is not finite: %v", i, value)
		}
	}
}
