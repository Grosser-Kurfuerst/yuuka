package main

import (
	"path/filepath"
	"testing"

	"github.com/yuuka-ai/yuuka/model/qwen35"
	"github.com/yuuka-ai/yuuka/tests/fixture"
)

func TestCPUEngineAcceptsKVFormatForQwen35(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35.gguf")
	if err := fixture.WriteTinyQwen35(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	runner, err := loaded.cpuEngine(16, nil, qwen35.KVQ8_0)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestCPUEngineLoadsQwen35MoE(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny-qwen35moe.gguf")
	if err := fixture.WriteTinyQwen35MoE(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	if loaded.qwen == nil || !loaded.qwen.Config.IsMoE() {
		t.Fatal("qwen35moe fixture did not load as a MoE model")
	}
	runner, err := loaded.cpuEngine(16, nil, qwen35.KVQ8_0)
	if err != nil {
		t.Fatal(err)
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}
