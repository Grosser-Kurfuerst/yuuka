// Package qwen35 implements the Qwen3.5/Qwen3.6 hybrid decoder: gated delta-net
// linear-attention layers interleaved with full GQA attention layers. The
// qwen35moe variant replaces the dense FFN with routed and shared experts.
package qwen35

import (
	"fmt"
	"strings"

	"github.com/yuuka-ai/yuuka/gguf"
)

const (
	Architecture    = "qwen35"
	MoEArchitecture = "qwen35moe"
)

// Config is the validated runtime configuration for one Qwen3.5/Qwen3.6 model.
type Config struct {
	Architecture          string
	BlockCount            int
	ContextLength         int
	HiddenSize            int
	FFNSize               int
	HeadCount             int
	KVHeadCount           int
	HeadDim               int
	ValueDim              int
	RoPEDim               int
	RoPESections          [4]int
	RoPEFreqBase          float64
	RMSNormEps            float32
	VocabSize             int
	SSMConvKernel         int
	SSMStateSize          int
	SSMGroupCount         int
	SSMTimeStepRank       int
	SSMInnerSize          int
	FullAttentionInterval int
	// MoE fields are set only for qwen35moe. ExpertFFNSize is one routed
	// expert's SwiGLU width; SharedFFNSize is the shared expert's width.
	ExpertCount     int
	ExpertUsedCount int
	ExpertFFNSize   int
	SharedFFNSize   int
}

// LoadConfig reads and validates Qwen3.5 hyperparameters from GGUF metadata.
func LoadConfig(r *gguf.Reader) (*Config, error) {
	arch, err := r.String("general.architecture")
	if err != nil {
		return nil, err
	}
	if arch != Architecture && arch != MoEArchitecture {
		return nil, fmt.Errorf("model architecture %q is not %q or %q", arch, Architecture, MoEArchitecture)
	}
	if value, ok := r.MetadataValue(arch + ".rope.scaling.type"); ok {
		if err := validateRoPEScaling(arch+".rope.scaling.type", value); err != nil {
			return nil, err
		}
	}
	cfg := &Config{Architecture: arch}
	metaKey := func(name string) string { return arch + "." + name }
	intKey := func(key string) (int, error) {
		value, err := r.Int(key)
		if err != nil {
			return 0, err
		}
		if value <= 0 || value > 1<<31 {
			return 0, fmt.Errorf("metadata %s has invalid value %d", key, value)
		}
		return int(value), nil
	}
	floatKey := func(key string) (float64, error) {
		value, err := r.Float(key)
		if err != nil {
			return 0, err
		}
		return value, nil
	}
	if cfg.BlockCount, err = intKey(metaKey("block_count")); err != nil {
		return nil, err
	}
	if cfg.ContextLength, err = intKey(metaKey("context_length")); err != nil {
		return nil, err
	}
	if cfg.HiddenSize, err = intKey(metaKey("embedding_length")); err != nil {
		return nil, err
	}
	if arch == MoEArchitecture {
		if cfg.ExpertCount, err = intKey(metaKey("expert_count")); err != nil {
			return nil, err
		}
		if cfg.ExpertUsedCount, err = intKey(metaKey("expert_used_count")); err != nil {
			return nil, err
		}
		if cfg.ExpertFFNSize, err = loadExpertFFNSize(r, metaKey("expert_feed_forward_length")); err != nil {
			return nil, err
		}
		if _, ok := r.MetadataValue(metaKey("expert_shared_feed_forward_length")); ok {
			if cfg.SharedFFNSize, err = intKey(metaKey("expert_shared_feed_forward_length")); err != nil {
				return nil, err
			}
		} else {
			cfg.SharedFFNSize = cfg.ExpertFFNSize
		}
	} else if cfg.FFNSize, err = intKey(metaKey("feed_forward_length")); err != nil {
		return nil, err
	}
	if cfg.HeadCount, err = intKey(metaKey("attention.head_count")); err != nil {
		return nil, err
	}
	if cfg.KVHeadCount, err = intKey(metaKey("attention.head_count_kv")); err != nil {
		return nil, err
	}
	if cfg.HeadDim, err = intKey(metaKey("attention.key_length")); err != nil {
		return nil, err
	}
	if cfg.ValueDim, err = intKey(metaKey("attention.value_length")); err != nil {
		return nil, err
	}
	if cfg.RoPEDim, err = intKey(metaKey("rope.dimension_count")); err != nil {
		return nil, err
	}
	if cfg.RoPEFreqBase, err = floatKey(metaKey("rope.freq_base")); err != nil {
		return nil, err
	}
	eps, err := floatKey(metaKey("attention.layer_norm_rms_epsilon"))
	if err != nil {
		return nil, err
	}
	cfg.RMSNormEps = float32(eps)
	if cfg.SSMConvKernel, err = intKey(metaKey("ssm.conv_kernel")); err != nil {
		return nil, err
	}
	if cfg.SSMStateSize, err = intKey(metaKey("ssm.state_size")); err != nil {
		return nil, err
	}
	if cfg.SSMGroupCount, err = intKey(metaKey("ssm.group_count")); err != nil {
		return nil, err
	}
	if cfg.SSMTimeStepRank, err = intKey(metaKey("ssm.time_step_rank")); err != nil {
		return nil, err
	}
	if cfg.SSMInnerSize, err = intKey(metaKey("ssm.inner_size")); err != nil {
		return nil, err
	}
	if cfg.FullAttentionInterval, err = intKey(metaKey("full_attention_interval")); err != nil {
		return nil, err
	}
	if value, ok := r.MetadataValue(metaKey("rope.dimension_sections")); ok {
		sections, err := value.AsInts()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", metaKey("rope.dimension_sections"), err)
		}
		if len(sections) != 4 {
			return nil, fmt.Errorf("%s has %d entries, want 4", metaKey("rope.dimension_sections"), len(sections))
		}
		for i, s := range sections {
			if s < 0 || s > 1<<20 {
				return nil, fmt.Errorf("invalid rope section %d", s)
			}
			cfg.RoPESections[i] = int(s)
		}
	}
	if value, ok := r.MetadataValue("tokenizer.ggml.tokens"); ok {
		tokens, err := value.AsStrings()
		if err != nil {
			return nil, fmt.Errorf("tokenizer.ggml.tokens: %w", err)
		}
		cfg.VocabSize = len(tokens)
	} else if cfg.VocabSize, err = intKey(metaKey("vocab_size")); err != nil {
		return nil, fmt.Errorf("vocabulary size is unavailable")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// validateRoPEScaling rejects scaling types the runtime does not execute. The
// default and explicit none/default values use the base RoPE table.
func validateRoPEScaling(name string, value gguf.Value) error {
	text, err := value.AsString()
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	switch strings.ToLower(strings.TrimSpace(text)) {
	case "", "none", "default":
		return nil
	default:
		return fmt.Errorf("%s %q is not supported", name, text)
	}
}

// loadExpertFFNSize reads a scalar or per-layer expert width. The runtime
// currently requires every layer to use the same width.
func loadExpertFFNSize(r *gguf.Reader, key string) (int, error) {
	value, ok := r.MetadataValue(key)
	if !ok {
		return 0, fmt.Errorf("missing required metadata %s", key)
	}
	var size int
	if value.Type == gguf.MetaArray {
		widths, err := value.AsInts()
		if err != nil {
			return 0, fmt.Errorf("%s: %w", key, err)
		}
		if len(widths) == 0 {
			return 0, fmt.Errorf("%s is empty", key)
		}
		size = int(widths[0])
		for i, width := range widths {
			if width != widths[0] {
				return 0, fmt.Errorf("%s varies at layer %d", key, i)
			}
		}
	} else {
		raw, err := value.AsInt()
		if err != nil {
			return 0, fmt.Errorf("%s: %w", key, err)
		}
		size = int(raw)
	}
	if size <= 0 || size > 1<<31 {
		return 0, fmt.Errorf("%s has invalid value %d", key, size)
	}
	return size, nil
}

func (c *Config) validate() error {
	if c.HeadCount%c.KVHeadCount != 0 {
		return fmt.Errorf("head count %d is not a multiple of KV head count %d",
			c.HeadCount, c.KVHeadCount)
	}
	if c.HeadDim != c.ValueDim {
		return fmt.Errorf("key head dim %d differs from value head dim %d",
			c.HeadDim, c.ValueDim)
	}
	if c.RoPEDim%2 != 0 || c.RoPEDim > c.HeadDim {
		return fmt.Errorf("invalid RoPE dimension %d for head dim %d", c.RoPEDim, c.HeadDim)
	}
	if sum := c.RoPESections[0] + c.RoPESections[1] + c.RoPESections[2] + c.RoPESections[3]; sum*2 != c.RoPEDim {
		return fmt.Errorf("RoPE sections sum to %d pairs, want %d", sum, c.RoPEDim/2)
	}
	if c.FullAttentionInterval <= 0 {
		return fmt.Errorf("full attention interval must be positive")
	}
	if c.SSMInnerSize != c.SSMTimeStepRank*c.SSMStateSize {
		return fmt.Errorf("ssm inner size %d != time_step_rank %d * state_size %d",
			c.SSMInnerSize, c.SSMTimeStepRank, c.SSMStateSize)
	}
	if c.SSMConvKernel < 2 {
		return fmt.Errorf("invalid ssm conv kernel %d", c.SSMConvKernel)
	}
	if c.IsMoE() {
		if c.ExpertUsedCount <= 0 || c.ExpertUsedCount > c.ExpertCount {
			return fmt.Errorf("expert used count %d is outside 1..%d",
				c.ExpertUsedCount, c.ExpertCount)
		}
		if c.ExpertFFNSize <= 0 {
			return fmt.Errorf("invalid expert feed-forward size %d", c.ExpertFFNSize)
		}
		if c.SharedFFNSize <= 0 {
			return fmt.Errorf("invalid shared expert feed-forward size %d", c.SharedFFNSize)
		}
	} else if c.FFNSize <= 0 {
		return fmt.Errorf("invalid feed-forward size %d", c.FFNSize)
	}
	if c.ContextLength <= 0 || c.VocabSize <= 0 {
		return fmt.Errorf("invalid context or vocabulary size")
	}
	return nil
}

// IsMoE reports whether the model uses routed and shared expert FFN layers.
func (c *Config) IsMoE() bool { return c.ExpertCount > 0 }

// IsRecurrent reports whether layer uses gated delta-net linear attention.
func (c *Config) IsRecurrent(layer int) bool {
	return (layer+1)%c.FullAttentionInterval != 0
}

// FullAttentionLayers returns layer indices that use full GQA attention.
func (c *Config) FullAttentionLayers() []int {
	var layers []int
	for i := 0; i < c.BlockCount; i++ {
		if !c.IsRecurrent(i) {
			layers = append(layers, i)
		}
	}
	return layers
}

// KVHeadsPerGroup returns query heads sharing one KV head.
func (c *Config) KVHeadsPerGroup() int {
	return c.HeadCount / c.KVHeadCount
}
