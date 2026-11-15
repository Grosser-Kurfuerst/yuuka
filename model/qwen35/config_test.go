package qwen35

import "testing"

func validTestConfig() Config {
	return Config{
		Architecture:          Architecture,
		BlockCount:            2,
		ContextLength:         256,
		HiddenSize:            32,
		FFNSize:               64,
		HeadCount:             2,
		KVHeadCount:           1,
		HeadDim:               8,
		ValueDim:              8,
		RoPEDim:               8,
		RoPESections:          [4]int{1, 1, 1, 1},
		RoPEFreqBase:          10000,
		RMSNormEps:            1e-6,
		VocabSize:             259,
		SSMConvKernel:         2,
		SSMStateSize:          8,
		SSMGroupCount:         1,
		SSMTimeStepRank:       2,
		SSMInnerSize:          16,
		FullAttentionInterval: 2,
	}
}

func TestConfigValidate(t *testing.T) {
	cfg := validTestConfig()
	if err := cfg.validate(); err != nil {
		t.Fatalf("valid config: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"head count not multiple", func(c *Config) { c.HeadCount = 3; c.KVHeadCount = 2 }},
		{"key and value dims differ", func(c *Config) { c.ValueDim = 4 }},
		{"odd rope dimension", func(c *Config) { c.RoPEDim = 7 }},
		{"rope dimension exceeds head", func(c *Config) { c.RoPEDim = 10 }},
		{"rope sections mismatch", func(c *Config) { c.RoPESections = [4]int{1, 1, 1, 0} }},
		{"zero attention interval", func(c *Config) { c.FullAttentionInterval = 0 }},
		{"ssm inner mismatch", func(c *Config) { c.SSMInnerSize = 8 }},
		{"short conv kernel", func(c *Config) { c.SSMConvKernel = 1 }},
		{"zero feed-forward", func(c *Config) { c.FFNSize = 0 }},
		{"zero context", func(c *Config) { c.ContextLength = 0 }},
		{"zero vocabulary", func(c *Config) { c.VocabSize = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validTestConfig()
			tc.mutate(&cfg)
			if err := cfg.validate(); err == nil {
				t.Fatal("invalid config was accepted")
			}
		})
	}
}

func TestMoEConfigValidate(t *testing.T) {
	cfg := validTestConfig()
	cfg.Architecture = MoEArchitecture
	cfg.FFNSize = 0
	cfg.ExpertCount = 4
	cfg.ExpertUsedCount = 2
	cfg.ExpertFFNSize = 16
	cfg.SharedFFNSize = 16
	if err := cfg.validate(); err != nil {
		t.Fatalf("valid MoE config: %v", err)
	}
	if !cfg.IsMoE() {
		t.Fatal("MoE config was not detected")
	}
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"zero experts", func(c *Config) { c.ExpertCount = 0 }},
		{"used exceeds experts", func(c *Config) { c.ExpertUsedCount = 5 }},
		{"zero used", func(c *Config) { c.ExpertUsedCount = 0 }},
		{"zero expert width", func(c *Config) { c.ExpertFFNSize = 0 }},
		{"zero shared width", func(c *Config) { c.SharedFFNSize = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := validTestConfig()
			cfg.Architecture = MoEArchitecture
			cfg.FFNSize = 0
			cfg.ExpertCount = 4
			cfg.ExpertUsedCount = 2
			cfg.ExpertFFNSize = 16
			cfg.SharedFFNSize = 16
			tc.mutate(&cfg)
			if err := cfg.validate(); err == nil {
				t.Fatal("invalid MoE config was accepted")
			}
		})
	}
}
