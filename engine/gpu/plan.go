//go:build hip && linux

package gpu

import (
	"fmt"
	"math"
	"strings"

	"github.com/yuuka-ai/yuuka/backend/hip"
)

// prefillMatmul appends the measured implementation for one prefill matmul.
func (m *Model) prefillMatmul(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	return m.prefillMatmulPath(plan, w, x, xOff, out, outOff, nIn, nOut, rows,
		hip.SelectPrefillPath(w.dtype, rows, nIn, nOut))
}

func (m *Model) prefillMatmulPath(plan *hip.Plan, w weight, x *hip.Buffer, xOff uint64,
	out *hip.Buffer, outOff uint64, nIn, nOut, rows int64, path hip.PrefillPath) error {
	switch path {
	case hip.PrefillBatchedGemv:
		return plan.AddGemvBatchAt(w.buf, 0, x, xOff, out, outOff, nIn, nOut, rows, w.dtype)
	case hip.PrefillBLAS:
		if err := plan.AddCastF16At(x, xOff, m.prefill.xF16, 0, rows*nIn); err != nil {
			return err
		}
		return plan.AddGemmBlasAt(w.buf, 0, m.prefill.xF16, m.prefill.yF16, out,
			outOff, nIn, nOut, rows)
	case hip.PrefillMMQ:
		return m.prefillMMQPrepared(plan, w, x, xOff, out, outOff, nIn, nOut, rows)
	default:
		return plan.AddGemmPathAt(w.buf, 0, x, xOff, out, outOff,
			nIn, nOut, rows, w.dtype, hip.GemmWMMA)
	}
}

// prefillProjection is one prefill matmul that reads the shared activation
// tile at offset zero.
type prefillProjection struct {
	w    weight
	out  *hip.Buffer
	nOut int64
}

// prefillProjections runs several projections that read the same activation
// tile. When two or more of them select the MMQ route, the Q8_1 quantization of
// that tile runs once for the group instead of once per projection; the q/k/v
// and qkv/gate groups are the measured case. Projections on other routes keep
// their own implementation and their own input handling.
func (m *Model) prefillProjections(plan *hip.Plan, x *hip.Buffer, nIn, rows int64,
	projections []prefillProjection) error {
	mmq := 0
	for _, projection := range projections {
		if hip.SelectPrefillPath(projection.w.dtype, rows, nIn, projection.nOut) == hip.PrefillMMQ {
			mmq++
		}
	}
	if mmq > 1 {
		if err := plan.AddQuantizeQ8_1At(x, 0, m.prefill.xQ8, m.prefill.xDS, nIn, rows); err != nil {
			return err
		}
	}
	for _, projection := range projections {
		if mmq > 1 && hip.SelectPrefillPath(projection.w.dtype, rows, nIn, projection.nOut) == hip.PrefillMMQ {
			if err := plan.AddGemmMmqAt(projection.w.buf, 0, m.prefill.xQ8, m.prefill.xDS,
				projection.out, 0, nIn, projection.nOut, rows, projection.w.dtype); err != nil {
				return err
			}
			continue
		}
		if err := m.prefillMatmul(plan, projection.w, x, 0, projection.out, 0, nIn,
			projection.nOut, rows); err != nil {
			return err
		}
	}
	return nil
}

// prefillMMQPrepared quantizes the activation tile to Q8_1 and runs the int8
// GEMM. The quantization is a separate plan operation so a gate/up pair that
// reads the same normalized tile can share it.
func (m *Model) prefillMMQPrepared(plan *hip.Plan, w weight, x *hip.Buffer,
	xOff uint64, out *hip.Buffer, outOff uint64, nIn, nOut, rows int64) error {
	if err := plan.AddQuantizeQ8_1At(x, xOff, m.prefill.xQ8, m.prefill.xDS, nIn, rows); err != nil {
		return err
	}
	return plan.AddGemmMmqAt(w.buf, 0, m.prefill.xQ8, m.prefill.xDS, out, outOff,
		nIn, nOut, rows, w.dtype)
}

// buildDecodePlan constructs the single-token graph over one session's block
// table and recurrent state.
func (m *Model) buildDecodePlan(s *Session, trace bool) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	embedding, err := m.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbedding(embedding.buf, m.act.x, int64(cfg.HiddenSize), int64(rowBytes), embedding.dtype); err != nil {
		return nil, err
	}
	for i := range cfg.BlockCount {
		prefix := fmt.Sprintf("blk.%d.", i)
		attnNorm, err := m.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := m.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			if err := m.addRecurrent(plan, s, i, attnNorm); err != nil {
				return nil, err
			}
		} else {
			if err := m.addFullAttention(plan, s, i, attnNorm); err != nil {
				return nil, err
			}
		}
		if err := m.addFFN(plan, prefix, postNorm); err != nil {
			return nil, err
		}
		if trace {
			if err := plan.AddCopy(s.traceBufs[i], m.act.x, int64(cfg.HiddenSize)); err != nil {
				return nil, err
			}
		}
	}
	plan.MarkTail()
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	if err := plan.AddRMSNorm(m.act.x, outputNorm.buf, m.act.norm, int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := m.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := m.weight(outputName)
	if err != nil {
		return nil, err
	}
	if err := plan.AddGemv(output.buf, m.act.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (m *Model) addFullAttention(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight) error {
	cfg := m.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := m.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := m.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := m.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := m.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := m.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := m.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNorm(m.act.x, attnNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(q.buf, m.act.norm, m.act.q, hidden, qDim, q.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(k.buf, m.act.norm, m.act.k, hidden, kvDim, k.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(v.buf, m.act.norm, m.act.v, hidden, kvDim, v.dtype); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(m.act.q, 0, qNorm.buf, m.act.q, 0,
		cfg.HeadCount, headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsAt(m.act.k, 0, kNorm.buf, m.act.k, 0,
		cfg.KVHeadCount, headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPE(m.act.q, cfg.HeadCount, int(2*headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddRoPE(m.act.k, cfg.KVHeadCount, int(headDim), int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase)); err != nil {
		return err
	}
	if err := plan.AddKVWritePagedFormatAt(m.act.k, m.act.v, m.poolK[layerIndex], m.poolV[layerIndex],
		s.tableBuf, kvDim, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	if err := plan.AddAttentionPagedFormatAt(m.act.q, m.poolK[layerIndex], m.poolV[layerIndex],
		m.act.attn, m.act.scores, s.tableBuf,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim),
		m.capacity, s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := plan.AddGemv(o.buf, m.act.attn, m.act.tmp, oDim, hidden, o.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

func (m *Model) addRecurrent(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight) error {
	cfg := m.config
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := m.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := m.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := m.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := m.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := m.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := m.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := m.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := m.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	keyDim := int64(cfg.SSMGroupCount * cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNorm(m.act.x, attnNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(qkv.buf, m.act.norm, m.act.qkv, hidden, convDim, qkv.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, m.act.norm, m.act.z, hidden, inner, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(beta.buf, m.act.norm, m.act.beta, hidden, rank, beta.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(alpha.buf, m.act.norm, m.act.alpha, hidden, rank, alpha.dtype); err != nil {
		return err
	}
	if err := plan.AddConv1D(s.conv[layerIndex], conv.buf, m.act.qkv, m.act.conv, int(convDim), cfg.SSMConvKernel); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	if err := plan.AddL2NormAt(m.act.conv, 0, m.act.conv, 0,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, qScale); err != nil {
		return err
	}
	if err := plan.AddL2NormAt(m.act.conv, uint64(keyDim)*4, m.act.conv, uint64(keyDim)*4,
		int64(cfg.SSMGroupCount), stateSize, cfg.RMSNormEps, 1); err != nil {
		return err
	}
	if err := plan.AddGDNAt(s.recState[layerIndex],
		m.act.conv, 0,
		m.act.conv, uint64(keyDim)*4,
		m.act.conv, uint64(2*keyDim)*4,
		m.act.alpha, m.act.beta, dt.buf, a.buf, m.act.gdn,
		cfg.SSMGroupCount, cfg.SSMTimeStepRank, int(stateSize)); err != nil {
		return err
	}
	if err := plan.AddGatedRMSNorm(m.act.gdn, norm.buf, m.act.z, m.act.gdnNorm,
		rank, stateSize, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(out.buf, m.act.gdnNorm, m.act.tmp, inner, hidden, out.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

func (m *Model) addFFN(plan *hip.Plan, prefix string, postNorm weight) error {
	cfg := m.config
	gate, err := m.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNorm(m.act.x, postNorm.buf, m.act.norm, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddGemv(gate.buf, m.act.norm, m.act.ffnGate, hidden, ffn, gate.dtype); err != nil {
		return err
	}
	if err := plan.AddGemv(up.buf, m.act.norm, m.act.ffnUp, hidden, ffn, up.dtype); err != nil {
		return err
	}
	if err := plan.AddSiluMul(m.act.ffnGate, m.act.ffnUp, m.act.ffnGate, ffn); err != nil {
		return err
	}
	if err := plan.AddGemv(down.buf, m.act.ffnGate, m.act.tmp, ffn, hidden, down.dtype); err != nil {
		return err
	}
	return plan.Add(m.act.x, m.act.tmp, hidden)
}

type prefillOutput uint8

const (
	prefillLastLogit prefillOutput = iota
	prefillAllLogits
	prefillLastLogitFullFFN // Full-row reference for correctness comparisons.
)

// buildPrefillPlan constructs the chunked prompt graph for one session.
func (m *Model) buildPrefillPlan(s *Session, rows int, outputMode prefillOutput) (*hip.Plan, error) {
	plan, err := m.device.NewPlan()
	if err != nil {
		return nil, err
	}
	cfg := m.config
	p := &m.prefill
	rows64 := int64(rows)
	lastOff := uint64(rows-1) * uint64(cfg.HiddenSize) * 4
	crop := outputMode == prefillLastLogit && cfg.BlockCount > 0 && !cfg.IsRecurrent(cfg.BlockCount-1)
	embedding, err := m.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	rowBytes, err := embedding.dtype.RowBytes(int64(cfg.HiddenSize))
	if err != nil {
		return nil, err
	}
	if err := plan.AddEmbeddingBatchAt(embedding.buf, p.x, p.tokens, 0, rows64,
		int64(cfg.HiddenSize), int64(rowBytes), int64(cfg.VocabSize), embedding.dtype); err != nil {
		return nil, err
	}
	for i := 0; i < cfg.BlockCount; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		attnNorm, err := m.weight(prefix + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		postNorm, err := m.weight(prefix + "post_attention_norm.weight")
		if err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(i) {
			if err := m.addPrefillRecurrent(plan, s, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		} else {
			if err := m.addPrefillFullAttention(plan, s, i, attnNorm, rows64); err != nil {
				return nil, err
			}
		}
		ffnRows := rows64
		if crop && i == cfg.BlockCount-1 {
			// Full attention has already written all K/V rows. The final
			// FFN's only consumer is the requested last prompt logit.
			plan.MarkTail()
			if rows > 1 {
				if err := plan.AddCopyAt(p.x, 0, p.x, lastOff, int64(cfg.HiddenSize)); err != nil {
					return nil, err
				}
			}
			ffnRows, lastOff = 1, 0
		}
		// Preserve the parent chunk's projection arithmetic for the compact row.
		if err := m.addPrefillFFN(plan, prefix, postNorm, ffnRows, rows64); err != nil {
			return nil, err
		}
	}
	if !crop {
		plan.MarkTail()
	}
	outputNorm, err := m.weight("output_norm.weight")
	if err != nil {
		return nil, err
	}
	outputName := "output.weight"
	if _, ok := m.weights[outputName]; !ok {
		outputName = "token_embd.weight"
	}
	output, err := m.weight(outputName)
	if err != nil {
		return nil, err
	}
	if outputMode == prefillAllLogits {
		if rows > maxSpecDraft {
			return nil, fmt.Errorf("gpu: speculative logits rows %d exceed %d", rows, maxSpecDraft)
		}
		if err := plan.AddRMSNormBatchAt(p.x, 0, outputNorm.buf, p.norm, 0,
			rows64, int64(cfg.HiddenSize), int64(cfg.HiddenSize), int64(cfg.HiddenSize),
			cfg.RMSNormEps); err != nil {
			return nil, err
		}
		if hip.SelectLMHeadPath(output.dtype, rows64, int64(cfg.HiddenSize), int64(cfg.VocabSize)) {
			// One streaming pass over the output matrix for every candidate
			// row instead of one GEMV per row.
			if err := plan.AddLMHeadAt(output.buf, 0, p.norm, 0, m.specLogits, 0,
				int64(cfg.HiddenSize), int64(cfg.VocabSize), rows64, output.dtype); err != nil {
				return nil, err
			}
			return plan, nil
		}
		for row := range rows {
			if err := plan.AddGemvAt(output.buf, 0, p.norm,
				uint64(row)*uint64(cfg.HiddenSize)*4, m.specLogits,
				uint64(row)*uint64(cfg.VocabSize)*4, int64(cfg.HiddenSize),
				int64(cfg.VocabSize), output.dtype); err != nil {
				return nil, err
			}
		}
		return plan, nil
	}
	if err := plan.AddRMSNormAt(p.x, lastOff, outputNorm.buf, p.norm, 0,
		int64(cfg.HiddenSize), cfg.RMSNormEps); err != nil {
		return nil, err
	}
	if hip.SelectLMHeadPath(output.dtype, 1, int64(cfg.HiddenSize), int64(cfg.VocabSize)) {
		// The chunk tail needs the last row only. The projection reads the
		// output matrix once at streaming bandwidth instead of running the
		// decode GEMV, which is latency bound at a single row.
		if err := plan.AddLMHeadAt(output.buf, 0, p.norm, 0, m.act.logits, 0,
			int64(cfg.HiddenSize), int64(cfg.VocabSize), 1, output.dtype); err != nil {
			return nil, err
		}
		return plan, nil
	}
	if err := plan.AddGemv(output.buf, p.norm, m.act.logits,
		int64(cfg.HiddenSize), int64(cfg.VocabSize), output.dtype); err != nil {
		return nil, err
	}
	return plan, nil
}

func (m *Model) addPrefillFullAttention(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight, rows int64) error {
	cfg := m.config
	p := &m.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	q, err := m.weight(prefix + "attn_q.weight")
	if err != nil {
		return err
	}
	k, err := m.weight(prefix + "attn_k.weight")
	if err != nil {
		return err
	}
	v, err := m.weight(prefix + "attn_v.weight")
	if err != nil {
		return err
	}
	o, err := m.weight(prefix + "attn_output.weight")
	if err != nil {
		return err
	}
	qNorm, err := m.weight(prefix + "attn_q_norm.weight")
	if err != nil {
		return err
	}
	kNorm, err := m.weight(prefix + "attn_k_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	headDim := int64(cfg.HeadDim)
	qDim := int64(cfg.HeadCount * cfg.HeadDim * 2)
	kvDim := int64(cfg.KVHeadCount * cfg.HeadDim)
	oDim := int64(cfg.HeadCount * cfg.HeadDim)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.prefillProjections(plan, p.norm, hidden, rows, []prefillProjection{
		{w: q, out: p.q, nOut: qDim},
		{w: k, out: p.k, nOut: kvDim},
		{w: v, out: p.v, nOut: kvDim},
	}); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.q, 0, qNorm.buf, p.q, 0,
		rows, int64(cfg.HeadCount), headDim, 2*headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRMSNormHeadsBatchAt(p.k, 0, kNorm.buf, p.k, 0,
		rows, int64(cfg.KVHeadCount), headDim, headDim, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(p.q, 0, int(rows), cfg.HeadCount, int(2*headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.HeadCount*2*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddRoPEBatchAt(p.k, 0, int(rows), cfg.KVHeadCount, int(headDim),
		int(headDim), cfg.RoPEDim, float32(cfg.RoPEFreqBase), int64(cfg.KVHeadCount*cfg.HeadDim)); err != nil {
		return err
	}
	if err := plan.AddKVWriteBatchPagedFormatAt(p.k, 0, p.v, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], s.tableBuf, kvDim, rows, int64(s.capacity), m.blockSize, m.kvType); err != nil {
		return err
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	// The rocBLAS path only runs where the FP16 MMA kernel does not cover the
	// shape, so F16 KV never selects the slower kernel by default.
	useBlas := m.attentionBlasEnabled && !m.forceTiledAttention &&
		!hip.PrefillAttentionMMA(cfg.HeadCount, cfg.KVHeadCount, int(headDim),
			int(rows), s.position, m.kvType)
	if m.forceTiledAttention {
		if err := plan.AddAttentionPrefillPagedTiledFormatAt(p.q, 0, m.poolK[layerIndex],
			m.poolV[layerIndex], p.attn, 0, s.tableBuf,
			cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(rows),
			s.capacity, m.blockSize, scale, m.kvType); err != nil {
			return err
		}
	} else if useBlas {
		if err := plan.AddAttentionPrefillPagedBlasFormatAt(p.q, 0, m.poolK[layerIndex],
			m.poolV[layerIndex], p.attn, 0, s.tableBuf, p.attentionBlas,
			cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(rows),
			s.capacity, m.blockSize, scale); err != nil {
			return err
		}
	} else if err := plan.AddAttentionPrefillPagedFormatAt(p.q, 0, m.poolK[layerIndex],
		m.poolV[layerIndex], p.attn, 0, s.tableBuf,
		cfg.KVHeadCount, cfg.HeadCount, int(headDim), int(kvDim), int(rows),
		s.capacity, m.blockSize, scale, m.kvType); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, o, p.attn, 0, p.tmp, 0, oDim, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

func (m *Model) addPrefillRecurrent(plan *hip.Plan, s *Session, layerIndex int, attnNorm weight, rows int64) error {
	cfg := m.config
	p := &m.prefill
	prefix := fmt.Sprintf("blk.%d.", layerIndex)
	qkv, err := m.weight(prefix + "attn_qkv.weight")
	if err != nil {
		return err
	}
	gate, err := m.weight(prefix + "attn_gate.weight")
	if err != nil {
		return err
	}
	out, err := m.weight(prefix + "ssm_out.weight")
	if err != nil {
		return err
	}
	beta, err := m.weight(prefix + "ssm_beta.weight")
	if err != nil {
		return err
	}
	alpha, err := m.weight(prefix + "ssm_alpha.weight")
	if err != nil {
		return err
	}
	conv, err := m.weight(prefix + "ssm_conv1d.weight")
	if err != nil {
		return err
	}
	dt, err := m.weight(prefix + "ssm_dt.bias")
	if err != nil {
		return err
	}
	a, err := m.weight(prefix + "ssm_a")
	if err != nil {
		return err
	}
	norm, err := m.weight(prefix + "ssm_norm.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	inner := int64(cfg.SSMInnerSize)
	rank := int64(cfg.SSMTimeStepRank)
	stateSize := int64(cfg.SSMStateSize)
	convDim := int64(2*cfg.SSMGroupCount*cfg.SSMStateSize + cfg.SSMInnerSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, attnNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.prefillProjections(plan, p.norm, hidden, rows, []prefillProjection{
		{w: qkv, out: p.qkv, nOut: convDim},
		{w: gate, out: p.z, nOut: inner},
		{w: beta, out: p.beta, nOut: rank},
		{w: alpha, out: p.alpha, nOut: rank},
	}); err != nil {
		return err
	}
	qScale := float32(1.0 / math.Sqrt(float64(stateSize)))
	var prepared *hip.Buffer
	arch, _, _ := strings.Cut(m.device.Info().GCNArch, ":")
	preparedWidth := uint64(convDim - inner + 2*rank)
	if arch == "gfx1151" && stateSize == 128 && rows >= 128 &&
		preparedWidth <= p.ffnGate.Bytes()/4/uint64(rows) {
		// The FFN writes this buffer after the recurrent op on the same stream.
		prepared = p.ffnGate
	}
	if err := plan.AddRecurrentChunkAt(hip.RecurrentChunkArgs{
		ConvState:   s.conv[layerIndex],
		State:       s.recState[layerIndex],
		QKV:         p.qkv,
		Z:           p.z,
		Alpha:       p.alpha,
		Beta:        p.beta,
		ConvWeights: conv.buf,
		DT:          dt.buf,
		A:           a.buf,
		Norm:        norm.buf,
		Conv:        p.conv,
		Out:         p.gdnNorm,
		Prepared:    prepared,
		Rows:        rows,
		Groups:      int64(cfg.SSMGroupCount),
		Heads:       rank,
		StateSize:   stateSize,
		ConvDim:     convDim,
		Inner:       inner,
		Kernel:      int64(cfg.SSMConvKernel),
		Eps:         cfg.RMSNormEps,
		QScale:      qScale,
	}); err != nil {
		return err
	}
	if err := m.prefillMatmul(plan, out, p.gdnNorm, 0, p.tmp, 0, inner, hidden, rows); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}

func (m *Model) addPrefillFFN(plan *hip.Plan, prefix string, postNorm weight, rows, dispatchRows int64) error {
	cfg := m.config
	p := &m.prefill
	gate, err := m.weight(prefix + "ffn_gate.weight")
	if err != nil {
		return err
	}
	up, err := m.weight(prefix + "ffn_up.weight")
	if err != nil {
		return err
	}
	down, err := m.weight(prefix + "ffn_down.weight")
	if err != nil {
		return err
	}
	hidden := int64(cfg.HiddenSize)
	ffn := int64(cfg.FFNSize)
	if err := plan.AddRMSNormBatchAt(p.x, 0, postNorm.buf, p.norm, 0,
		rows, hidden, hidden, hidden, cfg.RMSNormEps); err != nil {
		return err
	}
	// Fuse the gate/up pair when both matmuls select the custom WMMA path.
	// The fused kernel shares one activation tile and applies SwiGLU in the
	// epilogue, so neither intermediate is written.
	gatePath := hip.SelectPrefillPath(gate.dtype, dispatchRows, hidden, ffn)
	upPath := hip.SelectPrefillPath(up.dtype, dispatchRows, hidden, ffn)
	fused := gate.dtype == up.dtype && gatePath == hip.PrefillWMMA
	if fused {
		if err := plan.AddGemmSwigluAt(gate.buf, 0, up.buf, 0, p.norm, 0,
			p.ffnGate, 0, hidden, ffn, rows, gate.dtype); err != nil {
			return err
		}
	} else if gatePath == hip.PrefillMMQ && upPath == hip.PrefillMMQ {
		// The two projections read exactly the same normalized activation, so
		// the Q8_1 quantization runs once for the pair.
		if err := plan.AddQuantizeQ8_1At(p.norm, 0, p.xQ8, p.xDS, hidden, rows); err != nil {
			return err
		}
		if err := plan.AddGemmMmqAt(gate.buf, 0, p.xQ8, p.xDS, p.ffnGate, 0, hidden, ffn, rows, gate.dtype); err != nil {
			return err
		}
		if err := plan.AddGemmMmqAt(up.buf, 0, p.xQ8, p.xDS, p.ffnUp, 0, hidden, ffn, rows, up.dtype); err != nil {
			return err
		}
	} else {
		if err := m.prefillMatmulPath(plan, gate, p.norm, 0, p.ffnGate, 0, hidden, ffn, rows, gatePath); err != nil {
			return err
		}
		if err := m.prefillMatmulPath(plan, up, p.norm, 0, p.ffnUp, 0, hidden, ffn, rows, upPath); err != nil {
			return err
		}
	}
	if !fused {
		if err := plan.AddSiluMul(p.ffnGate, p.ffnUp, p.ffnGate, rows*ffn); err != nil {
			return err
		}
	}
	downPath := hip.SelectPrefillPath(down.dtype, dispatchRows, ffn, hidden)
	if err := m.prefillMatmulPath(plan, down, p.ffnGate, 0, p.tmp, 0, ffn, hidden, rows, downPath); err != nil {
		return err
	}
	return plan.Add(p.x, p.tmp, rows*hidden)
}
