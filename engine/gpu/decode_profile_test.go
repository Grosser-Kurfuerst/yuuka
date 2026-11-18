//go:build hip && linux

package gpu

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/yuuka-ai/yuuka/backend/hip"
	"github.com/yuuka-ai/yuuka/tensor"
)

// The recorded 2026-11-18 decode attribution of docs/roadmap4.md section 2.3,
// measured on one gfx1151 device with Qwen3.5-4B-Q4_K_M. The gates below compare
// against it, so a change that moves the attribution has to re-record the
// document rather than pass silently.
const (
	decodeBaselineOps     = 579
	decodeBaselineTotalMS = 97.605
	decodeBaselineGemvMS  = 87.556
)

// Gate bands. Device clocks are not pinned, so the absolute total is only gated
// loosely; the operation count is structural and exact, and the GEMV share is
// the invariant the later stages of roadmap4 are built on.
const (
	decodeTotalLowRatio  = 0.5
	decodeTotalHighRatio = 2.0
	decodeShareTolerance = 0.05
)

// TestGPUProfileDecode attributes one decode step per planned operation and per
// tensor. It reproduces the tables of docs/roadmap4.md sections 2.3 and 2.4 and
// gates the result against the recorded attribution. Event instrumentation adds
// a submission gap per operation, so the absolute numbers are a diagnosis, not
// a release measurement; see docs/testing.md section 6.3.
func TestGPUProfileDecode(t *testing.T) {
	reader, _, model, prompt := goldenPrompt(t)
	rt, err := New(reader, model, 64)
	if err != nil {
		t.Skipf("HIP runtime unavailable: %v", err)
	}
	defer rt.Close()
	logits, err := rt.Forward(prompt)
	if err != nil {
		t.Fatal(err)
	}
	token := argmax(logits)
	times, err := rt.ProfileStep(token)
	if err != nil {
		t.Fatal(err)
	}
	if len(times) == 0 {
		t.Fatal("profile returned no operations")
	}
	shadow := rt.session.plan.Ops()
	if len(shadow) != len(times) {
		t.Fatalf("the decode plan holds %d operations and the profile returned %d;"+
			" the shadow and the submission disagree", len(shadow), len(times))
	}
	names := decodeBufferNames(rt.model, rt.session)
	samples := decodeSamples(shadow, times, names)
	report := newDecodeReport(samples)
	printDecodeReport(t, rt, report)
	checkDecodeGates(t, report)
}

// decodeSample is one profiled operation with the model buffer that produced it
// resolved to a name.
type decodeSample struct {
	index int
	info  hip.OpInfo
	// role is the resolved name of the buffer that names the operation: the
	// packed weights for a matmul, otherwise the first slot that is not unused.
	role string
	// tensor is the leaf name of role, with the layer prefix removed, so the
	// 32 per-layer projections of one kind aggregate into one row.
	tensor string
	dtype  tensor.DType
	// hasDType is false for the operations whose u[0] is not a dtype.
	hasDType bool
	nIn      int64
	nOut     int64
	rows     int64
	ms       float64
	// moved is the bytes the operation is expected to move; the GEMV family
	// reads its packed weights once, the gated delta-net reads and writes its
	// recurrent state five times (docs/roadmap4.md section 2.6).
	moved  uint64
	matmul bool
}

func (s decodeSample) gbps() float64 {
	if s.ms <= 0 || s.moved == 0 {
		return 0
	}
	return float64(s.moved) / (s.ms * 1e6)
}

// shapeText describes the dimensions the attribution cares about: the matrix
// for a matmul, the state geometry for the gated delta-net, and the element
// count for everything else.
func (s decodeSample) shapeText() string {
	switch {
	case s.matmul:
		text := fmt.Sprintf("%d->%d", s.nIn, s.nOut)
		if s.rows > 1 {
			text = fmt.Sprintf("%s r%d", text, s.rows)
		}
		return text
	case s.info.Code == hip.OpGDN:
		return fmt.Sprintf("%d g x %d h x %d", s.info.N[0], s.info.N[1], s.info.N[2])
	case s.info.Code == hip.OpAttentionPaged || s.info.Code == hip.OpAttentionPrefillPaged ||
		s.info.Code == hip.OpAttention || s.info.Code == hip.OpAttentionPrefill:
		return fmt.Sprintf("%d kv x %d h x %d d", s.info.N[0], s.info.N[1], s.info.N[2])
	default:
		return fmt.Sprintf("%d", s.nIn)
	}
}

// decodeSamples pairs the plan shadow with the profiled times and resolves the
// buffers each operation touches.
func decodeSamples(shadow []hip.OpInfo, times []hip.OpTime, names map[uintptr]string) []decodeSample {
	samples := make([]decodeSample, 0, len(times))
	for i, item := range times {
		info := shadow[i]
		s := decodeSample{
			index: i,
			info:  info,
			ms:    float64(item.Nanos) / 1e6,
		}
		s.tensor, s.role = resolveOperationNames(info, names)
		s.moved, s.matmul, s.rows = operationTraffic(info)
		switch info.Code {
		case hip.OpGemv, hip.OpGemvBatch, hip.OpGemm, hip.OpLMHead,
			hip.OpGemmSwiglu, hip.OpQuantQ8_1:
			s.dtype = tensor.DType(info.U[0])
			s.hasDType = true
		}
		s.nIn = info.N[0]
		s.nOut = info.N[1]
		if s.matmul && s.rows == 0 {
			s.rows = 1
		}
		samples = append(samples, s)
	}
	return samples
}

// resolveOperationNames returns the leaf tensor name and the full buffer name of
// the operation. Slots are scanned in role priority order so a matmul is named
// by its weights, a stateful operation by its state, and an elementwise
// operation by its output.
func resolveOperationNames(info hip.OpInfo, names map[uintptr]string) (tensorName, role string) {
	priority := []hip.OpRole{
		hip.RoleWeight, hip.RoleState, hip.RoleOutput, hip.RoleAccumulator,
		hip.RoleActivation, hip.RolePool, hip.RoleTable, hip.RoleControl,
		hip.RoleScratch,
	}
	for _, want := range priority {
		for slot := range info.Bufs {
			if info.Roles[slot] != want {
				continue
			}
			if name := names[info.Bufs[slot]]; name != "" {
				return leafTensorName(name), name
			}
		}
	}
	return "?", "?"
}

// leafTensorName removes the "blk.N." prefix so the per-layer copies of one
// projection aggregate together.
func leafTensorName(name string) string {
	if bracket := strings.IndexByte(name, '['); bracket >= 0 {
		return name[:bracket]
	}
	if !strings.HasPrefix(name, "blk.") {
		return name
	}
	if dot := strings.Index(name[len("blk."):], "."); dot >= 0 {
		return name[len("blk.")+dot+1:]
	}
	return name
}

// operationTraffic returns the bytes an operation is expected to move, whether
// it is a packed-matrix read, and the row count it covers.
func operationTraffic(info hip.OpInfo) (moved uint64, matmul bool, rows int64) {
	batch := info.N[2]
	if batch <= 0 {
		batch = 1
	}
	switch info.Code {
	case hip.OpGemv, hip.OpLMHead:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, 1
	case hip.OpGemvBatch:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGemm:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGemmSwiglu:
		dtype := tensor.DType(info.U[0])
		rowBytes, err := dtype.RowBytes(info.N[0])
		if err != nil {
			return 0, false, 0
		}
		return 2 * uint64(rowBytes) * uint64(info.N[1]), true, batch
	case hip.OpGDN:
		// groups, heads, state size: one 4-byte element per state cell, moved
		// by the decay, the two S^T products, and the rank-1 update.
		state := uint64(info.N[1]) * uint64(info.N[2]) * uint64(info.N[2]) * 4
		return 5 * state, false, 1
	case hip.OpEmbedding, hip.OpEmbeddingBatch:
		// One packed weight row per token, read by the gather.
		return uint64(info.N[1]) * uint64(batch), false, batch
	default:
		// The remaining operations move activation-sized buffers, which the
		// plan does not describe well enough to publish a bandwidth for.
		return 0, false, batch
	}
}

// decodeTensorRow aggregates every operation that reads the same tensor.
type decodeTensorRow struct {
	tensor string
	dtype  tensor.DType
	shape  string
	ops    int
	ms     float64
	moved  uint64
	matmul bool
}

// decodeDTypeRow aggregates the matmul family by weight dtype.
type decodeDTypeRow struct {
	dtype tensor.DType
	ops   int
	ms    float64
	moved uint64
	rows  int64
}

// decodeCodeRow aggregates by operation code, which is the table of
// docs/roadmap4.md section 2.3.
type decodeCodeRow struct {
	code uint32
	ops  int
	ms   float64
}

type decodeReport struct {
	samples []decodeSample
	codes   []decodeCodeRow
	tensors []decodeTensorRow
	dtypes  []decodeDTypeRow
	total   float64
	matmul  float64
}

func newDecodeReport(samples []decodeSample) *decodeReport {
	r := &decodeReport{samples: samples}
	codes := map[uint32]*decodeCodeRow{}
	tensors := map[string]*decodeTensorRow{}
	dtypes := map[tensor.DType]*decodeDTypeRow{}
	for _, s := range samples {
		r.total += s.ms
		if s.info.Code == hip.OpGemv || s.info.Code == hip.OpLMHead ||
			s.info.Code == hip.OpGemvBatch || s.info.Code == hip.OpGemm {
			r.matmul += s.ms
		}
		if row, ok := codes[s.info.Code]; ok {
			row.ops++
			row.ms += s.ms
		} else {
			codes[s.info.Code] = &decodeCodeRow{code: s.info.Code, ops: 1, ms: s.ms}
		}
		key := fmt.Sprintf("%s|%s|%d|%d", s.tensor, s.dtype, s.nIn, s.nOut)
		if row, ok := tensors[key]; ok {
			row.ops++
			row.ms += s.ms
			row.moved += s.moved
		} else {
			tensors[key] = &decodeTensorRow{
				tensor: s.tensor, dtype: s.dtype, shape: s.shapeText(),
				ops: 1, ms: s.ms, moved: s.moved, matmul: s.matmul,
			}
		}
		if s.matmul && s.hasDType {
			if row, ok := dtypes[s.dtype]; ok {
				row.ops++
				row.ms += s.ms
				row.moved += s.moved
			} else {
				dtypes[s.dtype] = &decodeDTypeRow{
					dtype: s.dtype, ops: 1, ms: s.ms, moved: s.moved,
				}
			}
		}
	}
	for _, row := range codes {
		r.codes = append(r.codes, *row)
	}
	sort.Slice(r.codes, func(i, j int) bool { return r.codes[i].ms > r.codes[j].ms })
	for _, row := range tensors {
		r.tensors = append(r.tensors, *row)
	}
	sort.Slice(r.tensors, func(i, j int) bool { return r.tensors[i].ms > r.tensors[j].ms })
	for _, row := range dtypes {
		r.dtypes = append(r.dtypes, *row)
	}
	sort.Slice(r.dtypes, func(i, j int) bool { return r.dtypes[i].ms > r.dtypes[j].ms })
	return r
}

// printDecodeReport writes the four attribution tables. The per-operation table
// is one line per plan index, sorted by time, so a slow operation can be named
// without a counter database.
func printDecodeReport(t *testing.T, rt *Runtime, r *decodeReport) {
	t.Helper()
	info := rt.model.device.Info()
	// HIP reports half of rocminfo's compute-unit count on this APU, so the
	// figure is labelled with its source instead of being compared.
	t.Logf("device %s (%s), HIP reports %d compute units, HIP %s", info.Name,
		info.GCNArch, info.ComputeUnits, info.HIPVersion)
	t.Logf("profiled %d operations, total %.3f ms; recorded baseline %.3f ms in"+
		" %d operations", len(r.samples), r.total, decodeBaselineTotalMS,
		decodeBaselineOps)

	t.Logf("by operation code:")
	t.Logf("%-20s %5s %10s %7s", "operation", "ops", "ms", "share")
	for _, row := range r.codes {
		t.Logf("%-20s %5d %10.3f %6.1f%%", hip.OpName(row.code), row.ops, row.ms,
			100*row.ms/r.total)
	}

	t.Logf("by tensor (matmul weights read once per token, gated delta-net state five times):")
	t.Logf("%-26s %-6s %-14s %5s %9s %8s %8s %7s", "tensor", "dtype", "shape",
		"ops", "ms", "MB", "GB/s", "share")
	var matmulMS, stateMS float64
	var matmulMoved, stateMoved uint64
	for _, row := range r.tensors {
		shape := row.shape
		dtype := "-"
		if row.matmul {
			matmulMS += row.ms
			matmulMoved += row.moved
			dtype = row.dtype.String()
		}
		if !row.matmul && row.moved != 0 {
			stateMS += row.ms
			stateMoved += row.moved
		}
		gbps := 0.0
		if row.ms > 0 && row.moved != 0 {
			gbps = float64(row.moved) / (row.ms * 1e6)
		}
		t.Logf("%-26s %-6s %-14s %5d %9.3f %8.1f %8.1f %6.1f%%", row.tensor,
			dtype, shape, row.ops, row.ms, float64(row.moved)/1e6, gbps,
			100*row.ms/r.total)
	}
	t.Logf("%-26s %-6s %-14s %5d %9.3f %8.1f %8.1f %6.1f%%", "profiled total", "", "",
		len(r.samples), r.total, float64(matmulMoved+stateMoved)/1e6,
		float64(matmulMoved+stateMoved)/(r.total*1e6), 100.0)
	t.Logf("matmul family: %.3f ms of %.3f ms (%.1f%%), %.1f MB of packed weights,"+
		" %.1f GB/s effective", matmulMS, r.total, 100*matmulMS/r.total,
		float64(matmulMoved)/1e6, float64(matmulMoved)/(matmulMS*1e6))
	if stateMS > 0 {
		t.Logf("gated delta-net state: %.3f ms of %.3f ms (%.1f%%), %.1f MB moved,"+
			" %.1f GB/s effective", stateMS, r.total, 100*stateMS/r.total,
			float64(stateMoved)/1e6, float64(stateMoved)/(stateMS*1e6))
	}

	t.Logf("by weight dtype (matmul family only):")
	t.Logf("%-8s %5s %10s %8s %10s", "dtype", "ops", "ms", "MB", "GB/s")
	for _, row := range r.dtypes {
		gbps := 0.0
		if row.ms > 0 {
			gbps = float64(row.moved) / (row.ms * 1e6)
		}
		t.Logf("%-8s %5d %10.3f %8.1f %10.1f", row.dtype, row.ops, row.ms,
			float64(row.moved)/1e6, gbps)
	}

	sorted := make([]decodeSample, len(r.samples))
	copy(sorted, r.samples)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ms > sorted[j].ms })
	t.Logf("per operation index, sorted by time:")
	t.Logf("%-5s %-16s %-30s %-16s %9s %8s", "op", "kind", "buffer", "shape", "ms", "GB/s")
	for _, sample := range sorted {
		gbpsText := "-"
		if gbps := sample.gbps(); gbps > 0 {
			gbpsText = fmt.Sprintf("%.1f", gbps)
		}
		t.Logf("%-5d %-16s %-30s %-16s %9.4f %8s", sample.index, sample.info.Name(),
			sample.role, sample.shapeText(), sample.ms, gbpsText)
	}
}

// checkDecodeGates compares the measured attribution against the recorded one.
func checkDecodeGates(t *testing.T, r *decodeReport) {
	t.Helper()
	if got := len(r.samples); got != decodeBaselineOps {
		t.Fatalf("the decode step runs %d operations, the recorded attribution is %d;"+
			" update docs/roadmap4.md section 2.3 before changing the plan shape", got,
			decodeBaselineOps)
	}
	if r.total < decodeTotalLowRatio*decodeBaselineTotalMS ||
		r.total > decodeTotalHighRatio*decodeBaselineTotalMS {
		t.Fatalf("profiled decode step is %.3f ms, which is outside the recorded"+
			" %.1fx..%.1fx band of %.3f ms; re-record the attribution in"+
			" docs/roadmap4.md section 2.3 for this device", r.total,
			decodeTotalLowRatio, decodeTotalHighRatio, decodeBaselineTotalMS)
	}
	share := r.matmul / r.total
	baselineShare := decodeBaselineGemvMS / decodeBaselineTotalMS
	if share < baselineShare-decodeShareTolerance || share > baselineShare+decodeShareTolerance {
		t.Fatalf("the GEMV family is %.1f%% of the decode step, the recorded attribution"+
			" is %.1f%%; a change moved the bottleneck, so update docs/roadmap4.md"+
			" section 2.3 and the stage gates", 100*share, 100*baselineShare)
	}
	t.Logf("gate: %d operations (recorded %d), total %.3f ms (recorded %.3f ms),"+
		" GEMV share %.1f%% (recorded %.1f%%)", len(r.samples), decodeBaselineOps,
		r.total, decodeBaselineTotalMS, 100*share, 100*baselineShare)
}

// decodeBufferNames labels every buffer the decode plan can reference so a
// planned operation can be named. Weights keep their GGUF tensor names; the
// activation, pool, and session buffers get a stable role name.
func decodeBufferNames(m *Model, s *Session) map[uintptr]string {
	names := make(map[uintptr]string, len(m.weights)+64)
	add := func(b *hip.Buffer, name string) {
		if b == nil {
			return
		}
		if id := b.ID(); id != 0 {
			names[id] = name
		}
	}
	for name, w := range m.weights {
		add(w.buf, name)
	}
	act := m.act
	for _, item := range []struct {
		buf  *hip.Buffer
		name string
	}{
		{act.x, "act.x"},
		{act.norm, "act.norm"},
		{act.q, "act.q"},
		{act.k, "act.k"},
		{act.v, "act.v"},
		{act.attn, "act.attn"},
		{act.tmp, "act.tmp"},
		{act.qkv, "act.qkv"},
		{act.z, "act.z"},
		{act.beta, "act.beta"},
		{act.alpha, "act.alpha"},
		{act.conv, "act.conv"},
		{act.gdn, "act.gdn"},
		{act.gdnNorm, "act.gdn_norm"},
		{act.ffnGate, "act.ffn_gate"},
		{act.ffnUp, "act.ffn_up"},
		{act.logits, "act.logits"},
		{act.scores, "act.scores"},
	} {
		add(item.buf, item.name)
	}
	for i, buf := range m.poolK {
		add(buf, fmt.Sprintf("kv.pool_k[%d]", i))
	}
	for i, buf := range m.poolV {
		add(buf, fmt.Sprintf("kv.pool_v[%d]", i))
	}
	add(s.tableBuf, "session.block_table")
	add(s.dyn, "session.dynamic")
	for i, buf := range s.conv {
		add(buf, fmt.Sprintf("session.conv[blk.%d]", i))
	}
	for i, buf := range s.recState {
		add(buf, fmt.Sprintf("session.rec_state[blk.%d]", i))
	}
	for i, buf := range s.traceBufs {
		add(buf, fmt.Sprintf("session.trace[blk.%d]", i))
	}
	return names
}
