package qwen35

import (
	"math"
	"math/rand"
	"testing"
)

// TestGDNChunkedMatchesPerToken validates the chunkwise gated delta rule
// against the per-token recurrence on random inputs. The two forms sum the state
// update in a different order, so the tolerance is relative to the magnitude of
// the state entries rather than exact equality.
func TestGDNChunkedMatchesPerToken(t *testing.T) {
	const (
		stateSize = 128
		groups    = 4
		rank      = 8
		convDim   = 2*groups*stateSize + rank*stateSize
	)
	rng := rand.New(rand.NewSource(20261118))
	for _, rows := range []int{1, 5, 32, 33, 64, 100} {
		state := make([]float32, rank*stateSize*stateSize)
		conv := make([]float32, rows*convDim)
		alpha := make([]float32, rows*rank)
		beta := make([]float32, rows*rank)
		for i := range state {
			state[i] = float32(rng.NormFloat64() * 0.5)
		}
		keyDim := groups * stateSize
		for r := 0; r < rows; r++ {
			row := conv[r*convDim : (r+1)*convDim]
			// Unit-norm q and k per group, raw v: the scans see the same input.
			for g := 0; g < groups; g++ {
				q := row[g*stateSize : (g+1)*stateSize]
				k := row[keyDim+g*stateSize : keyDim+(g+1)*stateSize]
				fillUnit(rng, q)
				fillUnit(rng, k)
			}
			v := row[2*keyDim:]
			for i := range v {
				v[i] = float32(rng.NormFloat64())
			}
			for h := 0; h < rank; h++ {
				// Log decay in [-0.2, 0] keeps the state bounded.
				alpha[r*rank+h] = float32(-0.2 * rng.Float64())
				beta[r*rank+h] = float32(0.9*rng.Float64() + 0.05)
			}
		}
		refState := make([]float32, len(state))
		copy(refState, state)
		chunkState := make([]float32, len(state))
		copy(chunkState, state)
		refOut := make([]float32, rows*rank*stateSize)
		chunkOut := make([]float32, rows*rank*stateSize)
		sk := make([]float32, stateSize)
		d := make([]float32, stateSize)
		gdnPerTokenScan(refState, sk, d, conv, alpha, beta, refOut, rows, groups,
			rank, stateSize, convDim)
		gdnChunkedScan(chunkState, conv, alpha, beta, chunkOut, rows, groups, rank,
			stateSize, convDim, newGDNScratch(stateSize))

		maxOut := relDiff(refOut, chunkOut)
		maxState := relDiff(refState, chunkState)
		if maxOut > 1e-4 || maxState > 1e-4 {
			t.Fatalf("rows=%d: output difference %.3e, state difference %.3e, want <= 1e-4",
				rows, maxOut, maxState)
		}
		t.Logf("rows=%3d: output %.3e, state %.3e", rows, maxOut, maxState)
	}
}

// TestGDNChunkedLongSequence runs several chunks back to back so the state
// carried between chunks is covered as well.
func TestGDNChunkedLongSequence(t *testing.T) {
	const (
		stateSize = 128
		groups    = 2
		rank      = 4
		rows      = 160
	)
	convDim := 2*groups*stateSize + rank*stateSize
	keyDim := groups * stateSize
	rng := rand.New(rand.NewSource(7))
	conv := make([]float32, rows*convDim)
	alpha := make([]float32, rows*rank)
	beta := make([]float32, rows*rank)
	for r := 0; r < rows; r++ {
		row := conv[r*convDim : (r+1)*convDim]
		for g := 0; g < groups; g++ {
			fillUnit(rng, row[g*stateSize:(g+1)*stateSize])
			fillUnit(rng, row[keyDim+g*stateSize:keyDim+(g+1)*stateSize])
		}
		v := row[2*keyDim:]
		for i := range v {
			v[i] = float32(2 * rng.NormFloat64())
		}
		for h := 0; h < rank; h++ {
			alpha[r*rank+h] = float32(-0.05 * rng.Float64())
			beta[r*rank+h] = float32(rng.Float64())
		}
	}
	refState := make([]float32, rank*stateSize*stateSize)
	for i := range refState {
		refState[i] = float32(rng.NormFloat64())
	}
	chunkState := make([]float32, len(refState))
	copy(chunkState, refState)
	refOut := make([]float32, rows*rank*stateSize)
	chunkOut := make([]float32, rows*rank*stateSize)
	gdnPerTokenScan(refState, make([]float32, stateSize), make([]float32, stateSize),
		conv, alpha, beta, refOut, rows, groups, rank, stateSize, convDim)
	gdnChunkedScan(chunkState, conv, alpha, beta, chunkOut, rows, groups, rank,
		stateSize, convDim, newGDNScratch(stateSize))
	scale := float32(0)
	for _, v := range refOut {
		scale = max(scale, float32(math.Abs(float64(v))))
	}
	maxOut := relDiff(refOut, chunkOut)
	maxState := relDiff(refState, chunkState)
	t.Logf("out magnitude %.3e, output difference %.3e, state difference %.3e",
		scale, maxOut, maxState)
	if maxOut > 1e-4 || maxState > 1e-4 {
		t.Fatalf("output difference %.3e, state difference %.3e, want <= 1e-4",
			maxOut, maxState)
	}
}

// fillUnit fills x with a random unit vector.
func fillUnit(rng *rand.Rand, x []float32) {
	var norm float64
	for i := range x {
		v := rng.NormFloat64()
		x[i] = float32(v)
		norm += v * v
	}
	inv := float32(1 / math.Sqrt(norm))
	for i := range x {
		x[i] *= inv
	}
}

// relDiff returns the largest difference between a and b relative to the
// largest magnitude in a.
func relDiff(a, b []float32) float64 {
	var peak, diff float64
	for i := range a {
		peak = max(peak, math.Abs(float64(a[i])))
		diff = max(diff, math.Abs(float64(a[i]-b[i])))
	}
	if peak == 0 {
		return diff
	}
	return diff / peak
}
