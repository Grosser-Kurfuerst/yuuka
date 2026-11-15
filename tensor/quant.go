package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
)

const kQuantBlock = 256

// DequantizeRow expands one packed row into dst. dst must hold dstLen elements,
// which must equal the dtype's elements for len(src).
func DequantizeRow(dst []float32, dtype DType, src []byte) error {
	block, err := dtype.BlockLayout()
	if err != nil {
		return err
	}
	if block.Elements == 0 || len(src)%int(block.Bytes) != 0 {
		return fmt.Errorf("packed row length %d is not a multiple of %s block size %d",
			len(src), dtype, block.Bytes)
	}
	want := len(src) / int(block.Bytes) * int(block.Elements)
	if len(dst) != want {
		return fmt.Errorf("destination length %d, want %d", len(dst), want)
	}
	switch dtype {
	case F32:
		if len(src)%4 != 0 {
			return fmt.Errorf("F32 row length %d is not divisible by 4", len(src))
		}
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		}
	case F16:
		for i := range dst {
			dst[i] = Float16ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
		}
	case Q8_0:
		dequantizeQ8_0(dst, src)
	case Q4_0:
		dequantizeQ4_0(dst, src)
	case Q5_0:
		dequantizeQ5_0(dst, src)
	case Q4_K:
		dequantizeQ4K(dst, src)
	case Q5_K:
		dequantizeQ5K(dst, src)
	case Q6_K:
		dequantizeQ6K(dst, src)
	case IQ2XXS:
		dequantizeIQ2XXS(dst, src)
	case IQ2S:
		dequantizeIQ2S(dst, src)
	case IQ3S:
		dequantizeIQ3S(dst, src)
	default:
		return fmt.Errorf("%w: %s", ErrUnsupportedDType, dtype)
	}
	return nil
}

func dequantizeQ4_0(dst []float32, src []byte) {
	for off, si := 0, 0; si+18 <= len(src); off, si = off+32, si+18 {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
		for j := range 16 {
			dst[off+j] = d * float32(int(src[si+2+j]&0x0f)-8)
			dst[off+j+16] = d * float32(int(src[si+2+j]>>4)-8)
		}
	}
}

func dequantizeQ5_0(dst []float32, src []byte) {
	for off, si := 0, 0; si+22 <= len(src); off, si = off+32, si+22 {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
		qh := binary.LittleEndian.Uint32(src[si+2:])
		for j := range 16 {
			xh0 := uint8(((qh >> uint(j)) << 4) & 0x10)
			xh1 := uint8((qh >> uint(j+12)) & 0x10)
			dst[off+j] = d * float32(int(src[si+6+j]&0x0f|xh0)-16)
			dst[off+j+16] = d * float32(int(src[si+6+j]>>4|xh1)-16)
		}
	}
}

func dequantizeQ8_0(dst []float32, src []byte) {
	for off, si := 0, 0; si+34 <= len(src); off, si = off+32, si+34 {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
		for i := 0; i < 32; i++ {
			dst[off+i] = d * float32(int8(src[si+2+i]))
		}
	}
}

func getScaleMinK4(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	d := (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4)
	m := (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return d, m
}

func dequantizeQ4K(dst []float32, src []byte) {
	for off, si := 0, 0; si+144 <= len(src); off, si = off+256, si+144 {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
		dm := Float16ToFloat32(binary.LittleEndian.Uint16(src[si+2:]))
		scales := src[si+4 : si+16]
		qs := src[si+16 : si+144]
		is := 0
		for j := 0; j < 256; j += 64 {
			s0, m0 := getScaleMinK4(is, scales)
			s1, m1 := getScaleMinK4(is+1, scales)
			d1, min1 := d*float32(s0), dm*float32(m0)
			d2, min2 := d*float32(s1), dm*float32(m1)
			for l := 0; l < 32; l++ {
				dst[off+j+l] = d1*float32(qs[l]&0x0f) - min1
				dst[off+j+32+l] = d2*float32(qs[l]>>4) - min2
			}
			qs = qs[32:]
			is += 2
		}
	}
}

func dequantizeQ5K(dst []float32, src []byte) {
	for off, si := 0, 0; si+176 <= len(src); off, si = off+256, si+176 {
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
		dm := Float16ToFloat32(binary.LittleEndian.Uint16(src[si+2:]))
		scales := src[si+4 : si+16]
		qh := src[si+16 : si+48]
		ql := src[si+48 : si+176]
		is := 0
		for j := 0; j < 256; j += 64 {
			s0, m0 := getScaleMinK4(is, scales)
			s1, m1 := getScaleMinK4(is+1, scales)
			d1, min1 := d*float32(s0), dm*float32(m0)
			d2, min2 := d*float32(s1), dm*float32(m1)
			bit0 := uint8(1) << (2 * (j / 64))
			bit1 := bit0 << 1
			for l := 0; l < 32; l++ {
				q0 := ql[l] & 0x0f
				if qh[l]&bit0 != 0 {
					q0 += 16
				}
				q1 := ql[l] >> 4
				if qh[l]&bit1 != 0 {
					q1 += 16
				}
				dst[off+j+l] = d1*float32(q0) - min1
				dst[off+j+32+l] = d2*float32(q1) - min2
			}
			ql = ql[32:]
			is += 2
		}
	}
}

// dequantizeIQ2XXS expands one 256-value IQ2_XXS block. The codebook and sign
// tables are shared with the ggml reference implementation.
func dequantizeIQ2XXS(dst []float32, src []byte) {
	d := Float16ToFloat32(binary.LittleEndian.Uint16(src))
	for ib32 := 0; ib32 < 8; ib32++ {
		base := 2 + 8*ib32
		aux0 := binary.LittleEndian.Uint32(src[base:])
		aux1 := binary.LittleEndian.Uint32(src[base+4:])
		db := d * (0.5 + float32(aux1>>28)) * 0.25
		for l := 0; l < 4; l++ {
			grid := iq2xxs_grid[(aux0>>(8*l))&0xff]
			signs := ksigns_iq2xs[(aux1>>(7*l))&127]
			out := ib32*32 + l*8
			for j := 0; j < 8; j++ {
				sign := float32(1)
				if signs&(1<<uint(j)) != 0 {
					sign = -1
				}
				dst[out+j] = db * float32(uint8(grid>>(8*j))) * sign
			}
		}
	}
}

// dequantizeIQ2S expands one 256-value IQ2_S block.
func dequantizeIQ2S(dst []float32, src []byte) {
	d := Float16ToFloat32(binary.LittleEndian.Uint16(src))
	qs := src[2:66]
	qh := src[66:74]
	scales := src[74:82]
	signs := qs[32:64]
	for ib32 := 0; ib32 < 8; ib32++ {
		db0 := d * (0.5 + float32(scales[ib32]&0xf)) * 0.25
		db1 := d * (0.5 + float32(scales[ib32]>>4)) * 0.25
		for l := 0; l < 4; l++ {
			idx := uint16(qs[4*ib32+l]) | ((uint16(qh[ib32]) << (8 - 2*l)) & 0x300)
			grid := iq2s_grid[idx]
			signsByte := signs[4*ib32+l]
			db := db0
			if l >= 2 {
				db = db1
			}
			out := ib32*32 + l*8
			for j := 0; j < 8; j++ {
				sign := float32(1)
				if signsByte&(1<<uint(j)) != 0 {
					sign = -1
				}
				dst[out+j] = db * float32(uint8(grid>>(8*j))) * sign
			}
		}
	}
}

// dequantizeIQ3S expands one 256-value IQ3_S block.
func dequantizeIQ3S(dst []float32, src []byte) {
	d := Float16ToFloat32(binary.LittleEndian.Uint16(src))
	qs := src[2:66]
	qh := src[66:74]
	signs := src[74:106]
	scales := src[106:110]
	for ib32 := 0; ib32 < 8; ib32 += 2 {
		db1 := d * (1 + 2*float32(scales[ib32/2]&0xf))
		db2 := d * (1 + 2*float32(scales[ib32/2]>>4))
		qsOff := ib32 * 8
		signsOff := ib32 * 4
		qhOff := ib32
		for half := 0; half < 2; half++ {
			db := db1
			if half == 1 {
				db = db2
			}
			for l := 0; l < 4; l++ {
				idx1 := uint16(qs[qsOff+2*l]) | ((uint16(qh[qhOff]) << (8 - 2*l)) & 256)
				idx2 := uint16(qs[qsOff+2*l+1]) | ((uint16(qh[qhOff]) << (7 - 2*l)) & 256)
				grid1 := iq3s_grid[idx1]
				grid2 := iq3s_grid[idx2]
				signsByte := signs[signsOff+l]
				out := ib32*32 + half*32 + l*8
				for j := 0; j < 4; j++ {
					sign := float32(1)
					if signsByte&(1<<uint(j)) != 0 {
						sign = -1
					}
					dst[out+j] = db * float32(uint8(grid1>>(8*j))) * sign
				}
				for j := 0; j < 4; j++ {
					sign := float32(1)
					if signsByte&(1<<uint(j+4)) != 0 {
						sign = -1
					}
					dst[out+j+4] = db * float32(uint8(grid2>>(8*j))) * sign
				}
			}
			qhOff++
			qsOff += 8
			signsOff += 4
		}
	}
}

func dequantizeQ6K(dst []float32, src []byte) {
	for off, si := 0, 0; si+210 <= len(src); off, si = off+256, si+210 {
		ql := src[si : si+128]
		qh := src[si+128 : si+192]
		sc := src[si+192 : si+208]
		d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si+208:]))
		for n := 0; n < 2; n++ {
			qlN := ql[n*64:]
			qhN := qh[n*32:]
			scN := sc[n*8:]
			base := off + n*128
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((qlN[l]&0x0f)|(((qhN[l]>>0)&3)<<4)) - 32
				q2 := int8((qlN[l+32]&0x0f)|(((qhN[l]>>2)&3)<<4)) - 32
				q3 := int8((qlN[l]>>4)|(((qhN[l]>>4)&3)<<4)) - 32
				q4 := int8((qlN[l+32]>>4)|(((qhN[l]>>6)&3)<<4)) - 32
				dst[base+l] = d * float32(int8(scN[is+0])) * float32(q1)
				dst[base+l+32] = d * float32(int8(scN[is+2])) * float32(q2)
				dst[base+l+64] = d * float32(int8(scN[is+4])) * float32(q3)
				dst[base+l+96] = d * float32(int8(scN[is+6])) * float32(q4)
			}
		}
	}
}

// Matrix is a row-major matrix of Rows output rows, each Cols values wide.
// Data is the packed row storage for DType; weight storage is read-only.
type Matrix struct {
	Data  []byte
	Rows  int64
	Cols  int64
	DType DType
}

// Validate checks shape, dtype support, and byte length.
func (m Matrix) Validate() error {
	rowBytes, err := m.DType.RowBytes(m.Cols)
	if err != nil {
		return err
	}
	if m.Rows <= 0 {
		return fmt.Errorf("invalid row count %d", m.Rows)
	}
	if m.Rows > (1<<63-1)/int64(rowBytes) {
		return fmt.Errorf("matrix byte size overflows")
	}
	want := m.Rows * int64(rowBytes)
	if int64(len(m.Data)) != want {
		return fmt.Errorf("matrix data has %d bytes, want %d", len(m.Data), want)
	}
	return nil
}

// MatVec computes dst = M * x. dst must hold Rows values and x must hold Cols.
func (m Matrix) MatVec(dst, x []float32) error {
	if int64(len(x)) != m.Cols {
		return fmt.Errorf("input length %d, want %d", len(x), m.Cols)
	}
	if int64(len(dst)) != m.Rows {
		return fmt.Errorf("output length %d, want %d", len(dst), m.Rows)
	}
	return m.matmul(dst, x, 1)
}

// MatMul computes dst[b, :] = M * x[b, :] for batch rows. x is laid out as
// batch contiguous Cols-wide rows, and dst as batch contiguous Rows-wide rows.
func (m Matrix) MatMul(dst, x []float32, batch int) error {
	if batch <= 0 {
		return fmt.Errorf("invalid batch %d", batch)
	}
	if int64(len(x)) != int64(batch)*m.Cols {
		return fmt.Errorf("input length %d, want %d", len(x), int64(batch)*m.Cols)
	}
	if int64(len(dst)) != int64(batch)*m.Rows {
		return fmt.Errorf("output length %d, want %d", len(dst), int64(batch)*m.Rows)
	}
	return m.matmul(dst, x, batch)
}

func (m Matrix) matmul(dst, x []float32, batch int) error {
	if err := m.Validate(); err != nil {
		return err
	}
	rowBytes, _ := m.DType.RowBytes(m.Cols)
	workers := 1
	if int64(len(m.Data)) >= 64*1024 {
		workers = runtime.GOMAXPROCS(0)
		if int64(workers) > m.Rows {
			workers = int(m.Rows)
		}
	}
	if workers <= 1 {
		m.matmulRows(dst, x, batch, 0, m.Rows, rowBytes)
		return nil
	}
	var wg sync.WaitGroup
	chunk := (m.Rows + int64(workers) - 1) / int64(workers)
	for w := 0; w < workers; w++ {
		start := int64(w) * chunk
		if start >= m.Rows {
			break
		}
		end := start + chunk
		if end > m.Rows {
			end = m.Rows
		}
		wg.Add(1)
		go func(start, end int64) {
			defer wg.Done()
			m.matmulRows(dst, x, batch, start, end, rowBytes)
		}(start, end)
	}
	wg.Wait()
	return nil
}

func (m Matrix) matmulRows(dst, x []float32, batch int, rowStart, rowEnd int64, rowBytes uint64) {
	cols := int(m.Cols)
	for row := rowStart; row < rowEnd; row++ {
		src := m.Data[int64(row)*int64(rowBytes):]
		switch m.DType {
		case F32:
			rowPtr := src[:cols*4]
			for b := 0; b < batch; b++ {
				xb := x[b*cols : (b+1)*cols]
				out := dst[b*int(m.Rows)+int(row)]
				var sum float32
				for i := 0; i < cols; i++ {
					sum += xb[i] * math.Float32frombits(binary.LittleEndian.Uint32(rowPtr[i*4:]))
				}
				out = sum
				dst[b*int(m.Rows)+int(row)] = out
			}
		case F16:
			for b := 0; b < batch; b++ {
				xb := x[b*cols : (b+1)*cols]
				var sum float32
				for i := 0; i < cols; i++ {
					sum += xb[i] * Float16ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
				}
				dst[b*int(m.Rows)+int(row)] = sum
			}
		case Q8_0:
			for b := 0; b < batch; b++ {
				xb := x[b*cols : (b+1)*cols]
				var sum float32
				for block, si := 0, 0; block < cols/32; block, si = block+1, si+34 {
					d := Float16ToFloat32(binary.LittleEndian.Uint16(src[si:]))
					for i := 0; i < 32; i++ {
						sum += xb[block*32+i] * d * float32(int8(src[si+2+i]))
					}
				}
				dst[b*int(m.Rows)+int(row)] = sum
			}
		case Q4_0, Q5_0:
			var values [32]float32
			for b := 0; b < batch; b++ {
				xb := x[b*cols : (b+1)*cols]
				var sum float32
				blockBytes := 18
				if m.DType == Q5_0 {
					blockBytes = 22
				}
				for block, si := 0, 0; block < cols/32; block, si = block+1, si+blockBytes {
					if m.DType == Q4_0 {
						dequantizeQ4_0(values[:], src[si:si+18])
					} else {
						dequantizeQ5_0(values[:], src[si:si+22])
					}
					for i := range 32 {
						sum += xb[block*32+i] * values[i]
					}
				}
				dst[b*int(m.Rows)+int(row)] = sum
			}
		default:
			var values [kQuantBlock]float32
			for b := 0; b < batch; b++ {
				xb := x[b*cols : (b+1)*cols]
				var sum float32
				for off, si := 0, 0; off < cols; off, si = off+256, si+int(m.blockBytes()) {
					block := src[si:]
					switch m.DType {
					case Q4_K:
						dequantizeQ4K(values[:], block[:144])
					case Q5_K:
						dequantizeQ5K(values[:], block[:176])
					case Q6_K:
						dequantizeQ6K(values[:], block[:210])
					case IQ2XXS:
						dequantizeIQ2XXS(values[:], block[:66])
					case IQ2S:
						dequantizeIQ2S(values[:], block[:82])
					case IQ3S:
						dequantizeIQ3S(values[:], block[:110])
					}
					for i := 0; i < 256; i++ {
						sum += xb[off+i] * values[i]
					}
				}
				dst[b*int(m.Rows)+int(row)] = sum
			}
		}
	}
}

func (m Matrix) blockBytes() int64 {
	block, _ := m.DType.BlockLayout()
	return block.Bytes
}
