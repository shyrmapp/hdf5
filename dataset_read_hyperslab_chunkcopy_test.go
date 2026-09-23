package hdf5

import (
	"bytes"
	"fmt"
	"math/rand"
	"path/filepath"
	"testing"
)

// extractChunkPortionElementwise is the per-element extraction that
// extractChunkPortion replaced, kept verbatim as the reference: walk the
// whole selection, keep the coordinates inside the chunk, place each by
// selectionOutputIndex.
//
//nolint:gocognit // reference implementation, kept verbatim
func extractChunkPortionElementwise(
	chunkData []byte,
	chunkCoord, chunkDims, datasetDims []uint64,
	selection *HyperslabSelection,
	elementSize uint64,
	outputData []byte,
) {
	ndims := len(chunkCoord)
	chunkStart := make([]uint64, ndims)
	chunkEnd := make([]uint64, ndims)
	for i := 0; i < ndims; i++ {
		chunkStart[i] = chunkCoord[i] * chunkDims[i]
		chunkEnd[i] = chunkStart[i] + chunkDims[i]
		if chunkEnd[i] > datasetDims[i] {
			chunkEnd[i] = datasetDims[i]
		}
	}
	coords := make([]uint64, ndims)
	copy(coords, selection.Start)
	var rec func(dim int)
	rec = func(dim int) {
		if dim == ndims {
			for i := 0; i < ndims; i++ {
				if coords[i] < chunkStart[i] || coords[i] >= chunkEnd[i] {
					return
				}
			}
			off, st := uint64(0), uint64(1)
			for i := ndims - 1; i >= 0; i-- {
				off += (coords[i] - chunkStart[i]) * st
				st *= chunkDims[i]
			}
			src := off * elementSize
			dst := selectionOutputIndex(coords, selection) * elementSize
			if src+elementSize <= uint64(len(chunkData)) && dst+elementSize <= uint64(len(outputData)) {
				copy(outputData[dst:dst+elementSize], chunkData[src:src+elementSize])
			}
			return
		}
		for c := uint64(0); c < selection.Count[dim]; c++ {
			bs := selection.Start[dim] + c*selection.Stride[dim]
			for b := uint64(0); b < selection.Block[dim]; b++ {
				coords[dim] = bs + b
				if coords[dim] >= chunkEnd[dim] {
					return
				}
				rec(dim + 1)
			}
		}
	}
	rec(0)
}

// chunkCase is a dataset shape, a chunk shape and a valid selection on it.
type chunkCase struct {
	dims, chunk []uint64
	sel         HyperslabSelection
	es          uint64
}

func (c chunkCase) String() string {
	return fmt.Sprintf("dims=%v chunk=%v start=%v count=%v stride=%v block=%v es=%d",
		c.dims, c.chunk, c.sel.Start, c.sel.Count, c.sel.Stride, c.sel.Block, c.es)
}

// randomChunkCase draws a valid (non-overlapping, in-bounds) selection.
func randomChunkCase(r *rand.Rand) chunkCase {
	ndims := 1 + r.Intn(3)
	c := chunkCase{
		dims:  make([]uint64, ndims),
		chunk: make([]uint64, ndims),
		sel: HyperslabSelection{
			Start:  make([]uint64, ndims),
			Count:  make([]uint64, ndims),
			Stride: make([]uint64, ndims),
			Block:  make([]uint64, ndims),
		},
		es: []uint64{1, 2, 4, 8}[r.Intn(4)],
	}
	for i := 0; i < ndims; i++ {
		d := uint64(1 + r.Intn(24))
		c.dims[i] = d
		c.chunk[i] = uint64(1 + r.Intn(int(d)+3)) // may exceed the dim: edge chunks
		block := uint64(1 + r.Intn(3))
		stride := block + uint64(r.Intn(3)) // block <= stride
		if r.Intn(3) == 0 {
			stride, block = 1, 1
		}
		start := uint64(r.Intn(int(d)))
		// Largest count keeping start+(count-1)*stride+block <= d.
		if start+block > d {
			block = d - start
			if block > stride {
				stride = block
			}
		}
		maxCount := (d-start-block)/stride + 1
		c.sel.Start[i] = start
		c.sel.Stride[i] = stride
		c.sel.Block[i] = block
		c.sel.Count[i] = 1 + uint64(r.Int63n(int64(maxCount)))
	}
	return c
}

// forEachChunk runs fn over every chunk of the dataset with its chunk
// coordinate and the chunk's bytes cut from the row-major raw data (edge
// chunks padded, as HDF5 stores them).
//
//nolint:gocognit // test fixture builder
func forEachChunk(c chunkCase, raw []byte, fn func(coord []uint64, data []byte)) {
	ndims := len(c.dims)
	nchunks := make([]uint64, ndims)
	for i := range nchunks {
		nchunks[i] = (c.dims[i] + c.chunk[i] - 1) / c.chunk[i]
	}
	chunkElems := uint64(1)
	for _, v := range c.chunk {
		chunkElems *= v
	}
	coord := make([]uint64, ndims)
	var rec func(dim int)
	rec = func(dim int) {
		if dim == ndims {
			data := make([]byte, chunkElems*c.es)
			local := make([]uint64, ndims)
			for e := uint64(0); e < chunkElems; e++ {
				rem := e
				for i := ndims - 1; i >= 0; i-- {
					local[i] = rem % c.chunk[i]
					rem /= c.chunk[i]
				}
				off, st, in := uint64(0), uint64(1), true
				for i := ndims - 1; i >= 0; i-- {
					x := coord[i]*c.chunk[i] + local[i]
					if x >= c.dims[i] {
						in = false
						break
					}
					off += x * st
					st *= c.dims[i]
				}
				if in {
					copy(data[e*c.es:(e+1)*c.es], raw[off*c.es:(off+1)*c.es])
				}
			}
			fn(append([]uint64(nil), coord...), data)
			return
		}
		for k := uint64(0); k < nchunks[dim]; k++ {
			coord[dim] = k
			rec(dim + 1)
		}
	}
	rec(0)
}

// TestExtractChunkPortion_MatchesElementwise compares the run-copy
// extraction with the per-element reference, and both with the contiguous
// hyperslab extraction, over random dataset/chunk/selection shapes — every
// element width, strided and blocked selections, edge chunks.
func TestExtractChunkPortion_MatchesElementwise(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for n := 0; n < 3000; n++ {
		c := randomChunkCase(r)
		total := uint64(1)
		for _, d := range c.dims {
			total *= d
		}
		raw := make([]byte, total*c.es)
		r.Read(raw)

		outElems := calculateHyperslabOutputSize(&c.sel)
		got := make([]byte, outElems*c.es)
		ref := make([]byte, outElems*c.es)
		forEachChunk(c, raw, func(coord []uint64, data []byte) {
			extractChunkPortion(data, coord, c.chunk, c.dims, &c.sel, c.es, got)
			extractChunkPortionElementwise(data, coord, c.chunk, c.dims, &c.sel, c.es, ref)
		})
		if !bytes.Equal(got, ref) {
			t.Fatalf("case %d (%s): run copy differs from the per-element reference", n, c)
		}

		// Ground truth: the contiguous extraction walks the selection in
		// order with a running counter, independent of chunking.
		want := make([]byte, outElems*c.es)
		var idx uint64
		coords := append([]uint64(nil), c.sel.Start...)
		extractHyperslabRecursive(raw, want, c.dims, &c.sel, coords, 0, c.es, &idx)
		if !bytes.Equal(got, want) {
			t.Fatalf("case %d (%s): chunked extraction differs from the contiguous one", n, c)
		}
	}
}

// A chunk buffer shorter than the chunk (or an output shorter than the
// selection) drops the elements that don't fit, exactly as the per-element
// copy did, instead of panicking.
func TestExtractChunkPortion_ShortBuffers(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for n := 0; n < 1000; n++ {
		c := randomChunkCase(r)
		total := uint64(1)
		for _, d := range c.dims {
			total *= d
		}
		raw := make([]byte, total*c.es)
		r.Read(raw)
		outElems := calculateHyperslabOutputSize(&c.sel)
		outLen := r.Intn(int(outElems*c.es) + 1)
		got := make([]byte, outLen)
		ref := make([]byte, outLen)
		forEachChunk(c, raw, func(coord []uint64, data []byte) {
			data = data[:r.Intn(len(data)+1)]
			extractChunkPortion(data, coord, c.chunk, c.dims, &c.sel, c.es, got)
			extractChunkPortionElementwise(data, coord, c.chunk, c.dims, &c.sel, c.es, ref)
		})
		if !bytes.Equal(got, ref) {
			t.Fatalf("case %d (%s, out %d B): differs from the per-element reference", n, c, outLen)
		}
	}
}

// benchChunkCase is the OPERA NIMBUS RATE layout: 2200×1900 float64 in
// 69×119 chunks, read as 512-row bands.
func benchChunkCase() (dims, chunk []uint64) {
	return []uint64{2200, 1900}, []uint64{69, 119}
}

func benchmarkExtract(b *testing.B, extract func([]byte, []uint64, []uint64, []uint64, *HyperslabSelection, uint64, []byte)) {
	dims, chunk := benchChunkCase()
	const es, band = 8, 512
	data := make([]byte, chunk[0]*chunk[1]*es)
	sel := &HyperslabSelection{
		Start: []uint64{512, 0}, Count: []uint64{band, dims[1]},
		Stride: []uint64{1, 1}, Block: []uint64{1, 1},
	}
	out := make([]byte, band*dims[1]*es)
	coords := findOverlappingChunks(sel, chunk, dims)
	b.SetBytes(int64(len(out)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, cc := range coords {
			extract(data, cc, chunk, dims, sel, es, out)
		}
	}
}

// BenchmarkExtractChunkPortion measures placing one 512-row band of a
// many-chunk dataset from already-decoded chunks: run copy vs the old
// per-element walk.
func BenchmarkExtractChunkPortion(b *testing.B) {
	b.Run("runcopy", func(b *testing.B) { benchmarkExtract(b, extractChunkPortion) })
	b.Run("elementwise", func(b *testing.B) { benchmarkExtract(b, extractChunkPortionElementwise) })
}

// BenchmarkReadSliceChunkedBands reads a whole many-chunk dataset through
// ReadSlice in row bands, end to end (chunk index, filters, conversion).
func BenchmarkReadSliceChunkedBands(b *testing.B) {
	const rows, cols = 1100, 950
	path := filepath.Join(b.TempDir(), "bands.h5")
	fw, err := CreateForWrite(path, CreateTruncate)
	if err != nil {
		b.Fatal(err)
	}
	ds, err := fw.CreateDataset("/d", Float64, []uint64{rows, cols}, WithChunkDims([]uint64{69, 119}))
	if err != nil {
		b.Fatal(err)
	}
	vals := make([]float64, rows*cols)
	for i := range vals {
		vals[i] = float64(i)
	}
	if err := ds.Write(vals); err != nil {
		b.Fatal(err)
	}
	if err := fw.Close(); err != nil {
		b.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	rds, ok := findDatasetByName(f, "d")
	if !ok {
		b.Fatal("dataset not found")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for y := uint64(0); y < rows; y += 256 {
			h := min(uint64(256), rows-y)
			if _, err := rds.ReadSlice([]uint64{y, 0}, []uint64{h, cols}); err != nil {
				b.Fatal(err)
			}
		}
	}
}
