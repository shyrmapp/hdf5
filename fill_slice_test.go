package hdf5

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// Fixtures come from testdata/c_interop/gen_fill.c.
const (
	fillV3     = "testdata/c_interop/fill_v3.h5"
	fillLatest = "testdata/c_interop/fill_latest.h5"
)

func openDS(t *testing.T, path, name string, opts ...OpenOption) *Dataset {
	t.Helper()
	f, err := Open(path, opts...)
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })
	obj, err := f.Resolve("/" + name)
	require.NoError(t, err)
	return obj.(*Dataset)
}

func full(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestFillValues(t *testing.T) {
	partial := full(16, -999)
	partial[0], partial[1], partial[4], partial[5] = 1, 2, 3, 4
	nc := full(16, float64(float32(9.96921e36)))
	nc[0], nc[1], nc[4], nc[5] = 1.5, 2.5, 3.5, 4.5

	for _, file := range []string{fillV3, fillLatest} {
		cases := []struct {
			name string
			want []float64
		}{
			{"partial", partial},
			{"unwritten_chunked", full(16, 7)},
			{"unwritten_chunked_unlim", full(16, 7)},
			{"nc_fill", nc},
			{"unwritten_contig", full(6, 5)},
		}
		for _, c := range cases {
			t.Run(file+"/"+c.name, func(t *testing.T) {
				ds := openDS(t, file, c.name)
				got, err := ds.Read()
				require.NoError(t, err)
				require.Equal(t, c.want, got)

				n := uint64(len(c.want))
				if n == 16 {
					s, err := ds.ReadSlice([]uint64{0, 0}, []uint64{4, 4})
					require.NoError(t, err)
					require.Equal(t, c.want, s)
					s, err = ds.ReadSlice([]uint64{1, 1}, []uint64{3, 3})
					require.NoError(t, err)
					require.Equal(t, []float64{c.want[5], c.want[6], c.want[7], c.want[9], c.want[10], c.want[11], c.want[13], c.want[14], c.want[15]}, s)
				} else {
					s, err := ds.ReadSlice([]uint64{1}, []uint64{n - 1})
					require.NoError(t, err)
					require.Equal(t, c.want[1:], s)
				}
			})
		}
	}
}

func TestEmptyUnlimitedChunkIterator(t *testing.T) {
	for _, file := range []string{fillV3, fillLatest} {
		ds := openDS(t, file, "empty_unlim")
		it, err := ds.ChunkIterator()
		require.NoError(t, err)
		require.False(t, it.Next())
		require.NoError(t, it.Err())
		got, err := ds.Read()
		require.NoError(t, err)
		require.Empty(t, got)
	}
	// Unwritten chunked datasets have no chunks to iterate either.
	it, err := openDS(t, fillV3, "unwritten_chunked").ChunkIterator()
	require.NoError(t, err)
	require.Zero(t, it.Total())
}

func TestFillValueHelperEncodings(t *testing.T) {
	// ReadSlice on the partial dataset exercises layout v3 + B-tree; the
	// message variants are covered in internal/core.
	ds := openDS(t, fillV3, "partial")
	got, err := ds.ReadSlice([]uint64{2, 2}, []uint64{2, 2})
	require.NoError(t, err)
	require.Equal(t, full(4, -999), got)
}

// TestSliceMatchesRead compares ReadSlice/ReadHyperslab with a naive
// reference built from Read() (value == linear index) for contiguous and
// chunked layouts, rank 1-3, with stride and block.
func TestSliceMatchesRead(t *testing.T) {
	shapes := map[string][]uint64{"1d": {30}, "2d": {5, 6}, "3d": {4, 5, 6}}
	for _, layout := range []string{"contig", "chunk"} {
		for rank, dims := range shapes {
			name := layout + rank
			ds := openDS(t, fillV3, name)
			all, err := ds.Read()
			require.NoError(t, err)

			for _, sel := range selectionsFor(dims) {
				if !inBounds(sel, dims) {
					continue
				}
				t.Run(fmt.Sprintf("%s/%v", name, sel), func(t *testing.T) {
					s := sel
					got, err := ds.ReadHyperslab(&s)
					require.NoError(t, err)
					require.Equal(t, naive(all, dims, s), got)
				})
			}
		}
	}

	// The cases from the bug report.
	ds := openDS(t, fillV3, "contig3d")
	got, err := ds.ReadSlice([]uint64{0, 1, 0}, []uint64{2, 2, 6})
	require.NoError(t, err)
	want := append(seq(6, 18), seq(36, 48)...)
	require.Equal(t, want, got)

	got, err = openDS(t, fillV3, "contig1d").ReadHyperslab(&HyperslabSelection{
		Start: []uint64{14}, Count: []uint64{5}, Stride: []uint64{2}})
	require.NoError(t, err)
	require.Equal(t, []float64{14, 16, 18, 20, 22}, got)
}

func inBounds(s HyperslabSelection, dims []uint64) bool {
	for i, d := range dims {
		if s.Start[i]+(s.Count[i]-1)*s.Stride[i]+s.Block[i] > d {
			return false
		}
	}
	return true
}

func seq(lo, hi int) []float64 {
	var out []float64
	for i := lo; i < hi; i++ {
		out = append(out, float64(i))
	}
	return out
}

// selectionsFor generates a spread of selections for dims, including full
// rows, partial rows, strides and blocks in each dimension.
func selectionsFor(dims []uint64) []HyperslabSelection {
	n := len(dims)
	mk := func(start, count, stride, block uint64) (s, c, st, b []uint64) {
		for i := 0; i < n; i++ {
			s = append(s, start)
			c = append(c, count)
			st = append(st, stride)
			b = append(b, block)
		}
		return
	}
	var out []HyperslabSelection
	add := func(s, c, st, b []uint64) {
		out = append(out, HyperslabSelection{Start: s, Count: c, Stride: st, Block: b})
	}
	// Full dataset, then each leading-dimension variation.
	full := make([]uint64, n)
	zero := make([]uint64, n)
	ones := make([]uint64, n)
	for i := range dims {
		full[i], ones[i] = dims[i], 1
	}
	add(zero, full, ones, ones)
	// One row/plane in the first dimension, rest full.
	c := append([]uint64(nil), full...)
	c[0] = 1
	st := append([]uint64(nil), zero...)
	st[0] = 1
	add(st, c, ones, ones)
	// Two leading rows starting at 1, rest full (linear run for contiguous).
	c = append([]uint64(nil), full...)
	c[0] = 2
	add(st, c, ones, ones)
	// Partial in the middle dimension / last dimension.
	for d := 0; d < n; d++ {
		s := append([]uint64(nil), zero...)
		c := append([]uint64(nil), full...)
		s[d], c[d] = 1, 2
		add(s, c, ones, ones)
	}
	// Uniform strided and blocked selections.
	s, cc, sr, b := mk(0, 2, 2, 1)
	add(s, cc, sr, b)
	s, cc, sr, b = mk(1, 2, 3, 2)
	add(s, cc, sr, b)
	s, cc, sr, b = mk(0, 2, 3, 2)
	add(s, cc, sr, b)
	// Stride in the last dimension only.
	s = append([]uint64(nil), zero...)
	cc = append([]uint64(nil), full...)
	sr = append([]uint64(nil), ones...)
	cc[n-1], sr[n-1] = 2, 3
	add(s, cc, sr, ones)
	return out
}

// naive builds the expected selection from the full row-major data.
func naive(all []float64, dims []uint64, s HyperslabSelection) []float64 {
	coords := make([][]uint64, len(dims))
	for i := range dims {
		for c := uint64(0); c < s.Count[i]; c++ {
			for b := uint64(0); b < s.Block[i]; b++ {
				coords[i] = append(coords[i], s.Start[i]+c*s.Stride[i]+b)
			}
		}
	}
	var out []float64
	var rec func(d int, off uint64)
	rec = func(d int, off uint64) {
		if d == len(dims) {
			out = append(out, all[off])
			return
		}
		stride := uint64(1)
		for _, x := range dims[d+1:] {
			stride *= x
		}
		for _, c := range coords[d] {
			rec(d+1, off+c*stride)
		}
	}
	rec(0, 0)
	return out
}

func TestReadSliceOverflow(t *testing.T) {
	for _, name := range []string{"contig1d", "chunk1d"} {
		ds := openDS(t, fillV3, name)
		require.NotPanics(t, func() {
			_, err := ds.ReadSlice([]uint64{math.MaxUint64 - 1}, []uint64{4})
			require.Error(t, err)
			_, err = ds.ReadSlice([]uint64{2}, []uint64{math.MaxUint64})
			require.Error(t, err)
			_, err = ds.ReadHyperslab(&HyperslabSelection{
				Start: []uint64{math.MaxUint64 - 1}, Count: []uint64{4}, Stride: []uint64{2}, Block: []uint64{math.MaxUint64}})
			require.Error(t, err)
			_, err = ds.ReadHyperslab(&HyperslabSelection{
				Start: []uint64{math.MaxUint64 - 1}, Count: []uint64{2}})
			require.Error(t, err)
			_, err = ds.ReadHyperslab(&HyperslabSelection{
				Start: []uint64{0}, Count: []uint64{2}, Block: []uint64{math.MaxUint64}})
			require.Error(t, err)
		})
	}
}

// TestReadLimitBeforeAllocation patches a tiny file to declare 4 Gi int8
// elements and checks that a full-range ReadSlice fails against a 1 MiB
// limit without having allocated anything close to the declared size.
func TestReadLimitBeforeAllocation(t *testing.T) {
	data, err := os.ReadFile("testdata/c_interop/fill_v3.h5")
	require.NoError(t, err)
	// Dataspace v1 body with rank 1, dims {30}, maxdims {30}; patch both to 4 Gi.
	hdr := []byte{1, 1, 1, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0}
	patched := 0
	for i := 0; i+len(hdr) <= len(data); i++ {
		if bytes.Equal(data[i:i+len(hdr)], hdr) {
			data[i+8], data[i+12], data[i+16], data[i+20] = 0, 1, 0, 1
			patched++
		}
	}
	require.Positive(t, patched)
	path := t.TempDir() + "/huge.h5"
	require.NoError(t, os.WriteFile(path, data, 0o600))

	for _, name := range []string{"contig1d", "chunk1d"} {
		ds := openDS(t, path, name, WithMaxReadBytes(1<<20))
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		_, err := ds.ReadSlice([]uint64{0}, []uint64{1 << 32})
		runtime.ReadMemStats(&after)
		require.Error(t, err, name)
		require.Contains(t, err.Error(), "limit")
		require.Less(t, after.TotalAlloc-before.TotalAlloc, uint64(64<<20), name)
	}
}
