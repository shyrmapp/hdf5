package hdf5

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestFloat16Read reads a C-library-written H5T_IEEE_F16LE dataset covering
// zero, normals, the float16 maximum, and a subnormal.
func TestFloat16Read(t *testing.T) {
	f, err := Open("testdata/float16_default.h5")
	require.NoError(t, err)
	defer f.Close()

	var got []float64
	f.Walk(func(path string, obj Object) {
		if ds, ok := obj.(*Dataset); ok && path == "/f16" {
			got, err = ds.Read()
			require.NoError(t, err)
		}
	})
	require.Equal(t, []float64{0, 1, -2.5, 0.15625, 65504, math.Ldexp(1, -24)}, got)
}

// TestLZFWrite_RoundTrip writes an LZF-compressed chunked dataset and reads
// it back; the stored file must carry filter ID 32000 (h5py-compatible).
func TestLZFWrite_RoundTrip(t *testing.T) {
	file := filepath.Join(t.TempDir(), "lzf.h5")

	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i % 50)
	}

	fw, err := CreateForWrite(file, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/z", Float64, []uint64{1000},
		WithChunkDims([]uint64{250}), WithLZFCompression())
	require.NoError(t, err)
	require.NoError(t, ds.Write(data))
	require.NoError(t, fw.Close())

	f, err := Open(file)
	require.NoError(t, err)
	defer f.Close()

	var got []float64
	f.Walk(func(_ string, obj Object) {
		if d, ok := obj.(*Dataset); ok {
			got, err = d.Read()
			require.NoError(t, err)
		}
	})
	require.Equal(t, data, got)
}

// TestCInterop_LZFFilterHeader verifies the official h5dump identifies the
// LZF filter in a Go-written file (data needs the LZF plugin, so only the
// header is checked).
func TestCInterop_LZFFilterHeader(t *testing.T) {
	h5dump := h5tool(t, "h5dump")
	file := filepath.Join(t.TempDir(), "lzf_interop.h5")

	fw, err := CreateForWrite(file, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/z", Int32, []uint64{100},
		WithChunkDims([]uint64{25}), WithLZFCompression())
	require.NoError(t, err)
	vals := make([]int32, 100)
	for i := range vals {
		vals[i] = int32(i)
	}
	require.NoError(t, ds.Write(vals))
	require.NoError(t, fw.Close())

	out := runH5(t, h5dump, "-H", "-p", file)
	require.Contains(t, out, "FILTER_ID 32000")
	require.Contains(t, out, "lzf")
}
