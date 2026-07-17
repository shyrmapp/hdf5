package hdf5

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// counterValues returns [0, 1, ..., n-1] as float64.
func counterValues(n int) []float64 {
	vals := make([]float64, n)
	for i := range vals {
		vals[i] = float64(i)
	}
	return vals
}

// grid2DValues returns rows x cols values i*100+j flattened row-major.
func grid2DValues(rows, cols int) []float64 {
	vals := make([]float64, 0, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			vals = append(vals, float64(i*100+j))
		}
	}
	return vals
}

// readOnlyDataset opens path and returns the values of its single dataset /z.
func readOnlyDataset(t *testing.T, path string) []float64 {
	t.Helper()

	f, err := Open(path)
	require.NoError(t, err)
	defer f.Close()

	var got []float64
	var found bool
	f.Walk(func(_ string, obj Object) {
		if ds, ok := obj.(*Dataset); ok {
			found = true
			vals, err := ds.Read()
			require.NoError(t, err)
			got = vals
		}
	})
	require.True(t, found, "no dataset found in %s", path)
	return got
}

// TestLayoutV4ChunkIndexes reads HDF5 1.10+/2.x files written under
// H5F_LIBVER_LATEST, whose data layout messages are version 4/5 with the
// modern chunk indexes (Single Chunk, Implicit, Fixed Array, Extensible
// Array, v2 B-tree), and asserts the exact expected data per index type.
func TestLayoutV4ChunkIndexes(t *testing.T) {
	tests := []struct {
		file string
		want []float64
	}{
		{"v4_single.h5", counterValues(50)},    // Single Chunk index.
		{"v4_single_gz.h5", counterValues(50)}, // Single Chunk index, gzip, layout v5.
		{"v4_implicit.h5", counterValues(200)}, // Implicit index.
		{"v4_fixed.h5", counterValues(200)},    // Fixed Array index, gzip, layout v5.
		{"v4_ext.h5", counterValues(200)},      // Extensible Array index.
		{"v4_btree2.h5", grid2DValues(20, 30)}, // Version 2 B-tree index, 2D.
	}

	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			got := readOnlyDataset(t, "testdata/layout_v4/"+tt.file)
			require.Equal(t, tt.want, got)
		})
	}
}

// TestLayoutV4Hyperslab reads a hyperslab from a v2 B-tree indexed 2D dataset
// to cover the v4 chunk index path in ReadSlice.
func TestLayoutV4Hyperslab(t *testing.T) {
	f, err := Open("testdata/layout_v4/v4_btree2.h5")
	require.NoError(t, err)
	defer f.Close()

	var ds *Dataset
	f.Walk(func(_ string, obj Object) {
		if d, ok := obj.(*Dataset); ok {
			ds = d
		}
	})
	require.NotNil(t, ds)

	// Rows 5..6, columns 8..12: spans chunk boundary at column 10.
	got, err := ds.ReadSlice([]uint64{5, 8}, []uint64{2, 5})
	require.NoError(t, err)
	require.Equal(t, []float64{
		508, 509, 510, 511, 512,
		608, 609, 610, 611, 612,
	}, got)
}

// TestLayoutV4ChunkIterator iterates the chunks of an Extensible Array
// indexed dataset to cover the v4 chunk index path in ChunkIterator.
func TestLayoutV4ChunkIterator(t *testing.T) {
	f, err := Open("testdata/layout_v4/v4_ext.h5")
	require.NoError(t, err)
	defer f.Close()

	var ds *Dataset
	f.Walk(func(_ string, obj Object) {
		if d, ok := obj.(*Dataset); ok {
			ds = d
		}
	})
	require.NotNil(t, ds)

	iter, err := ds.ChunkIterator()
	require.NoError(t, err)
	require.Equal(t, 8, iter.Total()) // 200 elements / 25 per chunk.

	var all []float64
	for iter.Next() {
		chunk, err := iter.Chunk()
		require.NoError(t, err)
		all = append(all, chunk.([]float64)...)
	}
	require.NoError(t, iter.Err())
	require.Equal(t, counterValues(200), all)
}
