package hdf5

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCInteropIndexCases reads datasets written by testdata/c_interop/gen_index.c
// (HDF5 2.2 C library, libver latest). Expected values match h5dump: a v2
// filter pipeline (shuffle+deflate+fletcher32), a paged fixed array index
// (5000 chunks) and a v2 B-tree chunk index of depth > 1 (14400 chunks).
func TestCInteropIndexCases(t *testing.T) {
	f, err := Open("testdata/c_interop/index_cases.h5")
	require.NoError(t, err)
	defer f.Close()

	seq := func(n int) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = float64(i)
		}
		return v
	}
	grid := make([]float64, 0, 14400)
	for i := 0; i < 120; i++ {
		for j := 0; j < 120; j++ {
			grid = append(grid, float64(i*1000+j))
		}
	}
	want := map[string][]float64{
		"pipeline_v2":  seq(1000),
		"fixed_5000":   seq(10000),
		"unlimited_2d": grid,
	}

	seen := 0
	f.Walk(func(path string, obj Object) {
		ds, ok := obj.(*Dataset)
		if !ok {
			return
		}
		w, ok := want[ds.Name()]
		if !ok {
			return
		}
		seen++
		got, err := ds.Read()
		require.NoError(t, err, path)
		require.Equal(t, w, got, path)
	})
	require.Equal(t, len(want), seen)
}
