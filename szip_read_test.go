package hdf5

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestReadSZIPDataset reads an SZIP-compressed dataset from an official
// HDF5 test file (40x20 int32, chunked 20x10, NN coding, K13, LSB) and
// verifies every value against h5dump output (sequential 0..799).
func TestReadSZIPDataset(t *testing.T) {
	file, err := Open("testdata/hdf5_official/h5repack_szip.h5")
	require.NoError(t, err)
	defer file.Close()

	obj, err := file.Resolve("/dset_szip")
	require.NoError(t, err)
	ds, ok := obj.(*Dataset)
	require.True(t, ok, "/dset_szip is not a dataset")

	data, err := ds.Read()
	require.NoError(t, err)
	require.Len(t, data, 800)
	for i, v := range data {
		require.Equal(t, float64(i), v, "value mismatch at index %d", i)
	}
}
