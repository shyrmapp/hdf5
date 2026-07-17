package hdf5

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestReadComplex reads the complex128 dataset from a file written by the
// HDF5 2.x C library (H5T_COMPLEX_IEEE_F64LE, new in HDF5 2.0).
func TestReadComplex(t *testing.T) {
	f, err := Open("testdata/hdf5_v2_latest.h5")
	require.NoError(t, err)
	defer f.Close()

	var got []complex128
	f.Walk(func(path string, obj Object) {
		if ds, ok := obj.(*Dataset); ok && path == "/cplx" {
			got, err = ds.ReadComplex()
			require.NoError(t, err)
		}
	})
	require.Equal(t, []complex128{1 + 2i, 3 + 4i, 5 + 6i, 7 + 8i, 9 + 10i}, got)
}

// TestReadComplex_NotComplex rejects non-complex datasets with a clear error.
func TestReadComplex_NotComplex(t *testing.T) {
	f, err := Open("testdata/hdf5_v2_latest.h5")
	require.NoError(t, err)
	defer f.Close()

	f.Walk(func(path string, obj Object) {
		if ds, ok := obj.(*Dataset); ok && path == "/f64" {
			_, err := ds.ReadComplex()
			require.ErrorContains(t, err, "not complex")
		}
	})
}

// TestHDF5v2Latest_FullRead sweeps every dataset in the 2.x libver-latest
// file: superblock v3, layout v4 chunked+gzip, float16, dense links.
func TestHDF5v2Latest_FullRead(t *testing.T) {
	f, err := Open("testdata/hdf5_v2_latest.h5")
	require.NoError(t, err)
	defer f.Close()

	read := map[string]int{}
	f.Walk(func(path string, obj Object) {
		ds, ok := obj.(*Dataset)
		if !ok {
			return
		}
		if path == "/cplx" {
			v, err := ds.ReadComplex()
			require.NoError(t, err, path)
			read[path] = len(v)
			return
		}
		v, err := ds.Read()
		require.NoError(t, err, path)
		read[path] = len(v)
	})
	require.Equal(t, 100, read["/zint"], "layout v4 chunked+gzip")
	require.Equal(t, 5, read["/f16"], "float16")
	require.Equal(t, 5, read["/cplx"], "complex128")
	require.Equal(t, 5, read["/f64"])
	require.Len(t, read, 16, "12 dense-link datasets + f64 + f16 + cplx + zint")
}
