package hdf5

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCompoundVLenStringMember reads an h5py-written compound whose members are
// int32, a variable-length string, and float64 — in that order.
//
// Regression for two bugs that only surface when a member FOLLOWS a
// variable-length one:
//
//   - ParseDatatypeMessage gave vlen (and array/enum/opaque) members "all
//     remaining bytes" as their properties, so the compound walker's
//     `offset += 8 + len(Properties)` skipped past every later member and the
//     type failed to parse at all.
//   - readVariableString read the heap reference from offset 0, folding the
//     4-byte length prefix into the heap address.
//
// Values are the ones h5dump reports, empty string on row 3 included.
func TestCompoundVLenStringMember(t *testing.T) {
	f, err := Open(filepath.Join("testdata", "vlen_strings.h5"))
	require.NoError(t, err)
	defer f.Close()

	obj, err := f.Resolve("/compound_with_vlen")
	require.NoError(t, err)
	dset, ok := obj.(*Dataset)
	require.True(t, ok, "/compound_with_vlen should be a dataset")

	rows, err := dset.ReadCompound()
	require.NoError(t, err)
	require.Len(t, rows, 5)

	want := []struct {
		id    int32
		name  string
		value float64
	}{
		{1, "Alice", 3.14},
		{2, "Bob", 2.71},
		{3, "Charlie Brown", 1.41},
		{4, "", 0},
		{5, "This is a very long string to test heap storage", 9.99},
	}

	for i, w := range want {
		require.Equalf(t, w.id, rows[i]["id"], "row %d id", i)
		require.Equalf(t, w.name, rows[i]["name"], "row %d name", i)
		// The trailing member is the one the parser used to lose entirely.
		require.InDeltaf(t, w.value, rows[i]["value"], 1e-9, "row %d value", i)
	}
}
