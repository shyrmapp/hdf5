package hdf5

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// h5dumpClean runs h5dump on file and fails on any error output. When
// H5_EXPORT_DIR is set the file is copied there so it can also be checked
// with other h5dump versions (e.g. 1.10 in docker).
func h5dumpClean(t *testing.T, file string) string {
	t.Helper()
	h5dump := h5tool(t, "h5dump")
	if dir := os.Getenv("H5_EXPORT_DIR"); dir != "" {
		b, err := os.ReadFile(file) //nolint:gosec // test path
		require.NoError(t, err)
		name := strings.ReplaceAll(t.Name(), "/", "_") + ".h5"
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), b, 0o600))
	}
	out := runH5(t, h5dump, "--enable-error-stack", file)
	require.NotContains(t, out, "HDF5-DIAG", out)
	require.NotContains(t, out, "unable to print", out)
	return out
}

func newWriteFile(t *testing.T, opts ...WriteOption) (*FileWriter, string) {
	t.Helper()
	file := filepath.Join(t.TempDir(), "x.h5")
	fw, err := CreateForWrite(file, CreateTruncate, opts...)
	require.NoError(t, err)
	return fw, file
}

// A v0 group holds at most 256 children; the 257th must error without
// corrupting the file.
func TestCInterop_GroupChildLimit(t *testing.T) {
	fw, file := newWriteFile(t)
	for i := 0; i < 256; i++ {
		_, err := fw.CreateDataset(fmt.Sprintf("/d%03d", i), Int32, []uint64{1})
		require.NoError(t, err, "child %d", i)
	}
	_, err := fw.CreateDataset("/overflow", Int32, []uint64{1})
	require.Error(t, err)
	_, err = fw.CreateDataset("/overflow2", Int32, []uint64{1})
	require.Error(t, err)
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, `DATASET "d255"`)
	require.Contains(t, out, `DATASET "d000"`)
	require.NotContains(t, out, "overflow")
}
