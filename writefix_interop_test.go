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

func TestCInterop_FloatAttributes(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/ds", Int32, []uint64{1})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{1}))
	require.NoError(t, ds.WriteAttribute("f32", float32(1.25)))
	require.NoError(t, ds.WriteAttribute("f64s", []float64{1, 2, 3}))
	g, err := fw.CreateGroup("/g")
	require.NoError(t, err)
	require.NoError(t, g.WriteAttribute("gf", float64(2.5)))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, "(0): 1.25")
	require.Contains(t, out, "(0): 1, 2, 3")
	require.Contains(t, out, "(0): 2.5")
}

func TestCInterop_HeaderGrowthAfterReopen(t *testing.T) {
	fw, file := newWriteFile(t)
	for i := 0; i < 3; i++ {
		ds, err := fw.CreateDataset(fmt.Sprintf("/ds%d", i), Int32, []uint64{3})
		require.NoError(t, err)
		require.NoError(t, ds.Write([]int32{int32(i), 7, 8}))
	}
	require.NoError(t, fw.Close())

	fw, err := OpenForWrite(file, OpenReadWrite)
	require.NoError(t, err)
	ds, err := fw.OpenDataset("/ds0")
	require.NoError(t, err)
	require.NoError(t, ds.WriteAttribute("note", strings.Repeat("n", 900)))
	require.NoError(t, ds.WriteAttribute("note2", strings.Repeat("m", 900)))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, strings.Repeat("n", 900))
	require.Contains(t, out, strings.Repeat("m", 900))
	require.Contains(t, out, "(0): 1, 7, 8")
	require.Contains(t, out, "(0): 2, 7, 8")
}

func TestCInterop_ChunkedAttributesAroundWrite(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/c", Int32, []uint64{10}, WithChunkDims([]uint64{5}))
	require.NoError(t, err)
	require.NoError(t, ds.WriteAttribute("before", strings.Repeat("b", 300)))
	require.NoError(t, ds.Write([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
	require.NoError(t, ds.WriteAttribute("after", strings.Repeat("a", 300)))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, "(0): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
	require.Contains(t, out, strings.Repeat("b", 300))
	require.Contains(t, out, strings.Repeat("a", 300))
}

func TestCInterop_HardLinkThenSoftLink(t *testing.T) {
	fw, file := newWriteFile(t)
	_, err := fw.CreateGroup("/g")
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/x", Int32, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{4, 5}))
	require.NoError(t, fw.CreateHardLink("/h0", "/x"))
	require.NoError(t, fw.CreateHardLink("/h1", "/x"))
	require.NoError(t, fw.CreateHardLink("/h2", "/x"))
	require.NoError(t, fw.CreateSoftLink("/g/s", "/x"))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, `SOFTLINK "s"`)
	require.Contains(t, out, `HARDLINK`)
}

func TestCInterop_ResizeAfterWriteKeepsData(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/r", Int32, []uint64{10},
		WithChunkDims([]uint64{5}), WithMaxDims([]uint64{Unlimited}))
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}))
	require.NoError(t, ds.Resize([]uint64{20}))
	require.NoError(t, ds.WriteAttribute("kept", "yes"))
	vals := make([]int32, 20)
	for i := range vals {
		vals[i] = int32(100 + i)
	}
	require.NoError(t, ds.Write(vals))
	require.NoError(t, ds.Resize([]uint64{30}))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, "100, 101, 102")
	require.Contains(t, out, `"yes"`)
	require.Contains(t, out, "( 30 )")
}
