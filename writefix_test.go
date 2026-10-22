package hdf5

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOpenDatasetChunkedRefusesDataWrite(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/c", Int32, []uint64{10}, WithChunkDims([]uint64{5}))
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}))
	require.NoError(t, fw.Close())

	fw, err = OpenForWrite(file, OpenReadWrite)
	require.NoError(t, err)
	ds, err = fw.OpenDataset("/c")
	require.NoError(t, err)
	require.Error(t, ds.Write([]int32{9, 9, 9, 9, 9, 9, 9, 9, 9, 9}))
	require.Error(t, ds.WriteRaw(make([]byte, 40)))
	require.NoError(t, ds.WriteAttribute("still", "fine"))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, "(0): 0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
}

func TestUseAfterCloseReturnsError(t *testing.T) {
	fw, _ := newWriteFile(t)
	ds, err := fw.CreateDataset("/d", Int32, []uint64{1})
	require.NoError(t, err)
	g, err := fw.CreateGroup("/g")
	require.NoError(t, err)
	require.NoError(t, fw.Close())

	require.Error(t, ds.Write([]int32{1}))
	require.Error(t, ds.WriteAttribute("a", int32(1)))
	require.Error(t, g.WriteAttribute("a", int32(1)))
	_, err = fw.CreateDataset("/e", Int32, []uint64{1})
	require.Error(t, err)
	_, err = fw.CreateGroup("/h")
	require.Error(t, err)
	require.Error(t, fw.Delete("/d"))
	require.NoError(t, fw.Close())
}

func TestStaleWriterAfterDelete(t *testing.T) {
	fw, file := newWriteFile(t)
	old, err := fw.CreateDataset("/old", Int32, []uint64{4})
	require.NoError(t, err)
	require.NoError(t, old.Write([]int32{1, 1, 1, 1}))
	require.NoError(t, fw.Delete("/old"))
	repl, err := fw.CreateDataset("/new", Int32, []uint64{4})
	require.NoError(t, err)
	require.NoError(t, repl.Write([]int32{7, 7, 7, 7}))

	require.Error(t, old.Write([]int32{2, 2, 2, 2}))
	require.Error(t, old.WriteAttribute("x", int32(1)))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, "(0): 7, 7, 7, 7")
}

func TestDuplicateAndInvalidNames(t *testing.T) {
	fw, file := newWriteFile(t)
	_, err := fw.CreateDataset("/dup", Int32, []uint64{1})
	require.NoError(t, err)
	_, err = fw.CreateDataset("/dup", Int32, []uint64{1})
	require.ErrorContains(t, err, "already exists")

	g, err := fw.CreateGroup("/g")
	require.NoError(t, err)
	require.NoError(t, g.WriteAttribute("a", int32(1)))
	child, err := fw.CreateDataset("/g/child", Int32, []uint64{1})
	require.NoError(t, err)
	require.NoError(t, child.Write([]int32{5}))
	_, err = fw.CreateGroup("/g")
	require.ErrorContains(t, err, "already exists")
	_, err = fw.CreateDataset("/g/child2", Int32, []uint64{1}) // /g must still be intact
	require.NoError(t, err)

	_, err = fw.CreateDataset("/dup", Int32, []uint64{1})
	require.Error(t, err)
	_, err = fw.CreateDataset("/", Int32, []uint64{1})
	require.Error(t, err)
	_, err = fw.CreateDataset("/trail/", Int32, []uint64{1})
	require.Error(t, err)
	_, err = fw.CreateGroup("/trailg/")
	require.Error(t, err)
	require.Error(t, fw.CreateSoftLink("/dup", "/g"))
	require.NoError(t, fw.Close())

	out := h5dumpClean(t, file)
	require.Contains(t, out, `DATASET "child"`)
	require.Contains(t, out, `DATASET "child2"`)
	require.Equal(t, 1, strings.Count(out, `DATASET "dup"`))
}

func TestHugeAttributeRejected(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/d", Int32, []uint64{1})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{3}))
	require.Error(t, ds.WriteAttribute("big", make([]byte, 70000)))
	require.Error(t, ds.WriteAttribute("bigs", strings.Repeat("x", 70000)))
	require.NoError(t, ds.WriteAttribute("ok", "fine"))
	require.NoError(t, fw.Close())
	out := h5dumpClean(t, file)
	require.Contains(t, out, `"fine"`)
}

func TestOpenForWriteReadOnlyCloses(t *testing.T) {
	fw, file := newWriteFile(t)
	ds, err := fw.CreateDataset("/d", Int32, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{1, 2}))
	require.NoError(t, fw.Close())

	ro, err := OpenForWrite(file, OpenReadOnly)
	require.NoError(t, err)
	require.NoError(t, ro.Close())
	require.NoError(t, ro.Close(), "second Close is a no-op")
	_, err = ro.CreateDataset("/x", Int32, []uint64{1})
	require.Error(t, err)
}

func TestCreateDenseGroupRejectsBlankNames(t *testing.T) {
	fw, _ := newWriteFile(t)
	_, err := fw.CreateDataset("/d", Int32, []uint64{1})
	require.NoError(t, err)
	require.Error(t, fw.CreateDenseGroup("/", map[string]string{"a": "/d"}))
	require.Error(t, fw.CreateDenseGroup("/g/", map[string]string{"a": "/d"}))
	require.Error(t, fw.CreateDenseGroup("/g", map[string]string{"": "/d"}))
	require.NoError(t, fw.Close())
}
