package hdf5

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestGroupDeleteDenseAttribute covers deleteAttribute -> deleteDenseAttributeFromHeader,
// the uncached delete path. GroupWriter.DeleteAttribute always takes it (unlike
// DatasetWriter, which short-circuits to the cached-header variant), so a group
// with enough attributes to spill into dense storage is the way in.
func TestGroupDeleteDenseAttribute(t *testing.T) {
	filename := filepath.Join(t.TempDir(), "dense_group_attrs.h5")

	fw, err := CreateForWrite(filename, CreateTruncate)
	require.NoError(t, err)

	g, err := fw.CreateGroup("/measurements")
	require.NoError(t, err)

	// 10 attributes: compact storage transitions to dense at 8.
	for i := range 10 {
		require.NoError(t, g.WriteAttribute(fmt.Sprintf("attr%d", i), int32(i*10)))
	}

	// GroupWriter.DeleteAttribute re-reads the object header from the file, so
	// this takes the uncached path whether or not the file is reopened first.
	require.NoError(t, g.DeleteAttribute("attr4"))
	require.NoError(t, fw.Close())

	f, err := Open(filename)
	require.NoError(t, err)
	defer f.Close()

	obj, err := f.Resolve("/measurements")
	require.NoError(t, err)
	grp, ok := obj.(*Group)
	require.True(t, ok, "/measurements should be a group")

	attrs, err := grp.Attributes()
	require.NoError(t, err)

	names := make(map[string]bool, len(attrs))
	for _, a := range attrs {
		names[a.Name] = true
	}
	require.False(t, names["attr4"], "deleted attribute must be gone")
	require.Len(t, names, 9, "the other nine must survive")
}
