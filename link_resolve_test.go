package hdf5

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testWriteLinkFile writes a file with a dataset, a soft link to it, a
// dangling soft link, and a soft link cycle, then returns the file path.
func testWriteLinkFile(t *testing.T, dir string) string {
	t.Helper()
	filename := filepath.Join(dir, "links.h5")

	fw, err := CreateForWrite(filename, CreateTruncate)
	require.NoError(t, err)

	ds, err := fw.CreateDataset("/data", Float64, []uint64{3})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]float64{1.5, 2.5, 3.5}))

	require.NoError(t, fw.CreateSoftLink("/soft", "/data"))
	require.NoError(t, fw.CreateSoftLink("/dangling", "/missing"))
	require.NoError(t, fw.CreateSoftLink("/cycle_a", "/cycle_b"))
	require.NoError(t, fw.CreateSoftLink("/cycle_b", "/cycle_a"))

	require.NoError(t, fw.Close())
	return filename
}

// TestResolve_SoftLinks covers Walk surfacing and Resolve semantics for
// soft links written by this library.
func TestResolve_SoftLinks(t *testing.T) {
	filename := testWriteLinkFile(t, t.TempDir())

	f, err := Open(filename)
	require.NoError(t, err)
	defer f.Close()

	// Walk surfaces the soft links with their targets.
	links := map[string]string{}
	f.Walk(func(path string, obj Object) {
		if sl, ok := obj.(*SoftLink); ok {
			links[path] = sl.Target()
		}
	})
	assert.Equal(t, map[string]string{
		"/soft":     "/data",
		"/dangling": "/missing",
		"/cycle_a":  "/cycle_b",
		"/cycle_b":  "/cycle_a",
	}, links)

	// Resolve follows the soft link to the dataset.
	obj, err := f.Resolve("/soft")
	require.NoError(t, err)
	ds, ok := obj.(*Dataset)
	require.True(t, ok, "expected *Dataset, got %T", obj)
	vals, err := ds.Read()
	require.NoError(t, err)
	assert.Equal(t, []float64{1.5, 2.5, 3.5}, vals)

	// Direct paths still resolve.
	obj, err = f.Resolve("/data")
	require.NoError(t, err)
	assert.IsType(t, &Dataset{}, obj)

	// Root resolves to the root group.
	obj, err = f.Resolve("/")
	require.NoError(t, err)
	assert.IsType(t, &Group{}, obj)

	// Dangling soft link errors.
	_, err = f.Resolve("/dangling")
	require.ErrorContains(t, err, "not found")

	// Cycle errors instead of hanging.
	_, err = f.Resolve("/cycle_a")
	require.ErrorContains(t, err, "cycle")

	// Relative paths are rejected.
	_, err = f.Resolve("data")
	require.ErrorContains(t, err, "absolute")
}

// TestResolve_ExternalLink writes an external link to a second Go-written
// file and resolves it through ExternalLink.Resolve.
func TestResolve_ExternalLink(t *testing.T) {
	dir := t.TempDir()

	// Target file with a dataset.
	targetFile := filepath.Join(dir, "target.h5")
	fw, err := CreateForWrite(targetFile, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/remote", Int32, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{7, 9}))
	require.NoError(t, fw.Close())

	// Source file with the external link.
	sourceFile := filepath.Join(dir, "source.h5")
	fw, err = CreateForWrite(sourceFile, CreateTruncate)
	require.NoError(t, err)
	require.NoError(t, fw.CreateExternalLink("/ext", "target.h5", "/remote"))
	require.NoError(t, fw.Close())

	f, err := Open(sourceFile)
	require.NoError(t, err)
	defer f.Close()

	// Resolve stops at the external link.
	obj, err := f.Resolve("/ext")
	require.NoError(t, err)
	el, ok := obj.(*ExternalLink)
	require.True(t, ok, "expected *ExternalLink, got %T", obj)
	assert.Equal(t, "ext", el.Name())
	assert.Equal(t, "target.h5", el.FileName())
	assert.Equal(t, "/remote", el.ObjectPath())

	// Walk surfaces the external link.
	var walked *ExternalLink
	f.Walk(func(path string, obj Object) {
		if e, ok := obj.(*ExternalLink); ok && path == "/ext" {
			walked = e
		}
	})
	require.NotNil(t, walked, "Walk did not surface the external link")

	// ExternalLink.Resolve opens the target file and finds the dataset.
	ef, target, err := el.Resolve(dir)
	require.NoError(t, err)
	defer ef.Close()
	targetDS, ok := target.(*Dataset)
	require.True(t, ok, "expected *Dataset, got %T", target)
	vals, err := targetDS.Read()
	require.NoError(t, err)
	assert.Equal(t, []float64{7, 9}, vals)
}

// TestResolve_CFixture_SoftLinks resolves soft links in a C-written fixture
// file, when present.
func TestResolve_CFixture_SoftLinks(t *testing.T) {
	fixture := filepath.Join("testdata", "hdf5_official", "tsoftlinks.h5")
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("fixture %s not present", fixture)
	}

	f, err := Open(fixture)
	require.NoError(t, err)
	defer f.Close()

	softLinks := map[string]string{}
	f.Walk(func(path string, obj Object) {
		if sl, ok := obj.(*SoftLink); ok {
			softLinks[path] = sl.Target()
		}
	})
	require.NotEmpty(t, softLinks, "expected soft links in %s", fixture)
	assert.Equal(t, "/dset1", softLinks["/soft_dset1"])

	// A C-written soft link resolves to its target dataset.
	obj, err := f.Resolve("/soft_dset1")
	require.NoError(t, err)
	assert.IsType(t, &Dataset{}, obj)

	// A dangling C-written soft link ("not_yet" is relative) errors cleanly.
	_, err = f.Resolve("/soft_dangle")
	require.Error(t, err)

	// Every other soft link must either resolve or fail cleanly — never hang.
	for path, target := range softLinks {
		obj, err := f.Resolve(path)
		if err == nil {
			assert.NotNil(t, obj, "resolved %s (-> %s) to nil", path, target)
		}
	}
}
