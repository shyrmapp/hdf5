package hdf5

// Dense link / dense attribute storage and group aliasing, checked against
// fixtures produced by the HDF5 C library (testdata/c_interop/gen_dense.c).
// Expected contents mirror the generator: links "l%03d", attributes "a%03d"
// with int32 value i*10.

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

func openFixture(t *testing.T, name string) *File {
	t.Helper()
	f, err := Open(filepath.Join("testdata", "c_interop", name))
	require.NoError(t, err)
	t.Cleanup(func() { _ = f.Close() })
	return f
}

func childNames(g *Group) []string {
	names := make([]string, 0, len(g.Children()))
	for _, c := range g.Children() {
		names = append(names, c.Name())
	}
	sort.Strings(names)
	return names
}

func childGroup(t *testing.T, g *Group, name string) *Group {
	t.Helper()
	for _, c := range g.Children() {
		if c.Name() == name {
			cg, ok := c.(*Group)
			require.True(t, ok, "%s is not a group", name)
			return cg
		}
	}
	t.Fatalf("child %q not found", name)
	return nil
}

func TestDenseLinks_CFixtures(t *testing.T) {
	for _, n := range []int{9, 30, 100, 300} {
		t.Run(fmt.Sprintf("%d", n), func(t *testing.T) {
			f := openFixture(t, fmt.Sprintf("links_%d.h5", n))
			g := childGroup(t, f.Root(), "g")
			want := make([]string, n)
			for i := range want {
				want[i] = fmt.Sprintf("l%03d", i)
			}
			require.Equal(t, want, childNames(g))
		})
	}
}

// requireAttrNames checks that attrs are exactly a000..a(n-1).
func requireAttrNames(t *testing.T, attrs []*core.Attribute, n int) {
	t.Helper()
	got := names(attrs)
	sort.Strings(got)
	require.Len(t, got, n)
	for i, name := range got {
		require.Equal(t, fmt.Sprintf("a%03d", i), name)
	}
}

func TestDenseAttributes_CFixtures(t *testing.T) {
	for _, n := range []int{20, 300} {
		for _, order := range []string{"plain", "corder"} {
			t.Run(fmt.Sprintf("%d_%s", n, order), func(t *testing.T) {
				f := openFixture(t, fmt.Sprintf("attrs_%d_%s.h5", n, order))
				g := childGroup(t, f.Root(), "g")
				d := childDataset(t, f.Root(), "d")

				gattrs, err := g.Attributes()
				require.NoError(t, err)
				requireAttrNames(t, gattrs, n)

				dattrs, err := d.Attributes()
				require.NoError(t, err)
				requireAttrNames(t, dattrs, n)

				for i := 0; i < n; i += 37 {
					v, err := d.ReadAttribute(fmt.Sprintf("a%03d", i))
					require.NoError(t, err)
					require.EqualValues(t, i*10, v)
				}
			})
		}
	}
}

func childDataset(t *testing.T, g *Group, name string) *Dataset {
	t.Helper()
	for _, c := range g.Children() {
		if ds, ok := c.(*Dataset); ok && ds.Name() == name {
			return ds
		}
	}
	t.Fatalf("dataset %q not found", name)
	return nil
}

func names(attrs []*core.Attribute) []string {
	out := make([]string, 0, len(attrs))
	for _, x := range attrs {
		out = append(out, x.Name)
	}
	return out
}

// An aliased group (hard-linked under two names) must list its children on
// every path, not only the first one visited.
func TestHardLinkAlias_CFixture(t *testing.T) {
	for _, name := range []string{"alias.h5", "alias_old.h5"} { // new-style and symbol-table groups
		t.Run(name, func(t *testing.T) {
			f := openFixture(t, name)
			for _, p := range []string{"a", "b"} {
				g := childGroup(t, f.Root(), p)
				require.Equal(t, []string{"x"}, childNames(g), "children of /%s", p)
			}
		})
	}
}

// A child that cannot be loaded must fail the open instead of silently
// dropping the child (and everything below it).
func TestChildLoadError_NotSwallowed(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "c_interop", "links_30.h5"))
	require.NoError(t, err)
	// Break the signature of the second object header (the dataset "d").
	first := bytes.Index(data, []byte("OHDR"))
	second := first + 4 + bytes.Index(data[first+4:], []byte("OHDR"))
	require.Greater(t, second, first+4)
	copy(data[second:], "XXXX")
	path := filepath.Join(t.TempDir(), "broken.h5")
	require.NoError(t, os.WriteFile(path, data, 0o600))

	f, err := Open(path)
	if err == nil {
		_ = f.Close()
	}
	require.Error(t, err)
}

// Dense attributes written by this library must be readable by the C tools.
func TestCInterop_DenseAttributes(t *testing.T) {
	h5dump := h5tool(t, "h5dump")
	path := filepath.Join(t.TempDir(), "dense_attrs.h5")

	fw, err := CreateForWrite(path, CreateTruncate)
	require.NoError(t, err)
	ds, err := fw.CreateDataset("/d", Int32, []uint64{2})
	require.NoError(t, err)
	require.NoError(t, ds.Write([]int32{1, 2}))
	const n = 12
	for i := 0; i < n; i++ {
		require.NoError(t, ds.WriteAttribute(fmt.Sprintf("attr%02d", i), int32(i*10)))
	}
	require.NoError(t, fw.Close())

	out := runH5(t, h5dump, "-A", path)
	for i := 0; i < n; i++ {
		require.Contains(t, out, fmt.Sprintf("ATTRIBUTE \"attr%02d\"", i))
		require.Contains(t, out, fmt.Sprintf("(0): %d", i*10))
	}
	require.False(t, strings.Contains(out, "error"), out)

	// Lookup by name goes through the B-tree's name hash.
	out = runH5(t, h5dump, "-a", "/d/attr05", path)
	require.Contains(t, out, "(0): 50")
}
