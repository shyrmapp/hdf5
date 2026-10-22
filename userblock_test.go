package hdf5

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// fileSnapshot reads everything the read API exposes so two files can be compared.
func fileSnapshot(t *testing.T, path string) map[string]string {
	t.Helper()
	f, err := Open(path)
	require.NoError(t, err)
	defer func() { require.NoError(t, f.Close()) }()

	snap := map[string]string{}
	f.Walk(func(p string, obj Object) {
		var attrs []*core.Attribute
		var aerr error
		switch o := obj.(type) {
		case *Group:
			attrs, aerr = o.Attributes()
		case *Dataset:
			attrs, aerr = o.Attributes()
			vals, rerr := o.Read()
			snap[p+"#read"] = fmt.Sprint(vals, rerr)

			header, err := core.ReadObjectHeader(f.Reader(), o.Address(), f.Superblock())
			require.NoError(t, err)
			for _, msg := range header.Messages {
				if msg.Type != core.MsgDataspace {
					continue
				}
				space, err := core.ParseDataspaceMessage(msg.Data)
				require.NoError(t, err)
				start := make([]uint64, len(space.Dimensions))
				count := make([]uint64, len(space.Dimensions))
				for i, d := range space.Dimensions {
					start[i] = d / 3
					count[i] = d - d/3 - d/4
				}
				slice, serr := o.ReadSlice(start, count)
				snap[p+"#slice"] = fmt.Sprint(slice, serr)
			}
		}
		snap[p+"#attrerr"] = fmt.Sprint(aerr)
		for _, a := range attrs {
			v, err := a.ReadValue()
			snap[p+"@"+a.Name] = fmt.Sprint(v, err)
		}
	})
	return snap
}

// TestUserBlockMatchesPlainFile checks that a file wrapped in a user block reads
// identically to the original: objects, data, slices and attribute values, not
// just "no error". Fixtures were made with `h5repack -b <size> <src> ub<size>_<src>`.
func TestUserBlockMatchesPlainFile(t *testing.T) {
	for _, src := range []string{"fill_v3.h5", "attrs_20_corder.h5"} {
		want := fileSnapshot(t, filepath.Join("testdata", "c_interop", src))
		require.NotEmpty(t, want)
		for _, size := range []int{512, 4096} {
			t.Run(fmt.Sprintf("%s/ub%d", src, size), func(t *testing.T) {
				path := filepath.Join("testdata", "c_interop", fmt.Sprintf("ub%d_%s", size, src))
				f, err := Open(path)
				require.NoError(t, err)
				require.Equal(t, uint64(size), f.Superblock().BaseAddress)
				require.NoError(t, f.Close())

				require.Equal(t, want, fileSnapshot(t, path))
			})
		}
	}
}

func TestOpenForWriteRefusesUserBlock(t *testing.T) {
	_, err := OpenForWrite("testdata/c_interop/ub512_fill_v3.h5", OpenReadOnly)
	require.ErrorContains(t, err, "user block")
}
