package hdf5

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// bigDims is the official-corpus file that made this limit necessary: 6 KiB on
// disk, declaring a single chunked dataset of 4294967306 int8 elements. Read()
// widens int8 to float64, so it would allocate ~34 GiB.
const bigDims = "testdata/hdf5_official/tbigdims.h5"

// TestReadLimitRejectsOversizedDataset is the regression for a fatal OOM. Go
// reports allocation failure as "fatal error: runtime: out of memory", which
// kills the process and cannot be recovered, so the size must be rejected
// before the allocation happens. This must stay an error, never a crash.
func TestReadLimitRejectsOversizedDataset(t *testing.T) {
	f, err := Open(bigDims)
	require.NoError(t, err)
	defer f.Close()

	obj, err := f.Resolve("/dset4gb")
	require.NoError(t, err)
	ds, ok := obj.(*Dataset)
	require.True(t, ok)

	_, err = ds.Read()
	require.Error(t, err, "a 6KiB file must not be able to allocate 34GiB")

	// The message has to tell the caller what to do instead.
	require.Contains(t, err.Error(), "ReadSlice")
	require.Contains(t, err.Error(), "WithMaxReadBytes")
}

// TestWithMaxReadBytesRaisesLimit checks the option actually moves the ceiling.
// It is still set far below this dataset's size, so the read must still fail —
// asserting the limit is honored rather than ignored once set.
func TestWithMaxReadBytesRaisesLimit(t *testing.T) {
	f, err := Open(bigDims, WithMaxReadBytes(4<<30))
	require.NoError(t, err)
	defer f.Close()

	obj, err := f.Resolve("/dset4gb")
	require.NoError(t, err)

	_, err = obj.(*Dataset).Read()
	require.Error(t, err)
	require.Contains(t, err.Error(), "4294967296 byte limit",
		"the raised limit should appear in the error, not the default")
}

// TestReadLimitAllowsNormalDatasets guards against the limit being so eager it
// breaks ordinary reads.
func TestReadLimitAllowsNormalDatasets(t *testing.T) {
	f, err := Open(filepath.Join("testdata", "simple_float64.h5"))
	require.NoError(t, err)
	defer f.Close()

	var read int
	f.Walk(func(_ string, obj Object) {
		ds, ok := obj.(*Dataset)
		if !ok {
			return
		}
		if vals, err := ds.Read(); err == nil {
			read += len(vals)
		}
	})
	require.Positive(t, read, "normal datasets must still read")
}

// TestDefaultReadLimitIsOneGiB pins the documented default, which SECURITY.md
// and the WithMaxReadBytes doc comment both quote.
func TestDefaultReadLimitIsOneGiB(t *testing.T) {
	require.Equal(t, uint64(1<<30), uint64(core.DefaultMaxReadBytes))

	f, err := Open(bigDims)
	require.NoError(t, err)
	defer f.Close()

	obj, _ := f.Resolve("/dset4gb")
	_, err = obj.(*Dataset).Read()
	require.Error(t, err)
	require.True(t, strings.Contains(err.Error(), "1073741824 byte limit"),
		"default limit should be reported as 1 GiB, got: %v", err)
}
