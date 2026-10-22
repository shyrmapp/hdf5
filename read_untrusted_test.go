package hdf5

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

const chunksCases = "testdata/c_interop/chunks_cases.h5"

// allocated returns the bytes fn allocated in total (not live), so a reader
// that builds a huge temporary and drops it still shows up.
func allocated(fn func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// patchedCopy writes data to a temp file and returns its path.
func patchedCopy(t *testing.T, data []byte) string {
	t.Helper()
	path := t.TempDir() + "/patched.h5"
	require.NoError(t, os.WriteFile(path, data, 0o600))
	return path
}

// TestHostileChunkDims patches the chunk dimensions of a C-written layout-v3
// file: a zero dimension used to divide by zero in ReadSlice, a missing
// element-size dimension to slice out of range in Read and index out of range
// in ReadSlice. Each must be an error, never a panic.
func TestHostileChunkDims(t *testing.T) {
	orig, err := os.ReadFile(fillV3)
	require.NoError(t, err)

	// layout v3 chunked: 3, 2, ndims, btree addr (8), then 32-bit dims.
	patch := func(ndims byte, dims []byte, edit func(msg []byte)) []byte {
		data := bytes.Clone(orig)
		head := []byte{3, 2, ndims}
		n := 0
		for i := 0; i+11+len(dims) <= len(data); i++ {
			if bytes.Equal(data[i:i+3], head) && bytes.Equal(data[i+11:i+11+len(dims)], dims) {
				edit(data[i : i+11+len(dims)])
				n++
			}
		}
		require.Equal(t, 1, n)
		return data
	}
	le32s := func(v ...uint32) []byte {
		var b []byte
		for _, x := range v {
			b = binary.LittleEndian.AppendUint32(b, x)
		}
		return b
	}

	cases := map[string]struct {
		name string
		data []byte
		sel  [][]uint64 // start, count
	}{
		// chunk1d: dims {30}, chunk {7} + element size 4 -> chunk {0}.
		"zero": {"chunk1d", patch(2, le32s(7, 4), func(m []byte) { copy(m[11:], le32s(0, 4)) }), [][]uint64{{0}, {30}}},
		// chunk2d: dims {5,6}, chunk {2,4} + 4 -> drop the trailing dimension.
		"rank": {"chunk2d", patch(3, le32s(2, 4, 4), func(m []byte) { m[2] = 2 }), [][]uint64{{0, 0}, {5, 6}}},
	}
	for label, c := range cases {
		t.Run(label, func(t *testing.T) {
			ds := openDS(t, patchedCopy(t, c.data), c.name)
			require.NotPanics(t, func() {
				_, err := ds.Read()
				require.Error(t, err)
				_, err = ds.ReadSlice(c.sel[0], c.sel[1])
				require.Error(t, err)
				_, err = ds.ChunkIterator()
				require.Error(t, err)
			})
		})
	}
}

// blobDataset is a Dataset over an in-memory file, for exercising the
// hyperslab reader with hand-built layout messages.
func blobDataset(blob []byte, maxRead uint64) *Dataset {
	sb := &core.Superblock{Version: 2, OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}
	return &Dataset{file: &File{reader: bytes.NewReader(blob), sb: sb, maxReadBytes: maxRead}}
}

func floatType() *core.DatatypeMessage {
	return &core.DatatypeMessage{Class: core.DatatypeFloat, Size: 8}
}

func fullSel(n uint64) *HyperslabSelection {
	return &HyperslabSelection{Start: []uint64{0}, Count: []uint64{n}, Stride: []uint64{1}, Block: []uint64{1}}
}

// TestImplicitIndexZeroElementSize is a ~100-byte file whose implicit chunk
// index has a zero element-size dimension: every chunk then "fits" in the file
// and ReadSlice used to build one entry per declared chunk.
func TestImplicitIndexZeroElementSize(t *testing.T) {
	const n = 1 << 22
	layout := &core.DataLayoutMessage{
		Version: 4, Class: core.LayoutChunked, DataAddress: 8,
		ChunkSize: []uint64{1, 0}, ChunkIndexType: core.ChunkIndexImplicit,
	}
	ds := blobDataset(make([]byte, 100), 0)
	var err error
	got := allocated(func() {
		_, err = ds.readHyperslabChunked(fullSel(n), floatType(), &core.DataspaceMessage{Dimensions: []uint64{n}}, layout, nil, nil)
	})
	require.Error(t, err)
	require.Less(t, got, uint64(1<<20))

	// A well-formed implicit index larger than the file is refused too.
	layout.ChunkSize = []uint64{1, 8}
	_, err = ds.readHyperslabChunked(fullSel(n), floatType(), &core.DataspaceMessage{Dimensions: []uint64{n}}, layout, nil, nil)
	require.Error(t, err)
}

// TestSparseSliceNoCoordinateGrid reads all of a 4M-element dataset with 1
// element chunks of which only one is stored. ReadSlice used to materialize a
// coordinate slice for every one of the 4M grid cells (~250 MiB) before
// looking anything up.
func TestSparseSliceNoCoordinateGrid(t *testing.T) {
	const n = 1 << 22
	blob := make([]byte, 64)
	binary.LittleEndian.PutUint64(blob[16:], math.Float64bits(42))
	layout := &core.DataLayoutMessage{
		Version: 4, Class: core.LayoutChunked, DataAddress: 16,
		ChunkSize: []uint64{1, 8}, ChunkIndexType: core.ChunkIndexSingle,
	}
	ds := blobDataset(blob, 32<<20)
	var out interface{}
	var err error
	got := allocated(func() {
		out, err = ds.readHyperslabChunked(fullSel(n), floatType(), &core.DataspaceMessage{Dimensions: []uint64{n}}, layout, nil, nil)
	})
	require.NoError(t, err)
	vals := out.([]float64)
	require.Len(t, vals, n)
	require.Equal(t, 42.0, vals[0])
	require.Equal(t, 0.0, vals[n-1])
	// Output buffer + float64 result = 64 MiB; the coordinate grid added 250+.
	require.Less(t, got, uint64(100<<20))
}

func zlibBytes(t *testing.T, raw []byte) []byte {
	t.Helper()
	var b bytes.Buffer
	w := zlib.NewWriter(&b)
	_, err := w.Write(raw)
	require.NoError(t, err)
	require.NoError(t, w.Close())
	return b.Bytes()
}

// TestHyperslabChunkSizeBounds covers the decode bound and short chunks of the
// hyperslab path, which must match Read(): a chunk decompresses to exactly
// elemSize x rank-chunk-dims bytes (not also multiplied by the element-size
// pseudo-dimension), and a chunk that decodes short is an error.
func TestHyperslabChunkSizeBounds(t *testing.T) {
	deflate, err := core.ParseFilterPipelineMessage([]byte{2, 1, 1, 0, 0, 0, 1, 0, 6, 0, 0, 0})
	require.NoError(t, err)
	space := &core.DataspaceMessage{Dimensions: []uint64{4}}

	read := func(stored []byte) error {
		blob := append(make([]byte, 16), stored...)
		layout := &core.DataLayoutMessage{
			Version: 4, Class: core.LayoutChunked, DataAddress: 16,
			Flags:     core.LayoutChunkSingleIndexWithFilter,
			ChunkSize: []uint64{4, 8}, ChunkIndexType: core.ChunkIndexSingle,
			SingleChunkNBytes: uint64(len(stored)),
		}
		_, err := blobDataset(blob, 0).readHyperslabChunked(fullSel(4), floatType(), space, layout, deflate, nil)
		return err
	}

	require.NoError(t, read(zlibBytes(t, make([]byte, 32))), "exact chunk")
	require.Error(t, read(zlibBytes(t, make([]byte, 256))), "8x the chunk size must be refused")
	require.Error(t, read(zlibBytes(t, make([]byte, 16))), "short chunk must not read as partial data")
}

// TestPartialChunksStoredUnfiltered reads C-written datasets (layout v4,
// H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS) whose edge chunk is stored raw. The
// hyperslab path used to run the pipeline over it anyway: an error with
// deflate, silently wrong values with shuffle alone.
func TestPartialChunksStoredUnfiltered(t *testing.T) {
	for _, name := range []string{"partial_shuffle_deflate", "partial_shuffle"} {
		t.Run(name, func(t *testing.T) {
			ds := openDS(t, chunksCases, name)
			all, err := ds.Read()
			require.NoError(t, err)
			require.Equal(t, seq(0, 10), all)

			for _, s := range [][2]uint64{{8, 2}, {0, 10}, {5, 5}} {
				got, err := ds.ReadSlice([]uint64{s[0]}, []uint64{s[1]})
				require.NoError(t, err, s)
				require.Equal(t, seq(int(s[0]), int(s[0]+s[1])), got, s)
			}
		})
	}
}

// fletcherPayload locates the stored bytes of want in a file so a test can
// corrupt them.
func fletcherPayload(t *testing.T, data, want []byte) int {
	t.Helper()
	i := bytes.Index(data, want)
	require.GreaterOrEqual(t, i, 0)
	return i
}

// TestFletcher32Verified reads C-written Fletcher32 datasets (even and odd
// payload lengths) and checks a flipped stored byte is reported instead of
// the checksum being stripped unread.
func TestFletcher32Verified(t *testing.T) {
	orig, err := os.ReadFile(chunksCases)
	require.NoError(t, err)

	want := map[string][]float64{"fletcher32": seq(1, 9), "fletcher32_odd": seq(1, 8)}
	for name, vals := range want {
		got, err := openDS(t, chunksCases, name).Read()
		require.NoError(t, err, name)
		require.Equal(t, vals, got, name)
	}

	var raw8 []byte
	for i := 1; i <= 8; i++ {
		raw8 = binary.LittleEndian.AppendUint64(raw8, math.Float64bits(float64(i)))
	}
	for name, payload := range map[string][]byte{"fletcher32": raw8, "fletcher32_odd": {1, 2, 3, 4, 5, 6, 7}} {
		t.Run(name, func(t *testing.T) {
			data := bytes.Clone(orig)
			data[fletcherPayload(t, data, payload)+3] ^= 0x10
			ds := openDS(t, patchedCopy(t, data), name)

			_, err := ds.Read()
			require.ErrorContains(t, err, "fletcher32")
			_, err = ds.ReadSlice([]uint64{0}, []uint64{uint64(len(want[name]))})
			require.ErrorContains(t, err, "fletcher32")
		})
	}
}
