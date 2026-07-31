package core

import (
	"bytes"
	"compress/zlib"
	"testing"

	"github.com/shyrmapp/hdf5/internal/utils"
	"github.com/stretchr/testify/require"
)

// zlibBomb returns a tiny zlib stream that expands to size bytes of zeros.
func zlibBomb(t *testing.T, size int) []byte {
	t.Helper()

	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	_, err := w.Write(make([]byte, size))
	require.NoError(t, err)
	require.NoError(t, w.Close())
	return buf.Bytes()
}

// TestDeflateBombRejected is the compression-bomb guard. A chunk is fixed-size
// in HDF5, so a stream that expands past the chunk size is malformed however
// well-formed the zlib framing is. Without the bound, io.ReadAll would happily
// buffer the whole expansion.
func TestDeflateBombRejected(t *testing.T) {
	const chunkBytes = 64 << 10
	bomb := zlibBomb(t, 64<<20) // 64 MiB from a few KiB

	require.Less(t, len(bomb), chunkBytes,
		"the bomb must be smaller than the chunk it claims to be")

	_, err := applyDeflate(bomb, chunkBytes)
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceeds")
}

// TestDeflateExactChunkSizeAccepted pins the boundary: expanding to exactly the
// chunk size is legitimate and must not be rejected by an off-by-one.
func TestDeflateExactChunkSizeAccepted(t *testing.T) {
	const chunkBytes = 64 << 10

	out, err := applyDeflate(zlibBomb(t, chunkBytes), chunkBytes)
	require.NoError(t, err)
	require.Len(t, out, chunkBytes)
}

// TestDeflateUnknownLimitFallsBackToMaxChunkSize checks that a caller passing 0
// gets MaxChunkSize rather than "unbounded".
func TestDeflateUnknownLimitFallsBackToMaxChunkSize(t *testing.T) {
	out, err := applyDeflate(zlibBomb(t, 1<<20), 0)
	require.NoError(t, err, "1 MiB is under MaxChunkSize, so it should pass")
	require.Len(t, out, 1<<20)

	// readAllBounded is where 0 is normalized; confirm the ceiling it picks.
	_, err = readAllBounded(bytes.NewReader(make([]byte, 8)), 0)
	require.NoError(t, err)
	require.Equal(t, uint64(1<<30), uint64(utils.MaxChunkSize))
}

// TestLZFBombRejected covers the hand-rolled decompressor, which builds its
// output in a loop rather than through io.Reader, so it needs its own guard:
// each backreference can emit up to 264 bytes.
func TestLZFBombRejected(t *testing.T) {
	// A literal byte, then repeated long backreferences that each copy from
	// the growing output — classic LZF expansion.
	bomb := make([]byte, 0, 2+2000*3)
	bomb = append(bomb, 0x00, 0x41) // literal run of 1: "A"
	for range 2000 {
		// long backref: 111OXXXX XXXXXXXX RRRRRRRR -> offset 1, length 264
		bomb = append(bomb, 0xE0, 0x00, 0xFF)
	}

	_, err := lzfDecompress(bomb, 1024)
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceeds")
}

// TestPipelineBombRejectedEndToEnd drives the bomb through the public entry
// point, with the chunk size the reader would pass.
func TestPipelineBombRejectedEndToEnd(t *testing.T) {
	fp := &FilterPipelineMessage{
		Version:    1,
		NumFilters: 1,
		Filters:    []Filter{{ID: FilterDeflate}},
	}

	_, err := fp.ApplyFilters(zlibBomb(t, 64<<20), 0, 4096)
	require.Error(t, err)
	require.Contains(t, err.Error(), "exceeds")
}
