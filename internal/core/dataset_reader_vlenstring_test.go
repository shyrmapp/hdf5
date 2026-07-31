package core

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

// heapAddr is where the synthetic GCOL collection starts. It must be non-zero:
// readVariableString treats address 0 as a null string reference.
const heapAddr = 64

// buildGlobalHeapWithString lays out a GCOL collection at heapAddr holding a
// single object (ID 1) containing s, and returns the backing bytes.
func buildGlobalHeapWithString(s string) []byte {
	const size = 512
	data := make([]byte, heapAddr+size)
	c := data[heapAddr:]

	copy(c[0:4], "GCOL")
	c[4] = 1 // version
	binary.LittleEndian.PutUint64(c[8:16], size)

	// Object header: ID 1, 0 refs, 4 reserved, then the 8-byte length.
	binary.LittleEndian.PutUint16(c[16:18], 1)
	binary.LittleEndian.PutUint16(c[18:20], 0)
	binary.LittleEndian.PutUint64(c[24:32], uint64(len(s)))
	copy(c[32:], s)

	return data
}

// vlenRef builds an on-disk vlen element: a 4-byte length, then the global heap
// ID (heap address of offsetSize bytes, then a 4-byte object index).
//
// The length prefix matters — a reader that starts at offset 0 folds it into
// the heap address and chases a garbage pointer.
func vlenRef(heapAddr uint64, objIndex uint32, offsetSize int, length uint32) []byte {
	ref := make([]byte, 4+offsetSize+4)
	binary.LittleEndian.PutUint32(ref[0:4], length)
	if offsetSize == 8 {
		binary.LittleEndian.PutUint64(ref[4:12], heapAddr)
	} else {
		binary.LittleEndian.PutUint32(ref[4:8], uint32(heapAddr))
	}
	binary.LittleEndian.PutUint32(ref[4+offsetSize:], objIndex)
	return ref
}

// TestReadVariableString covers the compound vlen-string member path: the record
// holds only a global heap reference, and the value must be fetched from the heap.
func TestReadVariableString(t *testing.T) {
	const want = "north station"
	heap := buildGlobalHeapWithString(want)
	sb := &Superblock{OffsetSize: 8}

	got, err := readVariableString(bytes.NewReader(heap), vlenRef(heapAddr, 1, 8, uint32(len(want))), sb)
	require.NoError(t, err)
	require.Equal(t, want, got)
}

// TestReadVariableString_NullReference checks that a zero heap address decodes to
// the empty string rather than an error: HDF5 writes address 0 for null strings.
func TestReadVariableString_NullReference(t *testing.T) {
	sb := &Superblock{OffsetSize: 8}

	got, err := readVariableString(bytes.NewReader(make([]byte, 64)), vlenRef(0, 0, 8, 0), sb)
	require.NoError(t, err)
	require.Empty(t, got)
}

// TestReadVariableString_Errors checks the two failure modes that a malformed
// file can trigger: a truncated reference and a heap address pointing at
// something that is not a GCOL collection.
func TestReadVariableString_Errors(t *testing.T) {
	sb := &Superblock{OffsetSize: 8}

	t.Run("truncated element", func(t *testing.T) {
		_, err := readVariableString(bytes.NewReader(make([]byte, 64)), []byte{1, 2, 3}, sb)
		require.Error(t, err)
	})

	t.Run("heap address is not a collection", func(t *testing.T) {
		junk := make([]byte, heapAddr+256)
		copy(junk[heapAddr:heapAddr+4], "FAIL")
		_, err := readVariableString(bytes.NewReader(junk), vlenRef(heapAddr, 1, 8, 4), sb)
		require.Error(t, err)
	})
}
