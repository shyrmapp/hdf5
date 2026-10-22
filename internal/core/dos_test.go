package core

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// allocDelta reports the bytes allocated while fn runs.
func allocDelta(fn func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// denseBlob builds a link-name B-tree (nrec records, all pointing at one
// 64 KiB direct block) plus a fractal heap header with the given doubling table.
// Layout: BTHD@0, FRHP@64, BTLF@256, FHDB@0x10000.
func denseBlob(nrec int, width, startBlock, maxDirect uint64, curRows uint16) []byte {
	const (
		heapAddr  = 64
		leafAddr  = 256
		blockAddr = 0x10000
		recSize   = 11
		objOff    = 17
		objLen    = 65000
	)
	nodeSize := uint32(6 + recSize*nrec + 4 + 64)
	bthd := withChecksum(cat([]byte("BTHD"), []byte{0, 5}, le32(nodeSize), le16(recSize), le16(0), []byte{100, 40},
		le64(leafAddr), le16(uint16(nrec)), le64(uint64(nrec))))
	rec := cat(le32(0), []byte{0}, le32(objOff), le16(objLen)) // hash, managed heap ID: offset(4) length(2)
	btlf := withChecksum(cat([]byte("BTLF"), []byte{0, 5}, bytes.Repeat(rec, nrec)))
	frhp := withChecksum(cat([]byte("FRHP"), []byte{0}, le16(7), le16(0), []byte{0}, le32(65535),
		le64(0), le64(0), le64(0), le64(0), le64(0), le64(0), le64(0), le64(0), // 8 length-sized counters
		le64(0), le64(0), le64(0), le64(0), // addresses and remaining counters
		le16(uint16(width)), le64(startBlock), le64(maxDirect), le16(32), le16(0), le64(blockAddr), le16(curRows)))
	fhdb := cat([]byte("FHDB"), []byte{0}, le64(heapAddr), le32(0))
	buf := make([]byte, blockAddr+65536)
	copy(buf, bthd)
	copy(buf[heapAddr:], frhp)
	copy(buf[leafAddr:], btlf)
	copy(buf[blockAddr:], fhdb)
	return buf
}

func TestDenseHeapSingleBlockSane(t *testing.T) {
	buf := denseBlob(2, 4, 65536, 65536, 0)
	objs, err := ReadDenseHeapObjects(bytes.NewReader(buf), 0, 64, hostileSB())
	require.NoError(t, err)
	require.Len(t, objs, 2)
}

// N records aimed at one block used to be copied N times before parsing.
func TestDenseHeapSharedBlockNotAmplified(t *testing.T) {
	buf := denseBlob(5000, 4, 65536, 65536, 0)
	var err error
	alloc := allocDelta(func() {
		err = guard(t, func() error {
			_, e := ReadDenseHeapObjects(bytes.NewReader(buf), 0, 64, hostileSB())
			return e
		})
	})
	require.ErrorContains(t, err, "exceeding file size")
	require.Less(t, alloc, uint64(16<<20), "allocated %d bytes for a %d byte file", alloc, len(buf))
}

// width=2 with startBlock=2^63 wraps width*startBlock to 0.
func TestFractalHeapOverflowingTableRejected(t *testing.T) {
	buf := denseBlob(1, 2, 1<<63, 1<<63, 1)
	err := guard(t, func() error {
		_, e := ReadDenseHeapObjects(bytes.NewReader(buf), 0, 64, hostileSB())
		return e
	})
	require.ErrorContains(t, err, "doubling table")
}

// Each v1 continuation block points 24 bytes further in, so every block
// overlaps the next; this used to be re-read ~1000 times.
func TestOverlappingContinuationsRejected(t *testing.T) {
	const size = 400 << 10
	buf := make([]byte, size)
	copy(buf, ohdrV1(contMsgV1(40, size-40)))
	nilMsg := cat(le16(0), le16(8), []byte{0, 0, 0, 0}, make([]byte, 8)) // NIL message with 8 data bytes
	for off, blk := 40, uint64(size-40); off+24 < size; off, blk = off+24, blk-24 {
		copy(buf[off:], contMsgV1(uint64(off+24), blk-24))
		for p := off + 24; p+len(nilMsg) <= size; p += len(nilMsg) {
			copy(buf[p:], nilMsg)
		}
		if blk < 48 {
			break
		}
	}
	var err error
	alloc := allocDelta(func() {
		err = guard(t, func() error {
			_, _, _, e := parseV1Header(bytes.NewReader(buf), 0, hostileSB())
			return e
		})
	})
	require.ErrorContains(t, err, "exceeding file size")
	require.Less(t, alloc, uint64(64<<20), "allocated %d bytes for a %d byte file", alloc, size)
}
