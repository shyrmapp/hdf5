package core

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// maxMetadataBlock caps any single metadata block read into memory when the
// reader cannot report its size.
const maxMetadataBlock = 1 << 30

// readerSize reports the size of r when it can be determined
// (os.File, bytes.Reader, io.SectionReader, ...).
func readerSize(r io.ReaderAt) (uint64, bool) {
	switch v := r.(type) {
	case interface{ Size() int64 }:
		if n := v.Size(); n >= 0 {
			return uint64(n), true
		}
	case *os.File:
		if fi, err := v.Stat(); err == nil && fi.Size() >= 0 {
			return uint64(fi.Size()), true //nolint:gosec // G115: value bounded by the enclosing length checks
		}
	}
	return 0, false
}

// blockLimit returns the largest number of bytes a metadata block starting at
// address may occupy: the remaining file size, else maxMetadataBlock.
func blockLimit(r io.ReaderAt, address uint64) uint64 {
	if sz, ok := readerSize(r); ok {
		if address >= sz {
			return 0
		}
		return sz - address
	}
	return maxMetadataBlock
}

const (
	// maxContinuationBlocks bounds the number of continuation blocks per object header.
	maxContinuationBlocks = 1024
	// maxHeaderBlock bounds one object header chunk read for checksum verification.
	maxHeaderBlock = 256 << 20
)

// contTracker rejects object header continuation cycles and runaway chains.
type contTracker struct {
	r    io.ReaderAt
	seen map[uint64]bool
}

func newContTracker(r io.ReaderAt) *contTracker {
	return &contTracker{r: r, seen: map[uint64]bool{}}
}

func (t *contTracker) visit(c continuationInfo) error {
	if t.seen[c.Address] {
		return fmt.Errorf("object header continuation cycle at 0x%x", c.Address)
	}
	if len(t.seen) >= maxContinuationBlocks {
		return fmt.Errorf("object header has more than %d continuation blocks", maxContinuationBlocks)
	}
	if c.Size > maxHeaderBlock || c.Size > blockLimit(t.r, c.Address) {
		return fmt.Errorf("object header continuation block at 0x%x size %d exceeds file", c.Address, c.Size)
	}
	t.seen[c.Address] = true
	return nil
}

// verifyChecksum checks that the last 4 bytes of block hold the Jenkins
// checksum of the preceding bytes.
func verifyChecksum(block []byte, what string, addr uint64) error {
	n := len(block) - 4
	if n < 0 {
		return fmt.Errorf("%s block at 0x%x too small for checksum", what, addr)
	}
	if got, want := JenkinsChecksum(block[:n]), binary.LittleEndian.Uint32(block[n:]); got != want {
		return fmt.Errorf("%s block at 0x%x checksum mismatch: computed %08x, stored %08x", what, addr, got, want)
	}
	return nil
}

// verifyBlockChecksum reads the metadata block [start, end) and verifies its
// trailing Jenkins checksum.
func verifyBlockChecksum(r io.ReaderAt, start, end uint64, what string) error {
	if end < start+4 || end-start > maxHeaderBlock || end-start > blockLimit(r, start) {
		return fmt.Errorf("%s block at 0x%x has invalid extent", what, start)
	}
	buf := make([]byte, end-start)
	if _, err := r.ReadAt(buf, int64(start)); err != nil { //nolint:gosec // G115: bounded by file size
		return fmt.Errorf("%s checksum read failed: %w", what, err)
	}
	return verifyChecksum(buf, what, start)
}
