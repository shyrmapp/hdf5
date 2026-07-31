// Package structures provides parsers for HDF5 internal data structures.
package structures

import (
	"errors"
	"fmt"
	"io"

	"github.com/shyrmapp/hdf5/internal/core"
)

// BTreeEntry is a symbol-table B-tree entry. The on-disk record is identical
// to a symbol table node entry, so it is the same type.
type BTreeEntry = SymbolTableEntry

// ReadBTreeEntries reads B-tree entries from a leaf node at the specified address.
func ReadBTreeEntries(r io.ReaderAt, address uint64, sb *core.Superblock) ([]BTreeEntry, error) {
	buf := make([]byte, 6)

	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(buf, int64(address)); err != nil {
		return nil, fmt.Errorf("b-tree node read failed: %w", err)
	}

	if string(buf[0:4]) != "BTRE" {
		return nil, errors.New("invalid B-tree signature")
	}

	_ = buf[4]
	level := buf[5]

	if level != 0 {
		return nil, errors.New("non-leaf nodes not supported yet")
	}

	entryCountBuf := make([]byte, 2)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(entryCountBuf, int64(address+6)); err != nil {
		return nil, err
	}
	entryCount := sb.Endianness.Uint16(entryCountBuf)

	var entries []BTreeEntry
	entrySize := 24

	for i := uint16(0); i < entryCount; i++ {
		entryBuf := make([]byte, entrySize)
		offset := address + 8 + uint64(i)*uint64(entrySize)

		//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
		if _, err := r.ReadAt(entryBuf, int64(offset)); err != nil {
			return nil, err
		}

		entries = append(entries, BTreeEntry{
			LinkNameOffset: sb.Endianness.Uint64(entryBuf[0:8]),
			ObjectAddress:  sb.Endianness.Uint64(entryBuf[8:16]),
			CacheType:      sb.Endianness.Uint32(entryBuf[16:20]),
			Reserved:       sb.Endianness.Uint32(entryBuf[20:24]),
		})
	}

	return entries, nil
}
