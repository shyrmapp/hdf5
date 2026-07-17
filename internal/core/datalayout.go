package core

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// DataLayoutClass represents the storage layout type.
type DataLayoutClass uint8

// Data layout class constants define how dataset data is stored.
const (
	LayoutCompact    DataLayoutClass = 0 // Data stored in message.
	LayoutContiguous DataLayoutClass = 1 // Data stored contiguously in file.
	LayoutChunked    DataLayoutClass = 2 // Data stored in chunks.
	LayoutVirtual    DataLayoutClass = 3 // Virtual dataset (HDF5 1.10+).

	layoutUnknown = "unknown" // String representation for unknown layout class.
)

// Chunk indexing types used by data layout message versions 4 and 5.
// Reference: H5Dpkg.h (H5D_chunk_index_t) and the HDF5 file format spec,
// "Version 4 Data Layout Message: Chunked Storage Property Description".
const (
	ChunkIndexBTreeV1  uint8 = 0 // Implied for layout v3 (not stored in file).
	ChunkIndexSingle   uint8 = 1 // Single Chunk index.
	ChunkIndexImplicit uint8 = 2 // Implicit index (contiguous chunks, no metadata).
	ChunkIndexFixed    uint8 = 3 // Fixed Array index.
	ChunkIndexExt      uint8 = 4 // Extensible Array index.
	ChunkIndexBTreeV2  uint8 = 5 // Version 2 B-tree index.
)

// Layout v4/v5 chunked flags (H5O_LAYOUT_CHUNK_* in H5Olayout.h).
const (
	// LayoutChunkDontFilterPartialChunks: partial edge chunks are stored unfiltered.
	LayoutChunkDontFilterPartialChunks uint8 = 0x01
	// LayoutChunkSingleIndexWithFilter: single-chunk index stores filtered size + mask.
	LayoutChunkSingleIndexWithFilter uint8 = 0x02
)

// DataLayoutMessage represents HDF5 data layout message.
type DataLayoutMessage struct {
	Version      uint8
	Class        DataLayoutClass
	DataAddress  uint64   // Address where data is stored (for contiguous/chunked).
	DataSize     uint64   // Size of data (for contiguous).
	CompactData  []byte   // Data itself (for compact layout).
	ChunkSize    []uint64 // Chunk dimensions (for chunked layout) - uint64 for HDF5 2.0.0+ support.
	ChunkKeySize uint8    // Size of chunk keys in bytes: 4 (uint32) or 8 (uint64).

	// Fields below are only set for layout message versions 4 and 5 (chunked).
	Flags          uint8 // LayoutChunk* flags.
	ChunkIndexType uint8 // ChunkIndex* constant; ChunkIndexBTreeV1 for v3 layouts.

	// Single Chunk index with filter (Flags & LayoutChunkSingleIndexWithFilter).
	SingleChunkNBytes     uint64 // Filtered (stored) size of the single chunk.
	SingleChunkFilterMask uint32 // Filter mask of the single chunk.

	// Fixed Array index creation parameter.
	FixedArrayPageBits uint8 // Log2 of number of elements per data block page.

	// Extensible Array index creation parameters (decode order per H5Olayout.c).
	ExtArrayMaxNelmtsBits         uint8 // Log2 of maximum number of elements.
	ExtArrayIdxBlkElmts           uint8 // Number of elements in the index block.
	ExtArraySupBlkMinDataPtrs     uint8 // Minimum data block pointers per super block.
	ExtArrayDataBlkMinElmts       uint8 // Minimum elements per data block.
	ExtArrayMaxDblkPageNelmtsBits uint8 // Log2 of maximum elements per data block page.

	// Version 2 B-tree index creation parameters.
	BTreeV2NodeSize     uint32 // Size of B-tree nodes in bytes.
	BTreeV2SplitPercent uint8  // Node split percentage.
	BTreeV2MergePercent uint8  // Node merge percentage.
}

// ParseDataLayoutMessage parses a data layout message from header message data.
func ParseDataLayoutMessage(data []byte, sb *Superblock) (*DataLayoutMessage, error) {
	if len(data) < 1 {
		return nil, errors.New("data layout message too short")
	}

	version := data[0]

	// Version 3 (HDF5 1.8+) and versions 4/5 (HDF5 1.10+ with
	// H5F_LIBVER_LATEST; v5 is written for filtered datasets by 1.14+/2.x).
	if version < 3 || version > 5 {
		return nil, fmt.Errorf("unsupported data layout version: %d", version)
	}

	msg := &DataLayoutMessage{
		Version:      version,
		ChunkKeySize: determineChunkKeySize(sb.Version),
	}

	if version == 3 {
		return parseLayoutV3(data, sb, msg)
	}
	return parseLayoutV4(data, sb, msg)
}

// determineChunkKeySize determines the chunk key size based on file format version.
// HDF5 < 2.0.0 (superblock v0-v2) uses 32-bit chunk dimensions.
// Future versions may use 64-bit chunk dimensions.
func determineChunkKeySize(superblockVersion uint8) uint8 {
	// Conservative approach: use 32-bit for all current versions (0, 2, 3).
	// All tested files (including HDF5 2.0.0) work correctly with 32-bit.
	// This condition (>= 4) is prepared for potential future versions.
	if superblockVersion >= 4 {
		return 8
	}
	return 4
}

// parseLayoutV3 parses HDF5 Data Layout Message version 3.
// Cognitive complexity is high due to handling 3 distinct layout types
// (Compact, Contiguous, Chunked) with different binary formats and
// HDF5 version differences (32-bit vs 64-bit chunk dimensions).
// This complexity is inherent to the HDF5 format specification.
//
//nolint:gocognit,cyclop // Binary format parsing requires handling multiple layout types
func parseLayoutV3(data []byte, sb *Superblock, msg *DataLayoutMessage) (*DataLayoutMessage, error) {
	if len(data) < 2 {
		return nil, errors.New("layout v3 message too short")
	}

	msg.Class = DataLayoutClass(data[1])

	switch msg.Class {
	case LayoutCompact:
		// Compact layout: data is stored in the message itself.
		if len(data) < 4 {
			return nil, errors.New("compact layout message too short")
		}
		size := binary.LittleEndian.Uint16(data[2:4])
		if len(data) < 4+int(size) {
			return nil, errors.New("compact layout data truncated")
		}
		msg.CompactData = data[4 : 4+size]
		msg.DataSize = uint64(size)

	case LayoutContiguous:
		// Contiguous layout: data stored sequentially in file.
		if len(data) < 2+int(sb.OffsetSize)+int(sb.LengthSize) {
			return nil, errors.New("contiguous layout message too short")
		}

		offset := 2
		// Read data address.
		msg.DataAddress = readUint64(data[offset:], int(sb.OffsetSize), sb.Endianness)
		offset += int(sb.OffsetSize)

		// Read data size.
		msg.DataSize = readUint64(data[offset:], int(sb.LengthSize), sb.Endianness)

	case LayoutChunked:
		// Chunked layout v3: dimensionality + B-tree address + chunk dimensions.
		// Reference: H5Olayout.c - H5D_CHUNKED case.
		if len(data) < 3 {
			return nil, errors.New("chunked layout message too short")
		}

		dimensionality := data[2]
		offset := 3

		// For v3, B-tree address comes BEFORE chunk dimensions.
		// Read B-tree address (where chunk index is stored).
		if offset+int(sb.OffsetSize) > len(data) {
			return nil, errors.New("chunked layout address truncated")
		}
		msg.DataAddress = readUint64(data[offset:], int(sb.OffsetSize), sb.Endianness)
		offset += int(sb.OffsetSize)

		// Read chunk dimensions.
		// Current HDF5 formats (superblock v0-v3) use 32-bit chunk dimensions.
		// Future formats may use 64-bit chunk dimensions.
		msg.ChunkSize = make([]uint64, dimensionality)

		if msg.ChunkKeySize == 8 {
			// Read as uint64 (HDF5 2.0.0+).
			for i := 0; i < int(dimensionality); i++ {
				if offset+8 > len(data) {
					return nil, fmt.Errorf("chunked layout dimension %d truncated (64-bit)", i)
				}
				msg.ChunkSize[i] = binary.LittleEndian.Uint64(data[offset : offset+8])
				offset += 8
			}
		} else {
			// Read as uint32 (HDF5 < 2.0.0) and convert to uint64 for internal consistency.
			for i := 0; i < int(dimensionality); i++ {
				if offset+4 > len(data) {
					return nil, fmt.Errorf("chunked layout dimension %d truncated (32-bit)", i)
				}
				chunk32 := binary.LittleEndian.Uint32(data[offset : offset+4])
				msg.ChunkSize[i] = uint64(chunk32) // Safe widening conversion.
				offset += 4
			}
		}

	default:
		return nil, fmt.Errorf("unsupported layout class: %d", msg.Class)
	}

	return msg, nil
}

// parseLayoutV4 parses HDF5 Data Layout Message versions 4 and 5.
// Both versions share the same encoding for the fields read here; they differ
// only for layout classes/index features this reader does not yet support
// (e.g. virtual storage details). For compact and contiguous classes the
// encoding is identical to version 3, so those delegate to the v3 parser.
// Reference: H5Olayout.c (H5O__layout_decode), chunked branch.
func parseLayoutV4(data []byte, sb *Superblock, msg *DataLayoutMessage) (*DataLayoutMessage, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("layout v%d message too short", msg.Version)
	}

	msg.Class = DataLayoutClass(data[1])
	if msg.Class != LayoutChunked {
		// Compact and contiguous have the same wire format as v3.
		return parseLayoutV3(data, sb, msg)
	}

	// Chunked storage, v4/v5:
	//   Flags (1) | Dimensionality (1) | Dim Size Encoded Length (1) |
	//   Dimension Sizes (dimensionality x encoded length) |
	//   Chunk Indexing Type (1) | index-specific fields | Address (offsetSize).
	if len(data) < 5 {
		return nil, errors.New("chunked layout v4 message too short")
	}
	msg.Flags = data[2]
	dimensionality := int(data[3])
	encLen := int(data[4])
	if encLen < 1 || encLen > 8 {
		return nil, fmt.Errorf("invalid chunk dimension encoded length: %d", encLen)
	}

	offset := 5
	if offset+dimensionality*encLen > len(data) {
		return nil, errors.New("chunked layout v4 dimensions truncated")
	}
	// Same convention as v3: chunk dims for each dataset dimension, then the
	// element size as a trailing pseudo-dimension.
	msg.ChunkSize = make([]uint64, dimensionality)
	for i := 0; i < dimensionality; i++ {
		msg.ChunkSize[i] = readUint64(data[offset:], encLen, sb.Endianness)
		offset += encLen
	}

	if offset >= len(data) {
		return nil, errors.New("chunked layout v4 index type truncated")
	}
	msg.ChunkIndexType = data[offset]
	offset++

	offset, err := parseChunkIndexInfo(data, offset, msg, sb)
	if err != nil {
		return nil, err
	}

	if offset+int(sb.OffsetSize) > len(data) {
		return nil, errors.New("chunked layout v4 address truncated")
	}
	msg.DataAddress = readUint64(data[offset:], int(sb.OffsetSize), sb.Endianness)

	return msg, nil
}

// parseChunkIndexInfo parses the chunk-index-specific fields of a v4/v5
// chunked layout message and returns the offset past them.
func parseChunkIndexInfo(data []byte, offset int, msg *DataLayoutMessage, sb *Superblock) (int, error) {
	switch msg.ChunkIndexType {
	case ChunkIndexSingle:
		if msg.Flags&LayoutChunkSingleIndexWithFilter != 0 {
			// Filtered single chunk: stored size (lengthSize) + filter mask (4).
			if offset+int(sb.LengthSize)+4 > len(data) {
				return 0, errors.New("single chunk filter info truncated")
			}
			msg.SingleChunkNBytes = readUint64(data[offset:], int(sb.LengthSize), sb.Endianness)
			offset += int(sb.LengthSize)
			msg.SingleChunkFilterMask = sb.Endianness.Uint32(data[offset : offset+4])
			offset += 4
		}

	case ChunkIndexImplicit:
		// No index-specific fields.

	case ChunkIndexFixed:
		if offset+1 > len(data) {
			return 0, errors.New("fixed array index info truncated")
		}
		msg.FixedArrayPageBits = data[offset]
		offset++

	case ChunkIndexExt:
		if offset+5 > len(data) {
			return 0, errors.New("extensible array index info truncated")
		}
		// Decode order per H5Olayout.c: max nelmts bits, index block elements,
		// super block min data pointers, data block min elements, page bits.
		msg.ExtArrayMaxNelmtsBits = data[offset]
		msg.ExtArrayIdxBlkElmts = data[offset+1]
		msg.ExtArraySupBlkMinDataPtrs = data[offset+2]
		msg.ExtArrayDataBlkMinElmts = data[offset+3]
		msg.ExtArrayMaxDblkPageNelmtsBits = data[offset+4]
		offset += 5

	case ChunkIndexBTreeV2:
		if offset+6 > len(data) {
			return 0, errors.New("v2 b-tree index info truncated")
		}
		msg.BTreeV2NodeSize = sb.Endianness.Uint32(data[offset : offset+4])
		msg.BTreeV2SplitPercent = data[offset+4]
		msg.BTreeV2MergePercent = data[offset+5]
		offset += 6

	default:
		return 0, fmt.Errorf("unsupported chunk index type: %d", msg.ChunkIndexType)
	}

	return offset, nil
}

// Helper function to read variable-sized unsigned integers.
func readUint64(data []byte, size int, endianness binary.ByteOrder) uint64 {
	if size > len(data) {
		size = len(data)
	}

	switch size {
	case 1:
		return uint64(data[0])
	case 2:
		return uint64(endianness.Uint16(data[:2]))
	case 4:
		return uint64(endianness.Uint32(data[:4]))
	case 8:
		return endianness.Uint64(data[:8])
	default:
		// Pad to 8 bytes and read.
		var buf [8]byte
		copy(buf[:], data[:size])
		return endianness.Uint64(buf[:])
	}
}

// IsContiguous returns true if layout is contiguous.
func (dl *DataLayoutMessage) IsContiguous() bool {
	return dl.Class == LayoutContiguous
}

// IsCompact returns true if layout is compact (data in message).
func (dl *DataLayoutMessage) IsCompact() bool {
	return dl.Class == LayoutCompact
}

// IsChunked returns true if layout is chunked.
func (dl *DataLayoutMessage) IsChunked() bool {
	return dl.Class == LayoutChunked
}

// String returns human-readable layout description.
func (dl *DataLayoutMessage) String() string {
	switch dl.Class {
	case LayoutCompact:
		return fmt.Sprintf("compact (size=%d)", dl.DataSize)
	case LayoutContiguous:
		return fmt.Sprintf("contiguous (address=0x%X, size=%d)", dl.DataAddress, dl.DataSize)
	case LayoutChunked:
		return fmt.Sprintf("chunked (chunks=%v)", dl.ChunkSize)
	case LayoutVirtual:
		return "virtual"
	default:
		return layoutUnknown
	}
}
