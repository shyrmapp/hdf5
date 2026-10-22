package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// GlobalHeapSignature is the magic signature for global heap collections (4 bytes ASCII "GCOL").
var GlobalHeapSignature = [4]byte{'G', 'C', 'O', 'L'}

// GlobalHeapCollection represents a global heap collection (H5HG_heap_t in C).
// Format reference: H5HGpkg.h.
type GlobalHeapCollection struct {
	Address      uint64             // File address of this collection.
	Size         uint64             // Total size of collection in bytes.
	Objects      []GlobalHeapObject // Array of heap objects.
	CollectionID uint16             // Collection ID for debugging.
}

// GlobalHeapObject represents a single object in the global heap.
// Format reference: H5HGpkg.h:105-109.
type GlobalHeapObject struct {
	Index int    // Object index within collection.
	Size  uint64 // Size of object data.
	Data  []byte // Actual object data.
	NRefs uint16 // Reference count.
}

// GlobalHeapReference represents a reference to a global heap object.
// This is what's stored in dataset data for variable-length types.
// Format: heap_address (offset_size bytes) + object_index (4 bytes).
type GlobalHeapReference struct {
	HeapAddress uint64 // Address of heap collection.
	ObjectIndex uint32 // Index within the collection.
}

// ReadGlobalHeapCollection reads a global heap collection from the file.
// Collection format (H5HG.c:156-180):
//   - Signature (4 bytes): "GCOL".
//   - Version (1 byte): always 1.
//   - Reserved (3 bytes).
//   - Collection size (offset_size bytes).
//   - Then follows heap objects, each with:
//   - Object ID (2 bytes).
//   - Reference count (2 bytes).
//   - Reserved (4 bytes).
//   - Object size (offset_size bytes).
//   - Object data (size bytes, aligned to 8-byte boundary).
func ReadGlobalHeapCollection(r io.ReaderAt, address uint64, offsetSize int) (*GlobalHeapCollection, error) {
	if offsetSize != 4 && offsetSize != 8 {
		return nil, fmt.Errorf("invalid offset size: %d (must be 4 or 8)", offsetSize)
	}

	// Read collection header.
	headerSize := 4 + 1 + 3 + offsetSize // signature + version + reserved + size.
	headerBuf := make([]byte, headerSize)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(headerBuf, int64(address)); err != nil {
		return nil, fmt.Errorf("failed to read global heap header: %w", err)
	}

	// Verify signature.
	if string(headerBuf[0:4]) != "GCOL" {
		return nil, fmt.Errorf("invalid global heap signature: %q (expected GCOL)", string(headerBuf[0:4]))
	}

	// Parse version.
	version := headerBuf[4]
	if version != 1 {
		return nil, fmt.Errorf("unsupported global heap version: %d", version)
	}

	// Parse collection size.
	var collectionSize uint64
	if offsetSize == 8 {
		collectionSize = binary.LittleEndian.Uint64(headerBuf[8:16])
	} else {
		collectionSize = uint64(binary.LittleEndian.Uint32(headerBuf[8:12]))
	}

	//nolint:gosec // G115: Safe conversion for HDF5 structure sizes
	if collectionSize < uint64(headerSize) {
		return nil, fmt.Errorf("invalid collection size: %d (too small)", collectionSize)
	}

	if collectionSize > maxMetadataBlock || collectionSize > blockLimit(r, address) {
		return nil, fmt.Errorf("global heap collection size %d exceeds file or limit", collectionSize)
	}

	// Read entire collection into memory for easier parsing.
	collectionData := make([]byte, collectionSize)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(collectionData, int64(address)); err != nil {
		return nil, fmt.Errorf("failed to read global heap collection: %w", err)
	}

	collection := &GlobalHeapCollection{
		Address: address,
		Size:    collectionSize,
		Objects: make([]GlobalHeapObject, 0),
	}

	// Parse heap objects.
	// Start after header, aligned to 8-byte boundary.
	offset := headerSize
	// Align to 8-byte boundary.
	if offset%8 != 0 {
		offset += 8 - (offset % 8)
	}

	for offset < len(collectionData) {
		// Need at least object header: ID(2) + nrefs(2) + reserved(4) + size(offsetSize).
		objHeaderSize := 2 + 2 + 4 + offsetSize
		if offset+objHeaderSize > len(collectionData) {
			break
		}

		// Parse object header.
		objID := binary.LittleEndian.Uint16(collectionData[offset : offset+2])
		nRefs := binary.LittleEndian.Uint16(collectionData[offset+2 : offset+4])
		// Skip reserved (4 bytes).

		var objSize uint64
		if offsetSize == 8 {
			objSize = binary.LittleEndian.Uint64(collectionData[offset+8 : offset+16])
		} else {
			objSize = uint64(binary.LittleEndian.Uint32(collectionData[offset+8 : offset+12]))
		}

		// Object ID 0 is the free space object, skip it.
		if objID == 0 {
			// Free space - nothing follows it that we can parse once it
			// claims more than the collection holds.
			remaining := uint64(len(collectionData) - offset - objHeaderSize) //nolint:gosec // G115: value bounded by the enclosing length checks
			if objSize > remaining {
				break
			}
			//nolint:gosec // G115: objSize <= remaining <= len(collectionData)
			offset += objHeaderSize + int(objSize)
			offset = (offset + 7) &^ 7
			continue
		}

		// Read object data.
		dataStart := offset + objHeaderSize
		if objSize > uint64(len(collectionData)-dataStart) { //nolint:gosec // G115: value bounded by the enclosing length checks
			return nil, fmt.Errorf("object %d data extends beyond collection", objID)
		}
		//nolint:gosec // G115: objSize <= len(collectionData)
		objEnd := dataStart + int(objSize)

		objData := make([]byte, objSize)
		copy(objData, collectionData[dataStart:objEnd])

		obj := GlobalHeapObject{
			Index: int(objID),
			Size:  objSize,
			Data:  objData,
			NRefs: nRefs,
		}

		collection.Objects = append(collection.Objects, obj)

		// Advance to next object (data is aligned to 8-byte boundary).
		offset = (objEnd + 7) &^ 7
	}

	return collection, nil
}

// GetObject retrieves an object from the collection by index.
func (gc *GlobalHeapCollection) GetObject(index uint32) (*GlobalHeapObject, error) {
	for i := range gc.Objects {
		if gc.Objects[i].Index == int(index) {
			return &gc.Objects[i], nil
		}
	}
	return nil, fmt.Errorf("object with index %d not found in collection", index)
}

// ParseGlobalHeapReference parses a global heap reference from raw bytes.
// Format: heap_address (offsetSize bytes) + object_index (4 bytes).
func ParseGlobalHeapReference(data []byte, offsetSize int) (*GlobalHeapReference, error) {
	if offsetSize != 4 && offsetSize != 8 {
		return nil, errors.New("invalid offset size")
	}

	requiredSize := offsetSize + 4
	if len(data) < requiredSize {
		return nil, fmt.Errorf("insufficient data for global heap reference: got %d bytes, need %d", len(data), requiredSize)
	}

	var heapAddr uint64
	if offsetSize == 8 {
		heapAddr = binary.LittleEndian.Uint64(data[0:8])
	} else {
		heapAddr = uint64(binary.LittleEndian.Uint32(data[0:4]))
	}

	//nolint:gosec // G602: len(data) >= offsetSize+4 checked above
	objIndex := binary.LittleEndian.Uint32(data[offsetSize : offsetSize+4])

	return &GlobalHeapReference{
		HeapAddress: heapAddr,
		ObjectIndex: objIndex,
	}, nil
}
