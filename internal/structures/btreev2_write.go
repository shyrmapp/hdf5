// Copyright (c) 2025 SciGo HDF5 Library Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package structures

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/shyrmapp/hdf5/internal/utils"
)

// B-tree v2 constants.
const (
	BTreeV2HeaderSignature = "BTHD" // B-tree v2 header signature
	BTreeV2LeafSignature   = "BTLF" // B-tree v2 leaf node signature

	BTreeV2TypeLinkNameIndex = uint8(5) // Type 5 = Link Name Index for dense groups
	// BTreeV2TypeAttrNameIndex is the attribute name index (H5A_BT2_NAME):
	// records are heap ID (8), message flags (1), creation order (4), name hash (4).
	BTreeV2TypeAttrNameIndex = uint8(8)

	linkRecordSize = 11
	attrRecordSize = 17

	DefaultBTreeV2NodeSize     = uint32(4096) // 4KB default node size
	DefaultBTreeV2SplitPercent = uint8(100)   // Split at 100% full
	DefaultBTreeV2MergePercent = uint8(40)    // Merge at 40% full
)

// B-tree v2 error definitions.
var (
	ErrBTreeNodeFull    = errors.New("b-tree node is full")
	ErrInvalidBTreeType = errors.New("invalid b-tree type")
)

// BTreeV2Header represents B-tree v2 header structure.
//
// Format (from H5B2hdr.c and HDF5 Format Spec III.A.2):
//   - Signature: "BTHD" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 = Link Name Index (1 byte)
//   - Node Size: Typically 4096 bytes (4 bytes)
//   - Record Size: Variable (link name record) (2 bytes)
//   - Depth: 0 for single leaf (2 bytes)
//   - Split Percent: 100 (1 byte)
//   - Merge Percent: 40 (1 byte)
//   - Root Node Address: offsetSize bytes
//   - Number of Records in Root: 2 bytes
//   - Total Records: 8 bytes
//   - Checksum: CRC32 (4 bytes)
//
// Reference:
//   - HDF5 Format Spec: Section III.A.2 (B-tree v2)
//   - C Library: H5B2hdr.c - H5B2_hdr_t structure
//   - Serialization: H5B2cache.c - H5B2__hdr_serialize()
type BTreeV2Header struct {
	Signature      [4]byte // "BTHD"
	Version        uint8   // 0
	Type           uint8   // 5 = Link Name Index
	NodeSize       uint32  // Node size in bytes
	RecordSize     uint16  // Size of record (11 bytes for link name records)
	Depth          uint16  // Tree depth (0: single leaf)
	SplitPercent   uint8   // 100
	MergePercent   uint8   // 40
	RootNodeAddr   uint64  // Address of root (leaf) node
	NumRecordsRoot uint16  // Number of records in root node
	TotalRecords   uint64  // Total records in tree
}

// BTreeV2LeafNode represents B-tree v2 leaf node structure.
//
// Format (from H5B2leaf.c and HDF5 Format Spec III.A.2):
//   - Signature: "BTLF" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 = Link Name Index (1 byte)
//   - Records: Array of link name records
//   - Checksum: CRC32 (4 bytes)
//
// Reference:
//   - HDF5 Format Spec: Section III.A.2 (B-tree v2)
//   - C Library: H5B2leaf.c - H5B2_leaf_t structure
//   - Serialization: H5B2cache.c - H5B2__leaf_serialize()
type BTreeV2LeafNode struct {
	Signature [4]byte          // "BTLF"
	Version   uint8            // 0
	Type      uint8            // 5
	Records   []LinkNameRecord // Link records
}

// LinkNameRecord represents a link name index record in B-tree v2.
//
// Format (from H5Gbtree2.c - H5G_dense_btree2_name_rec_t):
//   - Name Hash: uint32 (Jenkins hash of link name)
//   - Heap ID: 7 bytes (fractal heap object ID)
//
// The heap ID points to a link message stored in the fractal heap.
// HDF5 uses a 7-byte heap ID in the B-tree (vs 8 bytes in the heap itself).
//
// Reference:
//   - C Library: H5Gbtree2.c - H5G_dense_btree2_name_rec_t
//   - Encoding: H5Gbtree2.c - H5G__dense_btree2_name_encode()
type LinkNameRecord struct {
	NameHash uint32  // Jenkins hash of link name
	HeapID   [7]byte // Fractal heap ID (7 bytes, not 8)
}

// WritableBTreeV2 manages B-tree v2 construction for link name indexing.
//
// Limitations:
//   - Single leaf node only (no internal nodes)
//   - No splits (error if node size exceeded)
//   - Records sorted by name hash
//   - Maximum records = (nodeSize - overhead) / recordSize
type WritableBTreeV2 struct {
	header   *BTreeV2Header
	leaf     *BTreeV2LeafNode
	records  []LinkNameRecord
	nodeSize uint32

	// Addresses loaded from file (for RMW scenarios)
	loadedHeaderAddress uint64
	loadedLeafAddress   uint64
}

// NewWritableBTreeV2 creates a new B-tree v2 for link name indexing.
//
// Parameters:
//   - nodeSize: size of the B-tree node in bytes (typically 4096)
//
// Returns:
//   - *WritableBTreeV2: B-tree structure ready for record insertion
func NewWritableBTreeV2(nodeSize uint32) *WritableBTreeV2 {
	return newWritableBTreeV2(nodeSize, BTreeV2TypeLinkNameIndex, linkRecordSize)
}

// NewWritableAttributeBTreeV2 creates a B-tree v2 for the dense attribute name
// index (record type 8), the layout the HDF5 C library expects.
func NewWritableAttributeBTreeV2(nodeSize uint32) *WritableBTreeV2 {
	return newWritableBTreeV2(nodeSize, BTreeV2TypeAttrNameIndex, attrRecordSize)
}

func newWritableBTreeV2(nodeSize uint32, recType uint8, recSize uint16) *WritableBTreeV2 {
	if nodeSize == 0 {
		nodeSize = DefaultBTreeV2NodeSize
	}

	return &WritableBTreeV2{
		header: &BTreeV2Header{
			Signature:      [4]byte{'B', 'T', 'H', 'D'},
			Version:        0,
			Type:           recType,
			NodeSize:       nodeSize,
			RecordSize:     recSize,
			Depth:          0, // Single leaf
			SplitPercent:   DefaultBTreeV2SplitPercent,
			MergePercent:   DefaultBTreeV2MergePercent,
			RootNodeAddr:   0, // Set when writing
			NumRecordsRoot: 0,
			TotalRecords:   0,
		},
		leaf: &BTreeV2LeafNode{
			Signature: [4]byte{'B', 'T', 'L', 'F'},
			Version:   0,
			Type:      recType,
			Records:   make([]LinkNameRecord, 0),
		},
		records:  make([]LinkNameRecord, 0),
		nodeSize: nodeSize,
	}
}

// InsertRecord adds a link name record to the B-tree.
//
// Parameters:
//   - linkName: name of the link (for hash calculation)
//   - heapID: 8-byte fractal heap object ID (we store 7 bytes)
//
// Stores in a single leaf node, sorted by name hash.
//
// Returns:
//   - error if node is full or insertion fails
func (bt *WritableBTreeV2) InsertRecord(linkName string, heapID uint64) error {
	// Calculate Jenkins hash for link name
	hash := jenkinsHash(linkName)

	// Convert 8-byte heap ID to 7-byte format
	// HDF5 stores heap IDs as 7 bytes in B-tree (first 7 bytes of the ID)
	var heapIDBytes [7]byte
	var temp [8]byte
	binary.LittleEndian.PutUint64(temp[:], heapID)
	copy(heapIDBytes[:], temp[:7])

	record := LinkNameRecord{
		NameHash: hash,
		HeapID:   heapIDBytes,
	}

	// Check if node will be full
	maxRecords := bt.calculateMaxRecords()
	if len(bt.records) >= maxRecords {
		return ErrBTreeNodeFull
	}

	// Insert sorted by hash
	bt.records = insertRecordSorted(bt.records, record)
	bt.header.TotalRecords++
	bt.header.NumRecordsRoot++
	bt.leaf.Records = bt.records

	return nil
}

// SearchRecord searches for a record by name and returns the heap ID.
//
// This function is used for attribute modification - to find an existing attribute
// by name and get its heap ID for reading or updating.
//
// Parameters:
//   - name: attribute/link name to search for
//
// Returns:
//   - []byte: 8-byte heap ID (converted from 7-byte stored format)
//   - bool: true if found, false if not found
//
// Searches the single leaf node by name hash.
//
// Reference: H5Adense.c - H5A__dense_write() searches B-tree by name.
func (bt *WritableBTreeV2) SearchRecord(name string) ([]byte, bool) {
	hash := jenkinsHash(name)

	// Search in records
	for _, record := range bt.records {
		if record.NameHash == hash {
			// Convert 7-byte heap ID to 8-byte format
			heapID := make([]byte, 8)
			copy(heapID, record.HeapID[:])
			// Last byte is 0 (7-byte format pads to 8 bytes)
			return heapID, true
		}
	}

	return nil, false
}

// UpdateRecord updates an existing record's heap ID.
//
// This function is used when modifying attributes with different sizes:
// 1. Delete old heap object
// 2. Insert new heap object → get new heap ID
// 3. Update B-tree record with new heap ID
//
// Parameters:
//   - name: attribute/link name to update
//   - newHeapID: new 8-byte heap ID
//
// Returns:
//   - error: if record not found or update fails
//
// Updates the record in the single leaf node.
//
// Reference: H5Adense.c - H5A__dense_write() updates B-tree when size changes.
func (bt *WritableBTreeV2) UpdateRecord(name string, newHeapID uint64) error {
	hash := jenkinsHash(name)

	// Find record
	for i, record := range bt.records {
		if record.NameHash != hash {
			continue
		}

		// Convert 8-byte heap ID to 7-byte format
		var heapIDBytes [7]byte
		var temp [8]byte
		binary.LittleEndian.PutUint64(temp[:], newHeapID)
		copy(heapIDBytes[:], temp[:7])

		// Update record
		bt.records[i].HeapID = heapIDBytes
		bt.leaf.Records = bt.records
		return nil
	}

	return fmt.Errorf("record not found for name: %s", name)
}

// DeleteRecord removes a record from the B-tree by name.
//
// The tree is single-leaf (depth 0), so removal is a slice delete plus
// count updates — no node merging is possible until multi-level trees
// exist. Reference: H5B2.c - H5B2_remove().
func (bt *WritableBTreeV2) DeleteRecord(name string) error {
	hash := jenkinsHash(name)

	recordIndex := -1
	for i, record := range bt.records {
		if record.NameHash == hash {
			recordIndex = i
			break
		}
	}

	if recordIndex == -1 {
		return fmt.Errorf("record not found for name: %s", name)
	}

	bt.records = append(bt.records[:recordIndex], bt.records[recordIndex+1:]...)
	bt.leaf.Records = bt.records

	bt.header.TotalRecords--
	bt.header.NumRecordsRoot--

	return nil
}

// WriteToFile writes B-tree v2 to file and returns header address.
//
// Writes:
//  1. Leaf node at allocated address
//  2. Header at allocated address (with leaf address)
//
// Returns:
//   - uint64: header address (store this in Link Info Message)
//   - error if write fails
func (bt *WritableBTreeV2) WriteToFile(writer Writer, allocator Allocator, sb *core.Superblock) (uint64, error) {
	if writer == nil || allocator == nil || sb == nil {
		return 0, errors.New("writer, allocator, or superblock is nil")
	}

	// Calculate leaf node size
	// IMPORTANT: For RMW (Read-Modify-Write), allocate FULL node size,
	// not just current content size. This allows leaf to grow without relocation.
	// We'll only write actual data, but reserve full node size for future growth.
	leafSize := uint64(bt.nodeSize) // Allocate FULL node size

	// Allocate space for leaf node
	leafAddr, err := allocator.Allocate(leafSize)
	if err != nil {
		return 0, fmt.Errorf("failed to allocate leaf node: %w", err)
	}

	// Encode leaf node
	leafData, err := bt.encodeLeafNode(sb)
	if err != nil {
		return 0, fmt.Errorf("failed to encode leaf node: %w", err)
	}

	// Write leaf node
	if err := writer.WriteAtAddress(leafData, leafAddr); err != nil {
		return 0, fmt.Errorf("failed to write leaf node: %w", err)
	}

	// Calculate header size
	headerSize := bt.calculateHeaderSize(sb)

	// Allocate space for header
	headerAddr, err := allocator.Allocate(headerSize)
	if err != nil {
		return 0, fmt.Errorf("failed to allocate header: %w", err)
	}

	// Set root node address in header
	bt.header.RootNodeAddr = leafAddr

	// Encode header
	headerData, err := bt.encodeHeader(sb)
	if err != nil {
		return 0, fmt.Errorf("failed to encode header: %w", err)
	}

	// Write header
	if err := writer.WriteAtAddress(headerData, headerAddr); err != nil {
		return 0, fmt.Errorf("failed to write header: %w", err)
	}

	return headerAddr, nil
}

// WriteAt writes B-tree v2 in-place at previously loaded addresses.
//
// This method is used for Read-Modify-Write (RMW) scenarios:
// - B-tree was loaded via LoadFromFile()
// - New records were inserted
// - Write back to same addresses
//
// Parameters:
//   - writer: File writer (must implement Writer interface)
//   - sb: Superblock for field sizes
//
// Returns:
//   - error: if write fails or B-tree was not loaded from file
//
// Reference: Same as WriteToFile, but uses stored addresses.
func (bt *WritableBTreeV2) WriteAt(writer Writer, sb *core.Superblock) error {
	if writer == nil || sb == nil {
		return errors.New("writer or superblock is nil")
	}

	// Verify this B-tree was loaded from file
	if bt.loadedHeaderAddress == 0 {
		return errors.New("cannot use WriteAt: B-tree not loaded from file (use WriteToFile for new B-trees)")
	}

	// Encode leaf node
	leafData, err := bt.encodeLeafNode(sb)
	if err != nil {
		return fmt.Errorf("failed to encode leaf node: %w", err)
	}

	// Write leaf node at loaded address
	if err := writer.WriteAtAddress(leafData, bt.loadedLeafAddress); err != nil {
		return fmt.Errorf("failed to write leaf node at 0x%X: %w", bt.loadedLeafAddress, err)
	}

	// Update header with leaf address (in case it was cleared)
	bt.header.RootNodeAddr = bt.loadedLeafAddress

	// Encode header
	headerData, err := bt.encodeHeader(sb)
	if err != nil {
		return fmt.Errorf("failed to encode header: %w", err)
	}

	// Write header at loaded address
	if err := writer.WriteAtAddress(headerData, bt.loadedHeaderAddress); err != nil {
		return fmt.Errorf("failed to write header at 0x%X: %w", bt.loadedHeaderAddress, err)
	}

	return nil
}

// encodeHeader encodes B-tree v2 header for writing.
//
// Format (from H5B2cache.c - H5B2__hdr_serialize):
//   - Signature: "BTHD" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 (1 byte)
//   - Node Size: 4 bytes
//   - Record Size: 2 bytes
//   - Depth: 2 bytes
//   - Split Percent: 1 byte
//   - Merge Percent: 1 byte
//   - Root Node Address: offsetSize bytes
//   - Number of Records in Root: 2 bytes
//   - Total Records: 8 bytes
//   - Checksum: CRC32 (4 bytes)
//
//nolint:unparam // error reserved for future validation/compression features
func (bt *WritableBTreeV2) encodeHeader(sb *core.Superblock) ([]byte, error) {
	size := bt.calculateHeaderSize(sb)
	buf := make([]byte, size)
	offset := 0

	// Signature (4 bytes)
	copy(buf[offset:], bt.header.Signature[:])
	offset += 4

	// Version (1 byte)
	buf[offset] = bt.header.Version
	offset++

	// Type (1 byte)
	buf[offset] = bt.header.Type
	offset++

	// Node Size (4 bytes)
	binary.LittleEndian.PutUint32(buf[offset:], bt.header.NodeSize)
	offset += 4

	// Record Size (2 bytes)
	binary.LittleEndian.PutUint16(buf[offset:], bt.header.RecordSize)
	offset += 2

	// Depth (2 bytes)
	binary.LittleEndian.PutUint16(buf[offset:], bt.header.Depth)
	offset += 2

	// Split Percent (1 byte)
	buf[offset] = bt.header.SplitPercent
	offset++

	// Merge Percent (1 byte)
	buf[offset] = bt.header.MergePercent
	offset++

	// Root Node Address (offsetSize bytes)
	utils.PutUint(buf[offset:], bt.header.RootNodeAddr, int(sb.OffsetSize), sb.Endianness)
	offset += int(sb.OffsetSize)

	// Number of Records in Root (2 bytes)
	binary.LittleEndian.PutUint16(buf[offset:], bt.header.NumRecordsRoot)
	offset += 2

	// Total Records (8 bytes)
	binary.LittleEndian.PutUint64(buf[offset:], bt.header.TotalRecords)
	offset += 8

	// Checksum (Jenkins lookup3, 4 bytes)
	checksum := core.JenkinsChecksum(buf[:offset])
	binary.LittleEndian.PutUint32(buf[offset:], checksum)

	return buf, nil
}

// encodeLeafNode encodes B-tree v2 leaf node for writing.
//
// Format (from H5B2cache.c - H5B2__leaf_serialize):
//   - Signature: "BTLF" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 (1 byte)
//   - Records: Array of link name records (numRecords * 11 bytes)
//   - Checksum: CRC32 (4 bytes)
//
//nolint:unparam // error reserved for future validation/compression features
func (bt *WritableBTreeV2) encodeLeafNode(sb *core.Superblock) ([]byte, error) {
	size := bt.calculateLeafSize(sb)
	buf := make([]byte, size)
	offset := 0

	// Signature (4 bytes)
	copy(buf[offset:], bt.leaf.Signature[:])
	offset += 4

	// Version (1 byte)
	buf[offset] = bt.leaf.Version
	offset++

	// Type (1 byte)
	buf[offset] = bt.leaf.Type
	offset++

	for _, record := range bt.leaf.Records {
		if bt.header.Type == BTreeV2TypeAttrNameIndex {
			// heap ID (8, last byte zero), flags (1), creation order (4), hash (4)
			copy(buf[offset:], record.HeapID[:])
			binary.LittleEndian.PutUint32(buf[offset+13:], record.NameHash)
			offset += attrRecordSize
			continue
		}
		// Link name record: hash (4), heap ID (7)
		binary.LittleEndian.PutUint32(buf[offset:], record.NameHash)
		offset += 4
		copy(buf[offset:], record.HeapID[:])
		offset += 7
	}

	// Checksum (Jenkins lookup3, 4 bytes)
	checksum := core.JenkinsChecksum(buf[:offset])
	binary.LittleEndian.PutUint32(buf[offset:], checksum)

	return buf, nil
}

// calculateHeaderSize calculates the size of the B-tree v2 header.
func (bt *WritableBTreeV2) calculateHeaderSize(sb *core.Superblock) uint64 {
	return uint64(4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + int(sb.OffsetSize) + 2 + 8 + 4)
	// Signature + Version + Type + NodeSize + RecordSize + Depth +
	// SplitPercent + MergePercent + RootNodeAddr + NumRecordsRoot + TotalRecords + Checksum
}

// calculateLeafSize calculates the size of the leaf node.
func (bt *WritableBTreeV2) calculateLeafSize(_ *core.Superblock) uint64 {
	numRecords := len(bt.leaf.Records)
	return uint64(4+1+1+4) + uint64(numRecords)*uint64(bt.header.RecordSize)
	// Signature + Version + Type + Records + Checksum
}

// calculateMaxRecords calculates maximum records for a single leaf node.
func (bt *WritableBTreeV2) calculateMaxRecords() int {
	// Node size - overhead (signature + version + type + checksum)
	overhead := uint32(4 + 1 + 1 + 4) // 10 bytes
	available := bt.nodeSize - overhead
	return int(available / uint32(bt.header.RecordSize))
}

// insertRecordSorted inserts a record into a sorted slice by hash.
func insertRecordSorted(records []LinkNameRecord, newRecord LinkNameRecord) []LinkNameRecord {
	// Find insertion point
	insertPos := len(records) // Default: append at end

	for i := range records {
		if records[i].NameHash >= newRecord.NameHash {
			insertPos = i
			break
		}
	}

	// Insert at position
	records = append(records, LinkNameRecord{})
	copy(records[insertPos+1:], records[insertPos:])
	records[insertPos] = newRecord

	return records
}

// LoadFromFile loads an existing B-tree v2 from file.
//
// This implements read-modify-write (RMW) support:
// 1. Read header and leaf from file
// 2. Populate internal structures
// 3. Ready for new record insertion
//
// Parameters:
//   - r: Reader to read from (io.ReaderAt)
//   - headerAddr: Address of B-tree v2 header
//   - sb: Superblock for offset/length sizes
//
// Returns:
//   - error if read fails or validation fails
//
// Limitations:
//   - Single leaf node only (no internal nodes)
//   - Assumes B-tree type 5 (Link Name Index)
func (bt *WritableBTreeV2) LoadFromFile(r io.ReaderAt, headerAddr uint64, sb *core.Superblock) error {
	if r == nil {
		return fmt.Errorf("reader is nil")
	}
	if sb == nil {
		return fmt.Errorf("superblock is nil")
	}

	// 1. Read and decode header
	header, err := readBTreeV2Header(r, headerAddr, sb)
	if err != nil {
		return fmt.Errorf("failed to read B-tree header: %w", err)
	}

	// 2. Validate header
	if header.Type != BTreeV2TypeLinkNameIndex && header.Type != BTreeV2TypeAttrNameIndex {
		return fmt.Errorf("%w: expected type %d or %d, got %d", ErrInvalidBTreeType,
			BTreeV2TypeLinkNameIndex, BTreeV2TypeAttrNameIndex, header.Type)
	}
	if int(header.RecordSize) != linkRecordSize && int(header.RecordSize) != attrRecordSize {
		return fmt.Errorf("unsupported B-tree record size %d", header.RecordSize)
	}

	if header.Depth != 0 {
		return fmt.Errorf("only single-leaf B-trees are supported (depth 0), got depth %d", header.Depth)
	}

	// 3. Store loaded addresses for WriteAt() support (RMW)
	bt.loadedHeaderAddress = headerAddr
	bt.loadedLeafAddress = header.RootNodeAddr

	// 4. Store header
	bt.header = header
	bt.nodeSize = header.NodeSize

	// 5. Read and decode leaf node (if not empty)
	if header.NumRecordsRoot > 0 {
		leaf, records, err := readBTreeV2LeafNode(r, header.RootNodeAddr, int(header.NumRecordsRoot), header.Type, sb)
		if err != nil {
			return fmt.Errorf("failed to read leaf node: %w", err)
		}

		bt.leaf = leaf
		bt.records = records
	} else {
		// Empty tree
		bt.leaf = &BTreeV2LeafNode{
			Signature: [4]byte{'B', 'T', 'L', 'F'},
			Version:   0,
			Type:      header.Type,
			Records:   make([]LinkNameRecord, 0),
		}
		bt.records = make([]LinkNameRecord, 0)
	}

	return nil
}

// readBTreeV2Header reads B-tree v2 header from file.
//
// Format (from H5B2cache.c - H5B2__hdr_deserialize):
//   - Signature: "BTHD" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 = Link Name Index (1 byte)
//   - Node Size: 4 bytes
//   - Record Size: 2 bytes
//   - Depth: 2 bytes
//   - Split Percent: 1 byte
//   - Merge Percent: 1 byte
//   - Root Node Address: offsetSize bytes
//   - Number of Records in Root: 2 bytes
//   - Total Records: 8 bytes
//   - Checksum: CRC32 (4 bytes)
func readBTreeV2Header(r io.ReaderAt, address uint64, sb *core.Superblock) (*BTreeV2Header, error) {
	// Calculate header size
	size := 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + int(sb.OffsetSize) + 2 + 8 + 4

	// Read header data
	buf := make([]byte, size)
	//nolint:gosec // G115: address conversion, valid for file I/O
	n, err := r.ReadAt(buf, int64(address))
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("failed to read header at 0x%X: %w", address, err)
	}
	if n < size {
		return nil, fmt.Errorf("incomplete header read: got %d bytes, want %d", n, size)
	}

	offset := 0

	// Signature (4 bytes)
	var signature [4]byte
	copy(signature[:], buf[offset:offset+4])
	if string(signature[:]) != BTreeV2HeaderSignature {
		return nil, fmt.Errorf("invalid B-tree header signature: got %q, want %q", signature, BTreeV2HeaderSignature)
	}
	offset += 4

	// Version (1 byte)
	version := buf[offset]
	if version != 0 {
		return nil, fmt.Errorf("unsupported B-tree version: %d", version)
	}
	offset++

	// Type (1 byte)
	btreeType := buf[offset]
	offset++

	// Node Size (4 bytes)
	nodeSize := binary.LittleEndian.Uint32(buf[offset : offset+4])
	offset += 4

	// Record Size (2 bytes)
	recordSize := binary.LittleEndian.Uint16(buf[offset : offset+2])
	offset += 2

	// Depth (2 bytes)
	depth := binary.LittleEndian.Uint16(buf[offset : offset+2])
	offset += 2

	// Split Percent (1 byte)
	splitPercent := buf[offset]
	offset++

	// Merge Percent (1 byte)
	mergePercent := buf[offset]
	offset++

	// Root Node Address (offsetSize bytes)
	rootNodeAddr := utils.ReadUint(buf[offset:offset+int(sb.OffsetSize)], int(sb.OffsetSize), sb.Endianness)
	offset += int(sb.OffsetSize)

	// Number of Records in Root (2 bytes)
	numRecordsRoot := binary.LittleEndian.Uint16(buf[offset : offset+2])
	offset += 2

	// Total Records (8 bytes)
	totalRecords := binary.LittleEndian.Uint64(buf[offset : offset+8])
	offset += 8

	// Checksum (Jenkins lookup3, 4 bytes)
	storedChecksum := binary.LittleEndian.Uint32(buf[offset : offset+4])
	expectedChecksum := core.JenkinsChecksum(buf[:offset])
	if storedChecksum != expectedChecksum {
		return nil, fmt.Errorf("b-tree header checksum mismatch: got 0x%X, want 0x%X", storedChecksum, expectedChecksum)
	}

	return &BTreeV2Header{
		Signature:      signature,
		Version:        version,
		Type:           btreeType,
		NodeSize:       nodeSize,
		RecordSize:     recordSize,
		Depth:          depth,
		SplitPercent:   splitPercent,
		MergePercent:   mergePercent,
		RootNodeAddr:   rootNodeAddr,
		NumRecordsRoot: numRecordsRoot,
		TotalRecords:   totalRecords,
	}, nil
}

// readBTreeV2LeafNode reads B-tree v2 leaf node from file.
//
// Format (from H5B2cache.c - H5B2__leaf_deserialize):
//   - Signature: "BTLF" (4 bytes)
//   - Version: 0 (1 byte)
//   - Type: 5 (1 byte)
//   - Records: Array of link name records (numRecords * 11 bytes)
//   - Checksum: CRC32 (4 bytes)
func readBTreeV2LeafNode(r io.ReaderAt, address uint64, numRecords int, recType uint8, _ *core.Superblock) (*BTreeV2LeafNode, []LinkNameRecord, error) {
	recSize := linkRecordSize
	if recType == BTreeV2TypeAttrNameIndex {
		recSize = attrRecordSize
	}
	// Calculate leaf size
	size := 4 + 1 + 1 + (numRecords * recSize) + 4

	// Read leaf data
	buf := make([]byte, size)
	//nolint:gosec // G115: address conversion, valid for file I/O
	n, err := r.ReadAt(buf, int64(address))
	if err != nil && err != io.EOF {
		return nil, nil, fmt.Errorf("failed to read leaf at 0x%X: %w", address, err)
	}
	if n < size {
		return nil, nil, fmt.Errorf("incomplete leaf read: got %d bytes, want %d", n, size)
	}

	offset := 0

	// Signature (4 bytes)
	var signature [4]byte
	copy(signature[:], buf[offset:offset+4])
	if string(signature[:]) != BTreeV2LeafSignature {
		return nil, nil, fmt.Errorf("invalid B-tree leaf signature: got %q, want %q", signature, BTreeV2LeafSignature)
	}
	offset += 4

	// Version (1 byte)
	version := buf[offset]
	if version != 0 {
		return nil, nil, fmt.Errorf("unsupported B-tree leaf version: %d", version)
	}
	offset++

	// Type (1 byte)
	leafType := buf[offset]
	offset++

	// Records (each record is 11 bytes: 4 bytes hash + 7 bytes heap ID)
	records := make([]LinkNameRecord, numRecords)
	for i := 0; i < numRecords; i++ {
		var nameHash uint32
		var heapID [7]byte
		if recType == BTreeV2TypeAttrNameIndex {
			copy(heapID[:], buf[offset:offset+7])
			nameHash = binary.LittleEndian.Uint32(buf[offset+13 : offset+17])
			offset += attrRecordSize
		} else {
			nameHash = binary.LittleEndian.Uint32(buf[offset : offset+4])
			copy(heapID[:], buf[offset+4:offset+11])
			offset += linkRecordSize
		}

		records[i] = LinkNameRecord{
			NameHash: nameHash,
			HeapID:   heapID,
		}
	}

	// Checksum (Jenkins lookup3, 4 bytes)
	storedChecksum := binary.LittleEndian.Uint32(buf[offset : offset+4])
	expectedChecksum := core.JenkinsChecksum(buf[:offset])
	if storedChecksum != expectedChecksum {
		return nil, nil, fmt.Errorf("b-tree leaf checksum mismatch: got 0x%X, want 0x%X", storedChecksum, expectedChecksum)
	}

	leaf := &BTreeV2LeafNode{
		Signature: signature,
		Version:   version,
		Type:      leafType,
		Records:   records,
	}

	return leaf, records, nil
}

// jenkinsHash computes Jenkins hash (lookup3) for a string.
//
// This is the hash function used by HDF5 for link name indexing.
// Based on Bob Jenkins' lookup3 hash algorithm.
//
// Reference:
//   - H5checksum.c - H5_checksum_lookup3()
//   - http://burtleburtle.net/bob/hash/doobs.html
func jenkinsHash(name string) uint32 {
	// Jenkins lookup3 hash implementation (simplified version of the C library's)

	length := len(name)
	a, b, c := uint32(0xdeadbeef)+uint32(length), uint32(0xdeadbeef)+uint32(length), uint32(0xdeadbeef)+uint32(length) //nolint:gosec // G115: Jenkins hash algorithm, length is string length

	// Process 12-byte chunks
	i := 0
	for i+12 <= length {
		a += uint32(name[i]) | uint32(name[i+1])<<8 | uint32(name[i+2])<<16 | uint32(name[i+3])<<24
		b += uint32(name[i+4]) | uint32(name[i+5])<<8 | uint32(name[i+6])<<16 | uint32(name[i+7])<<24
		c += uint32(name[i+8]) | uint32(name[i+9])<<8 | uint32(name[i+10])<<16 | uint32(name[i+11])<<24

		// Mix
		a -= c
		a ^= (c << 4) | (c >> 28)
		c += b
		b -= a
		b ^= (a << 6) | (a >> 26)
		a += c
		c -= b
		c ^= (b << 8) | (b >> 24)
		b += a
		a -= c
		a ^= (c << 16) | (c >> 16)
		c += b
		b -= a
		b ^= (a << 19) | (a >> 13)
		a += c
		c -= b
		c ^= (b << 4) | (b >> 28)
		b += a

		i += 12
	}

	// Handle remaining bytes
	remaining := length - i
	switch remaining {
	case 11:
		c += uint32(name[i+10]) << 16
		fallthrough
	case 10:
		c += uint32(name[i+9]) << 8
		fallthrough
	case 9:
		c += uint32(name[i+8])
		fallthrough
	case 8:
		b += uint32(name[i+7]) << 24
		fallthrough
	case 7:
		b += uint32(name[i+6]) << 16
		fallthrough
	case 6:
		b += uint32(name[i+5]) << 8
		fallthrough
	case 5:
		b += uint32(name[i+4])
		fallthrough
	case 4:
		a += uint32(name[i+3]) << 24
		fallthrough
	case 3:
		a += uint32(name[i+2]) << 16
		fallthrough
	case 2:
		a += uint32(name[i+1]) << 8
		fallthrough
	case 1:
		a += uint32(name[i])
	}

	// Final mix
	c ^= b
	c -= (b << 14) | (b >> 18)
	a ^= c
	a -= (c << 11) | (c >> 21)
	b ^= a
	b -= (a << 25) | (a >> 7)
	c ^= b
	c -= (b << 16) | (b >> 16)
	a ^= c
	a -= (c << 4) | (c >> 28)
	b ^= a
	b -= (a << 14) | (a >> 18)
	c ^= b
	c -= (b << 24) | (b >> 8)

	return c
}
