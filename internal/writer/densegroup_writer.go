// Copyright (c) 2025 SciGo HDF5 Library Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package writer

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/shyrmapp/hdf5/internal/structures"
	"github.com/shyrmapp/hdf5/internal/utils"
)

// DenseGroupWriter manages dense group creation.
//
// Dense groups (HDF5 1.8+) use:
//   - Link Info Message: Metadata about link storage
//   - Fractal Heap: Storage for link names and messages
//   - B-tree v2: Index for fast link lookup by name
//
// This coordinator:
//  1. Creates Fractal Heap for link storage
//  2. Creates B-tree v2 for link indexing
//  3. Stores link names and metadata in heap
//  4. Indexes links in B-tree
//  5. Builds Link Info Message with addresses
//  6. Constructs object header with all messages
//
// Reference: H5Gdense.c - H5G_dense_create(), H5G_dense_insert().
type DenseGroupWriter struct {
	name string

	// Components
	fractalHeap *structures.WritableFractalHeap
	btree       *structures.WritableBTreeV2
	linkInfo    *core.LinkInfoMessage

	// Links to add
	links []denseLink
}

// denseLink represents a link to be added to dense group.
type denseLink struct {
	name       string
	targetAddr uint64 // For hard links
	// Future: soft link path, external link file+path
}

// NewDenseGroupWriter creates new dense group writer.
//
// Parameters:
//   - name: Group name (for error messages)
//
// Returns:
//   - DenseGroupWriter ready to accept links
//
// Reference: H5Gdense.c - H5G_dense_create().
func NewDenseGroupWriter(name string) *DenseGroupWriter {
	return &DenseGroupWriter{
		name:        name,
		fractalHeap: structures.NewWritableFractalHeap(512 * 1024), // 512KB default
		btree:       structures.NewWritableBTreeV2(4096),           // 4KB node
		linkInfo: &core.LinkInfoMessage{
			Version: 0,
			Flags:   0, // No creation order tracking
		},
		links: make([]denseLink, 0),
	}
}

// AddLink adds hard link to dense group.
//
// Only hard links are supported (targetAddr points to object header).
// Future: Soft links, external links
//
// Parameters:
//   - name: Link name (UTF-8 string)
//   - targetAddr: File address of target object header
//
// Returns:
//   - error if name empty, duplicate, or invalid
//
// Reference: H5Gdense.c - H5G_dense_insert().
func (dgw *DenseGroupWriter) AddLink(name string, targetAddr uint64) error {
	if name == "" {
		return errors.New("link name cannot be empty")
	}

	// Check for duplicates
	for _, link := range dgw.links {
		if link.name == name {
			return fmt.Errorf("duplicate link name: %s", name)
		}
	}

	dgw.links = append(dgw.links, denseLink{
		name:       name,
		targetAddr: targetAddr,
	})

	return nil
}

// WriteToFile writes dense group to file, returns object header address.
//
// This method:
//  1. For each link:
//     a. Create link message (hard link format)
//     b. Insert link message into fractal heap
//     c. Insert (name, heapID) into B-tree v2
//  2. Write fractal heap to file
//  3. Write B-tree v2 to file
//  4. Create Link Info Message with heap/B-tree addresses
//  5. Create object header with Link Info + other messages
//  6. Write object header to file
//
// Parameters:
//   - fw: FileWriter for write operations
//   - allocator: Space allocator
//   - sb: Superblock for encoding parameters
//
// Returns:
//   - uint64: File address of group's object header
//   - error: Non-nil if write fails
//
// Reference: H5Gdense.c - H5G_dense_create() + H5G_dense_insert().
func (dgw *DenseGroupWriter) WriteToFile(fw *FileWriter, allocator *Allocator, sb *core.Superblock) (uint64, error) {
	if len(dgw.links) == 0 {
		return 0, errors.New("dense group must have at least one link")
	}

	// Step 1: Process all links
	for _, link := range dgw.links {
		// 1a. Create link message (hard link format)
		linkMsg, err := dgw.createLinkMessage(link, sb)
		if err != nil {
			return 0, fmt.Errorf("failed to encode link %s: %w", link.name, err)
		}

		// 1b. Insert into fractal heap
		heapID, err := dgw.fractalHeap.InsertObject(linkMsg)
		if err != nil {
			return 0, fmt.Errorf("failed to insert link %s into heap: %w", link.name, err)
		}

		// 1c. Convert heap ID to uint64 for B-tree
		// Heap ID is 8 bytes, read as little-endian uint64
		var heapIDUint64 uint64
		if len(heapID) >= 8 {
			heapIDUint64 = binary.LittleEndian.Uint64(heapID[:8])
		} else {
			return 0, fmt.Errorf("invalid heap ID length for link %s: %d bytes", link.name, len(heapID))
		}

		// 1d. Insert into B-tree v2
		err = dgw.btree.InsertRecord(link.name, heapIDUint64)
		if err != nil {
			return 0, fmt.Errorf("failed to insert link %s into B-tree: %w", link.name, err)
		}
	}

	// Step 2: Write fractal heap
	heapAddr, err := dgw.fractalHeap.WriteToFile(fw, allocator, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to write fractal heap: %w", err)
	}

	// Step 3: Write B-tree v2
	btreeAddr, err := dgw.btree.WriteToFile(fw, allocator, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to write B-tree v2: %w", err)
	}

	// Step 4: Create Link Info Message
	dgw.linkInfo.FractalHeapAddress = heapAddr
	dgw.linkInfo.NameBTreeAddress = btreeAddr
	dgw.linkInfo.CreationOrderBTreeAddress = 0 // No creation order tracking

	// Step 5: Create object header with Link Info Message
	ohAddr, err := dgw.createObjectHeader(fw, allocator, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to create object header: %w", err)
	}

	return ohAddr, nil
}

// createLinkMessage encodes a hard link message for fractal heap storage.
//
// Reference: H5Olink.c - H5O__link_encode().
func (dgw *DenseGroupWriter) createLinkMessage(link denseLink, sb *core.Superblock) ([]byte, error) {
	addr := make([]byte, sb.OffsetSize)
	utils.PutUint(addr, link.targetAddr, int(sb.OffsetSize), sb.Endianness)

	// Bits 0-1 pick the width of the name length field.
	var lenCode uint8
	switch n := len(link.name); {
	case n > math.MaxUint32:
		lenCode = 3
	case n > math.MaxUint16:
		lenCode = 2
	case n > math.MaxUint8:
		lenCode = 1
	}

	return core.EncodeLinkMessage(&core.LinkMessage{
		Version:   1,
		Flags:     lenCode | core.LinkFlagCharSetBit,
		Type:      core.LinkTypeHard,
		CharSet:   1, // UTF-8
		Name:      link.name,
		LinkValue: addr,
	}, sb)
}

// createObjectHeader creates object header with Link Info Message.
//
// Messages to include:
//   - Link Info Message (type 0x0002)
//   - Dataspace Message (type 0x0001) - scalar for groups
//   - Datatype Message (type 0x0003) - opaque for groups (optional, skipped)
//
// Reference: H5Oobj.c - H5O_obj_create().
func (dgw *DenseGroupWriter) createObjectHeader(fw *FileWriter, allocator *Allocator, sb *core.Superblock) (uint64, error) {
	// Encode Link Info Message
	linkInfoData, err := core.EncodeLinkInfoMessage(dgw.linkInfo, sb)
	if err != nil {
		return 0, fmt.Errorf("failed to encode link info message: %w", err)
	}

	// Create dataspace message (scalar for groups)
	dataspaceMsg := createScalarDataspaceMessage()

	// Create object header with messages
	ohw := &core.ObjectHeaderWriter{
		Version: 2,
		Flags:   0,
		Messages: []core.MessageWriter{
			{Type: core.MsgLinkInfo, Data: linkInfoData},
			{Type: core.MsgDataspace, Data: dataspaceMsg},
		},
	}

	// Calculate object header size using the writer's own method
	headerSize := ohw.Size()

	// Allocate space for object header
	headerAddr, err := allocator.Allocate(headerSize)
	if err != nil {
		return 0, fmt.Errorf("failed to allocate object header: %w", err)
	}

	// Write object header
	writtenSize, err := ohw.WriteTo(fw, headerAddr)
	if err != nil {
		return 0, fmt.Errorf("failed to write object header: %w", err)
	}

	if writtenSize != headerSize {
		return 0, fmt.Errorf("header size mismatch: expected %d, wrote %d", headerSize, writtenSize)
	}

	return headerAddr, nil
}

// createScalarDataspaceMessage creates a dataspace message for scalar dataspace.
//
// Groups have scalar dataspace (0 dimensions).
//
// Format:
//   - Version: 1 (1 byte) or 2
//   - Dimensionality: 0 (1 byte)
//   - Flags: 0 (1 byte)
//   - Type: 0 = scalar (1 byte) - optional in v1
//
// Reference: H5Odtype.c - H5O__dspace_encode().
func createScalarDataspaceMessage() []byte {
	// Use version 1 for simplicity
	// Version 1: 1 (ver) + 1 (dims) + 1 (flags) + 5 (reserved) = 8 bytes
	buf := make([]byte, 8)

	buf[0] = 1 // Version 1
	buf[1] = 0 // Dimensionality = 0 (scalar)
	buf[2] = 0 // Flags = 0
	// Bytes 3-7: reserved (zeros)

	return buf
}
