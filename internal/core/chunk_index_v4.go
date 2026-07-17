package core

import (
	"errors"
	"fmt"
	"io"

	"github.com/scigolib/hdf5/internal/utils"
)

// ChunkLocation describes one stored chunk located through a version 4/5 data
// layout chunk index. It is the common currency the modern chunk indexes
// (Single Chunk, Implicit, Fixed Array, Extensible Array, v2 B-tree) produce
// so the shared chunk assembly + filter pipeline can consume them uniformly.
type ChunkLocation struct {
	Scaled     []uint64 // Chunk grid coordinates, one per dataset dimension.
	Address    uint64   // File address of the stored chunk.
	NBytes     uint64   // Stored size in bytes (compressed size for filtered chunks).
	FilterMask uint32   // Per-chunk filter exclusion mask (0 = all filters applied).
}

// CollectChunksV4 lists all allocated chunks of a dataset whose data layout
// message is version 4 or 5, by walking the chunk index named in the message.
// Chunks that exist in the index but were never allocated (undefined address)
// are omitted; callers treat missing chunks as fill value.
func CollectChunksV4(r io.ReaderAt, layout *DataLayoutMessage, dataspace *DataspaceMessage, sb *Superblock) ([]ChunkLocation, error) {
	if len(layout.ChunkSize) == 0 {
		return nil, errors.New("chunked layout without chunk dimensions")
	}
	if isUndefinedAddress(layout.DataAddress, sb.OffsetSize) {
		// Index was never allocated: no chunks written yet.
		return nil, nil
	}

	grid, err := chunkGridDims(dataspace.Dimensions, layout.ChunkSize)
	if err != nil {
		return nil, err
	}

	switch layout.ChunkIndexType {
	case ChunkIndexSingle:
		return singleChunkEntries(layout, grid), nil
	case ChunkIndexImplicit:
		return implicitChunkEntries(layout, grid), nil
	case ChunkIndexFixed:
		return fixedArrayChunkEntries(r, layout, sb, grid)
	case ChunkIndexExt:
		return extensibleArrayChunkEntries(r, layout, sb, grid)
	case ChunkIndexBTreeV2:
		return btreeV2ChunkEntries(r, layout, sb, grid)
	default:
		return nil, fmt.Errorf("unsupported chunk index type: %d", layout.ChunkIndexType)
	}
}

// chunkStoreBytes returns the on-disk size of one unfiltered chunk. The layout
// message stores the element size as a trailing pseudo-dimension, so the plain
// product of all chunk "dimensions" is already a byte count.
func (dl *DataLayoutMessage) chunkStoreBytes() uint64 {
	size := uint64(1)
	for _, d := range dl.ChunkSize {
		size *= d
	}
	return size
}

// chunkGridDims computes the number of chunks along each dataset dimension
// (ceiling division). chunkDims may carry the trailing element-size entry.
func chunkGridDims(dims, chunkDims []uint64) ([]uint64, error) {
	if len(chunkDims) < len(dims) {
		return nil, fmt.Errorf("chunk rank %d smaller than dataset rank %d", len(chunkDims), len(dims))
	}
	grid := make([]uint64, len(dims))
	for i, d := range dims {
		if chunkDims[i] == 0 {
			return nil, fmt.Errorf("zero-size chunk dimension %d", i)
		}
		grid[i] = (d + chunkDims[i] - 1) / chunkDims[i]
	}
	return grid, nil
}

// gridChunkCount returns the total number of chunk slots in the grid.
func gridChunkCount(grid []uint64) uint64 {
	n := uint64(1)
	for _, g := range grid {
		n *= g
	}
	return n
}

// linearToScaled converts a row-major linear chunk index into per-dimension
// chunk grid coordinates.
func linearToScaled(idx uint64, grid []uint64) []uint64 {
	scaled := make([]uint64, len(grid))
	for i := len(grid) - 1; i >= 0; i-- {
		scaled[i] = idx % grid[i]
		idx /= grid[i]
	}
	return scaled
}

// isUndefinedAddress reports whether addr is the HDF5 "undefined address"
// marker (all bytes 0xFF at the file's offset size).
func isUndefinedAddress(addr uint64, offsetSize uint8) bool {
	if offsetSize >= 8 {
		return addr == ^uint64(0)
	}
	return addr == (uint64(1)<<(8*offsetSize))-1
}

// floorLog2 returns floor(log2(v)) for v >= 1 (0 for v <= 1).
func floorLog2(v uint64) uint64 {
	var n uint64
	for v > 1 {
		v >>= 1
		n++
	}
	return n
}

// readIndexBlob reads a chunk index structure of totalSize bytes at addr and
// verifies its 4-byte signature and 1-byte version (always 0 for the modern
// chunk index structures).
func readIndexBlob(r io.ReaderAt, addr, totalSize uint64, sig string) ([]byte, error) {
	if err := utils.ValidateBufferSize(totalSize, utils.MaxChunkSize, "chunk index block"); err != nil {
		return nil, fmt.Errorf("chunk index block at 0x%x: %w", addr, err)
	}
	buf := make([]byte, totalSize)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(buf, int64(addr)); err != nil {
		return nil, fmt.Errorf("failed to read %s block at 0x%x: %w", sig, addr, err)
	}
	if string(buf[0:4]) != sig {
		return nil, fmt.Errorf("invalid chunk index signature: %q (want %q)", buf[0:4], sig)
	}
	if buf[4] != 0 {
		return nil, fmt.Errorf("unsupported %s version: %d", sig, buf[4])
	}
	return buf, nil
}

// singleChunkEntries builds the one-entry chunk list of a Single Chunk index:
// the index address IS the chunk address (H5Dsingle.c).
func singleChunkEntries(layout *DataLayoutMessage, grid []uint64) []ChunkLocation {
	entry := ChunkLocation{
		Scaled:  make([]uint64, len(grid)),
		Address: layout.DataAddress,
		NBytes:  layout.chunkStoreBytes(),
	}
	if layout.Flags&LayoutChunkSingleIndexWithFilter != 0 {
		entry.NBytes = layout.SingleChunkNBytes
		entry.FilterMask = layout.SingleChunkFilterMask
	}
	return []ChunkLocation{entry}
}

// implicitChunkEntries builds the chunk list of an Implicit index: chunks are
// stored contiguously in row-major chunk order starting at the index address,
// with no per-chunk metadata (H5Dnone.c). Implicit indexes are never filtered.
func implicitChunkEntries(layout *DataLayoutMessage, grid []uint64) []ChunkLocation {
	chunkBytes := layout.chunkStoreBytes()
	n := gridChunkCount(grid)
	entries := make([]ChunkLocation, 0, n)
	for i := uint64(0); i < n; i++ {
		entries = append(entries, ChunkLocation{
			Scaled:  linearToScaled(i, grid),
			Address: layout.DataAddress + i*chunkBytes,
			NBytes:  chunkBytes,
		})
	}
	return entries
}

// decodeChunkElement decodes one fixed/extensible array element into a chunk
// entry. Client ID 0 (non-filtered chunks) stores just the address; client ID
// 1 (filtered chunks) stores address + variable-width stored size + 4-byte
// filter mask. Returns ok=false for unallocated slots (undefined address).
func decodeChunkElement(elt []byte, clientID uint8, sb *Superblock, layout *DataLayoutMessage) (ChunkLocation, bool, error) {
	offsetSize := int(sb.OffsetSize)
	var entry ChunkLocation

	switch clientID {
	case 0:
		if len(elt) < offsetSize {
			return entry, false, errors.New("chunk index element too short")
		}
		entry.Address = readUint64(elt, offsetSize, sb.Endianness)
		entry.NBytes = layout.chunkStoreBytes()
	case 1:
		sizeLen := len(elt) - offsetSize - 4
		if sizeLen < 1 || sizeLen > 8 {
			return entry, false, fmt.Errorf("invalid filtered chunk element size: %d bytes", len(elt))
		}
		entry.Address = readUint64(elt, offsetSize, sb.Endianness)
		entry.NBytes = readUint64(elt[offsetSize:], sizeLen, sb.Endianness)
		entry.FilterMask = sb.Endianness.Uint32(elt[offsetSize+sizeLen:])
	default:
		return entry, false, fmt.Errorf("unsupported chunk index client id: %d", clientID)
	}

	if isUndefinedAddress(entry.Address, sb.OffsetSize) {
		return entry, false, nil
	}
	return entry, true, nil
}

// fixedArrayHeader holds the decoded "FAHD" fields of a Fixed Array index.
type fixedArrayHeader struct {
	clientID  uint8
	entrySize uint64
	maxNelmts uint64
	dblkAddr  uint64
}

// parseFixedArrayHeader reads and validates a Fixed Array header ("FAHD"):
// sig(4) version(1) client(1) entry size(1) page bits(1) max nelmts(lengthSize)
// data block addr(offsetSize) checksum(4). Reference: H5FAhdr.c.
func parseFixedArrayHeader(r io.ReaderAt, addr uint64, sb *Superblock) (*fixedArrayHeader, error) {
	lengthSize := int(sb.LengthSize)
	hdrSize := uint64(8) + uint64(sb.LengthSize) + uint64(sb.OffsetSize) + 4
	buf, err := readIndexBlob(r, addr, hdrSize, "FAHD")
	if err != nil {
		return nil, err
	}

	hdr := &fixedArrayHeader{
		clientID:  buf[5],
		entrySize: uint64(buf[6]),
		maxNelmts: readUint64(buf[8:], lengthSize, sb.Endianness),
		dblkAddr:  readUint64(buf[8+lengthSize:], int(sb.OffsetSize), sb.Endianness),
	}
	pageBits := buf[7]

	if hdr.entrySize == 0 {
		return nil, errors.New("invalid fixed array entry size")
	}
	if pageBits < 64 && hdr.maxNelmts > uint64(1)<<pageBits {
		return nil, errors.New("paged fixed array data blocks not yet supported")
	}
	return hdr, nil
}

// fixedArrayChunkEntries reads a Fixed Array chunk index ("FAHD" header plus
// "FADB" data block). Reference: H5FAhdr.c / H5FAdblock.c / H5Dfarray.c.
func fixedArrayChunkEntries(r io.ReaderAt, layout *DataLayoutMessage, sb *Superblock, grid []uint64) ([]ChunkLocation, error) {
	hdr, err := parseFixedArrayHeader(r, layout.DataAddress, sb)
	if err != nil {
		return nil, err
	}
	if isUndefinedAddress(hdr.dblkAddr, sb.OffsetSize) {
		return nil, nil
	}

	// Data block: sig(4) version(1) client(1) header addr(offsetSize)
	// elements(maxNelmts x entrySize) checksum(4). (No page bitmap: unpaged.)
	dblkHdrSize := uint64(6) + uint64(sb.OffsetSize)
	eltBytes, err := utils.SafeMultiply(hdr.maxNelmts, hdr.entrySize)
	if err != nil {
		return nil, fmt.Errorf("fixed array size overflow: %w", err)
	}
	dblk, err := readIndexBlob(r, hdr.dblkAddr, dblkHdrSize+eltBytes+4, "FADB")
	if err != nil {
		return nil, err
	}
	if dblk[5] != hdr.clientID {
		return nil, fmt.Errorf("fixed array client id mismatch: header %d, data block %d", hdr.clientID, dblk[5])
	}

	nChunks := gridChunkCount(grid)
	if hdr.maxNelmts < nChunks {
		nChunks = hdr.maxNelmts
	}

	entries := make([]ChunkLocation, 0, nChunks)
	for i := uint64(0); i < nChunks; i++ {
		off := dblkHdrSize + i*hdr.entrySize
		entry, ok, err := decodeChunkElement(dblk[off:off+hdr.entrySize], hdr.clientID, sb, layout)
		if err != nil {
			return nil, fmt.Errorf("fixed array element %d: %w", i, err)
		}
		if !ok {
			continue
		}
		entry.Scaled = linearToScaled(i, grid)
		entries = append(entries, entry)
	}
	return entries, nil
}

// extArrayInfo carries the extensible array geometry derived from the "EAHD"
// header creation parameters. Reference: H5EAhdr.c / H5EAiblock.c / H5EApkg.h.
type extArrayInfo struct {
	clientID    uint8
	eltSize     uint64   // Raw element size in bytes.
	idxBlkElmts uint64   // Elements stored directly in the index block.
	arrOffSize  uint64   // Width of block offset fields: ceil(maxNelmtsBits/8).
	pageNelmts  uint64   // Elements per data block page (2^maxDblkPageNelmtsBits).
	iblkNsblks  uint64   // Super blocks whose data blocks the index block points to.
	nDblkAddrs  uint64   // Data block addresses stored in the index block.
	nSblkAddrs  uint64   // Super block addresses stored in the index block.
	sblkNdblks  []uint64 // Per super block: number of data blocks.
	sblkNelmts  []uint64 // Per super block: elements per data block.
	sblkStart   []uint64 // Per super block: first element index it covers.
	sblkDblk    []uint64 // Per super block: index of its first data block.
}

// newExtArrayInfo validates the EAHD creation parameters and precomputes the
// super block geometry, mirroring H5EA__hdr_init and H5EA__iblock_alloc:
//
//	nsblks       = 1 + (maxNelmtsBits - log2(dataBlkMinElmts))
//	iblkNsblks   = 2 * log2(supBlkMinDataPtrs)
//	nDblkAddrs   = 2 * (supBlkMinDataPtrs - 1)
//	sblk u: ndblks = 2^(u/2), dblkNelmts = 2^((u+1)/2) * dataBlkMinElmts
func newExtArrayInfo(clientID, eltSize, maxNelmtsBits, idxBlkElmts, dataBlkMinElmts, supBlkMinDataPtrs, pageBits uint8) (*extArrayInfo, error) {
	dmin := uint64(dataBlkMinElmts)
	smin := uint64(supBlkMinDataPtrs)
	if eltSize == 0 || dmin == 0 || smin < 2 || dmin&(dmin-1) != 0 || smin&(smin-1) != 0 {
		return nil, errors.New("invalid extensible array creation parameters")
	}
	if maxNelmtsBits == 0 || maxNelmtsBits > 64 {
		return nil, fmt.Errorf("invalid extensible array max nelmts bits: %d", maxNelmtsBits)
	}

	info := &extArrayInfo{
		clientID:    clientID,
		eltSize:     uint64(eltSize),
		idxBlkElmts: uint64(idxBlkElmts),
		arrOffSize:  (uint64(maxNelmtsBits) + 7) / 8,
		pageNelmts:  uint64(1) << pageBits,
		iblkNsblks:  2 * floorLog2(smin),
		nDblkAddrs:  2 * (smin - 1),
	}

	nsblks := 1 + uint64(maxNelmtsBits) - floorLog2(dmin)
	if info.iblkNsblks > nsblks {
		return nil, errors.New("inconsistent extensible array geometry")
	}
	info.nSblkAddrs = nsblks - info.iblkNsblks

	info.sblkNdblks = make([]uint64, nsblks)
	info.sblkNelmts = make([]uint64, nsblks)
	info.sblkStart = make([]uint64, nsblks)
	info.sblkDblk = make([]uint64, nsblks)
	startIdx, startDblk := uint64(0), uint64(0)
	for u := uint64(0); u < nsblks; u++ {
		info.sblkNdblks[u] = uint64(1) << (u / 2)
		info.sblkNelmts[u] = (uint64(1) << ((u + 1) / 2)) * dmin
		info.sblkStart[u] = startIdx
		info.sblkDblk[u] = startDblk
		startIdx += info.sblkNdblks[u] * info.sblkNelmts[u]
		startDblk += info.sblkNdblks[u]
	}
	return info, nil
}

// extArrayWalker resolves extensible array element slots to their raw bytes,
// caching the super/data blocks it loads along the way.
type extArrayWalker struct {
	r         io.ReaderAt
	sb        *Superblock
	info      *extArrayInfo
	iblk      []byte
	dblkCache map[uint64][]byte
	sblkCache map[uint64][]byte
}

// element returns the raw element bytes for array index idx, or ok=false when
// the covering super/data block was never allocated.
func (w *extArrayWalker) element(idx uint64) ([]byte, bool, error) {
	info := w.info
	offU := uint64(w.sb.OffsetSize)
	iblkHdrSize := 6 + offU

	if idx < info.idxBlkElmts {
		off := iblkHdrSize + idx*info.eltSize
		return w.iblk[off : off+info.eltSize], true, nil
	}

	// Element lives in a data block; locate its super block slot.
	// Mirrors H5EA__dblock_sblk_idx: sblk = log2(t/dataBlkMinElmts + 1).
	t := idx - info.idxBlkElmts
	sblk := floorLog2(t/info.sblkNelmts[0] + 1)
	if sblk >= uint64(len(info.sblkNdblks)) {
		return nil, false, fmt.Errorf("chunk index %d beyond extensible array capacity", idx)
	}
	within := t - info.sblkStart[sblk]
	dblkIdx := within / info.sblkNelmts[sblk]
	elmtIdx := within % info.sblkNelmts[sblk]
	if info.sblkNelmts[sblk] > info.pageNelmts {
		return nil, false, errors.New("paged extensible array data blocks not yet supported")
	}

	dblkAddr, ok, err := w.dataBlockAddress(sblk, dblkIdx)
	if err != nil || !ok {
		return nil, false, err
	}

	dblk, cached := w.dblkCache[dblkAddr]
	if !cached {
		dblk, err = w.readDataBlock(dblkAddr, info.sblkNelmts[sblk])
		if err != nil {
			return nil, false, err
		}
		w.dblkCache[dblkAddr] = dblk
	}
	off := elmtIdx * info.eltSize
	return dblk[off : off+info.eltSize], true, nil
}

// dataBlockAddress resolves the file address of data block dblkIdx of super
// block slot sblk, reading through the index block or an "EASB" super block.
func (w *extArrayWalker) dataBlockAddress(sblk, dblkIdx uint64) (uint64, bool, error) {
	info := w.info
	offU := uint64(w.sb.OffsetSize)
	dblkAddrsOff := 6 + offU + info.idxBlkElmts*info.eltSize

	var dblkAddr uint64
	if sblk < info.iblkNsblks {
		off := dblkAddrsOff + (info.sblkDblk[sblk]+dblkIdx)*offU
		dblkAddr = readUint64(w.iblk[off:], int(w.sb.OffsetSize), w.sb.Endianness)
	} else {
		var ok bool
		var err error
		dblkAddr, ok, err = w.superBlockDataBlockAddress(sblk, dblkIdx)
		if err != nil || !ok {
			return 0, false, err
		}
	}

	if isUndefinedAddress(dblkAddr, w.sb.OffsetSize) {
		return 0, false, nil
	}
	return dblkAddr, true, nil
}

// superBlockDataBlockAddress resolves a data block address through an "EASB"
// super block referenced from the index block's super block address area.
func (w *extArrayWalker) superBlockDataBlockAddress(sblk, dblkIdx uint64) (uint64, bool, error) {
	info := w.info
	offU := uint64(w.sb.OffsetSize)
	sblkAddrsOff := 6 + offU + info.idxBlkElmts*info.eltSize + info.nDblkAddrs*offU

	off := sblkAddrsOff + (sblk-info.iblkNsblks)*offU
	sblkAddr := readUint64(w.iblk[off:], int(w.sb.OffsetSize), w.sb.Endianness)
	if isUndefinedAddress(sblkAddr, w.sb.OffsetSize) {
		return 0, false, nil
	}

	sblkData, cached := w.sblkCache[sblkAddr]
	if !cached {
		var err error
		sblkData, err = w.readSuperBlock(sblkAddr, info.sblkNdblks[sblk])
		if err != nil {
			return 0, false, err
		}
		w.sblkCache[sblkAddr] = sblkData
	}
	return readUint64(sblkData[dblkIdx*offU:], int(w.sb.OffsetSize), w.sb.Endianness), true, nil
}

// readSuperBlock reads an "EASB" super block and returns its raw data block
// address area (ndblks x offsetSize bytes). Layout: sig(4) version(1)
// client(1) header addr(offsetSize) block offset(arrOffSize)
// data block addrs(ndblks x offsetSize) checksum(4). Reference: H5EAsblock.c.
func (w *extArrayWalker) readSuperBlock(addr, ndblks uint64) ([]byte, error) {
	offU := uint64(w.sb.OffsetSize)
	hdrSize := 6 + offU + w.info.arrOffSize
	buf, err := readIndexBlob(w.r, addr, hdrSize+ndblks*offU+4, "EASB")
	if err != nil {
		return nil, err
	}
	if buf[5] != w.info.clientID {
		return nil, fmt.Errorf("extensible array client id mismatch in super block: %d", buf[5])
	}
	return buf[hdrSize : hdrSize+ndblks*offU], nil
}

// readDataBlock reads an unpaged "EADB" data block and returns its raw
// element area (nelmts x eltSize bytes). Layout: sig(4) version(1) client(1)
// header addr(offsetSize) block offset(arrOffSize) elements(nelmts x eltSize)
// checksum(4). Reference: H5EAdblock.c.
func (w *extArrayWalker) readDataBlock(addr, nelmts uint64) ([]byte, error) {
	offU := uint64(w.sb.OffsetSize)
	hdrSize := 6 + offU + w.info.arrOffSize
	buf, err := readIndexBlob(w.r, addr, hdrSize+nelmts*w.info.eltSize+4, "EADB")
	if err != nil {
		return nil, err
	}
	if buf[5] != w.info.clientID {
		return nil, fmt.Errorf("extensible array client id mismatch in data block: %d", buf[5])
	}
	return buf[hdrSize : hdrSize+nelmts*w.info.eltSize], nil
}

// extensibleArrayChunkEntries reads an Extensible Array chunk index: "EAHD"
// header, "EAIB" index block and the "EASB"/"EADB" blocks it references.
// Reference: H5EAhdr.c / H5EAiblock.c / H5EAsblock.c / H5EAdblock.c and
// H5Dearray.c. Creation parameters are taken from the EAHD itself (they must
// match the copy in the layout message).
func extensibleArrayChunkEntries(r io.ReaderAt, layout *DataLayoutMessage, sb *Superblock, grid []uint64) ([]ChunkLocation, error) {
	offU := uint64(sb.OffsetSize)
	lenU := uint64(sb.LengthSize)

	// Header: sig(4) version(1) client(1) element size(1) max nelmts bits(1)
	// index blk elmts(1) data blk min elmts(1) sup blk min data ptrs(1)
	// max dblk page nelmts bits(1) statistics(6 x lengthSize)
	// index block addr(offsetSize) checksum(4).
	hdr, err := readIndexBlob(r, layout.DataAddress, 12+6*lenU+offU+4, "EAHD")
	if err != nil {
		return nil, err
	}
	info, err := newExtArrayInfo(hdr[5], hdr[6], hdr[7], hdr[8], hdr[9], hdr[10], hdr[11])
	if err != nil {
		return nil, err
	}
	iblkAddr := readUint64(hdr[12+6*lenU:], int(sb.OffsetSize), sb.Endianness)
	if isUndefinedAddress(iblkAddr, sb.OffsetSize) {
		return nil, nil
	}

	// Index block: sig(4) version(1) client(1) header addr(offsetSize)
	// elements(idxBlkElmts x eltSize) data block addrs(nDblkAddrs x offsetSize)
	// super block addrs(nSblkAddrs x offsetSize) checksum(4).
	iblkSize := 6 + offU + info.idxBlkElmts*info.eltSize +
		(info.nDblkAddrs+info.nSblkAddrs)*offU + 4
	iblk, err := readIndexBlob(r, iblkAddr, iblkSize, "EAIB")
	if err != nil {
		return nil, err
	}

	walker := &extArrayWalker{
		r:         r,
		sb:        sb,
		info:      info,
		iblk:      iblk,
		dblkCache: make(map[uint64][]byte),
		sblkCache: make(map[uint64][]byte),
	}

	nChunks := gridChunkCount(grid)
	entries := make([]ChunkLocation, 0, nChunks)
	for idx := uint64(0); idx < nChunks; idx++ {
		elt, ok, err := walker.element(idx)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue // Covering block never allocated: chunk missing.
		}
		entry, ok, err := decodeChunkElement(elt, info.clientID, sb, layout)
		if err != nil {
			return nil, fmt.Errorf("extensible array element %d: %w", idx, err)
		}
		if !ok {
			continue
		}
		entry.Scaled = linearToScaled(idx, grid)
		entries = append(entries, entry)
	}
	return entries, nil
}

// Version 2 B-tree record types for dataset chunk indexes (H5Bpkg / spec).
const (
	btreeV2TypeChunksUnfiltered uint8 = 10
	btreeV2TypeChunksFiltered   uint8 = 11
)

// decodeBTreeV2ChunkRecord decodes one v2 B-tree chunk record. Type 10
// (unfiltered): chunk address + one 8-byte scaled offset per dataset
// dimension. Type 11 (filtered): address + variable-width stored size +
// 4-byte filter mask + scaled offsets. Reference: H5Dbtree2.c.
func decodeBTreeV2ChunkRecord(rec []byte, recType uint8, sizeLen, rank int, layout *DataLayoutMessage, sb *Superblock) ChunkLocation {
	entry := ChunkLocation{
		Address: readUint64(rec, int(sb.OffsetSize), sb.Endianness),
		NBytes:  layout.chunkStoreBytes(),
	}
	off := int(sb.OffsetSize)
	if recType == btreeV2TypeChunksFiltered {
		entry.NBytes = readUint64(rec[off:], sizeLen, sb.Endianness)
		off += sizeLen
		entry.FilterMask = sb.Endianness.Uint32(rec[off:])
		off += 4
	}
	entry.Scaled = make([]uint64, rank)
	for d := 0; d < rank; d++ {
		entry.Scaled[d] = sb.Endianness.Uint64(rec[off:])
		off += 8
	}
	return entry
}

// btreeV2ChunkEntries reads a version 2 B-tree chunk index (record types 10
// and 11). Only depth-0 trees (all records in the root leaf node) are
// supported so far.
func btreeV2ChunkEntries(r io.ReaderAt, layout *DataLayoutMessage, sb *Superblock, grid []uint64) ([]ChunkLocation, error) {
	offsetSize := int(sb.OffsetSize)
	rank := len(grid)

	header, err := readBTreeV2HeaderRaw(r, layout.DataAddress, sb)
	if err != nil {
		return nil, fmt.Errorf("failed to read chunk v2 b-tree header: %w", err)
	}
	if header.Type != btreeV2TypeChunksUnfiltered && header.Type != btreeV2TypeChunksFiltered {
		return nil, fmt.Errorf("unexpected v2 b-tree record type for chunk index: %d", header.Type)
	}
	if header.Depth != 0 {
		return nil, fmt.Errorf("chunk v2 b-tree depth %d not yet supported", header.Depth)
	}
	if header.NumRecordsRoot == 0 || isUndefinedAddress(header.RootNodeAddr, sb.OffsetSize) {
		return nil, nil
	}

	recSize := int(header.RecordSize)
	sizeLen := 0
	if header.Type == btreeV2TypeChunksFiltered {
		sizeLen = recSize - offsetSize - 4 - 8*rank
		if sizeLen < 1 || sizeLen > 8 {
			return nil, fmt.Errorf("invalid filtered chunk record size: %d", recSize)
		}
	} else if recSize != offsetSize+8*rank {
		return nil, fmt.Errorf("invalid chunk record size %d for rank %d", recSize, rank)
	}

	// Leaf node: sig "BTLF"(4) version(1) type(1) records(n x recSize) checksum(4).
	nRecords := int(header.NumRecordsRoot)
	leafSize := 6 + uint64(header.NumRecordsRoot)*uint64(header.RecordSize) + 4
	buf, err := readIndexBlob(r, header.RootNodeAddr, leafSize, "BTLF")
	if err != nil {
		return nil, err
	}
	if buf[5] != header.Type {
		return nil, fmt.Errorf("v2 b-tree leaf record type mismatch: %d", buf[5])
	}

	entries := make([]ChunkLocation, 0, nRecords)
	for i := 0; i < nRecords; i++ {
		rec := buf[6+i*recSize : 6+(i+1)*recSize]
		entries = append(entries, decodeBTreeV2ChunkRecord(rec, header.Type, sizeLen, rank, layout, sb))
	}
	return entries, nil
}
