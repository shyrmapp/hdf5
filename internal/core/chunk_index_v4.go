package core

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"github.com/shyrmapp/hdf5/internal/utils"
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
	if err := layout.ValidateChunkDims(len(dataspace.Dimensions), 0); err != nil {
		return nil, err
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
		return implicitChunkEntries(r, layout, grid)
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
		hi, lo := bits.Mul64(size, d)
		if hi != 0 {
			return math.MaxUint64 // saturate; size checks downstream reject it
		}
		size = lo
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
		grid[i] = d / chunkDims[i]
		if d%chunkDims[i] != 0 {
			grid[i]++
		}
	}
	return grid, nil
}

// gridChunkCount returns the total number of chunk slots in the grid.
func gridChunkCount(grid []uint64) (uint64, error) {
	n := uint64(1)
	for _, g := range grid {
		var err error
		if n, err = utils.SafeMultiply(n, g); err != nil {
			return 0, fmt.Errorf("chunk grid too large: %w", err)
		}
	}
	return n, nil
}

// maxUnsizedEntries caps implicit-index entries when the reader cannot report
// the file size.
const maxUnsizedEntries = 1 << 24

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
	if totalSize < 10 || totalSize > blockLimit(r, addr) {
		return nil, fmt.Errorf("%s block at 0x%x of %d bytes exceeds file", sig, addr, totalSize)
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
	if err := verifyChecksum(buf, sig, addr); err != nil {
		return nil, err
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
func implicitChunkEntries(r io.ReaderAt, layout *DataLayoutMessage, grid []uint64) ([]ChunkLocation, error) {
	chunkBytes := layout.chunkStoreBytes()
	n, err := gridChunkCount(grid)
	if err != nil {
		return nil, err
	}
	// All chunks are stored contiguously, so they must fit in the file.
	if n > blockLimit(r, layout.DataAddress)/max(chunkBytes, 1) {
		return nil, fmt.Errorf("implicit chunk index: %d chunks of %d bytes do not fit in file", n, chunkBytes)
	}
	if _, known := readerSize(r); !known && n > maxUnsizedEntries {
		return nil, fmt.Errorf("implicit chunk index: %d chunks exceed limit", n)
	}
	entries := make([]ChunkLocation, 0, min(n, 4096))
	for i := uint64(0); i < n; i++ {
		entries = append(entries, ChunkLocation{
			Scaled:  linearToScaled(i, grid),
			Address: layout.DataAddress + i*chunkBytes,
			NBytes:  chunkBytes,
		})
	}
	return entries, nil
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
		entry.Address = utils.ReadUint(elt, offsetSize, sb.Endianness)
		entry.NBytes = layout.chunkStoreBytes()
	case 1:
		sizeLen := len(elt) - offsetSize - 4
		if sizeLen < 1 || sizeLen > 8 {
			return entry, false, fmt.Errorf("invalid filtered chunk element size: %d bytes", len(elt))
		}
		entry.Address = utils.ReadUint(elt, offsetSize, sb.Endianness)
		entry.NBytes = utils.ReadUint(elt[offsetSize:], sizeLen, sb.Endianness)
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
	pageBits  uint8
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
		maxNelmts: utils.ReadUint(buf[8:], lengthSize, sb.Endianness),
		dblkAddr:  utils.ReadUint(buf[8+lengthSize:], int(sb.OffsetSize), sb.Endianness),
	}
	hdr.pageBits = buf[7]

	if hdr.entrySize == 0 {
		return nil, errors.New("invalid fixed array entry size")
	}
	if hdr.pageBits == 0 || hdr.pageBits > 32 {
		return nil, fmt.Errorf("invalid fixed array page bits: %d", hdr.pageBits)
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

	nChunks, err := gridChunkCount(grid)
	if err != nil {
		return nil, err
	}
	nChunks = min(nChunks, hdr.maxNelmts)

	// Data block: sig(4) version(1) client(1) header addr(offsetSize)
	// then either the elements (maxNelmts x entrySize) when unpaged, or a page
	// init bitmap when paged, then checksum(4).
	dblkHdrSize := uint64(6) + uint64(sb.OffsetSize)
	pageNelmts := uint64(1) << hdr.pageBits
	if hdr.maxNelmts <= pageNelmts {
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
		return decodeFixedArrayElements(dblk[dblkHdrSize:], 0, nChunks, hdr, sb, layout, grid, nil)
	}

	// Paged: the prefix is followed by npages pages of (pageNelmts x
	// entrySize + 4-byte checksum) each (H5FAdblock.c / H5FAcache.c); the
	// last page holds the remainder. Pages whose bitmap bit is clear were never
	// written and hold only undefined addresses.
	npages := (hdr.maxNelmts + pageNelmts - 1) / pageNelmts
	bitmapBytes := (npages + 7) / 8
	prefix, err := readIndexBlob(r, hdr.dblkAddr, dblkHdrSize+bitmapBytes+4, "FADB")
	if err != nil {
		return nil, err
	}
	if prefix[5] != hdr.clientID {
		return nil, fmt.Errorf("fixed array client id mismatch: header %d, data block %d", hdr.clientID, prefix[5])
	}
	bitmap := prefix[dblkHdrSize : dblkHdrSize+bitmapBytes]
	pageStride := pageNelmts*hdr.entrySize + 4
	firstPage := hdr.dblkAddr + dblkHdrSize + bitmapBytes + 4

	var entries []ChunkLocation
	for p := uint64(0); p*pageNelmts < nChunks; p++ {
		if bitmap[p/8]&(0x80>>(p%8)) == 0 {
			continue
		}
		count := min(pageNelmts, hdr.maxNelmts-p*pageNelmts)
		page, err := readPageBlob(r, firstPage+p*pageStride, count*hdr.entrySize+4)
		if err != nil {
			return nil, fmt.Errorf("fixed array page %d: %w", p, err)
		}
		got, err := decodeFixedArrayElements(page, p*pageNelmts, min(nChunks, p*pageNelmts+count), hdr, sb, layout, grid, entries)
		if err != nil {
			return nil, err
		}
		entries = got
	}
	return entries, nil
}

// readPageBlob reads a signature-less metadata page ending in a checksum.
func readPageBlob(r io.ReaderAt, addr, size uint64) ([]byte, error) {
	if size < 4 || size > blockLimit(r, addr) || size > utils.MaxChunkSize {
		return nil, fmt.Errorf("page at 0x%x of %d bytes exceeds file", addr, size)
	}
	buf := make([]byte, size)
	if _, err := r.ReadAt(buf, int64(addr)); err != nil { //nolint:gosec // G115: bounded by file size
		return nil, err
	}
	if err := verifyChecksum(buf, "FA page", addr); err != nil {
		return nil, err
	}
	return buf, nil
}

// decodeFixedArrayElements decodes elements [first, end) of a fixed array from
// elts (which starts at element first) and appends them to entries.
func decodeFixedArrayElements(elts []byte, first, end uint64, hdr *fixedArrayHeader, sb *Superblock, layout *DataLayoutMessage, grid []uint64, entries []ChunkLocation) ([]ChunkLocation, error) {
	for i := first; i < end; i++ {
		off := (i - first) * hdr.entrySize
		entry, ok, err := decodeChunkElement(elts[off:off+hdr.entrySize], hdr.clientID, sb, layout)
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
	if maxNelmtsBits == 0 || maxNelmtsBits > 48 { // HDF5 chunk indexes use 32
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
	skipTo    uint64 // When element reports ok=false: first index past the unallocated range.
}

// element returns the raw element bytes for array index idx, or ok=false when
// the covering super/data block was never allocated.
func (w *extArrayWalker) element(idx uint64) ([]byte, bool, error) {
	info := w.info
	offU := uint64(w.sb.OffsetSize)
	iblkHdrSize := 6 + offU
	w.skipTo = idx + 1

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
	w.skipTo = info.idxBlkElmts + info.sblkStart[sblk] + (dblkIdx+1)*info.sblkNelmts[sblk]

	dblkAddr, ok, err := w.dataBlockAddress(sblk, dblkIdx)
	if err != nil || !ok {
		return nil, false, err
	}
	if info.sblkNelmts[sblk] > info.pageNelmts {
		return nil, false, errors.New("paged extensible array data blocks not yet supported")
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
		dblkAddr = utils.ReadUint(w.iblk[off:], int(w.sb.OffsetSize), w.sb.Endianness)
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
	sblkAddr := utils.ReadUint(w.iblk[off:], int(w.sb.OffsetSize), w.sb.Endianness)
	if isUndefinedAddress(sblkAddr, w.sb.OffsetSize) {
		w.skipTo = info.idxBlkElmts + info.sblkStart[sblk] + info.sblkNdblks[sblk]*info.sblkNelmts[sblk]
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
	return utils.ReadUint(sblkData[dblkIdx*offU:], int(w.sb.OffsetSize), w.sb.Endianness), true, nil
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
	iblkAddr := utils.ReadUint(hdr[12+6*lenU:], int(sb.OffsetSize), sb.Endianness)
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

	nChunks, err := gridChunkCount(grid)
	if err != nil {
		return nil, err
	}
	entries := make([]ChunkLocation, 0, min(nChunks, 4096))
	for idx := uint64(0); idx < nChunks; idx++ {
		elt, ok, err := walker.element(idx)
		if err != nil {
			return nil, err
		}
		if !ok {
			// Covering block never allocated: skip its whole range.
			idx = max(idx, walker.skipTo-1)
			continue
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
		Address: utils.ReadUint(rec, int(sb.OffsetSize), sb.Endianness),
		NBytes:  layout.chunkStoreBytes(),
	}
	off := int(sb.OffsetSize)
	if recType == btreeV2TypeChunksFiltered {
		entry.NBytes = utils.ReadUint(rec[off:], sizeLen, sb.Endianness)
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

// btreeV2Geometry holds the per-depth node capacities of a v2 B-tree
// (H5B2hdr.c: H5B2__hdr_init). Index 0 is the leaf level.
type btreeV2Geometry struct {
	recSize    int
	maxNrec    []uint64 // Max records per node at each depth.
	cumMaxNrec []uint64 // Max records in a whole subtree rooted at each depth.
	nrecSize   int      // Bytes used to encode a child's record count.
}

// limitEncSize is H5VM_limit_enc_size: bytes needed to encode values up to n.
func limitEncSize(n uint64) int {
	return int(floorLog2(n))/8 + 1 //nolint:gosec // G115: value bounded by the enclosing length checks
}

// maxBTreeV2Depth bounds tree depth; each level multiplies capacity by at
// least 2, so deeper trees cannot be valid.
const maxBTreeV2Depth = 16

func newBTreeV2Geometry(nodeSize uint32, recSize, depth, offsetSize int) (*btreeV2Geometry, error) {
	const prefix = 10 // sig + version + type + checksum
	if recSize <= 0 || int64(nodeSize) <= int64(prefix+recSize) || depth > maxBTreeV2Depth {
		return nil, fmt.Errorf("invalid v2 b-tree geometry: node size %d, record size %d, depth %d", nodeSize, recSize, depth)
	}
	g := &btreeV2Geometry{recSize: recSize}
	leaf := uint64(nodeSize-prefix) / uint64(recSize)
	g.maxNrec = []uint64{leaf}
	g.cumMaxNrec = []uint64{leaf}
	g.nrecSize = limitEncSize(leaf)
	for d := 1; d <= depth; d++ {
		ptr := offsetSize + g.nrecSize
		if d > 1 {
			ptr += limitEncSize(g.cumMaxNrec[d-1])
		}
		avail := int64(nodeSize) - prefix - int64(ptr)
		if avail < int64(recSize+ptr) {
			return nil, errors.New("invalid v2 b-tree geometry: node too small for internal node")
		}
		m := uint64(avail) / uint64(recSize+ptr) //nolint:gosec // G115: value bounded by the enclosing length checks
		g.maxNrec = append(g.maxNrec, m)
		hi, lo := bits.Mul64(m+1, g.cumMaxNrec[d-1])
		if hi != 0 || lo+m < lo {
			return nil, errors.New("invalid v2 b-tree geometry: capacity overflow")
		}
		g.cumMaxNrec = append(g.cumMaxNrec, lo+m)
	}
	return g, nil
}

// btreeV2ChunkWalker collects chunk records from a v2 B-tree of any depth.
type btreeV2ChunkWalker struct {
	r       io.ReaderAt
	sb      *Superblock
	geo     *btreeV2Geometry
	recType uint8
	limit   uint64 // Header's total record count; the walk may not exceed it.
	visited map[uint64]bool
	entries []ChunkLocation
	decode  func(rec []byte) ChunkLocation
}

// walk reads the node at addr (holding nrec records at the given depth) and
// everything below it.
func (w *btreeV2ChunkWalker) walk(addr, nrec uint64, depth int) error {
	if w.visited[addr] {
		return fmt.Errorf("v2 b-tree cycle: node at 0x%x visited twice", addr)
	}
	w.visited[addr] = true
	if nrec > w.geo.maxNrec[depth] {
		return fmt.Errorf("v2 b-tree node at 0x%x claims %d records, max %d", addr, nrec, w.geo.maxNrec[depth])
	}
	recSize := uint64(w.geo.recSize) //nolint:gosec // G115: value bounded by the enclosing length checks
	offU := uint64(w.sb.OffsetSize)

	if depth == 0 {
		// Leaf: sig "BTLF"(4) version(1) type(1) records checksum(4).
		buf, err := readIndexBlob(w.r, addr, 6+nrec*recSize+4, "BTLF")
		if err != nil {
			return err
		}
		if buf[5] != w.recType {
			return fmt.Errorf("v2 b-tree leaf record type mismatch: %d", buf[5])
		}
		return w.addRecords(buf[6:], nrec)
	}

	// Internal: sig "BTIN"(4) version(1) type(1) records, then nrec+1 child
	// pointers (address, record count, and for depth > 1 the subtree total),
	// checksum(4).
	ptrSize := offU + uint64(w.geo.nrecSize) //nolint:gosec // G115: value bounded by the enclosing length checks
	if depth > 1 {
		ptrSize += uint64(limitEncSize(w.geo.cumMaxNrec[depth-1])) //nolint:gosec // G115: value bounded by the enclosing length checks
	}
	buf, err := readIndexBlob(w.r, addr, 6+nrec*recSize+(nrec+1)*ptrSize+4, "BTIN")
	if err != nil {
		return err
	}
	if buf[5] != w.recType {
		return fmt.Errorf("v2 b-tree internal record type mismatch: %d", buf[5])
	}
	if err := w.addRecords(buf[6:], nrec); err != nil {
		return err
	}
	ptrs := buf[6+nrec*recSize:]
	for i := uint64(0); i <= nrec; i++ {
		p := ptrs[i*ptrSize:]
		child := utils.ReadUint(p, int(offU), w.sb.Endianness) //nolint:gosec // G115: value bounded by the enclosing length checks
		childN := utils.ReadUint(p[offU:], w.geo.nrecSize, w.sb.Endianness)
		if err := w.walk(child, childN, depth-1); err != nil {
			return err
		}
	}
	return nil
}

func (w *btreeV2ChunkWalker) addRecords(recs []byte, nrec uint64) error {
	if uint64(len(w.entries))+nrec > w.limit {
		return errors.New("v2 b-tree holds more records than its header declares")
	}
	rs := uint64(w.geo.recSize) //nolint:gosec // G115: value bounded by the enclosing length checks
	for i := uint64(0); i < nrec; i++ {
		w.entries = append(w.entries, w.decode(recs[i*rs:(i+1)*rs]))
	}
	return nil
}

// btreeV2ChunkEntries reads a version 2 B-tree chunk index (record types 10
// and 11) of any depth.
func btreeV2ChunkEntries(r io.ReaderAt, layout *DataLayoutMessage, sb *Superblock, grid []uint64) ([]ChunkLocation, error) {
	offsetSize := int(sb.OffsetSize)
	rank := len(grid)

	// Header: sig "BTHD"(4) version(1) type(1) node size(4) record size(2)
	// depth(2) split%(1) merge%(1) root addr(offsetSize) root nrec(2)
	// total records(lengthSize) checksum(4).
	buf, err := readIndexBlob(r, layout.DataAddress, 22+uint64(sb.OffsetSize)+uint64(sb.LengthSize), "BTHD")
	if err != nil {
		return nil, fmt.Errorf("failed to read chunk v2 b-tree header: %w", err)
	}
	recType := buf[5]
	nodeSize := sb.Endianness.Uint32(buf[6:])
	recSize := int(sb.Endianness.Uint16(buf[10:]))
	depth := int(sb.Endianness.Uint16(buf[12:]))
	rootAddr := utils.ReadUint(buf[16:], offsetSize, sb.Endianness)
	rootNrec := uint64(sb.Endianness.Uint16(buf[16+offsetSize:]))
	total := utils.ReadUint(buf[18+offsetSize:], int(sb.LengthSize), sb.Endianness)

	if recType != btreeV2TypeChunksUnfiltered && recType != btreeV2TypeChunksFiltered {
		return nil, fmt.Errorf("unexpected v2 b-tree record type for chunk index: %d", recType)
	}
	if rootNrec == 0 || isUndefinedAddress(rootAddr, sb.OffsetSize) {
		return nil, nil
	}

	sizeLen := 0
	if recType == btreeV2TypeChunksFiltered {
		sizeLen = recSize - offsetSize - 4 - 8*rank
		if sizeLen < 1 || sizeLen > 8 {
			return nil, fmt.Errorf("invalid filtered chunk record size: %d", recSize)
		}
	} else if recSize != offsetSize+8*rank {
		return nil, fmt.Errorf("invalid chunk record size %d for rank %d", recSize, rank)
	}

	geo, err := newBTreeV2Geometry(nodeSize, recSize, depth, offsetSize)
	if err != nil {
		return nil, err
	}
	w := &btreeV2ChunkWalker{
		r: r, sb: sb, geo: geo, recType: recType, limit: total,
		visited: map[uint64]bool{},
		decode: func(rec []byte) ChunkLocation {
			return decodeBTreeV2ChunkRecord(rec, recType, sizeLen, rank, layout, sb)
		},
	}
	if err := w.walk(rootAddr, rootNrec, depth); err != nil {
		return nil, err
	}
	return w.entries, nil
}
