package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"

	"github.com/shyrmapp/hdf5/internal/utils"
)

// Shared read path for dense storage: a v2 B-tree (name or creation-order
// index) whose records carry fractal-heap IDs. Dense links (B-tree record
// types 5/6) and dense attributes (types 8/9) both go through
// ReadDenseHeapObjects.
//
// Supported: B-tree v2 of any depth, fractal heaps whose root is a direct
// block or a (multi-level) indirect block, managed and tiny heap IDs, and
// huge IDs that embed address and length. Heaps with I/O filters and huge IDs
// indexed through a separate B-tree are rejected with an error. Every
// metadata block (BTHD/BTIN/BTLF, FRHP/FHIB/FHDB) has its Jenkins checksum
// verified.
//
// Reference: HDF5 format spec III.A.2 (B-tree v2) and III.G (fractal heap).

const maxHeapIndirect = 64

func readFull(r io.ReaderAt, addr uint64, n int) ([]byte, error) {
	if n < 0 || addr > 1<<62 {
		return nil, fmt.Errorf("invalid read of %d bytes at 0x%X", n, addr)
	}
	buf := make([]byte, n)
	got, err := r.ReadAt(buf, int64(addr))
	if got < n {
		if err == nil || errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return nil, fmt.Errorf("read of %d bytes at 0x%X: %w", n, addr, err)
	}
	return buf, nil
}

func readUintLE(b []byte) uint64 {
	var v uint64
	for i, c := range b {
		v |= uint64(c) << (8 * uint(i))
	}
	return v
}

// btreeV2 holds the header fields needed to walk a v2 B-tree.
type btreeV2 struct {
	r         io.ReaderAt
	sb        *Superblock
	recType   uint8
	rootAddr  uint64
	rootRecs  uint64
	depth     int
	totalRecs uint64
	geo       *btreeV2Geometry
	visited   map[uint64]bool
}

func openBTreeV2(r io.ReaderAt, addr uint64, sb *Superblock) (*btreeV2, error) {
	off := int(sb.OffsetSize)
	// sig(4) version(1) type(1) node size(4) record size(2) depth(2) split%(1)
	// merge%(1) root addr(off) root nrec(2) total records(length) checksum(4).
	buf, err := readIndexBlob(r, addr, 22+uint64(sb.OffsetSize)+uint64(sb.LengthSize), "BTHD")
	if err != nil {
		return nil, err
	}
	e := sb.Endianness
	b := &btreeV2{r: r, sb: sb, recType: buf[5], visited: map[uint64]bool{}}
	nodeSize := e.Uint32(buf[6:])
	recSize := int(e.Uint16(buf[10:]))
	b.depth = int(e.Uint16(buf[12:]))
	b.rootAddr = readAddress(buf[16:16+off], off)
	b.rootRecs = uint64(e.Uint16(buf[16+off:]))
	b.totalRecs = readUintLE(buf[18+off : 18+off+int(sb.LengthSize)])
	b.geo, err = newBTreeV2Geometry(nodeSize, recSize, b.depth, off)
	return b, err
}

// walk appends every record under the node at addr, in key order.
func (b *btreeV2) walk(addr, nrec uint64, depth int, out *[][]byte) error {
	if b.visited[addr] {
		return fmt.Errorf("btree v2 cycle: node at 0x%X visited twice", addr)
	}
	b.visited[addr] = true
	if nrec > b.geo.maxNrec[depth] {
		return fmt.Errorf("btree v2 node at 0x%X has %d records, max %d", addr, nrec, b.geo.maxNrec[depth])
	}
	rs := uint64(b.geo.recSize) //nolint:gosec // G115: validated by newBTreeV2Geometry
	offU := uint64(b.sb.OffsetSize)

	if depth == 0 {
		buf, err := readIndexBlob(b.r, addr, 6+nrec*rs+4, "BTLF")
		if err != nil {
			return err
		}
		return b.addRecords(buf[6:], nrec, out)
	}

	ptr := offU + uint64(b.geo.nrecSize) //nolint:gosec // G115: small
	if depth > 1 {
		ptr += uint64(limitEncSize(b.geo.cumMaxNrec[depth-1])) //nolint:gosec // G115: small
	}
	buf, err := readIndexBlob(b.r, addr, 6+nrec*rs+(nrec+1)*ptr+4, "BTIN")
	if err != nil {
		return err
	}
	recs := buf[6 : 6+nrec*rs]
	ptrs := buf[6+nrec*rs:]
	for i := uint64(0); i <= nrec; i++ {
		p := ptrs[i*ptr:]
		child := readUint64(p, int(offU), b.sb.Endianness) //nolint:gosec // G115: small
		childN := readUint64(p[offU:], b.geo.nrecSize, b.sb.Endianness)
		if err := b.walk(child, childN, depth-1, out); err != nil {
			return err
		}
		if i < nrec {
			if err := b.addRecords(recs[i*rs:], 1, out); err != nil {
				return err
			}
		}
	}
	return nil
}

func (b *btreeV2) addRecords(recs []byte, n uint64, out *[][]byte) error {
	if uint64(len(*out))+n > b.totalRecs {
		return fmt.Errorf("btree v2 holds more than the declared %d records", b.totalRecs)
	}
	rs := uint64(b.geo.recSize) //nolint:gosec // G115: validated by newBTreeV2Geometry
	for i := uint64(0); i < n; i++ {
		*out = append(*out, recs[i*rs:(i+1)*rs])
	}
	return nil
}

// heapIDs returns the fractal-heap ID of every record, in B-tree order.
func (b *btreeV2) heapIDs() ([][]byte, error) {
	var recs [][]byte
	if b.totalRecs == 0 || isUndefinedAddress(b.rootAddr, b.sb.OffsetSize) {
		return nil, nil
	}
	if err := b.walk(b.rootAddr, b.rootRecs, b.depth, &recs); err != nil {
		return nil, err
	}
	ids := make([][]byte, len(recs))
	for i, rec := range recs {
		var lo, hi int
		switch b.recType {
		case 5: // link name: hash(4) heapID(7)
			lo, hi = 4, 11
		case 6: // link creation order: order(8) heapID(7)
			lo, hi = 8, 15
		case 8, 9: // attribute name / creation order: heapID(8) ...
			lo, hi = 0, 8
		default:
			return nil, fmt.Errorf("btree v2 record type %d does not index a fractal heap", b.recType)
		}
		if hi > len(rec) {
			return nil, fmt.Errorf("btree v2 record of %d bytes too short for type %d", len(rec), b.recType)
		}
		ids[i] = rec[lo:hi]
	}
	return ids, nil
}

// fractalHeap holds the header fields needed to resolve heap IDs.
type fractalHeap struct {
	r          io.ReaderAt
	sb         *Superblock
	idLen      int
	flags      byte
	maxManObj  uint64
	width      uint64
	startBlock uint64
	maxDirect  uint64
	rootAddr   uint64
	curRows    uint64
	offBytes   int // heap offset width in IDs and block headers
	lenBytes   int // object length width in managed IDs
	blocks     map[uint64][]byte
	blockBytes uint64 // summed size of cached direct blocks, capped at the file size
}

// validDoublingTable requires powers of two within bounds that keep
// width*startBlock and the row shifts from overflowing (a wrapped product is a
// zero divisor).
func validDoublingTable(width, startBlock, maxDirect uint64) bool {
	pow2 := func(v uint64) bool { return v != 0 && v&(v-1) == 0 }
	return pow2(width) && width <= 1<<16 && pow2(startBlock) && startBlock <= 1<<32 &&
		pow2(maxDirect) && maxDirect >= startBlock && maxDirect <= 1<<32
}

func openFractalHeap(r io.ReaderAt, addr uint64, sb *Superblock) (*fractalHeap, error) {
	off, ln := int(sb.OffsetSize), int(sb.LengthSize)
	e := sb.Endianness
	// Filters change the header length, so check before the checksummed read.
	pre, err := readFull(r, addr, 9)
	if err != nil {
		return nil, err
	}
	if string(pre[0:4]) != "FRHP" {
		return nil, fmt.Errorf("invalid fractal heap signature: %q", pre[0:4])
	}
	if e.Uint16(pre[7:]) != 0 {
		return nil, errors.New("fractal heap with I/O filters is not supported")
	}
	size := 14 + 10*ln + 2*off + 2 + 2*ln + 2 + 2 + off + 2 + 4
	buf, err := readIndexBlob(r, addr, uint64(size), "FRHP") //nolint:gosec // G115: small
	if err != nil {
		return nil, err
	}
	h := &fractalHeap{r: r, sb: sb, idLen: int(e.Uint16(buf[5:])), flags: buf[9], maxManObj: uint64(e.Uint32(buf[10:])),
		blocks: map[uint64][]byte{}}
	p := 14 + 10*ln + 2*off
	h.width = uint64(e.Uint16(buf[p:]))
	p += 2
	h.startBlock = readUintLE(buf[p : p+ln])
	p += ln
	h.maxDirect = readUintLE(buf[p : p+ln])
	p += ln
	maxHeapBits := int(e.Uint16(buf[p:]))
	p += 4 // max heap size + starting rows in root
	h.rootAddr = readAddress(buf[p:p+off], off)
	p += off
	h.curRows = uint64(e.Uint16(buf[p:]))

	if !validDoublingTable(h.width, h.startBlock, h.maxDirect) || maxHeapBits == 0 || maxHeapBits > 64 {
		return nil, errors.New("invalid fractal heap doubling table")
	}
	h.offBytes = (maxHeapBits + 7) / 8
	h.lenBytes = min(limitEncSize(h.maxDirect), limitEncSize(h.maxManObj))
	return h, nil
}

func (h *fractalHeap) blockSize(row uint64) uint64 {
	if row == 0 {
		return h.startBlock
	}
	return h.startBlock << (row - 1)
}

func (h *fractalHeap) rowStart(row uint64) uint64 {
	if row == 0 {
		return 0
	}
	return h.width * h.startBlock << (row - 1)
}

// directRows is the number of doubling-table rows holding direct blocks.
func (h *fractalHeap) directRows() uint64 {
	return uint64(bits.TrailingZeros64(h.maxDirect)-bits.TrailingZeros64(h.startBlock)) + 2 //nolint:gosec // G115: small non-negative bit counts
}

// object resolves a heap ID to the object bytes.
func (h *fractalHeap) object(id []byte) ([]byte, error) {
	if len(id) == 0 {
		return nil, errors.New("empty heap ID")
	}
	switch (id[0] >> 4) & 3 {
	case 0: // managed
		if 1+h.offBytes+h.lenBytes > len(id) {
			return nil, fmt.Errorf("heap ID too short: %d bytes", len(id))
		}
		off := readUintLE(id[1 : 1+h.offBytes])
		length := readUintLE(id[1+h.offBytes : 1+h.offBytes+h.lenBytes])
		return h.managed(off, length)
	case 2: // tiny: length-1 in the low nibble, data follows
		n := int(id[0]&0x0F) + 1
		if h.idLen > 16 || 1+n > len(id) {
			return nil, errors.New("unsupported tiny heap ID")
		}
		return append([]byte(nil), id[1:1+n]...), nil
	case 1: // huge, address and length embedded in the ID
		o, l := int(h.sb.OffsetSize), int(h.sb.LengthSize)
		if 1+o+l > len(id) {
			return nil, errors.New("huge heap object indexed through a B-tree is not supported")
		}
		length := readUintLE(id[1+o : 1+o+l])
		if length > 1<<30 {
			return nil, fmt.Errorf("huge heap object of %d bytes refused", length)
		}
		return readFull(h.r, readAddress(id[1:1+o], o), int(length))
	}
	return nil, fmt.Errorf("unsupported heap ID type %d", (id[0]>>4)&3)
}

// managed reads length bytes at heap offset off, descending from the root block.
func (h *fractalHeap) managed(off, length uint64) ([]byte, error) {
	if h.curRows == 0 {
		return h.direct(h.rootAddr, h.startBlock, 0, off, length)
	}
	return h.indirect(h.rootAddr, h.curRows, 0, off, length, 0)
}

// indirectBlock returns the verified indirect block at addr with nrows rows.
func (h *fractalHeap) indirectBlock(addr, nrows uint64) ([]byte, error) {
	if b, ok := h.blocks[addr]; ok {
		return b, nil
	}
	size := 5 + uint64(h.sb.OffsetSize) + uint64(h.offBytes) + nrows*h.width*uint64(h.sb.OffsetSize) + 4 //nolint:gosec // G115: small
	b, err := readIndexBlob(h.r, addr, size, "FHIB")
	if err != nil {
		return nil, err
	}
	h.blocks[addr] = b
	return b, nil
}

func (h *fractalHeap) indirect(addr, nrows, base, off, length uint64, level int) ([]byte, error) {
	if level > maxHeapIndirect {
		return nil, errors.New("fractal heap nesting too deep")
	}
	if off < base {
		return nil, fmt.Errorf("heap offset 0x%X before block offset 0x%X", off, base)
	}
	rel := off - base
	row := uint64(0)
	if rel >= h.width*h.startBlock {
		row = uint64(bits.Len64(rel / (h.width * h.startBlock))) //nolint:gosec // G115: at most 64
	}
	if row >= nrows {
		return nil, fmt.Errorf("heap offset 0x%X outside indirect block at 0x%X", off, addr)
	}
	// width*startBlock<<(row-1) must stay below 2^63 so bs and rowStart cannot wrap to 0.
	if row > 1 && bits.Len64(h.width*h.startBlock)+int(row)-1 > 63 {
		return nil, fmt.Errorf("heap offset 0x%X outside the heap's address space", off)
	}
	bs := h.blockSize(row)
	col := (rel - h.rowStart(row)) / bs
	o := uint64(h.sb.OffsetSize)
	blk, err := h.indirectBlock(addr, nrows)
	if err != nil {
		return nil, err
	}
	entry := 5 + o + uint64(h.offBytes) + (row*h.width+col)*o //nolint:gosec // G115: small
	child := readAddress(blk[entry:entry+o], int(h.sb.OffsetSize))
	if isUndefinedAddress(child, h.sb.OffsetSize) {
		return nil, fmt.Errorf("heap offset 0x%X is in an unallocated block", off)
	}
	childOff := base + h.rowStart(row) + col*bs
	if row < h.directRows() {
		return h.direct(child, bs, childOff, off, length)
	}
	childRows := uint64(bits.TrailingZeros64(bs)-bits.TrailingZeros64(h.startBlock*h.width)) + 1 //nolint:gosec // G115: small non-negative bit counts
	return h.indirect(child, childRows, childOff, off, length, level+1)
}

// directBlock returns the verified direct block of the given size at addr. The
// checksum (when the heap enables it) sits in the block header and covers the
// whole block with the checksum field zeroed.
func (h *fractalHeap) directBlock(addr, size uint64) ([]byte, error) {
	if b, ok := h.blocks[addr]; ok {
		return b, nil
	}
	if err := utils.ValidateBufferSize(size, utils.MaxChunkSize, "fractal heap direct block"); err != nil {
		return nil, err
	}
	if size > blockLimit(h.r, addr) {
		return nil, fmt.Errorf("direct block at 0x%X of %d bytes exceeds file", addr, size)
	}
	// Overlapping blocks at distinct addresses must not multiply the file size.
	if total, ok := readerSize(h.r); ok && h.blockBytes+size > total {
		return nil, fmt.Errorf("fractal heap direct blocks exceed file size %d", total)
	}
	h.blockBytes += size
	b, err := readFull(h.r, addr, int(size)) //nolint:gosec // G115: bounded above
	if err != nil {
		return nil, err
	}
	if string(b[0:4]) != "FHDB" {
		return nil, fmt.Errorf("invalid direct block signature at 0x%X: %q", addr, b[0:4])
	}
	if h.flags&0x02 != 0 {
		at := 5 + int(h.sb.OffsetSize) + h.offBytes
		if at+4 > len(b) {
			return nil, fmt.Errorf("direct block at 0x%X too small for checksum", addr)
		}
		stored := binary.LittleEndian.Uint32(b[at:])
		clear(b[at : at+4])
		if got := JenkinsChecksum(b); got != stored {
			return nil, fmt.Errorf("FHDB block at 0x%X checksum mismatch: computed %08x, stored %08x", addr, got, stored)
		}
		binary.LittleEndian.PutUint32(b[at:], stored)
	}
	h.blocks[addr] = b
	return b, nil
}

func (h *fractalHeap) direct(addr, size, blockOff, off, length uint64) ([]byte, error) {
	if off < blockOff || off-blockOff+length > size || length > 1<<30 {
		return nil, fmt.Errorf("heap object 0x%X+%d outside direct block 0x%X+%d", off, length, blockOff, size)
	}
	b, err := h.directBlock(addr, size)
	if err != nil {
		return nil, err
	}
	rel := off - blockOff
	// Slice the cached block rather than copy: many records may share one block.
	return b[rel : rel+length : rel+length], nil
}

// ReadDenseHeapObjects walks the v2 B-tree at btreeAddr and returns the raw
// fractal-heap object (link message or attribute message bytes) behind each
// record, in B-tree order.
func ReadDenseHeapObjects(r io.ReaderAt, btreeAddr, heapAddr uint64, sb *Superblock) ([][]byte, error) {
	bt, err := openBTreeV2(r, btreeAddr, sb)
	if err != nil {
		return nil, fmt.Errorf("btree v2 header: %w", err)
	}
	ids, err := bt.heapIDs()
	if err != nil {
		return nil, fmt.Errorf("btree v2 records: %w", err)
	}
	if len(ids) == 0 {
		return nil, nil
	}
	heap, err := openFractalHeap(r, heapAddr, sb)
	if err != nil {
		return nil, fmt.Errorf("fractal heap header: %w", err)
	}
	out := make([][]byte, 0, len(ids))
	// Records may all point at one block; the objects of a sane heap are
	// disjoint, so their total cannot exceed the file.
	var total uint64
	fileSize, sized := readerSize(r)
	for i, id := range ids {
		data, err := heap.object(id)
		if err != nil {
			return nil, fmt.Errorf("heap object %d: %w", i, err)
		}
		total += uint64(len(data))
		if sized && total > fileSize {
			return nil, fmt.Errorf("heap objects total %d bytes, exceeding file size %d", total, fileSize)
		}
		out = append(out, data)
	}
	return out, nil
}
