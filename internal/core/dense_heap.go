package core

import (
	"errors"
	"fmt"
	"io"
	"math/bits"
)

// Shared read path for dense storage: a v2 B-tree (name or creation-order
// index) whose records carry fractal-heap IDs. Dense links (B-tree record
// types 5/6) and dense attributes (types 8/9) both go through
// ReadDenseHeapObjects.
//
// Supported: B-tree v2 of any depth, fractal heaps whose root is a direct
// block or a (multi-level) indirect block, managed and tiny heap IDs, and
// huge IDs that embed address and length. Heaps with I/O filters and huge IDs
// indexed through a separate B-tree are rejected with an error.
//
// Reference: HDF5 format spec III.A.2 (B-tree v2) and III.G (fractal heap).

const (
	maxBTreeV2Depth = 16
	maxHeapIndirect = 64
)

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

// limitEncSize returns the bytes needed to encode values up to n (H5VM_limit_enc_size).
func limitEncSize(n uint64) int {
	if n == 0 {
		return 1
	}
	return (bits.Len64(n) + 7) / 8
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
	r                io.ReaderAt
	sb               *Superblock
	Type             uint8
	nodeSize         uint32
	RecordSize       uint16
	Depth            uint16
	RootNodeAddr     uint64
	NumRecordsRoot   uint16
	totalRecs        uint64
	maxRecs, cumRecs []uint64 // per depth: max records in a node / in a whole subtree
}

// readBTreeV2HeaderRaw is the entry point shared with the chunk index reader.
var readBTreeV2HeaderRaw = openBTreeV2

func openBTreeV2(r io.ReaderAt, addr uint64, sb *Superblock) (*btreeV2, error) {
	off, ln := int(sb.OffsetSize), int(sb.LengthSize)
	buf, err := readFull(r, addr, 4+1+1+4+2+2+1+1+off+2+ln)
	if err != nil {
		return nil, err
	}
	if string(buf[0:4]) != "BTHD" {
		return nil, fmt.Errorf("invalid B-tree v2 signature: %q", buf[0:4])
	}
	e := sb.Endianness
	b := &btreeV2{r: r, sb: sb, Type: buf[5], nodeSize: e.Uint32(buf[6:]), RecordSize: e.Uint16(buf[10:]), Depth: e.Uint16(buf[12:])}
	p := 16
	b.RootNodeAddr = readAddress(buf[p:p+off], off)
	p += off
	b.NumRecordsRoot = e.Uint16(buf[p:])
	p += 2
	b.totalRecs = readUintLE(buf[p : p+ln])

	if b.Depth > maxBTreeV2Depth {
		return nil, fmt.Errorf("btree v2 depth %d too large", b.Depth)
	}
	if b.RecordSize == 0 || b.nodeSize < 10+uint32(b.RecordSize) {
		return nil, fmt.Errorf("invalid btree v2 node/record size %d/%d", b.nodeSize, b.RecordSize)
	}
	// Per-depth capacities determine the width of the child-pointer count fields.
	b.maxRecs = []uint64{uint64(b.nodeSize-10) / uint64(b.RecordSize)}
	b.cumRecs = []uint64{b.maxRecs[0]}
	for d := 1; d <= int(b.Depth); d++ {
		ptr := uint64(off + limitEncSize(b.maxRecs[d-1])) //nolint:gosec // G115: small non-negative bit/size counts
		if d > 1 {
			ptr += uint64(limitEncSize(b.cumRecs[d-1])) //nolint:gosec // G115: small non-negative bit/size counts
		}
		if uint64(b.nodeSize) < 10+ptr {
			return nil, fmt.Errorf("invalid btree v2 node size %d", b.nodeSize)
		}
		m := (uint64(b.nodeSize) - 10 - ptr) / (uint64(b.RecordSize) + ptr)
		b.maxRecs = append(b.maxRecs, m)
		b.cumRecs = append(b.cumRecs, (m+1)*b.cumRecs[d-1]+m)
	}
	return b, nil
}

// walk appends every record under the node at addr, in key order.
func (b *btreeV2) walk(addr uint64, nrec, depth uint16, out *[][]byte) error {
	if uint64(nrec) > b.maxRecs[depth] {
		return fmt.Errorf("btree v2 node at 0x%X has %d records, max %d", addr, nrec, b.maxRecs[depth])
	}
	rs := int(b.RecordSize)
	n := int(nrec)
	size := 6 + n*rs
	sig := "BTLF"
	off := int(b.sb.OffsetSize)
	var cntBytes, allBytes int
	if depth > 0 {
		sig = "BTIN"
		cntBytes = limitEncSize(b.maxRecs[depth-1])
		if depth > 1 {
			allBytes = limitEncSize(b.cumRecs[depth-1])
		}
		size += (n + 1) * (off + cntBytes + allBytes)
	}
	buf, err := readFull(b.r, addr, size)
	if err != nil {
		return err
	}
	if string(buf[0:4]) != sig {
		return fmt.Errorf("invalid B-tree v2 node signature at 0x%X: %q (want %s)", addr, buf[0:4], sig)
	}
	rec := func(i int) []byte { return buf[6+i*rs : 6+(i+1)*rs] }
	if depth == 0 {
		for i := 0; i < n; i++ {
			*out = append(*out, rec(i))
		}
		return nil
	}
	p := 6 + n*rs
	for i := 0; i <= n; i++ {
		child := readAddress(buf[p:p+off], off)
		cn := readUintLE(buf[p+off : p+off+cntBytes])
		p += off + cntBytes + allBytes
		if cn > 0xFFFF {
			return fmt.Errorf("btree v2 child record count %d too large", cn)
		}
		if err := b.walk(child, uint16(cn), depth-1, out); err != nil {
			return err
		}
		if i < n {
			*out = append(*out, rec(i))
		}
		if uint64(len(*out)) > b.totalRecs {
			return fmt.Errorf("btree v2 holds more than the declared %d records", b.totalRecs)
		}
	}
	return nil
}

// heapIDs returns the fractal-heap ID of every record, in B-tree order.
func (b *btreeV2) heapIDs() ([][]byte, error) {
	var recs [][]byte
	if b.totalRecs == 0 {
		return nil, nil
	}
	if err := b.walk(b.RootNodeAddr, b.NumRecordsRoot, b.Depth, &recs); err != nil {
		return nil, err
	}
	ids := make([][]byte, len(recs))
	for i, rec := range recs {
		switch b.Type {
		case 5: // link name: hash(4) heapID(7)
			ids[i] = rec[4:min(11, len(rec))]
		case 6: // link creation order: order(8) heapID(7)
			ids[i] = rec[8:min(15, len(rec))]
		case 8, 9: // attribute name / creation order: heapID(8) ...
			ids[i] = rec[:min(8, len(rec))]
		default:
			return nil, fmt.Errorf("btree v2 record type %d does not index a fractal heap", b.Type)
		}
	}
	return ids, nil
}

// fractalHeap holds the header fields needed to resolve heap IDs.
type fractalHeap struct {
	r          io.ReaderAt
	sb         *Superblock
	addr       uint64
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
}

func openFractalHeap(r io.ReaderAt, addr uint64, sb *Superblock) (*fractalHeap, error) {
	off, ln := int(sb.OffsetSize), int(sb.LengthSize)
	size := 14 + 10*ln + 2*off + 2 + 2*ln + 2 + 2 + off + 2
	buf, err := readFull(r, addr, size)
	if err != nil {
		return nil, err
	}
	if string(buf[0:4]) != "FRHP" {
		return nil, fmt.Errorf("invalid fractal heap signature: %q", buf[0:4])
	}
	e := sb.Endianness
	h := &fractalHeap{r: r, sb: sb, addr: addr, idLen: int(e.Uint16(buf[5:])), flags: buf[9], maxManObj: uint64(e.Uint32(buf[10:]))}
	if e.Uint16(buf[7:]) != 0 {
		return nil, errors.New("fractal heap with I/O filters is not supported")
	}
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

	if h.width == 0 || h.startBlock == 0 || h.startBlock&(h.startBlock-1) != 0 ||
		h.maxDirect < h.startBlock || h.maxDirect&(h.maxDirect-1) != 0 || maxHeapBits == 0 || maxHeapBits > 64 {
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
	return uint64(bits.TrailingZeros64(h.maxDirect)-bits.TrailingZeros64(h.startBlock)) + 2 //nolint:gosec // G115: small non-negative bit/size counts
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
		row = uint64(bits.Len64(rel / (h.width * h.startBlock))) //nolint:gosec // G115: small non-negative bit/size counts
	}
	if row >= nrows {
		return nil, fmt.Errorf("heap offset 0x%X outside indirect block at 0x%X", off, addr)
	}
	bs := h.blockSize(row)
	col := (rel - h.rowStart(row)) / bs
	o := int(h.sb.OffsetSize)
	hdr := 5 + o + h.offBytes
	entry := uint64(hdr) + (row*h.width+col)*uint64(o) //nolint:gosec // G115: small non-negative bit/size counts
	buf, err := readFull(h.r, addr, 4)
	if err != nil {
		return nil, err
	}
	if string(buf) != "FHIB" {
		return nil, fmt.Errorf("invalid indirect block signature at 0x%X: %q", addr, buf)
	}
	eb, err := readFull(h.r, addr+entry, o)
	if err != nil {
		return nil, err
	}
	child := readAddress(eb, o)
	if isUndefinedAddress(child, h.sb.OffsetSize) {
		return nil, fmt.Errorf("heap offset 0x%X is in an unallocated block", off)
	}
	childOff := base + h.rowStart(row) + col*bs
	if row < h.directRows() {
		return h.direct(child, bs, childOff, off, length)
	}
	childRows := uint64(bits.TrailingZeros64(bs)-bits.TrailingZeros64(h.startBlock*h.width)) + 1 //nolint:gosec // G115: small non-negative bit/size counts
	return h.indirect(child, childRows, childOff, off, length, level+1)
}

func (h *fractalHeap) direct(addr, size, blockOff, off, length uint64) ([]byte, error) {
	if off < blockOff || off-blockOff+length > size {
		return nil, fmt.Errorf("heap object 0x%X+%d outside direct block 0x%X+%d", off, length, blockOff, size)
	}
	sig, err := readFull(h.r, addr, 4)
	if err != nil {
		return nil, err
	}
	if string(sig) != "FHDB" {
		return nil, fmt.Errorf("invalid direct block signature at 0x%X: %q", addr, sig)
	}
	if length > 1<<30 {
		return nil, fmt.Errorf("heap object of %d bytes refused", length)
	}
	return readFull(h.r, addr+(off-blockOff), int(length))
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
	for i, id := range ids {
		data, err := heap.object(id)
		if err != nil {
			return nil, fmt.Errorf("heap object %d: %w", i, err)
		}
		out = append(out, data)
	}
	return out, nil
}
