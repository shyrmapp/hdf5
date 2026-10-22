package hdf5

import (
	"fmt"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/shyrmapp/hdf5/internal/utils"
)

// HyperslabSelection represents a rectangular selection in N-dimensional space.
// It follows the HDF5 hyperslab specification with start, count, stride, and block parameters.
//
// Parameters:
//   - Start: Starting coordinates in each dimension (0-based indexing)
//   - Count: Number of blocks to select in each dimension
//   - Stride: Step between blocks in each dimension (nil = default to all 1s)
//   - Block: Size of each block in each dimension (nil = default to all 1s)
//
// The total number of elements selected is: product(Count[i] * Block[i]) for all dimensions.
//
// Example 1 - Simple slice (start=100, count=50 in 1D array):
//
//	sel := &HyperslabSelection{
//	    Start: []uint64{100},
//	    Count: []uint64{50},
//	}
//
// Example 2 - Strided selection (every 2nd element):
//
//	sel := &HyperslabSelection{
//	    Start:  []uint64{0, 0},
//	    Count:  []uint64{25, 25},  // 25 blocks in each dimension
//	    Stride: []uint64{2, 2},     // Every 2nd element
//	    Block:  []uint64{1, 1},     // Each block is 1x1
//	}
type HyperslabSelection struct {
	Start  []uint64
	Count  []uint64
	Stride []uint64 // nil means all 1s (contiguous selection)
	Block  []uint64 // nil means all 1s (single element blocks)
}

// ReadSlice reads a rectangular block from the dataset using simple start/count parameters.
// This is a convenience method for the common case of reading a contiguous rectangular region.
//
// Parameters:
//   - start: Starting coordinates in each dimension (0-based)
//   - count: Number of elements to read in each dimension
//
// The number of dimensions in start and count must match the dataset's dimensionality.
//
// Example (2D dataset):
//
//	// Read 50x50 block starting at position (100, 200)
//	data, err := dataset.ReadSlice([]uint64{100, 200}, []uint64{50, 50})
//
// Returns:
//   - interface{}: The selected data in the dataset's native type ([]float64, []int32, etc.)
//   - error: Error if selection is invalid or reading fails
func (d *Dataset) ReadSlice(start, count []uint64) (interface{}, error) {
	// Read object header to get dataset metadata
	header, err := core.ReadObjectHeader(d.file.reader, d.address, d.file.sb)
	if err != nil {
		return nil, fmt.Errorf("failed to read object header: %w", err)
	}

	// Extract dataspace to validate dimensions
	var dataspaceMsg *core.HeaderMessage
	for _, msg := range header.Messages {
		if msg.Type == core.MsgDataspace {
			dataspaceMsg = msg
			break
		}
	}

	if dataspaceMsg == nil {
		return nil, fmt.Errorf("dataspace message not found in dataset")
	}

	dataspace, err := core.ParseDataspaceMessage(dataspaceMsg.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dataspace: %w", err)
	}

	// Validate dimensions match
	if len(start) != len(dataspace.Dimensions) {
		return nil, fmt.Errorf("start dimensions (%d) != dataset dimensions (%d)",
			len(start), len(dataspace.Dimensions))
	}
	if len(count) != len(dataspace.Dimensions) {
		return nil, fmt.Errorf("count dimensions (%d) != dataset dimensions (%d)",
			len(count), len(dataspace.Dimensions))
	}

	// Validate bounds (start + count must not exceed dataset dimensions)
	for i := range start {
		// Overflow-safe form of start+count > dim.
		if start[i] > dataspace.Dimensions[i] || count[i] > dataspace.Dimensions[i]-start[i] {
			return nil, fmt.Errorf("selection out of bounds in dimension %d: start=%d + count=%d > size=%d",
				i, start[i], count[i], dataspace.Dimensions[i])
		}
	}

	// Create simple hyperslab selection (stride=1, block=1)
	selection := &HyperslabSelection{
		Start:  start,
		Count:  count,
		Stride: nil, // Default to all 1s (contiguous)
		Block:  nil, // Default to all 1s (single elements)
	}

	// Fill in defaults for Stride and Block
	fillHyperslabDefaults(selection, len(dataspace.Dimensions))

	return d.readHyperslab(selection, header)
}

// ReadHyperslab reads data with full hyperslab parameters including stride and block.
// This provides complete control over the selection pattern, allowing strided and blocked selections.
//
// Parameters:
//   - selection: The hyperslab selection specification
//
// The selection is validated against the dataset's dimensions before reading.
//
// Example (read every 2nd element in 2D):
//
//	sel := &HyperslabSelection{
//	    Start:  []uint64{100, 200},
//	    Count:  []uint64{25, 25},   // 25 blocks
//	    Stride: []uint64{2, 2},      // Every 2nd element
//	    Block:  []uint64{1, 1},      // 1x1 blocks
//	}
//	data, err := dataset.ReadHyperslab(sel)
//
// Returns:
//   - interface{}: The selected data in the dataset's native type
//   - error: Error if selection is invalid or reading fails
func (d *Dataset) ReadHyperslab(selection *HyperslabSelection) (interface{}, error) {
	// Read object header to get dataset metadata
	header, err := core.ReadObjectHeader(d.file.reader, d.address, d.file.sb)
	if err != nil {
		return nil, fmt.Errorf("failed to read object header: %w", err)
	}

	// Extract dataspace to validate dimensions
	var dataspaceMsg *core.HeaderMessage
	for _, msg := range header.Messages {
		if msg.Type == core.MsgDataspace {
			dataspaceMsg = msg
			break
		}
	}

	if dataspaceMsg == nil {
		return nil, fmt.Errorf("dataspace message not found in dataset")
	}

	dataspace, err := core.ParseDataspaceMessage(dataspaceMsg.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dataspace: %w", err)
	}

	// Validate selection
	if err := validateHyperslabSelection(selection, dataspace.Dimensions); err != nil {
		return nil, fmt.Errorf("invalid selection: %w", err)
	}

	return d.readHyperslab(selection, header)
}

// validateHyperslabSelection validates a hyperslab selection against dataset dimensions.
// It checks dimension counts, bounds, and fills in default values for nil Stride/Block.
func validateHyperslabSelection(sel *HyperslabSelection, dims []uint64) error {
	ndims := len(dims)

	// Validate dimensionality
	if err := validateSelectionDimensions(sel, ndims); err != nil {
		return err
	}

	// Fill in defaults for nil Stride and Block
	fillHyperslabDefaults(sel, ndims)

	// Validate bounds for each dimension
	return validateHyperslabBounds(sel, dims)
}

// validateSelectionDimensions checks that selection arrays match dataset dimensionality.
func validateSelectionDimensions(sel *HyperslabSelection, ndims int) error {
	if len(sel.Start) != ndims {
		return fmt.Errorf("start dimensions (%d) != dataset dimensions (%d)",
			len(sel.Start), ndims)
	}
	if len(sel.Count) != ndims {
		return fmt.Errorf("count dimensions (%d) != dataset dimensions (%d)",
			len(sel.Count), ndims)
	}
	if sel.Stride != nil && len(sel.Stride) != ndims {
		return fmt.Errorf("stride dimensions (%d) != dataset dimensions (%d)",
			len(sel.Stride), ndims)
	}
	if sel.Block != nil && len(sel.Block) != ndims {
		return fmt.Errorf("block dimensions (%d) != dataset dimensions (%d)",
			len(sel.Block), ndims)
	}
	return nil
}

// fillHyperslabDefaults fills nil Stride and Block arrays with default values (all 1s).
func fillHyperslabDefaults(sel *HyperslabSelection, ndims int) {
	if sel.Stride == nil {
		sel.Stride = make([]uint64, ndims)
		for i := range sel.Stride {
			sel.Stride[i] = 1
		}
	}
	if sel.Block == nil {
		sel.Block = make([]uint64, ndims)
		for i := range sel.Block {
			sel.Block[i] = 1
		}
	}
}

// validateHyperslabBounds checks that selection parameters are valid and within bounds.
func validateHyperslabBounds(sel *HyperslabSelection, dims []uint64) error {
	// CVE-2025-44905 fix: Validate hyperslab bounds with overflow checking.
	if err := utils.ValidateHyperslabBounds(sel.Start, sel.Count, sel.Stride, dims); err != nil {
		return fmt.Errorf("hyperslab bounds validation failed: %w", err)
	}

	// CVE-2025-44905 fix: Calculate total elements with overflow check.
	totalElements, err := utils.CalculateHyperslabElements(sel.Count)
	if err != nil {
		return fmt.Errorf("hyperslab too large: %w", err)
	}

	// Additional validation: ensure reasonable size
	_ = totalElements // Used for validation above

	// Keep the original per-dimension validation for completeness
	for i := range dims {
		if err := validateDimensionBounds(sel, dims, i); err != nil {
			return err
		}
	}
	return nil
}

// validateDimensionBounds validates a single dimension's bounds.
func validateDimensionBounds(sel *HyperslabSelection, dims []uint64, dim int) error {
	if sel.Count[dim] == 0 {
		return fmt.Errorf("count must be > 0 in dimension %d", dim)
	}
	if sel.Stride[dim] == 0 {
		return fmt.Errorf("stride must be > 0 in dimension %d", dim)
	}
	if sel.Block[dim] == 0 {
		return fmt.Errorf("block must be > 0 in dimension %d", dim)
	}

	// Check bounds: start + (count-1)*stride + block must not exceed dimension
	// (ValidateHyperslabBounds already proved start+(count-1)*stride < dim, so
	// only the block addition can still overflow.)
	last := sel.Start[dim] + (sel.Count[dim]-1)*sel.Stride[dim]
	lastCoord := last + sel.Block[dim]
	if sel.Block[dim] > dims[dim]-last {
		return fmt.Errorf("selection out of bounds in dimension %d: "+
			"start=%d + (count-1)*stride + block = %d > size=%d",
			dim, sel.Start[dim], lastCoord, dims[dim])
	}
	return nil
}

// readHyperslab is the internal implementation for hyperslab reading.
// It dispatches to the appropriate layout-specific reader based on the dataset's storage layout.
func (d *Dataset) readHyperslab(selection *HyperslabSelection, header *core.ObjectHeader) (interface{}, error) {
	// Extract and parse messages
	messages, err := extractHyperslabMessages(header)
	if err != nil {
		return nil, err
	}

	parsedMsgs, err := parseHyperslabMessages(messages, d.file.sb)
	if err != nil {
		return nil, err
	}

	// Bound the output before any allocation: the converted float64 result
	// and the raw bytes must both fit the read limit.
	n, err := checkedSelectionElements(selection)
	if err != nil {
		return nil, err
	}
	if err := core.CheckReadSize(n, 8, d.file.maxReadBytes, "selection"); err != nil {
		return nil, err
	}
	if err := core.CheckReadSize(n, uint64(parsedMsgs.datatype.Size), d.file.maxReadBytes, "selection"); err != nil {
		return nil, err
	}
	parsedMsgs.fill = core.FillValueBytes(header, parsedMsgs.datatype.Size)

	// Dispatch to appropriate layout reader
	return d.dispatchHyperslabReader(selection, parsedMsgs)
}

// hyperslabMessages holds raw message data extracted from object header.
type hyperslabMessages struct {
	datatype       *core.HeaderMessage
	dataspace      *core.HeaderMessage
	layout         *core.HeaderMessage
	filterPipeline *core.HeaderMessage
}

// parsedHyperslabMessages holds parsed message structures.
type parsedHyperslabMessages struct {
	datatype       *core.DatatypeMessage
	dataspace      *core.DataspaceMessage
	layout         *core.DataLayoutMessage
	filterPipeline *core.FilterPipelineMessage
	fill           []byte // one element's fill value; nil = zeros
}

// extractHyperslabMessages extracts required messages from object header.
func extractHyperslabMessages(header *core.ObjectHeader) (*hyperslabMessages, error) {
	msgs := &hyperslabMessages{}

	for _, msg := range header.Messages {
		switch msg.Type {
		case core.MsgDatatype:
			msgs.datatype = msg
		case core.MsgDataspace:
			msgs.dataspace = msg
		case core.MsgDataLayout:
			msgs.layout = msg
		case core.MsgFilterPipeline:
			msgs.filterPipeline = msg
		}
	}

	// Validate required messages
	if msgs.datatype == nil {
		return nil, fmt.Errorf("datatype message not found")
	}
	if msgs.dataspace == nil {
		return nil, fmt.Errorf("dataspace message not found")
	}
	if msgs.layout == nil {
		return nil, fmt.Errorf("data layout message not found")
	}

	return msgs, nil
}

// parseHyperslabMessages parses raw messages into structured types.
func parseHyperslabMessages(msgs *hyperslabMessages, sb *core.Superblock) (*parsedHyperslabMessages, error) {
	parsed := &parsedHyperslabMessages{}

	var err error

	parsed.datatype, err = core.ParseDatatypeMessage(msgs.datatype.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse datatype: %w", err)
	}

	parsed.dataspace, err = core.ParseDataspaceMessage(msgs.dataspace.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dataspace: %w", err)
	}

	parsed.layout, err = core.ParseDataLayoutMessage(msgs.layout.Data, sb)
	if err != nil {
		return nil, fmt.Errorf("failed to parse layout: %w", err)
	}

	// Parse filter pipeline (optional)
	if msgs.filterPipeline != nil {
		parsed.filterPipeline, err = core.ParseFilterPipelineMessage(msgs.filterPipeline.Data)
		if err != nil {
			return nil, fmt.Errorf("failed to parse filter pipeline: %w", err)
		}
	}

	return parsed, nil
}

// dispatchHyperslabReader dispatches to appropriate layout-specific reader.
func (d *Dataset) dispatchHyperslabReader(
	selection *HyperslabSelection,
	msgs *parsedHyperslabMessages,
) (interface{}, error) {
	switch {
	case msgs.layout.IsCompact():
		return d.readHyperslabCompact(selection, msgs.datatype, msgs.dataspace, msgs.layout)
	case msgs.layout.IsContiguous():
		return d.readHyperslabContiguous(selection, msgs.datatype, msgs.dataspace, msgs.layout, msgs.fill)
	case msgs.layout.IsChunked():
		return d.readHyperslabChunked(selection, msgs.datatype, msgs.dataspace, msgs.layout, msgs.filterPipeline, msgs.fill)
	default:
		return nil, fmt.Errorf("unsupported layout class: %d", msgs.layout.Class)
	}
}

// checkedSelectionElements is the element count of a selection,
// product(Count[i]*Block[i]), erroring on uint64 overflow.
func checkedSelectionElements(sel *HyperslabSelection) (uint64, error) {
	total := uint64(1)
	for i := range sel.Count {
		span, err := utils.SafeMultiply(sel.Count[i], sel.Block[i])
		if err != nil {
			return 0, fmt.Errorf("selection too large: %w", err)
		}
		if total, err = utils.SafeMultiply(total, span); err != nil {
			return 0, fmt.Errorf("selection too large: %w", err)
		}
	}
	return total, nil
}

// calculateHyperslabOutputSize calculates the total number of elements in the hyperslab selection.
// For a hyperslab with stride and block parameters, the total is: product(Count[i] * Block[i]).
func calculateHyperslabOutputSize(sel *HyperslabSelection) uint64 {
	if len(sel.Count) == 0 {
		return 0
	}

	total := uint64(1)
	for i := range sel.Count {
		blockSize := sel.Block[i]
		if blockSize == 0 {
			blockSize = 1 // Default if not set
		}
		total *= sel.Count[i] * blockSize
	}

	return total
}

// readHyperslabCompact reads hyperslab from compact layout dataset.
// Compact layout stores data directly in the object header.
func (d *Dataset) readHyperslabCompact(
	selection *HyperslabSelection,
	datatype *core.DatatypeMessage,
	dataspace *core.DataspaceMessage,
	layout *core.DataLayoutMessage,
) (interface{}, error) {
	// Compact data is stored in layout.CompactData
	// We need to extract the selected region from this data
	return extractHyperslabFromRawData(selection, datatype, dataspace, layout.CompactData, d.file.maxReadBytes)
}

// readHyperslabChunked reads hyperslab from chunked layout dataset.
// Chunked layout stores data in separate chunks indexed by a B-tree.
//
// OPTIMIZED: Reads ONLY the chunks that overlap with the selection.
// For a small selection in a large dataset, this dramatically reduces I/O.
func (d *Dataset) readHyperslabChunked(
	selection *HyperslabSelection,
	datatype *core.DatatypeMessage,
	dataspace *core.DataspaceMessage,
	layout *core.DataLayoutMessage,
	filterPipeline *core.FilterPipelineMessage,
	fill []byte,
) (interface{}, error) {
	elementSize := uint64(datatype.Size)
	dims := dataspace.Dimensions
	if err := layout.ValidateChunkDims(len(dims), elementSize); err != nil {
		return nil, err
	}
	// Drop the trailing element-size pseudo-dimension.
	chunkDims := layout.ChunkSize[:len(dims)]

	// Calculate output size
	outputElements := calculateHyperslabOutputSize(selection)
	if outputElements == 0 {
		return []float64{}, nil
	}

	// An HDF5 chunk is fixed-size, edge chunks included, so this is exactly
	// what a chunk must decode to. It bounds the filter pipeline.
	chunkBytes := elementSize
	for _, cd := range chunkDims {
		var err error
		if chunkBytes, err = utils.SafeMultiply(chunkBytes, cd); err != nil {
			return nil, fmt.Errorf("chunk size overflow: %w", err)
		}
	}

	// Collect chunk locations from the layout's chunk index (v1 B-tree for
	// layout v3, modern chunk indexes for layout v4/v5).
	allChunks, err := core.CollectChunks(d.file.reader, layout, dataspace, d.file.sb)
	if err != nil {
		return nil, fmt.Errorf("failed to get chunk index: %w", err)
	}

	// Allocate output buffer
	outputData := make([]byte, outputElements*elementSize)
	core.FillBuffer(outputData, fill) // chunks never written read as the fill value

	// Visit the stored chunks inside the selection's chunk-grid box. Walking
	// the index (whose size the file bounds) instead of enumerating every grid
	// coordinate keeps the cost independent of how sparse the dataset is.
	// Each element is placed at the output offset computed from its
	// coordinates (see extractChunkPortion), NOT a running counter: chunks
	// arrive in index order, which is not the selection's row-major order,
	// and a missing sparse chunk would otherwise shift every later element.
	first, last := selectionChunkRange(selection, chunkDims, dims)
	for _, chunk := range allChunks {
		if !chunkInRange(chunk.Scaled, first, last) {
			continue
		}
		chunkData, err := core.ReadChunk(d.file.reader, layout, filterPipeline, chunk, chunkDims, dims, chunkBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to extract from chunk %v: %w", chunk.Scaled, err)
		}
		extractChunkPortion(chunkData, chunk.Scaled, chunkDims, dims, selection, elementSize, outputData)
	}

	// Convert bytes to float64
	return core.ConvertToFloat64(outputData, datatype, outputElements, d.file.maxReadBytes)
}

// selectionChunkRange returns, per dimension, the first and last chunk-grid
// index the selection touches.
func selectionChunkRange(sel *HyperslabSelection, chunkDims, datasetDims []uint64) (first, last []uint64) {
	first = make([]uint64, len(chunkDims))
	last = make([]uint64, len(chunkDims))
	for i := range chunkDims {
		first[i] = sel.Start[i] / chunkDims[i]
		// End position = start + (count-1)*stride + block - 1, clamped to the dataset.
		endPos := sel.Start[i] + (sel.Count[i]-1)*sel.Stride[i] + sel.Block[i] - 1
		if endPos >= datasetDims[i] {
			endPos = datasetDims[i] - 1
		}
		last[i] = endPos / chunkDims[i]
	}
	return first, last
}

// chunkInRange reports whether scaled lies in the box [first, last].
func chunkInRange(scaled, first, last []uint64) bool {
	for i, c := range scaled {
		if c < first[i] || c > last[i] {
			return false
		}
	}
	return true
}

// extractChunkPortion copies the part of one decoded chunk that intersects
// the selection into outputData.
//
// Each element lands at its row-major position within the selection's
// output (see selectionOutputIndex), not at a running counter: chunks are
// visited in chunk-grid order, which is not the selection's order whenever a
// chunk is narrower than the selection, and a missing sparse chunk would
// otherwise shift every later element.
//
// Only the chunk∩selection box is visited — per dimension, just the selected
// coordinates inside [chunkStart, chunkEnd) — and each contiguous run along
// the innermost dimension is moved with one copy(). The previous version
// walked the whole selection for every chunk and tested each element for
// membership, which made a banded read of a many-chunk dataset quadratic.
func extractChunkPortion(
	chunkData []byte,
	chunkCoord []uint64,
	chunkDims []uint64,
	datasetDims []uint64,
	selection *HyperslabSelection,
	elementSize uint64,
	outputData []byte,
) {
	ndims := len(chunkCoord)
	if ndims == 0 {
		return
	}
	for i := 0; i < ndims; i++ {
		if selection.Count[i] == 0 {
			return
		}
	}

	c := chunkCopy{
		chunkData:   chunkData,
		outputData:  outputData,
		sel:         selection,
		elementSize: elementSize,
		chunkStart:  make([]uint64, ndims),
		chunkEnd:    make([]uint64, ndims),
		srcStride:   make([]uint64, ndims),
		dstStride:   make([]uint64, ndims),
	}
	src, dst := uint64(1), uint64(1)
	for i := ndims - 1; i >= 0; i-- {
		c.chunkStart[i] = chunkCoord[i] * chunkDims[i]
		c.chunkEnd[i] = c.chunkStart[i] + chunkDims[i]
		if c.chunkEnd[i] > datasetDims[i] {
			c.chunkEnd[i] = datasetDims[i]
		}
		c.srcStride[i] = src
		c.dstStride[i] = dst
		src *= chunkDims[i]
		dst *= selection.Count[i] * selection.Block[i]
	}
	c.copyDim(0, 0, 0)
}

// chunkCopy carries the per-chunk state of extractChunkPortion. Offsets are
// in elements; srcStride is the chunk's row-major stride per dimension and
// dstStride the selection output's.
type chunkCopy struct {
	chunkData, outputData []byte
	sel                   *HyperslabSelection
	elementSize           uint64
	chunkStart, chunkEnd  []uint64
	srcStride, dstStride  []uint64
}

// copyDim visits the selected coordinates of dimension dim that fall inside
// the chunk, recursing into the next dimension or, at the innermost one,
// copying runs. srcBase/dstBase are the element offsets accumulated from the
// outer dimensions.
//
// A coordinate x maps to output position (rel/stride)*block + rel%stride
// with rel = x - start, or plain rel when stride <= 1 — the same mapping as
// selectionOutputIndex. Within one block of a non-overlapping selection
// (block <= stride) that is consecutive, so a block's intersection with the
// chunk is one run; with stride <= 1 the whole selected range is. Blocks
// that overlap (block > stride, not a valid HDF5 selection but not rejected
// either) keep the per-element mapping.
func (c *chunkCopy) copyDim(dim int, srcBase, dstBase uint64) {
	sel := c.sel
	lo, hi := c.chunkStart[dim], c.chunkEnd[dim]
	start, count, stride, block := sel.Start[dim], sel.Count[dim], sel.Stride[dim], sel.Block[dim]
	innermost := dim == len(c.chunkStart)-1

	// visit handles the chunk-local coordinate range [a, b) of this
	// dimension whose output positions start at pos and are consecutive.
	visit := func(a, b, pos uint64) {
		src := srcBase + (a-lo)*c.srcStride[dim]
		dst := dstBase + pos*c.dstStride[dim]
		if innermost {
			c.copyRun(src, dst, b-a)
			return
		}
		for x := a; x < b; x++ {
			c.copyDim(dim+1, src, dst)
			src += c.srcStride[dim]
			dst += c.dstStride[dim]
		}
	}

	if stride <= 1 {
		end := start + (count-1)*stride + block
		a, b := max(lo, start), min(hi, end)
		if a < b {
			visit(a, b, a-start)
		}
		return
	}

	// First block that can reach lo: start + c*stride + block > lo.
	c0 := uint64(0)
	if lo >= start+block {
		c0 = (lo - start - block + stride) / stride
	}
	for ci := c0; ci < count; ci++ {
		bs := start + ci*stride
		if bs >= hi {
			break
		}
		a, b := max(lo, bs), min(hi, bs+block)
		if a >= b {
			continue
		}
		if block <= stride {
			visit(a, b, ci*block+(a-bs))
			continue
		}
		for x := a; x < b; x++ {
			rel := x - start
			visit(x, x+1, (rel/stride)*block+rel%stride)
		}
	}
}

// copyRun copies n consecutive elements from chunk offset src to output
// offset dst. Like the per-element copy it replaces, elements that would
// fall outside either buffer (a short chunk, a clipped edge) are dropped
// rather than read or written out of bounds; since both offsets grow
// together, the ones that fit are a prefix of the run.
func (c *chunkCopy) copyRun(src, dst, n uint64) {
	es := c.elementSize
	if es == 0 || n == 0 {
		return
	}
	srcLen, dstLen := uint64(len(c.chunkData)), uint64(len(c.outputData))
	if src*es >= srcLen || dst*es >= dstLen {
		return
	}
	n = min(n, (srcLen-src*es)/es, (dstLen-dst*es)/es)
	copy(c.outputData[dst*es:(dst+n)*es], c.chunkData[src*es:(src+n)*es])
}

// selectionOutputIndex returns the row-major position of a dataset
// coordinate within the hyperslab selection's output buffer. The per-dimension
// output extent is Count*Block; for the common stride=1, block=1 ReadSlice
// this is simply coord-start. extractChunkPortion applies the same mapping
// per run; this per-coordinate form is kept as its reference.
func selectionOutputIndex(coords []uint64, sel *HyperslabSelection) uint64 {
	idx := uint64(0)
	stride := uint64(1)
	for i := len(coords) - 1; i >= 0; i-- {
		rel := coords[i] - sel.Start[i]
		var pos uint64
		if sel.Stride[i] <= 1 {
			pos = rel
		} else {
			pos = (rel/sel.Stride[i])*sel.Block[i] + rel%sel.Stride[i]
		}
		idx += pos * stride
		stride *= sel.Count[i] * sel.Block[i]
	}
	return idx
}

// extractHyperslabFromRawData extracts a hyperslab selection from raw dataset bytes.
// This handles the N-dimensional indexing and stride/block logic.
//
// The raw data is assumed to be in row-major (C-style) order, where the last dimension
// varies fastest. The hyperslab selection is also in row-major order.
//
// Returns []float64, matching the existing Read() method.
func extractHyperslabFromRawData(
	selection *HyperslabSelection,
	datatype *core.DatatypeMessage,
	dataspace *core.DataspaceMessage,
	rawData []byte,
	maxBytes uint64,
) (interface{}, error) {
	elementSize := uint64(datatype.Size)
	ndims := len(dataspace.Dimensions)

	// Calculate output size
	outputElements := calculateHyperslabOutputSize(selection)
	if outputElements == 0 {
		// Return empty array
		return []float64{}, nil
	}

	// Allocate output buffer
	outputData := make([]byte, outputElements*elementSize)
	outputIdx := uint64(0)

	// Use recursive iteration to handle arbitrary dimensionality
	coords := make([]uint64, ndims)
	copy(coords, selection.Start)

	extractHyperslabRecursive(
		rawData, outputData,
		dataspace.Dimensions, selection,
		coords, 0,
		elementSize, &outputIdx,
	)

	// Convert bytes to float64 (matching existing Read() behavior)
	// Future: support other types based on datatype
	return core.ConvertToFloat64(outputData, datatype, outputElements, maxBytes)
}

// extractHyperslabRecursive recursively iterates through hyperslab selection dimensions.
// This handles arbitrary dimensionality with stride and block parameters.
func extractHyperslabRecursive(
	rawData, outputData []byte,
	dims []uint64,
	selection *HyperslabSelection,
	coords []uint64,
	dimIdx int,
	elementSize uint64,
	outputIdx *uint64,
) {
	ndims := len(dims)

	if dimIdx == ndims {
		// Base case: we have a complete coordinate, copy the element
		// Calculate linear offset in raw data (row-major order)
		offset := calculateLinearOffset(coords, dims)
		byteOffset := offset * elementSize

		// Bounds check
		if byteOffset+elementSize > uint64(len(rawData)) {
			return // Skip out-of-bounds reads
		}

		// Copy element to output
		outputOffset := (*outputIdx) * elementSize
		copy(outputData[outputOffset:outputOffset+elementSize],
			rawData[byteOffset:byteOffset+elementSize])
		(*outputIdx)++
		return
	}

	// Recursive case: iterate through current dimension
	// For each count, we advance by stride and read block elements
	for c := uint64(0); c < selection.Count[dimIdx]; c++ {
		// Start position for this block
		blockStart := selection.Start[dimIdx] + c*selection.Stride[dimIdx]

		// Iterate through block elements
		for b := uint64(0); b < selection.Block[dimIdx]; b++ {
			coords[dimIdx] = blockStart + b

			// Bounds check for this dimension
			if coords[dimIdx] >= dims[dimIdx] {
				continue
			}

			// Recurse to next dimension
			extractHyperslabRecursive(
				rawData, outputData,
				dims, selection,
				coords, dimIdx+1,
				elementSize, outputIdx,
			)
		}
	}
}

// calculateLinearOffset calculates the linear byte offset for N-dimensional coordinates.
// Uses row-major (C-style) indexing: last dimension varies fastest.
func calculateLinearOffset(coords, dims []uint64) uint64 {
	offset := uint64(0)
	stride := uint64(1)

	// Start from last dimension (varies fastest in row-major order)
	for i := len(coords) - 1; i >= 0; i-- {
		offset += coords[i] * stride
		stride *= dims[i]
	}

	return offset
}

// (datatype conversion is delegated to core.ConvertToFloat64 — see the
// chunked/contiguous readers above — so the hyperslab path supports exactly
// the same datatypes as the whole-dataset Read(), including fixed-point
// integers of every width and sign.)
