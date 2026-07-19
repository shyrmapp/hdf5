package core

import (
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/scigolib/hdf5/internal/utils"
)

// ReadDatasetFloat64 reads a dataset and returns values as float64 array.
// This is the main entry point for reading numerical datasets.
func ReadDatasetFloat64(r io.ReaderAt, header *ObjectHeader, sb *Superblock) ([]float64, error) {
	rawData, datatype, totalElements, err := readDatasetRaw(r, header, sb)
	if err != nil {
		return nil, err
	}
	if totalElements == 0 {
		return []float64{}, nil
	}
	return convertToFloat64(rawData, datatype, totalElements)
}

// ReadDatasetComplex reads a complex-number dataset (datatype class 11,
// introduced in HDF5 2.0) and returns values as complex128. Both
// complex128 (16-byte, float64 parts) and complex64 (8-byte, float32
// parts) storage are supported; parts are stored (real, imaginary).
func ReadDatasetComplex(r io.ReaderAt, header *ObjectHeader, sb *Superblock) ([]complex128, error) {
	rawData, datatype, totalElements, err := readDatasetRaw(r, header, sb)
	if err != nil {
		return nil, err
	}
	if datatype.Class != DatatypeComplex {
		return nil, fmt.Errorf("dataset is not complex (class %d)", datatype.Class)
	}
	if totalElements == 0 {
		return []complex128{}, nil
	}

	// The complex datatype's properties embed the base floating-point
	// datatype; byte order comes from THAT bit field, not the complex
	// type's own (whose bit 0 means something else).
	byteOrder := datatype.GetByteOrder()
	if base, baseErr := ParseDatatypeMessage(datatype.Properties); baseErr == nil && base.Class == DatatypeFloat {
		byteOrder = base.GetByteOrder()
	}
	result := make([]complex128, totalElements)
	switch datatype.Size {
	case 16: // float64 parts
		if totalElements*16 > uint64(len(rawData)) {
			return nil, errors.New("data truncated (complex128)")
		}
		for i := uint64(0); i < totalElements; i++ {
			re := math.Float64frombits(byteOrder.Uint64(rawData[i*16 : i*16+8]))
			im := math.Float64frombits(byteOrder.Uint64(rawData[i*16+8 : i*16+16]))
			result[i] = complex(re, im)
		}
	case 8: // float32 parts
		if totalElements*8 > uint64(len(rawData)) {
			return nil, errors.New("data truncated (complex64)")
		}
		for i := uint64(0); i < totalElements; i++ {
			re := math.Float32frombits(byteOrder.Uint32(rawData[i*8 : i*8+4]))
			im := math.Float32frombits(byteOrder.Uint32(rawData[i*8+4 : i*8+8]))
			result[i] = complex(float64(re), float64(im))
		}
	default:
		return nil, fmt.Errorf("unsupported complex element size %d bytes", datatype.Size)
	}
	return result, nil
}

// readDatasetRaw extracts the datatype and raw element bytes of a dataset:
// message extraction, layout dispatch (compact/contiguous/chunked), and
// filter pipeline handling shared by the typed readers.
func readDatasetRaw(r io.ReaderAt, header *ObjectHeader, sb *Superblock) ([]byte, *DatatypeMessage, uint64, error) {
	// 1. Extract required messages from object header.
	var datatypeMsg, dataspaceMsg, layoutMsg, filterPipelineMsg *HeaderMessage

	for _, msg := range header.Messages {
		switch msg.Type {
		case MsgDatatype:
			datatypeMsg = msg
		case MsgDataspace:
			dataspaceMsg = msg
		case MsgDataLayout:
			layoutMsg = msg
		case MsgFilterPipeline:
			filterPipelineMsg = msg
		}
	}

	// Validate we have all required messages.
	if datatypeMsg == nil {
		return nil, nil, 0, errors.New("datatype message not found")
	}
	if dataspaceMsg == nil {
		return nil, nil, 0, errors.New("dataspace message not found")
	}
	if layoutMsg == nil {
		return nil, nil, 0, errors.New("data layout message not found")
	}

	// 2. Parse datatype.
	datatype, err := ParseDatatypeMessage(datatypeMsg.Data)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("failed to parse datatype: %w", err)
	}

	// 3. Parse dataspace.
	dataspace, err := ParseDataspaceMessage(dataspaceMsg.Data)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("failed to parse dataspace: %w", err)
	}

	// 4. Parse layout.
	layout, err := ParseDataLayoutMessage(layoutMsg.Data, sb)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("failed to parse layout: %w", err)
	}

	// 5. Parse filter pipeline (optional, for compression).
	var filterPipeline *FilterPipelineMessage
	if filterPipelineMsg != nil {
		filterPipeline, err = ParseFilterPipelineMessage(filterPipelineMsg.Data)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("failed to parse filter pipeline: %w", err)
		}
	}

	// 6. Calculate total number of elements.
	totalElements := dataspace.TotalElements()
	if totalElements == 0 {
		return []byte{}, datatype, 0, nil
	}

	// 6. Read data based on layout type.
	var rawData []byte

	switch {
	case layout.IsCompact():
		// Data is stored directly in the layout message.
		rawData = layout.CompactData

	case layout.IsContiguous():
		// Data is stored contiguously at specific address.
		dataSize := totalElements * uint64(datatype.Size)
		rawData = make([]byte, dataSize)

		//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
		_, err := r.ReadAt(rawData, int64(layout.DataAddress))
		if err != nil {
			return nil, nil, 0, fmt.Errorf("failed to read contiguous data: %w", err)
		}

	case layout.IsChunked():
		// Data is stored in chunks indexed by B-tree.
		rawData, err = readChunkedData(r, layout, dataspace, datatype, sb, filterPipeline)
		if err != nil {
			return nil, nil, 0, fmt.Errorf("failed to read chunked data: %w", err)
		}

	default:
		return nil, nil, 0, fmt.Errorf("unsupported layout class: %d", layout.Class)
	}

	return rawData, datatype, totalElements, nil
}

// ConvertToFloat64 converts raw element bytes to a float64 slice based on
// the datatype. Exported so the hyperslab reader shares the exact same
// datatype coverage as the whole-dataset Read() path (notably fixed-point
// integers of every width/sign — int16, uint8, …), instead of maintaining
// a second, narrower converter that silently rejected those types.
func ConvertToFloat64(rawData []byte, datatype *DatatypeMessage, numElements uint64) ([]float64, error) {
	return convertToFloat64(rawData, datatype, numElements)
}

// convertToFloat64 converts raw bytes to float64 array based on datatype.
func convertToFloat64(rawData []byte, datatype *DatatypeMessage, numElements uint64) ([]float64, error) {
	result := make([]float64, numElements)
	byteOrder := datatype.GetByteOrder()

	switch {
	case datatype.IsFloat64():
		// IEEE 754 double precision (64-bit).
		for i := uint64(0); i < numElements; i++ {
			offset := i * 8
			if offset+8 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (float64)")
			}

			bits := byteOrder.Uint64(rawData[offset : offset+8])
			result[i] = math.Float64frombits(bits)
		}

	case datatype.IsFloat32():
		// IEEE 754 single precision (32-bit).
		for i := uint64(0); i < numElements; i++ {
			offset := i * 4
			if offset+4 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (float32)")
			}

			bits := byteOrder.Uint32(rawData[offset : offset+4])
			result[i] = float64(math.Float32frombits(bits))
		}

	case datatype.IsFloat16():
		// IEEE 754 half precision (16-bit), H5T_IEEE_F16LE/BE.
		for i := uint64(0); i < numElements; i++ {
			offset := i * 2
			if offset+2 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (float16)")
			}

			result[i] = float16ToFloat64(byteOrder.Uint16(rawData[offset : offset+2]))
		}

	case datatype.IsFixedPoint():
		// Fixed-point integer of any width (1/2/4/8 bytes), signed or
		// unsigned. The HDF5 spec encodes width in datatype.Size and
		// signedness in bit 3 of ClassBitField. The branching is on
		// Size first because the byte read is the same regardless of
		// signedness; signedness only changes the reinterpretation.
		signed := datatype.IsSignedFixedPoint()
		switch datatype.Size {
		case 1:
			if numElements > uint64(len(rawData)) {
				return nil, errors.New("data truncated (1-byte int)")
			}
			if signed {
				for i := uint64(0); i < numElements; i++ {
					//nolint:gosec // G115: spec-mandated uint8→int8 reinterpretation
					result[i] = float64(int8(rawData[i]))
				}
			} else {
				for i := uint64(0); i < numElements; i++ {
					result[i] = float64(rawData[i])
				}
			}
		case 2:
			if numElements*2 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (2-byte int)")
			}
			if signed {
				for i := uint64(0); i < numElements; i++ {
					//nolint:gosec // G115: spec-mandated uint16→int16 reinterpretation
					result[i] = float64(int16(byteOrder.Uint16(rawData[i*2 : i*2+2])))
				}
			} else {
				for i := uint64(0); i < numElements; i++ {
					result[i] = float64(byteOrder.Uint16(rawData[i*2 : i*2+2]))
				}
			}
		case 4:
			if numElements*4 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (4-byte int)")
			}
			if signed {
				for i := uint64(0); i < numElements; i++ {
					//nolint:gosec // G115: spec-mandated uint32→int32 reinterpretation
					result[i] = float64(int32(byteOrder.Uint32(rawData[i*4 : i*4+4])))
				}
			} else {
				for i := uint64(0); i < numElements; i++ {
					result[i] = float64(byteOrder.Uint32(rawData[i*4 : i*4+4]))
				}
			}
		case 8:
			if numElements*8 > uint64(len(rawData)) {
				return nil, errors.New("data truncated (8-byte int)")
			}
			if signed {
				for i := uint64(0); i < numElements; i++ {
					//nolint:gosec // G115: spec-mandated uint64→int64 reinterpretation
					result[i] = float64(int64(byteOrder.Uint64(rawData[i*8 : i*8+8])))
				}
			} else {
				for i := uint64(0); i < numElements; i++ {
					// uint64 above 2^53 loses precision in float64; this
					// matches the existing int64 path's behavior and is
					// what the public Read API (returning []float64) can
					// represent.
					result[i] = float64(byteOrder.Uint64(rawData[i*8 : i*8+8]))
				}
			}
		default:
			return nil, fmt.Errorf("unsupported fixed-point width %d bytes", datatype.Size)
		}

	default:
		return nil, fmt.Errorf("unsupported datatype for conversion to float64: %s", datatype)
	}

	return result, nil
}

// ReadDatasetInfo returns dataset metadata without reading actual data.
func ReadDatasetInfo(header *ObjectHeader, sb *Superblock) (*DatasetInfo, error) {
	var datatypeMsg, dataspaceMsg, layoutMsg *HeaderMessage

	for _, msg := range header.Messages {
		switch msg.Type {
		case MsgDatatype:
			datatypeMsg = msg
		case MsgDataspace:
			dataspaceMsg = msg
		case MsgDataLayout:
			layoutMsg = msg
		}
	}

	if datatypeMsg == nil || dataspaceMsg == nil || layoutMsg == nil {
		return nil, errors.New("missing required messages")
	}

	datatype, err := ParseDatatypeMessage(datatypeMsg.Data)
	if err != nil {
		return nil, err
	}

	dataspace, err := ParseDataspaceMessage(dataspaceMsg.Data)
	if err != nil {
		return nil, err
	}

	layout, err := ParseDataLayoutMessage(layoutMsg.Data, sb)
	if err != nil {
		return nil, err
	}

	return &DatasetInfo{
		Datatype:  datatype,
		Dataspace: dataspace,
		Layout:    layout,
	}, nil
}

// DatasetInfo holds metadata about a dataset.
type DatasetInfo struct {
	Datatype  *DatatypeMessage
	Dataspace *DataspaceMessage
	Layout    *DataLayoutMessage
}

// String returns human-readable dataset info.
func (di *DatasetInfo) String() string {
	return fmt.Sprintf(
		"Dataset: %s, %s, %s",
		di.Datatype.String(),
		di.Dataspace.String(),
		di.Layout.String(),
	)
}

// readChunkedData reads data from chunked layout.
func readChunkedData(r io.ReaderAt, layout *DataLayoutMessage, dataspace *DataspaceMessage, datatype *DatatypeMessage, sb *Superblock, filterPipeline *FilterPipelineMessage) ([]byte, error) {
	// Calculate total data size.
	totalElements := dataspace.TotalElements()
	elementSize := uint64(datatype.Size)

	// CVE-2025-7067 fix: Check for overflow in total size calculation.
	totalBytes, err := utils.SafeMultiply(totalElements, elementSize)
	if err != nil {
		return nil, fmt.Errorf("dataset size overflow: %w", err)
	}

	// Validate total size is within reasonable limits.
	if err := utils.ValidateBufferSize(totalBytes, utils.MaxChunkSize*1024, "dataset"); err != nil {
		return nil, fmt.Errorf("dataset too large: %w", err)
	}

	// Allocate output buffer.
	rawData := make([]byte, totalBytes)

	// Collect the chunk list from whichever chunk index the layout uses:
	// layout v3 files index chunks with a v1 B-tree; layout v4/v5 files
	// (HDF5 1.10+ under H5F_LIBVER_LATEST) use one of the modern indexes.
	chunks, err := CollectChunks(r, layout, dataspace, sb)
	if err != nil {
		return nil, err
	}

	// Read each chunk and copy to correct position.
	dataDims := dataspace.Dimensions
	actualChunkDims := layout.ChunkSize[:len(dataDims)]

	for _, chunk := range chunks {
		// CVE-2025-7067 fix: Validate chunk size before allocation to prevent buffer overflow.
		if err := utils.ValidateBufferSize(chunk.NBytes, utils.MaxChunkSize, "chunk data"); err != nil {
			return nil, fmt.Errorf("invalid chunk size at 0x%x: %w", chunk.Address, err)
		}

		// Read chunk data.
		chunkData := make([]byte, chunk.NBytes)
		//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
		_, err := r.ReadAt(chunkData, int64(chunk.Address))
		if err != nil {
			return nil, fmt.Errorf("failed to read chunk at 0x%x: %w", chunk.Address, err)
		}

		// Apply filters (decompression, etc) if present.
		if filterPipeline != nil && !chunkStoredUnfiltered(layout, chunk, actualChunkDims, dataDims) {
			if chunk.FilterMask != 0 {
				return nil, fmt.Errorf("per-chunk filter masks not supported (chunk at 0x%x, mask 0x%x)",
					chunk.Address, chunk.FilterMask)
			}
			chunkData, err = filterPipeline.ApplyFilters(chunkData)
			if err != nil {
				return nil, fmt.Errorf("failed to apply filters to chunk at 0x%x: %w", chunk.Address, err)
			}
		}

		// Calculate where this chunk goes in the output array.
		// For N-dimensional dataset, chunk [i0, i1, ...] maps to elements:
		// [i0*chunk[0] : (i0+1)*chunk[0], i1*chunk[1] : (i1+1)*chunk[1], ...].
		err = copyChunkToArray(chunkData, rawData, chunk.Scaled, actualChunkDims, dataDims, elementSize)
		if err != nil {
			return nil, fmt.Errorf("failed to copy chunk %v: %w", chunk.Scaled, err)
		}
	}

	return rawData, nil
}

// CollectChunks lists all allocated chunks of a chunked dataset, dispatching
// on the layout message version: v3 walks the v1 B-tree, v4/v5 walk the chunk
// index named in the message. Scaled coordinates are trimmed to the dataset
// rank in both cases.
func CollectChunks(r io.ReaderAt, layout *DataLayoutMessage, dataspace *DataspaceMessage, sb *Superblock) ([]ChunkLocation, error) {
	if layout.Version >= 4 {
		return CollectChunksV4(r, layout, dataspace, sb)
	}

	// Layout v3: chunk index is a v1 B-tree. Note: chunk dimensions include
	// an extra trailing dimension for the datatype size (see H5Dbtree.c).
	ndims := len(layout.ChunkSize)
	btree, err := ParseBTreeV1Node(r, layout.DataAddress, sb.OffsetSize, ndims, layout.ChunkSize)
	if err != nil {
		return nil, fmt.Errorf("failed to parse B-tree: %w", err)
	}

	chunks, err := btree.CollectAllChunks(r, sb.OffsetSize, layout.ChunkSize)
	if err != nil {
		return nil, fmt.Errorf("failed to collect chunks: %w", err)
	}

	rank := len(dataspace.Dimensions)
	entries := make([]ChunkLocation, 0, len(chunks))
	for _, chunk := range chunks {
		if rank > len(chunk.Key.Scaled) {
			return nil, fmt.Errorf("chunk key rank %d smaller than dataset rank %d", len(chunk.Key.Scaled), rank)
		}
		entries = append(entries, ChunkLocation{
			Scaled:     chunk.Key.Scaled[:rank],
			Address:    chunk.Address,
			NBytes:     uint64(chunk.Key.Nbytes),
			FilterMask: chunk.Key.FilterMask,
		})
	}
	return entries, nil
}

// chunkStoredUnfiltered reports whether this particular chunk bypassed the
// filter pipeline: layout v4/v5 can flag partial edge chunks as stored raw
// (H5O_LAYOUT_CHUNK_DONT_FILTER_PARTIAL_BOUND_CHUNKS).
func chunkStoredUnfiltered(layout *DataLayoutMessage, chunk ChunkLocation, chunkDims, dataDims []uint64) bool {
	if layout.Version < 4 || layout.Flags&LayoutChunkDontFilterPartialChunks == 0 {
		return false
	}
	for i, coord := range chunk.Scaled {
		if (coord+1)*chunkDims[i] > dataDims[i] {
			return true // Partial edge chunk: stored without filtering.
		}
	}
	return false
}

// copyChunkToArray copies chunk data to the correct position in full array.
// This handles multi-dimensional indexing and partial chunks at boundaries.
func copyChunkToArray(chunkData, fullData []byte, chunkCoords, chunkSize, dataDims []uint64, elemSize uint64) error {
	ndims := len(chunkCoords)
	if ndims != len(chunkSize) || ndims != len(dataDims) {
		return errors.New("dimension mismatch")
	}

	// Use general N-dimensional algorithm.
	return copyNDChunk(chunkData, fullData, chunkCoords, chunkSize, dataDims, elemSize)
}

// copyNDChunk copies an N-dimensional chunk to the full N-dimensional array.
// Uses general algorithm that works for any number of dimensions.
func copyNDChunk(chunkData, fullData []byte, chunkCoords, chunkSize, dataDims []uint64, elemSize uint64) error {
	ndims := len(chunkCoords)

	// Calculate strides for both chunk and full array.
	// Stride[i] = product of all dimensions after i.
	chunkStrides := make([]uint64, ndims)
	dataStrides := make([]uint64, ndims)

	chunkStrides[ndims-1] = 1
	dataStrides[ndims-1] = 1
	for i := ndims - 2; i >= 0; i-- {
		chunkStrides[i] = chunkStrides[i+1] * chunkSize[i+1]
		dataStrides[i] = dataStrides[i+1] * dataDims[i+1]
	}

	// Calculate actual dimensions to copy (may be less than chunk size at boundaries).
	copyDims := make([]uint64, ndims)
	for i := 0; i < ndims; i++ {
		// Starting position of this chunk in dataset.
		startPos := chunkCoords[i] * chunkSize[i]
		// Maximum elements we can copy in this dimension.
		maxCopy := chunkSize[i]
		if startPos+maxCopy > dataDims[i] {
			maxCopy = dataDims[i] - startPos
		}
		copyDims[i] = maxCopy
	}

	// Calculate starting offset in full array for this chunk.
	dataOffset := uint64(0)
	for i := 0; i < ndims; i++ {
		dataOffset += chunkCoords[i] * chunkSize[i] * dataStrides[i]
	}

	// Use recursive N-dimensional iteration to copy elements.
	indices := make([]uint64, ndims)
	return copyNDChunkRecursive(chunkData, fullData, indices, 0, copyDims, chunkStrides, dataStrides, dataOffset, elemSize)
}

// copyNDChunkRecursive recursively iterates through N-dimensional indices.
func copyNDChunkRecursive(chunkData, fullData []byte, indices []uint64, dim int, copyDims, chunkStrides, dataStrides []uint64, dataBaseOffset, elemSize uint64) error {
	ndims := len(indices)

	if dim == ndims-1 {
		// Base case: copy a contiguous row.
		numElements := copyDims[dim]

		// Calculate source offset in chunk.
		chunkOffset := uint64(0)
		for i := 0; i < ndims; i++ {
			chunkOffset += indices[i] * chunkStrides[i]
		}
		chunkOffset *= elemSize

		// Calculate destination offset in full array.
		dataOffset := dataBaseOffset
		for i := 0; i < ndims-1; i++ {
			dataOffset += indices[i] * dataStrides[i]
		}
		dataOffset *= elemSize

		numBytes := numElements * elemSize

		// Bounds check.
		if chunkOffset+numBytes > uint64(len(chunkData)) {
			return fmt.Errorf("chunk data truncated: need %d bytes at offset %d, have %d total",
				numBytes, chunkOffset, len(chunkData))
		}
		if dataOffset+numBytes > uint64(len(fullData)) {
			return fmt.Errorf("full data overflow: need %d bytes at offset %d, have %d total",
				numBytes, dataOffset, len(fullData))
		}

		// Copy the row.
		copy(fullData[dataOffset:dataOffset+numBytes], chunkData[chunkOffset:chunkOffset+numBytes])
		return nil
	}

	// Recursive case: iterate through this dimension.
	for indices[dim] = 0; indices[dim] < copyDims[dim]; indices[dim]++ {
		err := copyNDChunkRecursive(chunkData, fullData, indices, dim+1, copyDims, chunkStrides, dataStrides, dataBaseOffset, elemSize)
		if err != nil {
			return err
		}
	}

	return nil
}

// float16ToFloat64 converts an IEEE 754 half-precision bit pattern to float64.
// Handles subnormals, infinities, and NaN per the standard.
func float16ToFloat64(bits uint16) float64 {
	sign := uint64(bits>>15) & 1
	exp := uint64(bits>>10) & 0x1F
	frac := uint64(bits) & 0x3FF

	var f64bits uint64
	switch exp {
	case 0x1F:
		// Inf / NaN: max exponent, preserve fraction (NaN payload).
		f64bits = sign<<63 | 0x7FF<<52 | frac<<42
	case 0:
		if frac == 0 {
			// Signed zero.
			f64bits = sign << 63
		} else {
			// Subnormal: value = frac * 2^-24. Convert exactly via math.
			return math.Copysign(float64(frac)*0x1p-24, float64(1-2*int(sign)))
		}
	default:
		// Normal: rebias exponent 15 -> 1023.
		f64bits = sign<<63 | (exp+1023-15)<<52 | frac<<42
	}
	return math.Float64frombits(f64bits)
}
