package core

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/shyrmapp/hdf5/internal/utils"
)

// CompoundValue represents a single compound structure instance as a map of field names to values.
type CompoundValue map[string]interface{}

// ReadDatasetCompound reads a dataset with compound datatype and returns array of compound values.
func ReadDatasetCompound(r io.ReaderAt, header *ObjectHeader, sb *Superblock, maxBytes uint64) ([]CompoundValue, error) {
	// 1. Extract required messages.
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

	// Validate.
	if datatypeMsg == nil {
		return nil, errors.New("datatype message not found")
	}
	if dataspaceMsg == nil {
		return nil, errors.New("dataspace message not found")
	}
	if layoutMsg == nil {
		return nil, errors.New("data layout message not found")
	}

	// 2. Parse datatype.
	datatype, err := ParseDatatypeMessage(datatypeMsg.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse datatype: %w", err)
	}

	if !datatype.IsCompound() {
		return nil, errors.New("not a compound datatype")
	}

	// Parse compound structure.
	compoundType, err := ParseCompoundType(datatype)
	if err != nil {
		return nil, fmt.Errorf("failed to parse compound type: %w", err)
	}

	// 3. Parse dataspace.
	dataspace, err := ParseDataspaceMessage(dataspaceMsg.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse dataspace: %w", err)
	}

	// 4. Parse layout.
	layout, err := ParseDataLayoutMessage(layoutMsg.Data, sb)
	if err != nil {
		return nil, fmt.Errorf("failed to parse layout: %w", err)
	}

	// 5. Parse filter pipeline (optional).
	var filterPipeline *FilterPipelineMessage
	if filterPipelineMsg != nil {
		filterPipeline, err = ParseFilterPipelineMessage(filterPipelineMsg.Data)
		if err != nil {
			return nil, fmt.Errorf("failed to parse filter pipeline: %w", err)
		}
	}

	// 6. Calculate total number of elements.
	totalElements := dataspace.TotalElements()
	if totalElements == 0 {
		return []CompoundValue{}, nil
	}

	// The decoded rows (one map each) cost far more than the raw bytes the
	// layout readers check, so bound the output before reading anything.
	if err := checkReadSize(totalElements, compoundRowCost(compoundType), maxBytes, "compound output"); err != nil {
		return nil, err
	}

	// 7. Read raw data based on layout.
	var rawData []byte

	switch {
	case layout.IsCompact():
		rawData = layout.CompactData
		if datatype.Size == 0 || totalElements > uint64(len(rawData))/uint64(datatype.Size) {
			return nil, fmt.Errorf("compact data holds %d bytes, fewer than %d rows of %d bytes", len(rawData), totalElements, datatype.Size)
		}

	case layout.IsContiguous():
		rawData, err = readContiguousData(r, layout.DataAddress, totalElements, uint64(datatype.Size), maxBytes, sb.OffsetSize, FillValueBytes(header, datatype.Size))
		if err != nil {
			return nil, err
		}

	case layout.IsChunked():
		rawData, err = readChunkedData(r, layout, dataspace, datatype, sb, filterPipeline, maxBytes, FillValueBytes(header, datatype.Size))
		if err != nil {
			return nil, fmt.Errorf("failed to read chunked data: %w", err)
		}

	default:
		return nil, fmt.Errorf("unsupported layout class: %d", layout.Class)
	}

	// 8. Convert raw bytes to compound values.
	return parseCompoundData(rawData, compoundType, totalElements, r, sb)
}

// parseCompoundData parses raw bytes into array of compound values.
func parseCompoundData(rawData []byte, compoundType *CompoundType, numElements uint64, r io.ReaderAt, sb *Superblock) ([]CompoundValue, error) {
	result := make([]CompoundValue, numElements)
	structSize := uint64(compoundType.Size)

	for i := uint64(0); i < numElements; i++ {
		structOffset := i * structSize

		if structOffset+structSize > uint64(len(rawData)) {
			return nil, fmt.Errorf("data truncated at element %d", i)
		}

		structData := rawData[structOffset : structOffset+structSize]
		value := make(CompoundValue)

		// Parse each member.
		for _, member := range compoundType.Members {
			if uint64(member.Offset) > structSize {
				return nil, fmt.Errorf("member %s offset %d exceeds compound size %d",
					member.Name, member.Offset, structSize)
			}
			memberData := structData[member.Offset:]

			var memberValue interface{}
			var err error
			if len(member.Dims) > 0 {
				memberValue, err = decodeArray(memberData, member.Type, member.Dims, r, sb)
			} else {
				memberValue, err = parseMemberValue(memberData, member.Type, r, sb)
			}
			if err != nil {
				return nil, fmt.Errorf("failed to parse member %s: %w", member.Name, err)
			}

			value[member.Name] = memberValue
		}

		result[i] = value
	}

	return result, nil
}

// compoundRowCost estimates the bytes one decoded row occupies: a map header
// plus, per member, a key, an interface value and its payload. Deliberately
// generous so a tiny struct cannot expand 100x past the caller's limit.
func compoundRowCost(c *CompoundType) uint64 {
	cost := uint64(64)
	for _, m := range c.Members {
		cost += 80
		if m.Type.IsCompound() {
			if nested, err := ParseCompoundType(m.Type); err == nil {
				cost += compoundRowCost(nested)
			}
		}
		n := uint64(1)
		for _, d := range m.Dims {
			n *= uint64(d)
		}
		cost += (n - 1) * 24
	}
	return cost
}

// arrayDatatype decodes an array datatype's dimensions and base type.
func arrayDatatype(dt *DatatypeMessage) ([]uint32, *DatatypeMessage, error) {
	p := dt.Properties
	ndims, dimsOff, off, ok := arrayLayout(dt.Version, p)
	if !ok {
		return nil, nil, errors.New("array datatype truncated")
	}
	dims := make([]uint32, ndims)
	for i := range dims {
		dims[i] = binary.LittleEndian.Uint32(p[dimsOff+4*i:])
	}
	base, err := ParseDatatypeMessage(p[off:])
	if err != nil {
		return nil, nil, fmt.Errorf("array base type: %w", err)
	}
	return dims, base, nil
}

// decodeArray decodes a row-major array member into a flat []interface{} of
// natural-typed elements (multi-dimensional arrays are flattened).
func decodeArray(data []byte, base *DatatypeMessage, dims []uint32, r io.ReaderAt, sb *Superblock) (interface{}, error) {
	if base.Size == 0 {
		return nil, errors.New("array member with zero-size base type")
	}
	maxN := uint64(len(data)) / uint64(base.Size)
	n := uint64(1)
	for _, d := range dims {
		if d != 0 && n > maxN/uint64(d) {
			return nil, errors.New("array member larger than its compound")
		}
		n *= uint64(d)
	}
	out := make([]interface{}, n)
	for i := range out {
		v, err := parseMemberValue(data[uint64(i)*uint64(base.Size):], base, r, sb)
		if err != nil {
			return nil, fmt.Errorf("array element %d: %w", i, err)
		}
		out[i] = v
	}
	return out, nil
}

// parseMemberValue parses a single member value based on its datatype.
func parseMemberValue(data []byte, datatype *DatatypeMessage, r io.ReaderAt, sb *Superblock) (interface{}, error) {
	byteOrder := datatype.GetByteOrder()

	switch {
	case datatype.IsFloat64():
		if len(data) < 8 {
			return nil, errors.New("insufficient data for float64")
		}
		bits := byteOrder.Uint64(data[0:8])
		return math.Float64frombits(bits), nil

	case datatype.IsFloat32():
		if len(data) < 4 {
			return nil, errors.New("insufficient data for float32")
		}
		bits := byteOrder.Uint32(data[0:4])
		return math.Float32frombits(bits), nil

	case datatype.Class == DatatypeFixed:
		// Same signedness/width/byte-order decoding as attributes.
		return readFixedAttr(&Attribute{Datatype: datatype, Data: data}, 1, true)

	case datatype.Class == DatatypeArray:
		dims, base, err := arrayDatatype(datatype)
		if err != nil {
			return nil, err
		}
		return decodeArray(data, base, dims, r, sb)

	case datatype.IsFixedString():
		// CVE-2025-2926 fix: Validate string size before processing.
		stringSize := uint64(datatype.Size)
		if err := utils.ValidateBufferSize(stringSize, utils.MaxStringSize, "compound string field"); err != nil {
			return nil, fmt.Errorf("string field too large: %w", err)
		}

		//nolint:gosec // G115: Safe length comparison
		if uint32(len(data)) < datatype.Size {
			return nil, errors.New("insufficient data for string")
		}
		// Extract string based on padding type.
		str := extractString(data[0:datatype.Size], datatype.GetStringPadding())
		return str, nil

	case datatype.IsVariableString():
		// Variable-length strings are stored as global heap references.
		// Parse the reference: heap_address (offset_size bytes) + object_index (4 bytes).
		return readVariableString(r, data, sb)

	case datatype.IsCompound():
		// Nested compound - recursive parse.
		nestedCompound, err := ParseCompoundType(datatype)
		if err != nil {
			return nil, fmt.Errorf("failed to parse nested compound: %w", err)
		}
		values, err := parseCompoundData(data, nestedCompound, 1, r, sb)
		if err != nil {
			return nil, err
		}
		return values[0], nil

	default:
		return nil, fmt.Errorf("unsupported member datatype: %s", datatype)
	}
}

// extractString extracts a string from fixed-length byte array based on padding type.
func extractString(data []byte, paddingType uint8) string {
	switch paddingType {
	case 0: // Null-terminated.
		for i, b := range data {
			if b == 0 {
				return string(data[:i])
			}
		}
		return string(data)

	case 1: // Null-padded.
		end := len(data)
		for end > 0 && data[end-1] == 0 {
			end--
		}
		return string(data[:end])

	case 2: // Space-padded.
		end := len(data)
		for end > 0 && (data[end-1] == ' ' || data[end-1] == 0) {
			end--
		}
		return string(data[:end])

	default:
		return string(data)
	}
}

// readVariableString reads a variable-length string from the Global Heap.
//
// An on-disk vlen element is a 4-byte length followed by the global heap ID
// (heap address, offset_size bytes, then a 4-byte object index). The length
// prefix is skipped: the true length comes from the heap object itself, and
// reading the reference from offset 0 would fold the length into the address.
func readVariableString(r io.ReaderAt, data []byte, sb *Superblock) (string, error) {
	offsetSize := int(sb.OffsetSize)

	const lengthPrefix = 4
	if len(data) < lengthPrefix+offsetSize+4 {
		return "", fmt.Errorf("vlen string element too short: got %d bytes, need %d",
			len(data), lengthPrefix+offsetSize+4)
	}

	ref, err := ParseGlobalHeapReference(data[lengthPrefix:], offsetSize)
	if err != nil {
		return "", fmt.Errorf("failed to parse global heap reference: %w", err)
	}

	// Check for null reference (address 0 typically means empty/null string).
	if ref.HeapAddress == 0 {
		return "", nil
	}

	// Read the global heap collection.
	collection, err := ReadGlobalHeapCollection(r, ref.HeapAddress, offsetSize)
	if err != nil {
		return "", fmt.Errorf("failed to read global heap collection at 0x%X: %w", ref.HeapAddress, err)
	}

	// Get the object from the collection.
	obj, err := collection.GetObject(ref.ObjectIndex)
	if err != nil {
		return "", fmt.Errorf("failed to get object %d from heap collection: %w", ref.ObjectIndex, err)
	}

	// Convert object data to string.
	// Variable-length strings are typically null-terminated.
	str := string(obj.Data)

	// Remove trailing null bytes if present.
	for str != "" && str[len(str)-1] == 0 {
		str = str[:len(str)-1]
	}

	return str, nil
}
