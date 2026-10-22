// Package core provides HDF5 file format parsing and manipulation functionality.
package core

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/shyrmapp/hdf5/internal/utils"
)

// Attribute represents an HDF5 attribute with metadata and data.
type Attribute struct {
	Name      string
	Datatype  *DatatypeMessage
	Dataspace *DataspaceMessage
	Data      []byte

	// For variable-length types, we need access to the file reader
	// to resolve Global Heap references.
	reader     io.ReaderAt
	offsetSize int
}

// AttributeInfoMessage represents the Attribute Info Message (0x0015).
// This message contains information about dense attribute storage.
// Reference: H5Adense.c in C library.
type AttributeInfoMessage struct {
	Version            uint8
	Flags              uint8
	FractalHeapAddr    uint64 // Address of fractal heap for dense attribute storage
	BTreeNameIndexAddr uint64 // Address of B-tree v2 for indexing attributes by name
	// Optional fields for creation order (if tracked):
	MaxCreationIndex    uint64 // Only present if creation order tracked
	BTreeOrderIndexAddr uint64 // Only present if creation order indexed
}

// ParseAttributeMessage parses an attribute message (type 0x000C).
// Format according to HDF5 spec:
// - Version (1 byte).
// - Flags (1 byte) - reserved, always 0.
// - Name size (2 bytes).
// - Datatype size (2 bytes).
// - Dataspace size (2 bytes).
// - Name encoding (1 byte) - for version 3+.
// - Name (variable, null-terminated).
// - Datatype message data.
// - Dataspace message data.
// - Data (variable).
func ParseAttributeMessage(data []byte, endianness binary.ByteOrder) (*Attribute, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("attribute message too short: %d bytes", len(data))
	}

	attr := &Attribute{}
	offset := 0

	// Version.
	version := data[offset]
	offset++

	// Flags (reserved).
	offset++

	// Name size (2 bytes).
	nameSize := endianness.Uint16(data[offset : offset+2])
	offset += 2

	// Datatype size (2 bytes).
	datatypeSize := endianness.Uint16(data[offset : offset+2])
	offset += 2

	// Dataspace size (2 bytes).
	dataspaceSize := endianness.Uint16(data[offset : offset+2])
	offset += 2

	// For version 3+, there's a name encoding byte.
	if version >= 3 {
		// Name encoding (1 byte) - 0 = ASCII, 1 = UTF-8.
		offset++
	}

	// Read name (null-terminated).
	if offset+int(nameSize) > len(data) {
		return nil, fmt.Errorf("name extends beyond message: offset=%d, nameSize=%d, dataLen=%d",
			offset, nameSize, len(data))
	}

	// Name is null-terminated, so subtract 1 for the null byte.
	if nameSize > 0 {
		attr.Name = string(data[offset : offset+int(nameSize)-1])
	}

	// For version 1 and 2, name/datatype/dataspace are padded to 8-byte boundaries.
	// For version 3+, no padding (sizes are exact).
	// Reference: H5Oattr.c - H5O_ALIGN_OLD macro: (8 * (((X) + 7) / 8))
	alignTo8 := func(size uint16) int {
		return int((size + 7) & ^uint16(7))
	}

	if version < 3 {
		// V1/V2: Pad to 8-byte boundaries
		offset += alignTo8(nameSize)
	} else {
		// V3+: Exact sizes
		offset += int(nameSize)
	}

	// Parse datatype.
	if offset+int(datatypeSize) > len(data) {
		return nil, fmt.Errorf("datatype extends beyond message")
	}

	datatypeData := data[offset : offset+int(datatypeSize)]
	var err error
	attr.Datatype, err = ParseDatatypeMessage(datatypeData)
	if err != nil {
		return nil, fmt.Errorf("datatype parse failed: %w", err)
	}

	if version < 3 {
		offset += alignTo8(datatypeSize)
	} else {
		offset += int(datatypeSize)
	}

	// Parse dataspace.
	if offset+int(dataspaceSize) > len(data) {
		return nil, fmt.Errorf("dataspace extends beyond message")
	}

	dataspaceData := data[offset : offset+int(dataspaceSize)]
	attr.Dataspace, err = ParseDataspaceMessage(dataspaceData)
	if err != nil {
		return nil, fmt.Errorf("dataspace parse failed: %w", err)
	}

	if version < 3 {
		offset += alignTo8(dataspaceSize)
	} else {
		offset += int(dataspaceSize)
	}

	// Remaining data is the attribute value.
	if offset < len(data) {
		//nolint:gosec // G115: Safe conversion - offset is bounded by len(data)
		attrDataSize := uint64(len(data) - offset)

		// CVE-2025-6269 fix: Validate attribute data size before allocation.
		if err := utils.ValidateBufferSize(attrDataSize, utils.MaxAttributeSize, "attribute data"); err != nil {
			return nil, fmt.Errorf("invalid attribute size: %w", err)
		}

		attr.Data = make([]byte, attrDataSize)
		copy(attr.Data, data[offset:])
	}

	return attr, nil
}

// decodeNumeric decodes totalElements fixed-width values from an attribute's raw
// data using decode, which reads exactly sizeof(T) bytes. Scalars are unwrapped.
//
// typeName only appears in the overflow error message.
func decodeNumeric[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64](
	a *Attribute, totalElements uint64, isScalar bool, typeName string, decode func([]byte) T,
) (interface{}, error) {
	elemSize := uint64(a.Datatype.Size)

	// CVE-2025-6269 fix: Check for multiplication overflow before processing.
	totalBytes, err := utils.SafeMultiply(totalElements, elemSize)
	if err != nil {
		return nil, fmt.Errorf("attribute size overflow (%s): %w", typeName, err)
	}

	if totalBytes > uint64(len(a.Data)) {
		return nil, fmt.Errorf("attribute data size mismatch: need %d bytes, have %d",
			totalBytes, len(a.Data))
	}

	values := make([]T, totalElements)
	for i := uint64(0); i < totalElements; i++ {
		offset := i * elemSize
		values[i] = decode(a.Data[offset : offset+elemSize])
	}
	if isScalar {
		return values[0], nil
	}
	return values, nil
}

// readFixedAttr decodes an integer attribute using the datatype's byte order
// and signedness, returning the natural Go type (int8..int64, uint8..uint64).
//
//nolint:gosec // G115: HDF5 binary format requires same-width sign reinterpretation
func readFixedAttr(a *Attribute, n uint64, isScalar bool) (interface{}, error) {
	bo := a.Datatype.GetByteOrder()
	signed := a.Datatype.IsSignedFixedPoint()
	switch a.Datatype.Size {
	case 1:
		if signed {
			return decodeNumeric(a, n, isScalar, "int8", func(b []byte) int8 { return int8(b[0]) })
		}
		return decodeNumeric(a, n, isScalar, "uint8", func(b []byte) uint8 { return b[0] })
	case 2:
		if signed {
			return decodeNumeric(a, n, isScalar, "int16", func(b []byte) int16 { return int16(bo.Uint16(b)) })
		}
		return decodeNumeric(a, n, isScalar, "uint16", bo.Uint16)
	case 4:
		if signed {
			return decodeNumeric(a, n, isScalar, "int32", func(b []byte) int32 { return int32(bo.Uint32(b)) })
		}
		return decodeNumeric(a, n, isScalar, "uint32", bo.Uint32)
	case 8:
		if signed {
			return decodeNumeric(a, n, isScalar, "int64", func(b []byte) int64 { return int64(bo.Uint64(b)) })
		}
		return decodeNumeric(a, n, isScalar, "uint64", bo.Uint64)
	}
	return nil, fmt.Errorf("unsupported datatype class %d or size %d", a.Datatype.Class, a.Datatype.Size)
}

// ReadValue reads the attribute value as the appropriate Go type.
func (a *Attribute) ReadValue() (interface{}, error) {
	if a.Datatype == nil || a.Dataspace == nil {
		return nil, fmt.Errorf("attribute missing datatype or dataspace")
	}

	totalElements := a.Dataspace.TotalElements()
	if totalElements == 0 || len(a.Data) == 0 {
		// Empty attribute - return empty slice instead of nil.
		return []interface{}{}, nil
	}

	// For scalar attributes, return single value.
	isScalar := len(a.Dataspace.Dimensions) == 0 ||
		(len(a.Dataspace.Dimensions) == 1 && a.Dataspace.Dimensions[0] == 1)

	switch a.Datatype.Class {
	case DatatypeFixed:
		return readFixedAttr(a, totalElements, isScalar)

	case DatatypeFloat:
		bo := a.Datatype.GetByteOrder()
		switch a.Datatype.Size {
		case 4:
			return decodeNumeric(a, totalElements, isScalar, "float32", func(b []byte) float32 {
				return math.Float32frombits(bo.Uint32(b))
			})
		case 8:
			return decodeNumeric(a, totalElements, isScalar, "float64", func(b []byte) float64 {
				return math.Float64frombits(bo.Uint64(b))
			})
		}

	case DatatypeString:
		// Fixed-length strings.
		if !a.Datatype.IsFixedString() {
			return nil, fmt.Errorf("variable-length strings not yet supported in attributes")
		}

		stringSize := uint64(a.Datatype.Size)
		paddingType := a.Datatype.GetStringPadding()
		// Bound by the data actually present before allocating.
		if need, err := utils.SafeMultiply(totalElements, stringSize); err != nil || need > uint64(len(a.Data)) {
			return nil, fmt.Errorf("attribute data too short for %d strings of %d bytes", totalElements, stringSize)
		}
		values := make([]string, totalElements)

		for i := uint64(0); i < totalElements; i++ {
			offset := i * stringSize
			if offset+stringSize > uint64(len(a.Data)) {
				return nil, fmt.Errorf("data too short for string element %d", i)
			}
			stringBytes := a.Data[offset : offset+stringSize]
			values[i] = decodeFixedString(stringBytes, paddingType)
		}

		if isScalar {
			return values[0], nil
		}
		return values, nil

	case DatatypeVarLen:
		// Variable-length types (most commonly variable-length strings).
		// Data is stored as Global Heap references.
		if a.reader == nil {
			return nil, fmt.Errorf("variable-length attribute requires file reader (not available)")
		}

		// Check if this is a variable-length string.
		if !a.Datatype.IsVariableString() {
			return nil, fmt.Errorf("variable-length non-string types not yet supported in attributes")
		}

		// Each vlen element is: length (4 bytes) + heap_address (offsetSize bytes) + object_index (4 bytes).
		//nolint:gosec // G115: offsetSize is bounded to 4 or 8 by HDF5 format specification
		refSize := uint64(a.offsetSize + 8)

		// CVE-2025-6269 fix: Check for multiplication overflow before processing.
		totalBytes, err := utils.SafeMultiply(totalElements, refSize)
		if err != nil {
			return nil, fmt.Errorf("attribute size overflow (vlen string): %w", err)
		}

		if totalBytes > uint64(len(a.Data)) {
			return nil, fmt.Errorf("attribute data size mismatch for vlen strings: need %d bytes, have %d",
				totalBytes, len(a.Data))
		}

		values := make([]string, totalElements)
		for i := uint64(0); i < totalElements; i++ {
			offset := i * refSize
			elementData := a.Data[offset : offset+refSize]

			str, err := a.readVariableLengthString(elementData)
			if err != nil {
				return nil, fmt.Errorf("failed to read variable-length string element %d: %w", i, err)
			}
			values[i] = str
		}

		if isScalar {
			return values[0], nil
		}
		return values, nil
	}

	return nil, fmt.Errorf("unsupported datatype class %d or size %d", a.Datatype.Class, a.Datatype.Size)
}

// readVariableLengthString reads a variable-length string from the Global Heap.
//
// For variable-length strings in attributes, the format is:
//   - Length (4 bytes): Number of bytes in the string
//   - Heap Address (offsetSize bytes): Address of the Global Heap collection
//   - Object Index (4 bytes): Index of the object within the collection
//
// Reference: HDF5 Format Specification III.E (Global Heap), H5Tvlen.c.
func (a *Attribute) readVariableLengthString(data []byte) (string, error) {
	// Expected size: 4 (length) + offsetSize (heap address) + 4 (object index)
	expectedSize := 4 + a.offsetSize + 4
	if len(data) < expectedSize {
		return "", fmt.Errorf("vlen string data too short: got %d bytes, need %d", len(data), expectedSize)
	}

	// Parse the length field (4 bytes, little-endian).
	// This is the length of the string in bytes.
	_ = binary.LittleEndian.Uint32(data[0:4]) // length, not used directly but verified via heap object

	// Parse the global heap reference (after the length field).
	ref, err := ParseGlobalHeapReference(data[4:], a.offsetSize)
	if err != nil {
		return "", fmt.Errorf("failed to parse global heap reference: %w", err)
	}

	// Check for null reference (address 0 typically means empty/null string).
	if ref.HeapAddress == 0 {
		return "", nil
	}

	// Read the global heap collection.
	collection, err := ReadGlobalHeapCollection(a.reader, ref.HeapAddress, a.offsetSize)
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

// ParseAttributesFromMessages extracts all attributes from object header messages.
// Supports both compact attributes (stored in object header) and dense attributes
// (stored in fractal heap).
// Reference: H5Adense.c in C library.
func ParseAttributesFromMessages(r io.ReaderAt, messages []*HeaderMessage, sb *Superblock) ([]*Attribute, error) {
	var attributes []*Attribute
	var attrInfo *AttributeInfoMessage

	// First pass: look for AttributeInfo message (dense storage)
	for _, msg := range messages {
		if msg.Type == MsgAttributeInfo {
			var err error
			attrInfo, err = ParseAttributeInfoMessage(msg.Data, sb)
			if err != nil {
				// If we can't parse attribute info, fall back to compact only
				attrInfo = nil
			}
			break
		}
	}

	// Second pass: collect compact attributes
	for _, msg := range messages {
		if msg.Type != MsgAttribute {
			continue
		}
		attr, err := ParseAttributeMessage(msg.Data, sb.Endianness)
		if err != nil {
			// Log error but continue with other attributes
			continue
		}
		// Set reader for variable-length type resolution
		attr.reader = r
		attr.offsetSize = int(sb.OffsetSize)
		attributes = append(attributes, attr)
	}

	// If dense storage exists, read attributes from fractal heap
	if attrInfo != nil && attrInfo.FractalHeapAddr != 0 {
		denseAttrs, err := readDenseAttributes(r, attrInfo, sb)
		if err != nil {
			return nil, fmt.Errorf("failed to read dense attributes: %w", err)
		}
		attributes = append(attributes, denseAttrs...)
	}

	return attributes, nil
}

// readDenseAttributes reads attributes from dense storage: the name-index
// B-tree v2 yields fractal-heap IDs, each resolving to an attribute message.
// Reference: H5Adense.c - H5A__dense_iterate().
func readDenseAttributes(r io.ReaderAt, attrInfo *AttributeInfoMessage, sb *Superblock) ([]*Attribute, error) {
	if attrInfo == nil || sb == nil {
		return nil, fmt.Errorf("attribute info or superblock is nil")
	}
	if attrInfo.FractalHeapAddr == 0 || attrInfo.BTreeNameIndexAddr == 0 {
		return nil, fmt.Errorf("invalid dense attribute addresses (heap=0x%X, btree=0x%X)",
			attrInfo.FractalHeapAddr, attrInfo.BTreeNameIndexAddr)
	}

	objects, err := ReadDenseHeapObjects(r, attrInfo.BTreeNameIndexAddr, attrInfo.FractalHeapAddr, sb)
	if err != nil {
		return nil, err
	}
	attributes := make([]*Attribute, 0, len(objects))
	for i, data := range objects {
		attr, err := ParseAttributeMessage(data, sb.Endianness)
		if err != nil {
			return nil, fmt.Errorf("failed to parse attribute %d: %w", i, err)
		}
		attr.reader = r
		attr.offsetSize = int(sb.OffsetSize)
		attributes = append(attributes, attr)
	}
	return attributes, nil
}

// EncodeAttributeFromStruct encodes an Attribute struct to bytes (attribute message format).
//
// This is a convenience wrapper around EncodeAttributeMessage (from messages_write.go)
// that takes an Attribute struct instead of individual parameters.
//
// Parameters:
//   - attr: Attribute to encode
//   - sb: Superblock for endianness (currently unused as we use little-endian)
//
// Returns:
//   - []byte: Encoded attribute message
//   - error: Non-nil if encoding fails
//
// Reference: H5Oattr.c - H5O__attr_encode().
func EncodeAttributeFromStruct(attr *Attribute, sb *Superblock) ([]byte, error) {
	if attr == nil {
		return nil, fmt.Errorf("attribute is nil")
	}

	if attr.Name == "" {
		return nil, fmt.Errorf("attribute name cannot be empty")
	}

	if attr.Datatype == nil {
		return nil, fmt.Errorf("attribute datatype is nil")
	}

	if attr.Dataspace == nil {
		return nil, fmt.Errorf("attribute dataspace is nil")
	}

	// Use the existing EncodeAttributeMessage from messages_write.go
	// It handles all the encoding logic according to HDF5 spec
	_ = sb // Currently unused, encoding is always little-endian

	return EncodeAttributeMessage(attr.Name, attr.Datatype, attr.Dataspace, attr.Data)
}

// ParseAttributeInfoMessage parses an Attribute Info Message (0x0015).
// Reference: H5Oainfo.c in C library.
// Format:
// - Version (1 byte)
// - Flags (1 byte):
//   - Bit 0: Track creation order
//   - Bit 1: Index creation order
//
// - Max Creation Index (2 bytes) - only if track creation order flag set
// - Fractal Heap Address (variable, based on superblock offset size)
// - B-tree Name Index Address (variable)
// - B-tree Order Index Address (variable) - only if index creation order flag set.
func ParseAttributeInfoMessage(data []byte, sb *Superblock) (*AttributeInfoMessage, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("attribute info message too short: %d bytes", len(data))
	}

	msg := &AttributeInfoMessage{}
	offset := 0

	// Version
	msg.Version = data[offset]
	offset++

	// Flags
	msg.Flags = data[offset]
	offset++

	trackCreationOrder := (msg.Flags & 0x01) != 0
	indexCreationOrder := (msg.Flags & 0x02) != 0

	// Maximum Creation Index (2 bytes) - only if creation order tracked
	if trackCreationOrder {
		if offset+2 > len(data) {
			return nil, fmt.Errorf("attribute info message too short for max creation index")
		}
		msg.MaxCreationIndex = uint64(sb.Endianness.Uint16(data[offset : offset+2]))
		offset += 2
	}

	// Fractal Heap Address (offset size bytes)
	offsetSize := int(sb.OffsetSize)
	if offset+offsetSize > len(data) {
		return nil, fmt.Errorf("attribute info message too short for fractal heap address")
	}
	msg.FractalHeapAddr = utils.ReadUint(data[offset:offset+offsetSize], offsetSize, binary.LittleEndian)
	offset += offsetSize
	// Objects that track creation order but still store their attributes in
	// the header carry undefined addresses; report those as "no dense storage" (0).
	undefined := isUndefinedAddress(msg.FractalHeapAddr, sb.OffsetSize)
	if undefined {
		msg.FractalHeapAddr = 0
	}

	// B-tree Name Index Address (offset size bytes)
	if offset+offsetSize > len(data) {
		return nil, fmt.Errorf("attribute info message too short for btree name index address")
	}
	msg.BTreeNameIndexAddr = utils.ReadUint(data[offset:offset+offsetSize], offsetSize, binary.LittleEndian)
	offset += offsetSize
	if undefined {
		msg.BTreeNameIndexAddr = 0
	}

	// B-tree Order Index Address (offset size bytes) - only if indexed
	if indexCreationOrder {
		if offset+offsetSize > len(data) {
			return nil, fmt.Errorf("attribute info message too short for btree order index address")
		}
		msg.BTreeOrderIndexAddr = utils.ReadUint(data[offset:offset+offsetSize], offsetSize, binary.LittleEndian)
		if isUndefinedAddress(msg.BTreeOrderIndexAddr, sb.OffsetSize) {
			msg.BTreeOrderIndexAddr = 0
		}
	}

	return msg, nil
}

// EncodeAttributeInfoMessage encodes Attribute Info Message for writing.
//
// This implements the encoding logic matching the C reference H5Oainfo.c:H5O__ainfo_encode().
//
// Format:
// - Version (1 byte): 0
// - Flags (1 byte): creation order tracking/indexing
// - Max Creation Index (2 bytes): Optional, if bit 0 of flags set
// - Fractal Heap Address (8 bytes)
// - B-tree Name Index Address (8 bytes)
// - B-tree Order Index Address (8 bytes): Optional, if bit 1 of flags set
//
// Parameters:
//   - aim: AttributeInfoMessage to encode
//   - sb: Superblock for endianness and offset size
//
// Returns:
//   - []byte: Encoded message
//   - error: Non-nil if encoding fails
//
// Reference: H5Oainfo.c - H5O__ainfo_encode().
func EncodeAttributeInfoMessage(aim *AttributeInfoMessage, sb *Superblock) ([]byte, error) {
	if aim == nil {
		return nil, fmt.Errorf("attribute info message is nil")
	}

	// Calculate size
	size := 2 // Version (1) + Flags (1)

	trackCreationOrder := (aim.Flags & 0x01) != 0
	indexCreationOrder := (aim.Flags & 0x02) != 0

	// Maximum Creation Index (2 bytes) if creation order tracked
	if trackCreationOrder {
		size += 2
	}

	// Heap address (offsetSize bytes)
	offsetSize := int(sb.OffsetSize)
	size += offsetSize

	// B-tree name index address (offsetSize bytes)
	size += offsetSize

	// B-tree order index address (offsetSize bytes) if indexed
	if indexCreationOrder {
		size += offsetSize
	}

	buf := make([]byte, size)
	offset := 0

	// Version
	buf[offset] = aim.Version
	offset++

	// Flags
	buf[offset] = aim.Flags
	offset++

	// Maximum Creation Index if creation order tracked
	if trackCreationOrder {
		sb.Endianness.PutUint16(buf[offset:], uint16(aim.MaxCreationIndex)) //nolint:gosec // Safe: validated range
		offset += 2
	}

	// Fractal Heap Address
	writeAddress(buf[offset:], aim.FractalHeapAddr, offsetSize, sb.Endianness)
	offset += offsetSize

	// B-tree Name Index Address
	writeAddress(buf[offset:], aim.BTreeNameIndexAddr, offsetSize, sb.Endianness)
	offset += offsetSize

	// B-tree Order Index Address if indexed
	if indexCreationOrder {
		writeAddress(buf[offset:], aim.BTreeOrderIndexAddr, offsetSize, sb.Endianness)
	}

	return buf, nil
}

// writeAddress writes an address to buffer with specified size and endianness.
// Helper function for encoding variable-size addresses.
func writeAddress(buf []byte, addr uint64, size int, endianness binary.ByteOrder) {
	switch size {
	case 1:
		buf[0] = byte(addr) //nolint:gosec // G115: variable-size address encoding
	case 2:
		endianness.PutUint16(buf, uint16(addr)) //nolint:gosec // Safe: size limited to 2 bytes
	case 4:
		endianness.PutUint32(buf, uint32(addr)) //nolint:gosec // Safe: size limited to 4 bytes
	case 8:
		endianness.PutUint64(buf, addr)
	}
}
