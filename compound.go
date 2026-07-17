package hdf5

import (
	"encoding/binary"
	"fmt"

	"github.com/scigolib/hdf5/internal/core"
)

// CompoundField describes one member of a compound (struct-like) datatype.
// Type is limited to the numeric datatypes (Int8..Int64, Uint8..Uint64,
// Float32, Float64) and String. String fields require StringSize > 0.
type CompoundField struct {
	Name       string   // Member name (must be unique within the compound).
	Type       Datatype // Member type (numeric constants or String).
	StringSize uint32   // Fixed string size in bytes (String fields only).
}

// CompoundType is an opaque compound datatype built from CompoundFields,
// ready for use with CreateCompoundDatasetFromFields.
type CompoundType struct {
	msg  *core.DatatypeMessage
	size uint32
}

// Size returns the total size in bytes of one compound element (the sum of
// all member sizes; members are packed contiguously in declaration order).
func (ct *CompoundType) Size() uint32 {
	return ct.size
}

// compoundFieldSize returns the on-disk size in bytes of a compound field.
func compoundFieldSize(f CompoundField) (uint32, error) {
	switch f.Type {
	case Int8, Uint8:
		return 1, nil
	case Int16, Uint16:
		return 2, nil
	case Int32, Uint32, Float32:
		return 4, nil
	case Int64, Uint64, Float64:
		return 8, nil
	case String:
		if f.StringSize == 0 {
			return 0, fmt.Errorf("field %q: String fields require StringSize > 0", f.Name)
		}
		return f.StringSize, nil
	default:
		return 0, fmt.Errorf("field %q: unsupported compound field type %d (only Int8..Float64 and String are supported)", f.Name, f.Type)
	}
}

// encodeCompoundMemberType encodes the member datatype message (header +
// properties) for a compound field.
//
// Numeric members reuse the standard encoder. String members are encoded as
// a bare 8-byte header (null-terminated ASCII, no property bytes) exactly as
// the C library writes them inside compound types — extra property bytes
// would desynchronize the C decoder for any member that follows.
func encodeCompoundMemberType(f CompoundField, size uint32) ([]byte, error) {
	if f.Type == String {
		buf := make([]byte, 8)
		// Class (bits 0-3) | version (bits 4-7) | class bit field (bits 8-31).
		classAndVersion := uint32(core.DatatypeString) | 1<<4
		binary.LittleEndian.PutUint32(buf[0:4], classAndVersion)
		binary.LittleEndian.PutUint32(buf[4:8], size)
		return buf, nil
	}

	info, err := getDatatypeInfo(f.Type, &datasetConfig{})
	if err != nil {
		return nil, fmt.Errorf("field %q: %w", f.Name, err)
	}
	return core.EncodeDatatypeMessage(&core.DatatypeMessage{
		Class:         info.class,
		Version:       1,
		Size:          info.size,
		ClassBitField: info.classBitField,
	})
}

// NewCompoundType builds a compound datatype from the given fields. Member
// offsets are assigned contiguously in declaration order; the total size is
// the sum of the member sizes.
//
// The compound is encoded as a version-1 datatype message (member names
// padded to 8-byte boundaries, member count in the class bit field), the
// layout the HDF5 C library both writes and reads.
func NewCompoundType(fields ...CompoundField) (*CompoundType, error) {
	if len(fields) == 0 {
		return nil, fmt.Errorf("compound type requires at least one field")
	}

	le := binary.LittleEndian
	seen := make(map[string]bool, len(fields))
	var props []byte
	var totalSize uint32
	var numMembers uint32

	for _, f := range fields {
		if f.Name == "" {
			return nil, fmt.Errorf("compound field name cannot be empty")
		}
		if seen[f.Name] {
			return nil, fmt.Errorf("duplicate compound field name %q", f.Name)
		}
		seen[f.Name] = true

		size, err := compoundFieldSize(f)
		if err != nil {
			return nil, err
		}

		memberType, err := encodeCompoundMemberType(f, size)
		if err != nil {
			return nil, err
		}

		// Version-1 member layout (H5Odtype.c):
		// name (null-terminated, padded to 8-byte boundary), byte offset
		// (uint32), array info (28 bytes, zeroed for scalar members),
		// member datatype message.
		padded := make([]byte, ((len(f.Name)+8)/8)*8)
		copy(padded, f.Name)
		props = append(props, padded...)
		props = le.AppendUint32(props, totalSize)
		props = append(props, make([]byte, 28)...)
		props = append(props, memberType...)

		totalSize += size
		numMembers++
	}

	return &CompoundType{
		msg: &core.DatatypeMessage{
			Class:         core.DatatypeCompound,
			Version:       1,
			Size:          totalSize,
			ClassBitField: numMembers, // Bits 0-15: number of members.
			Properties:    props,
		},
		size: totalSize,
	}, nil
}

// CreateCompoundDatasetFromFields creates a compound dataset from a plain
// field list, without requiring any internal types. It is a thin wrapper
// around NewCompoundType and CreateCompoundDataset.
//
// Example:
//
//	ds, err := fw.CreateCompoundDatasetFromFields("/data", []hdf5.CompoundField{
//	    {Name: "temp", Type: hdf5.Float64},
//	    {Name: "id", Type: hdf5.Int32},
//	    {Name: "name", Type: hdf5.String, StringSize: 8},
//	}, []uint64{100})
//
// Rows are written with WriteRaw as packed little-endian bytes: each element
// is the concatenation of its field values at contiguous offsets.
func (fw *FileWriter) CreateCompoundDatasetFromFields(name string, fields []CompoundField, dims []uint64, opts ...DatasetOption) (*DatasetWriter, error) {
	ct, err := NewCompoundType(fields...)
	if err != nil {
		return nil, fmt.Errorf("failed to build compound type: %w", err)
	}
	return fw.CreateCompoundDataset(name, ct.msg, dims, opts...)
}
