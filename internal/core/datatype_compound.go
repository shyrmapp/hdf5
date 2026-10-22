package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"
)

// CompoundMember represents a single field in a compound datatype.
type CompoundMember struct {
	Name   string           // Field name.
	Offset uint32           // Byte offset within the compound structure.
	Dims   []uint32         // Array dimensions of a v1 member with intrinsic arrayness; nil for scalars.
	Type   *DatatypeMessage // Member datatype (can be any type, including nested compound).
}

// CompoundType represents a parsed compound datatype with all its members.
type CompoundType struct {
	Size    uint32           // Total size of the compound structure in bytes.
	Members []CompoundMember // List of members/fields.
}

// maxCompoundVersion is the newest compound encoding known: HDF5 2.x writes
// versions 4 and 5 for the latest format, with the v3 member layout.
const maxCompoundVersion = 5

// ParseCompoundType parses compound datatype properties to extract member information.
// The member count is always in ClassBitField bits 0-15. Per member:
//   - v1: name padded to 8 bytes, uint32 offset, 28 bytes of array info, datatype.
//   - v2: name padded to 8 bytes, uint32 offset, datatype.
//   - v3+: unpadded name, offset in the fewest bytes that can encode the
//     compound's size, datatype.
func ParseCompoundType(dt *DatatypeMessage) (*CompoundType, error) {
	if dt.Class != DatatypeCompound {
		return nil, errors.New("not a compound datatype")
	}
	c, _, err := parseCompoundProps(dt.Version, dt.ClassBitField, dt.Size, dt.Properties, 0)
	return c, err
}

// compoundOffsetBytes is the v3 member-offset width: the fewest bytes that can
// encode the compound size (H5VM_limit_enc_size), so 255 -> 1 and 256 -> 2.
func compoundOffsetBytes(size uint32) int {
	return (bits.Len32(size)-1)/8 + 1
}

// parseCompoundProps decodes every member and returns the compound with the
// number of property bytes consumed. calculateCompoundPropsLen shares it so
// the two cannot disagree about where a nested compound ends.
func parseCompoundProps(version uint8, classBitField, size uint32, props []byte, depth int) (*CompoundType, int, error) {
	if version < 1 || version > maxCompoundVersion {
		return nil, 0, fmt.Errorf("unsupported compound datatype version: %d", version)
	}
	numMembers := classBitField & 0xFFFF
	compound := &CompoundType{Size: size}
	offset := 0

	for i := uint32(0); i < numMembers; i++ {
		m, n, err := parseCompoundMember(version, size, props[offset:], depth)
		if err != nil {
			return nil, 0, fmt.Errorf("member %d (%s): %w", i, m.Name, err)
		}
		offset += n
		compound.Members = append(compound.Members, m)
	}
	return compound, offset, nil
}

// parseCompoundMember decodes one member from the head of props and returns
// the bytes it occupies. On error the member may carry only its name.
func parseCompoundMember(version uint8, size uint32, props []byte, depth int) (CompoundMember, int, error) {
	var m CompoundMember

	end := bytes.IndexByte(props, 0)
	if end < 0 {
		return m, 0, errors.New("name not null-terminated")
	}
	m.Name = string(props[:end])
	offset := end + 1
	if version < 3 {
		offset = ((end + 8) / 8) * 8 // padded to a multiple of 8, terminator included
	}

	offBytes := 4
	if version >= 3 {
		offBytes = compoundOffsetBytes(size)
	}
	if offset+offBytes > len(props) {
		return m, 0, errors.New("offset field truncated")
	}
	for j := offBytes - 1; j >= 0; j-- {
		m.Offset = m.Offset<<8 | uint32(props[offset+j])
	}
	offset += offBytes

	if version == 1 {
		if offset+28 > len(props) {
			return m, 0, errors.New("array info truncated")
		}
		// Dimensionality, 3 reserved, 4 permutation, 4 reserved, 4 x uint32 sizes.
		ndims := int(props[offset])
		if ndims > 4 {
			return m, 0, fmt.Errorf("%d array dimensions (max 4)", ndims)
		}
		for d := 0; d < ndims; d++ {
			m.Dims = append(m.Dims, binary.LittleEndian.Uint32(props[offset+12+4*d:]))
		}
		offset += 28
	}

	if offset+8 > len(props) {
		return m, 0, fmt.Errorf("datatype header truncated (need 8 bytes, have %d)", len(props)-offset)
	}
	mt, err := parseDatatypeMessage(props[offset:], depth+1)
	if err != nil {
		return m, 0, fmt.Errorf("failed to parse datatype: %w", err)
	}
	m.Type = mt
	return m, offset + 8 + len(mt.Properties), nil
}
