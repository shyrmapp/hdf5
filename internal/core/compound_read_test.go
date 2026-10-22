package core

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCompoundOffsetBytes(t *testing.T) {
	for size, want := range map[uint32]int{
		1: 1, 24: 1, 255: 1, 256: 2, 65535: 2, 65536: 3, 1<<24 - 1: 3, 1 << 24: 4,
	} {
		require.Equal(t, want, compoundOffsetBytes(size), "size %d", size)
	}
}

// A v1 member with intrinsic arrayness (28 bytes of array info) is an array,
// not a scalar of its base type.
func TestCompoundV1ArrayMember(t *testing.T) {
	le := binary.LittleEndian
	props := append([]byte("m"), make([]byte, 7)...) // name padded to 8
	props = le.AppendUint32(props, 0)                // offset
	info := make([]byte, 28)
	info[0] = 2 // ndims
	le.PutUint32(info[12:], 2)
	le.PutUint32(info[16:], 2)
	props = append(props, info...)
	props = le.AppendUint32(props, uint32(DatatypeFixed)|1<<4|0x08<<8) // signed int32
	props = le.AppendUint32(props, 4)
	props = append(props, 0, 0, 32, 0)

	dt := &DatatypeMessage{Class: DatatypeCompound, Version: 1, Size: 16, ClassBitField: 1, Properties: props}
	ct, err := ParseCompoundType(dt)
	require.NoError(t, err)
	require.Equal(t, []uint32{2, 2}, ct.Members[0].Dims)

	raw := make([]byte, 16)
	for i := range 4 {
		le.PutUint32(raw[4*i:], uint32(i+1))
	}
	got, err := parseCompoundData(raw, ct, 1, nil, &Superblock{OffsetSize: 8})
	require.NoError(t, err)
	require.Equal(t, []interface{}{int32(1), int32(2), int32(3), int32(4)}, got[0]["m"])
}

func compoundReadHeader(t *testing.T, rows uint64, layout []byte) *ObjectHeader {
	t.Helper()
	// {float32}: 4 bytes per row on disk.
	dt := testEncodeCompoundV3(4, []testCompoundField{{name: "f", typ: testBasicDatatype(DatatypeFloat, 4)}})
	return &ObjectHeader{Messages: []*HeaderMessage{
		{Type: MsgDatatype, Data: dt},
		{Type: MsgDataspace, Data: binary.LittleEndian.AppendUint64([]byte{1, 1, 0, 0, 0, 0, 0, 0}, rows)}, // v1, rank 1
		{Type: MsgDataLayout, Data: layout},
	}}
}

// The decoded rows (a map each) dwarf the raw bytes, so the limit must be
// checked against the output before anything is allocated.
func TestReadDatasetCompound_LimitCountsOutput(t *testing.T) {
	const undef = ^uint64(0)
	h := compoundReadHeader(t, 256*1024, buildContiguousLayoutMessage(undef, 1<<20))
	sb := &Superblock{OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}

	// 256K rows x 4 raw bytes is exactly 1 MiB, within the old raw-byte check.
	_, err := ReadDatasetCompound(&emptyReaderAt{}, h, sb, 1<<20)
	require.ErrorContains(t, err, "compound output")
}

func TestReadDatasetCompound_CompactRowsExceedPayload(t *testing.T) {
	layout := []byte{3, byte(LayoutCompact), 8, 0, 1, 2, 3, 4, 5, 6, 7, 8}
	h := compoundReadHeader(t, 1000, layout)
	sb := &Superblock{OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}

	_, err := ReadDatasetCompound(&emptyReaderAt{}, h, sb, 0)
	require.ErrorContains(t, err, "compact data holds 8 bytes")
}
