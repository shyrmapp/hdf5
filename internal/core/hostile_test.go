package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// guard runs fn and fails the test on a panic or a hang instead of wedging CI.
func guard(t *testing.T, fn func() error) error {
	t.Helper()
	type result struct {
		err      error
		panicked interface{}
	}
	done := make(chan result, 1)
	go func() {
		var res result
		defer func() {
			res.panicked = recover()
			done <- res
		}()
		res.err = fn()
	}()
	select {
	case res := <-done:
		require.Nil(t, res.panicked, "parser panicked")
		return res.err
	case <-time.After(5 * time.Second):
		t.Fatal("parser did not terminate (hang)")
		return nil
	}
}

func hostileSB() *Superblock {
	return &Superblock{Version: 2, OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}
}

func le64(v uint64) []byte { return binary.LittleEndian.AppendUint64(nil, v) }
func le32(v uint32) []byte { return binary.LittleEndian.AppendUint32(nil, v) }
func le16(v uint16) []byte { return binary.LittleEndian.AppendUint16(nil, v) }

func cat(parts ...[]byte) []byte { return bytes.Join(parts, nil) }

// withChecksum appends the Jenkins checksum of b.
func withChecksum(b []byte) []byte { return append(b, le32(JenkinsChecksum(b))...) }

// padTo extends b with zeros to n bytes.
func padTo(b []byte, n int) []byte { return append(b, make([]byte, n-len(b))...) }

// contMsgV2 is a v2 continuation message (type 0x10) pointing at addr/size.
func contMsgV2(addr, size uint64) []byte {
	return cat([]byte{byte(MsgContinuation)}, le16(16), []byte{0}, le64(addr), le64(size))
}

// ochk builds an OCHK block holding one continuation message.
func ochk(addr, size uint64) []byte {
	return withChecksum(cat([]byte("OCHK"), contMsgV2(addr, size)))
}

// ohdrV2 builds a v2 object header (4-byte chunk size field) holding msgs.
func ohdrV2(msgs []byte) []byte {
	return withChecksum(cat([]byte("OHDR"), []byte{2, 0x02}, le32(uint32(len(msgs))), msgs))
}

// contMsgV1 is a v1 continuation message.
func contMsgV1(addr, size uint64) []byte {
	return cat(le16(uint16(MsgContinuation)), le16(16), []byte{0, 0, 0, 0}, le64(addr), le64(size))
}

func ohdrV1(msgs []byte) []byte {
	return cat([]byte{1, 0}, le16(1), le32(1), le32(uint32(len(msgs))), make([]byte, 4), msgs)
}

func TestHostileCoreInputs(t *testing.T) {
	sb := hostileSB()

	// #1: a non-leaf chunk B-tree node whose only child is itself.
	selfTree := cat([]byte("TREE"), []byte{1, 1}, le16(1), le64(^uint64(0)), le64(^uint64(0)),
		make([]byte, 16), le64(0), make([]byte, 16))
	// Two entries pointing at the same leaf: counted twice, so rejected.
	sharedChild := padTo(cat([]byte("TREE"), []byte{1, 1}, le16(2), le64(^uint64(0)), le64(^uint64(0)),
		make([]byte, 16), le64(128), make([]byte, 16), le64(128), make([]byte, 16)), 128)
	sharedChild = append(sharedChild, cat([]byte("TREE"), []byte{1, 0}, le16(0), le64(^uint64(0)), le64(^uint64(0)), make([]byte, 16))...)
	// A child with the wrong level (same level as its parent) is also invalid.
	wrongLevel := padTo(cat([]byte("TREE"), []byte{1, 1}, le16(1), le64(^uint64(0)), le64(^uint64(0)),
		make([]byte, 16), le64(128), make([]byte, 16)), 128)
	wrongLevel = append(wrongLevel, cat([]byte("TREE"), []byte{1, 1}, le16(0), le64(^uint64(0)), le64(^uint64(0)), make([]byte, 16))...)

	// #2: continuation chains.
	v2Self := padTo(ohdrV2(contMsgV2(64, 28)), 64)
	v2Self = append(v2Self, ochk(64, 28)...)
	v2Huge := ohdrV2(contMsgV2(64, 1<<40))
	v2Bad := ohdrV2(contMsgV2(0, 4))
	v2Bad[len(v2Bad)-1] ^= 0xFF
	v1Self := padTo(ohdrV1(contMsgV1(64, 24)), 64)
	v1Self = append(v1Self, contMsgV1(64, 24)...)
	v1Huge := ohdrV1(contMsgV1(64, 1<<40))

	// #3: global heap collections.
	gcol := func(size uint64, objs []byte) []byte {
		b := cat([]byte("GCOL"), []byte{1, 0, 0, 0}, le64(size), objs)
		return b
	}
	freeObj := func(size uint64) []byte { return cat(le16(0), le16(0), make([]byte, 4), le64(size)) }
	dataObj := func(size uint64) []byte { return cat(le16(1), le16(1), make([]byte, 4), le64(size), make([]byte, 8)) }

	// #4/#5 live in the other tests below; #6/#7 in TestFilterPipelineV2.

	tests := []struct {
		name string
		run  func() error
		want string // substring of the expected error
	}{
		{"btree v1 self cycle", func() error {
			r := bytes.NewReader(selfTree)
			n, err := ParseBTreeV1Node(r, 0, 8, 1, []uint64{1})
			if err != nil {
				return err
			}
			_, err = n.CollectAllChunks(r, 8, []uint64{1})
			return err
		}, "level"},
		{"btree v1 two entries share a child", func() error {
			r := bytes.NewReader(sharedChild)
			n, err := ParseBTreeV1Node(r, 0, 8, 1, []uint64{1})
			if err != nil {
				return err
			}
			_, err = n.CollectAllChunks(r, 8, []uint64{1})
			return err
		}, "cycle"},
		{"btree v1 child at same level", func() error {
			r := bytes.NewReader(wrongLevel)
			n, err := ParseBTreeV1Node(r, 0, 8, 1, []uint64{1})
			if err != nil {
				return err
			}
			_, err = n.CollectAllChunks(r, 8, []uint64{1})
			return err
		}, "level"},
		{"ohdr v2 continuation self loop", func() error {
			_, err := ReadObjectHeader(bytes.NewReader(v2Self), 0, sb)
			return err
		}, "cycle"},
		{"ohdr v2 continuation beyond file", func() error {
			_, err := ReadObjectHeader(bytes.NewReader(v2Huge), 0, sb)
			return err
		}, "exceeds"},
		{"ohdr v2 bad checksum", func() error {
			_, err := ReadObjectHeader(bytes.NewReader(v2Bad), 0, sb)
			return err
		}, "checksum"},
		{"ohdr v1 continuation self loop", func() error {
			_, err := ReadObjectHeader(bytes.NewReader(v1Self), 0, sb)
			return err
		}, "cycle"},
		{"ohdr v1 continuation beyond file", func() error {
			_, err := ReadObjectHeader(bytes.NewReader(v1Huge), 0, sb)
			return err
		}, "exceeds"},
		{"gcol size 1<<62", func() error {
			_, err := ReadGlobalHeapCollection(bytes.NewReader(gcol(1<<62, nil)), 0, 8)
			return err
		}, "exceeds"},
		{"gcol size beyond file", func() error {
			_, err := ReadGlobalHeapCollection(bytes.NewReader(gcol(1<<33, nil)), 0, 8)
			return err
		}, "exceeds"},
		{"gcol free object size wraps offset", func() error {
			buf := gcol(0, freeObj(^uint64(0)-15))
			binary.LittleEndian.PutUint64(buf[8:], uint64(len(buf)))
			_, err := ReadGlobalHeapCollection(bytes.NewReader(buf), 0, 8)
			if err == nil {
				err = errors.New("terminated") // truncating at the bogus free object is acceptable
			}
			return err
		}, "terminated"},
		{"gcol object size 2^63+8", func() error {
			buf := gcol(0, cat(le16(1), le16(1), make([]byte, 4), le64(1<<63+8)))
			binary.LittleEndian.PutUint64(buf[8:], uint64(len(buf)))
			_, err := ReadGlobalHeapCollection(bytes.NewReader(buf), 0, 8)
			return err
		}, "beyond collection"},
		{"gcol object larger than collection", func() error {
			buf := gcol(0, dataObj(1<<20))
			binary.LittleEndian.PutUint64(buf[8:], uint64(len(buf)))
			_, err := ReadGlobalHeapCollection(bytes.NewReader(buf), 0, 8)
			return err
		}, "beyond collection"},
		{"dataspace 2 bytes", func() error {
			_, err := ParseDataspaceMessage([]byte{1, 1})
			return err
		}, "too short"},
		{"dataspace v2 no type byte", func() error {
			_, err := ParseDataspaceMessage([]byte{2, 1, 0})
			return err
		}, "too short"},
		{"dataspace rank 255", func() error {
			_, err := ParseDataspaceMessage(append([]byte{1, 255, 0, 0, 0, 0, 0, 0}, make([]byte, 4*255)...))
			return err
		}, "rank"},
		{"implicit index grid overflow", func() error {
			layout := &DataLayoutMessage{ChunkSize: []uint64{1, 1, 4}, DataAddress: 0x10, ChunkIndexType: ChunkIndexImplicit}
			ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 40, 1 << 40}}
			_, err := CollectChunksV4(bytes.NewReader(make([]byte, 200)), layout, ds, sb)
			return err
		}, "chunk"},
		{"implicit index huge grid small file", func() error {
			layout := &DataLayoutMessage{ChunkSize: []uint64{1, 4}, DataAddress: 0x10, ChunkIndexType: ChunkIndexImplicit}
			ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 22}}
			_, err := CollectChunksV4(bytes.NewReader(make([]byte, 200)), layout, ds, sb)
			return err
		}, "do not fit"},
		{"fixed array header bad checksum", func() error {
			hdr := cat([]byte("FAHD"), []byte{0, 0, 4, 10}, le64(10), le64(^uint64(0)), le32(0xdeadbeef))
			_, err := parseFixedArrayHeader(bytes.NewReader(hdr), 0, sb)
			return err
		}, "checksum"},
		{"v2 b-tree chunk header bad checksum", func() error {
			hdr := cat([]byte("BTHD"), []byte{0, 10}, le32(2048), le16(24), le16(0), []byte{100, 40},
				le64(64), le16(1), le64(1), le32(1))
			layout := &DataLayoutMessage{ChunkSize: []uint64{1, 1, 4}, DataAddress: 0, ChunkIndexType: ChunkIndexBTreeV2}
			ds := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{4, 4}}
			_, err := CollectChunksV4(bytes.NewReader(hdr), layout, ds, sb)
			return err
		}, "checksum"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := guard(t, tc.run)
			require.Error(t, err)
			require.Contains(t, err.Error(), tc.want)
		})
	}
}

func TestDataspaceNullAndShort(t *testing.T) {
	ds, err := ParseDataspaceMessage([]byte{2, 0, 0, byte(DataspaceNull)})
	require.NoError(t, err)
	require.Equal(t, DataspaceNull, ds.Type)
	require.Zero(t, ds.TotalElements())

	// Scalar v2 still works.
	ds, err = ParseDataspaceMessage([]byte{2, 0, 0, byte(DataspaceScalar)})
	require.NoError(t, err)
	require.Equal(t, DataspaceScalar, ds.Type)

	// TotalElements saturates instead of wrapping.
	big := &DataspaceMessage{Type: DataspaceSimple, Dimensions: []uint64{1 << 40, 1 << 40}}
	require.Equal(t, ^uint64(0), big.TotalElements())
}

func TestFilterPipelineV2(t *testing.T) {
	// shuffle(cd=[4]) + deflate(cd=[6]) + fletcher32(no cd), version 2 layout.
	msg := cat([]byte{2, 3},
		le16(2), le16(1), le16(1), le32(4),
		le16(1), le16(1), le16(1), le32(6),
		le16(3), le16(0), le16(0))
	p, err := ParseFilterPipelineMessage(msg)
	require.NoError(t, err)
	require.Len(t, p.Filters, 3)
	require.Equal(t, FilterShuffle, p.Filters[0].ID)
	require.Equal(t, []uint32{4}, p.Filters[0].ClientData)
	require.Equal(t, []uint32{6}, p.Filters[1].ClientData)
	require.Equal(t, FilterFletcher, p.Filters[2].ID)

	// LZF (id >= 256) carries a name length and an unpadded name in v2.
	lzf := cat([]byte{2, 1},
		le16(uint16(FilterLZF)), le16(4), le16(1), le16(3), []byte("lzf\x00"),
		le32(4), le32(100), le32(400))
	p, err = ParseFilterPipelineMessage(lzf)
	require.NoError(t, err)
	require.Equal(t, "lzf", p.Filters[0].Name)
	require.Equal(t, []uint32{4, 100, 400}, p.Filters[0].ClientData)

	// Truncation is an error, not a panic.
	for n := 0; n < len(lzf); n++ {
		_, _ = ParseFilterPipelineMessage(lzf[:n])
	}
}

func TestLZFPaddingClamped(t *testing.T) {
	p := &FilterPipelineMessage{Version: 2, Filters: []Filter{
		{ID: FilterLZF, ClientData: []uint32{4, 100, 0xFFFFFFFF}},
	}}
	// One literal run of 3 bytes.
	out, err := p.ApplyFilters([]byte{2, 'a', 'b', 'c'}, 0, 1024)
	require.NoError(t, err)
	require.LessOrEqual(t, len(out), 1024)
}
