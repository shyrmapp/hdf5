package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFillValueBytes(t *testing.T) {
	v := []byte{0x19, 0xfc} // int16 -999
	hdr := func(typ MessageType, data []byte) *ObjectHeader {
		return &ObjectHeader{Messages: []*HeaderMessage{{Type: typ, Data: data}}}
	}
	cases := []struct {
		name string
		h    *ObjectHeader
		want []byte
	}{
		{"none", &ObjectHeader{}, nil},
		{"old", hdr(MsgFillValueOld, []byte{2, 0, 0, 0, 0x19, 0xfc}), v},
		{"old size 0", hdr(MsgFillValueOld, []byte{0, 0, 0, 0}), nil},
		{"v1 defined", hdr(MsgFillValue, []byte{1, 2, 2, 1, 2, 0, 0, 0, 0x19, 0xfc}), v},
		{"v1 undefined", hdr(MsgFillValue, []byte{1, 2, 2, 0}), nil},
		{"v2 defined", hdr(MsgFillValue, []byte{2, 3, 2, 1, 2, 0, 0, 0, 0x19, 0xfc}), v},
		{"v2 fill time never", hdr(MsgFillValue, []byte{2, 3, 1, 1, 2, 0, 0, 0, 0x19, 0xfc}), nil},
		{"v3 defined", hdr(MsgFillValue, []byte{3, 0x20 | 0x08, 2, 0, 0, 0, 0x19, 0xfc}), v},
		{"v3 not defined flag", hdr(MsgFillValue, []byte{3, 0x08, 2, 0, 0, 0, 0x19, 0xfc}), nil},
		{"v3 undefined flag", hdr(MsgFillValue, []byte{3, 0x30, 2, 0, 0, 0, 0x19, 0xfc}), nil},
		{"size mismatch", hdr(MsgFillValue, []byte{2, 3, 2, 1, 4, 0, 0, 0, 1, 2, 3, 4}), nil},
		{"truncated", hdr(MsgFillValue, []byte{2, 3, 2, 1, 2, 0, 0, 0, 0x19}), nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, FillValueBytes(c.h, 2))
		})
	}
}

func TestFillBuffer(t *testing.T) {
	buf := make([]byte, 7)
	fillBuffer(buf, []byte{1, 2})
	require.Equal(t, []byte{1, 2, 1, 2, 1, 2, 1}, buf)
}
