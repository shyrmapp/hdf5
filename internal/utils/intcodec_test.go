package utils

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestReadPutUint(t *testing.T) {
	const v = uint64(0x0807060504030201)
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for size := 1; size <= 8; size++ {
			want := v
			if size < 8 {
				want = v & (1<<(8*uint(size)) - 1)
			}
			var ref [8]byte
			order.PutUint64(ref[:], want)
			wantBytes := ref[:size] // little endian: low bytes first
			if order == binary.BigEndian {
				wantBytes = ref[8-size:]
			}
			buf := make([]byte, size)
			PutUint(buf, v, size, order)
			if !bytes.Equal(buf, wantBytes) {
				t.Fatalf("%v size %d: bytes %x want %x", order, size, buf, wantBytes)
			}
			if got := ReadUint(buf, size, order); got != want {
				t.Fatalf("%v size %d: got %#x want %#x", order, size, got, want)
			}
		}
	}
}
