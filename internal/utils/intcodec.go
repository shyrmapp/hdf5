package utils

import "encoding/binary"

// ReadUint decodes an unsigned integer of size bytes (1..8, odd widths included)
// from data in the given byte order. size is clamped to len(data) and to 8.
func ReadUint(data []byte, size int, order binary.ByteOrder) uint64 {
	size = min(size, len(data), 8)
	var v uint64
	for i := 0; i < size; i++ {
		if order == binary.BigEndian {
			v = v<<8 | uint64(data[i])
		} else {
			v |= uint64(data[i]) << (8 * uint(i))
		}
	}
	return v
}

// PutUint encodes the low size bytes (1..8, odd widths included) of v into buf
// in the given byte order. size is clamped to len(buf) and to 8.
func PutUint(buf []byte, v uint64, size int, order binary.ByteOrder) {
	size = min(size, len(buf), 8)
	for i := 0; i < size; i++ {
		b := byte(v >> (8 * uint(i))) //nolint:gosec // G115: intentional truncation
		if order == binary.BigEndian {
			buf[size-1-i] = b
		} else {
			buf[i] = b
		}
	}
}
