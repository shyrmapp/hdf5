package core

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestParseLayoutV4 tests version 4 layout parsing (delegates to v3).
func TestParseLayoutV4(t *testing.T) {
	sb := &Superblock{
		OffsetSize: 8,
		LengthSize: 8,
		Endianness: binary.LittleEndian,
	}

	// Test contiguous layout v4.
	data := make([]byte, 2+8+8) // version + class + address + size
	data[0] = 4                 // version 4
	data[1] = byte(LayoutContiguous)
	binary.LittleEndian.PutUint64(data[2:10], 0x1000)  // data address
	binary.LittleEndian.PutUint64(data[10:18], 0x4000) // data size

	msg, err := ParseDataLayoutMessage(data, sb)
	require.NoError(t, err)
	require.NotNil(t, msg)
	require.Equal(t, uint8(4), msg.Version)
	require.Equal(t, LayoutContiguous, msg.Class)
	require.Equal(t, uint64(0x1000), msg.DataAddress)
	require.Equal(t, uint64(0x4000), msg.DataSize)
}
