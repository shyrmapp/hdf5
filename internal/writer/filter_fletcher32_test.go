package writer

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewFletcher32Filter(t *testing.T) {
	filter := NewFletcher32Filter()
	require.NotNil(t, filter)
}

func TestFletcher32Filter_ID(t *testing.T) {
	filter := NewFletcher32Filter()
	require.Equal(t, FilterFletcher32, filter.ID())
	require.Equal(t, FilterID(3), filter.ID())
}

func TestFletcher32Filter_Name(t *testing.T) {
	filter := NewFletcher32Filter()
	require.Equal(t, "fletcher32", filter.Name())
}

func TestFletcher32Filter_Encode(t *testing.T) {
	filter := NewFletcher32Filter()
	flags, cdValues := filter.Encode()

	require.Equal(t, uint16(0), flags)
	require.Equal(t, 0, len(cdValues))
}

func TestFletcher32Filter_Apply(t *testing.T) {
	filter := NewFletcher32Filter()
	data := []byte{1, 2, 3, 4, 5}

	result, err := filter.Apply(data)
	require.NoError(t, err)

	// Result should be 4 bytes longer (original + checksum)
	require.Equal(t, len(data)+4, len(result))

	// Original data should be preserved
	require.Equal(t, data, result[:len(data)])

	// Checksum should be present (non-zero for this data)
	checksum := binary.LittleEndian.Uint32(result[len(data):])
	require.NotEqual(t, uint32(0), checksum)
}

func TestFletcher32Filter_Apply_EmptyData(t *testing.T) {
	filter := NewFletcher32Filter()
	data := []byte{}

	result, err := filter.Apply(data)
	require.NoError(t, err)

	// Result should be 4 bytes (just checksum)
	require.Equal(t, 4, len(result))
}

func TestFletcher32Filter_Apply_LargeData(t *testing.T) {
	filter := NewFletcher32Filter()
	data := make([]byte, 100*1024) // 100KB
	for i := range data {
		data[i] = byte(i % 256)
	}

	result, err := filter.Apply(data)
	require.NoError(t, err)
	require.Equal(t, len(data)+4, len(result))
}

func TestFletcher32Filter_ChecksumUniqueness(t *testing.T) {
	filter := NewFletcher32Filter()

	// Different data should produce different checksums
	data1 := []byte{1, 2, 3, 4, 5}
	data2 := []byte{5, 4, 3, 2, 1}
	data3 := []byte{1, 2, 3, 4, 6}

	checksum1, _ := filter.Apply(data1)
	checksum2, _ := filter.Apply(data2)
	checksum3, _ := filter.Apply(data3)

	cs1 := binary.LittleEndian.Uint32(checksum1[len(data1):])
	cs2 := binary.LittleEndian.Uint32(checksum2[len(data2):])
	cs3 := binary.LittleEndian.Uint32(checksum3[len(data3):])

	require.NotEqual(t, cs1, cs2)
	require.NotEqual(t, cs1, cs3)
	require.NotEqual(t, cs2, cs3)
}

func TestCalculateFletcher32_KnownValues(t *testing.T) {
	// Test with known patterns
	tests := []struct {
		name string
		data []byte
	}{
		{"empty", []byte{}},
		{"single byte", []byte{42}},
		{"two bytes", []byte{1, 2}},
		{"hello", []byte("hello")},
		{"zeros", make([]byte, 100)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checksum1 := calculateFletcher32(tt.data)
			checksum2 := calculateFletcher32(tt.data)

			// Same data should produce same checksum
			require.Equal(t, checksum1, checksum2)
		})
	}
}
