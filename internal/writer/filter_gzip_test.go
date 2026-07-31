package writer

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewGZIPFilter(t *testing.T) {
	tests := []struct {
		name          string
		inputLevel    int
		expectedLevel int
	}{
		{"valid level 1", 1, 1},
		{"valid level 6", 6, 6},
		{"valid level 9", 9, 9},
		{"invalid level 0", 0, 6},   // Should default to 6
		{"invalid level 10", 10, 6}, // Should default to 6
		{"invalid level -1", -1, 6}, // Should default to 6
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGZIPFilter(tt.inputLevel)
			require.NotNil(t, filter)
			require.Equal(t, tt.expectedLevel, filter.level)
		})
	}
}

func TestGZIPFilter_ID(t *testing.T) {
	filter := NewGZIPFilter(6)
	require.Equal(t, FilterGZIP, filter.ID())
	require.Equal(t, FilterID(1), filter.ID())
}

func TestGZIPFilter_Name(t *testing.T) {
	filter := NewGZIPFilter(6)
	require.Equal(t, "deflate", filter.Name())
}

func TestGZIPFilter_Encode(t *testing.T) {
	tests := []struct {
		name  string
		level int
	}{
		{"level 1", 1},
		{"level 6", 6},
		{"level 9", 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewGZIPFilter(tt.level)
			flags, cdValues := filter.Encode()

			require.Equal(t, uint16(0), flags)
			require.Equal(t, 1, len(cdValues))
			require.Equal(t, uint32(tt.level), cdValues[0])
		})
	}
}

func TestGZIPFilter_CompressSmallData(t *testing.T) {
	filter := NewGZIPFilter(6)
	data := []byte{1, 2, 3, 4, 5}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
	// Compressed data should have GZIP headers
	require.Greater(t, len(compressed), 10) // GZIP has minimum overhead
}

func TestGZIPFilter_CompressMediumData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create 1KB of data
	data := make([]byte, 1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
}

func TestGZIPFilter_CompressLargeData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create 100KB of data
	data := make([]byte, 100*1024)
	for i := range data {
		data[i] = byte(i % 256)
	}

	compressed, err := filter.Apply(data)
	require.NoError(t, err)
	require.NotNil(t, compressed)
	require.NotEqual(t, data, compressed)
}

func TestGZIPFilter_CompressRepetitiveData(t *testing.T) {
	filter := NewGZIPFilter(6)
	// Create highly repetitive data (should compress well)
	data := bytes.Repeat([]byte{42}, 10000)

	compressed, err := filter.Apply(data)
	require.NoError(t, err)

	// Should achieve good compression ratio (>50%)
	compressionRatio := float64(len(data)) / float64(len(compressed))
	require.Greater(t, compressionRatio, 2.0, "Expected compression ratio > 2:1 for repetitive data")
}
