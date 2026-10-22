package writer

import (
	"encoding/binary"

	"github.com/shyrmapp/hdf5/internal/core"
)

// Fletcher32Filter implements Fletcher32 checksum (FilterID = 3).
//
// The Fletcher32 filter adds a 4-byte checksum to the end of data to detect
// corruption during storage or transmission. It uses the Fletcher32 algorithm,
// which is faster than CRC32 but less robust against intentional tampering.
//
// The filter is commonly used in HDF5 to ensure data integrity, especially
// for compressed data where corruption could affect decompression.
//
// On write: checksum is calculated and appended (original_data + 4 bytes).
// On read: checksum is verified and stripped (returns original_data).
type Fletcher32Filter struct{}

// NewFletcher32Filter creates a Fletcher32 checksum filter.
func NewFletcher32Filter() *Fletcher32Filter {
	return &Fletcher32Filter{}
}

// ID returns the HDF5 filter identifier for Fletcher32.
func (f *Fletcher32Filter) ID() FilterID {
	return FilterFletcher32
}

// Name returns the HDF5 filter name.
func (f *Fletcher32Filter) Name() string {
	return "fletcher32"
}

// Apply calculates Fletcher32 checksum and appends it to the data.
//
// The returned data is 4 bytes longer than the input, with the checksum
// stored in little-endian format at the end.
func (f *Fletcher32Filter) Apply(data []byte) ([]byte, error) {
	checksum := core.Fletcher32(data)

	// Append 4-byte checksum (little-endian)
	result := make([]byte, len(data)+4)
	copy(result, data)
	binary.LittleEndian.PutUint32(result[len(data):], checksum)

	return result, nil
}

// Encode returns the filter parameters for the Pipeline message.
//
// Fletcher32 has no parameters, so this returns empty values.
func (f *Fletcher32Filter) Encode() (flags uint16, cdValues []uint32) {
	return 0, []uint32{}
}
