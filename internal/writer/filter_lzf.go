package writer

import (
	"fmt"
)

// HDF5 filter label for LZF. Extracted as a constant so goconst does not flag
// the duplicate string across source + helper tests.
const filterLZFName = "lzf"

// LZFFilter implements LZF compression (FilterID = 32000).
// LZF is a very fast compression algorithm designed by Marc Lehmann.
// It provides ~40-50% compression with 3-5x faster compression than GZIP
// and 2x faster decompression.
//
// This filter is commonly used by PyTables and h5py for fast compression.
// Filter ID 32000 was registered by Francesc Alted (PyTables maintainer).
//
// Reference: http://oldhome.schmorp.de/marc/liblzf.html
// HDF5 Registration: https://portal.hdfgroup.org/display/support/Filters
type LZFFilter struct {
	// LZF has no configuration parameters
}

// NewLZFFilter creates an LZF compression filter.
// LZF has no configuration parameters - it uses a fixed algorithm.
func NewLZFFilter() *LZFFilter {
	return &LZFFilter{}
}

// ID returns the HDF5 filter identifier for LZF.
func (f *LZFFilter) ID() FilterID {
	return FilterLZF
}

// Name returns the HDF5 filter name.
func (f *LZFFilter) Name() string {
	return filterLZFName
}

// Apply compresses data using LZF algorithm.
// Returns compressed data suitable for storage.
//
// LZF algorithm characteristics:
//   - Hash-based pattern matching (LZ77 family)
//   - 8KB sliding window
//   - Very fast compression (near memcpy speed)
//   - Typical compression ratio: 40-50%
func (f *LZFFilter) Apply(data []byte) ([]byte, error) {
	if len(data) == 0 {
		return data, nil
	}

	compressed, err := lzfCompress(data)
	if err != nil {
		return nil, fmt.Errorf("lzf compression failed: %w", err)
	}

	return compressed, nil
}

// Encode returns the filter parameters for the Pipeline message.
//
// For LZF in HDF5, the client data typically contains:
//   - cd_values[0]: Plugin revision number (usually 0)
//   - cd_values[1]: LZF filter version (usually 0)
//   - cd_values[2]: Pre-computed chunk size (0 = not pre-computed)
//
// For this implementation, we use minimal parameters.
func (f *LZFFilter) Encode() (flags uint16, cdValues []uint32) {
	return 0, []uint32{0, 0, 0} // Revision 0, Version 0, No pre-computed size
}

// lzfCompress compresses data using the LZF algorithm.
//
// LZF compressed format consists of segments:
//   - Literal run (000LLLLL): L+1 bytes of uncompressed data
//   - Short backref (RRROXXXX XXXXXXXX): 3-8 bytes from offset 1-8192
//   - Long backref (111OXXXX XXXXXXXX RRRRRRRR): 9-264 bytes from offset 1-8192
//
// Reference: https://gist.github.com/joshuawscott/758b407adfc141310b3993401cc41f2e
//
//nolint:gocognit,gocyclo,cyclop,nestif,unparam // LZF compression algorithm is inherently complex; error kept for API consistency
func lzfCompress(input []byte) ([]byte, error) {
	inLen := len(input)
	if inLen == 0 {
		return input, nil
	}

	// Allocate worst-case output buffer (data + overhead for all literals).
	// Worst case: every byte becomes a literal segment with 1-byte header.
	// Plus some extra for long literal runs.
	maxOut := inLen + (inLen / 32) + 256
	output := make([]byte, 0, maxOut)

	// Hash table for finding matches (2^14 = 16384 entries).
	// Maps 3-byte patterns to positions in input.
	const hlog = 14
	const hsize = 1 << hlog
	var htab [hsize]uint32

	inPos := 0
	litPos := 0 // Start of current literal run

	for inPos < inLen {
		// Need at least 3 bytes for pattern matching.
		if inPos+3 > inLen {
			break
		}

		// Calculate hash of next 3 bytes.
		hash := hashLZF(input[inPos], input[inPos+1], input[inPos+2])
		ref := int(htab[hash])
		htab[hash] = uint32(inPos)

		// Check if we have a valid match:
		// - ref must be before current position
		// - ref must be within 8KB window
		// - first 3 bytes must match
		offset := inPos - ref
		if ref > 0 && offset <= 8192 && offset > 0 &&
			input[ref] == input[inPos] &&
			input[ref+1] == input[inPos+1] &&
			input[ref+2] == input[inPos+2] {
			// Found a match! First, output any pending literal run.
			if litPos < inPos {
				output = appendLiteral(output, input[litPos:inPos])
			}

			// Extend match as far as possible.
			maxLen := inLen - inPos
			if maxLen > 264 {
				maxLen = 264 // LZF max match length
			}

			matchLen := 3
			for matchLen < maxLen && input[ref+matchLen] == input[inPos+matchLen] {
				matchLen++
			}

			// Encode backreference.
			output = appendBackref(output, offset, matchLen)

			inPos += matchLen
			litPos = inPos

			// Update hash table for all positions in match (except last 2).
			for i := 1; i < matchLen-2; i++ {
				pos := inPos - matchLen + i
				if pos+2 < inLen {
					h := hashLZF(input[pos], input[pos+1], input[pos+2])
					htab[h] = uint32(pos) //nolint:gosec // G115: pos < len(input), fits in uint32
				}
			}
		} else {
			// No match, continue scanning.
			inPos++
		}
	}

	// Output any remaining literal bytes.
	if litPos < inLen {
		output = appendLiteral(output, input[litPos:])
	}

	return output, nil
}

// hashLZF computes a hash value for 3 bytes.
// This hash function is designed to distribute 3-byte patterns across the hash table.
func hashLZF(b0, b1, b2 byte) uint32 {
	// LZF standard hash function.
	v := (uint32(b0) << 16) | (uint32(b1) << 8) | uint32(b2)
	v ^= v >> 16
	v *= 0x45d9f3b // Prime multiplier
	v ^= v >> 16
	return v & 0x3fff // Mask to 14 bits (16384 entries)
}

// appendLiteral appends a literal run to the output.
// Literal format: 000LLLLL [L+1 bytes]
// For runs > 32 bytes, multiple literal segments are used.
func appendLiteral(output, literal []byte) []byte {
	for len(literal) > 0 {
		runLen := len(literal)
		if runLen > 32 {
			runLen = 32 // Max literal run is 32 bytes (5 bits = 0-31, +1 = 1-32)
		}

		// Control byte: 000LLLLL where LLLLL = runLen - 1.
		ctrl := byte(runLen - 1) //nolint:gosec // G115: runLen bounded to 1-32
		output = append(output, ctrl)
		output = append(output, literal[:runLen]...)

		literal = literal[runLen:]
	}
	return output
}

// appendBackref appends a backreference to the output.
// Short format (3-8 bytes): RRROXXXX XXXXXXXX.
// Long format (9-264 bytes): 111OXXXX XXXXXXXX RRRRRRRR.
func appendBackref(output []byte, offset, length int) []byte {
	// Offset is 1-based in encoding (offset - 1).
	offset--

	if length <= 8 {
		// Short backreference: 3-8 bytes, offset 1-8192.
		// Format: RRR OXXXX XXXXXXXX
		// R = length - 2 (000 = 3 bytes, 110 = 8 bytes)
		// O + X = offset - 1 (13 bits, max 8191)
		runBits := (length - 2) << 5          // Bits 7-5
		ctrl := byte(runBits | (offset >> 8)) //nolint:gosec // G115: LZF backreference encoding
		output = append(output, ctrl, byte(offset&0xFF))
	} else {
		// Long backreference: 9-264 bytes, offset 1-8192.
		// Format: 111 OXXXX XXXXXXXX RRRRRRRR
		// R = length - 9 (0 = 9 bytes, 255 = 264 bytes)
		// O + X = offset - 1 (13 bits, max 8191)
		ctrl := byte(0xE0 | (offset >> 8))                               //nolint:gosec // G115: LZF backreference encoding
		output = append(output, ctrl, byte(offset&0xFF), byte(length-9)) //nolint:gosec // G115: LZF backreference encoding
	}

	return output
}
