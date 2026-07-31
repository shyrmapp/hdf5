package writer

import (
	"testing"
)

// TestLZFFilter_Basic tests basic LZF compression and decompression.
func TestLZFFilter_Metadata(t *testing.T) {
	filter := NewLZFFilter()

	if filter.ID() != FilterLZF {
		t.Errorf("ID() = %d, want %d", filter.ID(), FilterLZF)
	}

	if filter.Name() != "lzf" {
		t.Errorf("Name() = %q, want %q", filter.Name(), "lzf")
	}

	flags, cdValues := filter.Encode()
	if flags != 0 {
		t.Errorf("Encode() flags = %d, want 0", flags)
	}
	if len(cdValues) != 3 {
		t.Errorf("Encode() cd_values length = %d, want 3", len(cdValues))
	}
	// Expected: [0, 0, 0] (revision, version, chunk_size).
	for i, v := range cdValues {
		if v != 0 {
			t.Errorf("Encode() cd_values[%d] = %d, want 0", i, v)
		}
	}
}
