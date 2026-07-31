package core

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestFilterName tests filter name conversion.
func TestFilterName(t *testing.T) {
	tests := []struct {
		name     string
		filterID FilterID
		want     string
	}{
		{
			name:     "GZIP deflate",
			filterID: FilterDeflate,
			want:     "GZIP",
		},
		{
			name:     "Shuffle filter",
			filterID: FilterShuffle,
			want:     "Shuffle",
		},
		{
			name:     "Fletcher32 checksum",
			filterID: FilterFletcher,
			want:     "Fletcher32",
		},
		{
			name:     "SZIP compression",
			filterID: FilterSZIP,
			want:     "SZIP",
		},
		{
			name:     "N-bit compression",
			filterID: FilterNBit,
			want:     "N-bit",
		},
		{
			name:     "Scale-Offset filter",
			filterID: FilterScaleOffset,
			want:     "Scale-Offset",
		},
		{
			name:     "unknown filter ID",
			filterID: 999,
			want:     "Unknown-999",
		},
		{
			name:     "unknown filter ID 0",
			filterID: 0,
			want:     "Unknown-0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterName(tt.filterID)
			require.Equal(t, tt.want, got)
		})
	}
}
