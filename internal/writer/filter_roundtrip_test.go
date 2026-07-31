package writer

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// decodeWithReader runs data back through the real read path: it encodes the
// pipeline's own HDF5 Pipeline message, parses it with core.ParseFilterPipelineMessage,
// and applies the reversal core actually uses when reading a file.
//
// This is deliberately not a writer-internal inverse. The writer has no Remove();
// the only decoder that matters is the one core uses on real files, so that is
// what the write path is checked against.
func decodeWithReader(t *testing.T, fp *FilterPipeline, data []byte) []byte {
	t.Helper()

	msg, err := fp.EncodePipelineMessage()
	require.NoError(t, err)

	parsed, err := core.ParseFilterPipelineMessage(msg)
	require.NoError(t, err)

	out, err := parsed.ApplyFilters(data, 0)
	require.NoError(t, err)
	return out
}

func TestFilterRoundTripAgainstReader(t *testing.T) {
	// Float64 ramp: compresses well and exercises shuffle's element striding.
	original := make([]byte, 8*512)
	for i := range 512 {
		binary.LittleEndian.PutUint64(original[i*8:], math.Float64bits(float64(i)*0.25))
	}

	tests := []struct {
		name  string
		build func() *FilterPipeline
	}{
		{"gzip", func() *FilterPipeline {
			fp := NewFilterPipeline()
			fp.AddFilter(NewGZIPFilter(6))
			return fp
		}},
		{"lzf", func() *FilterPipeline {
			fp := NewFilterPipeline()
			fp.AddFilter(NewLZFFilter())
			return fp
		}},
		{"shuffle+gzip", func() *FilterPipeline {
			fp := NewFilterPipeline()
			fp.AddFilter(NewShuffleFilter(8))
			fp.AddFilter(NewGZIPFilter(6))
			return fp
		}},
		{"shuffle+gzip+fletcher32", func() *FilterPipeline {
			fp := NewFilterPipeline()
			fp.AddFilter(NewShuffleFilter(8))
			fp.AddFilter(NewGZIPFilter(6))
			fp.AddFilter(NewFletcher32Filter())
			return fp
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := tt.build()

			encoded, err := fp.Apply(original)
			require.NoError(t, err)

			require.Equal(t, original, decodeWithReader(t, fp, encoded))
		})
	}
}

func TestGZIPCompressionLevelsRoundTrip(t *testing.T) {
	original := make([]byte, 4096) // all zeros: maximally compressible
	for level := 1; level <= 9; level++ {
		fp := NewFilterPipeline()
		fp.AddFilter(NewGZIPFilter(level))

		encoded, err := fp.Apply(original)
		require.NoError(t, err)
		require.Less(t, len(encoded), len(original), "level %d should compress", level)
		require.Equal(t, original, decodeWithReader(t, fp, encoded))
	}
}

// TestLZFCompressPatterns covers the three LZF segment kinds (literal runs,
// short backrefs, long backrefs) that lzfCompress emits.
func TestLZFCompressPatterns(t *testing.T) {
	longRun := make([]byte, 1024) // one very long backref
	incompressible := make([]byte, 256)
	for i := range incompressible {
		incompressible[i] = byte(i) // literal runs, no repeats within window
	}
	shortRepeats := []byte("abcabcabcabcabcabcabcabc") // short backrefs

	for name, data := range map[string][]byte{
		"long-backref":   longRun,
		"literal-runs":   incompressible,
		"short-backrefs": shortRepeats,
		"empty":          {},
		"single-byte":    {0x42},
	} {
		t.Run(name, func(t *testing.T) {
			f := NewLZFFilter()
			encoded, err := f.Apply(data)
			require.NoError(t, err)

			fp := NewFilterPipeline()
			fp.AddFilter(f)
			if len(data) == 0 {
				require.Empty(t, encoded)
				return
			}
			require.Equal(t, data, decodeWithReader(t, fp, encoded))
		})
	}
}

func TestFilterPipelineAdd(t *testing.T) {
	fp := NewFilterPipeline()
	require.True(t, fp.IsEmpty())

	fp.AddFilter(NewGZIPFilter(6))
	require.False(t, fp.IsEmpty())

	// AddFilterAtStart must put shuffle ahead of gzip, or the pipeline message
	// declares the wrong order and readers decode garbage.
	fp.AddFilterAtStart(NewShuffleFilter(4))
	require.Equal(t, FilterShuffle, fp.filters[0].ID())
	require.Equal(t, FilterGZIP, fp.filters[1].ID())
}
