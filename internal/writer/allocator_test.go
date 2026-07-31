package writer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewAllocator(t *testing.T) {
	tests := []struct {
		name          string
		initialOffset uint64
		wantOffset    uint64
	}{
		{"zero offset", 0, 0},
		{"after superblock v2", 48, 48},
		{"custom offset", 1024, 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			alloc := NewAllocator(tt.initialOffset)
			assert.NotNil(t, alloc)
			assert.Equal(t, tt.wantOffset, alloc.EndOfFile())
			assert.Empty(t, alloc.blocks)
		})
	}
}

func TestAllocate(t *testing.T) {
	t.Run("sequential allocations", func(t *testing.T) {
		alloc := NewAllocator(48) // After superblock v2

		// First allocation
		addr1, err := alloc.Allocate(100)
		require.NoError(t, err)
		assert.Equal(t, uint64(48), addr1)
		assert.Equal(t, uint64(148), alloc.EndOfFile())

		// Second allocation (should be contiguous)
		addr2, err := alloc.Allocate(200)
		require.NoError(t, err)
		assert.Equal(t, uint64(148), addr2)
		assert.Equal(t, uint64(348), alloc.EndOfFile())

		// Third allocation
		addr3, err := alloc.Allocate(50)
		require.NoError(t, err)
		assert.Equal(t, uint64(348), addr3)
		assert.Equal(t, uint64(398), alloc.EndOfFile())
	})

	t.Run("zero size allocation fails", func(t *testing.T) {
		alloc := NewAllocator(0)

		addr, err := alloc.Allocate(0)
		assert.Error(t, err)
		assert.Equal(t, uint64(0), addr)
		assert.Contains(t, err.Error(), "cannot allocate zero bytes")
	})

	t.Run("large allocation", func(t *testing.T) {
		alloc := NewAllocator(0)

		// Allocate 10MB
		size := uint64(10 * 1024 * 1024)
		addr, err := alloc.Allocate(size)
		require.NoError(t, err)
		assert.Equal(t, uint64(0), addr)
		assert.Equal(t, size, alloc.EndOfFile())
	})
}

func TestAllocatorEndOfFile(t *testing.T) {
	tests := []struct {
		name          string
		initialOffset uint64
		allocations   []uint64
		expectedEOF   uint64
	}{
		{
			name:          "no allocations",
			initialOffset: 48,
			allocations:   []uint64{},
			expectedEOF:   48,
		},
		{
			name:          "single allocation",
			initialOffset: 48,
			allocations:   []uint64{100},
			expectedEOF:   148,
		},
		{
			name:          "multiple allocations",
			initialOffset: 48,
			allocations:   []uint64{100, 200, 50},
			expectedEOF:   398,
		},
		{
			name:          "large allocations",
			initialOffset: 0,
			allocations:   []uint64{1024, 2048, 4096},
			expectedEOF:   7168,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			alloc := NewAllocator(tt.initialOffset)

			for _, size := range tt.allocations {
				_, err := alloc.Allocate(size)
				require.NoError(t, err)
			}

			assert.Equal(t, tt.expectedEOF, alloc.EndOfFile())
		})
	}
}

// Benchmark allocation performance.
func BenchmarkAllocate(b *testing.B) {
	alloc := NewAllocator(0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = alloc.Allocate(1024)
	}
}

func TestAllocator_EdgeCases(t *testing.T) {
	t.Run("allocate size 1", func(t *testing.T) {
		alloc := NewAllocator(0)
		addr, err := alloc.Allocate(1)
		require.NoError(t, err)
		assert.Equal(t, uint64(0), addr)
		assert.Equal(t, uint64(1), alloc.EndOfFile())
	})

	t.Run("allocate max uint64 size", func(t *testing.T) {
		alloc := NewAllocator(0)

		// This might cause overflow in real implementation, but test it
		// In practice, filesystems can't handle this, but allocator should try
		size := uint64(1<<63 - 1) // Very large but not overflow-causing
		addr, err := alloc.Allocate(size)

		// Should succeed (allocator doesn't validate size limits in MVP)
		require.NoError(t, err)
		assert.Equal(t, uint64(0), addr)
	})

	t.Run("allocate from non-zero initial offset", func(t *testing.T) {
		initialOffset := uint64(12345)
		alloc := NewAllocator(initialOffset)

		addr, err := alloc.Allocate(100)
		require.NoError(t, err)
		assert.Equal(t, initialOffset, addr)
		assert.Equal(t, initialOffset+100, alloc.EndOfFile())
	})

	t.Run("many allocations preserve order", func(t *testing.T) {
		alloc := NewAllocator(0)

		// Allocate with varying sizes
		sizes := []uint64{10, 20, 5, 100, 1, 50}
		addrs := make([]uint64, len(sizes))

		for i, size := range sizes {
			addr, err := alloc.Allocate(size)
			require.NoError(t, err)
			addrs[i] = addr
		}

		// Verify addresses are in ascending order (sequential allocation)
		for i := 0; i < len(addrs)-1; i++ {
			assert.Less(t, addrs[i], addrs[i+1],
				"addresses should be sequential and ascending")
		}
	})
}

// TestAllocator_GetTotalAllocated tests total space tracking.
func TestAllocator_ConcurrentAccess(t *testing.T) {
	t.Skip("Allocator is NOT thread-safe in MVP - this is a documented limitation")

	// This test is skipped but documents the expected behavior.
	// In v0.11.0-RC, we might add thread safety and enable this test.
}

// BenchmarkAllocate_Sequential benchmarks sequential allocations.
func BenchmarkAllocate_Sequential(b *testing.B) {
	alloc := NewAllocator(0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = alloc.Allocate(1024)
	}
}

// BenchmarkBlocks benchmarks retrieving all blocks.
