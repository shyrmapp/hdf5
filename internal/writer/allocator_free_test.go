package writer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// TASK-042: Allocator.Free() tests
// ---------------------------------------------------------------------------

func TestFree_ZeroSize(t *testing.T) {
	alloc := NewAllocator(0)
	err := alloc.Free(100, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot free zero bytes")
}

func TestFree_BestFit(t *testing.T) {
	alloc := NewAllocator(0)

	// Allocate 5 blocks of different sizes interspersed with padding.
	_, _ = alloc.Allocate(50)  // [0,50)     — keep
	_, _ = alloc.Allocate(200) // [50,250)   — will free (large gap)
	_, _ = alloc.Allocate(50)  // [250,300)  — keep
	_, _ = alloc.Allocate(100) // [300,400)  — will free (medium gap)
	_, _ = alloc.Allocate(50)  // [400,450)  — keep
	_, _ = alloc.Allocate(60)  // [450,510)  — will free (small gap)
	_, _ = alloc.Allocate(50)  // [510,560)  — keep

	// Free 3 blocks to create gaps of different sizes.
	require.NoError(t, alloc.Free(50, 200))  // 200 byte gap
	require.NoError(t, alloc.Free(300, 100)) // 100 byte gap
	require.NoError(t, alloc.Free(450, 60))  // 60 byte gap

	// Allocate 55 bytes — should pick the 60-byte gap (best fit).
	addr, err := alloc.Allocate(55)
	require.NoError(t, err)
	assert.Equal(t, uint64(450), addr, "should pick best-fit (60-byte gap for 55-byte request)")

	// Allocate 100 bytes — should pick the 100-byte gap (exact fit).
	addr, err = alloc.Allocate(100)
	require.NoError(t, err)
	assert.Equal(t, uint64(300), addr, "should pick exact-fit 100-byte gap")
}

func BenchmarkFreeAndReuse(b *testing.B) {
	alloc := NewAllocator(0)

	// Pre-allocate a pool of blocks.
	addrs := make([]uint64, 100)
	for i := range addrs {
		addr, _ := alloc.Allocate(1024)
		addrs[i] = addr
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx := i % len(addrs)
		_ = alloc.Free(addrs[idx], 1024)
		addr, _ := alloc.Allocate(1024)
		addrs[idx] = addr
	}
}
