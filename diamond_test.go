package hdf5

import (
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// diamond.h5: /g0../g20, each /gi has two hard links (a, b) to /g(i+1), so
// /g20 has 2^20 paths. Per-path loading made Open exponential.
func TestOpenHardLinkDiamond(t *testing.T) {
	type result struct {
		f     *File
		err   error
		alloc uint64
	}
	done := make(chan result, 1)
	go func() {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		f, err := Open("testdata/c_interop/diamond.h5")
		runtime.ReadMemStats(&after)
		done <- result{f, err, after.TotalAlloc - before.TotalAlloc}
	}()
	var res result
	select {
	case res = <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("Open did not finish: diamond loaded once per path")
	}
	require.NoError(t, res.err)
	defer res.f.Close()
	require.Less(t, res.alloc, uint64(32<<20), "Open allocated %d bytes", res.alloc)

	// Aliases keep listing their children under every name.
	var g *Group
	for _, c := range res.f.Root().Children() {
		if c.Name() == "g0" {
			g = c.(*Group)
		}
	}
	require.NotNil(t, g)
	require.Len(t, g.Children(), 2)
	for _, c := range g.Children() {
		require.Len(t, c.(*Group).Children(), 2, "alias %s lost its children", c.Name())
	}

	// Walk reports every group alias but descends into shared children once.
	visits := 0
	res.f.Walk(func(string, Object) { visits++ })
	require.Less(t, visits, 200)
}
