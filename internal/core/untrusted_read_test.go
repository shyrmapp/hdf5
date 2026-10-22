package core

import (
	"bytes"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
)

// Hand-built dataset headers for the typed readers. The declared element count
// and layout are what a hostile file controls, so each test states them
// directly.

func dtMsg(class byte, size uint32, props []byte) []byte {
	return cat([]byte{1<<4 | class, 0, 0, 0}, le32(size), props)
}

func spaceMsg(n uint64) []byte { return cat([]byte{1, 1, 0, 0, 0, 0, 0, 0}, le64(n)) }

func compactMsg(payload []byte) []byte {
	return cat([]byte{3, 0}, le16(uint16(len(payload))), payload)
}

var undefAddr = bytes.Repeat([]byte{0xff}, 8)

func contiguousMsg(addr []byte) []byte { return cat([]byte{3, 1}, addr, le64(0)) }

func dsHeader(dt, space, layout []byte) *ObjectHeader {
	return &ObjectHeader{Messages: []*HeaderMessage{
		{Type: MsgDatatype, Data: dt},
		{Type: MsgDataspace, Data: space},
		{Type: MsgDataLayout, Data: layout},
	}}
}

func allocatedBytes(fn func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

var (
	stringType  = dtMsg(3, 1, nil) // fixed string, 1 byte, null-terminated
	complexType = dtMsg(11, 16, dtMsg(1, 8, make([]byte, 12)))
	vlenType    = dtMsg(9, 16, dtMsg(0, 1, make([]byte, 4)))
)

// TestReadLimitCountsOutputCost: the limit has to cover what the typed result
// costs (16 B per string, 16 B per complex128, 24 B per vlen slice header), not
// just the stored bytes, and compact data must actually hold the declared
// elements.
func TestReadLimitCountsOutputCost(t *testing.T) {
	const limit = 1 << 20
	sb := hostileSB()
	r := bytes.NewReader(nil)

	type reader func(h *ObjectHeader) error
	readers := map[string]struct {
		dt   []byte
		read reader
	}{
		"strings": {stringType, func(h *ObjectHeader) error { _, err := ReadDatasetStrings(r, h, sb, limit); return err }},
		"complex": {complexType, func(h *ObjectHeader) error { _, err := ReadDatasetComplex(r, h, sb, limit); return err }},
		"vlen":    {vlenType, func(h *ObjectHeader) error { _, err := ReadDatasetVLenBytes(r, h, sb, limit); return err }},
	}
	for name, c := range readers {
		t.Run(name+"/compact", func(t *testing.T) {
			// 4 bytes of payload declaring 16M elements.
			h := dsHeader(c.dt, spaceMsg(1<<24), compactMsg([]byte{1, 2, 3, 4}))
			var err error
			got := allocatedBytes(func() { err = guard(t, func() error { return c.read(h) }) })
			require.Error(t, err)
			require.Less(t, got, uint64(limit))
		})
	}

	t.Run("strings/undefined address", func(t *testing.T) {
		// 1M one-byte strings: 1 MB stored (under the limit), 16 MB as []string.
		h := dsHeader(stringType, spaceMsg(1<<20), contiguousMsg(undefAddr))
		var err error
		got := allocatedBytes(func() { _, err = ReadDatasetStrings(r, h, sb, limit) })
		require.ErrorContains(t, err, "limit")
		require.Less(t, got, uint64(4<<20))
	})

	t.Run("compact within payload still reads", func(t *testing.T) {
		h := dsHeader(stringType, spaceMsg(3), compactMsg([]byte("abc")))
		got, err := ReadDatasetStrings(r, h, sb, limit)
		require.NoError(t, err)
		require.Equal(t, []string{"a", "b", "c"}, got)
	})
}

// TestVLenPayloadsCountAgainstLimit: many elements may reference one large
// global heap object, so the copied payloads total far more than the heap IDs
// that were bounded up front.
func TestVLenPayloadsCountAgainstLimit(t *testing.T) {
	const n, objLen, refsAt = 1000, 400, 1024
	blob := buildGlobalHeapWithString(string(bytes.Repeat([]byte{'x'}, objLen)))
	blob = append(blob, make([]byte, refsAt-len(blob))...)
	for i := 0; i < n; i++ {
		blob = append(blob, cat(le32(objLen), le64(heapAddr), le32(1))...)
	}
	h := dsHeader(vlenType, spaceMsg(n), contiguousMsg(le64(refsAt)))

	got, err := ReadDatasetVLenBytes(bytes.NewReader(blob), h, hostileSB(), 4<<20)
	require.NoError(t, err)
	require.Len(t, got, n)

	// 24 KB headers + 16 KB IDs fit in 100 KB; the 400 KB of payloads do not.
	_, err = ReadDatasetVLenBytes(bytes.NewReader(blob), h, hostileSB(), 100<<10)
	require.ErrorContains(t, err, "limit")
}

// TestHostileChunkLayouts: chunk dimensions come from the file.
func TestHostileChunkLayouts(t *testing.T) {
	f64 := dtMsg(1, 8, make([]byte, 12))
	space2 := cat([]byte{1, 2, 0, 0, 0, 0, 0, 0}, le64(4), le64(4))

	t.Run("missing dimensions", func(t *testing.T) {
		// Rank 2 dataset, chunk layout with one dimension (v3: ndims, addr, dims).
		layout := cat([]byte{3, 2, 1}, undefAddr, le32(8))
		err := guard(t, func() error {
			_, err := ReadDatasetFloat64(bytes.NewReader(nil), dsHeader(f64, space2, layout), hostileSB(), 0)
			return err
		})
		require.ErrorContains(t, err, "dimensions")
	})

	t.Run("zero dimension", func(t *testing.T) {
		layout := cat([]byte{3, 2, 3}, undefAddr, le32(0), le32(2), le32(8))
		err := guard(t, func() error {
			_, err := ReadDatasetFloat64(bytes.NewReader(nil), dsHeader(f64, space2, layout), hostileSB(), 0)
			return err
		})
		require.ErrorContains(t, err, "zero-size chunk dimension")
	})

	t.Run("element size mismatch", func(t *testing.T) {
		layout := cat([]byte{3, 2, 3}, undefAddr, le32(2), le32(2), le32(4))
		err := guard(t, func() error {
			_, err := ReadDatasetFloat64(bytes.NewReader(nil), dsHeader(f64, space2, layout), hostileSB(), 0)
			return err
		})
		require.ErrorContains(t, err, "element size")
	})

	t.Run("implicit index with zero element size", func(t *testing.T) {
		// Layout v4, implicit index at 8, chunk {1, 0}: every chunk is 0 bytes,
		// so they "fit" in a 100 byte file however many there are.
		layout := cat([]byte{4, 2, 0, 2, 4}, le32(1), le32(0), []byte{ChunkIndexImplicit}, le64(8))
		space := cat([]byte{1, 1, 0, 0, 0, 0, 0, 0}, le64(1<<22))
		var err error
		got := allocatedBytes(func() {
			err = guard(t, func() error {
				_, err := ReadDatasetFloat64(bytes.NewReader(make([]byte, 100)), dsHeader(f64, space, layout), hostileSB(), 0)
				return err
			})
		})
		require.Error(t, err)
		require.Less(t, got, uint64(1<<20))
	})
}

// TestFletcher32AgainstReference pins fletcher32 to H5_checksum_fletcher32's
// definition on inputs a C-written fixture does not reach (long runs that
// exercise the 360-word folding, odd lengths, 0xff data that lands sums on
// 65535). The reference here is the textbook two-sum loop with the fold
// applied every word, which is equivalent.
func TestFletcher32AgainstReference(t *testing.T) {
	ref := func(d []byte) uint32 {
		fold := func(x uint32) uint32 { return x&0xffff + x>>16 }
		var s1, s2 uint32
		for i := 0; i < len(d); i += 2 {
			w := uint32(d[i]) << 8
			if i+1 < len(d) {
				w |= uint32(d[i+1])
			}
			s1 = fold(s1 + w)
			s2 = fold(s2 + s1)
		}
		return fold(s2)<<16 | fold(s1)
	}
	for _, n := range []int{0, 1, 2, 3, 719, 720, 721, 1441, 5000} {
		for _, fill := range []byte{0, 0xff, 0x5a} {
			d := bytes.Repeat([]byte{fill}, n)
			require.Equal(t, ref(d), fletcher32(d), "len %d fill %#x", n, fill)
		}
	}
}
