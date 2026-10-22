package hdf5

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

// Fixtures come from testdata/c_interop/gen_fill.c.
func TestAttributeTypesFromC(t *testing.T) {
	f, err := Open("testdata/c_interop/attr_types.h5")
	require.NoError(t, err)
	defer f.Close()

	attrs, err := f.Root().Attributes()
	require.NoError(t, err)
	got := map[string]interface{}{}
	for _, a := range attrs {
		v, err := a.ReadValue()
		require.NoError(t, err, a.Name)
		got[a.Name] = v
	}
	require.Equal(t, map[string]interface{}{
		"i8":    int8(-5),
		"i16":   int16(-300),
		"u8":    uint8(250),
		"u16":   uint16(60000),
		"u32":   uint32(4000000000),
		"u64":   uint64(math.MaxUint64),
		"f64be": 1.5,
		"i32be": int32(-7),
	}, got)
}

func TestSuperblockSizesFromC(t *testing.T) {
	for _, tc := range []struct {
		file    string
		version uint8
		off     uint8
		length  uint8
	}{
		{"sb_v0_a4_l8.h5", 0, 4, 8},
		{"sb_v0_a4_l4.h5", 0, 4, 4},
		{"sb_v2_a4_l8.h5", 3, 4, 8},
		{"sb_v2_a4_l4.h5", 3, 4, 4},
	} {
		t.Run(tc.file, func(t *testing.T) {
			f, err := Open("testdata/c_interop/" + tc.file)
			require.NoError(t, err)
			defer f.Close()
			sb := f.Superblock()
			require.Equal(t, tc.version, sb.Version)
			require.Equal(t, tc.off, sb.OffsetSize)
			require.Equal(t, tc.length, sb.LengthSize)

			obj, err := f.Resolve("/d")
			require.NoError(t, err)
			vals, err := obj.(*Dataset).Read()
			require.NoError(t, err)
			require.Equal(t, []float64{0, 1, 2, 3, 4}, vals)
		})
	}
}
