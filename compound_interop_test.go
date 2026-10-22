package hdf5

import (
	"math"
	"strings"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// Fixtures come from testdata/c_interop/gen_compound.c; expected values are
// what h5dump prints for them.
var compoundFixtures = []string{
	"testdata/c_interop/compound_v1.h5", // compound datatype v1
	"testdata/c_interop/compound_v3.h5", // compound datatype v3
}

func TestCompoundAllMembers(t *testing.T) {
	want := []map[string]any{
		{
			"d": 1.5, "i8": int8(-8), "i16": int16(-300), "i32": int32(-70000), "i64": int64(-5000000000),
			"u8": uint8(200), "u16": uint16(60000), "u32": uint32(3000000000), "u64": uint64(18000000000000000000),
			"be": int32(-123456), "fs": "abcd", "vs": "hello",
			"in":  core.CompoundValue{"a": int32(7), "b": 2.5},
			"arr": []any{int32(1), int32(-2), int32(3)},
		},
		{
			"d": -2.25, "i8": int8(127), "i16": int16(32767), "i32": int32(math.MaxInt32), "i64": int64(math.MaxInt64),
			"u8": uint8(255), "u16": uint16(65535), "u32": uint32(math.MaxUint32), "u64": uint64(math.MaxUint64),
			"be": int32(654321), "fs": "xy", "vs": "",
			"in":  core.CompoundValue{"a": int32(-9), "b": 0.125},
			"arr": []any{int32(10), int32(20), int32(30)},
		},
	}
	// "all" has an array member (compound v2 under EARLIEST); "noarr" is a
	// genuine v1 compound there.
	for _, path := range compoundFixtures {
		for _, name := range []string{"all", "noarr"} {
			t.Run(path+"/"+name, func(t *testing.T) {
				got, err := openDS(t, path, name).ReadCompound()
				require.NoError(t, err)
				require.Len(t, got, 2)
				for i := range want {
					exp := map[string]any{}
					for k, v := range want[i] {
						if k != "arr" || name == "all" {
							exp[k] = v
						}
					}
					require.Equal(t, exp, map[string]any(got[i]))
				}
			})
		}
	}
}

func TestCompoundSmallSize24(t *testing.T) {
	for _, path := range compoundFixtures {
		t.Run(path, func(t *testing.T) {
			got, err := openDS(t, path, "ddu").ReadCompound()
			require.NoError(t, err)
			require.Len(t, got, 2)
			require.Equal(t, map[string]any{"d": 1.5, "i": int32(-7), "u": uint32(3000000000)}, map[string]any(got[0]))
			require.Equal(t, map[string]any{"d": 2.5, "i": int32(8), "u": uint32(5)}, map[string]any(got[1]))
		})
	}
}

func TestCompoundLargeSize2ByteOffsets(t *testing.T) {
	for _, path := range compoundFixtures {
		t.Run(path, func(t *testing.T) {
			got, err := openDS(t, path, "big").ReadCompound()
			require.NoError(t, err)
			require.Len(t, got, 2)
			require.Equal(t, map[string]any{"d": 3.5, "s": "first", "i": int32(-42)}, map[string]any(got[0]))
			require.Equal(t, map[string]any{"d": 4.5, "s": strings.Repeat("z", 299), "i": int32(99)}, map[string]any(got[1]))
		})
	}
}
