package hdf5

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testWeatherFields is the compound layout used by the public-API tests:
// struct { float64 temp; int32 id; char name[8] } — 20 bytes per row.
var testWeatherFields = []CompoundField{
	{Name: "temp", Type: Float64},
	{Name: "id", Type: Int32},
	{Name: "name", Type: String, StringSize: 8},
}

// testWeatherRow packs one row of testWeatherFields as little-endian bytes.
func testWeatherRow(temp float64, id int32, name string) []byte {
	row := make([]byte, 20)
	binary.LittleEndian.PutUint64(row[0:8], math.Float64bits(temp))
	binary.LittleEndian.PutUint32(row[8:12], uint32(id)) // Intentional bit pattern
	copy(row[12:20], name)
	return row
}

// testWriteWeatherFile writes a compound dataset via the public field API and
// returns the file path.
func testWriteWeatherFile(t *testing.T, dir string) string {
	t.Helper()
	filename := filepath.Join(dir, "compound_public.h5")

	fw, err := CreateForWrite(filename, CreateTruncate)
	require.NoError(t, err)

	ds, err := fw.CreateCompoundDatasetFromFields("/weather", testWeatherFields, []uint64{3})
	require.NoError(t, err)

	data := make([]byte, 0, 60)
	data = append(data, testWeatherRow(21.5, 1, "berlin")...)
	data = append(data, testWeatherRow(18.25, 2, "oslo")...)
	data = append(data, testWeatherRow(-3.5, 3, "tromso")...)
	require.NoError(t, ds.WriteRaw(data))

	require.NoError(t, fw.Close())
	return filename
}

// TestCreateCompoundDatasetFromFields writes a compound dataset through the
// public API and reads the values back through Open().
func TestCreateCompoundDatasetFromFields(t *testing.T) {
	filename := testWriteWeatherFile(t, t.TempDir())

	f, err := Open(filename)
	require.NoError(t, err)
	defer f.Close()

	var ds *Dataset
	f.Walk(func(path string, obj Object) {
		if path == "/weather" {
			ds, _ = obj.(*Dataset)
		}
	})
	require.NotNil(t, ds, "dataset /weather not found")

	rows, err := ds.ReadCompound()
	require.NoError(t, err)
	require.Len(t, rows, 3)

	assert.InDelta(t, 21.5, rows[0]["temp"], 1e-12)
	assert.EqualValues(t, 1, rows[0]["id"])
	assert.Equal(t, "berlin", rows[0]["name"])
	assert.InDelta(t, -3.5, rows[2]["temp"], 1e-12)
	assert.EqualValues(t, 3, rows[2]["id"])
	assert.Equal(t, "tromso", rows[2]["name"])
}

// TestNewCompoundType_Validation covers rejection of invalid field lists.
func TestNewCompoundType_Validation(t *testing.T) {
	// No fields.
	_, err := NewCompoundType()
	require.Error(t, err)

	// String without size.
	_, err = NewCompoundType(CompoundField{Name: "s", Type: String})
	require.ErrorContains(t, err, "StringSize")

	// Unsupported type.
	_, err = NewCompoundType(CompoundField{Name: "v", Type: VLenString})
	require.ErrorContains(t, err, "unsupported compound field type")

	// Empty name.
	_, err = NewCompoundType(CompoundField{Type: Int32})
	require.ErrorContains(t, err, "empty")

	// Duplicate name.
	_, err = NewCompoundType(
		CompoundField{Name: "x", Type: Int32},
		CompoundField{Name: "x", Type: Int64},
	)
	require.ErrorContains(t, err, "duplicate")

	// Valid: size is the contiguous sum of member sizes.
	ct, err := NewCompoundType(testWeatherFields...)
	require.NoError(t, err)
	assert.Equal(t, uint32(20), ct.Size())
}

// TestCInterop_CompoundFromFields verifies the official h5dump reads a
// compound dataset written via the public field API and reports the member
// names in the H5T_COMPOUND datatype.
func TestCInterop_CompoundFromFields(t *testing.T) {
	h5dump := h5tool(t, "h5dump")
	filename := testWriteWeatherFile(t, t.TempDir())

	out := runH5(t, h5dump, "--enable-error-stack", "-p", filename)
	require.Contains(t, out, "H5T_COMPOUND")
	for _, member := range []string{`"temp"`, `"id"`, `"name"`} {
		require.Contains(t, out, member, "h5dump output missing member %s:\n%s", member, out)
	}
	// Field types and data values survive the round trip.
	require.Contains(t, out, "H5T_IEEE_F64LE")
	require.Contains(t, out, "H5T_STD_I32LE")
	require.Contains(t, out, "H5T_STRING")
	require.Contains(t, out, "berlin")
	require.Contains(t, out, "tromso")
}
