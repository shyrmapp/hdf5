package writer

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFileWriter_Allocate(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.h5")

	writer, err := NewFileWriter(path, ModeTruncate, 48)
	require.NoError(t, err)
	defer func() { _ = writer.Close() }()

	t.Run("sequential allocations", func(t *testing.T) {
		addr1, err := writer.Allocate(100)
		require.NoError(t, err)
		assert.Equal(t, uint64(48), addr1)
		assert.Equal(t, uint64(148), writer.EndOfFile())

		addr2, err := writer.Allocate(200)
		require.NoError(t, err)
		assert.Equal(t, uint64(148), addr2)
		assert.Equal(t, uint64(348), writer.EndOfFile())
	})

	t.Run("zero size allocation fails", func(t *testing.T) {
		_, err := writer.Allocate(0)
		assert.Error(t, err)
	})
}

func TestFileWriter_WriteAt(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.h5")

	writer, err := NewFileWriter(path, ModeTruncate, 0)
	require.NoError(t, err)
	defer func() { _ = writer.Close() }()

	t.Run("write data at address", func(t *testing.T) {
		data := []byte("Hello, HDF5!")
		addr, err := writer.Allocate(uint64(len(data)))
		require.NoError(t, err)

		n, err := writer.WriteAt(data, int64(addr))
		require.NoError(t, err)
		assert.Equal(t, len(data), n)

		// Read back to verify
		buf := make([]byte, len(data))
		n, err = writer.ReadAt(buf, int64(addr))
		require.NoError(t, err)
		assert.Equal(t, len(data), n)
		assert.Equal(t, data, buf)
	})

	t.Run("write empty data", func(t *testing.T) {
		n, err := writer.WriteAt([]byte{}, 0)
		assert.NoError(t, err)
		assert.Equal(t, 0, n) // Should be no-op
	})

	t.Run("write at specific address", func(t *testing.T) {
		data := []byte{0x01, 0x02, 0x03, 0x04}
		addr, _ := writer.Allocate(uint64(len(data)))

		n, err := writer.WriteAt(data, int64(addr))
		require.NoError(t, err)
		assert.Equal(t, len(data), n)

		// Verify
		buf := make([]byte, len(data))
		_, err = writer.ReadAt(buf, int64(addr))
		require.NoError(t, err)
		assert.Equal(t, data, buf)
	})
}

func TestFileWriter_Close(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.h5")

	writer, err := NewFileWriter(path, ModeTruncate, 0)
	require.NoError(t, err)

	// Close once
	err = writer.Close()
	assert.NoError(t, err)

	// Close again (should be safe)
	err = writer.Close()
	assert.NoError(t, err)

	// Operations after close should fail
	_, err = writer.Allocate(100)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	_, err = writer.WriteAt([]byte("test"), 0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")

	err = writer.Flush()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "closed")
}
