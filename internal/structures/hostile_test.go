package structures

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/shyrmapp/hdf5/internal/core"
	"github.com/stretchr/testify/require"
)

// guard runs fn and fails on a panic or a hang.
func guard(t *testing.T, fn func() error) error {
	t.Helper()
	type result struct {
		err      error
		panicked interface{}
	}
	done := make(chan result, 1)
	go func() {
		var res result
		defer func() {
			res.panicked = recover()
			done <- res
		}()
		res.err = fn()
	}()
	select {
	case res := <-done:
		require.Nil(t, res.panicked, "parser panicked")
		return res.err
	case <-time.After(5 * time.Second):
		t.Fatal("parser did not terminate (hang)")
		return nil
	}
}

func TestHostileLinkMessage(t *testing.T) {
	sb := &core.Superblock{Version: 2, OffsetSize: 8, LengthSize: 8, Endianness: binary.LittleEndian}

	// version 1, flags: 8-byte name length; length 2^63+1 wraps int(nameLen) negative.
	hugeName := append([]byte{1, 0x03}, binary.LittleEndian.AppendUint64(nil, 1<<63+1)...)
	hugeName = append(hugeName, "abcdefgh"...)

	// Soft link whose declared value length exceeds the data.
	softShort := []byte{1, 0x08, byte(LinkTypeSoft), 1, 'a', 0xFF, 0xFF, 'x'}

	for name, data := range map[string][]byte{
		"name length 2^63+1":       hugeName,
		"soft link value past end": softShort,
	} {
		t.Run(name, func(t *testing.T) {
			err := guard(t, func() error {
				_, err := ParseLinkMessage(data, sb)
				return err
			})
			require.Error(t, err)
		})
	}
}
