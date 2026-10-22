package hdf5

import (
	"errors"
	"fmt"

	"github.com/shyrmapp/hdf5/internal/core"
)

// errHeaderFull means a message fits neither in the object header's
// allocation nor, via a continuation message, next to its other messages.
var errHeaderFull = errors.New("object header full")

// appendHeaderMessage adds a message to the v2 object header at addr. The
// header is rewritten in place when it still fits its allocation; otherwise
// the message moves to a new OCHK continuation block and the header only
// gains a continuation message. Returns errHeaderFull if even that does not
// fit. oh must come from core.ReadObjectHeader; messages that live in
// continuation blocks stay where they are.
func (fw *FileWriter) appendHeaderMessage(addr uint64, oh *core.ObjectHeader, msgType core.MessageType,
	data []byte, sb *core.Superblock) error {
	allocSize := fw.lookupHeaderAllocSize(addr)
	if allocSize == 0 {
		return fmt.Errorf("cannot determine object header allocation at 0x%X", addr)
	}

	oh.Messages = filterMainChunkMessages(oh.Messages)
	if err := core.AddMessageToObjectHeader(oh, msgType, data); err != nil {
		return fmt.Errorf("failed to add message to header: %w", err)
	}
	if core.ObjectHeaderSizeFromParsed(oh) <= allocSize {
		return writeOHDRWithBoundsCheck(fw, addr, oh, sb)
	}

	// Swap the new message for a continuation message and check that fits
	// before allocating anything.
	oh.Messages = oh.Messages[:len(oh.Messages)-1]
	contData := core.EncodeContinuationMessage(0, 0, sb)
	if err := core.AddMessageToObjectHeader(oh, core.MsgContinuation, contData); err != nil {
		return err
	}
	if core.ObjectHeaderSizeFromParsed(oh) > allocSize {
		oh.Messages = oh.Messages[:len(oh.Messages)-1]
		return errHeaderFull
	}

	ochkMessages := []core.MessageWriter{{Type: msgType, Data: data}}
	ochkSize := core.ContinuationChunkSizeV2(ochkMessages)
	ochkAddr, err := fw.writer.Allocator().Allocate(ochkSize)
	if err != nil {
		return fmt.Errorf("failed to allocate OCHK continuation block: %w", err)
	}
	if _, err := core.WriteContinuationChunkV2(fw.writer, ochkAddr, ochkMessages); err != nil {
		return fmt.Errorf("failed to write OCHK continuation block: %w", err)
	}
	oh.Messages[len(oh.Messages)-1].Data = core.EncodeContinuationMessage(ochkAddr, ochkSize, sb)
	return writeOHDRWithBoundsCheck(fw, addr, oh, sb)
}

// patchHeaderMessage overwrites a same-sized message of the header at addr in
// place (in the main chunk or a continuation block) and refreshes the
// checksum of the chunk it lives in.
func (fw *FileWriter) patchHeaderMessage(addr uint64, oh *core.ObjectHeader, msg *core.HeaderMessage, data []byte) error {
	if len(data) != len(msg.Data) {
		return fmt.Errorf("patchHeaderMessage: size changed (%d -> %d)", len(msg.Data), len(data))
	}
	const msgHeaderLen = 4 // type(1) + size(2) + flags(1)
	if err := fw.writer.WriteAtAddress(data, msg.Offset+msgHeaderLen); err != nil {
		return fmt.Errorf("failed to patch header message: %w", err)
	}
	copy(msg.Data, data)

	if !msg.FromContinuation {
		return fw.refreshOHDRChecksum(addr)
	}
	sb := fw.file.sb
	for _, c := range oh.Messages {
		if c.Type != core.MsgContinuation || len(c.Data) < int(sb.OffsetSize+sb.LengthSize) {
			continue
		}
		caddr := readUint(c.Data[:sb.OffsetSize])
		csize := readUint(c.Data[sb.OffsetSize : sb.OffsetSize+sb.LengthSize])
		if msg.Offset >= caddr && msg.Offset < caddr+csize {
			return fw.refreshChecksum(caddr, int64(csize)-4) //nolint:gosec // G115: OCHK size is small
		}
	}
	return fmt.Errorf("continuation block of header message at 0x%X not found", msg.Offset)
}

// readUint decodes a little-endian unsigned integer of up to 8 bytes.
func readUint(b []byte) uint64 {
	var v uint64
	for i := len(b) - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}
