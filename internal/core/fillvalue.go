package core

import "encoding/binary"

// FillValueBytes returns the dataset's fill value as one element's bytes in
// the dataset's own datatype, or nil when elements are zero-filled: no fill
// message, fill undefined, fill time "never", or a value whose size does not
// match elemSize (e.g. a variable-length type). Unallocated storage and
// missing chunks read as this value (netCDF-4 stores NC_FILL here).
//
// Handles the old message (type 4: size + value) and the new message (type 5)
// versions 1-3.
func FillValueBytes(header *ObjectHeader, elemSize uint32) []byte {
	if header == nil || elemSize == 0 {
		return nil
	}
	var oldMsg, newMsg []byte
	for _, m := range header.Messages {
		switch m.Type {
		case MsgFillValueOld:
			oldMsg = m.Data
		case MsgFillValue:
			newMsg = m.Data
		}
	}

	var val []byte
	switch {
	case newMsg != nil:
		val = parseNewFillValue(newMsg)
	case oldMsg != nil:
		val = parseSizedValue(oldMsg)
	}
	if uint32(len(val)) != elemSize { //nolint:gosec // G115: len bounded by message size
		return nil
	}
	for _, b := range val {
		if b != 0 {
			return val
		}
	}
	return nil // all-zero fill is the default
}

// parseSizedValue decodes "u32 size, then size bytes".
func parseSizedValue(d []byte) []byte {
	if len(d) < 4 {
		return nil
	}
	n := uint64(binary.LittleEndian.Uint32(d))
	if n == 0 || n > uint64(len(d))-4 {
		return nil
	}
	return d[4 : 4+n]
}

// parseNewFillValue decodes fill value message v1-v3.
func parseNewFillValue(d []byte) []byte {
	if len(d) < 2 {
		return nil
	}
	switch d[0] {
	case 1, 2:
		// version, alloc time, fill write time, fill defined, [size, value]
		if len(d) < 4 || d[2] == 1 /* H5D_FILL_TIME_NEVER */ || d[3] == 0 {
			return nil
		}
		return parseSizedValue(d[4:])
	case 3:
		// version, flags: bits 0-1 alloc time, 2-3 write time, 4 undefined, 5 defined.
		flags := d[1]
		if (flags>>2)&3 == 1 /* H5D_FILL_TIME_NEVER */ || flags&0x10 != 0 || flags&0x20 == 0 {
			return nil
		}
		return parseSizedValue(d[2:])
	}
	return nil
}

// fillBuffer replicates the element pattern fill across buf. A nil fill
// leaves buf (already zero) untouched.
func fillBuffer(buf, fill []byte) {
	if len(fill) == 0 {
		return
	}
	n := copy(buf, fill)
	for n < len(buf) {
		n += copy(buf[n:], buf[:n])
	}
}

// FillBuffer is fillBuffer for callers outside the package.
func FillBuffer(buf, fill []byte) { fillBuffer(buf, fill) }

// IsUndefinedAddress reports whether addr is the HDF5 undefined-address marker.
func IsUndefinedAddress(addr uint64, offsetSize uint8) bool {
	return isUndefinedAddress(addr, offsetSize)
}

// CheckReadSize rejects an allocation of n*bytesPerElement over maxBytes
// (0 = default limit), before the caller allocates.
func CheckReadSize(n, bytesPerElement, maxBytes uint64, what string) error {
	return checkReadSize(n, bytesPerElement, maxBytes, what)
}
