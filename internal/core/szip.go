package core

import (
	"encoding/binary"
	"fmt"

	"github.com/shyrmapp/aec"

	"github.com/shyrmapp/hdf5/internal/utils"
)

// SZIP option mask bits from szlib.h. Only MSB and NN affect decoding;
// the remaining bits (K13, CHIP, EC, LSB, RAW) carry no stream-level
// meaning for the AEC decoder, matching libaec's sz_compat.c.
const (
	szMSBOptionMask = 16
	szNNOptionMask  = 32
)

// Positions of the SZIP parameters in the filter's client data, as stored
// by H5Zszip.c (H5Z_SZIP_PARM_*). Note PPB comes before BPP.
const (
	szParmMask = 0 // options mask
	szParmPPB  = 1 // pixels per block
	szParmBPP  = 2 // bits per pixel
	szParmPPS  = 3 // pixels per scanline
)

// szipConfig holds the decode parameters derived from the filter's client
// data, mirroring the setup in libaec's SZ_BufftoBuffDecompress.
type szipConfig struct {
	ppb, bpp, pps int
	bits          int // coded sample width (8 when byte-interleaved)
	pixelSize     int // bytes per coded sample in the output buffer
	rsi           int
	flags         int
	deinterleave  bool // 32/64-bit pixels are stored byte-plane interleaved
	padScanline   bool
}

func parseSZIPConfig(clientData []uint32) (szipConfig, error) {
	var c szipConfig
	if len(clientData) < 4 {
		return c, fmt.Errorf("szip: expected 4 client data values, got %d", len(clientData))
	}
	mask := int(clientData[szParmMask])
	c.ppb = int(clientData[szParmPPB])
	c.bpp = int(clientData[szParmBPP])
	c.pps = int(clientData[szParmPPS])

	// Same validation as sz_compat.c.
	if c.pps == 0 || c.ppb == 0 || c.ppb%2 != 0 || c.bpp == 0 || (c.bpp > 32 && c.bpp != 64) {
		return c, fmt.Errorf("szip: invalid parameters (bpp=%d ppb=%d pps=%d)", c.bpp, c.ppb, c.pps)
	}

	// 32/64-bit pixels are byte-interleaved by the encoder and coded as
	// 8-bit samples; everything else is coded at its pixel width.
	c.deinterleave = c.bpp == 32 || c.bpp == 64
	c.bits = c.bpp
	if c.deinterleave {
		c.bits = 8
	}
	switch {
	case c.bits > 16:
		c.pixelSize = 4
	case c.bits > 8:
		c.pixelSize = 2
	default:
		c.pixelSize = 1
	}

	c.rsi = (c.pps + c.ppb - 1) / c.ppb
	c.padScanline = c.pps%c.ppb != 0

	if mask&szMSBOptionMask != 0 {
		c.flags |= aec.FlagMSB
	}
	if mask&szNNOptionMask != 0 {
		c.flags |= aec.FlagPreprocess
	}
	return c, nil
}

// applySZIP decompresses an HDF5 SZIP-filtered chunk. The chunk starts with
// the uncompressed size as a little-endian uint32 (written by H5Z_filter_szip),
// followed by a raw CCSDS 121.0-B (AEC) stream. The decode path is a port of
// SZ_BufftoBuffDecompress from libaec's sz_compat.c.
func applySZIP(clientData []uint32, data []byte) ([]byte, error) {
	c, err := parseSZIPConfig(clientData)
	if err != nil {
		return nil, err
	}
	if len(data) < 4 {
		return nil, fmt.Errorf("szip: chunk shorter than 4-byte size header")
	}
	storedLen := binary.LittleEndian.Uint32(data)
	if err := utils.ValidateBufferSize(uint64(storedLen), utils.MaxChunkSize, "szip chunk"); err != nil {
		return nil, err
	}
	destLen := int(storedLen)
	src := data[4:]

	// When the scanline is not a whole number of blocks the encoder padded
	// each scanline up to rsi*ppb pixels; decode the padded size, then strip.
	bufSize := destLen
	scanlines := 0
	if c.padScanline {
		scanlines = (destLen/c.pixelSize + c.pps - 1) / c.pps
		bufSize = c.rsi * c.ppb * c.pixelSize * scanlines
		//nolint:gosec // G115: product of validated non-negative ints
		if err := utils.ValidateBufferSize(uint64(bufSize), utils.MaxChunkSize, "szip padded buffer"); err != nil {
			return nil, err
		}
	}

	samples, err := aec.Decode(src, aec.Params{
		BitsPerSample: c.bits,
		BlockSize:     c.ppb,
		RSI:           c.rsi,
		Flags:         c.flags,
		NumValues:     bufSize / c.pixelSize,
	})
	if err != nil {
		return nil, fmt.Errorf("szip: aec decode failed: %w", err)
	}

	buf := szipPackSamples(samples, c.pixelSize, c.flags&aec.FlagMSB != 0)

	if c.padScanline {
		lineSize := c.pps * c.pixelSize
		szipRemovePadding(buf, lineSize, c.rsi*c.ppb*c.pixelSize)
		if totalOut := scanlines * lineSize; totalOut < destLen {
			destLen = totalOut
		}
	}
	// A crafted destLen that is not a multiple of the sample width can exceed
	// the decoded size (libaec truncates to whole samples); clamp, don't panic.
	if destLen > len(buf) {
		destLen = len(buf)
	}

	if !c.deinterleave {
		return buf[:destLen], nil
	}
	// SZIP's byte-plane interleave for 32/64-bit pixels is exactly the HDF5
	// shuffle layout; reuse the shuffle filter's reverse transform.
	//nolint:gosec // G115: bpp validated to 32 or 64
	return applyShuffle(buf[:destLen], []uint32{uint32(c.bpp / 8)})
}

// szipPackSamples serializes decoded samples in libaec's output layout:
// 1/2/4 bytes per sample, big-endian when the MSB option is set.
func szipPackSamples(samples []uint32, pixelSize int, msb bool) []byte {
	var order binary.ByteOrder = binary.LittleEndian
	if msb {
		order = binary.BigEndian
	}
	buf := make([]byte, len(samples)*pixelSize)
	for i, s := range samples {
		writeUint64(buf[i*pixelSize:], uint64(s), pixelSize, order)
	}
	return buf
}

// szipRemovePadding compacts padded scanlines in place: keeps the first
// lineSize bytes of every paddedLine-sized stride (sz_compat remove_padding).
func szipRemovePadding(buf []byte, lineSize, paddedLine int) {
	i := lineSize
	for j := paddedLine; j < len(buf); j += paddedLine {
		copy(buf[i:i+lineSize], buf[j:j+lineSize])
		i += lineSize
	}
}
