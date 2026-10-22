package hdf5

import (
	"fmt"

	"github.com/shyrmapp/hdf5/internal/core"
)

// readHyperslabContiguous reads a hyperslab from a contiguous dataset. Only
// the bytes the selection spans are read: one ReadAt when the selection is a
// single linear run, otherwise one ReadAt per innermost row.
func (d *Dataset) readHyperslabContiguous(
	selection *HyperslabSelection,
	datatype *core.DatatypeMessage,
	dataspace *core.DataspaceMessage,
	layout *core.DataLayoutMessage,
	fill []byte,
) (interface{}, error) {
	es := uint64(datatype.Size)
	dims := dataspace.Dimensions
	n := calculateHyperslabOutputSize(selection)
	if n == 0 {
		return []float64{}, nil
	}
	out := make([]byte, n*es)

	// Storage never allocated: every element is the fill value.
	if core.IsUndefinedAddress(layout.DataAddress, d.file.sb.OffsetSize) {
		core.FillBuffer(out, fill)
		return core.ConvertToFloat64(out, datatype, n, d.file.maxReadBytes)
	}

	if isLinearRun(selection, dims) {
		off := layout.DataAddress + calculateLinearOffset(selection.Start, dims)*es
		//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
		if _, err := d.file.reader.ReadAt(out, int64(off)); err != nil {
			return nil, fmt.Errorf("failed to read contiguous data: %w", err)
		}
		return core.ConvertToFloat64(out, datatype, n, d.file.maxReadBytes)
	}

	// General case: per-dimension selected coordinates; one read per row of
	// the innermost dimension, gathering the selected columns from the span.
	nd := len(dims)
	coords := make([][]uint64, nd)
	for i := range coords {
		coords[i] = selectedCoords(selection, i)
	}
	cols := coords[nd-1]
	lo, hi := cols[0], cols[len(cols)-1]+1
	span := make([]byte, (hi-lo)*es)
	idx := make([]int, nd)
	at := make([]uint64, nd)
	pos := uint64(0)
	for {
		for i := 0; i < nd-1; i++ {
			at[i] = coords[i][idx[i]]
		}
		at[nd-1] = lo
		off := layout.DataAddress + calculateLinearOffset(at, dims)*es
		//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
		if _, err := d.file.reader.ReadAt(span, int64(off)); err != nil {
			return nil, fmt.Errorf("failed to read contiguous row: %w", err)
		}
		for _, c := range cols {
			copy(out[pos*es:(pos+1)*es], span[(c-lo)*es:(c-lo+1)*es])
			pos++
		}
		// Advance the odometer over the outer dimensions.
		i := nd - 2
		for ; i >= 0; i-- {
			idx[i]++
			if idx[i] < len(coords[i]) {
				break
			}
			idx[i] = 0
		}
		if i < 0 {
			break
		}
	}
	return core.ConvertToFloat64(out, datatype, n, d.file.maxReadBytes)
}

// selectedCoords lists the coordinates selected in dimension i, in output
// order: Count blocks of Block consecutive coordinates, Stride apart.
func selectedCoords(sel *HyperslabSelection, i int) []uint64 {
	out := make([]uint64, 0, sel.Count[i]*sel.Block[i])
	for c := uint64(0); c < sel.Count[i]; c++ {
		for b := uint64(0); b < sel.Block[i]; b++ {
			out = append(out, sel.Start[i]+c*sel.Stride[i]+b)
		}
	}
	return out
}

// isLinearRun reports whether the selection is one run of consecutive
// elements in row-major order: leading dimensions select a single index, the
// first dimension selecting more is a plain range, and every later dimension
// is selected whole.
func isLinearRun(sel *HyperslabSelection, dims []uint64) bool {
	// plainRange: the dimension's selected coordinates are consecutive.
	plainRange := func(i int) bool {
		return sel.Count[i] == 1 || sel.Stride[i] == sel.Block[i]
	}
	seenRange := false
	for i := range dims {
		span := sel.Count[i] * sel.Block[i]
		switch {
		case !seenRange && span == 1:
			// Leading single index.
		case !seenRange:
			if !plainRange(i) {
				return false
			}
			seenRange = true
		default:
			if !plainRange(i) || sel.Start[i] != 0 || span != dims[i] {
				return false
			}
		}
	}
	return true
}
