// Package hdf5 is a pure Go reader and writer for HDF5 files (superblock
// versions 0, 2 and 3), covering datasets, groups, attributes and links.
//
// Open reads a file; whole-dataset reads are capped at 1 GiB by default, see
// WithMaxReadBytes. A File is not safe for concurrent use; separate Files are
// independent. CreateForWrite and OpenForWrite return a FileWriter.
package hdf5

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/shyrmapp/hdf5/internal/core"
)

// File represents an open HDF5 file with its metadata and root group.
type File struct {
	osFile        *os.File
	reader        io.ReaderAt // osFile shifted by the superblock's base address
	sb            *core.Superblock
	root          *Group
	loadingBTrees map[uint64]bool   // B-trees on the current load path, to cut ancestor cycles
	loadingGroups map[uint64]bool   // Groups on the current load path (hard-link cycle guard)
	groupCache    map[uint64]*Group // loaded groups by header address; aliases share children
	groupLoading  map[uint64]bool   // groups mid-load, to keep cycle cut-offs out of groupCache
	maxReadBytes  uint64            // Ceiling for a single whole-dataset read; 0 = default
}

// OpenOption configures how a file is opened.
type OpenOption func(*File)

// WithMaxReadBytes sets the ceiling for a single whole-dataset read, such as
// [Dataset.Read] or [Dataset.ReadStrings]. The default is 1 GiB.
//
// Set this if you read files you did not produce. How big a dataset claims to
// be is a number taken from the file, and a 6 KiB file can legitimately claim
// to hold tens of gigabytes. Go cannot recover from an allocation that
// exhausts memory — it is a fatal runtime error, not a panic — so the claim is
// checked before any allocation, and the ceiling is what that check compares
// against. Sizing it to your data turns a hostile file into an error instead
// of a dead process.
//
// Pick the limit from the largest dataset you expect, with some headroom, not
// from the memory of the machine you happen to run on. A limit derived from
// available RAM makes the same file succeed on one host and fail on another,
// and makes a bigger host accept a bigger hostile allocation.
//
//	// largest real product is ~105 MiB once widened to float64
//	f, err := hdf5.Open(name, hdf5.WithMaxReadBytes(256<<20))
//
// The limit counts the memory the call allocates, not the bytes stored on
// disk. Reading an int8 dataset through [Dataset.Read] yields float64, so it
// costs 8 bytes per element, not 1.
//
// To read data larger than the limit without raising it, use
// [Dataset.ReadSlice] or [Dataset.ChunkIterator], which read in pieces and are
// bounded by the size of each piece.
func WithMaxReadBytes(n uint64) OpenOption {
	return func(f *File) { f.maxReadBytes = n }
}

// Open opens an HDF5 file for reading and returns a File handle.
// The file must be a valid HDF5 file with a supported format version.
func Open(filename string, opts ...OpenOption) (*File, error) {
	//nolint:gosec // G304: User-provided filename is intentional for HDF5 file library
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("file open failed: %w", err)
	}

	// Release the descriptor if a loader panics (the panic itself propagates).
	opened := false
	defer func() {
		if !opened {
			_ = f.Close()
		}
	}()

	// Get file size for address validation.
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("file stat failed: %w", err)
	}
	fileSize := fi.Size()

	sb, err := core.ReadSuperblock(f)
	if err != nil {
		_ = f.Close()
		if errors.Is(err, core.ErrNoSignature) {
			return nil, errors.New("not an HDF5 file")
		}
		return nil, fmt.Errorf("superblock read failed: %w", err)
	}

	// Addresses in the file are relative to the superblock, which a user block
	// can push past offset 0; the section reader shifts every read by that base.
	//nolint:gosec // G115: BaseAddress is a located file offset, so <= fileSize
	base := int64(sb.BaseAddress)
	file := &File{
		osFile:        f,
		reader:        io.NewSectionReader(f, base, fileSize-base),
		sb:            sb,
		loadingBTrees: make(map[uint64]bool),
		loadingGroups: make(map[uint64]bool),
		groupCache:    make(map[uint64]*Group),
		groupLoading:  make(map[uint64]bool),
	}
	for _, opt := range opts {
		opt(file)
	}

	// Validate root group address.
	//nolint:gosec // G115: File size is always positive, safe to convert int64 to uint64
	if sb.RootGroup >= uint64(fileSize-base) {
		_ = f.Close()
		return nil, fmt.Errorf("root group address %d beyond file size %d (after base address %d)",
			sb.RootGroup, fileSize-base, base)
	}

	// For all versions, sb.RootGroup now contains the correct object header address.
	file.root, err = loadGroup(file, sb.RootGroup)
	if err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("root group load failed: %w", err)
	}

	// Ensure root group always has name "/" (may be empty from object header)
	file.root.name = "/"

	opened = true
	return file, nil
}

// Close closes the HDF5 file and releases associated resources.
// It is safe to call Close multiple times.
func (f *File) Close() error {
	if f.osFile == nil {
		return nil // Already closed.
	}
	err := f.osFile.Close()
	f.osFile = nil // Prevent double close.
	return err
}

// Root returns the root group of the HDF5 file.
func (f *File) Root() *Group {
	return f.root
}

// Walk traverses the entire file structure, calling fn for each object.
// Objects are visited in depth-first order starting from the root group.
//
// A group reachable through several hard links is reported under every path,
// but its children are only walked under the first path, so a link diamond
// costs one visit per link rather than one per path (which is exponential in
// the diamond depth). Use Group.Children on the alias to list them.
func (f *File) Walk(fn func(path string, obj Object)) {
	walkGroup(f.root, "/", fn, map[*Object]bool{})
}

func walkGroup(g *Group, currentPath string, fn func(string, Object), walked map[*Object]bool) {
	fn(currentPath, g)

	// Aliases share one children slice, so its first element identifies it.
	if len(g.children) > 0 {
		if walked[&g.children[0]] {
			return
		}
		walked[&g.children[0]] = true
	}
	for _, child := range g.Children() {
		childPath := currentPath + child.Name()

		if childGroup, ok := child.(*Group); ok {
			walkGroup(childGroup, childPath+"/", fn, walked)
		} else {
			fn(childPath, child)
		}
	}
}

// SuperblockVersion returns the HDF5 superblock format version (0, 2, or 3).
func (f *File) SuperblockVersion() uint8 {
	return f.sb.Version
}

// Superblock returns the file's superblock metadata structure.
func (f *File) Superblock() *core.Superblock {
	return f.sb
}

// Reader returns the file reader for low-level access. Offsets are HDF5
// addresses, relative to Superblock().BaseAddress (non-zero only with a user block).
func (f *File) Reader() io.ReaderAt {
	return f.reader
}

// readSignature reads 4 bytes at address and returns string.
func readSignature(r io.ReaderAt, address uint64) string {
	buf := make([]byte, 4)
	//nolint:gosec // G115: HDF5 addresses fit in int64 for io.ReaderAt interface
	if _, err := r.ReadAt(buf, int64(address)); err != nil {
		return ""
	}
	return string(buf)
}
