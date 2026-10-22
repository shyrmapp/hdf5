package hdf5

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shyrmapp/hdf5/internal/core"
)

// FuzzRead feeds arbitrary bytes through the full untrusted-input surface:
// open, walk the object tree, and exercise every public read path. The
// invariant is "error, never panic" — any panic on malformed input is a bug.
// Seeded with the official HDF5 test corpus (small files only, to keep
// mutation throughput up).
func FuzzRead(f *testing.F) {
	seedCorpus(f)

	f.Fuzz(func(t *testing.T, data []byte) {
		path := filepath.Join(t.TempDir(), "fuzz.h5")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Skip()
		}
		file, err := Open(path)
		if err != nil {
			return
		}
		defer file.Close()

		file.Walk(func(_ string, obj Object) {
			switch v := obj.(type) {
			case *Dataset:
				_, _ = v.Read()
				_, _ = v.ReadStrings()
				_, _ = v.ReadCompound()
				_, _ = v.ReadVLenBytes()
				fuzzSlice(file, v)
				fuzzChunks(v)
				fuzzAttrs(v.Attributes())
			case *Group:
				fuzzAttrs(v.Attributes())
			}
		})
	})
}

// fuzzSlice reads a small corner hyperslab (up to 2 elements per dimension).
func fuzzSlice(file *File, d *Dataset) {
	header, err := core.ReadObjectHeader(file.osFile, d.address, file.sb)
	if err != nil {
		return
	}
	for _, msg := range header.Messages {
		if msg.Type != core.MsgDataspace {
			continue
		}
		ds, err := core.ParseDataspaceMessage(msg.Data)
		if err != nil || len(ds.Dimensions) == 0 {
			return
		}
		start := make([]uint64, len(ds.Dimensions))
		count := make([]uint64, len(ds.Dimensions))
		for i, dim := range ds.Dimensions {
			count[i] = min(dim, 2)
		}
		_, _ = d.ReadSlice(start, count)
		return
	}
}

// fuzzChunks pulls the first few chunks through the chunk iterator.
func fuzzChunks(d *Dataset) {
	it, err := d.ChunkIterator()
	if err != nil {
		return
	}
	for i := 0; i < 3 && it.Next(); i++ {
		_, _ = it.Chunk()
	}
}

// fuzzAttrs decodes every attribute value.
func fuzzAttrs(attrs []*core.Attribute, err error) {
	if err != nil {
		return
	}
	for _, a := range attrs {
		_, _ = a.ReadValue()
	}
}

func seedCorpus(f *testing.F) {
	for _, dir := range []string{filepath.Join("testdata", "hdf5_official"), filepath.Join("testdata", "layout_v4")} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			f.Fatal(err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".h5") {
				continue
			}
			if info, err := e.Info(); err != nil || info.Size() > 128<<10 {
				continue
			}
			if data, err := os.ReadFile(filepath.Join(dir, e.Name())); err == nil {
				f.Add(data)
			}
		}
	}
}
