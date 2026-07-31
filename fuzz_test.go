package hdf5

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
				if attrs, err := v.Attributes(); err == nil {
					for _, a := range attrs {
						_ = a
					}
				}
			case *Group:
				_, _ = v.Attributes()
			}
		})
	})
}

func seedCorpus(f *testing.F) {
	dir := filepath.Join("testdata", "hdf5_official")
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
