# Examples

Each example is a standalone `main.go`. Run with `go run`:

```bash
go run ./examples/01-basic          # open a file, read the superblock, walk the tree
go run ./examples/02-list-objects   # traverse groups and datasets
go run ./examples/03-read-dataset   # read numeric datasets and metadata
go run ./examples/04-vlen-strings   # read variable-length strings via the global heap
go run ./examples/05-comprehensive  # all superblock versions, layouts, datatypes, GZIP
go run ./examples/06-write-dataset  # create a file and write a dataset
```

The reading examples expect test files in `testdata/` (`v0.h5`, `v2.h5`,
`v3.h5`, `with_groups.h5`, `vlen_strings.h5`). Most auto-generate them if
Python with `h5py` is available; `06-write-dataset` writes one with this
library.

API reference: <https://pkg.go.dev/github.com/shyrmapp/hdf5>
