# HDF5 Go Library

> Pure Go implementation of the HDF5 file format — no CGo required

[![Version](https://img.shields.io/github/v/tag/shyrmapp/hdf5?style=flat-square&logo=github&color=blue&label=version)](https://github.com/shyrmapp/hdf5/tags)
[![Go Version](https://img.shields.io/github/go-mod/go-version/shyrmapp/hdf5?style=flat-square&logo=go)](https://go.dev)
[![CI](https://img.shields.io/github/actions/workflow/status/shyrmapp/hdf5/test.yml?branch=main&style=flat-square&logo=github&label=tests)](https://github.com/shyrmapp/hdf5/actions)
[![License](https://img.shields.io/github/license/shyrmapp/hdf5?style=flat-square&color=blue)](https://github.com/shyrmapp/hdf5/blob/main/LICENSE)

Reads files written by HDF5 1.x and 2.x libraries, including the 2.x "latest"
format (data layout v4/v5 with modern chunk indexes), complex datatypes, and
float16. Write output is round-tripped through the official HDF5 tools
(h5dump/h5diff/h5repack) in CI.

This is the maintained hard fork of [scigolib/hdf5](https://github.com/scigolib/hdf5).

## Quick Start

```bash
go get github.com/shyrmapp/hdf5
```

```go
package main

import (
    "fmt"
    "log"
    "github.com/shyrmapp/hdf5"
)

func main() {
    file, err := hdf5.Open("data.h5")
    if err != nil {
        log.Fatal(err)
    }
    defer file.Close()

    file.Walk(func(path string, obj hdf5.Object) {
        switch v := obj.(type) {
        case *hdf5.Group:
            fmt.Printf("group   %s (%d children)\n", path, len(v.Children()))
        case *hdf5.Dataset:
            fmt.Printf("dataset %s\n", path)
        }
    })
}
```

[More examples →](examples/)

## Features

**Reading**

- Superblocks v0/v2/v3, object headers v1/v2, traditional and dense groups
- All layouts: compact, contiguous, chunked (v1 B-tree and v4/v5 chunk
  indexes: single chunk, implicit, fixed/extensible array, v2 B-tree)
- Hyperslab selection (stride/block, reads only needed chunks) and a
  memory-efficient `ChunkIterator`
- Datatypes: fixed-point (every width + sign), float16/32/64, complex
  (HDF5 2.0 class 11), fixed and variable-length strings, compounds,
  arrays, enums, references, opaque
- Compression: GZIP, LZF, SZIP (via [shyrmapp/aec](https://github.com/shyrmapp/aec)), BZIP2; Shuffle and Fletcher32 filters
- Attributes (compact and dense), soft/hard/external link resolution

**Writing**

- Datasets in all layouts, all supported datatypes including compounds and
  variable-length strings
- Resizable datasets with unlimited dimensions
- GZIP and LZF compression, Shuffle, Fletcher32
- Groups, attributes (create/modify/delete), hard/soft/external links
- Object deletion with space reclamation

**Not supported**: SZIP write, SWMR, parallel I/O. Concurrent
access to the same `File` requires caller synchronization; separate `File`
instances are independent. See [ROADMAP.md](ROADMAP.md).

## Validation

- Tested against the official HDF5 test corpus (433 files;
  [documented exclusions](testdata/hdf5_official/KNOWN_FAILURES.md) for
  multi-file and intentionally corrupt inputs)
- C-library interop testbench in CI: written files verified with
  h5dump/h5diff/h5repack
- Defenses against malformed files: overflow-checked allocation, size
  limits, security regression tests

## Documentation

- [Quick Start](docs/guides/QUICKSTART.md) · [Installation](docs/guides/INSTALLATION.md) · [Reading Data](docs/guides/READING_DATA.md)
- [Datatypes](docs/guides/DATATYPES.md) · [Troubleshooting](docs/guides/TROUBLESHOOTING.md) · [FAQ](docs/guides/FAQ.md)
- [Architecture Overview](docs/architecture/OVERVIEW.md) · [API Reference](https://pkg.go.dev/github.com/shyrmapp/hdf5)

## Development

Requires Go 1.26+. Pure-Go dependencies only (no CGo).

```bash
go test ./...          # run tests
go test -race ./...    # with race detector
go build ./cmd/...     # build tools
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the
[issue tracker](https://github.com/shyrmapp/hdf5/issues).

## Resources

- [HDF5 Format Specification](https://docs.hdfgroup.org/documentation/hdf5/latest/_f_m_t3.html)
- [Official HDF5 Library](https://github.com/HDFGroup/hdf5)

## License

MIT — see [LICENSE](LICENSE). Forked from
[scigolib/hdf5](https://github.com/scigolib/hdf5); original work by the
scigolib contributors.
