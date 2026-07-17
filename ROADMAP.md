# HDF5 Go Library - Development Roadmap

**Current version**: v0.14.0 (see [CHANGELOG.md](CHANGELOG.md) for release history)

## 🎯 Vision

A pure-Go HDF5 library: no CGo, cross-platform, verified against the official
C library. Read support covers the default HDF5 format produced by any 1.x/2.x
release; write output is validated with h5dump/h5diff/h5repack in CI.

## 📅 Planned

### Shipped in v0.15 (this release)

- ✅ LZF write (`WithLZFCompression`, h5py/PyTables compatible)
- ✅ float16 read (`H5T_IEEE_F16LE/BE`)
- ✅ Public compound-write API (`CompoundField`/`NewCompoundType`/
  `CreateCompoundDatasetFromFields`)
- ✅ Soft/external link resolution (`File.Resolve`, `SoftLink`/`ExternalLink`
  objects, cycle detection)
- ✅ Data layout v4/v5 read with all five modern chunk indexes — files
  written under `H5F_LIBVER_LATEST` (h5py `libver="latest"`) now read
- ✅ Complex datatype read (`Dataset.ReadComplex`)

### Remaining gaps (v0.16.x candidates)

- Paged fixed/extensible-array data blocks; v2 B-tree chunk index depth > 1
- float16/complex in attributes and hyperslab selections
- Layout v4 write support (currently read-only)

### v1.0.0 — Long-term support

- API freeze after community validation of the v0.15/v0.16 features.
- Multi-level B-tree v2 (node splits/merges) if dense-group workloads
  demand it — deliberately deferred while trees fit one leaf.

## ❌ Not planned

- **SZIP write** — patent/libaec-encumbered; no pure-Go implementation.
- **BZIP2 write** — stdlib `compress/bzip2` is decompress-only; adding a
  dependency contradicts the zero-dep design. Read support stays.
- **User-configurable B-tree rebalancing** — the C library exposes no such
  API; deletions rebalance automatically. (An earlier speculative
  Smart/Lazy/Incremental rebalancing API was removed in the v0.15 cleanup.)

## 📚 Resources

- [CHANGELOG.md](CHANGELOG.md) — full release history
- [UPSTREAM.md](UPSTREAM.md) — format-spec and C-library version tracking
- [docs/architecture/OVERVIEW.md](docs/architecture/OVERVIEW.md) — internals
- [GitHub Discussions](https://github.com/scigolib/hdf5/discussions) — support
