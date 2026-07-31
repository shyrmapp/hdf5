# Security Policy

## Supported Versions

The project is pre-1.0. Security fixes land on the latest minor release only;
there are no backports.

| Version | Supported |
| ------- | --------- |
| 0.17.x  | ✅        |
| < 0.17  | ❌        |

## Reporting a Vulnerability

**Do not open a public issue for a security vulnerability.**

Report it via a
[private security advisory](https://github.com/shyrmapp/hdf5/security/advisories/new).

Please include:

- What the vulnerability is and its impact (DoS, information disclosure, ...)
- Steps to reproduce — a triggering `.h5` file is the most useful thing you
  can attach
- Which versions are affected
- A suggested fix, if you have one

Expect an acknowledgement within 72 hours and an initial assessment within a
week. Disclosure timing is coordinated with you, and you are credited in the
advisory unless you prefer otherwise.

## Threat Model

This library parses untrusted binary input. Every field in an HDF5 file —
sizes, offsets, counts, addresses — is attacker-controlled. The realistic
attacks are memory exhaustion, unbounded recursion, and out-of-range reads
driven by crafted size fields.

The library aims to **return an error, never panic**, on any input. A panic on
malformed input is a bug worth reporting.

### What is enforced

Size limits (`internal/utils`):

| Limit                  | Value  |
| ---------------------- | ------ |
| `MaxChunkSize`         | 1 GiB  |
| `MaxAttributeSize`     | 64 MiB |
| `MaxStringSize`        | 16 MiB |
| `MaxHyperslabElements` | 1e9    |
| Whole-dataset read     | 1 GiB, configurable via `WithMaxReadBytes` |

Other defenses:

- Overflow-checked multiplication (`SafeMultiply`) before every buffer
  allocation derived from a file-supplied size
- Signature and version validation on superblocks, object headers and heaps
- Hyperslab bounds validation, including overflow in stride arithmetic
- Soft-link cycle detection, capped at 32 hops (`maxSoftLinkHops`)
- Datatype nesting capped at 32 levels (`maxDatatypeNesting`)
- Decompression bounded by the chunk's own uncompressed size. An HDF5 chunk is
  fixed-size — edge chunks included — so the reader knows exactly how many
  bytes a chunk must expand to, and a stream producing more is rejected mid-
  decompression rather than buffered. This bounds compression bombs by the
  file's own declared geometry rather than by a guessed ratio.
- Whole-dataset reads size-checked before allocating, against the memory the
  call would need rather than the bytes on disk (reading int8 through `Read()`
  yields float64, 8x wider). Go reports allocation failure as an unrecoverable
  `fatal error: runtime: out of memory`, so this cannot be left to fail late
  the way it can in C, Java or Python.
- Contiguous datasets additionally validated against actual file length: a
  header claiming more bytes than the file holds is rejected before allocating
- Fletcher32 and metadata checksum validation where the format provides it

### Testing

- `FuzzRead` covers the untrusted-input read surface — open, walk, and every
  public read path — seeded with the official HDF5 test corpus. It runs on
  every CI build; the invariant is "error, never panic".
- `gosec` plus ~35 other linters run in CI.
- Written files are validated against the official C library
  (`h5dump`/`h5diff`/`h5repack`) in CI.
- The official HDF5 test corpus (433 files) is read in CI, with
  [documented exclusions](testdata/hdf5_official/KNOWN_FAILURES.md).

### Known gaps

These are real and unmitigated. Handle them yourself if they matter to you:

- **No global object-count limit** during traversal. A file declaring an
  enormous number of groups or datasets will allocate accordingly.
- **Group traversal is not depth-limited.** Only soft-link chains and datatype
  nesting are capped.
- **A single `File` is not safe for concurrent use.** Separate `File`
  instances are independent.

### Handling untrusted files

Set a read limit sized to your data, bound the file before you open it, and
check every error — a parse failure may be the only signal that a file is
hostile.

`WithMaxReadBytes` is the single most important thing to set here. A dataset's
declared size is a number taken from the file: a 6 KiB file can claim to hold
tens of gigabytes, and Go cannot recover from the resulting allocation. Size
the limit from the largest dataset you expect, with headroom — not from the
machine's RAM, which would make the same file behave differently per host and
would let a bigger host absorb a bigger hostile claim.

```go
info, err := os.Stat(name)
if err != nil || info.Size() > maxAllowedSize {
    return errors.New("rejected: file too large")
}

// Sized to the workload, not to the host.
f, err := hdf5.Open(name, hdf5.WithMaxReadBytes(256<<20))
if err != nil {
    return fmt.Errorf("open failed: %w", err) // may indicate a crafted file
}
defer f.Close()

obj, err := f.Resolve("/data")
if err != nil {
    return fmt.Errorf("resolve failed: %w", err)
}
ds, ok := obj.(*hdf5.Dataset)
if !ok {
    return errors.New("/data is not a dataset")
}

values, err := ds.Read()
if err != nil {
    return fmt.Errorf("read failed: %w", err)
}
```

HDF5 object names are arbitrary strings and may contain `..` or absolute
paths. This library never touches the filesystem based on them, but if you use
a name to build a path, sanitize it:

```go
safe := filepath.Base(ds.Name()) // the name comes from the file
```

## Disclosure History

No vulnerabilities have been reported against this library to date.
