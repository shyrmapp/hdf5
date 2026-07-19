package hdf5

import (
	"fmt"
	"path/filepath"
	"strings"
)

// maxSoftLinkHops caps how many soft links Resolve will follow in a chain.
const maxSoftLinkHops = 32

// SoftLink represents an unresolved symbolic link inside an HDF5 file.
// Soft links store a target path that is resolved on access; use
// File.Resolve to follow them. Dangling links (targets that do not exist)
// are valid and surface as *SoftLink children.
type SoftLink struct {
	name   string
	target string
}

// Name returns the link's name.
func (s *SoftLink) Name() string {
	return s.name
}

// Target returns the target path the link points to within the file.
func (s *SoftLink) Target() string {
	return s.target
}

// ExternalLink represents a link to an object in another HDF5 file.
// File.Resolve stops at external links and returns them; callers resolve
// them explicitly with ExternalLink.Resolve.
type ExternalLink struct {
	name       string
	fileName   string
	objectPath string
}

// Name returns the link's name.
func (e *ExternalLink) Name() string {
	return e.name
}

// FileName returns the external HDF5 file name the link references.
func (e *ExternalLink) FileName() string {
	return e.fileName
}

// ObjectPath returns the object path within the external file.
func (e *ExternalLink) ObjectPath() string {
	return e.objectPath
}

// Resolve opens the external file and resolves the object path within it.
// Relative file names are resolved against dir (typically the directory of
// the file containing the link). The returned *File must be closed by the
// caller; it stays open so the returned Object remains readable.
func (e *ExternalLink) Resolve(dir string) (*File, Object, error) {
	path := e.fileName
	if !filepath.IsAbs(path) {
		path = filepath.Join(dir, path)
	}

	f, err := Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open external file %q: %w", path, err)
	}

	obj, err := f.Resolve(e.objectPath)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("failed to resolve %q in external file %q: %w", e.objectPath, path, err)
	}
	return f, obj, nil
}

// Resolve walks to the object at the given absolute path, following soft
// links along the way (with cycle detection, capped at 32 hops). Soft link
// targets must be absolute paths; relative targets return an error.
//
// Resolve stops at external links and returns the *ExternalLink: the caller
// resolves those explicitly with ExternalLink.Resolve, which returns the
// opened external file for the caller to close.
func (f *File) Resolve(path string) (Object, error) {
	if !strings.HasPrefix(path, "/") {
		return nil, fmt.Errorf("path must be absolute (start with '/'), got %q", path)
	}
	return f.resolvePath(path, make(map[string]bool), 0)
}

// resolvePath is the recursive worker behind Resolve. visited tracks the
// paths of soft links already followed (cycle detection); hops bounds the
// total chain length.
func (f *File) resolvePath(path string, visited map[string]bool, hops int) (Object, error) {
	if hops > maxSoftLinkHops {
		return nil, fmt.Errorf("soft link chain exceeds %d hops", maxSoftLinkHops)
	}

	trimmed := strings.Trim(path, "/")
	if trimmed == "" {
		return f.root, nil
	}
	segments := strings.Split(trimmed, "/")

	var current Object = f.root
	for i, segment := range segments {
		group, ok := current.(*Group)
		if !ok {
			return nil, fmt.Errorf("object %q is not a group", "/"+strings.Join(segments[:i], "/"))
		}

		child := findChild(group, segment)
		if child == nil {
			return nil, fmt.Errorf("object %q not found", "/"+strings.Join(segments[:i+1], "/"))
		}

		switch obj := child.(type) {
		case *SoftLink:
			// Re-resolve from the root at the link target (plus any
			// remaining segments).
			target, err := followSoftLink(obj, segments, i, visited)
			if err != nil {
				return nil, err
			}
			return f.resolvePath(target, visited, hops+1)
		case *ExternalLink:
			if i == len(segments)-1 {
				return obj, nil
			}
			return nil, fmt.Errorf("cannot traverse external link %q: resolve it with ExternalLink.Resolve first",
				"/"+strings.Join(segments[:i+1], "/"))
		default:
			current = child
		}
	}

	return current, nil
}

// findChild returns the direct child of group with the given name, or nil.
func findChild(group *Group, name string) Object {
	for _, c := range group.Children() {
		if c.Name() == name {
			return c
		}
	}
	return nil
}

// followSoftLink validates a soft link encountered at segments[index] and
// returns the absolute path where resolution continues: the link target with
// any remaining segments appended.
func followSoftLink(link *SoftLink, segments []string, index int, visited map[string]bool) (string, error) {
	linkPath := "/" + strings.Join(segments[:index+1], "/")
	if visited[linkPath] {
		return "", fmt.Errorf("soft link cycle detected at %q", linkPath)
	}
	visited[linkPath] = true

	target := link.Target()
	if !strings.HasPrefix(target, "/") {
		return "", fmt.Errorf("relative soft link target %q not supported", target)
	}
	if rest := strings.Join(segments[index+1:], "/"); rest != "" {
		target = strings.TrimSuffix(target, "/") + "/" + rest
	}
	return target, nil
}
