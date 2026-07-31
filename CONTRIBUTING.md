# Contributing to HDF5 Go Library

Thank you for considering contributing to the HDF5 Go Library! This document outlines the development workflow and guidelines.

## Git Workflow (GitHub Flow)

This project uses GitHub Flow - a simplified branching model.

### Branch Structure

```
main                 # Production-ready code (tagged releases)
  ├─ feature/*       # New features
  ├─ fix/*           # Bug fixes
  └─ release/*       # Release preparation (for version updates)

legacy               # Historical development (read-only archive)
```

### Branch Purposes

- **main**: Production-ready code. Protected branch - all changes via PRs only.
- **feature/\***: New features. Branch from `main`, merge via PR with squash.
- **fix/\***: Bug fixes. Branch from `main`, merge via PR with squash.
- **release/\***: Release preparation. Used for version bumps and changelog updates.

### Workflow Commands

#### Starting a New Feature

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/my-new-feature

# Work on your feature (multiple commits OK)...
git add .
git commit -m "feat: add my new feature"

# Push and create PR
git push origin feature/my-new-feature
gh pr create --title "feat: add my new feature" --body "Description..."

# Wait for CI green, then merge with squash
gh pr checks --watch
gh pr merge --squash --delete-branch
```

#### Fixing a Bug

```bash
# Create fix branch from main
git checkout main
git pull origin main
git checkout -b fix/issue-123

# Fix the bug...
git add .
git commit -m "fix: resolve issue #123"

# Push and create PR
git push origin fix/issue-123
gh pr create --title "fix: resolve issue #123" --body "Fixes #123"

# Wait for CI green, then merge with squash
gh pr checks --watch
gh pr merge --squash --delete-branch
```

#### Creating a Release

```bash
# Create release branch from main
git checkout main
git pull origin main
git checkout -b release/v1.0.0

# Update version numbers, CHANGELOG, etc.
git add .
git commit -m "chore: prepare release v1.0.0"

# Push and create PR
git push origin release/v1.0.0
gh pr create --title "Release v1.0.0" --body "Release notes..."

# Wait for CI green, then merge with squash
gh pr checks --watch
gh pr merge --squash --delete-branch

# After PR merged, create tag on main
git checkout main
git pull origin main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin --tags
```

#### Hotfix (Critical Production Bug)

Same as regular bug fix - use `fix/` branch and PR workflow.
Hotfixes follow the same process since we work directly with main.

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (build, dependencies, etc.)
- **perf**: Performance improvements

### Examples

```bash
feat: add support for compound datatypes
fix: correct endianness handling in superblock
docs: update README with new examples
refactor: simplify datatype parsing logic
test: add tests for chunked dataset reading
chore: update golangci-lint configuration
```

## Code Quality Standards

### Before Committing

```bash
gofmt -w -s .                      # format
test -z "$(gofmt -l .)"            # ...and verify nothing is left unformatted
golangci-lint run                  # lint
go test ./...                      # build every package and run tests
```

As a single pre-push gate:

```bash
test -z "$(gofmt -l .)" && golangci-lint run && go test ./...
```

### Pull Request Requirements

- [ ] Code is formatted (`gofmt -l .` prints nothing)
- [ ] Linter passes (`golangci-lint run`)
- [ ] All tests pass (`go test ./...`)
- [ ] New code has tests (minimum 70% coverage)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] No sensitive data (credentials, tokens, etc.)

## Development Setup

### Prerequisites

- Go 1.26 or later
- golangci-lint (see [installation](https://golangci-lint.run/welcome/install/))
- Python 3 with h5py and numpy (optional, for test file generation)

```bash
pip install h5py numpy
```

### Running Tests

```bash
go test ./...                                  # all tests
go test -race ./...                            # with race detector
go test -bench=. -benchmem ./...               # benchmarks

# Coverage. Use -coverpkg=./... for a whole-repo number: without it each
# package is only credited for its own tests, which undercounts internal/.
go test -coverpkg=./... -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

## Project Structure

```
hdf5/
├── .golangci.yml         # Linter configuration
├── cmd/                  # Command-line utilities
├── docs/                 # Documentation
├── examples/             # Usage examples
├── internal/             # Internal implementation
│   ├── core/            # Core HDF5 structures
│   ├── structures/      # HDF5 data structures
│   ├── writer/          # File writing, allocation, filters
│   └── utils/           # Overflow-checked arithmetic, size limits
├── testdata/            # Test HDF5 files
├── file.go              # Public API - File operations
├── group.go             # Public API - Groups & Datasets
└── README.md            # Main documentation
```

## Adding New Features

1. Check if issue exists, if not create one
2. Discuss approach in the issue
3. Create feature branch from `main`
4. Implement feature with tests
5. Update documentation
6. Run quality checks (`test -z "$(gofmt -l .)" && golangci-lint run && go test ./...`)
7. Create pull request to `main`
8. Wait for CI green and code review
9. Address feedback
10. Merge with squash when approved

## Code Style Guidelines

### General Principles

- Follow Go conventions and idioms
- Write self-documenting code
- Add comments for complex logic
- Keep functions small and focused
- Use meaningful variable names

### Naming Conventions

- **Public types/functions**: `PascalCase` (e.g., `ReadSuperblock`)
- **Private types/functions**: `camelCase` (e.g., `readSignature`)
- **Constants**: `PascalCase` with context prefix (e.g., `Version0`)
- **Test functions**: `Test*` (e.g., `TestReadSuperblock`)

### Error Handling

- Always check and handle errors
- Use `utils.WrapError(context, err)` to add context
- Never ignore errors
- Validate inputs

### Testing

- Use `testify/require` for assertions
- Test both success and error cases
- Use table-driven tests when appropriate
- Mock external dependencies

## Getting Help

- Check existing issues and discussions
- Read documentation in `docs/`
- Review architecture documentation in `docs/architecture/`
- Ask questions in GitHub Issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the HDF5 Go Library!** 🎉
