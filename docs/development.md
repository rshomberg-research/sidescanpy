# Development Guide

## Setup

### Prerequisites
- Python 3.14+
- `uv` for package management (recommended) or `pip`
- Git

### Initial Setup

```bash
# Clone repo
git clone <repo-url>
cd sidescanpy

# Create virtual environment with uv
uv venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Verify installation
pytest tests/unit/ -v
```

## Project Structure

See `CLAUDE.md` for detailed architecture overview.

## Development Workflow

### Making Changes

1. **Read `CLAUDE.md`** — Project context and current focus
2. **Create/update planning doc** if it's a big change (`docs/planning/`)
3. **Write tests first** — TDD is preferred
4. **Implement** — Small, focused changes
5. **Run tests** — `pytest tests/unit/ -v`
6. **Commit** — Conventional Commits format
7. **Update `CLAUDE.md`** — If current focus or architecture changed

### Testing

```bash
# Fast unit tests
pytest tests/unit/ -v

# Integration tests (requires fixtures)
pytest tests/integration/ -v --slow

# Coverage
pytest tests/unit/ --cov=sidescanpy --cov-report=html

# Specific test
pytest tests/unit/grids/test_resample.py::TestResampleChannelToGrid3DValidation -v
```

### Code Style

- **Type hints:** Use for public APIs
- **Docstrings:** NumPy style for functions/classes
- **Line length:** 100 characters (soft limit)
- **Imports:** Standard library first, then third-party, then local

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`

**Scopes:** Use hyphenated module paths (e.g., `io/sdf`, `grids/resample`)

**Examples:**
- `feat(grids): add resample_channel_to_grid for 3D data`
- `fix(io/jsf): correct navigation timestamp parsing`
- `test(grids): add fixtures for 3D channel data`
- `docs(readme): update installation instructions`

### Running Scripts

Scripts in `scripts/` are CLI tools and demos.

```bash
# Example: minimal mapping demo
python scripts/minimal_map_demo.py data/sample_jsf_georef.ss.h5
```

## Working with Claude Code

This project is set up for use with Claude Code.

- **Project context:** `CLAUDE.md` (read automatically at session start)
- **Prompts:** `.claude/prompts/` (feature planning, TDD, refactoring templates)
- **Settings:** `.claude/settings.local.json` (personal, not committed)

### Recommended Session Flow

1. Claude Code reads `CLAUDE.md` — includes current work focus
2. Plan work (write to `docs/planning/` if complex)
3. Implement with TDD approach
4. Run tests after each change
5. Suggest commits (user approves and commits manually)
6. Update `CLAUDE.md` at session end if focus changed

## Docs Organization

- **User docs:** `README.md` (installation, quick start), `docs/` (module refs)
- **Development docs:** `docs/DEVELOPMENT.md`, `CLAUDE.md`
- **Planning docs:** `docs/planning/`
- **Changelog:** `CHANGELOG.md`

## Release Process

Rough flow:
1. All work committed on feature branch
2. All tests passing
3. `CHANGELOG.md` updated
4. Version bumped in `pyproject.toml`
5. Merge to main
6. Tag: `git tag v0.X.0`
7. Update `CLAUDE.md` — new "Current Work" state
8. Push tag: `git push origin v0.X.0`

### Pre-tag checklist

Before running `git tag vX.Y.Z`:

**Documentation**
- [ ] `CHANGELOG.md` has a dated section for the new version
- [ ] New features documented in `docs/`
- [ ] `README.md` reflects supported formats and current capabilities
- [ ] `CLAUDE.md` "Current Work Focus" updated to new state

**Code quality**
- [ ] No stray `print()` debug statements
- [ ] Logging levels appropriate (DEBUG / INFO / WARNING)
- [ ] TODOs related to this version resolved or explicitly deferred

**Tests**
- [ ] `pytest tests/unit -v` all green
- [ ] `pytest tests/integration -v --slow` all green
- [ ] No accidentally-skipped tests

**Build & packaging**
- [ ] Version bumped in `pyproject.toml`
- [ ] Version bumped in `sidescanpy/__init__.py` if `__version__` exists
- [ ] `python -m build` succeeds
- [ ] Fresh-venv install of the wheel works

**Git hygiene**
- [ ] All work committed
- [ ] Working tree clean
- [ ] Branch up to date with main