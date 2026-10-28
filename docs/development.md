# Development Guide

## Setup

### Prerequisites
- Python 3.12+
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
2. **Write a design note** first if it's a big change (kept outside the repo)
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

### Running Examples

Scripts in `examples/` run on synthetic data and are smoke-tested in
`tests/examples/`.

```bash
python examples/quickstart.py
```

## Working with Claude Code

This project is set up for use with Claude Code.

- **Project context:** `CLAUDE.md` (read automatically at session start)
- **Prompts:** `.claude/prompts/` (feature planning, TDD, refactoring templates)
- **Settings:** `.claude/settings.local.json` (personal, not committed)

### Recommended Session Flow

1. Claude Code reads `CLAUDE.md` — includes current work focus
2. Plan work (a design note outside the repo if complex)
3. Implement with TDD approach
4. Run tests after each change
5. Suggest commits (user approves and commits manually)
6. Update `CLAUDE.md` at session end if focus changed

## Docs Organization

- **User docs:** `README.md` (installation, quick start), `docs/` (module refs)
- **Development docs:** `docs/development.md`, `CLAUDE.md`
- **Changelog:** `CHANGELOG.md`

## Sharing before release

Until the public release the repository is private and shared by
invitation. [Pre-release status](prerelease.md) is the page for the
people invited; this section is the maintainer's side.

**Repository.** It lives in the `rshomberg-research` organization
(moved from a personal account, where collaborators always get write
access). Add people as outside collaborators with the **Read** role:
Settings → Collaborators and teams. If it moves again, update `origin`
in each clone, `[project.urls]` in `pyproject.toml`, `repo_url` in
`mkdocs.yml`, and the links in `README.md` and `docs/`
(`grep -rn "github.com/" README.md docs mkdocs.yml pyproject.toml`).

Before inviting anyone, everything on GitHub (every branch, tag and
commit message) is visible to them. Check it holds nothing that should
stay private, with the names to look for in place of the placeholders:

```bash
git fetch origin --tags --force
git log --remotes=origin --tags -p -i -G"<name>|<other name>" --format="%h %s"
git log --remotes=origin --tags -i --grep="<name>" --format="%h %s"
```

**Docs.** The site is public at
<https://rshomberg-research.github.io/sidescanpy/> while the code stays
private. GitHub Pages can't serve a private repository on the
organization's free plan, so the `docs` workflow pushes each build of
`main` into the `sidescanpy/` folder of the public repository
`rshomberg-research/rshomberg-research.github.io`, which Pages serves.
The workflow also keeps every build of `main` and of `v*` tags as the
`sidescanpy-docs` artifact (90 days).

One-time setup, as an organization owner:

```bash
# 1. The public site repository: a redirect to the docs, served by Pages.
gh repo create rshomberg-research/rshomberg-research.github.io --public \
   --description "Public site of rshomberg-research (sidescanpy docs)"
#    push an index.html that redirects to sidescanpy/ and an empty .nojekyll,
#    then: Settings -> Pages -> Deploy from a branch -> main, / (root)

# 2. A deploy key the docs workflow pushes with.
ssh-keygen -t ed25519 -N "" -C "sidescanpy docs deploy" -f docs_deploy_key
gh repo deploy-key add docs_deploy_key.pub --allow-write \
   -R rshomberg-research/rshomberg-research.github.io --title "sidescanpy docs"
gh secret set DOCS_DEPLOY_KEY -R rshomberg-research/sidescanpy < docs_deploy_key
rm docs_deploy_key docs_deploy_key.pub

# 3. Turn publishing on; the next push to main publishes.
gh variable set PUBLISH_DOCS --body true -R rshomberg-research/sidescanpy
```

Everything on the site is public, including the API reference built
from the docstrings (signatures and docstrings, not source), the example
scripts and the development guide. Until the release the pages carry a
`noindex` tag (`overrides/main.html`), so the link works for the people
it is given to but search engines leave it out. At the public release,
enable Pages on this repository instead (same URL), remove `sidescanpy/`
from the site repository, and delete `overrides/` and `custom_dir` in
`mkdocs.yml`.

**Gallery.** `make gallery` re-runs `examples/gallery/` and refreshes
the images in `docs/assets/gallery/` (the README uses two of them). The
scripts download real data, so they are not run in CI; CI only checks
that every dataset name they fetch is registered.

**Test data.** The public fixtures are shared as one zip, through a
link-shared folder that each invited person
is added to. Rebuild it whenever a fixture or `SOURCES.md` changes:

```bash
cd /path/to/fixtures/..       # the directory holding the fixtures folder
mkdir sidescanpy-public-fixtures
cp fixtures/*.jsf fixtures/*.xtf /path/to/sidescanpy/tests/data/public/SOURCES.md \
   sidescanpy-public-fixtures/
zip -qrX sidescanpy-public-fixtures-$(date +%F).zip sidescanpy-public-fixtures
rm -r sidescanpy-public-fixtures
```

Then run `test_public_fixtures.py` against the unzipped copy before
uploading it. Only openly licensed data goes in the zip (see
`SOURCES.md`). Don't mint a Zenodo DOI for the data before the public
release: a published Zenodo record can't be withdrawn, and even a
restricted one shows its metadata publicly.

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