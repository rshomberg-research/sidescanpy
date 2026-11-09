# Development

## Setup

You need Python 3.12 or newer, [uv](https://docs.astral.sh/uv/) and git.

```bash
git clone https://github.com/rshomberg-research/sidescanpy.git
cd sidescanpy
uv sync --all-extras
make test
```

`uv sync --all-extras` builds `.venv/` with every optional dependency,
so the plotting, GeoTIFF and docs code is tested too.

| Command | Runs |
|---|---|
| `make test` | the unit tests, with `PYTHONPATH` cleared |
| `make lint` | ruff |
| `make docs` | this site in `docs/_build/html`, running the example scripts (downloads about 140 MB the first time); any warning, such as a malformed docstring or a broken cross-reference, fails the build |
| `make docs-fast` | the same without running the example scripts (no figures) |
| `make docs-serve` | builds the site, then serves it at <http://127.0.0.1:8000> |
| `make docs-clean` | removes the build and the generated API and example pages |

`uv run pytest tests/integration --slow` runs the integration tests,
which need real survey files that are not in the repository and skip
without them. The public test fixtures are listed in
`tests/data/public/SOURCES.md`; set `SIDESCANPY_PUBLIC_DATA` to the
folder holding them to run `tests/unit/io/test_public_fixtures.py`.

## Layout

| Package | Holds |
|---|---|
| `core` | the data model, schema accessors, processing history |
| `io` | the JSF, SDF and XTF readers, `.ss.h5` and NetCDF I/O, target catalogs |
| `processing` | correction steps, slicing, bottom and noise detection, segments |
| `analysis` | measurements that return result dataclasses |
| `grids` | gridding and GeoTIFF output |
| `targets` | target catalogs and matching targets to pings |
| `export` | image windows, chips and COCO/YOLO datasets |
| `viz` | plotting helpers |
| `cli` | the `sidescan-*` commands |

Imports go one way, `core ← io ← processing ← analysis ← grids ← targets
← export ← viz ← cli`, and only `viz` imports matplotlib. A unit test
(`tests/unit/test_layering.py`) enforces this.

## Conventions

- **Library, not workflow.** Library functions take and return datasets
  and do one thing. Orchestration, run-everything scripts and multi-panel
  figures belong in `examples/` or in your own project.
- **Tests first** for new behaviour, on synthetic data with real shapes
  (three dimensions, two channels, realistic sample counts);
  `tests/unit/synthetic.py` has the builders. Never commit survey data.
- **Docstrings** are numpydoc and are the API reference.
- **Logging.** Every module has `logger = logging.getLogger(__name__)`
  and logs only at DEBUG and WARNING. No `print()` in library code.
- **New schema variables** go in `core/schema.py` before a step produces
  them.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  with the module path as scope: `feat(io/sdf): ...`,
  `fix(grids/resample): ...`, `docs(processing): ...`.
- **Versions** follow SemVer. Until 1.0, a minor release may break the
  API or the `.ss.h5` format; `CHANGELOG.md` lists every break.

## Publishing

**Docs.** The site is at
<https://rshomberg-research.github.io/sidescanpy/>. The `docs` workflow
pushes each build of `main` into the `sidescanpy/` folder of the
repository `rshomberg-research/rshomberg-research.github.io`, which
GitHub Pages serves. (This setup dates from when the code repository
was private; Pages on this repository at the same URL would replace it.)
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

**Gallery.** The example pages are built by
[sphinx-gallery](https://sphinx-gallery.github.io/), which runs every
`examples/gallery/plot_*.py` script during the docs build, locally and
in CI, and shows its code, printed output and figures. The scripts
download real data; CI caches it between runs. The README shows two of
the figures from `docs/assets/gallery/`; `make readme-images` refreshes
them from the last `make docs`. The unit suite checks offline that every
dataset name a script fetches is registered.

**Test data.** The public fixtures are kept outside the repository and
handed out as one zip. Rebuild it whenever a fixture or `SOURCES.md`
changes:

```bash
cd /path/to/fixtures/..       # the directory holding the fixtures folder
mkdir sidescanpy-public-fixtures
cp fixtures/*.jsf fixtures/*.xtf /path/to/sidescanpy/tests/data/public/SOURCES.md \
   sidescanpy-public-fixtures/
zip -qrX sidescanpy-public-fixtures-$(date +%F).zip sidescanpy-public-fixtures
rm -r sidescanpy-public-fixtures
```

Then run `test_public_fixtures.py` against the unzipped copy before
sharing it. Only openly licensed data goes in the zip (see
`SOURCES.md`). A published Zenodo record can't be withdrawn, so check
the zip before archiving it there.

## Releasing

1. Update `CHANGELOG.md` with a dated section and the version in
   `pyproject.toml`.
2. Run `make test`, `make lint`, `make docs` and the integration tests
   on real data; all pass.
3. Build and check the wheel: `uv build`, then install it in a fresh
   environment and run [Getting started](getting_started.md).
4. Merge to `main`, tag `vX.Y.Z` and push the tag.
