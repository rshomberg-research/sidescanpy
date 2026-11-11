# Contributing

How to set up sidescanpy for development, run its checks, and where things
go. Bug reports and proposals are welcome as
[GitHub issues](https://github.com/rshomberg-research/sidescanpy/issues);
for a reader problem see [Reporting a problem](limitations.md#reporting-a-problem).
To add a reader for a new sonar format, see
[Adding a source format](sources/new_source.md).

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
| `make docs` | this site in `docs/_build/html`, running the example scripts (downloads about 1.3 GB the first time, 1.9 GB on disk; the two published-mosaic examples account for most of it); any warning, such as a malformed docstring or a broken cross-reference, fails the build |
| `make docs-fast` | the same without running the example scripts (no figures) |
| `make docs-code` | runs the Python blocks of [Getting started](getting_started.md) and the quickstart on the home page, and checks the output Getting started shows (downloads the 41 MB recording they use if needed; CI runs it after `make docs`) |
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

## Test data

The repository holds code only, no survey data.

- **Examples.** The [worked examples](examples/index.rst) download public
  recordings with `sidescanpy.datasets.fetch()`; see
  [Example data](example_data.md). `examples/project_template/` runs on
  synthetic data in `tests/examples`. The gallery and Theory scripts run
  during `make docs` (locally and in CI), and the unit suite checks
  offline that every dataset name a script fetches is registered.
- **Public fixtures.** `tests/unit/io/test_public_fixtures.py` runs on
  small cuts of openly licensed recordings (EdgeTech JSF, and XTF from
  three writers), listed with source, licence and checksum in
  [`tests/data/public/SOURCES.md`](https://github.com/rshomberg-research/sidescanpy/blob/main/tests/data/public/SOURCES.md).
  They are kept outside the repository; point `SIDESCANPY_PUBLIC_DATA`
  at their folder to run those tests, which skip without it.
