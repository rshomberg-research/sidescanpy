# Install

sidescanpy needs Python 3.12 or newer. It is pure Python on top of
numpy, scipy, pandas, xarray, h5py/h5netcdf, dask and pyproj, all of
which ship wheels for the common platforms. The test suite runs on Linux
with Python 3.12; other platforms are expected to work but are not yet
tested in CI.

<!-- TODO(release): replace the git installs below with
     `pip install "sidescanpy[viz]"` once it is on PyPI. -->

sidescanpy is not on PyPI yet. Until it is, install from the GitHub
repository, pinned to a release tag:

```bash
pip install "sidescanpy[viz] @ git+https://github.com/rshomberg-research/sidescanpy.git@v0.9.0"
```

With [uv](https://docs.astral.sh/uv/), in a project:

```bash
uv add "sidescanpy[viz] @ git+https://github.com/rshomberg-research/sidescanpy.git@v0.9.0"
```

`uv add` records the pin in `pyproject.toml` and `uv.lock`. Use `@main`
in place of the tag for the latest fixes.

## Extras

The core install converts, corrects, grids, analyzes and exports
datasets. Plots, basemaps and GeoTIFF output are optional:

| Extra | Adds | Needed for |
|---|---|---|
| `viz` | matplotlib | `sidescanpy.viz`, figures from `sidescan-export` and `sidescan-analyze` |
| `maps` | cartopy | land and locator basemaps behind georeferenced plots |
| `geo` | rioxarray | `grids.to_geotiff`, `sidescan-export grid --geotiff` |
| `docs` | Sphinx, the pydata theme, MyST, numpydoc, sphinx-gallery | building this site |
| `dev` | pytest, ruff | running the tests and linters |

Combine them in the brackets: `sidescanpy[viz,maps,geo]`. The `maps`
extra includes matplotlib, so it implies `viz`.

## Check the install

```bash
python -c "import sidescanpy; print(sidescanpy.__version__)"
sidescan-convert --help
```

The first line prints the installed version (`0.8.0`). The second
confirms the command-line tools are on your `PATH`.

## Working on the code

Clone the repository and let uv build the environment with every extra:

```bash
git clone https://github.com/rshomberg-research/sidescanpy.git
cd sidescanpy
uv sync --all-extras
make test
```

See [Development](development.md) for the test, lint and docs commands.

Next: [Getting started](getting_started.md).
