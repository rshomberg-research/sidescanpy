# SideScanPy

Python library for processing sidescan sonar data from Edgetech JSF and Klein SDF surveys.

## Features

- **File I/O:**
  - Read Edgetech JSF (`.jsf`) format
  - Read Klein SDF (`.sdf`) format
  - HDF5 (`.ss.h5`) round-trip serialization with `xarray`
  - Single-channel NetCDF export

- **Processing:**
  - Aux stream interpolation (nav, attitude, DVL) onto ping timestamps
  - Time-varying gain (TVG) correction
  - Slant range → ground range → per-sample depth calculation
  - Per-sample lat/lon georeferencing

- **Visualization:**
  - Port + starboard waterfall displays
  - Multiple x-axis modes (sample, ground range, across-track)
  - Multiple y-axis modes (ping, time, along-track)
  - Batch PNG export

- **Gridding** (v0.6.0, in progress):
  - Geographic and projected grids (WGS84, UTM)
  - Multiple resampling methods (histogram, linear, nearest, cubic)
  - Statistical aggregations (mean, median, max)

## Installation

### Requirements
- Python 3.14+
- `uv` (recommended) or `pip`

### From Source

```bash
git clone <repo-url>
cd sidescanpy
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Development Install

```bash
uv pip install -e ".[dev]"
pytest tests/unit/ -v  # Verify installation
```

## Quick Start

### Load a JSF Survey

```python
from sidescanpy.io.jsf import read_jsf
from sidescanpy.io.jsf.converter import convert

# Read raw JSF messages
messages = read_jsf("survey.jsf")

# Convert to SideScanSurvey
survey = convert(messages, survey_name="my_survey")

print(survey)
# SideScanSurvey(name='my_survey', n_frequencies=2, nav=1234, att=567, dvl=890, sensors=1)
```

### Process a Survey

```python
from sidescanpy.processing.interpolate import interpolate_aux_onto_pings
from sidescanpy.processing.radiometric import apply_tvg
from sidescanpy.processing.geometric import compute_geometric_ranges
from sidescanpy.processing.georeferencing import compute_georeferencing

# Full pipeline
survey = interpolate_aux_onto_pings(survey)
survey = apply_tvg(survey)
survey = compute_geometric_ranges(survey)
survey = compute_georeferencing(survey)
```

### Save & Load HDF5

```python
from sidescanpy.io.hdf5 import write_survey, read_survey

# Write
write_survey(survey, "survey.ss.h5")

# Read
survey = read_survey("survey.ss.h5")
```

### Visualize

```python
import matplotlib.pyplot as plt
from sidescanpy.viz import plot_waterfall

channel = survey.get_frequency("hz_1495000")
fig, ax = plt.subplots(figsize=(14, 6))
plot_waterfall(
    channel.dataset,
    start=0,
    stop=500,
    x_axis="ground_range",
    y_axis="along_track",
    cmap="copper",
    ax=ax,
)
plt.show()
```

### CLI Tools

This library is a pure Python API — it doesn't ship its own CLI scripts.
The batch-conversion/visualization CLI tools that used to live in this
repo's own `scripts/` directory (`convert.py`, `batch_convert_and_plot.py`,
`export_waterfall_series.py`, `visualization_demo.py`, and others) moved
out to a downstream consumer project (`project-a`, as of 2026-07-07) —
see that project's own docs for usage. They're straightforward to
reproduce against any `sidescanpy` install using the Quick Start examples
above as a starting point.

## Architecture

See `CLAUDE.md` for detailed architecture and conventions.

Top-level structure:

```
sidescanpy/
├── core/         # Schema validation, survey containers
├── io/           # File format readers/writers (JSF, SDF, HDF5, NetCDF)
├── processing/   # Signal processing pipeline
├── grids/        # Gridding and resampling (v0.6.0)
└── viz/          # Visualization utilities
```

## Documentation

- **User docs:** This README + `docs/`
- **Development:** `docs/DEVELOPMENT.md`
- **Architecture:** `CLAUDE.md`
- **Changelog:** `CHANGELOG.md`
- **Planning:** `docs/planning/`

Module documentation:
- `docs/core.md` — Schema and survey containers
- `docs/io_jsf.md` — Edgetech JSF format
- `docs/io_sdf.md` — Klein SDF format
- `docs/io_hdf5.md` — `.ss.h5` HDF5 layout
- `docs/io_netcdf.md` — NetCDF export
- `docs/processing.md` — Processing pipeline
- `docs/vis.md` — Visualization (⚠️ typo: should be `viz.md`)
- `docs/jsf_aux_source_selection.md` — JSF aux stream selection

## Testing

```bash
# Fast unit tests (~30s)
pytest tests/unit/ -v

# Integration tests (requires real data fixtures)
pytest tests/integration/ -v --slow

# With coverage
pytest tests/unit/ --cov=sidescanpy --cov-report=html
```

Test fixtures for real data are not committed to git due to file size limits.
See `data/README.md` for details.

## License

MIT — see `LICENSE`

## Contact

Russ Shomberg <rshomberg@uri.edu>
University of Rhode Island