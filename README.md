# SideScanPy

Python tools for ingesting, processing, and analyzing sidescan sonar data.

> **Status:** Early development. APIs may change without notice.
> ~310 unit tests + ~27 integration tests passing.

## What it does

- **Read** multiple raw data formats into validated survey objects
- **Standardize** sonar data into an `xarray`-based schema with explicit
  per-ping coordinates, time axes, and aux streams
- **Persist** surveys to chunked HDF5 (full survey) or NetCDF (single frequency
  channel) with lazy/dask-backed loading
- **Interpolate** auxiliary navigation, attitude, and DVL records onto
  per-ping timestamps
- **Apply** radiometric corrections (time-varying gain) on raw
  backscatter

## Install

```bash
git clone https://github.com/<your-username>/sidescanpy
cd sidescanpy
pip install -e .
```

Requires Python ≥ 3.11.

## Quick example

```python
from sidescanpy.io.jsf import read_jsf, convert
from sidescanpy.processing import interpolate_aux_onto_pings, apply_tvg
from sidescanpy.io.hdf5 import write_survey

# Parse a JSF file into a survey
messages = read_jsf("survey.jsf")
survey = convert(messages, source_files=["survey.jsf"])

# Interpolate aux records onto ping times
survey = interpolate_aux_onto_pings(survey)

# Apply TVG correction to each frequency channel
for fk, channel in survey.freq_channels.items():
    corrected = apply_tvg(channel.dataset)
    survey.add_frequency(fk, corrected)

# Save as a single HDF5 file
write_survey(survey, "survey.ss.h5")
```

## Documentation

- `docs/core.md`
- `docs/processing.md`
- `docs/vis.md`
- `docs/io_hdf5.md`
- `docs/io_netcdf.md`
- `docs/io_sdf.md`
- [`docs/io_jsf.md`](docs/io_jsf.md) — JSF reader, converter, message types
- [`docs/jsf_aux_source_selection.md`](docs/jsf_aux_source_selection.md) — auxiliary stream comparison and per-field source decisions
- [`CHANGELOG.md`](CHANGELOG.md) — version history

More module docs (`core`, `io_netcdf`, `io_hdf5`, `processing`) are forthcoming.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use this package in academic work, please cite:

```
Shomberg, R. (2026). SideScanPy: Python tools for sidescan sonar processing.
University of Rhode Island.
```