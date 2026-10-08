# HDF5 IO

Read and write full :class:`SideScanSurvey` objects as `.ss.h5` files.
A single file holds all frequency channels and auxiliary records with a
grouped layout that supports lazy/chunked loading.

For single-frequency channels in standalone NetCDF, see
[`docs/io_netcdf.md`](io_netcdf.md). For the schema each group conforms
to, see [`docs/core.md`](core.md).

## Entry points

| Function | Use case |
|---|---|
| `write_survey(survey, path)` | Write a survey to disk. |
| `read_survey(path)` | Read back as a validated survey. |
| `read_survey(path, chunks={"ping": N})` | Lazy loading for frequency channels. |

## File layout

```
survey.ss.h5
├── /                            ← root attrs: survey_name,
│                                  source_format, source_files_json
├── /frequencies/
│   ├── hz_547500/                ← SideScanFrequencyChannel.dataset
│   └── hz_1495000/
└── /aux_streams/
    ├── navigation/               ← always written (empty if unset)
    ├── attitude/                 ← always written (empty if unset)
    ├── dvl/                      ← always written (empty if unset)
    └── sensors/                  ← only if any sensors present
        └── pressure/
```

Group constants are defined in `sidescanpy.core.schema` (`HDF5_GROUP_*`)
and used uniformly by reader and writer.

## Conventions

- **Engine.** `h5netcdf`.
- **Survey metadata.** Stored in root-level attrs:
  - `survey_name` (string)
  - `source_format` (string, e.g. `"JSF"`)
  - `source_files_json` (JSON-encoded list of paths)
- **Aux records always written.** A survey with no nav messages still
  writes an empty navigation group. This keeps the file structure
  consistent across surveys regardless of source content.
- **Sensors group is conditional.** If a survey has zero sensors, the
  `aux_streams/sensors/` parent group is omitted entirely. The reader
  handles its absence.
- **Encoding.** Every frequency-channel variable with two or more
  dimensions is written with gzip level 4 + shuffle, chunked along
  `ping`. float64 variables are stored as float32, except latitude and
  longitude (float32 resolves only ~0.4 m there). Per-ping 1-D fields are
  uncompressed. Files from 0.5.0 that also carry `backscatter_raw` still
  read; `correct_file` drops that variable on rewrite.
- **Chunking on disk.** `HDF5_CHUNK_PING = 256` along the ping axis for
  frequency channel data.
- **Lazy loading.** `chunks={"ping": N}` on read returns dask-backed
  arrays for frequency channel data. Aux records always load eagerly
  (they're tiny).

## Round-trip guarantee

After `write_survey(survey, path); read_survey(path)`, the loaded survey
matches the original:

- Same frequency channels (validated as `SideScanFrequencyChannel`)
- Same aux records (`xr.testing.assert_identical`)
- Same survey metadata (`survey_name`, `source_format`, `source_files`)
- Same processing history per channel
- Bit-exact backscatter values, including NaN and dtype preservation

Enforced by ~57 unit tests across `tests/unit/io/hdf5/test_writer.py`,
`test_reader.py`, and `test_round_trip.py`, plus integration tests
against real JSF data.

## Performance notes

- For typical surveys (10k pings × 2 channels × 2000 samples), file
  sizes after gzip compression are roughly 30-50% of raw payload size.
- Lazy reads with `chunks={"ping": 256}` align with on-disk chunks for
  zero-overhead access. Smaller chunk sizes work but emit a warning
  about misaligned reads.
- Aux records (typically a few KB to a few MB) load eagerly and don't
  affect lazy-loading performance for the main backscatter arrays.

## Inspecting files manually

For debugging or quick inspection without loading into Python:

```bash
h5ls survey.ss.h5                   # List top-level groups
h5ls -r survey.ss.h5                # Recursive listing
h5dump -A survey.ss.h5              # Root attributes only
h5dump -H survey.ss.h5/frequencies  # Show structure of one group
```

## Future work

- Add a multi-survey container format (multiple surveys in one file)
  for mosaicking workflows.
- Consider Zarr as an alternate backend for cloud-friendly access
  (S3, GCS). Same xarray semantics, different storage.
- Document the on-disk schema version and add a `schema_version` root
  attribute. Useful for forward-compat as the schema evolves.
- Consider compression tuning per dataset based on data
  characteristics (e.g. constant-valued backscatter compresses
  to almost nothing; noisy data benefits less).
