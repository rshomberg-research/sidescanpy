# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added
- `docs/core.md` — schema, validation model, survey container reference.
- `docs/io_jsf.md` — JSF reader, message hierarchy, conventions.
- `docs/io_netcdf.md` — single-channel NetCDF IO reference.
- `docs/io_hdf5.md` — full-survey HDF5 IO with group layout.
- `docs/processing.md` — interpolate and radiometric reference, including
  per-field source decision matrix cross-linked from
  `jsf_aux_source_selection.md`.
- `CHANGELOG.md` (Keep a Changelog format).
- `LICENSE` (MIT).
- README rewrite focused on package purpose and quick-start example.
- Module-level docstrings in `__init__.py` for `sidescanpy.core`,
  `sidescanpy.io.jsf`, `sidescanpy.io.netcdf`, `sidescanpy.io.hdf5`,
  and `sidescanpy.processing`. Each cross-links to its `docs/*.md` file.
- `sidescanpy.processing.geometric.compute_geometric_ranges` — attaches
  per-sample `slant_range_m` and `ground_range_m` data variables to a
  validated dataset. Lossless: backscatter is preserved at original
  sample positions; ranges are derived from per-ping altitude and
  sound velocity. Handles nadir gap (water-column samples) and NaN
  altitude/sound velocity gracefully. ~24 unit tests.
- `compute_geometric_ranges` now produces six per-sample data variables:
  `slant_range_m`, `ground_range_m`, `across_track_m`, `along_track_m`,
  `vertical_offset_m`, `depth_per_sample_m`. Includes pitch correction
  for the platform-frame XYZ components. New `expose_unmasked_z` kwarg
  controls whether vertical_offset and depth_per_sample mask the water
  column (default: yes).
- Sign conventions: across_track_m is negative for port, positive for
  starboard. along_track_m is positive forward of nadir.
- 53 unit tests covering the new variables, pitch math, sign conventions,
  edge cases, and metadata propagation.
- `sidescanpy.processing.georeferencing.compute_georeferencing` — adds
  `latitude_per_sample` and `longitude_per_sample` data variables to a
  dataset. Uses `pyproj.Geod` for vectorized geodetic forward calculations
  on a configurable ellipsoid (default WGS84). Handles NaN propagation
  through per-ping nav/heading and per-sample offset fields. 26 unit
  tests covering the math, NaN propagation, ellipsoid options, and
  metadata.
- `pyproj>=3.6` added as a dependency.


### Changed
- Documentation pattern established: short `__init__.py` blurb (purpose,
  entry points, conventions, cross-link) plus focused `docs/*.md`
  reference (entry points, conventions, known issues, future work).
- `KNOWN_PROCESSING_LEVELS` trimmed to ``{"raw", "radiometric"}``.
  Geometric and georeferencing operations are now treated as additive
  (they attach new variables) rather than backscatter-transforming
  (which would advance processing_level).
- `docs/core.md`, `docs/processing.md` updated to reflect the
  additive geometric/georeferencing model.

## [0.3.0] - 2026-06-03

### Added
- **Processing — radiometric correction.**
  `sidescanpy.processing.radiometric.apply_tvg` applies time-varying gain
  (spreading + absorption) to backscatter data. Frequency-dependent absorption
  defaults via `absorption_for_frequency`. ~22 unit tests covering the math,
  edge cases, and metadata propagation.
- **Processing — auxiliary interpolation.**
  `sidescanpy.processing.interpolate.interpolate_aux_onto_pings` replaces
  per-ping fields (lat/lon/heading/roll/pitch/heave) with values interpolated
  from the survey's aux records onto each ping's timestamp. Lower-level
  `interpolate_onto_channel` accepts arbitrary external `xr.Dataset` sources.
  ~28 unit tests.
- **Diagnostic script.** `scripts/plot_jsf_aux_streams.py` extracts and compares
  all auxiliary streams in a JSF file, with mirror detection, per-source
  change-rate stats, and matplotlib plotting.
- **Documentation.** `docs/jsf_aux_source_selection.md` describes the diagnostic
  methodology and per-field source-selection decisions.

### Fixed
- Removed lower-bound clip on TVG output that was forcing all water-column
  values to 0 dB.
- Restored `backscatter_raw` immutability in `apply_tvg`.
- `NMEAGLLMessage`: corrected lat/lon extraction (was using raw `DDMM.mmmm`
  string, now uses pynmea2's decimal-degree fields).
- `PitchRollDataMessage`: temperature unit conversion (`* 10` → `/ 10`).
- `DopplerVelocityLogDataMessage`: roll copy-paste bug (`_raw_pitch` →
  `_raw_roll`).

### Changed
- Field renames in message classes for schema alignment: `course_deg` →
  `course_true_deg` on `NMEARMCMessage`, similar suffixes on related fields.
- `NMEAGGAMessage` and `NMEAVTGMessage` now extract lat/lon and course/speed
  respectively (previously stubs).

## [0.2.0] - 2026-06-02

### Added
- **NetCDF IO.** `sidescanpy.io.netcdf.{reader,writer}` for single
  `SideScanFrequencyChannel` datasets. Eager-by-default, lazy via `chunks=`. ~31
  round-trip tests.
- **HDF5 IO.** `sidescanpy.io.hdf5.{reader,writer}` for full `SideScanSurvey`
  objects with grouped layout (`/frequencies/hz_*`,
  `/aux_streams/{navigation,attitude,dvl,sensors}`). Lazy/dask-backed loading
  via `chunks=`. ~57 unit tests + integration round-trip tests against real JSF
  data.
- **Survey-level metadata** in HDF5 root attrs: `survey_name`, `source_format`,
  JSON-serialized `source_files`.

### Changed
- Frequency-channel keys renamed: `freq_<N>_hz` → `hz_<N>` to align with HDF5
  group naming.
- `MultiFrequencySideScanSurvey` renamed to `SideScanSurvey`.
- `HDF5_GROUP_ENVIRONMENTAL` → `HDF5_GROUP_SENSORS`; added `HDF5_GROUP_DVL`.
- Aux record empty-builders moved from `converter.py` to `schema.py`. Imported
  by both converter and HDF5 writer.

## [0.1.0] - 2026-06-01

### Added
- **Schema layer.** `sidescanpy.core.schema` defines xarray accessors with eager
  validation: `SideScanAccessor`, `NavigationRecordAccessor`,
  `AttitudeRecordAccessor`, `DVLRecordAccessor`, `SensorRecordAccessor`.
  Required dims/coords/vars/attrs enforced on construction. Aux records support
  `is_empty()` for empty-but-valid zero-observation datasets.
- **Survey container.** `sidescanpy.core.survey` defines
  `SideScanFrequencyChannel` (single-frequency wrapper) and `SideScanSurvey`
  (multi-frequency container with optional aux records and sensor dict). Aux
  records exposed as `@property` + `@setter`; setting to `None` clears. Setters
  validate eagerly.
- **JSF reader.** `sidescanpy.io.jsf.reader.read_jsf` and `read_jsf_iter` parse
  Edgetech `.jsf` binary files into typed `JSFMessage` instances.
- **JSF converter.** `sidescanpy.io.jsf.converter.convert` dispatches a mixed
  message list into a `SideScanSurvey`. Frequency channels grouped by
  `(transmit_freq_start, transmit_freq_end)`. Aux records built from NMEA /
  pitch-roll / DVL / pressure messages. `repair_mode` controls handling of
  unmatched port/starboard pings.
- **Test infrastructure.** Shared fixtures in `tests/conftest.py` including
  `make_minimal_*` builders. Mocks for converter unit tests. Real JSF fixture
  for integration tests.
- ~310 unit tests + ~27 integration tests, all passing.