# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

#### Dataset-export module (v0.15.0, in progress)
- **`sidescanpy.export`** (new subpackage) — turns an already-corrected
  survey channel into 8-bit training-dataset images, distinct from
  `processing/` (physical corrections) and `targets/interchange/` (ML
  annotation file formats).
  - `export.eightbit`: `to_uint8()`/`from_uint8()` — invertible
    percentile+gamma 8-bit display transform (any of uint16/uint32/
    float32/float64), with a configurable normalization scope: default
    per-image (each array stretched against its own percentiles), or
    precomputed via `compute_reference_stats()` for survey-/dataset-wide
    scope. `gamma=1.0` default matches this library's existing
    display-gamma convention (`viz.waterfall`/`viz.gridded`).
  - `export.windows`: `extract_chip()` — fixed-size, edge-aware,
    optionally-jittered crop around a point. `extract_full_channel()` —
    full-sample-axis window spanning a fixed ping span, downsampled to a
    fixed output size; `square=True` (default) sizes the native ping
    span to the channel's own sample count rather than a fixed ping
    count, avoiding an aspect-ratio stretch a mismatched fixed count
    would introduce once downsampled to a square output.
    `sample_negative_window()` — picks a window location at least a
    given distance from every cataloged target, for no-target negative
    samples. `targets_within_window()` — every cataloged target whose
    position falls inside an extracted window's extent (not just the
    one a chip was centered on), for images with multiple targets.
    `rescale_target_position()` — translates a target position from a
    window's native (pre-downsample) space into its resized output
    space.
  - `export.geometric_window.extract_geometric_window()` — resamples an
    already ping/sample-sliced channel window onto a regular *local*
    grid (along-track x across-track meters, no CRS/UTM projection),
    reusing `processing.geometric`'s per-sample `across_track_m` and
    `viz.waterfall.along_track_distance_m()`.
  - `export.georef_window.extract_georef_window()` — thin windowed
    wrapper over the existing survey-wide georeferencing/gridding path
    (`grids.resample_channel_to_grid` + `grids.definition.utm_grid_from_wgs84_bbox`),
    extracting a region around one window instead of gridding an entire
    survey. Produces a `SideScanGriddedRaster`, not a GeoTIFF file
    directly — this library still hasn't taken on the `rasterio`
    dependency a real GeoTIFF write would need.
- **`targets.interchange.coco`/`yolo`**: `CocoImageEntry` gained
  `bbox_size` (optional per-entry override of `write_coco`/`write_yolo`'s
  global `box_size`, for a real measured target extent instead of the
  shared placeholder) and support for multiple entries sharing one
  `file_name` — they become one shared COCO image record with multiple
  annotations (or, for YOLO, multiple lines in one label file) instead
  of one image per target, for windows with more than one cataloged
  target. Both changes are fully backward compatible — every existing
  caller's behavior (one target per image, one shared `box_size`) is
  unchanged. Raises `ValueError` if entries sharing a `file_name`
  disagree on `width`/`height`.
- Tests: `tests/unit/export/` (new), plus new cases in
  `tests/unit/targets/interchange/test_coco.py`/`test_yolo.py`. Full
  suite: 1352 passing / 36 skipped (was 1316).
- **Fixed same day: `extract_geometric_window`/`extract_georef_window`
  stretched their output.** Both derived `resolution_m` from the
  window's own *larger* along/across (or NS/EW) extent, then forced
  *both* axes to exactly `output_size_px` samples regardless of the
  true aspect ratio — the smaller-extent axis came out visibly
  compressed/stretched to fill the same pixel count as the larger one.
  Fixed by changing the contract entirely: both functions now take an
  explicit `resolution_m` and `size_px`, centered on a given real-world
  point (`center_along_m`/`center_across_m`, or `center_lat`/`center_lon`)
  — extent is purely `size_px * resolution_m` on *both* axes by
  construction, so the result can't be stretched relative to the real
  world regardless of the source data's own aspect ratio. This also
  adds real target-centering (previously the caller had no way to
  request a specific real-world center) and, for
  `extract_geometric_window`, `channel_idx=None` to pool both channels
  into one frame spanning the full swath width (no explicit mirroring
  needed — `across_track_m`'s own sign convention already separates
  port/starboard in real-world space). Both gain a `locate_point()`
  companion function to place a *different* real-world point (e.g. a
  second target) within an already-built grid. Tests rewritten
  accordingly. Full suite: 1361 passing / 36 skipped.
- **Fixed same day: `extract_full_channel` washed real images out to
  near-white.** It downsampled the native window (via `scipy.ndimage.zoom`)
  *before* a caller's percentile+gamma stretch ran, so the stretch measured
  its percentiles against an already-blurred, low-variance array instead
  of the real per-sample distribution — found by actually rendering real
  survey data and comparing against this project's own known-good
  waterfall/gridded output, which stretch first and only downsample for
  display. Fixed by removing the resize from `extract_full_channel`
  entirely (now returns the native-resolution window) and adding
  `resize_to()` as an explicit, separate, deliberately-last step —
  `to_uint8()` then `resize_to()`, never the other order. Full suite:
  1354 passing / 36 skipped.

#### CLI subpackage — sidescan-convert / sidescan-correct / sidescan-export / sidescan-analyze (2026-09-16)

Consolidates the pipeline scripts that used to be duplicated,
near-identically, across every consumer project's own `scripts/`
directory. Direct comparison found 9 of 13 numbered scripts
byte-identical between `project-b` and `project-a` except docstring
path text — pure duplication, no per-project logic — and the one real
divergence found (`project-b/04_radiometric.py` silently missing the
TVG term-override flags) was an already-occurred drift bug, not a
hypothetical risk. New `sidescanpy.cli` subpackage, four console-script
entry points (`[project.scripts]` in `pyproject.toml`):

- **`sidescan-correct`** (replaces `02_crop_farrange.py`/
  `04_radiometric.py`/`06_geometric.py`/`08_georef.py`) — one command,
  explicit `--crop-farrange`/`--tvg`/`--geometric`/`--georef` step
  flags, independently toggleable and independently idempotent.
  Requesting several steps in one invocation now reads + writes each
  file once (`sidescanpy/cli/_common.py: process_files_in_place()`),
  not once per stage. Verified byte-identical output against real JSF
  and Klein SDF survey data (including the `--no-spreading`/
  `--no-absorption` TVG override path) vs. running the four old scripts
  in sequence.
- **`sidescan-convert`** (replaces `01_convert.py`/`11_read_targets.py`)
  — `survey` (unchanged) and `targets`, which dispatches on the target
  catalog directory's own contents (`*.target` vs `Target_*.tgt`), not
  the paired survey format. Promotes `project-b/11_read_targets.py`'s
  Klein clock-offset calibration logic into `sidescanpy.io.sdf.targets`
  as new public functions — `calibrate_clock_offset()`,
  `resolve_targets_offset_corrected()`, `resolve_targets_nearest_ping()`
  — this is a Klein-acquisition-general fix (a constant per-survey
  epoch shift), not project-specific, and belongs where
  `read_klein_targets()` already lives. Also new:
  `viz.targets.plot_target_position_scatter()`/
  `export_target_position_scatter_png()`, promoted from the
  near-duplicated scatter-plot block in both old `11_read_targets.py`
  copies. Verified byte-identical `targets.json` output against real
  data for both formats, including calibrated-offset Klein resolution
  (same derived offset, 241760407.5s, on the real two-file survey
  checked).
- **`sidescan-export`** (replaces `03_track_maps.py`/`05_waterfalls.py`/
  `07_ground_waterfalls.py`/`09_grid.py`/`12_track_with_targets.py`/
  `13_export_chips.py`, promotes `project-a/scripts/
  14_export_dataset.py`'s `chip`/`survey-window`/`progression`
  subcommands essentially unchanged — that script was already
  format-agnostic and cross-project proven) — new `waterfall`
  (`--range-mode {slant,ground}` merges the old 05/07 split), `grid`,
  and `track` (optional `--targets-json` overlay merges the old 03/12
  split) subcommands. Verified byte-identical output against real data
  for waterfall/ground-waterfall/grid/track, and byte-identical chip
  PNGs + `targets.coco.json`/`.yolo` (contact sheet differs only in a
  deliberate cosmetic title-text change) against real target data.
- **`sidescan-analyze`** (new) — `investigate` promotes
  `project-a/scripts/investigate_dataset.py` essentially unchanged
  (a thin driver over `sidescanpy.investigation.run_investigation()`).
  `tvg-fit` is a from-scratch driver built on `stage4_decay()`'s own
  native `r_ref_m="auto"`/`r_fit_min_m="auto"` support (see "v0.14.0
  follow-up 11" below) rather than porting
  `project-a/scripts/debug/plot_official_tvg_fit.py`'s now-partially-
  superseded manual recipe verbatim — that script's exploratory
  predecessors (`compare_tvg_fit_recipes.py`/
  `validate_tvg_fit_flatness.py`/`tvg_range_residual_diagnostic.py`)
  were one-time comparison tools that led to the adopted recipe, not an
  ongoing capability, so they weren't promoted either. Verified against
  real data: identical fit window, winning candidate, and RMSE to
  `plot_official_tvg_fit.py`'s own manual recipe (only `r_ref_m` differs
  — 15.00m auto-placed vs. 14.50m manual mid-swath — a documented,
  expected difference between the two recipes that doesn't change the
  fit outcome).
- **`sidescanpy.cli._config`** — shared `--config path.json` support
  across all four commands: a small per-dataset JSON file (not YAML —
  this codebase already uses JSON everywhere and has no YAML
  dependency) seeds a (sub)command's argparse defaults via
  `set_defaults()`; an explicit CLI flag still overrides a config
  value. A single file can hold settings for several different
  (sub)commands at once — unrecognized keys are silently ignored, not
  an error.
- Tests: `tests/unit/cli/` (new — `test_correct.py`, `test_convert.py`,
  `test_export.py`, `test_analyze.py`, `test_config.py`), plus new
  cases in `tests/unit/io/sdf/test_targets.py` and
  `tests/unit/viz/test_targets.py` for the promoted Klein calibration
  functions and scatter-plot helper. Full suite: 1428 passing / 36
  skipped (was 1373 before this work).
- Consumer projects: `project-b`/`project-a` had their numbered
  scripts (and `project-a`'s `14_export_dataset.py`/
  `investigate_dataset.py`) deleted, replaced by a `datasets/*.json`
  per-survey config file each (see each project's own `datasets/`
  directory) — see each project's own `README.md`/`CLAUDE.md` for the
  consumer-project side of this consolidation.

#### Gridding module
- **`sidescanpy.grids.GridDefinition`** — grid schema with bounding box
  (south, west, north, east), resolution in meters, and CRS support.
  - Validates bounds (rejects inverted/empty boxes, antimeridian crossings)
  - Geodetic distance calculations via pyproj (Web Mercator projection)
  - Cell count derivation: `num_cells_ns`, `num_cells_ew`, `num_cells_total`
  - Grid center coordinate generation as 2D arrays
  - CRS reprojection with `reproject_to(crs)` helper
  - Round-trip WGS84 ↔ UTM transformation accurate to ~0.03°
  - Serialization: `to_dict()` / `from_dict()` for config storage
- **`sidescanpy.grids.SideScanGriddedRaster`** — xarray-validated wrapper for
  gridded backscatter (bbox, resolution, CRS, `n_contributing_samples`,
  `units`). Supports `reproject_to()`, `slice_by_bbox()`,
  `slice_by_indices()`, and dict serialization.
- **`sidescanpy.grids.resample_channel_to_grid()`** — resamples georeferenced
  channel backscatter onto a `GridDefinition` using `scipy.spatial.cKDTree`.
  - `ResamplingMethod`: `mean`, `median`, `max`, `nearest`, `idw`
  - Correctly treats 3D `(ping, channel, sample)` port/starboard samples as
    independent observations (does not average channels together)
  - `backscatter_var` parameter selects which backscatter variable to grid
    (default `"backscatter_raw"`; pass `"backscatter_intensity"` to grid
    TVG-corrected data — see Fixed, below)
  - Validates 3D shape and lat/lon/backscatter shape agreement, raising
    `ValueError` with an actionable message on 2D or mismatched input
- **`sidescanpy.viz.gridded`** — `plot_gridded_raster()`,
  `plot_gridded_with_coverage()`, `plot_gridded_multiple_methods()`,
  `export_gridded_png()`
- `scripts/demo_gridding_real_data.py` — end-to-end gridding demo against a
  real `.ss.h5` survey; `--method` flag to run a single resampling method
- `tests/unit/grids/conftest.py` — shared 3D channel fixtures
  (`channel_3d`, `channel_3d_real_proportions`) matching production
  `(ping, channel, sample)` proportions

### Changed
- All script/demo output directories now default under `./out/`, matching
  existing repo convention.

### Fixed
- **`resample_channel_to_grid()` accepted 2D input silently.** No shape
  validation existed; malformed or pre-flattened input could silently
  misalign backscatter values with the wrong coordinates instead of erroring.
  Now validates `backscatter_var` is 3D and that `latitude_per_sample` /
  `longitude_per_sample` match its shape, raising a clear `ValueError`
  otherwise.
- **Gridding demos always plotted raw, uncorrected backscatter — even on
  radiometrically corrected surveys.** `apply_tvg()`
  (`processing/radiometric.py`) writes TVG-corrected values to
  `backscatter_intensity` and deliberately leaves `backscatter_raw`
  untouched as a linear-amplitude reference. `resample_channel_to_grid()`
  hardcoded `backscatter_raw` with no way to select another variable, and
  `demo_gridding_real_data.py`'s variable-selection fallback always matched
  `backscatter_raw` first regardless of `processing_level`. Every gridded
  map produced against radiometric-labeled data was silently ungridding the
  uncorrected variable. Fixed via the new `backscatter_var` parameter; the
  demo now selects `backscatter_intensity` automatically when
  `processing_level == "radiometric"`.
- **`SideScanGriddedRaster` and its colorbar hardcoded `units="linear"`
  regardless of what was actually gridded.** TVG-corrected data is in dB, so
  plots of corrected surveys were mislabeled. `SideScanGriddedRaster` now
  takes a `units` parameter (propagated through `reproject_to()`,
  `slice_by_bbox()`, `slice_by_indices()`, and dict serialization);
  `resample_channel_to_grid()` sets it to `"dB"` when
  `backscatter_var="backscatter_intensity"`, `"linear"` otherwise; the
  colorbar label in `viz/gridded.py` now reads the raster's actual units
  instead of a hardcoded string.
- **`io/sdf/converter.py`'s pad-to-max-with-NaN behavior (for pages with
  differing sample counts) was silent, and discarded each ping's true
  recorded length.** Now logs a warning when padding actually occurs,
  and `_build_channel_dataset()` adds a new `valid_samples` (ping,
  channel) variable recording each ping's true length before padding —
  downstream code (e.g. `processing/farrange.py`) can now distinguish
  real short pings from padded filler instead of the padded shape
  looking like every ping genuinely recorded that many samples.
- **`01_convert.py`/`02_crop_farrange.py` (consumer-project pipeline
  scripts, `project-a` and `project-b`) wrote a survey's full,
  uncropped far-range data to disk during conversion, then
  `02_crop_farrange.py` read it back and rewrote it cropped** — doubling
  disk I/O for exactly the data that ends up thrown away.
  `01_convert.py` now runs `crop_far_range_noise()` in memory before its
  one and only write; `02_crop_farrange.py` remains for files converted
  before this change and is now a fast no-op (one read, no rewrite) on
  files already cropped at convert time.
- **JSF conversion crashed (`AttributeError`) on any file containing DVL
  (Doppler Velocity Log) messages.** `io/jsf/converter.py`'s
  `_build_dvl_record()` read `m.altitude_m` from each
  `DopplerVelocityLogDataMessage`, but that class has no such
  attribute — its altitude-equivalent field is `bottom_dist` (mean of
  the DVL's 4 beam bottom-distance readings). Every real JSF file with
  DVL data failed to convert at all. The existing unit tests never
  caught this because the DVL test fixture was a `Mock(spec=...)` with
  `.altitude_m` set directly rather than a real message object — masked
  the bug instead of exercising it (same class of issue as this file's
  "Passing Tests ≠ Correct Code" lesson). Fixed: converter now reads
  `m.bottom_dist`; the output variable is still named `altitude_m`
  (unchanged, matches the DVL record schema); fixture corrected to set
  `.bottom_dist`; added a regression test asserting real values flow
  through, not just that the field exists.
- **`export.eightbit.to_uint8()`'s gamma was inverted relative to this
  library's own established convention, and its docstring claimed the
  opposite (that it matched).** `viz.waterfall.plot_waterfall`/
  `viz.gridded` hand `gamma` straight to `matplotlib.colors.PowerNorm`,
  whose convention is `normalized ** gamma` (`gamma > 1` darkens
  midtones/background, the classic contrast-enhancing direction).
  `to_uint8()` computed `normalized ** (1/gamma)` instead — the inverse
  — so a gamma value already tuned against a `PowerNorm`-based render
  (e.g. 2.5) produced the opposite visual effect through this function:
  brightened/washed-out instead of darkened. Found on real data (a
  dataset-export progression grid at gamma=2.5 came out uniformly
  washed-out/whited-out, the same nominal gamma that correctly enhanced
  contrast in this survey's own `plot_waterfall` renders). Fixed:
  `to_uint8()` now computes `normalized ** gamma`; `from_uint8()`'s
  inverse updated to match (`normalized ** (1/gamma)`). Regression test:
  `tests/unit/export/test_eightbit.py::test_gamma_above_1_darkens_matching_powernorm_convention`.
  **Any caller that picked a gamma value empirically against the old
  (inverted) behavior needs to re-tune it** — the fix changes real
  output, not just documentation.
- **`extract_geometric_window`/`extract_georef_window`'s fixed
  `1.5 * resolution_m` KDTree search radius left periodic empty stripes
  on a survey whose GPS updates far less often than its ping rate.**
  Found on a real Klein SDF survey: real GPS fixes land only every ~8
  pings, each jumping ~2m — a grid resolution chosen from the *average*
  along-track distance per ping (correctly avoiding an earlier,
  differently-broken resolution heuristic) still can't find a real point
  within `1.5x` that resolution for most cells between fixes, since the
  real gaps are wider than that at the fix-to-fix boundaries. Widening
  the resolution itself to bridge every gap would zoom a close-up window
  out absurdly (200px at ~2m/px spans ~400m). Fixed: `extract_geometric_window`
  gained `search_radius_m`, `extract_georef_window` gained
  `max_search_radius_m` (threaded straight to `resample_channel_to_grid`'s
  existing parameter of the same name) — both optional, defaulting to
  the prior `1.5 * resolution_m` for full backward compatibility. A
  caller widens the radius to the real worst-case gap (e.g. the
  95th-percentile along-track diff) to bridge duty-cycle gaps without
  touching resolution/zoom at all. Real-data confirmation: one chip-scale
  target's geometric coverage roughly doubled (17,355→29,599 of 40,401
  cells filled) at the same resolution, radius alone changed.

## [0.5.0] — 2026-06-17

### Added
- **Klein SDF support**: Read Klein SDF (Side-scan Data File) format produced
  by Klein 4900-series side-scan sonars. Includes:
  - `sidescanpy.io.sdf.SDFPageHeader` — typed dataclass for the 512-byte
    page header per the Klein SDF/SDFX Data Page Definitions spec, with
    100+ named fields and 26 reserved uint32 slots.
  - `sidescanpy.io.sdf.SDFSonarPage` — sonar data page (versions 3501/3502)
    wrapping a header plus port and starboard sample arrays.
  - `sidescanpy.io.sdf.SDFUnknownPage` — non-sonar page types parsed for
    page-level introspection but with raw body bytes preserved.
  - `read_sdf(path)` — eager file reader returning all pages.
  - `read_sdf_iter(path)` — lazy iterator over pages.
  - `read_sdf_header(f)` — low-level helper that reads marker + header at
    the current file position.
  - `convert(pages, ...)` — produces a `SideScanSurvey` from SDF pages,
    grouping by frequency and building navigation/attitude aux records
    from per-page values.

- **Format-dispatching CLI**: `scripts/convert.py` now dispatches on file
  extension. Same command works for `.jsf` and `.sdf` input.

- **Batch processing script**: `scripts/batch_convert_and_plot.py`
  processes a directory of sonar files, converting each to `.ss.h5` and
  exporting waterfall PNGs per frequency. Failures on individual files
  are logged and skipped; processing continues. Re-running skips
  already-processed files unless `--force` is given.

- **`sidescanpy.processing.override_altitude(survey, altitude_m)`** for
  setting altitude on surveys where the source format doesn't record it
  (e.g. Klein with depth/altitude sensors disabled).

### Changed
- Reader behavior on bad SDF markers: previously raised `ValueError`,
  now logs a warning and scans forward up to 1 MB looking for the next
  valid marker. Robust to small file corruption.
- `parse_page_header()` removed from public SDF API (still available
  internally). Users should call `SDFPageHeader.from_bytes()` directly
  for header-only parsing or use the `read_sdf*` functions for full
  page reading.

### Fixed
- Klein SDF sample data masking: Klein stores sample amplitudes as
  uint32 with the low 16 bits carrying amplitude. Bit 17 is occasionally
  set (~0.001% in observed data) as undocumented metadata. Reader now
  masks samples to the low 16 bits, eliminating spurious bright pixels
  in waterfall visualizations. Masking is logged at DEBUG level.

- Klein SDF `sonar_freq_khz` was reading at the wrong byte offset for
  3500-series Klein systems. The Klein spec lists field numbers 93
  (`processedPingNumber`) and 94 (`processedPingSpacing`) as N/A for
  3500-series; their byte slots don't exist in 3500 data. Reader
  STRUCT updated to match the 3500 layout. Reserved field count
  increased from 24 to 26 to maintain the 512-byte header total.

### Notes on Klein SDF
- The `depth_m` field in Klein SDF is stored in Volts and requires
  INI-file calibration to convert to meters; we don't currently parse
  Klein INI files. Surveys converted from SDF have `depth_m` set to
  NaN. Users with calibration data can override directly via xarray:
  `ch.dataset["depth_m"] = (["ping"], calibrated_depths_array)`.
- Klein doesn't provide sound velocity in the page header. Surveys
  have `sound_vel_ms` set to NaN; downstream `compute_geometric_ranges`
  falls back to 1500 m/s when sound velocity is NaN.
- 26 reserved uint32 fields in the page header are exposed as
  `header._reserved` (a tuple of 26 uint32 values) for forward-
  compatibility with future Klein firmware.

## [0.4.0] - 2026-06-09

Major release adding geometric and georeferencing computations,
slicing operations, visualization utilities, and a real-data
integration test that surfaced and fixed three end-to-end bugs.

### Added

#### Processing
- **`compute_geometric_ranges`** (`sidescanpy.processing.geometric`) —
  computes six per-sample data variables: `slant_range_m`,
  `ground_range_m`, `across_track_m`, `along_track_m`,
  `vertical_offset_m`, `depth_per_sample_m`. Includes pitch correction
  for the platform-frame XYZ components. Sign conventions: across_track
  is negative for port, positive for starboard; along_track is positive
  forward of nadir. Lossless: backscatter preserved at original sample
  positions.
- **`compute_georeferencing`** (`sidescanpy.processing.georeferencing`)
  — computes per-sample `latitude_per_sample` and `longitude_per_sample`
  via `pyproj.Geod` forward calculations on a configurable ellipsoid
  (default WGS84). Vectorized; handles NaN propagation cleanly.
- **Slicing module** (`sidescanpy.processing.slicing`) — five pure
  functions for slicing `SideScanFrequencyChannel`: `slice_by_pings`,
  `slice_by_samples`, `slice_by_time`, `slice_by_bbox`,
  `slice_by_ground_range`. Schema-preserving.

#### Visualization
- **`plot_waterfall`** (`sidescanpy.viz`) — port+starboard combined
  waterfall with configurable x-axis (sample/ground_range/across_track)
  and y-axis (ping/time/along_track) modes. Percentile-based auto color
  scaling, NaN-aware coordinate handling for water-column gap.

#### Scripts
- `scripts/convert_jsf.py` — convert raw JSF to `.ss.h5` with optional
  pipeline processing (raw → radiometric → geometric → georef).
  Supports multi-file input.
- `scripts/export_waterfall_series.py` — export a series of waterfall
  PNGs from a survey, sliced by ping count or time window.
- `scripts/diagnose_pipeline.py` — print finite-count and sample values
  at each pipeline stage; useful for surfacing integration bugs.

#### Testing
- Full-pipeline integration test (`tests/integration/test_full_pipeline.py`)
  exercising read → convert → interpolate → TVG → geometric → georef
  → write → read on the real sample JSF fixture. Surfaced three real-data
  bugs (see Fixed below).
- ~150 new unit tests across the new modules.

#### Documentation
- `CHANGELOG.md` (Keep a Changelog format).
- `LICENSE` (MIT).
- README rewritten as quick-start.
- Module-level docstrings cross-linking to `docs/*.md` for `core`,
  `io.jsf`, `io.netcdf`, `io.hdf5`, `processing`, `viz`.
- New: `docs/core.md`, `docs/io_jsf.md`, `docs/io_netcdf.md`,
  `docs/io_hdf5.md`, `docs/processing.md`, `docs/viz.md`.

### Changed

- **BREAKING**: `KNOWN_PROCESSING_LEVELS` trimmed to `{"raw",
  "radiometric"}`. Geometric and georeferencing are now additive (they
  attach new data variables) rather than backscatter-transforming.
  Datasets with `processing_level == "geometric"` or `"georeferenced"`
  written by older versions will now fail validation on load.
- **BREAKING**: Schema rename — added `fathometer_reading_m` as a
  required per-ping field (raw altimeter measurement, attitude-affected).
  `altitude_m` semantics shifted to "vertical altitude after attitude
  correction." For now `altitude_m == fathometer_reading_m` until
  proper attitude correction lands. Existing `.ss.h5` files written
  under the old schema will fail validation.
- `scripts/plot_sonar_ping.py` rewritten: uses `read_survey()` and
  `plot_waterfall()`. New `--process`, `--x-axis`, `--y-axis` flags.
  Multi-file directory loading dropped.

### Fixed

- Navigation timestamp conversion in aux record builders was inverted,
  causing all aux timestamps to land near 1970-01-01 instead of the
  actual survey time. (`int(m.timestamp * 1e9)` is now unconditional.)
- `interpolate_aux_onto_pings` no longer overwrites valid per-ping
  fields with NaN when the source aux variable is structurally all NaN
  (e.g. attitude data on devices with broken validity flags). Skips and
  preserves the original.
- `compute_geometric_ranges` treats `sound_vel_ms <= 0` as invalid
  (in addition to NaN). Devices that report zero as a "no data"
  sentinel previously cascaded to all-NaN geometric output. Falls back
  to 1500 m/s with a warning.
- `scripts/plot_sonar_ping.py` along-track y-axis previously mismatched
  between waterfall (slice-relative) and marker line (survey-relative).
  Both now use survey-relative reference consistently.

### Known issues

- `plot_sonar_ping.py` slider re-renders the full waterfall on every
  step; lags noticeably on 5000-ping surveys. Future: cache waterfall,
  update only line plot and marker.

### Dependencies

- Added: `pyproj>=3.6`
- 
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