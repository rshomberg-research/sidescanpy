# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- `analysis.detect_shadow_proposals`: object proposals from acoustic
  shadows in native slant-range data. Cells well below a running-median
  seabed level form shadow regions; each is paired with the highlight in
  front of it and returned as a box in native (ping, sample, channel)
  coordinates with sizes, contrast, a flat-seabed height estimate and a
  score (`ShadowProposals`). Meant to pre-fill annotation boxes.
  A second, weaker threshold (`weak_shadow_db`) finds the short, shallow
  shadows of objects near nadir and keeps them only with a highlight in
  front (`weak` column). The score uses the estimated height rather than
  the shadow length, so the same object scores alike at any range.
  `viz.plot_shadow_proposals` draws them on a tile or chip.

- `export.windows.native`: `native_chip` and `swath_tiles` cut windows that
  are square (or a fixed length) in metres from the native ping x sample
  grid and resize them with linear interpolation, keeping NaN cells NaN.
  `NativeWindow` maps image pixels to (ping, sample, channel) and back
  (`to_pixel`, `to_native`, `box_to_native`), for annotation round trips.
  For sonars whose ping spacing is ~10x the sample spacing, the metric
  render levels repeat or smear pings and native pixel crops squash shapes.

- `DatasetSpec.frequencies`: restrict target and background windows to
  the listed frequency keys. Without it a target is matched to the
  frequency whose sample spacing is closest to its catalog's, which is
  ambiguous when two frequencies share a sample spacing (and can put a
  dataset on a dead channel).

### Fixed

- **Distance-axis waterfalls no longer smear.** `viz.plot_waterfall` with
  `across_track`/`ground_range`/`slant_range` axes set every sample without
  a coordinate (water column, NaN padding of shorter pings after
  concatenating files of different lengths) to x = 0, so the last real cell
  of each ping stretched across the swath and whole rows came out one
  colour. Missing coordinates are now interpolated within the ping (cells
  stay masked) and all-padding columns are dropped.
- **SDF files whose range setting changes mid-file are split.** When a
  frequency's sample interval changes (e.g. 2500 samples at 60 m, then
  3334 at 40 m), `io.read_and_convert_segments` now cuts the file into one
  survey per constant-rate run (`<stem>_seg01`, ...), as it already did for
  XTF (`io.sdf.split`). Previously the whole file used the first page's
  sample rate and the later pings were mapped to the wrong ranges (twice
  their true range in the case seen). `read_and_convert` is unchanged. The
  project template's `01_convert` now converts through
  `read_and_convert_segments`.
- **Klein targets are placed at the ping their catalog names.** The
  `.tgt` string fields hold the target ping's own time and ping number on
  the sonar's clock (the clock of the converted pings). The reader now
  parses them into `extra_attributes["catalog_ping_time"]` /
  `["catalog_ping_number"]`, and the new
  `targets.correlate.resolve_targets_by_catalog_ping` matches them exactly,
  with no clock offset. `sidescan-convert targets` uses it by default for
  Klein catalogs. Previously targets were matched by `pick_timestamp`, the
  save time, which on real catalogs is 30-90 s after the target ping
  (hours for picks made in playback), so targets landed roughly 100 m
  along track from where they were. `--use-pick-time` (and the
  clock-offset options) keep the old behaviour for catalogs without those
  fields. Matched targets get `position_confidence="approximate"` and
  record `ss_h5` / `ping_index` / `ping_number`, which
  `targets.correlate.correlate_target` (and so exports and datasets) now
  uses directly instead of re-matching the nearest vessel position (a
  navigation fix repeated over ~10 pings put several targets on the same,
  wrong ping).
- **Klein targets are matched to the right frequency.** The reader now
  records the `.tgd` window's sample spacing
  (`extra_attributes["ping_meters_per_sample"]`), which
  `correlate_target` uses to pick the frequency `sample` is counted on.
  Without it a target fell to the first frequency in the file and was
  "projected" onto the high-frequency sample axis, doubling its range where
  the two frequencies differ in spacing (40 m range on some Klein 4900
  files); some targets left the swath and were dropped.
- `io.sdf.targets` documents the `.tgd` layout: every second ping, newest
  first; `window_row` is not always the target's ping.

## [0.7.0] — 2026-10-08

New XTF reader and a changed TVG default (results change for callers who
relied on the default spreading coefficient). The deprecation shims from
0.6.0 are kept for one more release and removed in 0.8.0.

### Changed

- **TVG spreading default is now 40 log R** (two-way spherical spreading),
  up from 20 log R. Samples are amplitude converted with `20*log10`, so the
  two-way spherical loss is 40 dB per decade; 20 was the one-way term,
  inconsistent with the two-way absorption term `2*alpha*R`. Affects
  `TVGParams`, the `"tvg"` step, `fit_range_decay`'s candidates and
  `sidescan-correct --tvg` when no coefficient is given. Configs that set
  `spreading_loss_db` explicitly are unchanged, and files already corrected
  record their coefficient in history, so `invert()` still undoes them
  exactly. 30 log R (insonified area growing with range) is a documented
  alternative still to be evaluated on real data; set
  `spreading_loss_db: 30.0` to use it.

### Added

- XTF reader (`io.xtf`): numpy structured dtypes for the file, channel and
  ping headers, packet framing, a streaming sonar-ping reader and a
  converter to `SideScanSurvey`. Files are split into segments where the
  range setting changes; `io.dispatch` and `sidescan-convert` produce one
  survey per segment. Port samples are reversed so sample 0 is nearest the
  sonar, as in the other readers. See the "XTF" guide page.

## [0.6.2] — 2026-10-08

### Fixed

- `processing.detect_far_range_cutoff_sample` no longer crops inside the
  water column. On recordings that start at the transmit pulse, the quiet
  water column read as the noise floor and the seafloor return as noise
  rising again, cropping away nearly the whole swath. A cutoff whose slant
  range is not beyond the median `altitude_m` is now rejected with a
  WARNING.
- `examples/project_template/04_export.py` skips grid segments with no
  georeferenced samples (noted in the report) instead of failing.

## [0.6.1] — 2026-10-08

### Changed

- Default TVG absorption (Francois-Garrison, when `absorption_db_per_m` is
  not given) and `analysis.fit_range_decay`'s absorption model use the
  channel's acoustic center frequency, `(freq_hz_start + freq_hz_end) / 2`,
  instead of the chirp start. New `processing.center_frequency_hz(ds)`.
  For a 1495-1605 kHz chirp the default rises from about 0.74 to 0.79
  dB/m; single-frequency channels (start == end) are unchanged. Channel
  labels (`hz_<freq_hz_start>`) are unchanged.
- `grids.bbox_from_channel_dataset`: the minimum margin is now
  `min_margin_m=5.0` (meters, converted at the data's latitude) instead of
  a fixed 0.001 deg (about 110 m), which dominated the margin of a typical
  grid segment. `min_margin_deg=` still sets a degree minimum explicitly.
  Auto-fitted grids (`grid_channel`, `sidescan-export grid`) are tighter.

## [0.6.0] — 2026-10-08

A restructure into a library API with a thin CLI. Workflow, numbered
"stages" and figure layouts moved out of the library into the CLI and
`examples/`. Many module paths changed; see the migration table below.
No deprecation shims are kept except where noted, and those are removed
in 0.8.0 (originally planned for 0.7.0).

Between 0.5.0 and this release, `main` carried a 0.4.x patch series
(0.4.6–0.4.59) that added the `sidescan-*` CLI, the investigation tools,
gridding fixes and the target dataset export. Those changes are included
here; their history is in git.

### Added

- **CLI:** `sidescan-convert`, `sidescan-correct`, `sidescan-export`
  (`waterfall`, `grid`, `track`, `chip`) and `sidescan-analyze`
  (`investigate`, `tvg-fit`, `segment`, `nadir`, `beam-geometry`,
  `usable-range`, `units`, `nav-attitude`, `recommend-config`). Every
  subcommand takes `--config` (YAML or JSON); unknown keys warn, and
  `--strict-config` makes them errors. Shared `-v`/`-q`/`--log-file`.
- **One correction API:** `processing.correct(ds, steps, params)`,
  `correct_survey`, `correct_file(path, chunk_pings=...)`. Built-in steps
  `crop_farrange`, `interpolate_aux`, `tvg`, `geometric`,
  `range_correction`, `georef`; any callable or `Step(name, fn)` is also a
  step. Applied steps and their parameters go to `processing_history`;
  re-running skips recorded steps unless `force=True`.
  `correct_survey` and `correct_file` take `frequency_params` to override
  step parameters for one frequency (e.g. absorption per frequency).
- **Reversible corrections:** `processing.invert(ds)` and
  `uncorrected_backscatter(ds)` undo radiometric steps from their
  recorded parameters.
- `processing.francois_garrison_absorption_db_per_m` (Francois &
  Garrison 1982), used by the `tvg` step with temperature, salinity,
  depth and pH parameters.
- **Analysis as small functions** returning dataclasses:
  `characterize_values`, `compare_outlier_rejection`, `analyze_nadir`,
  `estimate_beam_geometry`, `estimate_usable_range`, `fit_range_decay`,
  `fit_tvg_curve`, `infer_units`, `summarize_navigation`,
  `measure_ping_geometry`.
- `processing.segments` (`find_segments`, `split_by_segments`, gap /
  setting-change / garbage / instability detectors) and
  `processing.bottom` (first-bottom detection, nadir masking).
- `processing.slicing`: `slice_by_range`, `slice_survey`; every slicer
  takes and returns an `xr.Dataset`, and sample slicing uses the absolute
  `sample` coordinate, so trimming samples no longer shifts range.
- **Gridding:** `grids.grid_channel`, `grids.recommend_resolution`,
  `grids.to_geotiff` (new `[geo]` extra, rioxarray).
- **Targets and datasets:** `Target.category`; `targets.correlate`
  (`build_position_index`, `correlate_target`, `correlate_catalog`);
  `export.ml.build_target_dataset(sources, targets, DatasetSpec, out_dir)`
  writing COCO and YOLO (optional Ultralytics layout and train/val split),
  with jitter draws, random flips, slant-range or ground-range geometric
  chips, swath frames sized to the data, meter-sized placeholder boxes,
  multi-category annotations and background (no-target) images.
- `sidescanpy.log.configure_logging` and `log_timing`. The library logs
  only at DEBUG and WARNING.
- `examples/`: `quickstart.py`, `tvg_calibration.py`,
  `gridding_and_geotiff.py`, `build_yolo_dataset.py`,
  `investigate_dataset.py`, `figures/`. All are smoke-tested.

### Changed (breaking)

- `backscatter_raw` is no longer written. Converters store one
  `backscatter_intensity`, and corrections modify it in place; the
  uncorrected signal is recovered by inversion. Files that still carry
  `backscatter_raw` are read, and `correct_file` drops it on rewrite.
- `processing_level` is derived from `processing_history`
  (`raw` / `tvg` / `geometric` / `range_correction` / `georef`).
- Default absorption follows Francois-Garrison: about 0.74 dB/m at
  1.5 MHz (was 0.2) and 0.14 dB/m at 547.5 kHz (was 0.10). TVG-corrected
  output is brighter at long range than with 0.5.0.
- The bundled empirical range-correction table is gone.
  `range_correction` requires a table (object or path).
- `.ss.h5` writing: one encoding policy (gzip 4 + shuffle, ping-chunked),
  float64 stored as float32 except latitude/longitude. Corrected files are
  roughly 4–5x smaller; values match to float32 rounding.
- matplotlib is optional (`[viz]` extra). Everything except `viz` and the
  figure-producing CLI subcommands runs without it.
- CLI: `--value-var raw` / `backscatter_raw` → `uncorrected` (old
  spellings warn for one release); `--stage3-*` → `--usable-range-*`,
  `--stage4-*` → `--decay-*` (old flags and config keys warn for one
  release); `sidescan-export survey-window` and `progression` removed.
- `sidescan-convert targets` writes `position_scatter.png` only with
  `--plot`. `sidescan-convert --calibrate-offset` requires
  `--local-utc-offset-hours` (no timezone default).
- `resample_channel_to_grid` names rasters from `freq_hz_start`, so
  titles read e.g. "455 kHz" instead of "unknown".

### Fixed

- Target-to-ping matching used `111000 * hypot(dlat, dlon)` without
  `cos(lat)`, so east-west offsets read 1/cos(lat) too long (1.33x at
  41°). It is now geodesic (WGS84), and pings without a position fix are
  skipped. Targets near `max_dist_m` may now match, and the matched ping
  can change.
- Gridding IDW weighted over every in-radius sample, NaN included, then
  applied the weights to the finite values only. Weights now cover finite
  contributors only.
- Gridding aggregation is vectorized (2–10x faster); results match the
  previous per-cell loop exactly for count/mask/max/median/nearest and to
  1e-12 for mean/IDW.
- `apply_tvg` treats a zero sound velocity like a missing one.
- Georeferenced chips with no position at the window center are skipped
  instead of crashing.

### Removed

- Modules that did not import or had no users: `io.standard`,
  `io.ingest`, `analysis.comparison`, `analysis.plotting`,
  `analysis.geoplot`, `utils`, `processing.auxiliary`.
- `processing.pipeline` (`run_pipeline`, `run_correction_steps`,
  `apply_correction_steps`, `reconstruct_raw`) and `processing.chunked`.
- `sidescanpy.investigation.run_investigation` and every `stageN_*`
  function.
- Tracked binary test fixtures; tests use generated synthetic data.

### Migration

| 0.5.0 / 0.4.x | 0.6.0 |
|---|---|
| `processing.pipeline.run_pipeline`, `run_correction_steps`, `apply_correction_steps` | `processing.correct(ds, steps, params)` / `correct_survey` |
| `processing.chunked.correct_file_chunked` | `processing.correct_file(path, chunk_pings=...)` |
| `processing.pipeline.reconstruct_raw` | `processing.uncorrected_backscatter(ds)` or `invert(ds)` |
| `ds["backscatter_raw"]` | `processing.uncorrected_backscatter(ds)` |
| `investigation.run_investigation` | `sidescan-analyze investigate`, or `examples/investigate_dataset.py` |
| `analysis.investigation.stage1_characterize` | `analysis.characterize_values` |
| `analysis.investigation.stage1_5_outlier_comparison` | `analysis.compare_outlier_rejection` |
| `analysis.investigation.stage2_nadir` | `analysis.analyze_nadir` |
| `analysis.investigation.stage2_5_beam_geometry` | `analysis.estimate_beam_geometry` |
| `analysis.investigation.stage3_usable_range` | `analysis.estimate_usable_range` |
| `analysis.investigation.stage4_decay` | `analysis.fit_range_decay` |
| `analysis.investigation.stage5_units` | `analysis.infer_units` |
| `analysis.tvg_diagnostics.fit_hardware_tvg_curve` | `analysis.fit_tvg_curve` |
| `analysis.tvg_diagnostics.mean_backscatter_vs_range` | `processing.backscatter_vs_range` |
| `analysis.investigation.detect_first_bottom_return`, `mask_nadir_gap`, `blank_nadir_noise` | `processing.detect_first_bottom_return`, `mask_nadir_gap`, `blank_nadir_noise` |
| `analysis.investigation.reject_range_outliers` | `processing.reject_range_outliers` |
| `analysis.segmentation.find_segments` | `processing.find_segments` |
| `analysis.recommend.measure_ping_geometry` | `analysis.measure_ping_geometry` |
| `viz.investigation`, `viz.tvg_diagnostics`, `viz.nav_attitude` | `viz.analysis`, `viz.tvg` (plots take `ax`, never save) |
| `viz.waterfall.along_track_distance_m` | `processing.along_track_distance_m` (old path warns until 0.8.0) |
| `viz.track.build_survey_track` | `processing.build_survey_track` (returns `SurveyTrack`) |
| `targets.core.Target` | `sidescanpy.targets.Target` (defined in `core.target`) |
| `targets.interchange.coco` / `yolo` | `export.ml.write_coco` / `write_yolo` |
| `export.eightbit` | `export.image` |
| `targets.window.decode_and_correct_window` | `export.target_window.decode_and_correct_window` |
| `io.sdf.targets.calibrate_clock_offset` | `targets.correlate.calibrate_clock_offset` |
| `cli._common.read_survey_dir` and friends | `io.hdf5.read_survey_dir`, `available_frequencies`, `load_channel_dataset`, `atomic_write_survey` |
| `processing.range_correction.RangeCorrectionTable.load_default()` | pass your own table to the `range_correction` step |
| `sidescan-export survey-window` | `sidescan-export chip --n-background N` |
| `sidescan-export progression` | `examples/figures/correction_progression.py` |

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