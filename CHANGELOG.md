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

##### CLI subpackage follow-up (2026-09-16, same day — real-data smoke test)

Found by actually running each new command against real survey data end
to end (`project-a/scripts/run_20251015_survey-a.sh`,
`project-b/scripts/run_site-b.sh` — new try-it-out scripts,
one per consumer project, mirrored in structure with format differences
called out inline as comments) rather than only unit tests:

- **`sidescan-convert`'s `-v`/`--verbose` was silently dropped when
  passed after the subcommand** (`sidescan-convert survey ... -v`) —
  argparse requires top-level flags *before* the subcommand name, and
  the fix isn't "just add `-v` to each subparser too": doing that with
  a plain `default=False` would have silently *overwritten* a real
  `-v` already set before the subcommand, since argparse copies a
  subparser's entire parsed sub-namespace (including unset-flag
  defaults) onto the shared one. Fixed with `default=argparse.SUPPRESS`
  on the subparser copy in `convert.py`/`export.py`/`analyze.py`, which
  only ever adds capability (accepting `-v` in either position) and
  never removes it. 2 new regression tests
  (`tests/unit/cli/test_convert.py`) cover both positions plus the
  overwrite risk directly.
- **`sidescan-convert` never set matplotlib's `Agg` backend before
  transitively importing `pyplot`** (via `viz.targets`'s position-
  scatter plot) — unlike `export.py`/`analyze.py`, which both already
  did. Harmless-but-noisy on this machine (matplotlib thrashed through
  several unavailable GUI backends — macOS, Qt, GTK3/4 — before
  eventually falling back), but a real crash risk on a stricter/more
  minimal environment. Fixed: same `matplotlib.use("Agg")` guard added
  before the `viz.targets` import, matching the other two CLIs exactly.
- **`sidescan-analyze tvg-fit` had no way to proceed when Stage 3
  couldn't auto-determine `usable_range_m`** — a known, previously-
  documented Klein SDF failure mode (real-time hardware AGC flattens
  backscatter enough that the SNR-vs-noise-floor threshold never trips,
  confirmed across every Klein channel checked in this project's own
  investigation history), reproduced immediately on real `project-b`
  data. New `--usable-range-m` flag bypasses Stage 3's auto-detection
  entirely when passed. 2 new tests
  (`tests/unit/cli/test_analyze.py`) — a flat/no-decay synthetic
  fixture reproduces the real failure mode and confirms the override
  fixes it.
- Full suite: 1432 passing / 36 skipped (was 1428). Version bumped
  0.4.10 → 0.4.11.

##### CLI subpackage follow-up 2 — real bounding-box visualization + a silent-success bug (2026-09-16, same day)

Found asking directly "is target bounding-box visualization part of the
system" after reviewing a real try-it-out run's output — it wasn't, for
either of the two things actually wanted (a full waterfall with boxes at
every matched target; a per-target context image with a drawn box), even
though the underlying box-drawing primitive
(`viz.contact_sheet.render_target_context()`) already existed from the
old, now-deleted `13_export_chips.py`.

- **`sidescan-export waterfall` gained `--targets-json`** (plus
  `--max-match-dist-m`/`--box-size`, matching `chip`'s own flags) — draws
  a red box + target-id label at every matched target's real position on
  the full rendered waterfall. Scoped to the default `sample`/`ping`
  axes only (rejected with a clear error otherwise) — `plot_waterfall`
  renders in real matplotlib *data* coordinates via `pcolormesh`, not raw
  pixel-array indices (x = ∓sample by channel, y = the *global* ping
  index `ping_offset` already bakes in), a genuinely different
  coordinate system from `render_target_context`'s plain `imshow` — an
  early version of this reused `bbox_from_center`/local-row math meant
  for the *other* rendering path and would have drawn every box in the
  wrong place; caught before shipping by writing `_target_waterfall_xy()`
  as a small pure function and unit-testing the coordinate math directly
  rather than only checking that a PNG got written.
- **`sidescan-export chip` gained `--with-context`** — alongside the
  existing tight `target_<id>.png` crop, saves `context/context_<id>.png`
  via `render_target_context()` on the full loaded ping neighborhood
  (wider view, real box drawn at the target's true position). Reuses the
  same real position-correlation machinery `chip` already has (matching
  each target's recorded lat/lon against the real per-ping GPS track) —
  not the old, since-removed `13_export_chips.py`'s approach (decoding
  the target catalog's own embedded window snippet), which the
  consolidation's own history already found to be wrong for JSF surveys.
- **Real bug, same session: `chip`/`survey-window` always reported
  success even when nothing was exported.** Both had an early
  `return` (bare, implicitly `None`) on their "nothing correlated/
  extracted" path, logged as `ERROR` -- but `main()`'s own
  `result if isinstance(result, int) else 0` fallback (needed for
  `progression`, which has no such early-exit path) turned that `None`
  into a *successful* exit code. Caught directly: a real try-it-out
  script using `set -e` to fail loudly on this exact condition kept
  running instead. Fixed: both now return 1 on that path, 0 on success
  (type hints corrected `-> None` to `-> int` to match).
- 11 new tests: `TestTargetWaterfallXY` (4, the coordinate math in
  isolation), `TestRunWaterfall`/`TestRunChip` additions (target-box/
  context-image integration, plus the no-match-returns-1 regression).
  Full suite: 1441 passing / 36 skipped (was 1432). Version bumped
  0.4.11 → 0.4.12.

##### CLI subpackage follow-up 3 — chip's --min-confidence never reached the COCO/YOLO writers (2026-09-16, same day)

Found running `chip --min-confidence unknown --with-context` against real
Klein SDF data end to end: real target crops were written for all 36
correlated targets, but `targets.coco.json`/`targets.yolo` came back
completely empty (0 images, 0 annotations) despite that.

- **Root cause: two independent confidence filters, only one of which
  `run_chip` actually controlled.** `run_chip`'s own `--min-confidence`
  filters the *target list* before correlation — already fixed to accept
  `unknown` in the previous follow-up. But `write_coco()`/`write_yolo()`
  each have their *own* separate `min_confidence: str = "verified"`
  default, applied again when building the COCO/YOLO entries — and
  `run_chip` never passed `args.min_confidence` through to either call.
  Klein's `.tgt` reader unconditionally sets `position_confidence=
  "unknown"` on every target (no vendor data exists to assess it any
  other way), so this second, unreachable filter silently discarded
  100% of entries regardless of what `--min-confidence` was set to.
- **Fixed:** both `write_coco()`/`write_yolo()` calls in `run_chip` now
  pass `min_confidence=args.min_confidence` explicitly.
- **Real-data confirmation:** the same 36-target Klein SDF run that
  previously wrote "COCO export: 0 image(s), 0 annotation(s)" now
  writes 36 images / 84 annotations (COCO) and 36 label files (YOLO).
- 1 new test (`test_min_confidence_reaches_coco_yolo_writers`). Full
  suite: 1442 passing / 36 skipped (was 1441). Version bumped
  0.4.12 → 0.4.13.

##### CLI subpackage follow-up 4 — chip's 3 parallel styles + selectable render level + dual-channel context (2026-09-16, same day)

User feedback after reviewing the first full-survey runs: liked the
`--with-context` QA images but wanted both channels shown, not just the
target's own side; wanted three separate COCO/YOLO datasets from one run
(tight chip, single-channel full swath, dual-channel full swath, the
latter two square); and wanted the option to render any of those from
raw/radiometric/geometric/georef data, geometric/georef putting the
image into real meters-square instead of pixel/sample space.

- **`--with-context` now combines port+starboard** (`_combine_channels_
  mirrored` + `_target_combined_col`, the same helpers `progression`
  already used) before calling `render_target_context` — no change to
  that function itself, since it only ever needed a plain 2D array plus
  a `(row, col)` position, both of which now describe the combined
  array's own space instead of one channel's.
- **`chip` gained `--styles {chip,single-swath,dual-swath}`** (default:
  all three) — each gets its own `output/<style>/` with its own
  `target_<id>.png` files, `contact_sheet.png`, `targets.coco.json`,
  `targets.yolo/`. `single-swath`/`dual-swath` reuse `export.windows.
  extract_full_channel()` (already built for `progression`'s full-scale
  stages) — native square window resized to `--swath-size-px` *before*
  the percentile stretch/uint8 conversion, a deliberate compromise for
  memory (a native dual-swath crop can be ~9500x9500 samples on a real
  JSF survey) that trades a small amount of stretch fidelity for keeping
  a `--normalization-scope shared` batch's memory bounded — documented,
  not silently different from `chip` style's own native-resolution
  shared-stats behavior. `dual-swath`'s co-target lookup now spans both
  physical channels (grouping changed from `(file, freq, channel)` to
  `(file, freq)`, with per-channel and channel-combined target lists
  built once per group) — needed since a dual-swath window can show a
  target from *either* channel, not just the primary's own side.
- **`chip` gained `--render-level {raw,radiometric,geometric,georef}`**
  (default `radiometric`, matching prior behavior exactly) — applies to
  every requested style's COCO *and* YOLO output alike. `geometric`/
  `georef` reuse `export.geometric_window.extract_geometric_window()`/
  `export.georef_window.extract_georef_window()` (`progression`'s own
  per-stage extractors), replicating that function's established
  centering/jitter/resolution conventions per style: `chip` centers on
  the target (jittered) at a resolution matching its native pixel
  footprint; `single-swath`/`dual-swath` center on the swath's own
  real-world extent (target located within it via `locate_point_*`, not
  assumed centered) at a resolution derived from the real swath width.
  Both require that correction to already exist in the source `.ss.h5`
  (via `sidescan-correct --geometric`/`--georef`) — requesting one
  against a file that never had it fails per-target with a warning
  (`KeyError`/`ValueError` caught), not a crash; a style that ends up
  with zero windows this way skips its own COCO/YOLO write with a clear
  warning instead of shipping an empty dataset silently.
- **Known, documented limitation:** unlike the pixel-space styles,
  `geometric`/`georef` windows only ever annotate the *primary* target,
  not other cataloged targets that happen to also fall inside the same
  resampled grid — checking every target's real-world position against
  an arbitrary resampled extent is materially more work than a pixel-
  window bounds check and wasn't asked for.
- **Real-data verification, `survey-a`'s full 5-file survey**
  (32 cataloged targets): `radiometric` level, all 3 styles + `--with-
  context` — `chip` 32/32, `single-swath` 32/32, `dual-swath` 9/32 (the
  rest skipped with a clear per-target warning — genuinely too close to
  their own file's start/end for dual-swath's ~9500-ping native window,
  `extract_full_channel`'s own documented, pre-existing edge case, not a
  regression). `geometric` level, `chip`+`single-swath`: 32/32 both
  (slow — ~7.75 minutes for 64 total KDTree-resampled windows at a small
  64/96px test size, a pre-existing `extract_geometric_window` cost
  inherited from `progression`, not introduced here). `georef` level,
  `chip`: 32/32 in ~24s (vectorized `resample_channel_to_grid` path, no
  Python-loop KDTree cost). Every output visually confirmed: the context
  image shows a real port+starboard waterfall with the box on the
  correct side; `dual-swath` chips show both channels with the nadir gap
  centered; `geometric`/`georef` chips show real grid-resampled crops,
  not pixel windows.
- 20 new tests (`tests/unit/cli/test_export.py`, new `_swath_friendly_
  survey`/`swath_ss_h5_dir` fixtures — proportioned oppositely from the
  existing `_geo_friendly_survey`, many pings/few samples, so a
  full-swath style's native window actually fits): default-all-3-styles,
  dual-channel context (via a `render_target_context` call spy, since
  the saved PNG's own pixel size doesn't reflect the source array's
  shape), geometric/georef dispatch (via `extract_geometric_window`/
  `extract_georef_window` call spies), raw-vs-radiometric pixel-content
  divergence, and graceful (not crashing) failure requesting `geometric`
  against a file with no geometric correction. 4 existing `TestRunChip`
  tests updated for the new `output/<style>/` layout (previously flat
  `output/`) and the new required `styles`/`render_level`/
  `swath_size_px` namespace fields. Full suite: 1448 passing / 36
  skipped (was 1428). Version bumped 0.4.13 → 0.4.14.

##### CLI subpackage follow-up 5 — geometric/georef chip crash on a target whose sample index doesn't fit its matched channel (2026-09-16, same day)

Found running the new `--render-level geometric` feature (follow-up 4,
above) against real project-b Klein SDF data for the first time at full
scale: crashed partway through with an uncaught `IndexError` instead of
producing output, on a target that the identical `radiometric`-level run
moments earlier had handled without complaint.

- **Root cause:** `_extract_geo_style_window`'s ``target_across_m``
  lookup does an exact `.isel(sample=t.sample)` against the *matched*
  channel's own array. `t.sample` is native to whichever channel/file
  the target's position was originally captured against — for Klein
  targets, `_match_frequency` has no `ping_meters_per_sample` to prefer
  one frequency channel over another when a target's catalog entry
  doesn't carry it, and picks arbitrarily; if that pick differs from the
  channel the sample index actually belongs to, and the two channels
  have different sample counts (real here: `hz_455000` vs `hz_900000`),
  the index can fall outside the matched channel's own array entirely.
  The pixel-space styles never hit this because `extract_chip`/
  `extract_full_channel`'s own edge-shifting window placement clamps
  silently instead of indexing out of bounds — this was a real, latent
  data-mismatch that geometric/georef's exact index lookup was the first
  thing to actually notice.
- **Fixed:** an explicit `0 <= t.sample < masked.sizes["sample"]` bounds
  check before the lookup, returning `None` (skip this target/style,
  warn, keep going) — matching this same function's existing "don't
  crash the whole run over one bad target" contract for missing
  geometric/georef data. Does **not** fix the underlying frequency-
  matching ambiguity itself (a separate, pre-existing question in
  `_match_frequency`, out of scope for this fix).
- **A second bug this exposed, unrelated to the crash itself:** the
  shell command that ran this used `cmd 2>&1 | grep ... > file 2>&1`
  without `set -o pipefail` — the logged exit status was `grep`'s (0,
  since it matched lines), not the actual crashed `sidescan-export`
  process's, so the failure was silently reported as a success. Caught
  by checking the *file's actual contents* (a raw Python traceback)
  after being suspicious of an empty-looking log, not by the reported
  exit code — a reminder that a piped shell command's own exit code
  doesn't reflect an earlier stage's failure without `pipefail`.
- 1 new regression test (`test_geometric_level_out_of_bounds_sample_skips_gracefully`).
  Full suite: 1449 passing / 36 skipped (was 1448). Version bumped
  0.4.14 → 0.4.15.

##### CLI subpackage follow-up 6 — waterfall rebuilt for whole-survey windows + cross-frequency target projection (2026-09-17)

User feedback after a full end-to-end run: didn't want waterfall segments
restarting at each file boundary; wanted window/overlap control instead
of fixed non-overlapping segments; wanted a fixed-size whole-survey
overview regardless of survey length; wanted raw/radiometric selectable;
and wanted every target visible on *both* frequency channels, not just
whichever one it was actually correlated to, using slant range + ping
number to derive its position on the other channel when the source
catalog only ever marks one.

- **New `sidescanpy.targets.core.project_target_to_frequency()`** — the
  physical trick: two frequency channels share the same ping (both
  pinged in lockstep — confirmed on real data: identical ping count per
  channel within one file) and the same physical world, so a target's
  slant range (`sample * meters_per_sample`) computed on its own
  channel converts cleanly to a sample index on any other channel via
  that channel's own meters/sample — no catalog entry for the second
  channel needed. Returns `None` (not a clamped guess) when the
  projected sample would land outside the target channel's own
  recorded window — a real, physical "this channel's range doesn't
  reach that far" case, not a bug.
- **`sidescan-export waterfall` rebuilt around one lazy, whole-survey
  concatenated dataset per frequency** (new `_concat_survey_frequency()`
  — every `.ss.h5` file in `--ss-h5-dir`, dask-backed, `xr.concat` along
  `ping`, tracking each file's own start-ping offset in the combined
  axis) instead of looping file-by-file and restarting segmentation at
  0 for each one. Output layout simplified to match:
  `output_dir/<freq>/segment_NNN.png` (no more `<file>.ss/` subfolder).
- **`--window-pings`/`--overlap-pings`** replace `--segment-size` —
  `--overlap-pings=0` (default) reproduces the old non-overlapping
  behavior exactly; a caller wanting overlap gets it via one new flag
  instead of pre/post-processing segments themselves. Rejects
  `overlap-pings >= window-pings` explicitly (would never advance)
  rather than looping forever or producing an unbounded segment count.
- **`--overview-rows N`** — additionally writes one `overview.png` per
  frequency: the *already-built* `viz.waterfall.plot_survey_overview()`
  (existed in the library, never wired into any CLI since the old
  `visualization_demo.py` was retired) compressed to approximately `N`
  rows via a computed `ping_stride = total_pings // N`, so every
  survey's overview comes out roughly the same size regardless of
  length, with a red marker line at every file boundary (using the same
  file-offset tracking `_concat_survey_frequency` already computes).
  Not an exact `N` (integer-division stride, not literal `linspace`
  point selection) — `plot_survey_overview`'s existing tick-label and
  break-marker math assumes one uniform stride value throughout, which
  true `linspace` sampling (generally non-uniform spacing after
  rounding) would have broken; a computed stride achieves the same
  practical goal (same approximate size for every survey) without
  touching that already-tested math.
- **`--value-var {raw,radiometric}`** (new; `plot_waterfall()`/
  `plot_survey_overview()` both gained the underlying `value_var`
  parameter, default `"backscatter_intensity"` — exactly today's prior
  behavior) — `raw` plots `backscatter_raw` instead. Combine with the
  already-existing `--range-mode ground` for the "geometric" half of
  the ask; **known limitation, not silently dropped**: `--targets-json`
  still only supports the default sample/ping axes (unchanged, pre-
  existing restriction) — `--range-mode ground` and target-box overlay
  are mutually exclusive in the same run, so getting both means two
  separate `sidescan-export waterfall` invocations (see the consumer
  scripts' own step 4 for the concrete pattern now used).
- **Cross-frequency target overlay wired directly into `run_waterfall`**:
  every matched target is now drawn on *every* rendered frequency of its
  own file — its real native position on the frequency it was actually
  correlated to, `project_target_to_frequency()`'s projection on every
  other one (silently skipped for that one frequency only if the
  projection lands out of bounds there).
- **Real-data verification, `survey-a`'s full 5-file, 64,627-
  ping survey:** overview's file-boundary markers (17547 / 22763 /
  40310 / 57858) matched this survey's already-known per-file ping
  counts (17547, 5216, 17547, 17548, 6769) exactly. A window/overlap run
  (3000/500) produced 26 windows spanning the whole concatenated survey
  with no per-file restart. A target correlated to `hz_1495000` was
  confirmed, visually, to also appear on `hz_547500`'s own rendering at
  the identical ping and a proportionally-scaled sample position (LF's
  coarser meters/sample puts the same physical range at a smaller
  sample index than HF) — the projection math confirmed correct against
  real data, not just synthetic fixtures. A one-ping HF/LF total-count
  mismatch surfaced on this real survey (64627 vs 64626) — handled
  correctly since offsets are tracked per-frequency independently, not
  assumed equal.
- 14 new tests (`tests/unit/cli/test_export.py`, new `two_file_ss_h5_dir`
  fixture for genuine no-restart-across-files coverage; `tests/unit/
  targets/test_core.py`'s new `TestProjectTargetToFrequency`, 8 tests).
  4 existing `TestRunWaterfall` tests updated for the new namespace
  fields and flat `<freq>/segment_NNN.png` output layout. Full suite:
  1463 passing / 36 skipped (was 1449). Version bumped 0.4.15 → 0.4.16.

##### CLI subpackage follow-up 7 — contact sheets: boxes drawn, real square-display bug fixed (2026-09-17)

User feedback reviewing the new chip datasets: wanted bounding boxes on
the contact sheets themselves (not just the individual chip files), and
reported some full-swath thumbnails not appearing square.

- **Root cause of the square complaint, confirmed not a data bug**:
  every underlying `target_<id>.png` file was independently verified
  square on disk (direct `PIL.Image.open(...).size` check, every style,
  every real dataset generated this session) — the distortion was in
  `viz.contact_sheet.contact_sheet()`'s own `imshow(..., aspect="auto")`,
  which stretches each thumbnail to fill whatever box shape the grid
  layout happens to produce for that cell, regardless of the source
  array's own shape. **Fixed**: `aspect="equal"`.
- **`contact_sheet()` gained `boxes`/`box_size` parameters** — a list of
  `Target` per thumbnail (reusing the same `bbox_from_center()`
  convention as COCO/YOLO/`render_target_context`, so all four agree on
  the same box for a given target), drawn as unfilled red rectangles.
  `run_chip` now passes each window's own `co_targets` list through, so
  a busy thumbnail correctly shows every target inside it, not just the
  one the crop was centered on (confirmed on real data: one dual-swath
  thumbnail legitimately shows 8 boxes). `run_survey_window`'s own
  contact sheet call is unaffected (no-target negatives by definition,
  nothing to box).
- **`render_target_context()`'s intentionally non-square wide-aspect
  QA view was left untouched** — that aspect is a deliberate, documented
  choice (matching EdgeTech Discover's own per-target image convention),
  not the same "auto" bug.
- 7 new tests (`tests/unit/viz/test_contact_sheet.py`: equal-aspect
  assertion, mismatched-`boxes`-length validation, boxes drawn/omitted/
  multi-target cases; `tests/unit/cli/test_export.py`: `run_chip` passes
  real `co_targets` through as `boxes`). Full suite: 1470 passing / 36
  skipped (was 1463). Version bumped 0.4.16 → 0.4.17.

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
- **`io/jsf/converter.py` stored `sample_rate_hz` from the JSF header's
  own `_raw_sample_freq` field, which is unreliable** (its own doc
  comment reads "sample freqency (Hz) usually (see Data Format)",
  already hedged) — confirmed on real data (`project-a`'s
  `survey-a` survey) 2.2x (HF) to 3.0x (LF) off from the real
  per-sample rate, survey-specific. This cascaded into every downstream
  range-dependent computation (TVG, geometric ranges, georeferencing,
  gridding) and, most visibly, target-catalog frequency matching — 29 of
  32 cataloged targets in that survey were being exported from the wrong
  frequency channel entirely. Fixed: `sample_rate_hz` now derives from
  `sample_interval` (`1.0 / first.sample_interval`), decoded
  independently from the header's own sampling-interval bytes and
  already correct (`messages.py` had a long-standing `TODO` flagging
  exactly this substitution). `sample_interval` added to
  `_validate_file_level_attrs`'s constancy check alongside the existing
  fields. Regression test:
  `tests/unit/io/jsf/test_converter.py::test_sample_rate_hz_from_sample_interval_not_sample_freq`.
  **Any `.ss.h5` file already converted under the old behavior has a
  wrong `sample_rate_hz` and needs reconversion from raw** — this fix
  only affects future conversions.
- **Chip/context exports (`cli/export.py: run_chip()`) never matched
  EdgeTech Discover's own per-target thumbnails, despite
  `targets.core.orient_for_display()` already existing and already
  validated for exactly this — it was just never called from the chip
  export path** (confirmed via grep: zero references outside
  `io/jsf/targets.py`). Two real display-orientation bugs: (1) a missing
  horizontal port-only mirror, matching this library's own
  `viz/waterfall.py` sign convention (port mirrored, starboard not) that
  chip exports were silently inconsistent with; (2) a missing vertical
  ping-order flip, a genuine Discover-vs-this-library display
  convention difference, scoped to chip/context exports only (not the
  whole-survey waterfall/gridded pipeline, whose own top-down ping
  convention is deliberate and used consistently elsewhere). Fixed via
  new `_orient_pixel_window()` — not a direct call to
  `orient_for_display()`, which assumes one target owns its own window;
  a chip/full-swath window here can carry several co-located targets
  that all need remapping together. Wired into
  `_extract_pixel_style_window()` (raw/radiometric render levels) and
  `run_chip()`'s context-image path; horizontal flip is skipped for
  "dual-swath" windows, which `_combine_channels_mirrored()` already
  port-mirrors at combination time. `geometric`/`georef` render levels
  are real-meters grids, not raw pixel arrays, and are unaffected.
  Tests: `tests/unit/cli/test_export.py::TestOrientPixelWindow` (5
  tests).
- **Chip exports defaulted to `--render-level radiometric` (raw pixel
  space), causing crops to be non-uniformly stretched 5-11x depending on
  vehicle speed at each target** (along-track pixel spacing isn't
  constant in sample/ping space; `--render-level geometric`/`georef`
  resample onto real, uniform meters instead). Not changed as
  `sidescan-export chip`'s own shared default (a real behavior change
  for every consumer, and geometric/georef windows are only trustworthy
  once a survey's `sample_rate_hz` is correct and it's been reconverted
  — see the fix above) — instead pinned explicitly via `--render-level
  geometric` in `project-a/scripts/dataset_20251015_survey-a.sh` and
  `project-b/scripts/dataset_site-b.sh`, the two scripts that
  actually produce each project's final chip/COCO/YOLO dataset.
- **`--render-level geometric` chip exports had three further real gaps,
  found reviewing the first real-data run of the fix above.**
  1. **Vertical orientation still didn't match Discover.** The
     `_orient_pixel_window()` fix above only covers the raw/radiometric
     pixel path — `extract_geometric_window()` builds its grid with row
     0 = the minimum along-track distance (ascending ping order,
     top-down), the same "wrong" direction `_orient_pixel_window` exists
     to reverse for pixel-space, but nothing applied an equivalent flip
     to the geometric path. Fixed: new `_flip_geo_vertical()` in
     `cli/export.py`, applied to every geometric-level chip/single-swath/
     dual-swath window. Verified two ways on real data: (1) an isolated
     no-jitter/no-edge-avoidance test showing the flipped render's
     large-shadow/small-speck vertical arrangement matches EdgeTech's own
     thumbnail (the unflipped render has it backwards); (2) a raw-pixel
     crosshair check confirming the underlying position (ping/sample) is
     itself correct — the shadow trails away from the marked point
     exactly as expected for a real sonar contact.
  2. **A window pushed to the edge of recorded data filled the rest with
     blanks instead of shifting to stay on real data.** Real case: target
     0011 sits almost at the max-range edge of its channel's recorded
     swath, so a window centered exactly on it was half no-data.
     `extract_geometric_window()` gained `avoid_edge_gaps: bool = False`
     (opt-in, not a default-behavior change for the one existing caller's
     other uses) — when `True`, shifts the center inward just enough to
     keep the full window on real data, the same trade-off `extract_chip`
     already makes for raw-pixel windows near an array edge. Wired to
     `True` for chip export specifically. New `_clamp_center()` helper;
     found and fixed a real bug while adding it — an un-cast
     `np.float32` from `along_valid.min()`/`.max()` silently tainted the
     clamped center (and everything derived from it) with a numpy dtype
     `json.dumps` can't serialize, breaking COCO export outright.
  3. **Targets were always exactly centered — `--jitter-px` was never
     actually wired into either dataset script**, despite the CLI
     supporting it since chip export was first built (default `0`,
     opt-in per run). Not a library bug — `project-a`'s/`project-b`'s
     own `dataset_*.sh` scripts simply never passed it. Added
     `JITTER_PX=30` (10% of `CHIP_SIZE_PX`) to both `_config_*.sh` files
     and wired `--jitter-px "$JITTER_PX"` into both `dataset_*.sh`
     scripts.
  4. **Geometric/georef windows never searched for co-occurring
     targets** — `_extract_geo_style_window()`'s own docstring admitted
     this explicitly ("documented, not silently missing"), unlike the
     raw-pixel styles' `targets_within_window()`. Fixed for `geometric`
     specifically: new `_geo_targets_within_window()` (the real-world
     equivalent — translates every other matched target's own along/
     across position into this window's local pixel coordinates, skips
     one that's out of range or off the current channel's sample axis
     rather than crashing the whole run) plus the same flip-and-remap
     pattern as `_flip_geo_vertical()`. `georef` still doesn't have
     this — its row-axis convention (northing-based) isn't obviously the
     same "match Discover" question, needs its own separate look before
     assuming the same fix applies; documented as a known gap, not
     silently missing.
  Real-data re-run, all 32 targets, all three styles: chip 37
  annotations (was 32, one per image with no co-target support), single-
  swath 46, dual-swath 63 — confirms co-target search is finding real
  matches, not just passing an empty-case test. Tests: 8 new in
  `tests/unit/export/test_geometric_window.py` (`avoid_edge_gaps`
  behavior, `_clamp_center` including the float32 regression),
  `TestGeoTargetsWithinWindow`/`TestFlipGeoVertical` (new, 5 tests) and
  a `TestRunChip` integration test proving two co-located targets both
  land in one geometric chip's own COCO annotations, in
  `tests/unit/cli/test_export.py`. Full suite: 1494 passing / 36 skipped
  (was 1482).

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