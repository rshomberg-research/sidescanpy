# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Commits follow [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- `sidescanpy.grids.to_geotiff(raster_or_dataset, path, *, variable=,
  dtype=, nodata=, compress=)`: single-band, north-up GeoTIFF with CRS,
  transform, nodata and attrs as tags, via rioxarray. New optional extra
  `geo` (`rioxarray>=0.15`); a clear ImportError names the extra when it
  is missing. `examples/gridding_and_geotiff.py` now uses it.
- `SideScanGriddedRaster.to_dataset()`: plain `xr.Dataset` on 1-D `y`/`x`
  cell-center axes with the CRS in a CF `spatial_ref` coordinate.
- `sidescanpy.grids.recommend_resolution(ds_or_geometry)`: half the finer
  of median ping spacing and sample interval (the rule
  `sidescan-analyze recommend-config` already used; it now calls this).
- `docs/planning/grids-library-swaps.md`: library-replacement candidates
  in grids/export with the parity test each needs.

- `sidescanpy.export.ml.build_target_dataset(sources, targets, spec,
  out_dir, *, contact_sheet=None, context=None) -> DatasetResult`: the
  whole `sidescan-export chip` workflow as a library call, configured by
  a `DatasetSpec` (`from_dict`/`from_yaml`, unknown keys rejected). It
  writes the same files as before plus a top-level `manifest.json`, and
  returns a typed summary (images per style, skipped targets with
  reasons). Plotting is injected as optional callables, so the builder
  needs no matplotlib. New options: several `resolution_m` values (one
  dataset per `res_<m>m/`), a train/val split (`val_fraction`, off by
  default: targets by id hash, backgrounds by file/ping-block hash) and
  `layout="ultralytics"` (`images/{train,val}`, `labels/{train,val}`,
  `data.yaml`). CLI: `--val-fraction`, `--split-block-pings`, `--layout`,
  and `--resolution-m` now takes several values.
- `sidescanpy.targets.waterfall`: target boxes in waterfall coordinates
  (`waterfall_boxes`, `target_waterfall_position`), and
  `targets.correlate.correlate_catalog`/`place_catalog` (a catalog
  correlated and projected onto every frequency of its file).
- `export.image.save_grayscale_png` and `reference_stats_for_scope`;
  `write_yolo(..., labels_dir=, data_yaml=)`; `build_position_index`
  accepts a list of files.
- Examples: `examples/build_yolo_dataset.py` (synthetic survey and
  catalog to an Ultralytics dataset with backgrounds) and
  `examples/figures/correction_progression.py` (the per-target
  correction-step QA figure, now built from the export extractors).

- Multi-category COCO/YOLO annotations, additive (uncategorized input
  produces byte-identical output):
  - New optional `Target.category`, read/written in targets JSON (key
    omitted when unset) and CSV (`category` column only when any target
    has one); `read_targets_csv(..., category_column=...)` maps any column.
  - `write_coco`/`write_yolo` take `categories=[...]`; otherwise classes
    are the sorted set present (COCO ids 1..N, YOLO 0..N-1, `data.yaml`
    in the same order). A category outside an explicit list is an error.
  - `sidescan-export chip --categories a,b,c` for stable ids across
    separately-exported datasets.
- `--strict-config` on every `sidescan-*` command with `--config`: config
  keys matching no option are an error instead of a warning.
- `sidescan-export chip` dataset-augmentation and projection options, all
  opt-in (defaults reproduce prior output exactly):
  - `--n-augment N`: N independent jitter draws per target per style, each
    its own image (`target_<id>_a<k>.png`). The target neighborhood loads
    once per target; only the resample repeats.
  - `--random-flip`: independent 50% horizontal and 50% vertical flip per
    image, annotations remapped to match; each image's draw is recorded in
    `<style>/augmentations.json`.
  - `--across-axis {ground,slant}` (geometric level): `slant` builds the
    real-meters grid on signed slant range instead of ground-projected
    `across_track_m` (no ground projection, water column kept), so
    `--resolution-m` still means the same real extent on both axes.
  - `--box-size-m`: annotation box in meters for center-only targets,
    converted per image by its own resolution.
  - `--swath-fill` (geometric single-swath/dual-swath, no `--resolution-m`):
    frame spans the across-track extent the data actually fills (far edge
    most pings reach, 10th percentile; dual swath port-far to
    starboard-far rather than nadir-centered), square pixels, no
    ping-spacing floor on resolution. The old frame left black side
    margins (survey-a: median ~20 px, up to 150 px per side; far wider on
    surveys whose ping spacing exceeds swath width / size).
- Under `--normalization-scope image` (default) each image is written as
  soon as it is extracted instead of holding every float window until the
  end; contact sheets show the first draw per target only.
- Background (no-target) dataset images:
  `sidescanpy.export.windows.sample_background_windows` samples window
  centers uniformly by ping (optionally within per-file ping ranges),
  renders them through the same extractors as target windows, and rejects
  any window with a cataloged target (any confidence) inside it plus a
  margin, or with too many NaN cells. `sidescan-export chip
  --n-background N` (default 0) writes `<style>/bg_<n>.png` with empty
  YOLO labels and annotation-free COCO image records
  (`--background-margin-m`, `--background-max-nan-frac`,
  `--background-ping-ranges`). `CocoImageEntry(target=None, ...)` records
  an image with no annotations in `write_coco`/`write_yolo`.
- `sidescanpy.processing.mask_bright_ping_rows` (whole-ping MAD outlier
  masking, from the CLI's progression grid).

### Changed

- `resample_channel_to_grid` aggregates all cells at once
  (`cKDTree.sparse_distance_matrix` in bounded blocks + grouped
  reductions) instead of one `query_ball_point` per cell: 2-10x faster
  aggregation (about 2-4x end to end; the tree build and reprojection now
  dominate), bounded memory at any sample density. Counts, max, median and
  nearest are identical; mean and idw agree to float64 rounding
  (summation order).

- **Breaking:** `sidescan-export chip` is a thin wrapper over
  `build_target_dataset`; chip output is byte-identical (plus
  `manifest.json`).
- **Breaking:** the chip/window machinery moved out of `cli/export.py`
  into the library as public functions: `sidescanpy.export.windows`
  (now a package: `pixel`, `levels`, `neighborhood`, `geo`, `background`),
  `sidescanpy.export.orientation` (with `orient_for_display` and
  `detrend_by_range`, formerly `sidescanpy.targets.display`, removed) and
  `sidescanpy.export.augment`. `sidescanpy.targets.window` moved to
  `sidescanpy.export.target_window`. Chip output is byte-identical.

- `--config` keys that match no option of any subcommand now log a
  WARNING naming the subcommand, instead of being silently ignored.
- Geometric level no longer loads per-sample latitude/longitude into the
  target neighborhood (only georef reads them).

### Removed

- **Breaking:** `sidescan-export survey-window` (superseded by
  `sidescan-export chip --n-background`, which screens windows against
  the target catalog) and `sidescan-export progression` (now
  `examples/figures/correction_progression.py`).

### Known

- `--resample-method griddata` at 1024 px dominates peak memory on dense
  surveys: one dual-swath window on survey-a peaked at 4.3 GB
  (kdtree: 1.9 GB, ~3x faster).

### Fixed

- Gridded rasters were always labeled channel `"unknown"` and
  `frequency_hz=0`: `resample_channel_to_grid` read a `frequency_hz`
  attribute nothing writes. It now reads `freq_hz_start` (what the
  converters write and `freq_key` is built from), e.g. `"455 kHz"`.
- `resample_channel_to_grid(method="idw")`: with a NaN sample in a cell's
  radius, weights were normalized over all samples and then applied
  positionally to the finite ones only (misaligned). Weights now cover
  the finite contributors only; NaN-free cells are unchanged.

#### Geometric resampling fills short navigation stalls (2026-09-29)

Navigation can stall: several consecutive pings keep (nearly) the same
position while the vessel keeps moving, and it resumes without catching
up. `along_track_distance_m()` is flat across them, and every geometric
window stacked those pings at one along-track point, collapsing their
seafloor into a line. On survey-a ~7 pings at each checked
target (42ms apart, distinct data) advanced only ~0.07m in total instead
of ~0.5m: at fine resolution a hard seam cut through the target itself.
New `along_track_distance_m(..., fill_nav_stalls=True)` replaces each
stalled step (below half the local median step) with the local median
step, for runs of at most 20 pings with movement on both sides; longer
runs are treated as a real stop. All three `extract_geometric_window*`
functions and `sidescan-export`'s geometric target placement use it.
Default stays `False`, so waterfall axes and gap/stationarity detection
still see raw navigation. Georef windows use per-sample lat/lon and are
not changed. (A first version interpolated between the surrounding fixes
instead; that can't restore spacing when the fixes themselves never
catch up, and left the seam.)

#### QA images draw each target's real box, not the --box-size placeholder (2026-09-29)

For catalogs that record a real target extent (a `bbox` on the target),
`sidescan-export chip` already wrote that measured size, converted to
output pixels, into COCO/YOLO. But `contact_sheet.png` and
`context/context_<id>.png` always drew the fixed `--box-size` square, so
real annotations looked generated. Both now draw
`extra_attributes["bbox_size_px"]` when present (new
`viz.contact_sheet._target_box_size`, the same choice COCO/YOLO make),
falling back to `--box-size` for center-only catalogs (JSF/Discover
picks). The native-pixel context view carries the target's extent at
1:1. The geometric/georef context view now renders square
(`aspect="equal"`) instead of the fixed 2.56:1 landscape meant for
native-pixel windows, with axes in meters (the export passes the
window's resolution; `render_target_context(resolution_m=...)`) -- they
were labeled meters while counting output pixels. Without a resolution,
geometric/georef axes are labeled as pixels. The geometric/georef title
no longer shows the resampled column as a "sample".

### Changed

#### Geometric export speed: crop before griddata, context reuses the dual-swath window (2026-09-29)

- `extract_geometric_window_griddata()` triangulates only the points
  within 2x `max_fill_distance_m` of the output window, not the whole
  loaded neighborhood (full swath across-track plus along-track slack).
  Delaunay cost scales with point count; a tight chip can be a few
  percent of what's loaded. Cells, NaN areas and box positions are
  unchanged; ~5% of interpolated pixels differ slightly because Qhull
  breaks ties between the two diagonals of each near-regular sample
  cell differently for a different point set (reordering the same
  points does the same, ~half of cells on a regular grid).
- `sidescan-export chip --with-context` at a geometric/georef level
  reuses the dual-swath window when `dual-swath` is also being exported,
  instead of a second, identical griddata pass per target. The context
  view then carries the dual-swath jitter; its box is still the target's
  true position. Seeded outputs are unchanged (the jitter-0 context call
  never drew from the RNG).
- survey-a, one target, dual-swath + context: ~22s of griddata
  -> 11.5s total run.

### Changed

#### `sidescan-analyze recommend-config`: resolution_m = half the finer axis (2026-09-29)

Recommended `resolution_m` is now `min(ping spacing, sample interval) / 2`
instead of `max(...)`, matching the project decision to oversample past
the finest native axis and let the geometric resample (griddata by
default) interpolate. The report still prints the old `max(...)` value
as the no-interpolation alternative, the interpolation ratio along the
coarser axis, and the resulting 300 px chip footprint. Without a
measured sample interval it falls back to half the ping spacing.
Affects every consumer's recommendations, e.g. ~0.06m -> ~0.004m on
survey-a. Size `size_px`/`box_size` to the targets
accordingly.

### Added

#### `sidescan-export chip`: `db` processing level and `--bit-depth 16` (2026-09-29)

- `--render-level db`: native pixel-space crop of `20*log10(backscatter_raw)`
  with no TVG gain, between `raw` (linear) and `radiometric` (dB + TVG).
  The progression grid already had this stage; chip export didn't.
  `--value-var db` does the same for `geometric`/`georef` levels.
  `backscatter_db` is computed per loaded neighborhood, only when one of
  those two flags asks for it, and never written to the `.ss.h5`.
- `--bit-depth {8,16}` (default 8): 16 writes single-channel 16-bit PNGs
  with the same percentile stretch and `--gamma`, via new
  `sidescanpy.export.to_uint16()`. `from_uint8()` now also inverts a
  `uint16` image. Contact sheets stay 8-bit. COCO/YOLO annotations are
  unchanged.

### Fixed

#### Geometric chips: neighborhood sized in meters, no invented seafloor past the swath (2026-09-29)

- Geometric/georef `chip` loaded `size_px` pings around the target, but
  the window covers `size_px * resolution_m` meters. At
  survey-a's ~6cm ping spacing a 300px / 0.1m chip (30m) got
  ~25m of pings, and griddata's nearest fill smeared the top/bottom
  rows into streaks. All geometric/georef styles now size the ping
  neighborhood from the output footprint in meters (plus jitter and
  1.5x slack) via `_geo_decimation`, and load through the same
  block-average path as the swath styles below.
- `extract_geometric_window_griddata()` gained `max_fill_distance_m`:
  cells with no real sample within that distance are NaN instead of
  nearest-filled. Default derives it from the data (1.5x the larger of
  the p95 ping gap, median sample step, and output resolution), so gaps
  between real pings still interpolate; `np.inf` restores never-NaN.
  Real case: 512px/0.1m and 256px/0.4m single-channel chips extend past
  the port swath's max range and across nadir, and were filled with
  wedge/streak artifacts there.

#### Geometric/georef swath exports: block-average before resampling (2026-09-29)

`single-swath`/`dual-swath` (and the geometric `--with-context` view) at
`--render-level geometric|georef` loaded a native-resolution neighborhood
of ~2x n_samples pings with every per-sample array in the corrected
file (~3GB per target on survey-a), then resampled all of it.
One 4-target run reached 44GB RSS. These styles now:

1. Estimate the output resolution (`--resolution-m`, or the same auto
   value as before: ground swath width / `--swath-size-px`).
2. Load the neighborhood in 2048-ping blocks, only the variables the
   resample and nadir mask read, mask each block at native resolution,
   and block-average it to ~2 source cells per output pixel per axis
   (`_SWATH_OVERSAMPLE`).
3. Resample and crop to the final size exactly as before.

Target positions are looked up on the averaged array (error at most
half an averaged cell, i.e. about 1/4 output pixel). The `chip` style
and the pixel-space levels are unchanged and still use native
resolution.

### Changed

#### Geometric chip export gains a selectable `resample_method` (kdtree/binned/griddata), default switched to griddata (2026-09-29)

Direct follow-up to the same-day "scipy binned as the standard" change
above. Real user pushback after seeing production output: the
non-square shape that change introduced ("KNOWN ISSUE" entry below) was
confirmed a hard blocker for training ("That is actually a huge problem
for training. I need them to be square and to have the same pixel
resolution to meters."). Investigated the user's own proposed fix
(oversample well past the finest native resolution, then interpolate)
and explained concretely why it doesn't work: at a resolution finer
than the real along-track ping spacing, interpolation doesn't recover
resolution that was never sampled — it fabricates the large majority of
the image between real, independent measurements, which risks a model
learning the interpolator's own smoothing signature as real seafloor
texture. Built a real-data, established-library comparison (5 methods,
two real targets, same square resolution) and published it before
picking a direction — see this repo's git history/session notes for the
artifact. **User's own conclusion after reviewing the real images,
explicitly overriding the recommendation above: adopt
`scipy.interpolate.griddata` as the default anyway** ("show me the
interpolation. This is exactly what it is for."), with `kdtree`/`binned`
kept available as explicit alternatives.

- New `export/geometric_window.py: extract_geometric_window_griddata()`
  -- same `(grid, meta)` contract as the other two
  `extract_geometric_window*` functions, built on
  `scipy.interpolate.griddata`. Full scattered-data interpolation, never
  NaN: `linear`/`cubic`'s own convex-hull edge NaN is filled by an
  automatic second `method="nearest"` pass. `interp_method` param
  (`"linear"` default, `"cubic"`, `"nearest"`) passed straight through.
  Real-data cost, confirmed by direct benchmark at production settings
  (320x320, ~0.073m resolution, one real target): **~14x slower than
  the KDTree average, ~68x slower than binned** (14.4s vs. 1.1s vs.
  0.2s) — a real, not-yet-optimized cost of using the full scattered
  point cloud for every window rather than a locally-bounded subset;
  worth knowing before a large dataset-scale export.
- `cli/export.py`'s `_extract_geo_style_window()`/`run_progression()`
  gained a `resample_method: "kdtree" | "binned" | "griddata"` param
  (**default `"griddata"`**), reverting the non-square anisotropic-pixel
  workaround from the entry below entirely — **every `resample_method`
  now produces plain square `size_px x size_px` output**, one fixed
  `resolution_m` on both axes, no exceptions. `sidescan-export chip`
  gained `--resample-method {kdtree,binned,griddata}`; `progression`
  gained the same flag. `_flip_geo_vertical()`/`_geo_targets_within_window()`/
  `_orient_geo_window()` keep their `along_size_px` param from the
  entry below (harmless, optional, unused by every current caller now
  that none of them produce non-square output) rather than being
  reverted/removed — still real, still tested, available if a future
  caller needs it again.
- Tests: `tests/unit/cli/test_export.py::TestRunChipResampleMethod` (6
  tests: each option dispatches to its own function, omitting the flag
  defaults to griddata, all three options produce square output) plus
  `tests/unit/export/test_geometric_window.py::
  TestExtractGeometricWindowGriddata` (8 tests). 2 existing `TestRunChip`
  tests updated to spy on `extract_geometric_window_griddata` (the new
  default) instead of `extract_geometric_window_binned`. Full suite:
  1741 passing / 36 skipped (was 1727).
- Real-data re-check, same target that showed the 16.25% NaN regression
  under the entry below: griddata default now produces `320x320`, 0
  near-black rows, target clearly visible -- confirmed via direct pixel
  inspection, not just trusted from the unit tests.

#### "geometric" render-level chip/progression export now uses scipy binned_statistic_2d, with anisotropic along-track pixel resolution (2026-09-29)

Real user direction, after reviewing the new `extract_geometric_window_binned()`
comparison panel directly: "I want to use this as the standard for
geometric correction." `cli/export.py`'s `_extract_geo_style_window()`
(the shared helper behind `sidescan-export chip`'s `chip`/`single-swath`/
`dual-swath` styles) and `run_progression()`'s geometric stage both now
call `extract_geometric_window_binned()` instead of
`extract_geometric_window()` for the `"geometric"` render level.
`"georef"` is unaffected (still `extract_georef_window()`, its own
separate, still-isotropic KDTree path -- Tier 2, a harder, separate fix,
see `docs/planning/anisotropic-crop-search-radius.md`).

**Real quality regression found and fixed before shipping, not shipped
blind:** a first version kept output square (`size_px x size_px`,
matching the prior KDTree behavior exactly) and just swapped the
function call. Real-data check (project-a stand-in, one real target)
showed **16.25% of rows entirely NaN/black** -- a visible venetian-blind
striping pattern -- because `extract_geometric_window_binned()` has no
search-radius/gap-bridging at all (a bin with no exact sample inside it
is just NaN), unlike the KDTree version's `along_search_radius_m`
(tuned to real ping-to-ping gaps specifically to avoid this). Shown to
the user directly (rendered chip + exact NaN-row percentage) before
deciding how to proceed.

**Fix, user-confirmed:** widen the along-track pixel itself
(`along_resolution_m`, same value the KDTree call used to use as
`along_search_radius_m`) so each along-track bin is sized to actually
contain a real sample -- same treatment
`generate_processing_slides.py`'s own "Anisotropic pixels"/"Anisotropic
pixels (scipy binned)" comparison panels already use. Re-verified on the
same real target: 0/213 rows NaN, real target clearly visible, no
striping.

**Known, accepted, deliberately-not-yet-resolved consequence: `"geometric"`
render-level chip/single-swath/dual-swath output is no longer
guaranteed square.** A chip that would have been `320x320` may now come
out e.g. `213x320` (fewer along-track rows, same real extent). COCO/YOLO
width/height are read from the actual array shape, so those stay
correct -- but this hasn't been checked against every downstream
consumer of these chips (e.g. training code that may assume square
input images). Flagged for follow-up, not silently assumed safe.

Two real, pre-existing bugs found and fixed while making this change
(both only reachable once row count and column count can genuinely
differ, which the prior square-only code path never exercised):
`_flip_geo_vertical()`'s vertical-flip math and `_geo_targets_within_window()`'s
co-target row-bound check both used to reuse `size_px` (the column
count) for the row axis too -- renamed/added an explicit
`along_size_px` param to both (and to `_orient_geo_window()`, which
threads it through), defaulting to `size_px` so every existing
square-only caller is unaffected.

`run_survey_window()`'s own `"geometric"` mode (a separate, `survey-window`
subcommand) was found, while auditing every `extract_geometric_window`
call site, to already be missing required `resolution_m`/
`center_along_m`/`center_across_m` arguments -- a pre-existing, unrelated
bug (this call would raise `TypeError` if ever exercised), left
untouched and not conflated with this change.

Tests: `tests/unit/cli/test_export.py` -- 2 existing tests updated to
spy on `extract_geometric_window_binned` instead of
`extract_geometric_window`; 3 new regression tests
(`TestFlipGeoVertical::test_uses_along_size_px_not_column_count_when_non_square`,
`TestOrientGeoWindow::test_uses_along_size_px_for_vertical_flip_when_non_square`,
`TestGeoTargetsWithinWindow::test_along_size_px_bounds_row_independent_of_size_px`).
Full suite: 1727 passing / 36 skipped (was 1724).

### Fixed

#### Port/starboard "geometric" render-level windows were unflipped mirror images of each other (2026-09-29)

Real user report, found while diagnosing a customer-facing slide tool
(a separate, project-local consumer script): different targets in the
same comparison grid showed inconsistent left-right orientation, and
the user correctly traced it to channel handling. Confirmed via direct
real-coordinate inspection, not a guess: `extract_geometric_window()`
builds columns in ascending `across_track_m` order. Port's
`across_track_m` is negative, so ascending order puts the
least-negative (nearest-nadir) value LAST — near-range lands on the
right edge. Starboard's is positive, so ascending order puts the
smallest (nearest-nadir) value FIRST — near-range lands on the left
edge. Every `"geometric"` render-level window for a single physical
channel (`chip`/`single-swath`'s own per-channel crops, and this
family's processing-slide tooling) has been rendering port and
starboard as genuine, uncorrected mirror images of each other — real
target coordinates confirmed it directly (target 0001, port,
`across_min=-16.28m`; target 0002, starboard, `across_min=+15.55m` —
near-range on opposite sides).

- New `cli/export.py: _orient_geo_window()` — wraps the existing
  `_flip_geo_vertical()` (unchanged, still tested independently) with
  an additional horizontal port-mirror flip for a single port
  channel's own window (`combine=False, channel_idx=0`). `combine=True`
  (dual-swath, a pooled real-meters point cloud spanning both true
  signs at once) never needs this — nadir is already correctly placed
  in the middle from the data's own real coordinates, same reasoning
  `_orient_pixel_window` already applies for its own dual-swath case.
  This is a **different reason** than `_orient_pixel_window`'s own
  port-mirror flip, which exists purely to match EdgeTech Discover's
  vendor display convention for pixel-space renders (sample-index
  ordering has no inherent near/far-range sign issue to begin with) —
  there is no Discover convention to match in real-meters geometric
  space; this fix exists purely for genuine near/far-range consistency
  between channels, with no vendor precedent either way. `georef`
  render-level windows are unaffected — northing/easting are absolute
  map coordinates with no port/starboard sign convention to invert.
- The one real call site (`_extract_geo_style_window`'s `"geometric"`
  branch) now calls `_orient_geo_window()` instead of
  `_flip_geo_vertical()` directly.
- 6 new tests (`tests/unit/cli/test_export.py::TestOrientGeoWindow`):
  port gets both flips, starboard gets only the vertical flip,
  dual-swath never gets the horizontal flip even if passed
  `channel_idx=0` directly, target column remapping is correct for
  port and correctly absent for starboard, and a real-data check
  (`_geo_channel_ds()`) confirming a real port window's near-range edge
  actually moves to column 0 after orientation. Full suite: 1709
  passing / 36 skipped (was 1703).
- **Not fixed here, confirmed unaffected:** the `georef` render level
  (no port/starboard sign concept), and the anisotropic-search-radius
  Tier 2 gap already documented in
  `docs/planning/anisotropic-crop-search-radius.md` (unrelated —
  orientation vs. resampling neighborhood shape are independent
  issues).

#### NaN cells render black everywhere, not transparent-onto-white (2026-09-28)

Real user instruction, from the same target-crop verification work as
the anisotropic-search-radius fix below: "make sure that we always use
black for nan across all of our outputs as the standard." No `set_bad`
call existed anywhere in the codebase — NaN rendered via matplotlib's
own default (transparent), which every export path checked composites
over a white background, matching the user's "filled white" report
exactly.

- New `viz/_cmap.py: resolve_cmap()` — the one place a `cmap: str |
  Colormap` argument gets resolved to an actual `Colormap` object,
  applied by default (not opt-in) rather than duplicated per call site.
  Accepts either a registered name or an existing `Colormap` instance;
  always returns a copy via `with_extremes(bad="black")` so the shared,
  registry-wide colormap a bare name resolves to is never mutated.
- Wired into every real backscatter-imagery plotting call:
  `viz/contact_sheet.py` (`contact_sheet()`, `render_target_context()`),
  `viz/waterfall.py` (`plot_waterfall()`, and transitively
  `plot_survey_overview()`), `viz/gridded.py` (`plot_gridded_raster()`,
  `plot_gridded_with_coverage()`, `plot_gridded_multiple_methods()`, and
  transitively `export_gridded_png()`). `cli/export.py`'s cmap-consuming
  call sites all pass through to these functions already — no direct
  `imshow`/`pcolormesh` calls of its own, so no changes needed there.
- 10 new tests: `tests/unit/viz/test_cmap.py` (the helper directly —
  NaN maps to the bad color, bad color is black, finite values
  unaffected, accepts a name or a `Colormap` instance, never mutates
  the shared registry colormap) plus one rendering-level regression
  test per plotting function above confirming the actual returned
  mesh/image's colormap has a black bad color. Full suite: 1703 passing
  / 36 skipped (was 1693).

#### `extract_geometric_window()`'s search radius is now anisotropic — blurry tight chips (2026-09-28)

Real user report, direct follow-up to the earlier `resolution_m` fix:
a chip exported at a tight, correctly-sized `resolution_m` still came
out blurry. Root cause, already anticipated and documented in
`docs/planning/anisotropic-crop-search-radius.md`: the KDTree
averaging neighborhood was a single isotropic radius, and it had to
stay wide enough to bridge real along-track ping gaps regardless of
how fine `resolution_m` was set — so every output pixel, no matter how
small, still averaged over that same wide real neighborhood. More
pixels at a finer resolution just meant more pixels of an equally
over-smoothed image.

- `export/geometric_window.py: extract_geometric_window()` gained
  `along_search_radius_m`/`across_search_radius_m` (an ellipse instead
  of a circle, via axis-rescaling before the `cKDTree` build/query —
  no custom distance metric needed). `search_radius_m` still works
  exactly as before (isotropic); the new params are opt-in and fully
  backward compatible (verified: equal along/across radii reproduce
  `search_radius_m`'s output byte-for-byte).
- `cli/export.py`'s target-crop path (`chip`/`single-swath`/
  `dual-swath` styles, and the `--with-context` QA view — all share
  one code path) now passes `along_search_radius_m` (still
  `max(1.5*resolution_m, gap_p95_m)`, unchanged — has to keep bridging
  real ping gaps) and a much narrower `across_search_radius_m`
  (`max(1.5*resolution_m, 1.5*mps)`, sized to the real sample
  interval) instead of one shared radius for both axes. The `georef`
  render level (northing/easting frame) is unchanged — a real fix
  there needs a heading-oriented ellipse, not just two fixed axes; see
  the planning doc's Tier 2.
- 4 new tests in `tests/unit/export/test_geometric_window.py`
  (isotropic-via-new-params matches old `search_radius_m` exactly;
  narrowing either axis alone increases NaN count without zeroing out
  the whole grid; omitting all radius params still reproduces the
  original default). Full suite: 1693 passing / 36 skipped (was 1689).

#### `--with-context` mislabeled geometric/georef QA views as pixel space (2026-09-28)

Real user report from a full real-data pipeline run: "the context
images look geocorrected, but they are labeled as ping x sample."
`viz/contact_sheet.py: render_target_context()` always labeled axes
"Sample"/"Ping (window-local)" -- correct for the `raw`/`radiometric`
render levels, but the `geometric`/`georef` QA view (added 2026-09-28,
resamples into real-world meters) never updated the labels to match,
so a genuinely correct real-meters image carried leftover pixel-space
axis text. Added `coordinate_system` param (`"pixel"` default,
`"geometric"` -> across/along-track meters, `"georef"` ->
easting/northing meters); `cli/export.py`'s two `render_target_context`
call sites now pass the right one. 3 new tests. Full suite: 1689
passing / 36 skipped (was 1686).

#### Correction-loop progress line was DEBUG-only; per-chunk TVG/geometric/georef summaries weren't (2026-09-27)

Real user complaint, follow-up to the 2026-09-28 CLI logging cleanup:
"too many log messages, impossible to tell progress." Two separate
problems, opposite direction each:

- The 2026-09-28 cleanup moved the per-file `[i/N] <name> (elapsed:
  ...)` progress line to DEBUG along with everything else — but that
  line *is* the "what file and how far along" signal, so hiding it was
  backwards. Promoted back to INFO in all three places it exists:
  `cli/_common.py`'s chunked and non-chunked correction loops, and
  `cli/convert.py`'s own loop.
- The real dominant noise source wasn't touched by that cleanup at all:
  `processing/radiometric.py: apply_tvg()` (2 calls),
  `processing/geometric.py: compute_geometric_ranges()`, and
  `processing/georeferencing.py: compute_georeferencing()` each log an
  "applied successfully" line at INFO on *every call* — fine for the
  non-chunked path (once per file), but `sidescan-correct
  --ping-chunk-size` (`processing/chunked.py: correct_file_chunked()`)
  calls all of these once **per chunk**, not once per file. A 5000-ping
  file at `--ping-chunk-size 512` produced ~10 chunks x 4 lines = ~40
  lines from correction alone, per file, with no `-v` involved. Demoted
  all four to DEBUG.
- Real-data verification: 5 real `project-a` files through
  `sidescan-correct --tvg --geometric --georef --ping-chunk-size 512`
  produced 24 lines total (one `[i/N]` line per file + final summary +
  unrelated pre-existing warnings) — previously this would have been
  dozens of lines per file from the chunked per-step summaries alone.
  Full suite unaffected (no test asserts these specific log levels):
  1686 passing / 36 skipped.

#### `recommend-config`'s `resolution_m` now checks the range-axis sample interval too, not just ping spacing (2026-09-27)

Real user question: "why is the recommended resolution equal to the
ping spacing -- should it be at least 1/2 that for Nyquist, and why not
use the sampling resolution instead since that's much lower?"
`measure_ping_geometry()` measured only the along-track ping spacing
and `render_recommended_config()` emitted it directly as `resolution_m`
-- silently assuming ping spacing is always the coarser (limiting) of
the sidescan swath's two anisotropic axes, without ever measuring the
across-track range-axis sample interval to check. That assumption
usually holds (ping rate is Hz-scale, sample rate is kHz-scale --
confirmed on real data below, ~12x apart) but was never verified, and
didn't match `sidescan-export chip`'s own runtime auto-calculation
(`cli/export.py`), which already computes `max(mps, ping_spacing_m)` --
the correct principle for a single isotropic output pixel size: it
can't be finer than either axis actually supports without inventing
detail on whichever axis is coarser.

(Nyquist itself argues the opposite direction from "half": a sampled
signal's smallest *reliably resolvable* feature is about 2x the sample
spacing, not less than it -- if `resolution_m` meant "smallest
resolvable feature" a literal Nyquist correction would make it
*coarser*, not finer. What `resolution_m` actually is here is an output
pixel/grid size fed straight to `sidescan-export`'s resampling, where
the established convention -- already implemented in `export.py`, this
fix just brings `recommend-config` in line with it -- is 1:1 with
whichever native axis is coarser.)

- `analysis/recommend.py: measure_ping_geometry()` now also returns
  `sample_interval_m` -- measured directly off `slant_range_m` (constant
  per ping, since it's linear in sample index) rather than recomputed
  from `sound_vel_ms`/`sample_rate_hz` a second time, so it stays
  correct regardless of how the caller's geometry was computed.
- `render_recommended_config()` now emits `resolution_m =
  max(ping_spacing_m_median, sample_interval_m)`, with both measured
  numbers shown as evidence and the limiting axis named -- not just
  ping spacing silently assumed. Backward compatible: a `geometry` dict
  without `sample_interval_m` (or `None`) falls back to the prior
  ping-spacing-only behavior exactly.
- Real-data check, `project-a` `20251015170335.ss.h5`/`hz_1495000`,
  pings 0:3000: ping spacing 0.0772 m vs. sample interval 0.0062 m --
  ping spacing is indeed the coarser axis on this survey (as usually
  expected), so the emitted number is numerically unchanged from
  before, but now measured and shown, not assumed.
- Tests: `tests/unit/analysis/test_recommend.py` (4 new, including an
  edge case where the sample interval actually is the coarser axis, to
  prove `max()` doesn't just always pick ping spacing by habit). Full
  suite: 1686 passing / 36 skipped (was 1682).

### Added

#### `extract_geometric_window_binned()` -- scipy-native alternative resampling, same contract (2026-09-29)

Real user request, direct follow-up to the anisotropic-pixel-resolution
addition below: once each axis gets its own pixel resolution, each
output cell is really just "mean of every real sample whose (along,
across) position falls inside this cell's rectangle" -- an ordinary 2D
binning, not a radius search. `export/geometric_window.py`'s existing
`extract_geometric_window()` computes this via a hand-written Python
loop over `cKDTree.query_ball_point()` per output cell; per this
project's own established-libraries-first convention (see
`sidescanpy/CLAUDE.md`'s "Dependencies" section, added same day),
`scipy.stats.binned_statistic_2d` already does exactly this in one
vectorized call.

- New `export/geometric_window.py: extract_geometric_window_binned()`
  -- same signature/contract as `extract_geometric_window()` (same
  `(grid, meta)` return, same `meta` keys, drop-in interchangeable with
  `locate_point()`), minus the `search_radius_m`/
  `along_search_radius_m`/`across_search_radius_m` params, which have
  no equivalent here -- a point either falls inside a cell's exact
  rectangle or it doesn't, there's no overlap/gap-bridging radius to
  tune. **Not a superset, a genuinely different resampling
  philosophy** -- see the function's own docstring for why a bin
  finer than real ping-to-ping spacing shows real empty cells instead
  of the KDTree version's radius-bridged fill, and why the two
  methods' output values are NOT expected to closely agree even at a
  coarse shared resolution (KDTree's default search radius,
  `1.5 * resolution_m`, is centered per-cell and by construction wider
  than the cell itself, so it always pools a different, larger,
  overlapping neighborhood than binned's exact rectangle).
- Re-exported from `sidescanpy.export`.
- Tests: `tests/unit/export/test_geometric_window.py::
  TestExtractGeometricWindowBinned` (9 tests) -- mirrors
  `TestExtractGeometricWindow`'s/`TestAnisotropicPixels`'s own cases
  (shape, finite-values, centering, all-water-column raises,
  non-square output, `avoid_edge_gaps`, `locate_point` compatibility)
  plus a real-fixture same-frame comparison against the KDTree version
  (confirms both place the window identically and both produce finite
  output over the same real data -- deliberately not asserting the
  grid *values* match, per the different-philosophy note above; an
  earlier version of this test wrongly expected close numeric
  agreement and failed against this fixture's synthetic random
  backscatter, which has no real spatial smoothness for the two
  differently-shaped neighborhoods to converge over). Full suite: 1724
  passing / 36 skipped (was 1715).

#### `extract_geometric_window()` gains independent along/across-track pixel resolution -- non-square output (2026-09-29)

Real user report, direct follow-up to the anisotropic-search-radius
fix: even with `along_search_radius_m`/`across_search_radius_m` set
correctly, a single shared `resolution_m` still forces the SAME pixel
density on both axes. On a survey whose along-track native resolution
is much coarser than across-track (ping spacing an order of magnitude
or more wider than sample interval — real vendor-format data measured at
~16:1), the along-track search radius floor (tied to real ping-to-ping
gaps, independent of `resolution_m`) ends up wide relative to a fine
shared pixel spacing, visible in the output as elliptical averaging
footprints — "you can clearly see the ellipses which appear wide...
that is not an issue w/ the radiometric [pixel-space resize], which
looks consistently smooth." No amount of tuning a single shared
`resolution_m`/search radius eliminates this, because a plain
pixel-space resize never averages a real neighborhood at all, so it
has nothing to show a footprint of — it was never solving the same
problem to begin with.

- `extract_geometric_window()` gained `along_resolution_m`/
  `across_resolution_m`/`along_size_px`/`across_size_px` — all
  optional, default `None` falls back to the shared `resolution_m`/
  `size_px`, so every existing caller gets byte-identical square-pixel
  output (verified: `test_omitting_new_params_reproduces_square_output_exactly`).
  Setting `along_resolution_m` coarser than `across_resolution_m` (close
  to the real ping spacing) lets each axis's pixel size match its OWN
  real resolving power — eliminating the visible-footprint artifact
  without averaging away real across-track detail that IS there to
  resolve. The returned array is genuinely non-square in shape when
  used this way (e.g. `(along_size_px, across_size_px)` = `(20, 320)`
  covering the same real square extent as a 320×320 window at the finer
  resolution alone) — most plotting libraries' default aspect handling
  (e.g. matplotlib `imshow`'s `aspect="equal"`) renders this correctly
  from the array shape alone, no `extent` needed.
- `meta` now always carries `along_resolution_m`/`across_resolution_m`
  (the resolved per-axis values, even in the isotropic case) — `locate_point()`
  now reads these instead of the single `resolution_m` key. `meta["resolution_m"]`
  stays populated as a legacy key (equal to `along_resolution_m`) for
  any existing caller reading it directly.
- 6 new tests (`tests/unit/export/test_geometric_window.py::TestAnisotropicPixels`):
  omitting the new params reproduces square output exactly, independent
  resolution produces a genuinely non-square array, `meta` carries both
  resolved resolutions (plus the legacy key), overriding only one axis
  falls the other back to the shared `resolution_m`, both axes cover
  the real extent they were asked to despite very different pixel
  counts, and `locate_point()` uses the correct per-axis resolution.
  Full suite: 1715 passing / 36 skipped (was 1709).
- **Not yet wired into any consumer project's own export tooling** —
  this is a library capability; a caller has to actually choose
  `along_resolution_m`/`across_resolution_m` values and decide how to
  display a non-square result. `docs/planning/anisotropic-crop-search-radius.md`'s
  Tier 2 (georef) gap is unrelated and still open.

#### `sidescan-analyze nav-attitude` -- navigation/attitude diagnostic plot (2026-09-27)

New subcommand, independent of the 5-stage TVG/geometry investigation
pipeline: plots heading/pitch/roll/altitude/depth/heave and the lon/lat
position track for one ping range, entirely from the aux-stream fields
already on every `.ss.h5` channel -- no backscatter, no geometric ranges
needed. Motivated by a heading/yaw value reported outside both
conventional ranges ([0,360) compass bearing, [-180,180) signed --
this library's own convention is compass bearing, CW from N, see
`CLAUDE.md`'s "Sign Conventions") during an upstream conversion step's
own plausibility checks: a human needs to see the actual
heading/attitude/position trace for the affected ping range to tell a
real sensor/conversion problem apart from an ordinary survey-line turn.

- `sidescanpy.analysis.nav_attitude.nav_attitude_summary()` -- per-field
  min/max/mean/NaN-count for heading/pitch/roll/altitude/depth/heave,
  plus a heading-convention check against `heading_deg` (or any field
  via `--heading-var`) mirroring the compass-vs-signed bucket test;
  `classification` is `"compass_bearing"`/`"signed"`/`"ambiguous"`/`None`.
- `sidescanpy.viz.nav_attitude.plot_nav_attitude()` -- heading (with the
  two convention bands shaded and any out-of-both-range points marked
  red), pitch+roll, altitude+depth+heave, and a ping-colored position
  track, sharing one ping-number x-axis across the first three panels.
- `sidescan-analyze nav-attitude --input <.ss.h5 or dir> --pings
  start:stop --output-dir <dir>` -- writes `metrics.json` +
  `<stem>_<freq>_nav_attitude.png`, same convention as every other
  per-tool subcommand here (`nadir`/`beam-geometry`/`usable-range`/
  `units`).
- Tests: `tests/unit/analysis/test_nav_attitude.py`,
  `tests/unit/viz/test_nav_attitude.py` (16 new). Full suite: 1682
  passing / 36 skipped (was 1666).

### Changed

#### CLI logging: per-item progress moved to DEBUG, INFO reserved for milestones (2026-09-28)

Real user complaint: a full field pipeline run (many missions x
part files x analyze tools) produced an overwhelming amount of INFO
output, and there was no way to quiet it down -- every CLI subcommand
already defaulted to `logging.INFO` (not `--verbose`), but essentially
nothing in `convert.py`/`_common.py`/`analyze.py` used `logger.debug`
at all, so there was no level to actually filter on.

- `convert.py: run_survey()`, `_common.py: process_files_in_place()`/
  the chunked-correction loop -- per-file `[i/N] <name> (elapsed: ...)`
  progress lines and per-file success/skip lines moved to DEBUG. The
  "Found N file(s)" line and the final `Success/Skipped/Failed` summary
  block stay at INFO -- these are the real milestones for a batch run,
  not each file individually.
- `analyze.py`: every per-tool subcommand (`nadir`/`beam-geometry`/
  `usable-range`/`units`/`tvg-fit`/`segment`/`investigate`) now logs
  exactly one INFO-level result line (classification + the one or two
  headline numbers) instead of a multi-line `"="*70`-bordered report
  block -- the per-metric breakdown, `note:` lines, and `wrote <path>`
  lines all moved to DEBUG. `tvg-fit`'s prior 12-line block collapses to
  one line: `<file> <freq>: tvg-fit classification=... best_candidate=...
  rmse=...dB`. Real effect on a pipeline run with N missions x M part
  files x 5 analyze tools x K frequencies: INFO output drops from
  roughly one block per (mission, file, tool, frequency) to one line per
  (mission, tool, frequency) -- part-file-level and per-metric detail is
  still there under `-v`/`--verbose`, just not printed by default.
- No behavior change to what's computed or written to disk -- this is
  purely which log level each existing message uses. `-v`/`--verbose`
  (already wired to `logging.DEBUG` on every subcommand) recovers the
  full previous verbosity exactly.
- Full suite: 1666 passing / 36 skipped (unchanged -- no test asserted
  on any of the demoted messages' log level; one existing test
  (`test_export.py`'s `test_matched_target_gets_a_box_drawn`) already
  asserted an unrelated INFO message via `caplog.at_level("INFO")` and
  still passes unchanged).

### Added

#### `sidescan-export chip --value-var` -- raw vs. radiometric, independently of the geometric/georef coordinate choice; `--with-context` follows `--render-level` (2026-09-28)

Real user request: coordinate system (native pixel vs. geometric/georef
real-meters) and value processing (raw vs. TVG-corrected) were
conflated into one `--render-level` enum -- `raw`/`radiometric` only
ever differed in *values* (both native pixel space), `geometric`/
`georef` only ever differed in *coordinates* (both hardcoded
radiometric). There was no way to get geometric coordinates with raw
values, or vice versa. Also asked directly: does `--with-context`
support a geometric/georef view at all? It didn't -- always a fixed
native-pixel-space, always-radiometric QA snapshot, regardless of
`--render-level`.

- New `--value-var {raw,radiometric}` (default `radiometric`, matching
  every existing caller's behavior exactly) on `chip` -- selects
  `backscatter_raw` vs. `backscatter_intensity` for geometric/georef
  chip styles and for `--with-context` in either coordinate mode.
  `extract_geometric_window`/`extract_georef_window` (in
  `sidescanpy.export`) already had a `backscatter_var` parameter for
  this at the library level -- only the CLI wiring (`_extract_geo_style_
  window`, `run_chip`) was missing. No effect (logged, not an error,
  matching `--resolution-m`'s own existing pattern) under `--render-level
  raw`/`radiometric`, which already fully determines this via
  render-level itself.
- `--with-context` now follows `--render-level`'s own coordinate choice
  instead of being hardcoded to native pixel space: under `geometric`/
  `georef`, it reuses the same "dual-swath" extraction (both channels
  pooled into one real-meters grid, sized via `--swath-size-px`) with
  `jitter_px=0` so the drawn box sits at the target's TRUE position
  (chip's own jitter is a training-data augmentation, not appropriate
  for a QA view). Under `raw`/`radiometric`, unchanged -- still the full
  loaded pixel-space neighborhood, now honoring `--value-var` instead of
  being hardcoded to `backscatter_intensity`.
- Tests: `TestRunChip` gained 4 -- context under `geometric`/`georef`
  writes a real, non-empty image; `--value-var raw` produces genuinely
  different chip pixels than `radiometric` at the same
  `render_level="geometric"`; `--value-var raw` under `render_level=
  "radiometric"` has no effect and logs why. Full suite: 1666 passing /
  36 skipped (was 1662).
- Version bumped 0.4.38 -> 0.4.39.

### Fixed

#### Lazy-read chunk sizes across the CLI weren't multiples of the on-disk HDF5 chunk size (2026-09-28)

Real user report: a `UserWarning` ("The specified chunks separate the
stored chunks along dimension 'ping'... This could degrade
performance...") fired on every `sidescan-analyze segment` call against
real data -- and, once traced, on several other commands too. Root
cause: `write_survey()` stores each frequency channel's backscatter on
disk chunked at `HDF5_CHUNK_PING` (256) pings; every lazy
`read_survey(..., chunks={"ping": N})` call across the CLI is a
*virtual* re-chunking request, and when `N` isn't a multiple of 256,
`xr.open_dataset` has to split real HDF5 chunks to satisfy it --
genuinely degraded read performance, not just console noise.

- `cli/analyze.py`: every one of `segment`/`nadir`/`beam-geometry`/
  `usable-range`/`units`/`tvg-fit`/`recommend-config`'s lazy reads
  hardcoded `chunks={"ping": 500}` -- replaced with a shared
  `_PING_CHUNKS = {"ping": HDF5_CHUNK_PING * 2}` (512) constant.
- `cli/export.py`: `_DEFAULT_SEGMENT_SIZE` (shared by `waterfall
  --window-pings` and, more consequentially, `grid --segment-size`,
  which feeds directly into a lazy read) was `2000`; `run_track`'s own
  chunk size was also `2000` independently. Both now
  `HDF5_CHUNK_PING * 8` (2048).
- User-facing tunables that legitimately vary per machine/mission
  (`sidescan-correct --ping-chunk-size`, `sidescan-export grid
  --segment-size` when explicitly overridden) are NOT forced to a
  specific value -- picking an aligned one is now documented as the
  right way to tune them, not silently overridden.
- Tests: `TestPingChunksAlignment` added to both `test_analyze.py` and
  `test_export.py` -- the relevant constant is an exact multiple of
  `HDF5_CHUNK_PING`, and reading a real >1-chunk survey with it raises
  no alignment warning. Full suite: 1662 passing / 36 skipped (was
  1658) -- the only remaining instances of this warning anywhere in the
  suite are `test_chunked.py`'s own deliberately-tiny (3-7 ping) test
  fixtures, unrelated to production code.
- Version bumped 0.4.37 -> 0.4.38.

#### `sidescan-export chip`'s ping-fallback target correlation assumed every file's ping axis starts at 0 (2026-09-27)

Real bug, found from a downstream consumer's multi-part-converted
mission (see the "Added" entry above -- built the same day for the same
underlying need): `_correlate_target()`'s ping-based fallback (for a
target with no recorded GPS lat/lon, or one too far from the survey's
own track -- the exact case `vendor_hdf5`'s vendor-format `targets.csv`
targets always hit, since that catalog has no GPS column at all) checked
`0 <= target.ping < n_pings` -- a bare ping-*count* bound that silently
assumed every `.ss.h5` file's own ping axis is locally zero-based. True
for a single, lone file (the only case ever exercised before), false for
ANY directory of several `.ss.h5` files sharing one globally-renumbered
ping/ping_number coordinate -- which includes `sidescan-analyze
segment`'s own `sectionN.ss.h5` outputs, not just the new per-source-file
conversion output that actually triggered this report. Every target on
such a mission failed to correlate at all ("no ss_h5 ping within Nm of
its recorded position" for every single one, `sidescan-export chip`
producing zero output).

- `_build_position_index()` now also records each channel's own
  `ping_start` (that file's first ping's value on the mission-global
  axis -- 0 for a lone/first file, not always 0 otherwise).
- `_correlate_target()`'s ping fallback converts `target.ping` (always
  mission-global) into the correct LOCAL array index via
  `target.ping - ping_start`, instead of assuming they're the same
  number.
- Tests: `TestCorrelateTarget` gained 3 (`ping_fallback_uses_global_
  ping_start_not_local_zero_based_index`, `..._before_this_files_range_
  returns_none`, `..._across_two_globally_offset_files_picks_the_right_
  one`); the existing 6 updated to set `ping_start` explicitly (default
  0, preserving their exact prior behavior/assertions). Full suite: 1658
  passing / 36 skipped (was 1655).
- Version bumped 0.4.36 -> 0.4.37.

### Added

#### `sidescan-analyze`'s subcommands accept a directory of `.ss.h5` files, same convention as `sidescan-export --ss-h5-dir` (2026-09-27)

Motivated by a real field OOM in a downstream consumer project
(`downstream-project`) converting a large multi-file vendor-format mission: the
container's memory ceiling can't hold one fully-assembled,
concatenated mission dataset in memory during conversion. The fix on
that side is to stop concatenating at conversion time and write one
`.ss.h5` per source file instead -- but every `sidescan-analyze`
subcommand (`segment`/`nadir`/`beam-geometry`/`usable-range`/`units`/
`tvg-fit`/`recommend-config`) only accepted a single file, unlike
`sidescan-export`'s `--ss-h5-dir`, which already lazily (dask-backed)
concatenates a directory of `.ss.h5` files into one logical survey.
Without this, a consumer switching to per-file conversion output would
lose `segment`'s whole point: detecting gaps/instability *across* what
used to be separate files, by looking at one concatenated timeline.

- `cli/_common.py` gained `available_frequencies()` and
  `load_channel_dataset()` (plus `concat_survey_frequency()`, promoted
  from `cli/export.py`'s previously-private `_concat_survey_frequency`,
  unchanged body) -- both accept a single `.ss.h5` file (byte-identical
  original behavior) or a directory of them (lazily dask-concatenated in
  filename/acquisition order). `cli/export.py` now imports the promoted
  function instead of keeping its own copy.
- `cli/analyze.py`: every one of the 7 subcommands above now resolves
  `--input`/`--ss-h5` through these shared helpers instead of a direct
  `read_survey(args.input, ...)` call. `segment`'s own multi-frequency
  write-out loop (needs every channel, not just the one segmentation was
  computed from) works identically for either input shape. No CLI flag
  changes -- `--input`/`--ss-h5` simply accept either a file or a
  directory now (help text updated to say so); every existing
  single-file caller's behavior is unchanged (confirmed: full suite
  green with zero pre-existing test modifications needed).
- Tests: `tests/unit/cli/test_analyze.py` gained `TestDirectoryInput` (6
  tests) -- `nadir` against a directory matches a single-file run
  byte-for-byte on classification/metrics; `tvg-fit`/`recommend-config`
  succeed against a directory; a missing/empty directory raises
  `FileNotFoundError`; and the one that actually matters:
  `segment` against a 2-file directory with a deliberate ~440km position
  jump exactly at the file seam correctly produces two clean segments
  split precisely at that seam, not one falsely-merged 200-ping segment
  (what a gap-blind concatenation would produce) and not two independent
  "section1"s with no relationship to each other (what two separate
  per-file calls would produce, the alternative this capability exists
  to avoid). Full suite: 1655 passing / 36 skipped (was 1650).
- Not done: `investigate` (the older monolithic subcommand, superseded
  by the 7 above for iterative use) was NOT extended -- it dispatches on
  file *format* (`.jsf`/`.sdf`/`.ss.h5`) via `detect_format`, a
  meaningfully different code path; out of scope for what was asked.
- Version bumped 0.4.35 -> 0.4.36.

### Fixed

#### `sidescan-export track` output dimensions weren't consistent across surveys (2026-09-26)

Real user report: track output should all look the same regardless of
how long/skinny or wide a given survey's real track is. It turned out
the figure itself was already created at one fixed size
(`figsize=(12, 10)`) with `ax.set_aspect("equal")` (correctly preserving
real-world geographic proportions, not distorting a skinny survey) --
the bug was in how the file got *saved*, not how it was plotted.

- Both `export_track_and_scanlines_png` (`viz/track.py`) and
  `export_track_with_targets_png` (`viz/targets.py` -- the one
  `sidescan-export track` actually calls whenever `--targets-json` is
  given, i.e. every real mission with an `targets.csv`) called
  `fig.savefig(..., bbox_inches="tight")`, which crops the saved PNG
  down to just the drawn content's own bounding box -- discarding the
  letterboxed blank margins `set_aspect("equal")` reserves for a
  non-square survey extent. A wide/square survey (already filling the
  canvas) saved close to the full 12x10in size; a long/skinny one
  (mostly blank canvas, a thin diagonal strip of real content) got
  cropped down to that thin strip -- so saved files ended up
  inconsistently sized despite the figure always being created the same
  way. Confirmed directly: a real repro before the fix saved
  983x1313px for a wide track vs. 614x1335px for a skinny one from the
  identical code path.
- Fixed: dropped `bbox_inches="tight"` from both calls. Every saved
  `track.png` is now exactly `figsize * dpi` pixels (`1800x1500` at the
  current 12x10in/150dpi defaults), full stop, regardless of the real
  survey's shape -- padding fills the rest, per the user's own explicit
  direction ("keep the geographic accuracy... can use padding to fill
  the space if needed").
- Tests: both `tests/unit/viz/test_track.py` and `test_targets.py`
  gained a regression test asserting identical saved-PNG pixel
  dimensions for a wide-aspect vs. a genuinely long/skinny synthetic
  track -- confirmed each fails against the pre-fix code (983x1313 vs.
  614x1335) before confirming it passes against the fix. Full suite:
  1650 passing / 36 skipped (was 1648).
- **Not touched, flagged only**: `viz/gridded.py`'s `export_gridded_png`
  has the identical `bbox_inches="tight"` pattern -- grid tiling was
  separately discussed the same session (leave the multi-tile chunking
  as-is, just clarify the terminology) without this specific
  save-dimension question being asked, so left alone; worth a follow-up
  if grid output turns out to have the same complaint.
- Version bumped 0.4.34 -> 0.4.35.

#### `find_segments` merge bug + `detect_gaps` now also checks raw ping timing (2026-09-25)

Found retuning `STABILITY_THRESHOLD` against the real `project-a` survey
from the pipeline-demo artifact: raising the threshold to correctly stop
flagging ordinary attitude wobble as "unstable" (see the `segment`
overview finding two entries below) exposed a real, more serious bug
than that entry's own "known coarse edge case" note had characterized.

- **`find_segments`**: merging raw intervals shorter than
  `min_segment_pings` computed each merged range's status from its own
  post-merge overlap with the excluded spans — so a short **excluded**
  span (100 pings, under the 200-ping default) merging into a much
  larger clean interval flipped the *entire* merged range to
  `"excluded"`, not just its own 100 pings. Confirmed on real data: a
  1100-ping range that was ~900 pings genuinely clean and 100 pings
  genuinely unstable came back entirely `"excluded"`. Fixed: status is
  now resolved per raw interval *before* any merging, and only a
  too-short **clean** interval is ever merged into a clean neighbor —
  `min_segment_pings` exists to avoid an unusably-short *clean* output
  section; a short excluded interval has no such requirement (nothing
  downstream reads it) and must never be merged into a clean one.
- **`detect_gaps`** (direct user request, prompted by the real
  segmentation result above turning out to be an attitude, not a
  position/time, event: "I am really looking for discontinuities...
  I suspect they correspond to jumps in time or locations as well") —
  gained a second, independent check: a jump in the raw per-ping `time`
  coordinate itself (`time_gap_factor`, same convention as the existing
  distance-based `gap_factor`), not just along-track distance. A real
  discontinuity (a paused/resumed recording) isn't guaranteed to show
  up in both signals — a logging pause could replay a stale-but-
  plausible position through it, which along-track distance alone
  wouldn't catch. New `sidescan-analyze segment --time-gap-factor`.
- Tests: `TestDetectGaps` gained 2 (an injected time-only jump with no
  position jump, both factors independently controllable);
  `TestFindSegments` gained a direct reproduction of the real bug (a
  short garbage run inside a long clean stretch — confirms the clean
  pings on both sides stay clean, not swallowed into the short excluded
  span). Full suite: 1648 passing / 36 skipped (was 1645).
- Version bumped 0.4.33 -> 0.4.34.

#### `sidescan-analyze segment --pings` — was the only new subcommand with no memory bound (2026-09-25)

Found preparing to demo the new pipeline against a real multi-GB
`project-a` survey file: `nadir`/`beam-geometry`/`usable-range`/
`units`/`recommend-config`/`tvg-fit`/`investigate` all bound how much of
a `.ss.h5` they materialize via `--pings` (default matching
`investigate`'s own 8000), but `segment` had no such flag at all —
`find_segments()`'s detectors need the whole requested ping window's
arrays in memory at once (no per-chunk scanning logic, a separate,
larger piece of work if a survey turns out to need it), so without a
bound it would try to load an entire multi-GB production file.

- `sidescan-analyze segment` gained `--pings` (default `(0, 8000)`,
  same convention as every other subcommand).
- **Second bug found fixing the first**: `write_segments()` needs every
  frequency channel sliced to the exact same `[start:stop]` window
  `find_segments()`'s own (0-indexed-from-`start`) ping numbers are
  relative to — passing the original, full-range survey (the prior
  behavior, harmless only because nothing had ever exercised `start !=
  0` before this fix) would have silently written the wrong pings into
  every section whenever a non-zero `--pings` start was used. Fixed:
  `run_segment` now builds a genuinely `[start:stop]`-sliced
  `SideScanSurvey` (every frequency channel, via `processing.slicing.
  slice_by_pings`) before calling `write_segments()`.
- Test: `tests/unit/cli/test_analyze.py::TestRunSegment::test_pings_bound_offsets_written_section_correctly`
  — writes with a non-zero `--pings` start, reads the section back,
  confirms its ping 0 matches the source survey's true ping 100, not
  ping 0. Full suite: 1645 passing / 36 skipped (was 1644).
- Version bumped 0.4.32 -> 0.4.33.

### Added

#### `sidescan-analyze recommend-config` — evidence-commented recommended config YAML (2026-09-25)

Same session, third increment of the analyze/config redesign (after
`sidescan-analyze segment` and the `nadir`/`beam-geometry`/
`usable-range`/`units` subcommands, both below). Direct user request:
"I would even be interested in analyze generating a config file for
each dataset, but with the opportunity for me to edit it manually, and
the analysis should provide evidence for every decision it suggests."

- New `sidescanpy.analysis.recommend` module. `measure_ping_geometry()`
  is the actual missing measurement behind `sidescan-export chip
  --resolution-m` (added earlier this session) -- real local **median**
  ping spacing and a **percentile-bounded** swath width for an
  already-clean section, robust to a handful of outlier jumps the same
  way `analysis.segmentation`'s own detectors already are (a plain
  mean/exact-span was exactly the bug `--resolution-m` was added to let
  operators route around in the first place). `render_recommended_config()`
  turns a section's already-computed `StageResult`s plus that
  measurement into one hand-formatted (not `yaml.dump`-generated --
  a plain dump can't attach a comment to a specific key) YAML: every
  emitted key states its evidence directly above it, using the same key
  names as real `sidescan-correct`/`sidescan-export` CLI flag dests, so
  the result is directly usable via `--config` once reviewed.
- **Only two stages currently map to a real downstream config key**,
  confirmed by the same audit that motivated this whole redesign:
  Stage 4 (TVG term selection -- `no_spreading`/`no_absorption`, the
  same boolean-disable convention already established in this family's
  own `configs/default.yaml`) and the new geometry measurement
  (`resolution_m`). `nadir`/`beam_geometry`/`usable_range_m`/`units`
  are reported as evidence/context in a comment, not invented as
  settable keys nothing downstream reads yet.
- New `sidescan-analyze recommend-config --input <section.ss.h5>
  --frequency <hz> --output <path.yaml>` -- runs Stage 1/2/2.5/3/4/5
  against one clean section (meant to follow `segment`), using the
  same adopted TVG-fit recipe `tvg-fit` itself uses
  (`r_ref_m="auto"`/`r_fit_min_m="auto"`, `range_var="slant_range_m"`).
  Degrades gracefully, not a hard failure, when Stage 3 can't determine
  `usable_range_m` (the known Klein SDF AGC failure mode already
  documented elsewhere in this file) -- Stage 4 is simply omitted, with
  a comment explaining why, rather than the whole command aborting.
- Tests: `tests/unit/analysis/test_recommend.py` (15) +
  `tests/unit/cli/test_analyze.py::TestRunRecommendConfig` (4). Full
  suite: 1644 passing / 36 skipped (was 1625).
- Version bumped 0.4.31 -> 0.4.32.

#### `sidescan-analyze nadir` / `beam-geometry` / `usable-range` / `units` — per-tool subcommands, not one monolithic pass (2026-09-25)

Same session, second increment of the analyze/config redesign (after
`sidescan-analyze segment` below). Direct user request: "each 'stage' of
analysis is a different analysis tool that I will want to run. The
script is set up to run them all, but the cli should make it easy to
run them individually" -- and analysis needs to be an iterative process
(adjust a config value, re-run one tool, look at the result, adjust
again), which a five-stage-at-once `investigate` pass doesn't support
cheaply.

- Four new subcommands, each `run_investigation()`'s corresponding stage
  on its own against one frequency channel (`--frequency` required,
  matching `tvg-fit`'s existing single-channel convention rather than
  `investigate`'s all/one-frequency choice): `nadir` (Stage 2),
  `beam-geometry` (Stage 2.5 -- runs Stage 2/3 internally as
  prerequisites, matching `stage2_5_beam_geometry()`'s own real
  dependency), `usable-range` (Stage 3), `units` (Stage 1 + Stage 5).
- Each writes only the plot(s) that are the *direct evidence* for its
  own classification -- not the full per-stage diagnostic set
  `investigate` produces (e.g. `nadir` writes just the zoomed
  boundary-detection plot by default; `--verbose-diagnostics` adds the
  full-survey waterfall overlay). `investigate` itself is unchanged --
  still the occasional full five-stage report, not touched or
  refactored, since it already thinly wraps `run_investigation()`.
- Every new subcommand's flags are its own analysis parameters (nadir
  search windows, stability thresholds, bin widths, SNR thresholds --
  the "initial guess" the user described adjusting between reruns) --
  already `--config`-seedable via the existing YAML loader, no changes
  needed there, same mechanism every other subcommand already uses.
- Tests: `tests/unit/cli/test_analyze.py` gained `TestRunNadir` (3),
  `TestRunBeamGeometry`/`TestRunUsableRange`/`TestRunUnits` (1 each).
  Full suite: 1625 passing / 36 skipped (was 1619).
- Version bumped 0.4.30 -> 0.4.31.

#### `sidescan-analyze segment` — cut gaps/garbage/setting-changes out of a mission before analyzing it (2026-09-25)

Same session, direct follow-up to the `--resolution-m` fix and the
resulting audit of what `sidescan-analyze` actually feeds into a real
config decision (see the entry below). The user's own framing: every
other analysis tool (nadir detection, beam geometry, usable range, TVG
fit) is only as trustworthy as the data it's pointed at, and there was
no way to cut garbage, gaps, or mid-file hardware setting changes out of
a mission before running any of them. First increment of a larger
analyze/config redesign (config-driven analysis parameters, per-tool CLI
subcommands, evidence-based recommended-config emission) — this piece
was singled out as the biggest, genuinely prerequisite gap, and is
scoped and delivered on its own rather than bundled with the rest.

- New `sidescanpy.analysis.segmentation` module — four independent
  detectors, each producing real evidence, not just a boolean:
  `detect_gaps` (an along-track distance jump between consecutive pings
  far beyond the survey's own typical spacing), `detect_setting_changes`
  (`valid_samples`, when present, or a `sound_vel_ms` step — see the
  function's own docstring for a real, stated limitation: a raw sample-
  count/range change already padded away by an upstream format-specific
  converter, e.g. JSF/vendor-format as of this writing, isn't detectable
  after the fact from the written `.ss.h5` alone), `detect_garbage_runs`
  (a sustained NaN/zero run across the ping axis, distinct from the
  nadir gap's own per-ping leading blanking run), and
  `detect_unstable_spans` (a rolling attitude/altitude stability check,
  `score_stability` ported from `downstream-project/scripts/apply_config_pipeline.py:
  _section_stability_score` — flags deployment/turn transients like the
  real site-c ping 0-500 case already documented above in this file).
- `find_segments()` combines all four into a list of `Segment`s covering
  the whole file with no gaps between them, each labeled `"clean"` or
  `"excluded"` with human-readable `reasons` — the evidence for the
  decision, not just the decision itself.
- `write_segments()` slices every frequency channel (via
  `processing.slicing.slice_by_pings`, no new slicing logic) to each
  `"clean"` segment's ping range and writes one independent `.ss.h5`
  per segment; `write_segments_report()` writes the full evidence as
  JSON.
- New `sidescan-analyze segment --input ... --output-dir ...` CLI —
  writes `sectionN.ss.h5` per clean segment, `segments.json`, and one
  minimal two-panel overview plot (altitude + pitch vs. ping index,
  boundaries and excluded spans marked — `viz.investigation.
  plot_segments_overview`, deliberately not a full diagnostic report).
  Every detector threshold is a CLI flag, already config-file-settable
  via the existing YAML `--config` loader (no changes needed there).
- Tests: `tests/unit/analysis/test_segmentation.py` (24) +
  `tests/unit/cli/test_analyze.py::TestRunSegment` (4), synthetic
  fixtures with deliberately injected gaps/setting-changes/garbage-runs/
  instability. Full suite: 1619 passing / 36 skipped (was 1591).
- **Explicitly deferred, not this pass** (tracked as the same redesign's
  next increments): pulling `nadir`/`beam-geometry`/`usable-range`/
  `units` out of the monolithic `investigate` into their own
  subcommands; evidence-based recommended-config YAML emission; the
  `apply_config_pipeline.py`/`investigate_vendor-format_dataset.py`/
  `generate_full_survey_overview.py` keep-or-archive decision.
- Version bumped 0.4.29 -> 0.4.30.

#### `sidescan-export chip --resolution-m` — config-driven, not auto-calculated (2026-09-25)

Real field report, direct follow-up to the scale-bar work above:
some `chip`/`single-swath`/`dual-swath` crops (all three, same target,
together) came out "way zoomed out" — mostly black, the real swath
shrunk to a speck in the center. Root cause: under `geometric`/`georef`
render level, every style's `resolution_m` (meters per output pixel)
was auto-calculated per target from `ping_spacing_m` — a window-average
vessel speed over a ping neighborhood sized from the *sample count per
ping* (`n_samples_single`, used as a ping-count bound in `run_chip`, a
holdover from the raw-pixel styles' own square-crop needs, which
geometric/georef extraction doesn't share). A target sitting in a
locally slow/dense-ping region (e.g. where the vessel slowed for a
closer look — exactly where interesting targets tend to get marked)
gets an average pulled from much faster surrounding transit pings,
inflating `resolution_m` and zooming every style out together — chip
included, since its own formula (`max(mps, ping_spacing_m)`) shares the
same inflated `ping_spacing_m`.

Rather than patch the auto-calculation's averaging window (still
fragile — no window size is right for every survey/target), the user
asked for the opposite: dataset/export parameters like this should be
determined by config, not auto-calculated at all, with the `analyze`
step providing the evidence to choose them. This is a first, scoped
step toward that, not the full redesign (tracked separately).

- New `sidescan-export chip --resolution-m METERS` — when set, used
  directly as `resolution_m` for every style under `geometric`/`georef`
  render level, bypassing the per-target auto-calculation entirely.
  Already config-file-settable for free via the existing YAML `--config`
  loader (`cli/_config.py`, no changes needed there). No effect (logged,
  not silently ignored) under `raw`/`radiometric` render level, which
  crops in native pixel space and has no `resolution_m` concept.
- `_extract_geo_style_window()` gained `resolution_m_override` — swaps
  in for both the chip-style and swath-style resolution formulas;
  `ping_spacing_m` itself is still computed either way, since it also
  feeds `bbox_size_px`'s unit conversion independent of which
  resolution the crop renders at (a known, separate, smaller-impact
  imprecision in that conversion — not addressed here).
- Tests: `tests/unit/cli/test_export.py::TestRunChip` gained 3 — the
  override reaches `extract_geometric_window`/`extract_georef_window`
  for every requested style, and is ignored with a logged warning under
  `raw`/`radiometric`. Full suite: 1591 passing / 36 skipped (was 1588).
- Version bumped 0.4.28 -> 0.4.29.

#### Land/water context for georeferenced plots (2026-09-18)
- **`sidescanpy.viz.basemap`** (new module, requires the new optional
  `maps` extra — cartopy) — land/water context for georeferenced plots.
  `add_land_background(ax, bbox, crs)` fills `ax` with a water color and
  draws land polygons behind whatever the caller plots next (a gridded
  raster, a survey track); `plot_locator_thumbnail(lon, lat)` is a small
  zoomed-out map with a marker showing where the main plot sits —
  "where is this, relative to the coast." Land geometry is a vendored
  1:50m Natural Earth land shapefile (`assets/natural_earth/`, public
  domain) — no network access needed at runtime, and coarse by design
  (good for "near land vs. open water" orientation and a regional
  locator, not a substitute for chart-scale shoreline detail). Every
  function raises `ImportError` with an install hint if cartopy isn't
  present, rather than silently skipping the background. Wired in as
  opt-in `show_land`/`show_locator` params on `viz.gridded.plot_gridded_raster()`/
  `export_gridded_png()`, `viz.track.plot_track_and_scanlines()`/
  `export_track_and_scanlines_png()`, and `viz.targets.plot_track_with_targets()`/
  `export_track_with_targets_png()` (default `False` — no behavior
  change for existing callers), plus `--show-land`/`--show-locator` on
  `sidescan-export grid`/`track`. Found wanting investigating
  `project-a`'s `survey-a` gridded/track exports having no
  way to show a survey's location relative to the coastline. 21 new
  tests across `tests/unit/viz/test_basemap.py` (new) and integration
  cases in `test_gridded.py`/`test_track.py`/`test_targets.py`, all
  gated on the optional extra via `pytest.importorskip`. Full suite:
  1516 passing / 36 skipped (was 1497).

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

##### CLI subpackage follow-up 8 — `sidescan-convert targets` now also writes targets.csv (2026-09-17)

`targets.io.write_targets_csv()`/`read_targets_csv()` have existed since
the original target-catalog work, fully tested, round-tripping every
field losslessly except `bbox`/`extra_attributes` (JSON-encoded into
single cells) — but nothing in any CLI command ever called
`write_targets_csv()`. `run_targets()` now writes `targets.csv`
unconditionally alongside `targets.json` (JSON stays the source-of-truth
format; CSV is a portable, human-reviewable secondary view, not an
opt-in extra). 1 existing test extended (`test_jsf_catalog`) to assert
`targets.csv` round-trips too. Full suite: 1470 passing / 36 skipped
(unchanged count — an assertion was added to an existing test, not a new
one). Version bumped 0.4.17 → 0.4.18.

##### CLI subpackage follow-up 9 — ground-mode waterfalls now draw target boxes too (2026-09-17)

Follow-up 6 added target boxes to `sidescan-export waterfall`, but hard-
rejected `--targets-json` whenever `--range-mode ground` was in effect
(that mode always forces `x_axis="across_track"`/`y_axis="along_track"`
via `resolve_waterfall_axes`, and the box-placement helper only knew how
to place a box in the default sample/ping pixel space). Not structural —
ground mode's own coordinates (`across_track_m`, cumulative along-track
distance) are already computed and already present in a georeferenced
`.ss.h5`, just not wired into the box math.

- `cli/export.py: _target_waterfall_xy()` now takes `ds`, the box's
  ping_offset, and the active `x_axis`/`y_axis`, and handles two
  supported pairings: the original `sample`/`ping` (x = ±sample by
  channel, y = the global ping index) and the new `across_track`/
  `along_track` (x = the target's own real `across_track_m` value, read
  directly off `ds` at its own local ping/channel/sample — not
  derivable from sample index alone, since across-track distance is
  altitude/geometry-dependent; y = cumulative along-track distance at
  the target's local ping). Returns `None` — box silently skipped, not
  drawn at a garbage position — when the target's ping falls outside
  the window or (ground mode only) its own across-track value is NaN
  (still in the water column at that exact sample). Any other
  `x_axis`/`y_axis` combination also returns `None`.
- `_export_waterfall_png()` computes the along-track distance array once
  per window (not once per box — `along_track_distance_m()` does a real
  `pyproj.Geod.inv` call over every ping) and passes it through.
- `run_waterfall()`'s `--targets-json` rejection narrowed to only the
  combinations that are genuinely unsupported (any pairing other than
  the two above) — ground mode itself is no longer rejected.
- Tests: `TestTargetWaterfallXY` gained 4 (ground-mode position read,
  out-of-window ping → `None`, NaN across-track → `None`, unsupported
  axis combo → `None`); `TestRunWaterfall` gained
  `test_ground_mode_matched_target_gets_a_real_box` (confirms a real
  match resolves to a non-`None` position under `--range-mode ground`)
  and `test_targets_json_unsupported_axes_still_rejected` (replaces the
  old ground-mode-is-rejected test, now checking a genuinely-unsupported
  combo instead). Full suite: 1475 passing / 36 skipped (was 1470).
  Version bumped 0.4.18 → 0.4.19.

##### CLI subpackage follow-up 10 — `--config` accepts YAML, not just JSON (2026-09-18)

`cli/_config.py` was JSON-only by a documented decision (this codebase's
one interchange format) now superseded by `CLAUDE.md`'s "Consumer-Project
Architecture Contract" section: YAML for anything a human reads or edits
(a `--config` file is exactly that case), JSON reserved for
machine-generated artifacts (`targets.json`, `metrics.json`, COCO
annotations).

- New `cli/_config.py: load_config(path)` — dispatches on extension
  (`.yaml`/`.yml` via `yaml.safe_load`, anything else via `json.loads`,
  so every existing extensionless/`.json` config already in use across
  the family keeps parsing exactly as before). All four CLIs
  (`convert.py`/`correct.py`/`export.py`/`analyze.py`) switched their
  inline `json.loads(config_path.read_text())` call to this, and their
  `--config` help text updated to say "YAML or JSON."
- New dependency: `pyyaml` (already a real dependency elsewhere in the
  family, e.g. `downstream-project`'s `pyproject.toml` — not new to the family).
- Tests: `tests/unit/cli/test_config.py: TestLoadConfig` (5 new — JSON by
  extension, YAML by extension, `.yml`, extensionless falls back to
  JSON, an end-to-end YAML-seeds-argparse-defaults case). Full suite:
  1522 passing / 36 skipped (was 1517).

##### CLI subpackage follow-up 11 — `--frequency` no longer assumes every survey uses JSF/SDF's `hz_<freq>` freq_key naming (2026-09-18)

Found empirically while verifying `sidescan-*` against an vendor-format-format
synthetic mission (`sensor1_data1`-style freq_keys, not `hz_<freq>`) for
`downstream-project`'s single-pipeline-migration plan — `analyze.py: run_tvg_fit`
hardcoded `freq_key = f"hz_{args.frequency}"`, and `export.py`'s
`run_waterfall`/`run_grid`/`_resolve_frequencies` (`run_track`) did the
same for any non-`"all"` request. Not vendor-format-specific: any consumer
whose freq_key convention isn't `hz_<freq_hz_start>` would hit this.

- New `cli/_common.py: resolve_freq_keys(available, requested)` —
  `"all"` returns every available key; otherwise tries `f"hz_{requested}"`
  first (every existing caller passing a bare frequency number keeps
  resolving exactly as before), falling back to `requested` itself as a
  literal freq_key. All four call sites switched to it.
- Tests: new `tests/unit/cli/test_common.py` (4), plus
  `tests/unit/cli/test_analyze.py::TestRunTvgFit::test_literal_freq_key_fallback_for_non_hz_naming`
  (1). Full suite: 1527 passing / 36 skipped (was 1522).

##### Gridding — non-finite projected coordinates dropped with a warning, not a bare scipy crash (2026-09-18)

Also found via the same vendor-format synthetic-mission exercise:
`resample_channel_to_grid()` filters out NaN *WGS84* lat/lon before
reprojecting into the grid's own CRS, but never checked the *projected*
result for finiteness — a physically-invalid latitude (>90, traced in
the triggering case to a corrupted `compute_georeferencing()` result for
a synthetic vessel position that wandered too close to a pole; not
something a healthy GPS fix produces, but not NaN either, so the
pre-transform filter let it through) can project to `inf`, which
`scipy.spatial.cKDTree` then rejects with an opaque "data must be
finite" `ValueError` pointing nowhere near the real cause.

- `resample_channel_to_grid()`: after projecting into a non-geographic
  (projected) CRS, samples whose projected x/y aren't finite are now
  dropped with a logged warning (count + CRS), mirroring the existing
  pre-transform NaN-lat/lon handling; if every sample is dropped this
  way, raises the same clear "No valid sample coordinates in channel"
  `ValueError` the pre-existing all-invalid-coordinates case already
  used, instead of reaching `cKDTree` at all.
- Tests: `tests/unit/grids/test_resample.py::TestResampleChannelToGridUTM`
  gained 2 (`test_physically_invalid_latitude_dropped_not_crashed`,
  `test_all_samples_invalid_latitude_raises_clear_error`). Full suite:
  1529 passing / 36 skipped (was 1527).

##### Gridding — `bbox_from_channel_dataset()` also excludes physically-invalid coordinates (2026-09-18, same-day follow-up)

Found continuing the same exercise: even after the fix above, a channel
with a few physically-invalid-but-finite samples could still crash
`sidescan-export grid` further upstream -- `bbox_from_channel_dataset()`
(used to auto-select a UTM zone before gridding) only filtered
`np.isfinite`, so those samples blew up its computed bbox extent, which
then produced an infinite auto-selected grid cell count and an
`OverflowError` converting it to `int` -- before `resample_channel_to_grid()`'s
own new finite-check ever got a chance to run.

- `bbox_from_channel_dataset()`: latitude/longitude filtering now also
  excludes anything outside `[-90, 90]`/`[-180, 180]`, not just non-finite
  values.
- Tests: `tests/unit/grids/test_definition.py::TestBboxFromChannelDataset`
  gained 2 (`test_physically_invalid_but_finite_coordinates_excluded`,
  `test_raises_when_every_coordinate_is_physically_invalid`). Full suite:
  1531 passing / 36 skipped (was 1529).

##### `__version__` was a hardcoded, long-stale literal -- fixed (2026-09-19)

Found via a real field report: `import sidescanpy; sidescanpy.__version__`
printed `"0.1.0"` right after installing the 0.4.23 wheel from this same
transfer. Root cause: `__init__.py` hardcoded `__version__ = "0.1.0"` as
a plain literal, never updated across the entire 0.4.x version history
-- `pyproject.toml`'s real version and this constant have been two
independent, silently-diverging sources of truth the whole time. Not
just cosmetic: this value gets stamped into real `.ss.h5`
`processing_history` records (`processing/{georeferencing,interpolate,
geometric,farrange}.py` all import and record it) as which library
version performed a correction step -- every one of those records has
been silently wrong.

- `__init__.py` now reads `importlib.metadata.version("sidescanpy")`
  (the installed package's real, build-time-embedded metadata) instead
  of a hardcoded string. Falls back to `"0.0.0+unknown"` only if the
  package somehow isn't installed at all (bare `sys.path` hack) --
  every real caller (editable install, wheel install) has metadata.
- Tests: new `tests/unit/test_version.py` (3) -- matches installed
  metadata, matches `pyproject.toml`'s real version, isn't the old
  stale literal. Full suite: 1534 passing / 36 skipped (was 1531).
- Version bumped 0.4.23 -> 0.4.24.

##### `sidescan-correct --ping-chunk-size` — memory-bounded, ping-chunked correction (2026-09-19)

Same real field report as the memory-split fix in `downstream-project`'s own
history (running `--tvg --geometric --georef` combined holds ~7x the
raw file size in memory before one write): that project's own
`batch script` mitigated it by splitting into 3 separate
invocations (2-4x per step). This is the actual root fix underneath —
each requested step is applied one ping-chunk at a time instead of
across the whole file, so peak memory is bounded by the *chunk* size,
not the file size, at any combination of steps in one invocation.

- New `sidescanpy.processing.chunked.correct_file_chunked()` — reuses
  the exact same `apply_correction_steps()`/`step_already_done()` the
  non-chunked path uses (no separate numerical implementation, so
  chunked results are identical to eager ones, chunk boundaries
  included — verified directly, not assumed). Each chunk is written to
  a temporary file; the chunk files are then lazily reassembled
  (`xr.open_mfdataset`, dask-backed, never fully materialized) and
  written to the real output via the existing `write_survey()`.
- **Real, empirically-measured gotcha, not a guess:** dask's *default*
  threaded scheduler does not stream that final write safely — measured
  peak memory at ~3x the *total* array size regardless of chunking
  (almost certainly concurrent worker threads racing for the underlying
  HDF5 file's write lock, each holding its own already-computed chunk
  while waiting). Forcing `dask.config.set(scheduler="single-threaded")`
  around the write fixed this completely: measured peak memory then
  scales with *per-chunk* size, confirmed across 100/500/2000-ping
  chunks (consistently ~2-3x the chunk's own size, not the file's).
  `correct_file_chunked()` always uses this scheduler for its write.
- Only steps that are genuinely per-ping-independent are supported:
  `tvg`/`geometric`/`range_correction`/`georef`. `crop_farrange` is
  rejected with a clear error — it needs to see a channel's entire
  ping range to pick one good far-range cutoff, so it isn't chunkable
  this way; run it (if needed) via the regular eager path first.
- `cli/correct.py`: new `--ping-chunk-size N` flag, routing through a
  new `cli/_common.py: process_files_in_place_chunked()` (same disk-
  space guard, same per-file success/skipped/failed accounting/logging
  shape as the non-chunked `process_files_in_place()`). Rejected
  together with `--crop-farrange` at argument-parsing time, not left to
  fail deep inside processing.
- Refactored `cli/correct.py`'s own per-survey already-done checks
  (`_tvg_already_done`, etc.) to call the new
  `pipeline.py: step_already_done()` per channel instead of duplicating
  the per-dataset logic — one place this now lives, shared by the CLI's
  per-survey checks and the new chunked path's per-chunk checks.
- Tests: new `tests/unit/processing/test_chunked.py` (7 — numerically
  identical to eager for `{tvg,geometric,georef}` and `{geometric}`
  alone, all-already-done is a true no-op (file untouched, `mtime`
  unchanged), partial-already-done applies only the missing steps
  without re-running `tvg` (checked by exact value equality, not just
  the history entry), rejects `crop_farrange`/unknown steps, handles
  multiple frequency channels), `tests/unit/processing/test_pipeline.py::TestStepAlreadyDone`
  (7), `tests/unit/cli/test_correct.py::TestPingChunkSize`/new
  `TestArgValidation` cases (5). Full suite: 1552 passing / 36 skipped
  (was 1534).
- Version bumped 0.4.24 -> 0.4.25.

##### `sidescan-export grid` was the one exporter still loading eagerly (2026-09-19, same-day follow-up)

Same real field OOM report, extended to the export side on request:
"we should be using the same lazy loading technique for the exports and
dataset generation as well." Checked every exporter, not just grid --
`waterfall`/`track` already lazily concatenate via `chunks=`
(`_concat_survey_frequency`), and `chip` already reads lazily *and*
slices a bounded per-target ping neighborhood before `.load()`ing
(with its own comment citing real 14.8GB files as the motivation) --
both were already doing exactly this. `run_grid` was the one exception:
`read_survey(path)` with no `chunks=` at all, loading the whole file
eagerly before `--segment-size`'s own per-segment loop ever ran,
defeating that flag's entire purpose on a large real survey.

- `run_grid` now reads via `chunks={"ping": args.segment_size}` and
  calls `.load()` on each segment slice explicitly (matching `chip`'s
  own established pattern) -- only that one segment is ever
  materialized, regardless of file size.
- Tests: `tests/unit/cli/test_export.py::TestRunGrid::test_reads_lazily_chunked_by_segment_size`
  (spies on `read_survey`, confirms `chunks={"ping": args.segment_size}`
  is what's actually passed). Full suite: 1553 passing / 36 skipped
  (was 1552).
- Version bumped 0.4.25 -> 0.4.26.

##### `sidescan-export chip` real bugs found on real production data (2026-09-25)

Two real, confirmed bugs from a real field report against real
vendor-format output (`downstream-project`'s `batch script`): a lot of "no
ss_h5 ping within 25.0m of its recorded position" warnings, some
targets missing entirely, and "all the bounding boxes are the same
size where I am expecting multiple sizes."

**1. Every target's COCO/YOLO box was the shared `--box-size`
placeholder, regardless of a real, measured extent.** `CocoImageEntry`
has carried a per-entry `bbox_size` override since 2026-09-08
specifically for catalogs with a real measured extent (e.g. vendor-format's
targets.csv 4-corner columns, unlike JSF's center-only picks) --
`run_chip` never actually populated it, so every entry fell back to the
shared placeholder no matter what `Target.bbox` said.

- New `_with_bbox_size_px(t, scale_x, scale_y)` converts `t.bbox`'s
  `(w, h)` into this specific window's own output pixel units and
  stashes it on `extra_attributes["bbox_size_px"]` (survives every
  `replace()` call downstream unchanged, no signature threading
  needed). Wired in at all 3 places a target's window-local position is
  finalized: the raw-pixel styles (`chip`/`single-swath`/`dual-swath`
  at `raw`/`radiometric` levels -- native window pixels are the same
  sample/ping units `t.bbox` was recorded in, until/unless the final
  resize step changes them) and both geo styles' `geometric`/`georef`
  branches (`t.bbox`'s sample/ping counts converted to real meters via
  `mps`/`ping_spacing_m`, then to this window's own pixel scale via
  `resolution_m`). `None` (a center-only pick) still falls back to the
  shared placeholder exactly as before.
- Tests: `tests/unit/cli/test_export.py::TestRunChip` gained 4 --
  exact-value check for the pixel-style path (chosen `size_px` avoids
  any resize, so no scaling ambiguity), a center-only-pick-still-uses-
  placeholder regression, and two-different-targets-get-two-different-
  sizes for both the pixel-style and geo-style (the actual render level
  `downstream-project`'s own pipeline uses) paths.

**2. `_correlate_target` required a GPS lat/lon match for every target,
silently dropping any target with no recorded GPS position (or one that
didn't match closely enough) -- even when its real, already-known
`ping` position (e.g. from a pixel-based source catalog, already
correctly projected onto the survey's global ping index before this
ever runs) was sitting right there on the object.** "No ss_h5 ping
within Nm" was the wrong diagnosis on a real production run for a
target that had no recorded *GPS* position to compare against at all.

- `_correlate_target` now falls back to `target.ping` when GPS matching
  is unavailable or fails -- but only when it's genuinely unambiguous:
  exactly one `(path, freq_key)` in the index has that ping within its
  own recorded range. Zero or multiple candidates both still return
  `None` rather than guess. Can only *add* matches a GPS-only search was
  dropping; never changes what an already-successful GPS match
  resolves to.
- `_build_position_index` now also caches each channel's `n_pings`
  (needed for the range check above).
- Tests: new `tests/unit/cli/test_export.py::TestCorrelateTarget` (6) --
  GPS match unchanged when it already works, no-GPS-falls-back-to-ping,
  GPS-too-far-falls-back-to-ping, ping-out-of-range still `None`,
  no-position-at-all still `None`, ambiguous-ping-across-multiple-files
  still `None` (not a guess).

Full suite: 1563 passing / 36 skipped (was 1553). Version bumped
0.4.26 -> 0.4.27.

##### Scale bars on track and gridded map plots (2026-09-25)

Same field session, same-day follow-up request: "track and any maps
need a scale bar as well." Neither `viz.track.plot_track_and_scanlines()`
nor `viz.gridded.plot_gridded_raster()` had one -- a real gap for output
meant to be read as a map, not just a diagnostic view.

- New `sidescanpy.viz.scalebar` module -- `add_scale_bar(ax, *,
  meters_per_data_unit=1.0, location="lower right", ...)`, a
  cartographic line-plus-ticks-plus-label scale bar drawn in
  axes-fraction coordinates (robust to whatever the current data range
  is), rounded to a "nice" 1-2-5x10^n length
  (`_nice_scale_length`/`_format_scale_label`, switches to km at
  1000m). `meters_per_data_unit=1.0` is correct as-is for both this
  library's map plots, since both already default to real projected
  (auto-UTM) meters, not raw lat/lon degrees -- no per-plot conversion
  math needed at the call site.
- Wired in as `show_scale_bar: bool = True` on
  `viz.track.plot_track_and_scanlines()`/`export_track_and_scanlines_png()`
  and `viz.gridded.plot_gridded_raster()`/`export_gridded_png()`,
  default on for every existing caller. Deliberately skipped (not drawn,
  not an error) whenever the plot's own CRS is geographic (WGS84
  degrees) rather than projected -- meters-per-degree isn't a single
  constant across a plot's extent, so a correct bar would need the
  plot's own latitude, which neither function has a principled way to
  supply automatically.
- Tests: new `tests/unit/viz/test_scalebar.py` (19), plus
  shown-by-default/disable/skipped-for-geographic-CRS cases added to
  both `tests/unit/viz/test_track.py` and `tests/unit/viz/test_gridded.py`.
  Full suite: 1588 passing / 36 skipped (was 1563).
- Version bumped 0.4.27 -> 0.4.28.

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
- **`sidescan-correct` had no way to apply `processing.range_correction`'s
  bundled empirical range correction at all** — `run_correction_steps`/
  `apply_correction_steps` have supported `"range_correction"` as a
  valid step since v0.9.0, but the CLI script that actually wraps them
  for every consumer project never exposed a flag for it, silently
  capping every real correction run at physics-based TVG alone. Found
  investigating "gridded segments look uncorrected — bright center, dark
  edges" on `project-a`'s `survey-a` survey: a real,
  altitude-invariant residual TVG (spreading+absorption alone) doesn't
  capture (documented in `processing/range_correction.py`'s own module
  docstring as the reason this module exists) — invisible in waterfall
  renders (percentile scaling dominated by floor-clipped outliers hides
  it) but plainly visible in gridded planview output (mean-aggregation
  reveals it as a vignette). This survey's own two frequencies
  (`hz_1495000`/`hz_547500`) are exactly what the bundled default
  calibration table was fit from originally. Fixed: new `--range-correction`
  flag, same idempotent/`--config`-seedable pattern as every other step
  (`_range_correction_already_done()` checks `processing_history` for
  `"empirical_range_correction"`, matching `apply_empirical_range_correction`'s
  own double-apply guard). 4 new tests (combined-pass, rerun-skip,
  added-in-a-later-standalone-invocation, matching `TestCombinedSteps`/
  `TestIdempotency`'s existing conventions). Full suite: 1497 passing /
  36 skipped (was 1494).
- **Bundled default range-correction calibration was fit against a
  since-superseded TVG recipe, and was actively making the residual
  worse, not better, once layered on top of the current one (2026-09-18).**
  Follow-up to the `--range-correction` flag fix directly above: exposing
  the flag surfaced that applying it on real `survey-a` data
  still left gridded output looking bright-center/dark-edge, not flat.
  Root cause, confirmed quantitatively: the bundled calibration
  (`range_correction_default.json`) was fit 2026-07-17 against the
  `absorption_only` TVG era later superseded by `project-a`'s own
  2026-09-16 `full_physics`-recipe decision (`datasets/survey-a.json`) —
  a different, larger real residual than the one the calibration was
  built to remove. Direct measurement of production `backscatter_intensity`
  (post range-correction) showed a ~33dB near-to-far decline (HF) /
  ~48dB (LF) over 0-30m ground range — the stale calibration wasn't just
  ineffective, applying it measurably worsened the held-out residual
  compared to plain TVG+geometric alone. **Fixed by refitting**, not by
  changing any library code: reran `fit_range_correction()` against
  `reconstruct_raw()` + `apply_correction_steps({"tvg", "geometric"})` on
  the current production recipe (plain-default full-physics TVG) using
  this survey's own already-corrected `ss_h5` file, then validated on a
  held-out ping range — held-out peak-to-trough residual dropped from
  13.10dB to 1.18dB (HF, 91% reduction) and 5.49dB to 1.02dB (LF, 81%
  reduction). Same known limitation as before, unchanged: fit and
  validation ranges are both from this one survey file, not genuine
  cross-survey validation; this file's own recorded/cropped range only
  reaches ~30m, so the refit table's 6 bins span 2.5-27.5m and any
  caller's `ground_range_m` beyond that clamps to the last bin's value
  (`RangeCorrectionTable`'s existing, deliberate constant-extrapolation
  behavior) rather than extrapolating an unverified shape further out.
  No test changes needed — no test asserts the bundled table's actual
  values, only that `load_default()` returns a usable table.
- **Gridded raster's background land/water context (`show_land=True`)
  compounded two independent margins into one visibly oversized "way
  more padding than necessary around the swath" frame (2026-09-18).**
  `bbox_from_channel_dataset()` — the grid-construction step run before
  a `SideScanGriddedRaster` ever exists — already pads the channel's
  true data extent by its own default 10% margin, so `raster.bbox`
  arriving at `plot_gridded_raster()` is never a tight fit to begin
  with. `add_land_background()`'s own default `margin_frac=0.08`
  (reasonable for a caller passing an unpadded, tight bbox, e.g.
  `viz/track.py`'s own track+scanline extent) was stacking a second,
  independent margin on top of the first for the gridded-raster case
  specifically, nearly doubling the effective padding around the real
  swath. Fixed at the `plot_gridded_raster()` call site only (not
  `add_land_background()`'s own default, which is still correct for a
  tight-bbox caller): passes `margin_frac=0.0` explicitly, since
  `raster.bbox`'s own pre-existing 10% margin already serves
  `add_land_background()`'s "don't clip land right at the edge" purpose.
- **Version bumped 0.4.17 → 0.4.18** for this round (covers this entry,
  the `--range-correction` CLI flag fix directly above, and the
  vertical-flip/edge-gap/jitter/multi-target geometric-chip fixes from
  the prior round — none had bumped the version yet).

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