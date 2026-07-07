# SideScanPy — Claude Code Context

**Project:** Python library for processing sidescan sonar data
**Owner:** Russ Shomberg <rshomberg@uri.edu>
**Version:** 0.5.0 tagged; v0.6.0 gridding module and v0.7.0 conversion/visualization CLI complete, unreleased
**Python:** 3.14+

The package can:
- Ingest Edgetech JSF (`.jsf`) and Klein SDF (`.sdf`) sidescan surveys
- Convert them to a standardized `xarray`-based `SideScanSurvey` model
- Run them through a processing pipeline: interpolate aux streams → TVG correction → geometric ranges → georeferencing
- Round-trip serialize to `.ss.h5` (HDF5) bit-exactly
- Resample georeferenced channels onto a regular grid (`sidescanpy.grids`, v0.6.0)
- Produce waterfall and gridded-raster visualizations, plus PNG exports

**Note (2026-07-07):** This library is consumed by several separate,
private downstream projects (each their own git repo, living as siblings
under `<workspace>/` rather than nested in this one).
None of their code, data, or process details belong in this repo — that
includes not naming or describing them here beyond what's needed for
this library's own context; some may have confidentiality requirements
significant enough that their existence itself shouldn't be disclosed. As
part of the same reorganization, this repo's own former `scripts/`
directory (CLI tools built on top of the library) moved out to one of
those consumer projects — see "Quick Reference" below.

---

## Quick Reference

- **Source code:** `src/sidescanpy/`
- **Tests:** `tests/unit/` (fast), `tests/integration/` (slow, requires fixtures)
- **Docs:** `docs/` (module refs), `docs/planning/` (feature plans)
- **Scripts:** moved out of this repo 2026-07-07 (see note above) — CLI
  tools and demos now live in the `project-a` consumer project
- **Changelog:** `CHANGELOG.md` (Keep a Changelog format)
- **Development guide:** `docs/DEVELOPMENT.md`

---

## Architecture

```
sidescanpy/
├── core/
│   ├── schema.py       # xarray accessors, HDF5 constants, validation
│   └── survey.py       # SideScanSurvey, SideScanFrequencyChannel
├── io/
│   ├── dispatch.py     # detect_format(), read_and_convert() — NEW in v0.7.0
│   ├── jsf/            # Edgetech JSF format
│   ├── sdf/            # Klein SDF format
│   ├── netcdf/         # Single-channel NetCDF
│   └── hdf5/           # Full-survey HDF5 (.ss.h5)
├── processing/
│   ├── interpolate.py  # Aux stream → ping timestamps
│   ├── radiometric.py  # TVG correction
│   ├── geometric.py    # Slant/ground range, per-sample depth
│   ├── georeferencing.py  # Per-sample lat/lon
│   ├── override_altitude.py  # Manual altitude injection (v0.5.0)
│   ├── slicing.py      # Subset operations
│   └── pipeline.py     # run_pipeline() (ladder), run_correction_steps()/
│                        #   apply_correction_steps() (explicit step-set) — NEW in v0.7.0
├── grids/              # v0.6.0
│   ├── definition.py   # GridDefinition (bbox + resolution + CRS)
│   ├── raster.py       # SideScanGriddedRaster (xarray wrapper)
│   └── resample.py     # resample_channel_to_grid()
└── viz/
    ├── waterfall.py    # Port+starboard waterfall display; along_track_distance_m()
    ├── gridded.py      # gridded raster plots (v0.6.0)
    └── track.py        # NEW v0.7.0: full-survey track + scan-line plot (UTM)

```

`scripts/` (CLI tools built on the library: `convert_survey.py`,
`visualization_demo.py`, `convert.py`, `batch_convert_and_plot.py`,
`plot_sonar_ping.py`, `export_waterfall_series.py`, the superseded
`scripts/old/` ones, gridding demos, diagnostic scripts) moved to the
`project-a` consumer project's own `scripts/` directory as of
2026-07-07 — see that project's docs, not this file, for their current
state. Every mention of `scripts/...` in the "Current Work Focus"/
changelog sections below predates that move and describes work done while
they still lived here; left as historical record, not updated.

### Data flow

```
.jsf or .sdf binary
    │  read_jsf() / read_sdf()
    ▼
list[Message]  (typed message instances)
    │  convert() → format-dispatched
    ▼
SideScanSurvey (frequency channels + aux streams)
    │  interpolate_aux_onto_pings()
    ▼
SideScanSurvey  (per-ping fields refined)
    │  apply_tvg()
    ▼
processing_level = "radiometric"
    │  compute_geometric_ranges()
    ▼
+ slant_range_m, ground_range_m, across_track_m, etc.
    │  compute_georeferencing()
    ▼
+ latitude_per_sample, longitude_per_sample
    │  write_survey()
    ▼
.ss.h5 file
    │  resample_channel_to_grid()   (v0.6.0)
    ▼
SideScanGriddedRaster
```

End-to-end pipeline runs in ~60 seconds on a 5000-ping survey.

---

## Data Directory Convention (v0.7.0+)

Survey data lives under `data/<survey_name>/`, colocated — raw files,
converted files, and every generated export travel together in one
directory rather than being split between `data/` and an unrelated
`out/` tree:

```
data/<survey_name>/
├── raw/       # original .jsf / .sdf files (untouched, source of truth)
├── ss_h5/     # converted .ss.h5 — scripts/convert_survey.py output
└── exports/   # waterfalls/gridded/manifest.json — scripts/visualization_demo.py output
```

See `data/README.md` for the full naming table (frequency subfolders,
segment file naming, `manifest.json`) and
`docs/planning/v0.7.0-visualization-demo.md` for the CLI reference.
Nothing under `data/` except `data/README.md` is committed to git —
raw files exceed GitHub's size limit and generated exports are
regenerable, so neither belongs in version control (Known Issue #1).

This is the standard going forward for new survey work. Older survey
directories that predate it aren't retroactively reorganized. Scripts
that predate this convention (`convert.py`, `batch_convert_and_plot.py`,
`export_waterfall_series.py`, `plot_sonar_ping.py`) still take
explicit input/output paths and work fine pointed at the new layout —
they just don't default to it or document it themselves.

---

## Data Structure — CRITICAL

Sidescan data is **3D**, not 2D:

```
backscatter_raw:      (ping, channel, sample)   e.g., (5216, 2, 4792)
latitude_per_sample:  (ping, channel, sample)
longitude_per_sample: (ping, channel, sample)
```

**Channel dimension:**
- `channel=0` → port side
- `channel=1` → starboard side
- Currently always size 2 (validation enforces this)

**Common mistake:** Averaging the channel dimension. Don't do this — treat each channel's samples as separate observations that span perpendicular to the survey track.

---

## Conventions

### Code Style
- **xarray-first:** All datasets validated by xarray accessors
- **Lossless transforms:** `compute_*` adds variables; `apply_*` modifies existing ones
- **`processing_level`:** Only tracks backscatter state (`"raw"` or `"radiometric"`)
- **Sign conventions:**
  - Across-track: port negative, starboard positive
  - Along-track: forward of nadir positive
  - Heading: compass bearing CW from N
- **No silent transformations:** Log at DEBUG for masks; INFO for unknown data
- **Schema-first:** Add to `core/schema.py` before implementing pipeline output

### Testing
- **TDD:** Write tests first, then implementation
- **Fixtures:** Use `_make_minimal_*` helpers, avoid real data in unit tests
- **Test naming:** `test_<what>_<when>()` (e.g., `test_rejects_2d_data`)
- **Test classes:** Group by feature/method (e.g., `TestGriddedRasterValidation`)

### Git & Commits
- **Conventional Commits:** `feat(scope):`, `fix(scope):`, `test(scope):`, etc.
- **Scopes:** Use hyphenated (`feat(io/sdf):`, `fix(grids/resample):`)
- **Commit each logical change** — don't bundle unrelated changes
- **Suggest commits before running them** — user reviews and commits manually

### Dependencies
- **Restraint:** Only add deps with justification
- **Current stack:** numpy, scipy, xarray, h5netcdf, pyproj, matplotlib, pandas
- **New deps:** Discuss before adding

---

## Working Style

### Before Making Changes
1. **Read relevant files first** — use `Read` tool
2. **Ask before major changes** — user prefers to approve
3. **Plan first, code second** — write a plan for complex tasks
4. **Test locally** — run pytest before suggesting commits

### After Making Changes
1. **Run tests:** `pytest tests/unit/ -v` (fast) or specific test file
2. **Suggest a commit message** — Conventional Commits format
3. **Wait for user to commit** — don't auto-commit

### Common Commands

**Environment:** This project uses `uv` for dependency management. Run
`uv sync --extra dev` once after cloning or pulling to install/update the
environment. No manual venv activation needed — `uv run` handles it.

**ROS2 note:** ROS2 Jazzy is installed system-wide and its `setup.bash`
injects `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` into every
shell. The Makefile clears this automatically. Always use `make test` or
prefix commands with `PYTHONPATH=""`. Long-term fix: conda env for ROS work,
keep this project in uv. See Known Issues #14.

**Other workstation:** The same uv migration was applied here on 2026-07-07.
If the other workstation is still on the old pip/venv setup, run:
```bash
# One-time migration on other workstation
rm -rf .venv
uv sync --extra dev
```
The same ROS PYTHONPATH issue likely applies there too — verify with
`echo $PYTHONPATH` and use `make test` rather than invoking pytest directly.

```bash
# Install / sync environment
uv sync --extra dev

# Run unit tests
make test       # quiet
make test-v     # verbose
make test-cov   # with coverage

# Run a specific test file
PYTHONPATH="" uv run pytest tests/unit/grids/test_resample.py -v

# Run with coverage manually
PYTHONPATH="" uv run pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing
```

---

## Lessons Learned

### Passing Tests ≠ Correct Code
A refactor of `resample_channel_to_grid()` produced tests that passed while using
2D synthetic data. The real data was 3D. Tests never exercised the actual data
structure the library operated on in production. **Test fixtures must match real
data shapes.** A test that passes with unrealistic data is false confidence.

### Diagnostic Views vs. Data Products
- **Data product** (`plot_gridded_raster()`, GeoTIFF): clean output for GIS/reports. Must be trustworthy.
- **Diagnostic view** (`plot_gridded_with_coverage()`): reveals internal data structure. Expected to show "artifacts" because that's the point.

Don't judge output quality by diagnostic views. Judge data products by whether
they look right in their target use case (e.g., open the GeoTIFF in QGIS).

### Path A/B/C Refactor Framework
When a component has known issues and limited time:
- **Path A:** Accept and document current state
- **Path B:** Full refactor as originally planned
- **Path C:** Minimal hardening — lock in current behavior with tests and validation, defer scope expansion

Path C was chosen for v0.6.0 gridding and worked well. Consider this when
"good enough" beats "complete."

### Documentation Drift Is Real
`CLAUDE.md` was updated to claim gridding was "complete and verified" when it
wasn't. Diagnostic work later found: no 3D validation, no channel-handling tests,
demo scripts pre-flattening data before calling the library, dead code misleading
about capabilities. **Verify documentation against actual code state periodically.
Trust but verify AI-generated summaries of "what's done."**

### Dev Environment Isolation
ROS2's `setup.bash` (sourced unconditionally in `~/.bashrc`) injects
`PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages`. This leaks into pytest
and caused silent test failures when `launch_testing` tried to import `lark`. The
venv's `include-system-site-packages = false` does **not** block `PYTHONPATH`
injection — that's a shell env var, not a site-packages flag. Always prefix
pytest with `PYTHONPATH=""` or use `make test`. The correct long-term fix is a
conda environment that fully isolates this project from the system ROS install.

---

## Current Work Focus

### v0.6.0 — Gridding Module (Complete, hardened, and verified usable — closed out)

**Status:** All three phases complete, hardened (2026-07-01), and verified to produce GIS-usable output on real data. Gridding work is closed out; focus has moved to v0.7.0 (below). `pyproject.toml` version bump still pending — not yet released.

**Completed:**
- ✅ `GridDefinition` (bbox + resolution + CRS)
- ✅ `SideScanGriddedRaster` (xarray wrapper) — carries a `units` param (`"linear"`/`"dB"`), propagated through `reproject_to()`/`slice_by_bbox()`/`slice_by_indices()`/dict serialization
- ✅ `resample_channel_to_grid()` — verified (not just assumed) to handle 3D `(ping, channel, sample)` data correctly: treats each channel's samples as separate observations, does not average port/starboard. Validates 3D shape and lat/lon/backscatter shape agreement, raising `ValueError` on 2D or mismatched input. `backscatter_var` param (default `"backscatter_raw"`) selects which variable to grid. Final API differs from the original plan in `docs/planning/v0.6.0-gridding-refactor.md` (that doc describes the initial `histogram`/`linear`/`nearest`/`cubic` design; the shipped implementation uses a `ResamplingMethod` enum — `mean`, `median`, `max`, `nearest`, `idw` — built on `scipy.spatial.cKDTree`). `separate_sides` param exists in the signature but is **not implemented** (dead code, deferred). The planning doc has a "What Actually Shipped" / "Not Implemented" section with the full delta.
- ✅ `viz/gridded.py` — gridded raster plots; colorbar labels read the raster's actual `units` instead of a hardcoded `"linear"` string. `plot_gridded_with_coverage()` is a **diagnostic/QC view, not a data product** — its alpha channel exaggerates normal per-cell sample-density variation into visible "streaks" fanning out from the survey track (most visible at fine resolution / low ping counts, but present at any scale — confirmed still present, unchanged, on the full 5216-ping survey). The opaque view (`plot_gridded_raster()` / `export_gridded_png()`) does not exhibit this and is the correct thing to treat as the actual output. Documented directly in the function's docstring.
- ✅ Integration test: `tests/integration/test_gridding_pipeline.py`
- ✅ Demo scripts: `scripts/demo_gridding_real_data.py` (now has `--method` flag, selects `backscatter_intensity` automatically on radiometric-processed surveys), `scripts/demo_gridding_workflow.py`, `scripts/demo_min_mapping.py`, `scripts/demo_min_visualization.py`
- ✅ `tests/unit/grids/conftest.py` — shared 3D channel fixtures matching production proportions

**2026-07-01 hardening pass — bugs found and fixed:**
1. `resample_channel_to_grid()` had zero test coverage of 3D behavior and no shape validation; it happened to handle 3D correctly by accident (generic `.ravel()` over whatever shape arrives) rather than by design. Added validation + regression tests.
2. Every gridding demo run against radiometrically-corrected surveys was silently gridding the **uncorrected** `backscatter_raw` variable, not the TVG-corrected `backscatter_intensity` written by `apply_tvg()`. Root cause: `resample_channel_to_grid()` hardcoded the variable name, and the demo's variable-selection fallback always matched `backscatter_raw` first. Fixed via new `backscatter_var` param.
3. `SideScanGriddedRaster` and the colorbar both hardcoded `units="linear"` regardless of what was gridded, mislabeling dB-scale corrected data. Fixed via a `units` param threaded through the raster and viz layer.

**2026-07-01 quality verification — "fan of streaks" investigated and resolved (as a non-issue):**
Diagnosed an apparent "fan of thin radiating streaks" artifact seen in early gridding demos. Root cause: `plot_gridded_with_coverage()`'s alpha-transparency rendering exaggerates ordinary per-cell sample-density variation (KDTree search radius vs. real ping/sample spacing) into visible rays — present throughout the full survey, not just during vessel turns as first suspected. Confirmed via direct comparison that the pattern is **identical before and after the v0.6.0 hardening pass** (same coverage geometry, only backscatter units/values differ) — hardening never touched this and the pattern isn't a regression. The primary data product (opaque raster / GeoTIFF-bound output) does not exhibit it at all. Verified full-survey output at 1m resolution is clean, coherent, and usable for GIS work.

### v0.7.0 — Format Conversion CLI + Visualization Demo (Complete)

**Status:** Implemented 2026-07-01. Plan: `docs/planning/v0.7.0-visualization-demo.md` (includes the user's resolutions to all open questions inline). Distinct from the also-`v0.7.0`-labeled GeoTIFF+cropping plan below (`docs/planning/v0.7.0-geotiff-and-cropping.md`) — two separate planning docs share the version label; not yet reconciled into a single roadmap entry.

**Completed:**
- ✅ `scripts/convert_survey.py` — standalone batch `.jsf`/`.sdf` → `.ss.h5` converter (`--input-dir`/`--output-dir`/`--input-format`/`--output-format`/`--process`/`--force`)
- ✅ `scripts/visualization_demo.py` — given a directory of `.ss.h5` files, lazily concatenates each frequency across files (dask-backed via `read_survey(chunks=...)`, aux-interpolated per file *before* concatenation), segments by pings/time/distance, and renders the full `{waterfall,gridded} x {raw,radiometric,geometric,radiometric+geometric,full}` matrix, writing `manifest.json` and pointing users at `plot_sonar_ping.py` for interactive follow-up. Verified end-to-end against a synthetic multi-file fixture, including segments that span a file boundary.
- ✅ **Standard data directory layout** (`data/<survey_name>/{raw,ss_h5,exports}`) — see the new "Data Directory Convention" section above and `data/README.md`. Supersedes the old convention of writing script output to `./out/`.
- ✅ `sidescanpy/io/dispatch.py` (`detect_format`, `read_and_convert`) and `sidescanpy/processing/pipeline.py` (`run_pipeline`, `run_correction_steps`, `apply_correction_steps`) — promoted out of the now-deleted `scripts/_workflow_helpers.py` into the library proper, per the project convention that domain logic belongs in `src/sidescanpy/`, not `scripts/`. `run_correction_steps`/`apply_correction_steps` add a capability the old ladder-only `_workflow_helpers.process_pipeline()` didn't have: correction combinations outside the ladder, e.g. geometric ranges **without** TVG — confirmed as an intentionally-supported combination (verified pipeline-independence table in `docs/planning/workflow-decisions.md`), not a ladder artifact.
- ✅ `sidescanpy/viz/along_track_distance_m()` — public wrapper around the along-track-distance primitive previously private to `viz/waterfall.py`; needed by `visualization_demo.py`'s distance-based segmentation.
- ✅ `scripts/batch_convert_and_plot.py` updated to import from `sidescanpy.io`/`sidescanpy.processing` instead of the deleted `_workflow_helpers.py`; its waterfall-export loop is now a small local copy rather than a shared import (deliberate — some duplication between scripts is fine; shared *domain* logic belongs in the library, shared *script* loops don't need to).
- ✅ `scripts/_workflow_helpers.py` deleted. `scripts/01_convert_jsf_to_standard.py`, `01_jsf_to_hdf5.py`, `01_jsf_to_netcdf.py`, `02_refine_hdf5.py`, `03_export_netcdf.py`, `03_generate_plots.py`, `convert_jsf.py`, `convert_jsf_to_hdf5.py` moved to `scripts/old/` (superseded by `convert.py`/`convert_survey.py`; not deleted, not maintained).
- ✅ Unit tests: `tests/unit/io/test_dispatch.py`, `tests/unit/processing/test_pipeline.py` — cover both the ladder and the explicit step-set primitive, including the geometric-without-TVG case.
- ✅ Run against a real multi-file JSF survey (`data/survey-a/`, 5 files) — the local `raw/ss_h5/exports` layout there confirms both scripts work end-to-end on real fieldwork data, not just the synthetic fixture. This also surfaced a real, unrelated pre-existing bug: two test fixtures (`tests/unit/io/jsf/conftest.py`'s `real_jsf_file_path`, `tests/data/conftest.py`'s `generate_sample_jsf_messages`) hardcoded a path to this same local survey's old flat `data/survey-a/jsf/...` location; once the directory was reorganized into the new layout the file moved to `raw/...` and those fixtures broke. Fixed by updating both hardcoded paths — worth knowing about since these fixtures point at machine-local, not-in-repo data (Known Issue #1) and will silently break again for anyone else with a differently-named local survey directory.

**2026-07-02 follow-up fixes (usability, post-real-data-check):**
1. `--segment-size` default raised 200 → 2000 (10x, a rough heuristic per user direction, not a measured value) — 200 pings produced a tall, thin waterfall strip rather than something close to square once the x-axis is real distance instead of sample index.
2. **Ping y-axis was always 0-based per-segment, not the survey-global index.** `plot_waterfall()` computed `y_axis="ping"` values as `arange(ds.sizes["ping"])` — correct for a caller passing the *whole* dataset with `start`/`stop` slice bounds, but `visualization_demo.py` pre-slices+loads each segment before correction/plotting, so the resulting per-segment dataset's own ping axis always restarts at 0. Fixed by adding a `ping_offset: int` parameter to `plot_waterfall()`/`_compute_y_full()` (default 0, backward compatible) — `visualization_demo.py` now passes the segment's true global start ping. Title also fixed to show the corrected global range. Covered by `tests/unit/viz/test_waterfall.py` (new).
3. **Geometric-corrected waterfalls didn't actually show the correction.** All correction levels were plotted with the same CLI `--x-axis`/`--y-axis` (default `sample`/`ping`) — since only backscatter *values* change under TVG (not pixel geometry), a `geometric`/`radiometric+geometric`/`full` waterfall looked identical in layout to `raw`, defeating the point of generating it. Fixed: any correction whose step set includes `"geometric"` now always renders `x_axis="across_track"`, `y_axis="along_track"` (both meters) via `scripts/visualization_demo.py`'s new `resolve_waterfall_axes()`, regardless of the CLI flags — those flags now only govern `raw`/`radiometric` levels, where sample/ping/time are the only meaningful choices anyway.

**2026-07-02 gridding correctness pass (northing/easting request + "why does gridded look less normalized" investigation):**

The user asked for gridded output in proper projected coordinates (northing/easting + CRS metadata) and asked why gridded output looked less normalized than the waterfall despite radiometric correction. Both questions led to real, previously-undiscovered bugs in code CLAUDE.md had documented as "hardened" (v0.6.0):

1. **`resample_channel_to_grid()` only ever worked for WGS84 — projected-CRS grids were silently unimplemented.** The KDTree/search-radius math was hardcoded to lat/lon-degree math regardless of `grid_def.crs`; passing a projected `GridDefinition` (e.g. UTM) would silently produce an empty or garbage raster, since sample points (always WGS84) were never reprojected into the grid's CRS before the nearest-neighbor query. This was already flagged, honestly, by two `pytest.skip("Channel reprojection not yet implemented")` tests in `tests/unit/grids/test_resample.py` — not a hidden regression, an acknowledged gap. **Fixed:** sample points are now reprojected into `grid_def.crs` before building the KDTree; search radius is computed in the grid's native units (degrees for geographic, meters for projected). Both skipped tests implemented and passing, plus a new cross-check test verifying a UTM grid and its WGS84-equivalent grid produce comparable coverage over the same channel.
2. **`GridDefinition.extent_m_ns`/`extent_m_ew` used Web Mercator (EPSG:3857) Euclidean distance, not true geodesic distance.** Web Mercator's vertical scale is `1/cos(latitude)`, so this overstated true ground distance away from the equator — **~39% too large at 44°N**, which is where this project's actual survey data (`data/survey-a`, site-b) sits. Since `num_cells_ns`/`num_cells_ew = ceil(extent_m / resolution_m)`, every WGS84 grid at this latitude has silently been using ~39% more (and smaller) cells than `resolution_m` requested — directly contributing to noisier-looking gridded output, since more, smaller cells means fewer raw samples land in each one. **Fixed:** `_geodetic_distance_m()` now uses `pyproj.Geod(ellps="WGS84").inv(...)` (the same technique already used correctly elsewhere in this codebase, e.g. `viz/waterfall.py`'s `_along_track_distance`), not a Web Mercator transform. Two existing tests (`test_cell_count_ns`/`test_cell_count_ew`) had the old, distorted numbers baked into their expected ranges (with a comment acknowledging "Web Mercator distorts" rather than flagging it as wrong) — updated to the corrected values. A third test (`test_near_pole`) was exploiting the same bug with a degenerate `-180..180` longitude bbox (two coincident points at the antimeridian); narrowed to a realistic few-degree span.
3. **`plot_gridded_raster()` auto-scaled on the 2nd/98th percentile; `plot_waterfall()` uses the 1st/99th.** On real survey data this is a much bigger deal than the 1-percentile difference suggests: the backscatter distribution has a sharp density cliff right at that boundary (near-zero/floor-clipped no-return samples), so p1=-125.8 dB but p2=-52.0 dB — a **~74 dB swing** from that one percentile point alone. This alone was likely the dominant driver of the "gridded doesn't look normalized" observation. **Fixed:** `vmin_pct`/`vmax_pct` parameterized on `plot_gridded_raster`/`plot_gridded_with_coverage`/`plot_gridded_multiple_methods`/`export_gridded_png`, defaulting to 1.0/99.0 to match `plot_waterfall`. `visualization_demo.py` relies on this shared default rather than passing its own value, so waterfall and gridded exports of the same segment stay comparably normalized without needing to keep two settings in sync by hand.
4. **Remaining visual difference after all three fixes above has two causes, one benign, one real** — revisited 2026-07-02 after the user pushed back that gridded output still didn't resemble the waterfall. (a) Heavy cell averaging: at 1m resolution this survey's point cloud is extremely dense (median ~7100 raw samples averaged into each 1m² cell, traced to long near-stationary stretches — median ping-to-ping spacing ≈0m over the first 2000 pings tested), which is real but benign — a resolution/aggregation-method tuning question (`--grid-method median` helps, coarser/finer resolution trades detail for coverage), not a bug. (b) **Not benign:** a genuine ~37 dB range-dependent TVG residual, quantitatively confirmed and written up under Known Issue #4 below — the waterfall's percentile-based color scale is stretched wide by Known Issue #5's floor-clipped outliers, which visually hides this residual, while gridding's mean aggregation suppresses those same outliers and reveals it as a vignette-like darkening toward the swath edges. This is a real data/algorithm issue in `processing/radiometric.py`, not something fixable in the gridding or visualization code — see Known Issue #4 for the full finding.

**Northing/easting output (the original ask):** `sidescanpy.grids.utm_crs_for_lonlat()` / `utm_grid_from_wgs84_bbox()` (new, `grids/definition.py`) auto-select a WGS84 UTM zone from a bbox's center and reproject into it. `SideScanGriddedRaster` now names its position coordinates `northing`/`easting` (meters) instead of `latitude`/`longitude` (degrees) whenever its CRS is projected — a real coordinate rename, not just a relabeled unit, since a variable literally named "latitude" holding meter values would be actively misleading (`y_coord_name`/`x_coord_name`/`y_values`/`x_values` properties added so callers don't need CRS-branching logic of their own). `viz/gridded.py`'s plot functions show "Easting (m)"/"Northing (m)" axis labels and the CRS (e.g. "WGS 84 / UTM zone 19N (EPSG:32619)") in the title for a projected raster. `visualization_demo.py`'s `export_gridded()` now grids into an auto-selected UTM zone by default rather than raw WGS84 degrees.

**Track + scan-lines plot (2026-07-02, new):** `sidescanpy.viz.plot_track_and_scanlines()`/`export_track_and_scanlines_png()` (new module `viz/track.py`) — a full-survey overview showing the vessel's nadir track plus periodic port/starboard scan-line "rungs", in a projected (auto-UTM by default) coordinate system, no backscatter. Wired into `visualization_demo.py` as a new `track` output type (`build_survey_track()`) — one plot per frequency covering the *whole* multi-file survey, not per-segment/correction like `waterfall`/`gridded`, written to `track_full/hz_<freq>/track.png`; not in `--output-types`' default set (opt-in via `--output-types track`). New `--track-scanline-stride` (default 50) controls rung density.
  - **Real bug found and fixed while building this:** the first version tried to save time by slicing each ping down to just its farthest-range sample *before* calling `compute_geometric_ranges`/`compute_georeferencing`, on the assumption that per-sample geometry has no cross-sample dependency. That assumption is wrong in one specific way: `compute_geometric_ranges` computes each sample's position from `np.arange(ds.sizes["sample"])` — a *local* index into whatever sample axis it's handed, not the sample's true position within the original ping. Slicing to 1 sample first makes the function see "sample 0" (point-blank range) instead of the true far-range edge, which the water-column mask then silently nulls to NaN — every scan line came out empty. **Fixed** by pre-selecting the strided *pings* instead (safe — ping-axis slicing doesn't touch `ds.sizes["sample"]`) while keeping each selected ping's full sample axis intact through georeferencing, slicing to the last sample only afterward. Verified on real data: swath width came out as a physically sensible ~130m (was 100% NaN before the fix). This is a real footgun in `compute_geometric_ranges`'s API (sample position is implicit in axis length, not an explicit coordinate) worth remembering before ever pre-slicing the sample axis ahead of a geometric/georef call elsewhere in this codebase.

**Not done / explicitly deferred:**
- `manifest.json`'s schema was decided during implementation (not pre-specified in the plan) — see the file itself or `visualization_demo.py`'s `main()` for the current shape (per-frequency segment list: index, ping range, time range, source files).
- No new CLI flag for the dask ping-chunk size — hardcoded to 512 (`visualization_demo.py`'s `_PING_CHUNK_SIZE`), per user direction; revisit only if profiling on a real multi-file survey suggests otherwise.
- The new 2000-ping segment-size default is a rough heuristic ("10x the old default"), not derived from real pings-per-meter vs. samples-per-meter ratios — revisit once there's a basis for a real estimate.
- `utm_crs_for_lonlat()` doesn't handle a segment whose bbox straddles a UTM zone boundary specially — it just picks the zone containing the bbox center, which is fine for typical segment sizes but could put a wide segment's edges in an adjacent zone's "wrong" region. Not addressed; flag if a survey's segments turn out to be UTM-zone-width or wider.
- The "why gridded looks smoothed relative to waterfall" finding (#4 above) suggests `resolution_m`/`--grid-method` may need per-survey tuning rather than a one-size-fits-all default; no auto-tuning implemented.

### Test Status (as of 2026-07-01)

```
tests/unit/                   950 passing, 36 skipped
tests/integration/             (run with --slow; requires local JSF/SDF fixtures — see Known Issues #1)
```

---

## Roadmap

### v0.7.0 — GeoTIFF export + CSV-driven target cropping

**Status:** Planning (2026-07-01). See `docs/planning/v0.7.0-geotiff-and-cropping.md` for the full plan — awaiting approval before implementation starts. (Not the same v0.7.0 work as the Format Conversion CLI + Visualization Demo above, which is already implemented — two planning docs currently share the `v0.7.0` label.)

**Priority 1 — GeoTIFF export**
- `write_geotiff()` in `sidescanpy/grids/`, built on `rasterio` (new dependency — not yet in `pyproject.toml`)
- CRS/resolution/bounds metadata; verify by opening in QGIS

**Priority 2 — Target cropping from CSV**
- New module: `sidescanpy/targets/`
- CSV input (`target_id, ping, channel, sample, window_pings, window_samples, label`) → cropped PNGs + optional metadata sidecar
- Attitude-corrected via existing `processing/geometric.py` pitch correction + `processing/slicing.py` (`slice_by_pings`/`slice_by_samples`)
- CLI: `scripts/crop_targets.py`

**Priority 3 (deferred)** — YOLO bounding-box overlays; detection-based (threshold/anomaly) auto target export, superseding the old 7b sketch below.

### v0.8.0+ candidates (not yet ordered)

- **TVG diagnostics** — `docs/planning/v0.8.0-tvg-diagnostics.md` is a written, scoped plan (2026-07-03) for diagnosing the range-dependent TVG residual and floor-clipped-sample issue (Known Issues #4/#5) before changing `processing/radiometric.py` — awaiting approval, not yet just an idea.
- **Klein INI parsing** — unblocks proper `depth_m` for SDF (~1 session; see Known Issues #5)
- **Bottom detection** (`processing/detect_bottom.py`) — fallback altitude
- **Per-side diagnostic mode** — port/starboard bands in gridded raster
- **Slider performance fix** in `plot_sonar_ping.py`
- **XTF support** — industry-standard format
- **Radiometric pipeline cleanup** — address TODOs before quantitative work
- **Test fixture solution** — synthetic or small real fixtures in repo
- **Lazy/dask processing verification** — confirm pipeline works chunked
- **Detection-based target export** (old v0.7.0 "7b" sketch: threshold/local-max/anomaly detection on grids, auto-export candidates) — folded into v0.7.0 Priority 3, deferred

### Deferred (deliberately, not forgotten)

- **Streaming HDF5 write.** Memory not yet a problem. Largest tested workflow: 14 SDF files × 382 MB; JSF surveys up to 629 MB per file. No OOM yet. Defer until a real workload hits limits.
- **Multi-frequency depth/altitude reconciliation.** When two frequencies have different per-ping altitude estimates, we don't merge them. v0.7+.
- **Per-survey bias/offset corrections.** Heading bias, layback, etc. Needed for serious operational use; not yet first-class.

---

## Known Issues / Gotchas

### High Priority
1. **Test fixtures not in repo** — JSF/SDF files exceed GitHub's 100 MB limit. See `data/README.md`.
2. **Integration tests don't skip gracefully** when fixtures are missing.
3. **Bit-17 masking in SDF** — Undocumented, empirically correct.

### Medium Priority
4. **`processing/radiometric.py`** — Has correctness TODOs:
   - Assumes raw `backscatter_intensity` is linear amplitude
   - Uses single-mean sound velocity across all pings
   - dB conversion assumptions need review
   - Frequency-dependent absorption modeling is approximate
   - Works "well enough" for visualization but not for quantitative analysis. Revisit before any acoustic-quantitative work.
   - As of 2026-07-01, gridding demos correctly grid the TVG-corrected `backscatter_intensity` variable (see `resample_channel_to_grid()`'s `backscatter_var` param) instead of silently ignoring it — but the above correctness TODOs on the correction itself still apply.
   - **2026-07-02 — quantitatively confirmed a real range-dependent TVG residual**, found while investigating why `visualization_demo.py`'s `gridded_full` output still didn't visually resemble `waterfall_full` for the same segment even after the CRS/percentile fixes above. Binning already-TVG-corrected `backscatter_intensity` by `ground_range_m` (real data, `20251015170335.ss.h5`, first 2000 pings, floor-clipped samples from Issue #5 excluded) shows a strong, monotonic falloff — mean ≈ **+1 dB at 10–15m range down to ≈ -36 dB at 55–65m range**, a ~37 dB systematic gradient that TVG is specifically supposed to remove (spreading + absorption loss compensation). This is **not** a bug in `resample_channel_to_grid()`, `SideScanGriddedRaster`, or the percentile-scaling fix (all three verified correct/wired-in) — it's a real residual in the TVG correction itself, and the two views disagree cosmetically only because of how each one's color scale reacts to it: the waterfall's percentile-based `vmin` is dominated by the ~1% Issue #5 floor-clipped population (pinned to ≈-125 dB), stretching its color range wide enough to visually flatten/hide the ~37 dB real gradient sitting entirely within the upper end of that range; gridding's cell-*mean* aggregation **dilutes, not removes,** that same outlier population — the floor-clipped samples are *not* excluded from gridding (they generally have finite georeferenced coordinates and pass `resample_channel_to_grid()`'s finite-coordinate filter same as any other sample; see that function's docstring "Notes" section), they're just statistically swamped by the thousands of real samples sharing their cell at typical resolutions, narrowing the effective color range enough to make the gradient plainly visible as a vignette-like darkening toward the swath edges. Confirms this file's existing "not for quantitative analysis" caveat with a concrete number — revisit the TVG spreading/absorption model before trusting either view's absolute dB values, not just before "acoustic-quantitative work" in the abstract. Full writeup: `docs/planning/handoff-tvg-range-residual.md`.
5. **Sharp low-end "cliff" in `backscatter_intensity` (~1% of samples pinned at the TVG dB-conversion floor, ~-120 dB) — cause not yet confirmed, needs investigation.** Found 2026-07-02 while investigating why gridded output looked less normalized than the waterfall (see v0.7.0 "gridding correctness pass" above). Evidence gathered so far, on real data (`data/survey-a/20251015170335.ss.h5`, first 500 pings, `hz_1495000`):
   - The floor values come from **exactly-zero raw linear amplitude** in `backscatter_raw` (100% of sampled floor points, not just near-zero/noise) — `apply_tvg()`'s `20*log10(clip(x, 1e-6))` floors these to ≈-120 dB, which is a correct response to genuinely-zero input, not a bug in the TVG conversion itself.
   - These zero-amplitude samples form a **leading run at the start of every ping** — median run length 56 samples (out of 4792), tight across ~500 pings (p10=51, p90=60, std≈3.5) — **not** scattered through the range or concentrated at long range.
   - Run length has **~zero correlation with `altitude_m`** (r≈-0.04) in the sampled pings (though altitude itself barely varied in that window — p10=3.87m, p90=4.07m — so this isn't a strong test of geometry-dependence yet).
   - `ground_range_m` at these points is **not** collapsing toward zero (median ≈36m, close to the non-floor median ≈35m) — i.e. the existing water-column mask (`compute_geometric_ranges`, `slant_range < altitude` ⇒ NaN) does **not** currently treat these as water-column, because the geometric model has no concept of a fixed acquisition-time blanking period — it only compares slant range (sample_index × sample_interval) to altitude.
   - **Net read so far:** looks more consistent with a fixed transmit/receiver blanking or ring-down period (a hardware/acquisition-time artifact, roughly constant in *samples*, not tied to bottom geometry) than with either hypothesis raised when this was found — not the "water column, ground range → 0" pattern (ground range doesn't collapse here), and not a smooth incident-angle falloff either (values are hard exact zeros, not a gradual attenuation curve). **Not confirmed** — would need to check whether the run length is constant in *time* (varies with any sample-rate/range-scale changes across the survey) vs. constant in *samples* regardless of settings, and whether this pattern holds on the SDF/Klein side or is JSF/Edgetech-specific, before treating it as settled.
   - Not yet fixed or worked around anywhere — `resample_channel_to_grid()`/`plot_waterfall()` currently include these samples like any other data point.
   - See `docs/planning/handoff-tvg-range-residual.md` for the full writeup tying this together with the range-dependent TVG residual (Known Issue #4) into one investigation.
6. **Klein SDF `depth_m` is NaN** — Klein records depth as a sensor voltage; calibration constants are in companion INI files which we don't yet parse. Requires INI calibration parsing (deferred; ~1 session). **Workaround:** users with calibration override directly: `ch.dataset["depth_m"] = (["ping"], calibrated_array)`.
7. **`compute_geometric_ranges` falls back to 1500 m/s** — When `sound_vel_ms` is NaN (always the case for SDF surveys). Acceptable for typical work; would need refinement for cold-water or thermocline-crossing surveys.
8. **Klein 3500-series assumption** — SDF reader STRUCT matches Klein 3500 layout (no fields 93/94). 5000/5900 series may differ; would need per-version STRUCTs if encountered.
9. **`channel_id` not an indexed coordinate — `.sel(channel_id=...)` breaks in xarray 2025.10+.** Tests and some internal code use `.sel(channel_id="port"/"starboard")` but `channel_id` is a non-indexed coordinate. Newer xarray raises `KeyError: "no index found for coordinate 'channel_id'"`. Fix: use `.isel(channel=0)` / `.isel(channel=1)`, or call `.set_index(channel="channel_id")` before label selection. Currently causes 3 unit test failures.

### Low Priority
9. **SDF `--x-axis ground_range` rendering is "blurry."** Likely due to inaccurate altitude (depth not calibrated → altitude derived poorly). Will improve when bottom detection or INI parsing lands. Cosmetic for now since JSF data renders correctly.
10. **Slider performance** in `plot_sonar_ping.py` — Full waterfall re-renders each step. Lags on 5000-ping surveys.
11. **DVL & attitude validity flags** — Reject all data on tested devices; embedded values used as fallback.
12. **`fathometer_reading_m` not attitude-corrected** — Will need `* cos(roll) * cos(pitch)` once attitude correction is implemented.
13. **`messages.py` (JSF) has uneven naming** — Some `_deg` suffixes, some not. Mix of `Optional[float] = None` and `float = nan`. Cosmetic cleanup deferred.

---

## Session Handoff

At the end of a major work session, update the "Current Work Focus" section above with:
- What was accomplished
- New test counts
- Any new issues discovered
- Next steps

This keeps `CLAUDE.md` as the single source of truth for project state.