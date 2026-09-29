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
- **Scripts:** moved out of this repo 2026-07-07 (see note above); the
  standard pipeline CLI moved back in as `src/sidescanpy/cli/` in
  v0.15.0 (`sidescan-convert`/`sidescan-correct`/`sidescan-export`/
  `sidescan-analyze` console scripts) — genuinely project-specific
  one-offs/demos still live in consumer projects' own `scripts/debug/`
- **Changelog:** `CHANGELOG.md` (Keep a Changelog format)
- **Development guide:** `docs/DEVELOPMENT.md`

---

## Consumer-Project Architecture Contract (added 2026-09-18 — read this before adding project-local logic anywhere)

**This section exists because the family has drifted from it multiple
times without a written record to check against.** Every project
consuming this library — several source formats, several data sources,
one shared goal — is expected to converge on the same two-layer split:

1. **This library (`sidescanpy`) + its `sidescan-*` CLI carries every
   genuinely generic capability**: format ingestion, TVG/geometric/
   georef correction, investigation/diagnostics, visualization
   (waterfall/track/grid, with options), and dataset export (COCO/YOLO,
   chips, with options). If a capability would benefit more than one
   consumer project, it belongs here — not duplicated per-project.
2. **Each consumer project carries only a handful of thin scripts +
   configs** whose entire job is: resolve that dataset's own
   format/hardware quirks, get the data converted into this library's
   shared `.ss.h5` format as fast as possible, then hand off to the same
   `sidescan-*` CLI every other project uses. A project-local script
   that reimplements correction, investigation, or dataset-export logic
   already generic in this library is exactly the failure mode this
   section exists to prevent — not a stylistic preference.

**Analysis is diagnostic, not load-bearing at runtime.** `sidescan-analyze`
(or any investigation tooling) exists to identify a dataset's quirks —
hardware TVG state, nadir behavior, sample-rate assumptions — once, with
a human in the loop deciding what the finding actually means. The
*correction* that results gets hardcoded into that project's own
script/config (a fixed `TVGParams` recipe, a fixed nadir margin, etc.),
not re-derived by an automated classifier on every run. An automated
classifier that gets overruled by a human is expected and normal (see
`CHANGELOG.md`'s TVG-recipe history for several real examples); what
must not happen is a stale automated finding sitting uncorrected in a
config, or a human override happening in conversation without the
config being updated to match.

**Config-file format convention**: YAML for anything a human reads or
edits (per-dataset settings, correction recipes — the object of the
"analysis together, script recommends, human decides" workflow), JSON
reserved for machine-generated/machine-only artifacts a human would
essentially never hand-edit (`targets.json`, `metrics.json`,
`summary.json`, COCO annotations). **`sidescan-*`'s own `--config`
loader (`src/sidescanpy/cli/_config.py`) is currently JSON-only, by a
documented decision now superseded by this convention — adding YAML
support there (`pyyaml`, already a dependency elsewhere in this family)
is real, not-yet-done work, not just a documentation change.**

**Low-side develop, field run.** A consumer project's scripts/configs
are expected to work unmodified on a field/offline copy of the
same project, on the assumption that the field data has already been
brought into the same structure/format as the low-side reference —
because the actual correction/export logic lives in this library and its
CLI, not in anything environment-specific.

**Where a project currently stands relative to this contract is that
project's own concern, documented in its own `CLAUDE.md`/planning docs —
not summarized here, so this file doesn't need updating every time a
project's status changes.**

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
├── viz/
│   ├── waterfall.py    # Port+starboard waterfall display; along_track_distance_m()
│   ├── gridded.py      # gridded raster plots (v0.6.0)
│   └── track.py        # NEW v0.7.0: full-survey track + scan-line plot (UTM)
└── cli/                # NEW v0.15.0: console-script CLIs (see "v0.15.0" below)
    ├── convert.py      # sidescan-convert: survey | targets
    ├── correct.py      # sidescan-correct: --crop-farrange/--tvg/--geometric/--georef
    ├── export.py       # sidescan-export: waterfall | grid | track | chip | survey-window | progression
    └── analyze.py      # sidescan-analyze: investigate | tvg-fit

```

(This tree predates `targets/`, `export/`, and `analysis/`, added in
later versions — not kept exhaustively current; see each subpackage's
own module docstring for what's actually there.)

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
- **Prefer established libraries over custom implementations, except in
  extreme cases.** When a generic capability (resampling, gridding,
  interpolation, reprojection, etc.) has a well-established
  implementation in an already-used dependency (numpy/scipy/xarray/
  pyproj) or a well-justified new one, reach for that before writing
  custom logic — even if a hand-written version would be faster to
  produce initially. Custom code is acceptable where the established
  option doesn't fit the domain need (e.g. a library assumes isotropic
  binning and the real data is genuinely anisotropic) — but that's a
  documented exception, not the default.

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

### v0.8.0 — TVG Diagnostics, Stage 1 (Complete)

**Status:** Implemented 2026-07-16, per `docs/planning/v0.8.0-tvg-diagnostics.md`. Purely additive — `processing/radiometric.py` untouched (verified via `git diff`).

**Completed:**
- ✅ `sidescanpy/analysis/tvg_diagnostics.py` (new file in the pre-existing but previously-empty-of-TVG-content `analysis/` subpackage — see caveat below) — `before_after_backscatter()`, `mean_backscatter_vs_range()`, `mean_backscatter_vs_sample_index()`, `bin_pings_by_altitude()`, `_linear_to_db()`.
- ✅ `sidescanpy/viz/tvg_diagnostics.py` — `plot_mean_vs_range()`, `plot_tvg_gain_curve()`, `plot_per_ping_profiles()`, `plot_multi_frequency_comparison()`, `plot_altitude_bin_comparison()`.
- ✅ `tests/unit/analysis/test_tvg_diagnostics.py` (24 tests) + `tests/unit/viz/test_tvg_diagnostics.py` (9 smoke tests) — all synthetic fixtures, no real data. Full suite: 983 passing / 36 skipped (was 950/36 — grew by exactly the 33 new tests, zero regressions).
- ✅ `project-a/scripts/analyze_tvg.py` (separate repo) — CLI wrapping the above, with `--absorption-db-per-m`/`--spreading-loss-db` sweep overrides and CSV/JSON dumps alongside PNGs. Supersedes `debug_radiometric.py`/`debug_radiometric_visualization.py` (moved to `project-a/scripts/old/`).
- ✅ Validated against real data: converted `~/data/sidescan/project-a/survey-a/jsf/20251015170335.jsf` fresh (no `.ss.h5` existed for it anymore) and reproduced `handoff-tvg-range-residual.md`'s ~37 dB residual table numerically to within 0.05 dB — confirms the new tooling measures the same thing the original manual investigation found.

**⚠️ Found something bigger than diagnostics while validating — see `docs/planning/handoff-tvg-sound-velocity-zero-bug.md`:** the absorption-coefficient sweep (0.2–0.72 dB/m) barely moved the residual (<0.6 dB), which traced to `sound_vel_ms` being exactly `0.0` (not NaN) for **100% of pings** in the real survey tested. `apply_tvg()`'s sound-velocity fallback only checks `np.isnan(...)`, not `<= 0` (unlike `compute_geometric_ranges()`'s already-correct check — Known Issue #7) — so `apply_tvg()` has been computing TVG gain as if every sample were at the clipped `min_range_m = 0.5` floor, regardless of true range. This substantially — but, per the simulated-fix numbers in that doc, not fully — explains Known Issue #4's ~37 dB residual. **Not fixed yet**, deliberately: this is a real behavior change to `apply_tvg()`'s output on real data and was out of scope for Stage 1's diagnostics-only mandate; see the handoff doc for the suggested one-line fix and why the remaining (smaller, non-monotonic) residual after that fix now looks more like evidence for a missing angular/grazing-angle correction than for simple absorption mistuning.

**Pre-existing discovery, unrelated to this work, not touched:** `src/sidescanpy/analysis/` already existed (not created by this Stage 1 work) containing `comparison.py`, `geoplot.py`, `plotting.py` — none referenced anywhere in `src/` or `tests/` (confirmed via grep), and `geoplot.py` imports `sidescanpy.io.hdf5_manager.HDF5Loader`, a class that doesn't exist in the current `io/` architecture (`io/hdf5/` with `read_survey`/`write_survey` functions). Looks like orphaned code from before the 2026-07-07 family reorg, undocumented anywhere in this file. Left alone — out of scope for this session, but worth someone deciding whether to delete or resurrect it.

### v0.8.0 follow-up — sound-velocity fix applied, residual investigation deepened (2026-07-16)

**Status:** Fix applied and merged into this same work; deepened
investigation complete; no further code change made (see rationale
below). Full writeups: `docs/planning/handoff-tvg-residual-post-fix-findings.md`
(the investigation), `docs/planning/v0.9.0-empirical-range-correction.md`
(the resulting design proposal, awaiting approval, not implemented).

- ✅ **Applied** the sound-velocity fallback fix flagged (not applied) in
  the previous session: `apply_tvg()`'s check widened from
  `np.isnan(sound_velocity)` to `np.isnan(sound_velocity) or
  sound_velocity <= 0`, matching `compute_geometric_ranges()`'s pattern.
  Two new regression tests in `tests/unit/processing/test_radiometric.py`
  (`test_zero_sound_velocity_falls_back_to_default`,
  `test_nan_sound_velocity_still_falls_back_to_default`). Suite: 985
  passing / 36 skipped (was 983 — zero regressions).
- ✅ Re-validated against real data: the fix's real-world effect matches
  last session's *simulated* prediction almost exactly (e.g. 10–15m bin:
  predicted ≈30.5dB, actual 30.53dB).
- ✅ Deepened the investigation using existing Stage 1 tooling (no new
  code needed): multi-frequency comparison (`hz_1495000` vs `hz_547500`),
  altitude-bin comparison, and a post-fix absorption-coefficient sweep.
- **Result: neither original candidate cleanly confirmed.** The residual
  is altitude-invariant (evidence against candidate 2 as originally
  framed — a true grazing-angle effect should vary with geometry) but
  non-monotonic for `hz_1495000` and a completely different
  rise-then-plateau shape for `hz_547500` (evidence against candidate 1 —
  no single retuned absorption coefficient produces either shape, and
  the sweep confirmed no tested value flattens the curve; the *current*
  default of 0.20 dB/m for 1495kHz is already near-locally-optimal among
  values tried). Leading hypothesis, unconfirmed: a hardware TVG/AGC
  artifact baked into the raw signal before `sidescanpy` sees it, not a
  missing software correction term.
- **No `absorption_for_frequency()` coefficient change made** — the
  evidence pointed away from a simple retune (see above), and changing
  the model's functional form (the only thing that could actually help)
  is a real design decision belonging in
  `docs/planning/v0.9.0-empirical-range-correction.md`'s proposal, not a
  mechanical fix to make autonomously. That doc is explicitly
  awaiting-approval, not implemented, with several open questions for
  the user (is this worth pursuing now; is a hardware TVG curve
  documented anywhere; per-survey vs. fixed calibration).

### v0.9.0 — Empirical Range Correction (Complete)

**Status:** Implemented 2026-07-22, per
`docs/planning/v0.9.0-empirical-range-correction.md` (design proposal —
that doc's status line is stale and predates this entry; treat this
entry as the source of truth for what actually shipped). Answers that
doc's open questions #2-5 as follows: non-parametric piecewise-linear
interpolation over observed bins (not a low-order polynomial — the two
calibrated frequencies have qualitatively different residual shapes, a
smooth bowl vs. a sharp near-field rise-then-plateau, that a single
polynomial risks fitting poorly); a **fixed, bundled default
calibration** (`load_default()`) rather than per-survey fitting; scoped
to the two frequencies actually observed (`hz_1495000`, `hz_547500`),
matched by `freq_hz_start` with a small tolerance for float round-trip,
not built generically for arbitrary frequencies. Question #2 (is a
hardware TVG/AGC curve documented anywhere) was never answered — this
empirical fit is still a workaround for an unconfirmed root cause, not a
substitute for the real curve; see the "v0.13.0" entry below for the
tooling built later to investigate that question directly per-survey.

**Completed:**
- ✅ `processing/range_correction.py` (new module, not folded into
  `apply_tvg()`): `RangeCorrectionTable` (per-frequency piecewise-linear
  lookup, JSON-serializable, `load_default()` reads a bundled calibration
  resource), `fit_range_correction()` (fits `target_level_db - mean` per
  range bin from one TVG-corrected channel, count-weighted target level
  so the fit has ~zero net effect on absolute level), `apply_empirical_range_correction()`
  (adds the fitted correction to `backscatter_intensity`; refuses to
  double-apply via `processing_history`; skips with a warning, doesn't
  guess, for an uncalibrated frequency; non-finite `ground_range_m`
  samples get zero correction, not propagated NaN).
- ✅ New, distinct, **opt-in only** step (`"range_correction"` in
  `pipeline.py`'s step framework) — not added to the default correction
  ladder, so existing consumers see no behavior change.
- ✅ `project-a/scripts/debug/fit_range_correction.py` — the
  fitting/validation CLI that produced the bundled default calibration.
- ✅ Tests: `tests/unit/processing/test_range_correction.py`.
- **Known, documented limitation, not hidden:** the bundled default
  calibration is fit from one real survey and validated only against a
  held-out ping range of that *same* survey — not genuine cross-survey
  validation. Revisit once a second independent real survey is
  available for calibration.

### Test Status (as of 2026-07-01)

```
tests/unit/                   950 passing, 36 skipped
tests/integration/             (run with --slow; requires local JSF/SDF fixtures — see Known Issues #1)
```

---

## Roadmap

### Cross-Project Standard Pipeline Scripts

**Status:** In progress (started 2026-08-06). See
`docs/planning/v0.12.0-standard-pipeline-scripts.md` for the full plan
and its updated status note. **Scope for this round: `sidescanpy` +
`project-a` only** — `project-b` is an empty scaffold with zero real
data, deferred to a future round rather than mirrored unverified.

`sidescanpy` side (library gap-filling + promotion out of
`project-a/scripts/` project-local helpers) is **complete**:
- ✅ `targets/interchange/yolo.py: write_yolo()` — new, mirrors
  `write_coco()`. `targets/core.py` gained public `clip_bbox()`/
  `CONFIDENCE_RANK` (promoted from `interchange/coco.py`'s private
  copies) so both writers share them.
- ✅ `viz/waterfall.py: resolve_waterfall_axes()` — promoted from
  `visualization_demo.py`.
- ✅ `viz/track.py: build_survey_track()` — promoted from
  `visualization_demo.py`.
- ✅ New `viz/targets.py: plot_track_with_targets()` /
  `export_track_with_targets_png()` — promoted from
  `plot_track_with_targets.py`'s `main()` body. Separate module from
  `viz/track.py` (needs `targets.core.Target`, matches the existing
  `viz/contact_sheet.py` precedent of isolating that dependency).
- ✅ `grids/definition.py: bbox_from_channel_dataset()` — promoted from
  `visualization_demo.py`'s `auto_grid_bbox()`.
- ✅ `io/jsf/targets.py: decode_and_correct_window()` — new, wraps the
  decode → TVG → orient-for-display → range-detrend chain previously
  duplicated inline in `export_target_dataset.py`. Deliberately **not**
  in `targets/core.py` (where the original plan sketch put it) —
  `io/jsf/targets.py` already imports `targets.core.Target`, so the
  reverse would be a circular import and a layering violation of
  `targets/core.py`'s own "format-agnostic, no format-specific IO"
  contract.
- All new/promoted code has unit tests (`tests/unit/targets/interchange/
  test_yolo.py`, cases added to `test_waterfall.py`/`test_track.py`/
  `test_definition.py`, new `tests/unit/viz/test_targets.py`, a case
  added to `tests/unit/io/jsf/test_targets.py`). Full suite: 1095
  passing / 36 skipped (was 985 as of 2026-07-16 — grew by more than
  just this session's new tests; not independently reconciled against
  the last recorded count). Two test directories (`tests/unit/viz/`,
  `tests/unit/io/jsf/`) were missing `__init__.py` (every sibling test
  dir has one) — this only surfaced as an actual collection error once
  a new `tests/unit/viz/test_targets.py` collided on basename with the
  pre-existing `tests/unit/io/jsf/test_targets.py`; fixed by adding the
  missing `__init__.py` files to match convention, not by renaming
  either test file.
- **GeoTIFF (stage 10, renumbered — see below) deliberately skipped** —
  still needs the explicit `rasterio` dependency sign-off
  `v0.7.0-geotiff-and-cropping.md` flagged and never got.

`project-a` side (the actual standardized scripts + cleanup) is
tracked in that project's own docs, not here — see its `README.md`/
`CLAUDE.md` for current state.

**2026-08-06 follow-up — far-range crop stage added, pipeline
restructured to edit `.ss.h5` files in place (13 stages, `10` now the
GeoTIFF gap):**
- ✅ New `processing/farrange.py`: `detect_far_range_cutoff_sample()` /
  `crop_far_range_noise()` — implements candidate solution (a) from
  `docs/planning/handoff-hf-farrange-noise-amplification.md` (Known
  Issue #6). Bins raw backscatter by sample index (no geometric
  computation needed — runs immediately after conversion), finds the
  bin with the minimum mean among statistically-reliable bins, and crops
  the sample axis there only if a later bin's mean rises
  `rise_threshold_db` (default 3.0) or more above that floor —
  conservative by design: a channel with no detected rise is left
  untouched, not cropped to a guessed constant.
  - **Real bug found and fixed while building this**: Known Issue #5's
    leading run of exact-zero raw amplitude converts to exactly -120dB
    (`_linear_to_db`'s clip floor), which — if not excluded — is *lower*
    than the genuine far-range floor and gets picked as "the floor" by
    the bin-minimum search; the real, much stronger near-nadir signal in
    the very next bin then reads as a huge spurious "rise," producing a
    false-positive crop right at the start of a channel that has no
    far-range problem at all. Confirmed on real data
    (`unknown/20241017181822.ss.h5`, `hz_557500`: bin 0 read -9.62dB
    only because ~23% of it was blanking-run zeros, vs. bins 1+ reading
    30-36dB). Fixed by masking exact -120dB before binning, mirroring
    `viz/waterfall.py`'s existing `_mask_floor_clip`. Regression test:
    `tests/unit/processing/test_farrange.py::test_leading_floor_clip_run_not_mistaken_for_the_floor`.
  - **Second bug found via real-data idempotency testing**: a channel
    where no crop was needed never got a `processing_history` entry (the
    function only appended history on an actual crop), so a caller
    checking "has this file already been through this stage?" couldn't
    tell "never checked" apart from "checked, nothing to crop" — the
    channel got needlessly re-scanned on every rerun. Fixed:
    `crop_far_range_noise()` now always appends a history entry
    (`params={"cropped": True/False, ...}`).
- ✅ `project-a/scripts/` restructured: correction stages
  (`02_crop_farrange.py`, `04_radiometric.py`, `06_geometric.py`,
  `08_georef.py`) now read-and-overwrite one `ss_h5/` directory in place
  rather than writing a separate directory per stage
  (`ss_h5_radiometric/`/`ss_h5_geometric/`/`ss_h5_georef/`, the original
  v0.12.0 design). Reason: real-data measurement on one 898MB file from
  `survey-a` showed radiometric at 2.8x and geometric at 6.3x
  raw size *each*, kept simultaneously — with three real surveys on this
  machine's actual free space (16GB), the original design didn't fit.
  In-place editing works because `backscatter_raw` stays immutable and
  correction steps only add data (matches this file's own "Lossless
  transforms: `compute_*` adds variables; `apply_*` modifies existing"
  convention) — one file already carries raw + every correction applied
  to it. Writes are atomic (temp file + `Path.replace()`) and each stage
  is idempotent (skips files already past that point). Real-data
  verification: 3 files through the full corrected pipeline settled at
  420MB total (single copy each), vs. an estimated 900MB+ under the old
  4-copy design for the same 3 files — and the crop stage alone cut the
  pre-correction total from 50MB to 27MB before the expensive
  geometric/georef arrays (6 + 2 new per-sample float arrays) even got
  added.
  - **Real bug found via this same real-data testing, root-caused (not
    a `sidescanpy` bug)**: an early version of the old per-directory
    design failed reading a geometric-stage output file with
    `OSError: [Errno group not found: aux_streams] 'aux_streams'`, right
    as free disk dropped from 16GB to 6.2GB during that same test.
    `write_survey()` always writes `aux_streams` unconditionally — no
    code path skips it — so this reads as disk-pressure-induced write
    corruption (`to_netcdf(mode="a")`'s sequential appends likely
    failing/truncating silently under `ENOSPC`), not a logic bug. The
    atomic-write pattern above directly prevents this recurring: a
    write that fails partway never overwrites or is mistaken for a valid
    file, regardless of why it failed.
  - Renumbering: `02_track_maps.py`→`03`, `03_radiometric.py`→`04`
    (rewritten), `04_waterfalls.py`→`05`, `05_geometric.py`→`06`
    (rewritten), `06_ground_waterfalls.py`→`07`, `07_georef.py`→`08`
    (rewritten), `08_grid.py`→`09`, `10_read_targets.py`→`11`,
    `11_track_with_targets.py`→`12`, `12_export_chips.py`→`13`. GeoTIFF's
    gap moves from `09` to `10`.

### v0.13.0 — Selectable TVG Terms + Hardware-Curve Fitting (Complete)

**Status:** Implemented 2026-08-13, per
`docs/planning/v0.13.0-tvg-term-selection-and-hardware-curve-fitting.md`.
Different angle from v0.9.0's empirical range correction above: that
module fits-and-removes whatever residual shape is observed, agnostic to
cause; this one is investigative tooling to help figure out, per survey,
*whether* hardware/acquisition already applied part of TVG before
`sidescanpy` sees the data (Known Issue #4's leading hypothesis, Known
Issue #6's `_gain_factor` evidence) — and, if so, a pipeline option to
avoid double-correcting it. Does **not** resolve the open hardware-AGC
question generally; it's a tool to investigate it per-survey, and the
JSF LF channel's honest result (see below) is itself evidence the
question stays open.

**Completed:**
- ✅ `processing/pipeline.py`: `apply_correction_steps()`/
  `run_correction_steps()` gained a general `step_params: dict[str,
  dict] | None` passthrough (`step_params={"tvg": {"params":
  TVGParams(...)}}`) — closes a plumbing gap, not a math gap:
  `tvg_gain_db()`'s spreading and absorption terms were already
  independently zero-able via `TVGParams`, there was just no way to get
  a custom `TVGParams` through the CLI/pipeline machinery.
- ✅ `analysis/tvg_diagnostics.py`: `fit_hardware_tvg_curve()` +
  `TVGCandidateFit`/`TVGCandidateFitResult`. Fits 4 candidate models
  (`flat`, `spreading_only`, `absorption_only`, `full_physics`) against
  an observed raw-backscatter-vs-range curve, **each with one free scale
  parameter, independently — deliberately not a joint 2-parameter
  regression** (tried during the investigation that motivated this;
  found poorly conditioned — the two terms are too collinear over
  realistic range spans, producing wild/sign-flipped coefficients even
  when the combined curve visually fit well). `r_ref_m` (the loss
  reference bin) auto-selects the max-mean bin among count-eligible
  bins, restricted to the near-field half of the range
  (`r_ref_search_fraction`, default 0.5) — not simply the nearest bin
  (Known Issue #5's blanking/ring-down artifact reads anomalously *low*
  right at the start of every ping and would corrupt the fit's sign if
  picked), and not an unrestricted global max either (found, against a
  real near-flat Klein survey, to occasionally land on a far-range
  noise-level bump purely by chance, collapsing the fit window to a
  single bin and making every candidate degenerate to RMSE=0).
- ✅ `viz/tvg_diagnostics.py`: `plot_candidate_fit()` — observed loss +
  each candidate's scaled model curve, fit window shaded.
- ✅ `project-a/scripts/04_radiometric.py`: `--spreading-loss-db`/
  `--absorption-db-per-m`/`--no-spreading`/`--no-absorption` flags
  (mutually exclusive per term — errors out rather than picking a silent
  winner, since a silent precedence rule risks exactly the kind of
  undetected double-correction this tooling exists to prevent).
- ✅ `project-a/scripts/debug/fit_tvg_hardware_curve.py` (new) — CLI
  wrapping the fit + plot, reusing `tvg_range_residual_diagnostic.py`'s
  `_reconstruct_raw()` helper for already-pipelined `ss_h5` files.
- ✅ Tests: `tests/unit/processing/test_pipeline.py`,
  `tests/unit/analysis/test_tvg_diagnostics.py`,
  `tests/unit/viz/test_tvg_diagnostics.py`. Full suite: 1140 passing / 36
  skipped (was 1122 before this work).
- **Real-data findings (JSF `survey-a`, both channels;
  reconstructed-raw via `_reconstruct_raw()` since production files are
  already fully pipelined):**
  - `hz_1495000` (HF): `absorption_only` wins clearly, scale≈2.18,
    RMSE≈1.07dB — consistent with hardware having already removed
    spreading loss, absorption loss remaining uncompensated.
  - `hz_547500` (LF): `absorption_only` also wins clearly, scale≈2.03,
    RMSE≈0.40dB — **this supersedes an earlier, less careful ad-hoc
    finding** (in the "TVG Parameters or Sound Velocity?" artifact) that
    LF didn't fit any of the 4 candidates well; that ad-hoc analysis
    predates this session's sign-convention and near-field-artifact
    fixes to the fitting logic and was very likely corrupted by the same
    bugs. Both JSF channels now show the same qualitative pattern.
  - Klein clean file (`project-b`, `survey-b`,
    `hz_455000`, `sonar_data180914090300.ss.h5`): near-flat raw curve
    (~0.3dB peak-to-trough over 1-39m) — winning candidate's scale is
    ≈-0.02 (effectively zero), RMSE≈0.01dB, distinguishing it clearly
    from JSF's scale≈2. Consistent with hardware having compensated both
    terms. (The same survey's `hz_900000` channel only records to ~8m
    range — too short a span for a meaningful fit; not a data-quality
    problem, just a short-range high-resolution channel.)
- **Not done:** confirming the hardware-AGC hypothesis directly (no
  instrument documentation was examined) — this tooling produces
  evidence consistent with it, not proof. No change to `absorption_for_frequency()`
  or `tvg_gain_db()`'s formula — both terms were already correct and
  independently zero-able; this work is entirely about exposing that
  capability and adding tooling to decide *when* to use it.

### v0.14.0 — Data Investigation Pipeline (Complete)

**Status:** Implemented 2026-08-23, from an external handoff doc (a
generic 5-stage investigation framework drafted outside this repo) that
asked for it to be built reusing existing tooling wherever possible.
Answers "what does this unknown/unfamiliar data source actually
contain" — units, TVG status, nadir handling, usable range — as a
repeatable pipeline, distinct from `tvg_diagnostics.py`'s
investigate-one-specific-residual scope (this module *uses*
`tvg_diagnostics.py`'s binning/fitting directly rather than duplicating
it).

- ✅ `analysis/investigation.py` (matplotlib-free, mirrors
  `tvg_diagnostics.py`'s analysis/viz split) — `StageResult` dataclass
  + `stage1_characterize`/`stage2_nadir`/`stage3_usable_range`/
  `stage4_decay`/`stage5_units`. Stage 3 reuses
  `mean_backscatter_vs_range`; Stage 4 reuses that plus
  `fit_hardware_tvg_curve` — the candidate-name → TVG-status mapping is
  deliberately inverted (candidate names describe what's still
  *present* in raw data, status describes what's already been
  *removed*) and spelled out explicitly in `_TVG_CANDIDATE_TO_STATUS`
  to avoid silently flipping the whole stage's conclusion.
  `stage4_decay`'s linear→dB conversion (`db_reference_percentile`,
  default 99th) is deliberately **not** the same "dB re 1" convention
  `analysis/tvg_diagnostics.py: _linear_to_db`/`apply_tvg()` use — this
  stage self-references to the channel's own signal level instead
  (no vendor calibration is ever available for this data, so an
  absolute dB reference is arbitrary; a documented relative one is
  reproducible). Provably inert to classification (every downstream
  number is a dB *difference*, which cancels a constant reference
  shift) — see the docstring and
  `test_classification_invariant_to_db_reference` for the proof this
  actually holds in the implementation, not just in theory. Also
  gained `db_reference_value` (overrides the percentile-computed
  reference with an exact linear value, e.g. `1.0` to reproduce the
  "dB re 1" convention directly) — added specifically to empirically
  run all three reference options from the strategy doc (99th
  percentile / fixed dB-re-1 / median) side by side on real data for
  the investigation report's "Final results" section; all three agreed
  to 6+ significant figures on both channels, confirming the proof
  above in practice, not just in the abstract.
- ✅ **Outlier rejection for Stage 3/4 (2026-08-23, user-requested):**
  `reject_range_outliers()` — per-range-bin, median/MAD-based rejection
  (deliberately aggressive default `mad_k=2.5`, deliberately willing to
  discard genuine hard-target/complex-terrain returns) so Stage 3's
  noise floor and Stage 4's decay fit can target a flat,
  sandy-bottom-like population instead of being pulled around by real
  but calibration-irrelevant returns. Opt-in only
  (`reject_outliers=False` default on both stages, plus the CLI's
  `--reject-outliers`) — explicitly **not** the right default for
  anything display/export-facing, where those same bright returns are
  usually genuine signal worth keeping (see this same session's "why is
  LF's range so high" investigation-report finding, which argues the
  opposite call for that use case). Tests:
  `tests/unit/analysis/test_investigation.py::TestRejectRangeOutliers`
  (4 tests) plus one integration test each on Stage 3/Stage 4 showing
  the option actually changes their output on injected-spike synthetic
  data. Full suite (through the `value_col` follow-up below too):
  1163 passing / 36 skipped (was 1155).
  **Real-data check, mixed result, worth knowing before reaching for
  this by default:** on `survey-a` at `mad_k=2.5`, ~4-5% of
  samples get rejected per channel (not just the ~0.0001%-rare true
  point-target population the earlier LF-outlier investigation found —
  a MAD threshold this aggressive on right-skewed backscatter trims a
  meaningful slice of ordinary speckle too, which is expected given how
  aggressive `mad_k=2.5` deliberately is). Stage 3's usable-range
  estimate benefited clearly (LF: noise_std 1.67→1.14, usable range
  46m→52m). **Stage 4's fit RMSE got measurably *worse*, not better**
  (HF: 0.22→0.39dB; LF: 0.28→0.38dB).
- ✅ **`fit_hardware_tvg_curve(..., value_col="mean"|"median")` +
  `stage4_decay(..., fit_value_col=...)` (2026-08-23, follow-up to the
  RMSE regression above) — tried, did NOT reliably fix it.** Theory:
  symmetric MAD-based trimming isn't distribution-neutral on a
  right-skewed population, so it shifts the plain arithmetic `mean`
  `fit_hardware_tvg_curve` fits against; `median` should be robust to
  that. **Real-data result was mixed, not a clean win:** switching
  `"mean"`→`"median"` under `reject_outliers=True` improved HF (0.39→
  0.35dB, still worse than the 0.22dB no-rejection baseline) but made
  LF *worse* (0.38→0.66dB). **What stayed stable across every
  configuration tested (both `value_col` choices, rejection on or
  off): the winning candidate and TVG-status classification** — only
  the RMSE number moved around. Treat Stage 4's RMSE as noisy/
  unreliable under `reject_outliers=True` regardless of `value_col`;
  treat the classification as the trustworthy output. Left both
  options in (real, useful, well-tested capabilities — `value_col` is
  also a clean, zero-risk addition to `fit_hardware_tvg_curve` in its
  own right, opt-in default `"mean"` preserves every existing caller's
  behavior exactly) but documented the honest result in both
  functions' docstrings rather than claim a fix that didn't
  materialize. This survey also has very little genuine target/terrain
  contamination to begin with (same finding above), so none of this
  may generalize to a survey that actually needs the option — revisit
  if/when one comes up.
- ✅ **2026-08-23 same-day follow-up, user pushback ("maybe I rejected
  too aggressively") — retuned `mad_k` default 2.5→3.5** (the
  commonly-cited Iglewicz & Hoaglin "definite outlier" threshold —
  literature-backed, not arbitrary). Real-data rejection dropped from
  ~4-5% to **1.6% (HF) / 2.4% (LF)** per channel — much closer to a
  genuine tail-trim. Stage 3 benefit still holds at the new default
  (LF noise_std 1.67→1.31, usable range 46m→50m). **Re-ran the
  `fit_value_col` comparison at the new default and found something
  worse than the earlier "mixed RMSE" result: pairing
  `reject_outliers=True` with `fit_value_col="median"` flipped LF's
  classification itself**, from `spreading_only` to `ambiguous` (best
  candidate changed `absorption_only`→`full_physics`, runner-up gap
  closed enough to trip the ambiguous threshold) — the earlier
  "classification always stays stable regardless of value_col" claim
  in this same file was true at `mad_k=2.5` but **is not true in
  general** and has been corrected in both this entry and
  `stage4_decay`'s own docstring. Updated guidance:
  `reject_outliers=True` paired with the `"mean"` default preserved
  classification in every real-data case tried; pairing with
  `"median"` did not — don't reach for `"median"` as a default fix,
  treat it as an experimental option to sanity-check against `"mean"`.
- ✅ **Stage 2 water-column boundary made explicit (2026-08-23,
  user-requested — "I do not think you are identifying the water
  column"). Superseded same day — see the follow-up entry below; the
  geometric-only boundary this entry describes turned out to be
  substantially wrong, not just incomplete. Left for historical
  record.** `stage2_nadir()` gained a `range_var` param (default
  `"ground_range_m"`): surfaces the water-column boundary that
  `compute_geometric_ranges()` already encodes (`ground_range_m` is
  NaN wherever `slant_range_m <= altitude_m`) as explicit per-ping
  metrics (`water_column_end_sample_median`/`p10`/`p90`) and a
  `first_bottom_sample` array in `.data`, rather than leaving it an
  invisible side effect of NaN masking. Also computes a corroborating
  signal-step check (does `value_var` actually step up across the
  boundary on a representative ping?) — **real-data result itself is
  informative, not just confirmatory**: LF shows a clear step (1.35×,
  no note), HF doesn't (0.95×, essentially flat, generates a note).
  Not a bug — the geometric crossing marks where grazing angle first
  becomes nonzero, not where backscatter has ramped up to a strong
  echo, which can build gradually further out. Confirmed via the new
  waterfall boundary overlay (below) that the boundary sits safely
  inside the dark/empty region on both channels, well before the
  visible bright signal wall — conservative, not permissive, which is
  the safe direction for Stage 3/4's fitting.
  - **Confirmed structurally (not a new behavior, just newly visible):
    Stage 3/4 already excluded water-column samples from curve
    fitting before this session** — both bin by `ground_range_m`,
    and `mean_backscatter_vs_range`'s own `isfinite` check already
    drops NaN ranges. Both stages now report this explicitly via a
    new `water_column_n_excluded` metric (real data: ~2.36M HF /
    ~1.39M LF samples, ~3.1%/~2.2% of each channel) instead of it
    being invisible. Directly answers the user's ask ("we do not want
    to include the water column in our curve fitting... I think we
    want that water column return to get pushed to near zero or just
    identified and removed entirely") — it already was, now provably so.
  - `viz/investigation.py`: `plot_stage2_nadir_zoom()` gained a
    `stage2` param — widens its zoom window to actually reach the
    detected boundary (previously fixed at `nadir_n_samples *
    zoom_factor`, e.g. ±100 samples, vs. a real median boundary of
    ~130-190 samples — far too narrow to show the transition at all,
    which was the direct cause of the user's "I'm not confident"
    concern) and marks the boundary with a shaded region + line.
    `plot_stage2_waterfall()` gained the same param — traces the
    per-ping boundary as an overlaid line across the full survey, the
    most direct way to visually audit it against the real signal
    structure (see the confirmation above).
  - Tests: 5 new (`test_water_column_boundary_skipped_without_range_var`,
    `test_water_column_boundary_matches_geometry`,
    `test_signal_step_note_when_no_real_step`,
    `test_signal_step_no_note_with_real_step`,
    `test_water_column_n_excluded_matches_geometry`). Full suite: 1170
    passing / 36 skipped.
- ✅ **Same-day correction: the geometric boundary above was itself
  substantially wrong — caught by a direct human read of the raw
  waterfall, not by any tooling.** The user disputed the reported
  boundary directly: *"Your estimate of the water-column boundary is
  off by an order of magnitude... I would guess for ping 4000 it
  starts at about sample 250."* Precise sample-by-sample inspection
  confirmed it: HF ping 4000's geometric crossing is genuinely at
  sample 141, but amplitude stays weak (0.05-0.4 vs. a ~22 peak) all
  the way to ~sample 300, then jumps sharply by ~sample 325-330 — the
  geometric crossing (`slant_range > altitude`) marks where the beam
  *mathematically* reaches bottom, not where the return is actually
  *strong*; grazing angle right at that crossing is still too shallow.
  **New:** `detect_first_strong_return()` — a genuine amplitude-based
  detector (rolling-mean smoothing + a per-ping-relative threshold +
  sustained-run confirmation, vectorized, no per-ping Python loop),
  validated against the same ping (lands at sample 320-333 across
  `threshold_fraction` 0.1-0.3, matching the manually-confirmed ~325-330
  onset). `stage2_nadir()` now reports **both** boundaries, clearly
  named and distinguished (`geometric_bottom_contact_sample_*` vs.
  `first_strong_return_sample_*`), plus their gap
  (`geometric_to_strong_gap_sample_median`) and a signal-step check
  re-anchored on the *strong* boundary — real data now shows a clean
  step there on both channels (HF 9.4×, LF 6.0×; the old
  geometric-anchored check had read 0.95×/1.35×, which in hindsight
  was itself the tell that something was off).
  - **`stage3_usable_range`/`stage4_decay` gained `exclude_before_sample`**
    (a `(ping, channel)` NaN mask applied before binning, layered on
    top of the existing geometric exclusion) and
    `run_investigation()` wires it in by default
    (`exclude_weak_zone=True`) using `stage2`'s
    `first_strong_return_sample`. **Second bug found while wiring this
    in, same day:** feeding the raw *per-ping* boundary straight into
    aggregate binning is wrong — pings' boundary location correlates
    with their own near-field amplitude (shallower altitude -> both an
    earlier onset and a stronger near return), so per-ping masking
    selectively keeps only the strongest, earliest-onset pings' data
    in near-range bins. On HF this produced a spurious near-field spike
    (mean 6.3 vs. 2.9 in the next bin out) that `fit_hardware_tvg_curve`'s
    max-mean `r_ref_m` auto-selection picked up, moving the reference
    from a physically sensible 7.5m to 2.5m and nearly quadrupling
    RMSE (0.22dB -> 0.87dB). **Fixed:** new
    `uniform_exclusion_boundary()` collapses the per-ping array to one
    conservative, survey-wide value (90th percentile) before masking —
    `r_ref_m` returned to exactly 7.5m/12.5m (matching the pre-exclusion
    values), confirming the fix. Real-data RMSE after both fixes: HF
    0.70dB, LF 0.37dB (higher than the original un-excluded 0.22/0.28dB
    — expected and correct, since the fit is now against a properly
    cleaned curve, not a regression). Classification unchanged
    (`spreading_only` both channels).
  - **This also resolved the earlier `reject_outliers`/`fit_value_col`
    "mixed result" writeup above (2026-08-23 mad_k retune entry) —
    that check was confounded by the per-ping exclusion bug just
    described, not a real property of `reject_outliers` or
    `fit_value_col`.** Re-run against the corrected uniform-boundary
    baseline: `reject_outliers=True` *improves* RMSE on both channels
    (HF 0.70->0.36-0.39dB, LF 0.37->0.30-0.38dB depending on
    `fit_value_col`), and classification stays `spreading_only` in
    every combination tried, `"mean"` or `"median"` — no ambiguous
    flip. Both functions' docstrings updated to record this as a
    caution about verifying a surprising result isn't a symptom of a
    different bug before writing it down as a property of the thing
    under test.
  - `viz/investigation.py`: `plot_stage2_nadir_zoom()`/
    `plot_stage2_waterfall()` now mark and label both boundaries
    distinctly (dotted olive = geometric, solid teal = strong-return)
    and shade the weak near-grazing zone between them.
  - Tests: 9 new/rewritten (`TestDetectFirstStrongReturn` — 4,
    `TestUniformExclusionBoundary` — 3 including a synthetic
    reproduction of the selection-bias bug, plus 2 Stage 2/Stage 3
    integration tests). Full suite: 1179 passing / 36 skipped.
- ✅ **New Stage 2.5 — beam geometry (2026-08-23, user pushback on the
  interpretation, not just the numbers).** After the boundary-detector
  fix above, the user offered a competing physical explanation for the
  weak zone: not gradual near-grazing-incidence scattering (Stage 2's
  implicit assumption — a beam pointed straight down at nadir), but a
  sidescan beam mounted at a genuine tilt that simply doesn't
  illuminate anywhere near directly below the towfish — "the weak zone
  isn't weak backscatter, it's the sonar looking at nothing." Also
  raised a second, harder-to-rule-out alternative: `altitude_m` itself
  could simply be wrong. Built a real diagnostic rather than argue
  from priors: `stage2_5_beam_geometry()` — for a set of candidate
  beam-tilt angles (straight lines, no real directivity/beam-pattern
  modeling — deliberately simple per the user's own "just straight
  angles for now" scope), computes where each would first intersect a
  flat seafloor at the ping's own `altitude_m`, and back-calculates the
  angle *implied* by where the real signal actually starts
  (`arccos(altitude_m / observed_slant_range_m)` at
  `first_strong_return_sample`). `viz.investigation.plot_stage2_5_beam_geometry()`
  draws this as an axial (side-view) diagram directly above the real
  single-ping profile on a shared across-track-distance x-axis, so
  candidates can be checked by eye, not just by a table of numbers.
  - **Real-data result favors the tilted-beam explanation clearly:**
    implied angle 25° below horizontal (HF) / 18° (LF) — both
    ordinary, physically unremarkable sidescan mounting angles, not
    extreme values — and the implied-angle line lands almost exactly
    on the visible boundary between the flat/near-zero region and the
    real signal on both channels. This is evidence against Stage 2's
    original 0°-from-vertical framing, not just an alternative that
    also fits.
  - **Cannot rule out the altitude-is-wrong alternative from this
    survey's data alone** — a wrong `altitude_m` shifts every
    candidate ray (and the implied angle) together without changing
    which one fits best relative to the others. Said so explicitly in
    both the docstring and the report rather than overclaim the tilted-
    beam explanation as proven; an independent altitude/depth source
    (DVL, chart depth, a second sensor) would be needed to separate the
    two hypotheses.
  - **Deliberately not hardcoded to this survey's beam pattern** — the
    user flagged that other datasets show nadir as *bright* rather than
    a *gap*, implying a genuinely different near-nadir beam geometry
    that would need a different candidate angle set and produce a
    different-looking picture; the function takes
    `candidate_angles_from_vertical_deg` as a parameter for exactly
    this reason, and the skill doc says explicitly not to reuse one
    survey's angle numbers for a differently-behaved one.
  - **Real bug caught by a test, not by inspection:** the "impossible
    geometry" check (observed slant range less than altitude) only
    tested `slant > 0`, not `slant >= altitude` — so a genuinely
    impossible case (slant < altitude) silently produced `angle = 0°`
    via `arccos(clip(ratio, -1, 1))` instead of being flagged. Fixed;
    regression test added.
  - Tests: `TestStage2_5BeamGeometry` (4 tests, including one that
    isolates the angle math from `detect_first_strong_return`'s own
    smoothing imprecision by feeding a hand-built `StageResult` with an
    exact known boundary, rather than conflating two things worth
    testing separately). Full suite: 1183 passing / 36 skipped.
- ✅ **Stage 1.5 linear-panel readability fix (2026-08-23,
  user-requested).** `plot_stage1_5_before_after()`'s linear-scale
  panel previously let matplotlib auto-scale independently per `hist()`
  call — with `before` spanning the full range (up to 1280 on LF), the
  entire readable bulk distribution (before *and* after) collapsed
  into an unreadable sliver at the left edge. Fixed: both panels now
  use one shared set of bin edges (matplotlib picks different edges
  per call otherwise, silently misrepresenting the overlay), and the
  linear panel's x-axis is capped at the `before` population's 99.5th
  percentile (`linear_xlim_percentile` param) rather than its true
  max, with an annotation reporting how many/how-large the off-screen
  values are — nothing is dropped from the underlying data, just the
  view.
- ✅ **Stage 1.5 (`stage1_5_outlier_comparison` + `plot_stage1_5_before_after`,
  2026-08-23, user-requested)** — re-runs Stage 1's characterization
  before/after `reject_range_outliers`, always computed by
  `run_investigation()` regardless of whether `reject_outliers` is on
  for Stage 3/4 (a preview, not gated behind that flag). Real-data
  check: both channels show a visible trim across the upper-middle of
  the distribution, not just the extreme tail — consistent with (and a
  good visual confirmation of) the ~4-5%-of-samples-rejected finding
  above, not a bug. Tests:
  `TestStage1_5OutlierComparison` (2 tests). Full suite: 1165 passing.
- ✅ `viz/tvg_diagnostics.py: plot_candidate_fit()` — "observed" markers
  changed from tiny filled dots (`markersize=3`) to larger open circles
  (`"o"`, `markersize=7`, unfilled) per user request, for legibility
  against the dashed candidate-fit lines. Purely cosmetic, no test
  asserted the old style.
- ✅ `viz/investigation.py` — one plot function per stage's qualitative
  output; reuses `viz.waterfall.plot_waterfall` (via aliasing
  `value_var` onto `backscatter_intensity` on a shallow copy, since
  that function reads a fixed variable name) and
  `viz.tvg_diagnostics.plot_candidate_fit` directly rather than
  reimplementing either.
- ✅ `investigation.py` (new top-level module) — `run_investigation()`,
  the library-level orchestrator the handoff asked for: loads a
  `.ss.h5`/`.jsf`/`.sdf` file, reconstructs raw if needed, computes
  geometric ranges if missing, runs all five stages, and (if
  `output_dir` given) saves one PNG per stage plus `metrics.json`.
  Meant to be reused as-is by any consumer, including a future
  field port — the actual CLI wrapper stays in the consumer
  project (`project-a/scripts/investigate_dataset.py`), per this
  file's own "domain logic in the library, CLI/script orchestration in
  the consumer project" convention — `run_investigation()` itself isn't
  a CLI, so it's the one exception living here rather than there.
- ✅ `processing/pipeline.py: reconstruct_raw()` — promoted from a
  helper duplicated across two `project-a` debug scripts
  (`tvg_range_residual_diagnostic.py`, `fit_tvg_hardware_curve.py`);
  this module's own need for it would have been a third independent
  copy.
- ✅ Tests: `tests/unit/analysis/test_investigation.py` (15 tests, small
  synthetic fixtures per TDD convention) — includes a regression guard
  documenting `fit_hardware_tvg_curve`'s own near-flat-data
  fit-degeneracy behavior (a truly flat+noise curve can have a
  non-`flat` candidate technically edge out `flat` on RMSE by
  overfitting the noise; the test asserts "no real decay detected"
  rather than a specific winning candidate name — see that test's
  docstring) and `test_classification_invariant_to_db_reference`
  (proves the dB-reference choice above is actually inert to
  classification, not just in theory). Full suite: 1155 passing / 36
  skipped (was 1140 — zero
  regressions).
- **Real-data validation** (`project-a`'s `survey-a`
  survey, both frequencies, 8000 pings, blind on `backscatter_raw`):
  independently reproduced three things this project had previously
  found by hand for this exact survey — units (`raw_counts`, matching
  the recorded `processing_history` metadata), TVG status (spreading
  already removed / absorption remaining, matching the v0.13.0 finding
  above almost exactly), and nadir (`gap`, matching Known Issue #5's
  documented leading zero-run).
- **New, previously-undocumented finding surfaced by this validation
  run:** `apply_tvg()`'s logged `processing_history` metadata
  (`radiometric.py:404`) hardcodes `"raw_format": "linear_amplitude
  (0-81)"` for every channel regardless of that channel's actual range
  — confirmed wrong for `survey-a`'s LF channel
  (`hz_547500`), whose real raw range is 0-1280. Cosmetic only (the
  string isn't used in the conversion math), but actively misdescribes
  the data to anyone who trusts that field later. **Not fixed** — flagged
  as a recommendation in the investigation's report artifact, not
  addressed in this session (out of scope for what was asked).
- Skill: `sidescan-data-investigation`
  (`<workspace>/.claude/skills/sidescan-data-investigation/SKILL.md`,
  family-root-scoped like `sidescan-sonar`) documents the pipeline and
  how to invoke/extend it — separate from `sidescan-sonar`, which stays
  the domain-physics reference.
- ✅ **Stage 3/4 exclusion-boundary split — per-ping for Stage 3,
  uniform for Stage 4 (2026-08-23, user-caught real artifact).** After
  the Stage 2.5 beam-geometry finding above confirmed
  `first_strong_return_sample` as a real, working detector, the user
  asked for an early masking step that removes every nadir/water-column
  sample before Stage 3/4 run at all ("only run the analysis on actual
  bottom hits"). First implementation fed
  `uniform_exclusion_boundary()`'s single survey-wide (p90) cutoff into
  *both* stages (reusing the value already needed for Stage 4's
  near-field selection-bias fix, above). The user caught a real
  consequence directly off the regenerated plot: LF's usable-range
  curve had an anomalous near-zero first point. Root cause: a uniform
  cutoff under-masks whichever ~10% of pings' true boundary exceeds it;
  because ground range at a fixed sample index depends on that ping's
  own altitude, those pings' still-invalid leaked samples can land in a
  *different* (nearer) range bin than where the rest of the survey's
  valid signal starts — confirmed on real data by comparing bin
  contents directly: the uniform boundary left a `range=8-10m,
  count=30, median=0.39` bin (thin and near-zero — contamination),
  while the exact per-ping boundary's nearest bin was `range=2-4m,
  count=305, median=14.75` (thin but genuinely valid — a few pings'
  real near boundary, not leaked invalid samples). Fixed in
  `run_investigation()`: Stage 3 now gets the exact per-ping
  `first_strong_return_sample` array directly (safe — Stage 3 does no
  aggregate max-search across bins, so per-ping precision has no
  downside there); Stage 4 keeps the uniform boundary (still needed —
  its `r_ref_m` auto-selection *does* do a max-mean search, which is
  exactly what per-ping masking was found to bias in the original
  near-field selection-bias fix, above). Real-data re-run: LF/HF
  classification unchanged (`usable_range_m` 46.0/36.0,
  `best_fit_model=absorption_only`, `tvg_status=spreading_only`); LF's
  usable-range plot no longer shows the near-zero artifact. Regression
  test: `TestStage3UsableRange.test_per_ping_boundary_avoids_thin_contaminated_near_bin_vs_uniform`
  (synthetic repro using the same correlated-boundary/altitude physics
  pattern as `test_avoids_near_field_selection_bias_vs_per_ping_boundary`,
  adapted for Stage 3's opposite conclusion). Full suite: 1184 passing
  / 36 skipped.
- ✅ **Stage 4 decay-fit plots now use distance-from-reference on the
  x-axis, not absolute range (2026-08-23, user-requested).** The y-axis
  was already "loss relative to r_ref" but the x-axis stayed absolute
  range in meters — inconsistent, and made it harder to read the curve
  as "how much loss accumulates past the point we normalized against."
  `viz/tvg_diagnostics.py: plot_candidate_fit()` and
  `viz/investigation.py: plot_stage4_residuals()` both now plot against
  `range_m - r_ref_m`, with a vertical line and the shaded fit-window
  span anchored at `x=0`; the x-axis label states `r_ref` explicitly
  (`"Distance from reference range (r_ref=7.5m) (m)"`). The reference
  vline is drawn once per `ax` (on `show_observed=True`), matching the
  existing "draw the observed curve once, overlay each candidate"
  calling convention, not once per candidate. Test updated
  (`test_repeated_calls_show_observed_once`: 5→6 expected lines, for
  the added vline). Full suite: 1184 passing / 36 skipped.
- ✅ **New `analysis/investigation.py: mask_weak_zone()` — water column
  now NaN'ed in displayed/exported waterfalls, not just excluded from
  curve fitting (2026-08-23, user-caught gap).** The Stage 3/4 fixes
  above stopped *fitting* against the nadir gap + weak near-grazing
  zone, but corrected/recommended waterfall exports kept *displaying*
  it — caught directly on the report's processing-step progression
  figure, where the panel captioned "the version to actually use
  downstream" still showed the water-column return. `mask_weak_zone(ds,
  value_vars=("backscatter_raw", "backscatter_intensity"))` reuses
  `detect_first_strong_return` (the same amplitude-based boundary
  Stage 2/3 already validated) and NaNs every sample before it, per
  (ping, channel), in whichever of `value_vars` are present — skips
  vars not present rather than raising, so it works whether or not TVG
  has been applied yet; always uses the exact per-ping boundary (no
  aggregation-bias risk for a display copy — see
  `uniform_exclusion_boundary`'s docstring for why Stage 3 vs Stage 4
  need different granularities of this same boundary internally, which
  doesn't apply here). Applied to the report's progression figure
  (panels 2–4, the corrected/recommended views); panel 1 (raw)
  deliberately left unmasked since it exists specifically to show what
  the sensor actually recorded. Tests: `TestMaskWeakZone` (3 tests —
  masks before boundary/leaves rest untouched, only masks vars present,
  doesn't mutate the input dataset). Full suite: 1187 passing / 36
  skipped.
- ✅ **Stage 2.5 now classifies the weak zone's physical interpretation
  automatically, computed across every ping, not left implicit
  (2026-08-23, user pushback: "you are still referring to the water
  column as a weak zone... this may not be the case in other datasets,
  the automated pipeline should spit that out").** `stage2_5_beam_geometry()`
  gained a `classification` (`"water_column_tilted_beam"` /
  `"near_grazing_weak_backscatter"` / `"ambiguous"` /
  `"undetermined"`), based on the cross-ping median/std of the implied
  angle (vectorized across every ping's own
  `first_strong_return_sample`, not just the single `ping_idx` used for
  the diagram plot — a single ping's detector noise is a weak basis for
  a survey-wide conclusion). Surfaced in `InvestigationResult.summary_row()`
  as `weak_zone_physical`/`weak_zone_implied_angle_deg`, so it shows up
  in the CLI's printed summary and `summary.json` without digging into
  a specific stage's metrics. Real-data re-run on `survey-a`:
  median implied angle 64.6°/73.0° from vertical (std 0.8°/1.0° across
  8000 pings, 100% valid) → `water_column_tilted_beam` both channels —
  matches the earlier single-ping 25°/18°-from-horizontal finding
  almost exactly (`90 - 64.6 = 25.4`, `90 - 73.0 = 17.0`), confirming
  the aggregate computation is consistent with the original manual
  read. Tests: `TestStage2_5BeamGeometry` gained 4 classification tests
  (one per outcome, using a new `_make_angle_ds` helper that directly
  constructs `slant_range_m` for an exact target angle, bypassing
  `compute_geometric_ranges`' coarse 0.75m/sample discretization which
  can't hit small target angles). Full suite: 1191 passing / 36
  skipped.
- ✅ **Battle-tested the pipeline hands-off across 6 more real surveys
  beyond `survey-a` (2026-08-23)** — 2 more `project-a` JSF
  surveys (`unknown`, `20250417-Unknown`) and 4 `project-b` Klein SDF
  surveys (`survey-b`, `site-c survey`,
  `site-d`, `site-e`), 14 frequency-channel runs
  total, **zero crashes**, both raw `.jsf`/`.sdf` files (never
  converted) and already-converted `.ss.h5` accepted directly with no
  manual intervention — the CLI already supported this
  (`_load_channel_dataset` dispatches on extension), confirmed for the
  first time on genuinely fresh files this session. Real findings, not
  hypothetical:
  - **JSF (`project-a`, nadir=`gap`) generalizes cleanly**: both new
    surveys reproduce `survey-a`'s shape almost exactly —
    `water_column_tilted_beam`, clean monotonic usable-range decay,
    sensible `usable_range_m` (34m/26m and 28m/46m). The tilted-beam
    finding isn't a one-survey fluke; it's an EdgeTech-hardware-wide
    characteristic on every JSF survey checked so far.
  - **Klein SDF (`project-b`, nadir=`bright_spike` on every survey
    checked) is a genuinely different animal, not just a cosmetic
    nadir-handling difference — confirms the user's suspicion.**
    `detect_first_strong_return`'s core assumption (weak near range,
    strong far range, detect the transition) is violated by a bright
    nadir: the very first samples are the *brightest* part of the
    ping (a real specular return, confirmed by direct inspection of
    `stage2/nadir_zoom.png`), so the detector fires at sample ≈0 on
    nearly every ping — not a real "first bottom contact," a spurious
    trigger on the nadir spike itself. **The existing safety nets
    correctly catch this rather than silently drawing a wrong
    conclusion**: the "impossible geometry" check (sample-0's slant
    range is always less than altitude) flags nearly every ping, so
    `implied_angle_valid_fraction` stays under the 20% threshold and
    Stage 2.5 correctly reports `undetermined` rather than a false
    `water_column_tilted_beam`/`near_grazing_weak_backscatter`. Treat
    any `first_strong_return_sample ≈ 0` finding on bright-nadir data
    as suspect by default, not confirmation that real bottom starts at
    nadir.
  - **`usable_range_m` came back `None` on all 6 Klein SDF
    frequency-channels tested — a real, reproducible format
    characteristic, confirmed root-caused, not a bug:** two distinct
    mechanisms, both found by direct inspection, not guessed.
    (1) One channel (`site-b`/900kHz) had **zero** finite
    `ground_range_m` samples — its recorded window's max slant range
    (13.4m) never reaches the vehicle's own altitude (17-31m) at *any*
    sample, so geometrically nothing in that channel ever reaches
    bottom (a short-range acquisition setting mismatched to how high
    the vehicle was actually flying that pass, not a processing bug).
    `stage2_nadir`'s all-NaN-`ground_range_m` note now names this
    specific cause automatically when detectable, instead of a generic
    "no finite values" (`TestStage2Nadir.test_all_nan_ground_range_names_short_recorded_window_cause`).
    (2) On every other Klein channel, per-range-bin **median** stays
    essentially flat/pinned at a single value across the *entire*
    recorded window regardless of range (real numbers: 64.0 exactly,
    repeated identically across three unrelated surveys/locations for
    the 455kHz channel) — Stage 3's SNR calc is hardcoded to `median`
    (`stage3_usable_range`, unlike Stage 4 which already has
    `fit_value_col`), so `binned["median"] - noise_floor` is ~0
    everywhere and SNR never crosses threshold. Checked whether `mean`
    would do better (direct binning, with and without
    `reject_outliers`): it doesn't — mean stays similarly flat
    (69-91 across 56m) once outlier-contaminated far-range spikes are
    stripped, so this isn't a value_col fix waiting to happen, it's
    consistent with Klein's real-time AGC (independently confirmed
    active via `_gain_factor`-style evidence — see Known Issue #6's
    "2026-08-12 follow-up" above) having already flattened backscatter
    within these particular short-range survey segments. **Not fixed
    (correctly, by the user's own scope instruction this session: "you
    should not be doing work on the actual pipeline... if that doesn't
    work fix it" — investigated, root-caused, confirmed not a bug, so
    left alone)** — if a real Klein-SDF usable-range need comes up
    later, the fix is a `stage3_usable_range` `value_col` param
    mirroring Stage 4's, not yet built since there's no evidence a
    different statistic would actually help on the data checked so
    far.
  - Result artifact (cross-format comparison, all 7 surveys/14 runs):
    see the `sidescan-data-investigation` skill's "Validated real-data
    run" section for the link.
- ✅ **Terminology correction + a real nadir-noise-blanking fix + Stage
  2.5 redesign (2026-08-23, same day, user physics correction: "you are
  still messing some stuff up").** The battle test above surfaced a
  genuinely wrong mental model, not just a naming nitpick — the user
  corrected it directly:
  - **"Geometric bottom contact" is actually just altitude** — slant
    range crossing altitude isn't a bottom-contact boundary, it's
    altitude re-expressed in sample units. Renamed throughout:
    `geometric_bottom_contact_sample_*` → `altitude_sample_*`,
    `first_strong_return_sample_*` → `first_bottom_return_sample_*`
    (the detector was correctly finding the real bottom echo all
    along, the old name undersold it), `geometric_to_strong_gap_sample_median`
    → `altitude_to_bottom_gap_sample_median`. `detect_first_strong_return`
    → `detect_first_bottom_return`.
  - **The span between altitude and the first bottom return isn't
    "weak near-grazing backscatter" — it's part of the nadir gap.** A
    real beam's near edge is never at 0° from vertical, so that span
    is simply never illuminated at all, same underlying reason as the
    acquisition-blanking gap at the very start of the ping, not a
    separate physical phenomenon. `mask_weak_zone()` → `mask_nadir_gap()`,
    `weak_zone_n_excluded` → `nadir_gap_n_excluded`,
    `exclude_weak_zone` → `exclude_nadir_gap` (all in
    `analysis/investigation.py` and `investigation.py`).
  - **New real processing step: `analysis/investigation.py: blank_nadir_noise()`
    + `detect_nadir_noise_extent()`.** A high-noise burst sits right at
    nadir on some hardware — pure electronics/ringdown noise, not a
    return. EdgeTech already zeros this before a JSF file is written
    (confirmed: JSF's nadir has always read as an exact-zero `gap`).
    Klein/SDF doesn't, and — this was the user's key correction — what
    the pipeline had been calling a bright-nadir "specular return" was
    actually just this un-blanked noise, not a real echo.
    `detect_nadir_noise_extent()` is the mirror image of
    `detect_first_bottom_return` (finds where an initially strong
    signal falls to a sustained low level, instead of where a weak
    signal rises to a sustained high one), narrowed to a
    `search_window` near the very start of the ping so a bright real
    bottom return much further out can't inflate the threshold and
    mask a genuinely short burst. `blank_nadir_noise()` zeros the
    detected extent — verified idempotent (a no-op) on already-zero
    JSF data via the same math (peak=0 → threshold=0 → sustained from
    sample 0 → boundary=0). `run_investigation()` now calls this
    automatically, before every other stage, no toggle (matches
    `reconstruct_raw`/`compute_geometric_ranges`'s "always-on
    preprocessing" convention, not a debatable option like
    `reject_outliers`).
  - **Stage 2.5 completely redesigned**: no longer sweeps a fixed
    candidate-angle set (0/15/30/45/60/70/80° from vertical) and asks
    which fits best. Now triangulates the beam's real **near edge**
    (from the first bottom return + altitude, cross-ping median/std)
    and real **far edge** (from Stage 3's `usable_range_m` + altitude
    — `arctan(usable_range_m / altitude)`), reporting
    `near_angle_deg_median`, `far_angle_deg`, `beam_center_angle_deg`
    (mean of the two), and `beam_width_deg` (far minus near) directly.
    Requires `stage3` as a new positional arg —
    `run_investigation()`'s stage order changed accordingly (**Stage 3
    now runs before Stage 2.5**, since the far edge needs
    `usable_range_m` already computed). `classification` dropped the
    old `water_column_tilted_beam`/`near_grazing_weak_backscatter`
    dichotomy entirely — that distinction assumed two different
    physical stories were possible (weak-but-real backscatter vs. true
    water column), and the user's correction established there's only
    one: the near-nadir span is never illuminated, full stop, regardless
    of how small the measured near angle happens to be. `classification`
    now reports measurement confidence only: `"determined"` /
    `"ambiguous"` / `"undetermined"`. `viz.investigation.plot_stage2_5_beam_geometry()`
    redrawn to match — one near ray (solid), one far ray (dashed), the
    wedge between them shaded as the beam's actual illuminated span,
    instead of 7 candidate rays.
  - **Re-ran the same 7-survey/14-run battle test with all of the
    above in place — the finding changed, not just the label.** Every
    Klein SDF survey's nadir classification flipped from `bright_spike`
    to `gap` (confirmed visually: the ~40,000-amplitude spike at
    sample 0 in site-c/900kHz's `nadir_zoom.png` is now flat zero,
    real terrain clearly visible on both sides starting further out).
    Several channels that were `beam_geometry: undetermined` before
    blanking now report `determined` with a real near-angle
    measurement (e.g. site-c/455kHz: 73.4°, site-e/900kHz:
    79.3°) — blanking fixed the specific noise-burst failure mode, not
    every source of per-ping detector noise on this hardware, so some
    channels remain `undetermined`. `usable_range_m` is still `None`
    on every SDF channel — confirmed unrelated to nadir noise (root
    cause is the separate per-bin-median-pinned-flat / recorded-window
    mechanisms documented above), unaffected by this fix, correctly
    unchanged.
  - Tests: `TestStage2Nadir` renamed/updated
    (`test_altitude_sample_matches_geometry`,
    `test_altitude_and_bottom_boundaries_diverge_generates_note`),
    `TestStage2_5BeamGeometry` rewritten around near/far rays
    (`test_near_and_far_ray_match_known_geometry`,
    `test_far_ray_angle_at_45_degrees_when_usable_range_equals_altitude`,
    `test_far_ray_none_when_usable_range_none`,
    `test_classification_determined_even_at_near_zero_angle` — locks in
    the dropped near-grazing/water-column dichotomy), new
    `TestDetectNadirNoiseExtent` (4 tests) and `TestBlankNadirNoise` (3
    tests), `TestMaskNadirGap` (renamed from `TestMaskWeakZone`). Full
    suite: 1200 passing / 36 skipped.
  - The 7-survey battle-test comparison artifact (same URL as above)
    republished in place with corrected terminology and the re-run
    findings. The original `survey-a`-only report's underlying
    data was regenerated (confirms numbers unchanged: `usable_range_m`
    36.0/46.0, `absorption_only`/`spreading_only`) but its own HTML
    prose was **not** rebuilt this session — it still uses the old
    "geometric bottom contact"/"weak zone" language if read directly;
    treat the battle-test report as the current source for terminology.
- ✅ **A second real bug, caught by the user reviewing the report above,
  same day: `detect_first_bottom_return` had no floor tied to the one
  absolute physical law in this whole problem — the real first bottom
  return can never land closer than the vehicle's own altitude.** Even
  after `blank_nadir_noise`, the detector could (and on real
  `project-b` data, did) return a boundary *before* altitude —
  confirmed directly: sample 36, which is *less than* the nadir-noise
  burst's own detected extent (43) and far short of the altitude
  sample (~451) on a real site-c/900kHz ping — a physically
  impossible chain (bottom before noise-burst-end before altitude).
  User's diagnosis was precise and directly falsifiable: "the actual
  first bottom return is close to sample 1000, at its closest in a
  small spot close to 500" — matched almost exactly once fixed
  (median 891, p10 599).
  - **Fix**: `detect_first_bottom_return` gained a `min_sample`
    parameter (per-(ping,channel) or scalar) — samples before it are
    excluded from both the peak reference *and* the sustained-rise
    search, not just clamped after the fact. `stage2_nadir` now always
    passes its own `altitude_sample` here. `mask_nadir_gap` (which
    recomputes the boundary independently, not via `stage2_nadir`)
    gained the same floor via a new `range_var` parameter, computing
    its own altitude sample the same way — same bug existed there
    too, same fix applied.
  - **Real-data re-run, all 14 frequency-channels**: JSF numbers
    unchanged (never affected — EdgeTech's noise is already zeroed, so
    the detector was never finding a pre-altitude spike there). Klein
    SDF: `beam_geometry` outcomes now vary genuinely by channel rather
    than uniformly — site-c/900kHz `determined` at 59° (matches the
    hand-verified case above), site-d/900kHz `determined` at
    11°, most 455kHz channels `ambiguous` (a small 2-4° median angle
    but high ping-to-ping std — first bottom return often lands right
    at the altitude floor rather than consistently further out).
    **Left as an open question, not resolved**: whether that reflects
    a real near-vertical beam edge on 455kHz specifically, or a
    detector still not fully robust to this hardware's near-floor
    noise on some channels — the classification system is doing its
    job either way (reporting low confidence instead of a
    clean-looking but unreliable number), so no further tuning applied
    without more evidence.
  - Tests: `TestDetectFirstBottomReturn` gained 3 tests
    (`test_min_sample_floor_skips_pre_altitude_spike` reproduces the
    exact real-data bug shape, `test_min_sample_scalar_applies_to_every_row`,
    `test_min_sample_nan_row_gets_no_floor`); `TestStage2Nadir` gained
    `test_pre_altitude_spike_does_not_win_as_first_bottom_return`
    (integration-level: confirms `stage2_nadir` actually wires the
    floor through, not just that the underlying detector supports it);
    `TestMaskNadirGap` gained
    `test_pre_altitude_spike_does_not_win_the_boundary` (same check for
    the independent `mask_nadir_gap` codepath). Full suite: 1205
    passing / 36 skipped.
  - The battle-test comparison artifact (same URL as above) republished
    again with the corrected numbers and this fix's writeup; the
    original `survey-a`-only report was not touched (still
    not rebuilt since the terminology-correction commit above — its
    numbers are unaffected by this fix since it's JSF-only).
- ✅ **`smooth_boundary_across_pings()` — new, same day, user feedback
  after the altitude-floor fix: "generally correct but shows massive
  jumps that are unexplained. Smoothing that line might help."**
  `detect_first_bottom_return` runs independently per ping — real
  per-ping threshold noise produces occasional large, physically
  implausible jumps between adjacent pings even after both fixes
  above. Confirmed on real `project-b` data (site-c/900kHz, first
  60 pings): a smooth ~1350-sample decline over 60 pings, but with
  single-ping spikes of 80-3466 samples that don't correspond to any
  real terrain change (adjacent pings are meters apart). New function:
  a per-channel NaN-aware rolling **median** across the ping axis
  (`window=15` default) — median rather than mean specifically so it's
  robust to a handful of outlier pings within the window without
  smearing across a genuine multi-ping trend the way a running mean
  would. Wired into `stage2_nadir` (`boundary_smoothing_window=15`
  param, applied to `first_bottom_return_sample` right after
  detection) and independently into `mask_nadir_gap` (same param name,
  since it recomputes its own boundary rather than reusing
  `stage2_nadir`'s). `<= 1` disables it in both.
  - **Real-data confirmation, visual, not just numeric**: regenerated
    `stage2/waterfall.png` for the same site-c/900kHz channel — the
    traced first-bottom-return line now smoothly tracks the visible
    dark/bright terrain boundary the whole way down the survey,
    instead of the previous erratic jumps. A few short flat "shelf"
    plateaus remain around specific ping ranges (a median filter can
    lock onto a repeated value if enough consecutive pings genuinely
    share a detection issue) — not fully eliminated, but a clear,
    visually obvious improvement over the unsmoothed trace.
  - Existing tests were unaffected by turning smoothing on by default
    (1205→1212 passing after new tests) because nearly every existing
    synthetic fixture uses a ping-invariant boundary (same value at
    every ping by construction) — a median filter is a no-op on a
    constant array, so none of those assertions needed updating; only
    genuinely new tests were needed to cover the smoothing behavior
    itself.
  - Tests: `TestSmoothBoundaryAcrossPings` (6 tests — rejects an
    isolated outlier ping, preserves a gradual monotonic trend, even
    window rounds up to odd, `window=1` is an exact identity transform,
    all-NaN window stays NaN, invalid window raises).
    `TestStage2Nadir.test_smooths_isolated_per_ping_detection_outlier`
    (integration-level: confirms `stage2_nadir` wires the smoothing
    through and that disabling it via `boundary_smoothing_window=1`
    reproduces the raw, unsmoothed detection). Full suite: 1212
    passing / 36 skipped.
- ✅ **2026-08-24 — three real findings from user review of the two new
  full investigation reports (`unknown`, site-c), all confirmed
  directly on real data. No pipeline code changed — all three are
  investigation-methodology/report-narrative corrections, not bugs in
  `sidescanpy` itself.**
  - **Real methodology gap in Stage 4: RMSE-of-fitted-loss-curve does
    not imply "this correction flattens the real data," when the
    usable-range fit window is narrow relative to `r_ref_m`.** Found on
    `project-a/unknown`'s HF channel (`hz_1495000`): Stage 4 picked
    `absorption_only` by RMSE (0.31dB, beating `full_physics`'s 3.21dB)
    over the 2.5-34m fit window. But binning the *actual corrected*
    `backscatter_intensity` by range within that same window tells a
    different story — `absorption_only` declines from median 20.6dB
    (0-3m) to **-4.8dB by 30-33m** (a ~25dB fade entirely inside the
    "usable" window); `full_physics` over the same span declines only
    29.8dB → 25.1dB (~5dB). Root cause: over a short span (here, r/r_ref
    ≈ 14×), a logarithmic spreading term doesn't curve enough to be
    statistically distinguishable from a linear absorption term, so a
    single-term model with one free scale can fit a mixed-physics
    decay's *shape* well without actually removing the loss. **Lesson
    for future investigations, not yet enforced in code:** treat a
    Stage 4 RMSE win as a hypothesis, and confirm it by applying the
    winning correction to the real data and checking for genuine
    flatness within the fit window — especially when
    `usable_range_m`/`r_ref_m` is small. No change made to
    `fit_hardware_tvg_curve()` itself (RMSE-per-candidate is still a
    reasonable first pass, just not sufficient alone); flagged here and
    in the skill doc rather than silently trusted.
  - **Real, confirmed sensor-quality finding on site-c's port-side
    LF channel — not a positioning artifact as originally guessed in
    the report.** Broken out per side (port=channel 0, starboard=
    channel 1) instead of only looking at each channel's aggregate
    median: HF's own port/starboard first-bottom-return ranges agree
    closely (22.0m / 20.1m, tight std both sides) — the physical sanity
    check ("LF and HF should see roughly the same bottom range") that
    exposed the problem. LF-port's median (26.75m) is in the same
    ballpark as HF, but its cross-ping std (20.1°, vs. HF's 5.7°) is
    enormous — real detection, badly corrupted by intermittent noise.
    LF-starboard's median (10.32m) is essentially identical to the
    altitude sample itself — its detector isn't finding a real bottom
    echo at all, it's pinned at the nadir-gap floor, which is why it
    looks deceptively clean (low std) despite being physically wrong.
    Confirmed visually too: the LF progression figure's port side shows
    repeated horizontal white dropout streaks (tens of pings at a time)
    completely absent from starboard. **Lesson: a smooth/low-std
    per-ping trace is not itself evidence of a correct measurement — it
    can mean a detector is stuck at a floor value.** When a channel's
    aggregate beam-geometry classification comes back `ambiguous`,
    breaking it out per side (and, where available, against another
    frequency's same-ping measurement) can reveal that one side is
    real-but-noisy and the other is smooth-but-wrong, rather than both
    sides being equally uncertain.
  - **Confirmed (not a new finding, a validated one): pings 0-500 of
    the site-c file are a vehicle turn/deployment transient, not
    part of the survey line.** Heading rotates ~59°→237°, altitude
    swings 10-56m, pitch shows a real -40° excursion, all stabilizing
    by ping ~320-500 — confirmed directly on `heading_deg`/
    `altitude_m`/`pitch_deg`. Re-running the full investigation on pings
    500+ only (vs. the full 0: range) moved HF's beam geometry from
    `ambiguous` (std 15.8°) to `determined` at 60° (std 4.9°), and both
    channels' TVG status from `ambiguous` to `spreading_only` removed —
    matching every JSF survey's pattern. `usable_range_m` stayed `None`
    either way (unrelated — see the existing Klein-AGC root cause
    above). **No automatic transit-phase detector exists** —
    `run_investigation()` takes an explicit ping range but nothing
    flags a deployment transient within a file; this one was found and
    excluded by direct inspection of attitude variables, not by any
    pipeline stage. Worth building a detector someday (e.g. a
    heading-rate or altitude-variance check over a sliding ping window)
    if this pattern recurs, but not implemented — out of scope for what
    was asked this round.
  - Both full investigation report artifacts (`unknown`, site-c)
    republished with corrected content: `unknown`'s TVG recommendation
    changed from `absorption_only` to `full_physics` for both channels
    (with the methodology-gap writeup above included in the report
    itself); site-c rebuilt entirely on the pings-500+ segment, with
    a new dedicated port-side-LF section replacing the original
    (incorrect) "survey positioning artifact" guess, and a new
    dedicated ping-500-transit-phase section. The wider 7-survey
    "Pipeline Battle Test" comparison artifact was **not** touched this
    round — it already used its own separately-computed site-c
    numbers from before this correction; revisit if it's used as a
    reference again.

### v0.14.0 follow-up — Stage 4 `exclude_first_n_bins`, prototyped 2026-09-02 (not yet adopted as a default — awaiting team review)

**Status:** Small, additive, backward-compatible parameter added and
tested (`exclude_first_n_bins=0` reproduces every existing caller's
behavior exactly); the *recipe* it enables (mid-swath `r_ref_m` + finer
`bin_width_m` + this new parameter, together) is a proposal being
demonstrated for team review, not yet adopted anywhere as a default.

- ✅ `analysis/investigation.py: stage4_decay()` gained
  `exclude_first_n_bins: int = 0` — drops this many *bins* (not samples)
  from the near end of the already-binned mean-vs-range curve before
  `r_ref_m` auto-selection or fitting ever see them. Motivation: a
  bright leading-edge specular artifact right at the first-bottom-return
  boundary — the opposite-sign case from the already-handled ring-down
  artifact at a ping's very start (which reads anomalously *low* and is
  naturally avoided by `r_ref_m`'s max-mean auto-pick) — can itself win
  that same near-field max-mean search, corrupting the reference and
  every candidate's fit against it. 4 new tests in
  `tests/unit/analysis/test_investigation.py::TestStage4Decay`
  (`exclude_first_n_bins=0` no-op; a synthetic bright-artifact-bin case
  showing the auto-picked `r_ref_m` moves off the artifact and RMSE
  improves once it's excluded; graceful handling when the requested
  exclusion exceeds the available bin count). Full suite: 1275 passing /
  36 skipped (was 1272).
- ✅ `project-a/scripts/debug/compare_tvg_fit_recipes.py` (new, that
  repo) — runs `stage4_decay` twice (today's default recipe vs. the
  proposed one: `r_ref_m` pinned to the middle of the swath instead of
  the near-field auto-pick, `bin_width_m` 2.0→1.0, `exclude_first_n_bins`
  default 2) and saves a 4-panel comparison figure (decay-fit curve +
  residuals, old vs. new) per run.
- **Real-data result, both channels of `survey-a`
  (`20251015170335.ss.h5`, pings 0:5000) — a genuine classification
  change, not just a cleaner-looking number:**
  - HF (`hz_1495000`): OLD is `ambiguous` (`spreading_only` RMSE=4.66dB
    barely beating `full_physics` RMSE=4.75dB — a near-coin-flip). NEW
    is a clean, decisively-separated `none`/`full_physics`
    (RMSE=2.22dB vs. runner-up `absorption_only` at 6.20dB).
  - LF (`hz_547500`): OLD is `spreading_only` (matching this project's
    long-standing documented finding for this exact survey, see the
    v0.13.0 section above) — but NEW flips the winning candidate to
    `full_physics`/`none` (RMSE=0.51dB, a very tight fit, vs. OLD's
    `spreading_only` RMSE=1.27dB). **This is a real disagreement with
    previously-trusted output for this survey, not a refinement of it —
    flagging prominently rather than quietly superseding it.**
  - Visually, in both channels' saved figures, two-to-four real
    near-boundary artifact points sit far above the surrounding trend
    (e.g. HF: ~28dB and ~23dB loss right at the start, vs. neighboring
    points near 0dB) — exactly the kind of artifact this parameter
    targets, clearly visible by eye once plotted, not subtle.
- **Flatness validated same day, per this file's own "RMSE win ≠
  genuine flatness" lesson (2026-08-24 methodology-gap entry above) —
  and it changes the read on both channels, not just confirms it.**
  `project-a/scripts/debug/validate_tvg_fit_flatness.py` (new)
  applies each recipe's winning candidate as a real `TVGParams` via
  `apply_tvg()`, bins the actually-corrected `backscatter_intensity` by
  range within the fit window, and fits a line (slope, dB/m) — the real
  flatness check, not a proxy for it.
  - HF: OLD's `spreading_only` winner, once actually applied, is **not
    flat at all** — a strong -0.446 dB/m slope, 14.12dB peak-to-trough
    across the fit window (a real, visible near-field bump then a
    clean decline, not noise). NEW's `full_physics` winner is
    substantially flatter — -0.227 dB/m, 3.68dB peak-to-trough — real
    improvement, but **not fully flat either**; some real residual
    remains unexplained by either recipe on this channel.
  - LF: OLD's `spreading_only` winner is also not flat once applied
    (-0.118 dB/m, 3.84dB peak-to-trough). NEW's `full_physics` winner
    comes out **essentially flat** (+0.026 dB/m, 0.74dB peak-to-trough)
    — this directly substantiates the classification flip flagged
    above as a real disagreement, not an artifact of the fit
    comparison: the new recipe's answer is the one that actually
    matches physical reality here, on the evidence gathered so far.
  - Same script also renders the raw artifact directly (via
    `sidescanpy.viz.investigation.plot_stage2_nadir_zoom`, no new
    plotting code) rather than only inferring it from the binned decay
    curve: HF shows an unmistakable sharp amplitude spike (raw
    `backscatter_raw` jumping to ~15-20+ from a ~2-5 baseline)
    immediately on both sides of the first-bottom-return boundary,
    visually confirming this is a real, sharp leading-edge-specular-like
    feature, not a statistical binning artifact. LF's version is
    noisier/less singularly spike-shaped in the same zoom (elevated,
    high-variance amplitude right at the boundary rather than one clean
    spike) — same general phenomenon, different visual character by
    channel/format, worth keeping in mind before assuming HF's crisp
    single-spike picture generalizes.
  - Still open: HF's remaining -0.227 dB/m residual after the new
    recipe isn't explained by any of the four candidate models tried —
    consistent with this file's standing "hardware TVG/AGC artifact"
    and "missing angular/grazing correction" open questions (see the
    Known Issues section), not resolved by this exercise.
- Not yet run against any Klein SDF (`project-b`) survey — this
  project's nadir/first-bottom-return handling differs substantially by
  format (see the "Battle-tested across 6 more real surveys" entry
  above), so this finding shouldn't be assumed to generalize there
  without checking.

### v0.14.0 follow-up 2 — two-sided fitting tried, tested properly, and rejected (2026-09-02, same day)

**Status:** `fit_hardware_tvg_curve()`/`stage4_decay()` gained a real,
tested, backward-compatible `r_fit_min_m` parameter (fits both sides of
`r_ref_m` instead of only outward from it — needed once `r_ref_m` is
pinned mid-swath, or the old one-sided window throws away the entire
near half of the usable range). **The recipe it enables did not
outperform the simpler one-sided mid-swath recipe from the entry above
on real data, after two rounds of honest testing** — kept as a tested
capability, not adopted as part of the recommended recipe.

- Team direction: "reference range out far is good, but then we should
  fit both sides — just throw out the values close to first bottom
  return." Implemented `r_fit_min_m` (default `None` = prior one-sided
  behavior, matches every existing caller exactly; `TVGCandidateFit`
  gained the same field, `None`-safe in both `plot_candidate_fit`/
  `plot_stage4_residuals`). 4 new tests
  (`tests/unit/analysis/test_tvg_diagnostics.py`,
  `tests/unit/analysis/test_investigation.py`). Full suite: 1279
  passing / 36 skipped (was 1275).
- **First real-data attempt (small exclusion, 2 bins) was much worse on
  both channels** — HF RMSE 4.66→6.98dB, peak-to-trough 3.7→31.3dB; LF
  RMSE 1.27→11.37dB (classification collapsed to `ambiguous`/`flat`,
  implausible), peak-to-trough 0.7→42.7dB. Root cause, found by
  measuring the *raw* (uncorrected) binned curve directly: there's a
  **real, broad near-field ramp already present in raw backscatter**,
  not just a narrow artifact — HF peaks around 4-5m past the boundary,
  LF around 9-10m, before declining into the normal decay. 2 bins
  (1-2m) doesn't come close to clearing it. Consistent with this
  file's own beam-geometry near-field findings elsewhere (grazing angle
  stays shallow for a real physical distance past the boundary) — this
  is evidence for that same mechanism, not a new one.
- **Second attempt (data-driven exclusion, past the raw peak: 5 bins
  HF, 10 bins LF) still did not beat the one-sided mid-swath recipe.**
  HF: RMSE 2.15dB (comparable to the one-sided recipe's 2.22dB) but
  peak-to-trough 4.80dB (worse than 3.68dB). LF: RMSE 1.96dB and
  peak-to-trough 8.87dB, both clearly worse than the one-sided recipe's
  0.51dB/0.74dB. Visually, the corrected curve's near-field peak sits
  further out (~10-11m on HF) than the *raw* curve's peak (~4-5m) —
  TVG's gain term, which grows with range, partially offsets the raw
  decline past its own peak, pushing the effective ramp out further
  under correction than a raw-data-only measurement would suggest.
- **A real bug in the validation script itself, caught and fixed
  mid-investigation, worth remembering:** the first flatness check
  after adding `exclude_first_n_bins` support scored the corrected data
  over a *wider* window than the fit actually used — conflating two
  independent filters (`exclude_first_n_bins` trims rows out of
  `binned` before the fit ever runs; `r_fit_min_m`/`r_ref_m` mask which
  of the *remaining* rows participate in the fit, without removing any
  rows). Scoring flatness over the wider, untrimmed window blamed the
  correction for a near-field region it was never asked to explain,
  and initially made the two-sided recipe look catastrophically worse
  than it actually was. Fixed in `project-a/scripts/debug/
  compare_tvg_fit_recipes.py`'s `_actual_fit_near_edge_m()` (takes the
  max of both filters). **General lesson: when validating a fit by
  reapplying it to real data, the validation window must exactly match
  what the fit itself saw — recomputing it independently is an easy
  way to silently test the wrong thing.**
- **Current recommendation, unchanged from the entry above:** one-sided
  fit outward from a mid-swath-pinned `r_ref_m`, finer bin width, small
  `exclude_first_n_bins` — no `r_fit_min_m` override. Two-sided fitting
  remains available and tested if a better near-field characterization
  ever justifies revisiting it, but isn't paying off with the
  straightforward approaches tried so far.

### v0.14.0 follow-up 3 — Stage 3/4's `range_var` default doesn't match `apply_tvg()`'s own convention (found 2026-09-02, same day)

**Status:** Real, confirmed, pre-existing inconsistency — not introduced
by this session's work, and not fixed here (no default changed; see
below for why). Flagged by a team member reviewing the fit plots
("Are you using ground range for the decay fit? It should be slant
range!").

- **The mismatch:** `stage3_usable_range()`/`stage4_decay()` both
  default to `range_var="ground_range_m"`. But `apply_tvg()` — the
  function that actually *applies* the correction these diagnostics are
  choosing a recipe for — computes its own gain curve directly against
  **slant range**, independently, inside itself (`radiometric.py`:
  `slant_range = np.arange(n_samples) * sample_interval_m`, never reads
  `ground_range_m` at all). So the diagnostic that picks which
  `TVGParams` to recommend has been fitting a different range variable
  than the one the real correction physically uses. No documented
  rationale found anywhere for the `ground_range_m` default on either
  Stage 3 or Stage 4 — looks like an unexamined choice (a reasonable
  one for georeferencing/gridding contexts, where `ground_range_m` is
  the right variable, just not for a physics-based loss-vs-range fit).
- **Real-data check, both channels of `survey-a`, the
  recommended mid-swath recipe specifically: negligible effect.**
  Switching `range_var` from `ground_range_m` to `slant_range_m`: HF
  RMSE 2.22→2.30dB, applied peak-to-trough 3.68→3.68dB (unchanged); LF
  RMSE 0.51→0.50dB, peak-to-trough 0.74→0.75dB. Same winning candidate
  both channels either way. **Why it barely moved here:** `ground_range
  = sqrt(slant_range² − altitude²)`, and this survey's altitude is only
  ~2m — at `r_ref_m` (18–23m, deliberately mid-swath), `ground_range`
  and `slant_range` have already converged to within centimeters.
- **Not a general "this never matters" conclusion — the opposite: a
  survey flown higher off bottom, or an r_ref pinned closer to nadir
  (e.g. the OLD near-field auto-pick, or any near-field analysis), would
  see this diverge far more** — ground_range→0 right at the boundary
  while slant_range→altitude there, a large *relative* mismatch exactly
  in the region every TVG-fit investigation this project has done cares
  most about getting right.
- **No default changed in `stage3_usable_range`/`stage4_decay`** —
  both already accept `range_var` as a parameter; changing the default
  would silently alter every existing caller's behavior across every
  consumer project, the same category of decision this file's own
  convention treats as needing explicit sign-off, not a same-day
  unilateral fix for what turned out to be a low-impact case on this
  one survey. `project-a/scripts/debug/compare_tvg_fit_recipes.py`
  now defaults to `--range-var slant_range_m` (the physically-correct
  choice, matching `apply_tvg()`) with `ground_range_m` still available
  via the flag for direct comparison.
- **Open, worth deciding deliberately rather than defaulting past:**
  whether `stage3_usable_range`/`stage4_decay`'s own default should
  change to `slant_range_m` project-wide — this session found one
  low-altitude survey where it doesn't matter, not evidence it never
  will.

### v0.14.0 follow-up 4 — the recipe made official, RMSE trusted directly (2026-09-02, same day)

**Status:** Settled, for real use — not another comparison round. The
fit-window widening explored in follow-up 2 (three rounds, all
rejected against a flatness check) turned out to just need a better
near edge: auto-detected from the raw data itself, with margin, instead
of a fixed bin count or the raw peak alone. `project-a/scripts/debug/
plot_official_tvg_fit.py` (new, replaces `compare_tvg_fit_recipes.py`/
`validate_tvg_fit_flatness.py` as the tool actually used going forward
— both kept, not deleted, as the historical record of how this was
found) renders exactly this recipe and nothing else: no old-vs-new, no
paired flatness check.

- **The recipe:** `range_var="slant_range_m"`; `r_ref_m = usable_range_m
  / 2`; fit window near edge = raw backscatter's own near-field peak bin
  (auto-detected, same search-and-bin approach `r_ref_m`'s own auto-pick
  uses) plus a 5m margin; far edge = `usable_range_m`, unchanged. Decided
  by RMSE (fixed-scale) directly — no separate flatness pass.
- **Why RMSE alone is trusted now, when it wasn't in follow-up 2:** once
  the window was repositioned this way (auto-margin past the *measured*
  near-field peak, not a guessed fixed cutoff), RMSE improving stopped
  disagreeing with what a flatness check would have said — the two
  follow-up-2 rounds that trusted a narrower/differently-placed window
  were exactly the cases where they diverged. This is a real,
  user-directed decision to stop pairing the two, not a claim that RMSE
  was always sufficient on its own.
- **Real-data numbers, `survey-a`, both channels:** HF window
  [11.0, 36.0]m, r_ref=18.0m, `full_physics` wins at RMSE=1.97dB
  (runner-up `absorption_only` 5.50dB). LF window [15.0, 47.0]m,
  r_ref=23.5m, `full_physics` wins at RMSE=0.59dB (runner-up
  `spreading_only` 2.01dB) — the widest separation of any recipe tried
  this session, on either metric.
- **Plot changes, all requested directly:** decay-fit and residuals
  panels now share one x-axis (`plot_stage4_residuals(...,
  relative_x=False)`, stacked with matplotlib `sharex=True`) labeled
  "Slant range (m)" explicitly rather than the generic "Range (m)" both
  functions default to. Y-axis limits on the fit panel are computed from
  the fit window's own observed values (not the full unfiltered curve),
  so a near-boundary outlier bin — the first-bottom-return artifact this
  whole recipe exists to route around — can no longer compress the
  actual fit curves into an unreadable sliver by stretching the axis to
  include it.
- **Explicitly not portable as literal numbers:** the recipe (method) is
  intended to transfer to other surveys/formats, including vendor-format data
  on the field — the specific numbers above (`r_ref_m`, the near
  edge) are this one JSF survey's own measurements and must be
  re-derived fresh anywhere else, via the same auto-detection, not
  copied.

### v0.14.0 follow-up 5 — `stage3_usable_range`/`stage4_decay`'s own `range_var` default changed to `slant_range_m` (2026-09-02, same day)

**Status:** Done — the "open decision" flagged in follow-up 3 above is
now resolved, at the user's direction ("Both should be using slant
range"). Both functions' `range_var` parameter now defaults to
`slant_range_m`, matching `apply_tvg()`'s own convention, everywhere —
not just in `project-a`'s `plot_official_tvg_fit.py`, which already
passed it explicitly and is unaffected by this change.

- **A second, more consequential bug found while making the change, not
  just the default flip itself:** both functions' water-column exclusion
  was keyed to `range_var`'s own NaN pattern (`ground_range_m` is NaN
  in the water column; `slant_range_m` — pure time-of-flight — is finite
  *everywhere*, including the water column). Flipping the default alone
  would have silently stopped excluding the water column from Stage
  3/4's binning and fitting entirely for every caller that doesn't pass
  its own `exclude_before_sample`. **Fixed properly, not just avoided:**
  water-column exclusion is now always computed from `ground_range_m`
  specifically (falls back to `range_var`'s own finiteness only if
  `ground_range_m` isn't present in `ds` at all), decoupled from
  whichever variable `range_var` names for the fit's own x-axis. Both
  functions' metrics now also record `"range_var"` directly, for
  traceability.
- **Real-data check: zero effect on `plot_official_tvg_fit.py`'s already-
  reported numbers.** Re-ran both channels of `survey-a` after
  this fix — RMSE, classification, and the auto-detected fit window are
  byte-identical to before (HF 1.965dB/[11.0,36.0]m, LF 0.593dB/
  [15.0,47.0]m). Expected: this survey's ~2m altitude is small enough
  relative to the auto-detected near edges (11m/15m) that the water
  column was never close to either fit window regardless. **Not
  evidence this bug was harmless in general** — a survey with a deeper
  water column relative to its fit window (plausible on the field,
  different platform/altitude) could have seen real contamination; this
  is exactly the kind of survey-specific difference follow-up 4's "not
  portable as literal numbers" note above already warned about, now
  with a second concrete mechanism behind it.
- **4 new regression tests**
  (`tests/unit/analysis/test_investigation.py::TestWaterColumnExclusionIndependentOfRangeVar`):
  water-column-excluded counts match between `range_var="slant_range_m"`
  and `range_var="ground_range_m"` on the same data (2 tests), and a
  direct behavioral check — a huge garbage value placed only in the
  water column does not appear in either stage's binned output under the
  `slant_range_m` default (2 tests, one per stage). 4 existing tests
  needed updating: 2 asserted the literal string `"ground_range_m"` in a
  missing-variable error message (now `"slant_range_m"`, the new
  default); 2 more (one in this file, one in `tests/unit/viz/
  test_investigation.py`, 4 call sites) had synthetic curves built as an
  *exact* function of `ground_range_m` specifically — pinned those to
  `range_var="ground_range_m"` explicitly rather than re-deriving new
  expected numbers, since none of the four were testing `range_var`
  choice in the first place. Full suite: 1283 passing / 36 skipped (was
  1279).

### v0.14.0 follow-up 6 — `r_fit_min_m` reaches `run_investigation()`, auto-detected near edge ported into the library (2026-09-02, same day)

**Status:** Done. Two changes, both driven by getting the two-sided-fit
recipe (follow-ups 2-5) usable from a `run_investigation()`-based caller
on the field, not just from `project-a`'s standalone debug script:

- **`run_investigation()` now threads `stage4_r_fit_min_m` and
  `stage4_exclude_first_n_bins` through to `stage4_decay()`.** These
  existed on `stage4_decay()` itself since follow-up 1/2 but weren't
  reachable from the orchestrator — every consumer that calls
  `run_investigation()` (not `stage4_decay()` directly) had no way to
  turn on two-sided fitting at all. No test coverage exists for
  `run_investigation()` itself in this repo's `tests/unit/` tree (only
  the individual stage functions and viz functions are unit-tested
  directly) — confirmed via `find`/`grep` before assuming silence meant
  nothing needed updating.
- **`stage4_decay(..., r_fit_min_m="auto")` — a new capability, not just
  a passthrough.** Real motivation: a fixed near-edge number picked by
  eye on one field mission looked fine on the first file, but the
  broader dataset's near-field ramp varies enough mission-to-mission
  that hand-picking one number per mission wasn't going to scale.
  Ported `project-a/scripts/debug/plot_official_tvg_fit.py`'s
  `_auto_near_edge_m` (bins raw `value_var`, finds the near-field peak
  bin's far edge, adds a margin — default 5.0m, same empirical buffer
  that script settled on) directly into `sidescanpy.analysis.investigation`
  as `_auto_near_edge_m` + a new `r_fit_min_auto_margin_m` parameter.
  Pass the literal string `"auto"` for `r_fit_min_m` (on `stage4_decay`
  or `run_investigation`'s `stage4_r_fit_min_m`) instead of a number.
  Needs `r_ref_m` or `usable_range_m` to bound the near-field search
  (falls back to `r_ref_search_fraction * usable_range_m` when `r_ref_m`
  isn't pinned) — raises `ValueError` with an explicit message if
  neither is available, rather than silently picking an arbitrary bound.
  `metrics["r_fit_min_m"]` carries the *resolved* number either way;
  `metrics["r_fit_min_m_override"]` keeps the literal `"auto"` so the
  JSON/reproducibility export still shows which mode produced the run.
- **7 new tests**: `TestStage4Decay` gained 4 (`auto` resolves to a
  numeric value and records the override correctly; raises without
  `r_ref_m`/`usable_range_m`; falls back to `r_ref_search_fraction *
  usable_range_m`; rejects any string other than `"auto"`). New
  `TestAutoNearEdgeM` class gained 3, testing `_auto_near_edge_m`
  directly against a hand-built non-monotonic ramp (distinct from the
  pure-decay fixtures the rest of the file uses) — confirms it finds
  the true peak bin's far edge + margin, respects a `search_max_m`
  narrower than the true peak, and returns `None` cleanly when the
  search window is empty. Full suite: 1290 passing / 36 skipped (was
  1283).
- **Version bumped 0.4.1 → 0.4.2** for this round's wheel (0.4.1 already
  covered the follow-up 5 `range_var`/water-column fix and was
  transferred to the field separately before this `"auto"` need
  came up there).

### v0.14.0 follow-up 7 — `r_ref_m="mid_swath"` + fit-window axis limits ported, fixing a real bug on the field (2026-09-02, same day)

**Status:** Done. Found from a live report after follow-up 6's `"auto"`
near edge actually ran on real vendor-format data: "the reference range is
often outside of the fit window," "not seeing observations from before
the reference range," and a direct question — was the earlier artifact
demo custom, or did it run through the automated code? **Answer: custom
— `project-a/scripts/debug/plot_official_tvg_fit.py` computed
`r_ref_m = usable_range_m / 2` and the axis limits inline; neither had
been ported into the library, so `run_investigation()`-based callers
(downstream-project) couldn't reproduce them.** Two real gaps, not one:

- **`r_ref_m=None` and `r_fit_min_m="auto"` fight each other by
  construction.** `r_ref_m=None` auto-picks the near-field *max-mean
  bin* — the peak itself. `r_fit_min_m="auto"` (follow-up 6) resolves to
  that same peak's far edge *plus a margin*. Left independent, `r_ref_m`
  (at the peak) nearly always lands *nearer* than the auto-detected fit
  window's near edge — i.e. the reference sits outside the window it's
  supposedly the reference for. This is exactly the first bug report.
  **Fixed:** `stage4_decay`'s `r_ref_m` now also accepts the literal
  string `"mid_swath"`, resolving to `usable_range_m / 2` (the recipe's
  own choice) — pinned well past the near-field peak on any normal
  survey, so it can't collide with the auto-detected near edge.
  `r_ref_m="mid_swath"` + `r_fit_min_m="auto"` is now the documented
  *required* pairing, not "pick one or leave both auto" as follow-up 6's
  own guidance had carelessly implied.
- **The axis-limit/zoom logic lived only in the standalone script, never
  in the library.** `plot_official_tvg_fit.py` computes robust y-limits
  from the fit window's own observed values (not the full curve — an
  out-of-window outlier can't compress the real fit into a sliver) and
  x-limits zoomed to `[r_fit_min_m, r_fit_max_m]` plus padding.
  `downstream-project`'s `_build_decay_fit_and_residuals` reused the same
  `plot_candidate_fit`/`plot_stage4_residuals` calls but left
  matplotlib's default full-data autoscale in place — which, combined
  with the `r_ref_m` bug above, is what produced the second report ("not
  seeing observations before the reference"): the reference sat so close
  to the near edge that there was barely any span between them to render
  meaningfully, and no window-relative zoom to make that legible.
  **Fixed:** new `viz.investigation.apply_stage4_fit_window_limits(ax,
  stage4)` — call once on the top (fit) panel of a `sharex=True` stack;
  reads `stage4.metrics["r_fit_min_m"]`/`["r_fit_max_m"]` directly, so
  any caller using the standard `stage4_decay` → `plot_candidate_fit`
  pipeline gets the same zoom without recomputing it by hand.
- **7 new tests**: `TestStage4Decay` gained 4 for `r_ref_m="mid_swath"`
  (resolves to `usable_range_m/2`; raises without `usable_range_m`;
  rejects any string other than `"mid_swath"`; a direct regression
  reproducing the bug — `r_ref_m=None` + `r_fit_min_m="auto"` puts the
  reference outside the window, `r_ref_m="mid_swath"` paired with it
  does not). New `TestApplyStage4FitWindowLimits` (viz) gained 3: no-op
  without a fit result; zooms to the fit window with the documented
  padding; an out-of-window outlier injected directly into
  `observed_loss_db` does not affect the computed y-limits. Full suite:
  1297 passing / 36 skipped (was 1294).
- **Version bumped 0.4.2 → 0.4.3.**

### v0.14.0 follow-up 8 — `r_ref_m="mid_swath"` wasn't sufficient; auto-detected near edge clamped, x-limits stopped cropping (2026-09-02, same day)

**Status:** Done. Follow-up 7's `r_ref_m="mid_swath"` fix reduced but did
not eliminate the "reference outside the fit window" bug — a second,
independent live report ("r_ref is often outside of the fit region and
it is not showing the full plot just the region around the fit") traced
to two more real gaps:

- **The near-edge auto-detect's search bound doesn't bound its output.**
  `r_fit_min_m="auto"`'s peak search restricts by *bin center* <=
  `r_ref_m`, but the value it returns is that peak bin's *far edge plus
  the margin* — on data whose true near-field peak sits close to the
  search boundary (real on this dataset: the near-field ramp varies
  enough mission-to-mission that it isn't reliably confined inside the
  near half of the swath), the returned value can still land past
  `r_ref_m` even though the search itself never looked beyond it.
  Pinning `r_ref_m="mid_swath"` narrowed how often this triggers but
  can't prevent it outright — mid-swath isn't guaranteed to be past
  wherever the real peak happens to sit on a given mission. **Fixed:**
  the auto-detected value is now clamped down to `r_ref_m` whenever it
  would exceed it (a locally one-sided fit for just that mission/
  section, not a crash), with a note recording when the clamp fired —
  `metrics["r_fit_min_m"] <= metrics["r_ref_m"]` is now a hard guarantee
  whenever `r_ref_m` is known, not just the common case.
- **The ported axis-zoom (follow-up 7) cropped to the fit window
  itself — the wrong call.** Matching `plot_official_tvg_fit.py`'s tight
  zoom meant that whenever the reference *did* still land outside the
  window (the bug above), the chart cropped that mismatch away instead
  of showing it — "not showing the full plot" was a direct symptom of
  the first bug, not a separate complaint. **Fixed:**
  `apply_stage4_fit_window_limits`'s x-limits no longer crop to
  `[r_fit_min_m, r_fit_max_m]` — they now always start at 0.0 and extend
  through the far edge (padded), showing the full observed curve. Only
  the y-limits stay computed from the fit window's own in-window values
  (the one piece of the original zoom actually worth keeping — protects
  against a near-boundary outlier compressing the real fit curves). A
  wrongly-placed reference or fit window is now visible on the chart
  instead of silently cropped out of frame.
- **1 new test** in `TestStage4Decay` (the clamp, using a hand-built
  ramp dataset whose peak sits deliberately close to `r_ref_m` — distinct
  from `_make_full_physics_ds`'s monotonic decay, whose peak always
  sits at range=0 and never triggers this path). `TestApplyStage4FitWindowLimits`'s
  zoom test rewritten for the new always-starts-at-0 contract instead of
  asserting the old crop-to-window behavior. Full suite: 1299 passing /
  36 skipped (was 1298; net +1 since one test was rewritten, not added).
- **Version bumped 0.4.4 → 0.4.5.**

### v0.14.0 follow-up 9 — `mid_swath`/`auto` made resilient enough to be a default, not just an opt-in override (2026-09-02, same day)

**Status:** Done. Triggered by `downstream-project` deciding to make
`r_ref_m="mid_swath"` + `r_fit_min_m="auto"` its own **default** config
(so missions don't need per-config overrides to get the validated
recipe) — running that against its synthetic test suite immediately
surfaced a real gap: `stage4_decay: r_ref_m='mid_swath' needs
usable_range_m -- pass it, or an explicit numeric r_ref_m.` on a
perfectly ordinary synthetic mission where Stage 3 simply couldn't
determine `usable_range_m`. 14 of `downstream-project`'s own tests failed the
same way once its `MissionParams` defaults flipped.

**The design mistake:** follow-up 7/8 built `"mid_swath"`/`"auto"` as
*hard requirements* — raise `ValueError` if the inputs they need aren't
available. That's a reasonable design for an explicit opt-in override
(a caller setting it by hand presumably knows their data supports it),
but wrong for something meant to be a blanket default across every
mission/section in a survey — some of those will always have a
genuinely undeterminable `usable_range_m` (short, noisy, or otherwise
difficult data), and raising there takes down the *entire* analyze run
over one difficult section instead of just that section's Stage 4
result.

**Fixed:** both now degrade gracefully instead of raising, matching
every other auto-adaptation already in this function (e.g. `min_bins`'s
`bin_width_m` shrink) — a note explaining the fallback, never a hard
failure over a missing optional input:
- `r_ref_m="mid_swath"` without `usable_range_m` → falls back to the
  prior near-field auto-pick (`r_ref_m=None`).
- `r_fit_min_m="auto"` without enough to bound its search (or with no
  bins in that search window) → falls back to the prior one-sided fit
  window (`r_fit_min_m=None`, starts at `r_ref_m`).
- Invalid string values (anything other than `"mid_swath"`/`"auto"`)
  still raise — that's a caller bug (a typo), not a data limitation, and
  should still fail loudly.
- **2 existing tests rewritten** (they asserted the old raise-on-missing-
  input behavior) to assert the new fallback + note instead. No new
  tests needed beyond that — the fallback paths reuse the same resolved-
  vs-`None` branches the clamp logic (follow-up 8) already exercises.
  Full suite: 1299 passing / 36 skipped (unchanged count — 2 rewritten,
  0 net new).
- **Version bumped 0.4.5 → 0.4.6.**

### v0.14.0 follow-up 10 — Stage 3/4 gained a `channel` parameter; they were silently pooling port and starboard (2026-09-02, same day)

**Status:** Done. Found while investigating a live report ("I do not
see multiple channels in the TVG fit charts") that turned out to name
the wrong axis at first — `downstream-project`'s `analyze` step already loops
over every *frequency* channel (`freq_key`, e.g. HF/LF) and produces
separate output per one. The actual gap: `mean_backscatter_vs_range`
(Stage 3/4's shared binning helper) `.ravel()`s **every** array
dimension before binning, including the *physical* `channel` dim
(port/starboard) — `run_investigation()`'s own docstring even said "one
frequency channel," using "channel" ambiguously for both meanings. Every
`stage3_usable_range`/`stage4_decay` call ever made (including every
number reported earlier this session) silently blended port and
starboard into one fitted curve, with nothing in the output signaling a
blend happened.

- **`stage3_usable_range`/`stage4_decay` gained `channel: int | str |
  None = None`** — `0`/`1` by position or `"port"`/`"starboard"` by
  `ds["channel_id"]`. `None` (default) is byte-identical to every
  existing caller's prior behavior. **Masks samples, does not slice
  `ds`** — the sidescan schema hard-requires `channel` be exactly size 2
  everywhere in the pipeline (`SideScanAccessor.validate()`, called
  internally by `compute_geometric_ranges`/`compute_georeferencing`),
  and `stage2_5_beam_geometry` separately hardcodes a `(0, "port"), (1,
  "starboard")` iteration regardless of `validate()` — an earlier
  attempt at slicing `ds` itself down to one channel hit both of these
  (`ValueError: 'channel' dimension must be size 2` from the first,
  `IndexError` from the second) before landing on the masking approach,
  which leaves `ds`'s shape untouched everywhere and only narrows the
  working array these two functions bin from. New shared
  `_resolve_channel_idx()` helper does the int/string resolution once.
- **`run_investigation()` gained the same `channel` parameter**,
  threaded through to both `stage3_usable_range`/`stage4_decay` calls
  only — **Stages 1, 1.5, 2, 2.5, and 5 are deliberately NOT
  restricted**, they keep seeing the full pooled `ds` regardless (those
  stages already report both channels' own geometry independently where
  it matters, e.g. `stage2_5_beam_geometry`'s `near_ray`/`far_ray`; only
  Stage 3/4's flat-ravel binning actually pools). `params["channel"]`
  records the requested value for reproducibility. Call once per
  physical channel (typically both) for a genuine per-channel Stage 3/4
  comparison.
- **16 new tests**: `TestStage3UsableRange`/`TestStage4Decay` each
  gained 5 (`channel=None` matches omitting it; restricting to one
  channel produces a measurably different, independently-correct
  result; string lookup matches int position; invalid string raises;
  pooled differs from either single channel), using two-channel
  fixtures whose sides are deliberately built to classify/measure
  differently (so a passing "differs" assertion is a real proof, not a
  coincidental pass). New top-level `tests/unit/test_investigation.py`
  (this repo's first direct test coverage of `run_investigation()`
  itself — confirmed via search that none existed before) gained 5 more
  at the orchestration level, round-tripping a real synthetic `.ss.h5`
  file through `write_survey`/`run_investigation` rather than calling
  the stage functions directly. Full suite: 1313 passing / 36 skipped
  (was 1299).
- **Version bumped 0.4.6 → 0.4.7.**

### v0.14.0 follow-up 11 — r_ref_m resolution order reversed: window first, then reference inside it (2026-09-02, same day)

**Status:** Done. Direct, sharp user pushback on the follow-up 10
channel-split artifact, after being told the frequency-channel question
still hadn't actually been answered: "you chose the reference and it
[is] at the start of the decay curve fit window, but the value of the
reference is clearly an outlier. All of the fits are terrible which
makes your claim that the results are different ridiculous... Maybe the
right way to do this is to generate the fit window first and then
select a value 1/3 from the start." Correct on both counts.

- **Root cause, finally addressed properly instead of patched:**
  follow-up 7/8's `r_ref_m="mid_swath"` design picks the reference
  *independently* of the fit window (`usable_range_m / 2`), then
  follow-up 8's clamp forces `r_fit_min_m` down to meet it when they
  collide — which lands the reference EXACTLY on the window's own
  boundary. That boundary is precisely the near-field transition zone
  the window exists to exclude, i.e. usually a noisy bin. Since
  `observed_loss_db = means[i_ref] - means`, a bad reference corrupts
  *every* bin's number, not just its own — every candidate's fit looks
  equally bad, and RMSE "differences" between runs (or channels) become
  reference-selection noise, not a physics signal. The follow-up-10
  artifact's port/starboard comparison reproduced exactly this, and the
  user caught it directly from the chart.
- **The actual fix (not another clamp):** resolution order reversed.
  `r_fit_min_m` (the window's near edge) now resolves FIRST — its own
  `"auto"` peak-plus-margin search bound can no longer come from a
  pending `r_ref_m="auto"`/`"mid_swath"` string (falls through to
  `r_ref_search_fraction * usable_range_m` instead; an already-numeric
  `r_ref_m` still bounds it, unchanged). `r_ref_m="auto"` — new,
  replaces `"mid_swath"` as the recommended default — is resolved
  SECOND, placed `r_ref_window_fraction` (new parameter, default 1/3,
  the user's own suggested split) of the way from the now-known near
  edge to the far edge (`usable_range_m`). Guaranteed strictly inside
  `[r_fit_min_m, r_fit_max_m]` by construction for any fraction in
  `(0, 1)` — not just usually, and the old clamp can never fire for
  this pairing. `"mid_swath"` stays available (legacy, explicit/
  backward-compat use only) with its original clamp-based safety net
  unchanged, since it still doesn't derive the reference from the
  window.
- **4 new tests**: the same pathological-ramp fixture that proved the
  clamp fires (follow-up 8) now proves `r_ref_m="auto"` avoids ever
  needing it — reference lands strictly inside the window, not on
  either boundary. Plus `r_ref_window_fraction` customization, and the
  graceful-fallback-without-usable_range_m case. 1 existing test's error
  message updated (`"must be 'mid_swath'"` → `"must be 'auto' or
  'mid_swath'"`). Full suite: 1316 passing / 36 skipped (was 1313).
- **Also directly answered** (separately, not a code change): the
  user's original, still-unanswered question about HF/LF frequency-
  channel handling — every step processes each `freq_key` fully
  independently, including frequency-specific absorption physics via
  `absorption_for_frequency(freq_hz)`; only `spreading_loss_db` is one
  survey-wide setting by design (a hardware property, not a frequency-
  dependent physics constant). The port/starboard channel-split work
  (follow-up 10) was answering a *different* question than the one
  originally asked — a real misread, called out directly by the user.
- **Version bumped 0.4.7 → 0.4.8.**

### v0.15.0 — CLI subpackage consolidating the consumer-project numbered scripts (2026-09-16)

**Status:** Complete. Full writeup in `CHANGELOG.md`'s "Unreleased ->
Added -> CLI subpackage" entry — this note just points at it and records
what changed outside `sidescanpy`'s own source tree, per this file's own
"Session Handoff" convention.

- New `sidescanpy.cli` subpackage (`convert.py`/`correct.py`/
  `export.py`/`analyze.py`/`_common.py`/`_config.py`), four
  console-script entry points added to `pyproject.toml`
  (`sidescan-convert`/`sidescan-correct`/`sidescan-export`/
  `sidescan-analyze`). Replaces the numbered pipeline scripts that were
  duplicated near-identically across `project-b`'s and `project-a`'s
  own `scripts/` directories (9 of 13 confirmed byte-identical except
  docstring text) — the exact category of drift that already caused a
  real bug (`project-b/04_radiometric.py` missing a feature) is now
  structurally impossible, since there's only one copy of this code.
- Two small, genuinely general fixes landed on existing modules as part
  of this: `io/sdf/targets.py` gained `calibrate_clock_offset()`/
  `resolve_targets_offset_corrected()`/`resolve_targets_nearest_ping()`
  (promoted from an `project-b`-local script — Klein-acquisition-
  general, not project-specific), and `viz/targets.py` gained
  `plot_target_position_scatter()`/`export_target_position_scatter_png()`
  (promoted from a near-duplicated block in both formats' own
  target-catalog-reading stage).
- Every new command verified against real survey data (both JSF and
  Klein SDF) with byte-identical (or, for `tvg-fit`, behaviorally
  equivalent — it intentionally uses the more current `r_ref_m="auto"`
  library recipe rather than porting an older manual script verbatim)
  output vs. the scripts it replaces, before any old script was deleted.
- `project-b`/`project-a` had their numbered scripts (plus
  `project-a`'s `14_export_dataset.py`/`investigate_dataset.py`)
  deleted and replaced with one `datasets/<survey>.json` `--config` file
  each (real, cited settings — not placeholders) — see those projects'
  own `README.md`/`CLAUDE.md` for their side of this.
- Version bumped 0.4.10 (pyproject.toml) for this round.
- Full suite: 1428 passing / 36 skipped (was 1373 before this work).

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

- ~~**TVG diagnostics**~~ — Stage 1 complete 2026-07-16 (see "v0.8.0 — TVG Diagnostics, Stage 1" above), sound-velocity fix applied and residual investigation deepened same day (see "v0.8.0 follow-up" above). Empirical range correction (the proposed next step) complete 2026-07-22 (see "v0.9.0 — Empirical Range Correction" above) — not Stage 2's original beam-pattern framing (evidence didn't cleanly support that framing either — see the follow-up section). Selectable TVG terms + hardware-curve-fitting tooling (a different, complementary angle on the same open question) complete 2026-08-13 (see "v0.13.0" below).
- **Klein INI parsing** — unblocks proper `depth_m` for SDF (~1 session; see Known Issues #5)
- **Bottom detection** (`processing/detect_bottom.py`) — fallback altitude
- **Per-side diagnostic mode** — port/starboard bands in gridded raster
- **Slider performance fix** in `plot_sonar_ping.py`
- **XTF support** — industry-standard format
- **Radiometric pipeline cleanup** — address TODOs before quantitative work
- **Test fixture solution** — synthetic or small real fixtures in repo
- **Lazy/dask processing verification** — confirm pipeline works chunked
- **Detection-based target export** (old v0.7.0 "7b" sketch: threshold/local-max/anomaly detection on grids, auto-export candidates) — folded into v0.7.0 Priority 3, deferred
- **Local/tile-based contrast normalization for large waterfall/gridded views** — a same-day fix (2026-07-30, floor-clip NaN masking + tightened percentile in `visualization_demo.py`) narrowed but didn't close the gap between how punchy target crops (`viz/contact_sheet.py`) look vs. big multi-segment exports. See `docs/planning/v0.11.0-local-contrast-normalization.md` for the tile-based approach (deferred: tile-seam risk, needs real visual QA) and a related deferred idea (upstream floor-clip masking so gridding's cell averages aren't pulled down either — gated on confirming Known Issue #5's hypothesis first).
- **Target-crop pipeline module/project split** — same day (2026-07-30) as the local-contrast item above, a separate exercise matched target-crop rendering (`targets/core.py`'s new `orient_for_display`/`detrend_by_range`, plus `apply_tvg_to_amplitude`) against EdgeTech Discover's own per-target images. Full writeup, including exactly what TVG correction is/isn't applied and why `detrend_by_range` is not a substitute for a calibrated TVG, lives in the consumer project: `project-a/docs/pipeline_walkthrough.md`. That doc's recommendations, summarized: `apply_tvg_to_amplitude`/`tvg_gain_db` and `orient_for_display`'s horizontal (port/starboard) flip are genuinely general and belong in the module (already do); `orient_for_display`'s vertical flip is inferred from one vendor's convention on one dataset and should be split into an independently-toggleable option before another consumer project inherits it unquestioned; `detrend_by_range` needs a loud "display-only, not physically grounded" warning wherever it ends up, not a name that reads as a standard step; `orient_for_display`/`detrend_by_range` should be re-exported from `sidescanpy.targets`'s top-level `__init__.py` (currently only reachable via `sidescanpy.targets.core`, inconsistent with `bbox_from_center`); a YOLO export alongside the existing COCO one (`targets/interchange/coco.py`) is anticipated in `targets/__init__.py`'s own docstring but was never built.
- **Anisotropic search radius for target-crop resampling (2026-09-27)** —
  `export/geometric_window.py: extract_geometric_window()`'s KDTree
  averaging neighborhood is a single isotropic radius pinned to the
  coarser of the along-track/across-track native resolutions, so a
  finer `resolution_m` alone can't recover real detail on the finer
  axis (it's averaged away regardless). Real user question, investigated
  and scoped, not implemented — see
  `docs/planning/anisotropic-crop-search-radius.md` for the full
  mechanism, the tiered scope (contained fix for the local along/across
  frame; a harder, separate one for the georef/whole-survey-grid path),
  and why an "interpolate extra pings" alternative was considered and
  rejected for the training-crop use case specifically.
- **`gamma`/`show_colorbar` display params (2026-07-31)** — a same-day follow-up to the item above, on the main waterfall/gridded pipeline rather than target crops. `viz/waterfall.py`'s `plot_waterfall()` and `viz/gridded.py`'s `plot_gridded_raster()`/`export_gridded_png()` gained `gamma: float = 1.0` (power-law display contrast) and `show_colorbar: bool`. `gamma != 1.0` uses `matplotlib.colors.PowerNorm` as the plot's `norm=` rather than pre-transforming the backscatter array to `[0, 1]` — the plotted array and colorbar stay in real units (dB) at any gamma, avoiding a real bug found in an earlier one-off script that pre-normalized the array and then mislabeled its colorbar "dB" for a unitless `[0, 1]` scale. Full writeup (gamma-vs-TVG explanation, gamma sweep findings, why `PowerNorm` and not a manual transform): `project-a/docs/pipeline_walkthrough.md`'s "Gamma correction, precisely" section.

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
   - **2026-07-16 — root cause found and fixed: this residual was substantially explained by `sound_vel_ms` being exactly `0.0` (not NaN) for every ping in the tested real survey, which `apply_tvg()`'s sound-velocity fallback didn't catch (only checked `np.isnan`, unlike `compute_geometric_ranges()`'s correct `np.isnan(...) | (... <= 0)` — Known Issue #7).** This silently pinned every sample's TVG-computed slant range at the `min_range_m = 0.5` floor, making the "range-varying" correction actually constant per ping. **Fix applied same day** (`apply_tvg()`'s check widened to match `geometric.py`'s), changing the residual from a ~37 dB monotonic falloff to a smaller (~6-8 dB peak-to-trough), **non-monotonic, altitude-invariant "bowl"** residual (real-data confirmation, not just simulation, matches last session's predicted numbers almost exactly). Full writeup: `docs/planning/handoff-tvg-sound-velocity-zero-bug.md`.
   - **2026-07-16 — deepened investigation: neither original candidate (mistuned absorption vs. missing angular/grazing correction) cleanly explains the remaining bowl.** Altitude-invariance argues against a grazing-angle effect; a qualitatively different residual shape between `hz_1495000` (bowl) and `hz_547500` (sharp rise then flat plateau), plus an absorption-coefficient sweep confirming no single value flattens the curve (the current 0.20 dB/m default for 1495kHz is already near-locally-optimal among values tried), argues against simple coefficient retuning. Leading hypothesis, unconfirmed: a hardware TVG/AGC artifact baked into the raw signal upstream of `sidescanpy`. No further code change made — see `docs/planning/v0.9.0-empirical-range-correction.md` for the resulting design proposal (awaiting approval, not implemented; several genuine judgment-call questions for the user). Full writeup: `docs/planning/handoff-tvg-residual-post-fix-findings.md`.
5. **Sharp low-end "cliff" in `backscatter_intensity` (~1% of samples pinned at the TVG dB-conversion floor, ~-120 dB) — cause not yet confirmed, needs investigation.** Found 2026-07-02 while investigating why gridded output looked less normalized than the waterfall (see v0.7.0 "gridding correctness pass" above). Evidence gathered so far, on real data (`data/survey-a/20251015170335.ss.h5`, first 500 pings, `hz_1495000`):
   - The floor values come from **exactly-zero raw linear amplitude** in `backscatter_raw` (100% of sampled floor points, not just near-zero/noise) — `apply_tvg()`'s `20*log10(clip(x, 1e-6))` floors these to ≈-120 dB, which is a correct response to genuinely-zero input, not a bug in the TVG conversion itself.
   - These zero-amplitude samples form a **leading run at the start of every ping** — median run length 56 samples (out of 4792), tight across ~500 pings (p10=51, p90=60, std≈3.5) — **not** scattered through the range or concentrated at long range.
   - Run length has **~zero correlation with `altitude_m`** (r≈-0.04) in the sampled pings (though altitude itself barely varied in that window — p10=3.87m, p90=4.07m — so this isn't a strong test of geometry-dependence yet).
   - `ground_range_m` at these points is **not** collapsing toward zero (median ≈36m, close to the non-floor median ≈35m) — i.e. the existing water-column mask (`compute_geometric_ranges`, `slant_range < altitude` ⇒ NaN) does **not** currently treat these as water-column, because the geometric model has no concept of a fixed acquisition-time blanking period — it only compares slant range (sample_index × sample_interval) to altitude.
   - **Net read so far:** looks more consistent with a fixed transmit/receiver blanking or ring-down period (a hardware/acquisition-time artifact, roughly constant in *samples*, not tied to bottom geometry) than with either hypothesis raised when this was found — not the "water column, ground range → 0" pattern (ground range doesn't collapse here), and not a smooth incident-angle falloff either (values are hard exact zeros, not a gradual attenuation curve). **Not confirmed** — would need to check whether the run length is constant in *time* (varies with any sample-rate/range-scale changes across the survey) vs. constant in *samples* regardless of settings, and whether this pattern holds on the SDF/Klein side or is JSF/Edgetech-specific, before treating it as settled.
   - Not yet fixed or worked around anywhere — `resample_channel_to_grid()`/`plot_waterfall()` currently include these samples like any other data point.
   - See `docs/planning/handoff-tvg-range-residual.md` for the full writeup tying this together with the range-dependent TVG residual (Known Issue #4) into one investigation.
6. **Far-range noise amplification on channels whose recorded range window exceeds their real useful range — confirmed 2026-08-06, distinct from #4/#5.** A per-survey acquisition setting (which channel gets the longer recorded window — not fixed by frequency, confirmed by comparing two real surveys with opposite HF/LF asymmetry) can leave a channel's array extending well past where real backscatter decays into noise. On `project-a`'s `unknown` survey, `hz_1495000`'s recorded window (≈165m) extends far past its real signal cutoff (≈50-60m, matching the original ~60m visual observation that started this investigation). Raw amplitude past that cutoff doesn't stay flat — it climbs back up over the remaining ~100m (-30dB at 55-60m → -21dB at 160-165m), new supporting evidence for the "hardware TVG/AGC artifact baked into the raw signal" hypothesis from the Known Issue #4 investigation below. `apply_tvg()`'s gain formula (`tvg_gain_db()`, uncapped, monotonically increasing with range) then compounds this: corrected mean balloons from ~31dB (real signal peak, 5-15m) to ~88dB by 160-165m, a ~57dB overshoot that dominates this codebase's percentile-based color scaling and visually flattens the legitimate near-nadir signal into "one strip of real data, the rest flat." **2026-08-06 fixed** via `processing/farrange.py: crop_far_range_noise()` (candidate fix (a), data-driven mask/crop) — see the "Cross-Project Standard Pipeline Scripts" roadmap entry above for details; the other two candidate fixes (cap the TVG gain formula; fix percentile scaling) remain undone, not needed once the affected samples are cropped out entirely. **Open issue this surfaced, broader than this one finding: we have no reliable way to know whether TVG/AGC was already applied by the sonar hardware or EdgeTech Discover's acquisition software before a `.jsf` file reaches `sidescanpy`** — if so, `apply_tvg()` double-corrects, uncompensated, which alone could produce residuals like Known Issue #4's without any bug in this codebase's math. **2026-08-06 follow-up, partially resolved:** read raw `_gain_factor`/`_user_transmit_level` JSF header fields directly (previously parsed but completely unused) across one file from each of `project-a`'s three surveys. `_user_transmit_level` is constant (100) everywhere — not informative. `_gain_factor` is real and actively varying, strongly anti-correlated with the raw signal's own mean amplitude on *every* survey checked (r = −0.80 to −0.86) — the exact signature of an active per-ping automatic gain control applied before the `.jsf` file is written. This confirms hardware/acquisition-side AGC is genuinely active (not hypothetical), but doesn't by itself prove `apply_tvg()` double-corrects: `_gain_factor` is one scalar per ping (broadband), not a per-sample range-dependent curve, so it can't alone produce the per-sample far-range climb or Known Issue #4's shape. Still unresolved: what `_gain_factor`'s units represent, and whether it's applied to `_raw_trace_data` before storage (making "raw" already gain-compensated) or merely recorded alongside an uncompensated signal. Full writeup: `docs/planning/handoff-hf-farrange-noise-amplification.md`'s "2026-08-06 follow-up" section.

**2026-08-12 follow-up — owner disputed the JSF diagnosis; re-tested and re-confirmed, plus a real SDF-side bug found and fixed.** The owner independently observed the same category of symptom on `project-b` (Klein SDF) and suspected a shared `sidescanpy` bug rather than two coincidental hardware artifacts, and specifically distrusted the JSF conclusion above. Investigated properly rather than defended on priors:
- **Owner's alternative hypothesis for JSF, tested directly and refuted:** "TVG got applied to invalid/near-zero far-range data, and TVG's uncapped range-growing gain formula manufactured a fake climb." Both required premises are false: `apply_tvg()` (`processing/radiometric.py`) only ever reads/writes `backscatter_intensity` — confirmed by code read and a live `np.shares_memory()` check that `backscatter_raw` is untouched. The exact file the original climb evidence came from (`unknown/ss_h5/20241017181753.ss.h5`, `hz_1495000`) has `processing_level="raw"`, empty `processing_history`, and `backscatter_raw == backscatter_intensity` byte-for-byte — TVG had definitely not run. The original evidence itself used only `_linear_to_db()` (plain unit conversion), never `apply_tvg()`'s gain formula. And raw *linear* amplitude itself (no dB, no TVG at all) climbs ~2.6x from floor to far edge — nowhere near the `1e-6` clip floor that would make "invalid" data behave pathologically. The JSF diagnosis stands, now on stronger evidence than the original investigation had.
- **JSF decode path independently audited** (`io/jsf/messages.py`): the sample-count field is cross-validated against the message's actual byte length with a hard `ValueError` on mismatch, and the converter hard-fails (not pads) on any per-ping inconsistency — rules out a fabricated-sample decode bug with high confidence. Found one unrelated, currently-dormant bug (`num_samples` uses only the 16-bit LSB, not the MSB-extended 20-bit value, until `__post_init__` recomputes it — harmless while real counts stay `<<65536`) and several parsed-but-unused header fields (`_starting_depth`, `_gain_factor`, `_user_transmit_level`, `_raw_pipe_dist_km`) — real gaps, not the cause of this issue. A fully independent clean-room byte decode (bypassing `sidescanpy.io.jsf` entirely) was spec'd but not run — the converging evidence above made it a nice-to-have, not a blocker.
- **SDF side: real bug found and fixed, distinct from the JSF issue.** `io/sdf/converter.py`'s `_build_channel_dataset` pads every ping to the group's max sample count with NaN — previously silent, and the true per-ping length was discarded. **Fixed**: now logs a warning when padding actually occurs, and a new `valid_samples` (ping, channel) variable preserves each ping's true recorded length through padding. In practice this is currently scoped to pages within one physical `.sdf` file (`01_convert.py` calls `read_and_convert()` once per file), not across a whole multi-file survey — `dispatch.py`'s multi-file-merge capability would pad across files too if ever used that way, worth knowing before doing so.
- **Independently confirmed on real `project-b` data: the same floor-then-climb hardware-artifact pattern found on JSF also exists on Klein/SDF hardware.** Raw-byte-level analysis of `survey-b/sonar_data180914072000.sdf` (`hz_900000`) reproduced the pattern directly from raw pages (bypassing sidescanpy's converter entirely) — this is new cross-vendor evidence for the "hardware TVG/AGC baked into raw signal" hypothesis, not previously checked on any non-JSF system.
- **Convert→crop pipeline inefficiency fixed**: `01_convert.py` used to write the full uncropped array to disk, then `02_crop_farrange.py` read it back and rewrote it cropped — doubling I/O for the data being thrown away. `01_convert.py` now runs `crop_far_range_noise()` in memory before its first (only) write; `02_crop_farrange.py` remains for already-converted files and becomes a fast no-op when run against files already-cropped at convert time. Verified end-to-end on real `project-a` data (converted 10 files fresh; `02_crop_farrange.py` run afterward skipped all 10 in ~1s instead of rewriting them) and by running the crop cleanup against already-existing real data: `project-a/unknown/ss_h5` (converted before this fix) reclaimed ~1.7G (4.5G → 2.8G) when cropped; `project-b/survey-b/ss_h5` was already fully cropped from an earlier session, so this run found nothing left to reclaim there (confirms the idempotency skip works correctly, not a null result from a bug).
7. **Klein SDF `depth_m` is NaN** — Klein records depth as a sensor voltage; calibration constants are in companion INI files which we don't yet parse. Requires INI calibration parsing (deferred; ~1 session). **Workaround:** users with calibration override directly: `ch.dataset["depth_m"] = (["ping"], calibrated_array)`.
8. **`compute_geometric_ranges` falls back to 1500 m/s** — When `sound_vel_ms` is NaN (always the case for SDF surveys). Acceptable for typical work; would need refinement for cold-water or thermocline-crossing surveys.
9. **Klein 3500-series assumption** — SDF reader STRUCT matches Klein 3500 layout (no fields 93/94). 5000/5900 series may differ; would need per-version STRUCTs if encountered.
10. **`channel_id` not an indexed coordinate — `.sel(channel_id=...)` breaks in xarray 2025.10+.** Tests and some internal code use `.sel(channel_id="port"/"starboard")` but `channel_id` is a non-indexed coordinate. Newer xarray raises `KeyError: "no index found for coordinate 'channel_id'"`. Fix: use `.isel(channel=0)` / `.isel(channel=1)`, or call `.set_index(channel="channel_id")` before label selection. Currently causes 3 unit test failures.
11. **`sidescan-export chip`'s "geometric" render-level output is no longer guaranteed square — not yet checked against downstream consumers (2026-09-29).** Since the scipy-binned-resampling swap (see `CHANGELOG.md`'s "Unreleased" entry), `chip`/`single-swath`/`dual-swath` styles at `--render-level geometric` can output e.g. `213x320` instead of a fixed `size_px x size_px` square, because along-track pixel resolution is now widened (fewer, coarser rows) to avoid real NaN gaps from strict binning. COCO/YOLO width/height read the actual array shape so those stay correct, but no consumer project's downstream training code has been audited for a square-input assumption. **If you're about to train on/consume these chips: check this first**, don't assume square.

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