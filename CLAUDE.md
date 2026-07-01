# SideScanPy — Claude Code Context

**Project:** Python library for processing sidescan sonar data
**Owner:** Russ Shomberg <rshomberg@uri.edu>
**Version:** 0.5.0 tagged; v0.6.0 gridding module complete, unreleased
**Python:** 3.14+

The package can:
- Ingest Edgetech JSF (`.jsf`) and Klein SDF (`.sdf`) sidescan surveys
- Convert them to a standardized `xarray`-based `SideScanSurvey` model
- Run them through a processing pipeline: interpolate aux streams → TVG correction → geometric ranges → georeferencing
- Round-trip serialize to `.ss.h5` (HDF5) bit-exactly
- Resample georeferenced channels onto a regular grid (`sidescanpy.grids`, v0.6.0)
- Produce waterfall and gridded-raster visualizations, plus PNG exports

---

## Quick Reference

- **Source code:** `src/sidescanpy/`
- **Tests:** `tests/unit/` (fast), `tests/integration/` (slow, requires fixtures)
- **Docs:** `docs/` (module refs), `docs/planning/` (feature plans)
- **Scripts:** `scripts/` (CLI tools and demos)
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
│   └── slicing.py      # Subset operations
├── grids/              # NEW in v0.6.0
│   ├── definition.py   # GridDefinition (bbox + resolution + CRS)
│   ├── raster.py       # SideScanGriddedRaster (xarray wrapper)
│   └── resample.py     # resample_channel_to_grid()
└── viz/
    ├── waterfall.py    # Port+starboard waterfall display
    └── gridded.py      # gridded raster plots (v0.6.0)

scripts/
├── convert.py                    # Format-dispatching converter (.jsf or .sdf)
├── batch_convert_and_plot.py     # Directory batch processing
├── plot_sonar_ping.py            # Interactive viewer
├── export_waterfall_series.py    # Slice + PNG export
└── (gridding demos, diagnostic scripts)
```

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
```bash
# Activate venv
source .venv/bin/activate

# Run unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/grids/test_resample.py -v

# Run with coverage
pytest tests/unit/ --cov=sidescanpy --cov-report=term-missing
```

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

### Test Status (as of 2026-07-01)

```
tests/unit/                   880 passing, 38 skipped
tests/integration/             (run with --slow; requires local JSF/SDF fixtures — see Known Issues #1)
```

---

## Roadmap

### v0.7.0 — GeoTIFF export + CSV-driven target cropping

**Status:** Planning (2026-07-01). See `docs/planning/v0.7.0-geotiff-and-cropping.md` for the full plan — awaiting approval before implementation starts.

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
5. **Klein SDF `depth_m` is NaN** — Klein records depth as a sensor voltage; calibration constants are in companion INI files which we don't yet parse. Requires INI calibration parsing (deferred; ~1 session). **Workaround:** users with calibration override directly: `ch.dataset["depth_m"] = (["ping"], calibrated_array)`.
6. **`compute_geometric_ranges` falls back to 1500 m/s** — When `sound_vel_ms` is NaN (always the case for SDF surveys). Acceptable for typical work; would need refinement for cold-water or thermocline-crossing surveys.
7. **Klein 3500-series assumption** — SDF reader STRUCT matches Klein 3500 layout (no fields 93/94). 5000/5900 series may differ; would need per-version STRUCTs if encountered.

### Low Priority
8. **SDF `--x-axis ground_range` rendering is "blurry."** Likely due to inaccurate altitude (depth not calibrated → altitude derived poorly). Will improve when bottom detection or INI parsing lands. Cosmetic for now since JSF data renders correctly.
9. **Slider performance** in `plot_sonar_ping.py` — Full waterfall re-renders each step. Lags on 5000-ping surveys.
10. **DVL & attitude validity flags** — Reject all data on tested devices; embedded values used as fallback.
11. **`fathometer_reading_m` not attitude-corrected** — Will need `* cos(roll) * cos(pitch)` once attitude correction is implemented.
12. **`messages.py` (JSF) has uneven naming** — Some `_deg` suffixes, some not. Mix of `Optional[float] = None` and `float = nan`. Cosmetic cleanup deferred.

---

## Session Handoff

At the end of a major work session, update the "Current Work Focus" section above with:
- What was accomplished
- New test counts
- Any new issues discovered
- Next steps

This keeps `CLAUDE.md` as the single source of truth for project state.