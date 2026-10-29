# sidescanpy — Claude Code Context

**What:** Python library to read, correct, grid and export sidescan sonar
data (EdgeTech JSF, Klein SDF, XTF → `.ss.h5`).
**Owner:** Russ Shomberg <rshomberg@uri.edu>
**Version:** 0.8.0. History: `CHANGELOG.md` and `git log`.
**Python:** 3.12+. **License:** MIT (private repo for now, publication planned).
**Pre-release:** code shared by invitation (read access) until the
public release; the docs site is public
(https://rshomberg-research.github.io/sidescanpy/, pushed into the org's
public site repo, so everything in `docs/` and every docstring is
public). The `.ss.h5` format, API and CLI may change in any 0.x release,
and the README and docs say so. Keep those warnings until 1.0. Sharing
setup (site publishing, fixture zip, no Zenodo yet):
`docs/development.md` → "Sharing before release".

This file is the working guide, not a log. Session narratives belong in
commit messages; design notes are kept outside this repo, in the parent
workspace's `notes/sidescanpy/` (not published).

Private downstream projects consume this library. Never name or describe
them, their surveys, sites or deployments anywhere in this repo (source,
tests, docs, CHANGELOG, commit messages). Describe format and hardware
knowledge generically ("some Klein 4900 files …").

---

## Architecture contract

**sidescanpy is a library API.** It provides:

- format ingestion (`io`) into the shared `.ss.h5` format
- corrections: radiometric, geometric, georeferencing
- slicing across pings/samples/frequencies, and manual cuts and masks
- gridding in both the swath (along/across-track) and georeferenced frames
- targets, and dataset export (chips/windows, COCO/YOLO, GeoTIFF)
- small, scoped quantitative analysis tools (nadir/first-bottom detection,
  TVG curve fitting, usable range, units, beam geometry, segments, …).
  Each takes data and returns a result dataclass or `xr.Dataset`. None
  runs a workflow.
- thin plotting helpers (`viz`) that render one result and return
  Axes/Figure without saving anything
- logging throughout

**The `sidescan-*` CLI is a thin quick-run convenience.** Each subcommand
parses arguments, calls library functions and logs a short result.
Library-grade logic never lives in `cli/`.

**Workflow belongs to the caller.** Orchestration, run-everything drivers,
numbered steps, multi-panel figure layouts and per-survey decisions live
in consumer projects, notebooks and `examples/`. A project typically has
one script per step (convert → analyze → correct → export) sharing one
config; each step writes outputs, QA figures and a report so a human can
inspect before the next step. Every correction step must be usable on its
own, skippable, and replaceable by a caller-supplied function.

### Rules

- Dependencies point one way: `core ← io ← processing ← analysis ← grids
  ← targets ← export ← viz ← cli` (enforced by `tests/unit/test_layering.py`).
  Nothing below `viz` imports matplotlib.
- Functions take and return `xr.Dataset`. `SideScanSurvey` is the
  multi-frequency container plus file I/O.
- Corrections modify `backscatter_intensity` in place; no raw copy is kept.
  Raw source files are the archive, radiometric steps are invertible from
  their recorded parameters, and every applied step is recorded in
  `processing_history`. What a project knows about its input (units,
  hardware TVG state) is the project's to record.
- Every module has `logger = logging.getLogger(__name__)`. Library code
  logs only at DEBUG and WARNING. No `print()`, no `basicConfig`, no
  `__main__` blocks (enforced by `tests/unit/test_logging_policy.py`).
- The word "stage" does not appear in library code (`tests/unit/test_gates.py`).
- Lean on established libraries (numpy, scipy, xarray, pyproj,
  rioxarray). Identify candidates aggressively, swap conservatively: tag
  them `# TODO(lib): …` and swap only once a parity test proves the
  output matches (design note `grids-library-swaps.md`). Justify every
  new dependency in `pyproject.toml`.
- Config files: YAML for anything a human edits, JSON for
  machine-written artifacts (`targets.json`, `metrics.json`, COCO).

---

## Layout

```
src/sidescanpy/
├── core/        schema accessors, SideScanSurvey, Target, history, units
├── io/          jsf/, sdf/, xtf/ readers+converters; hdf5/ (.ss.h5, survey
│                dirs); netcdf/ (single channel); dispatch, catalogs
├── processing/  correct (step API), radiometric (TVG, Francois-Garrison),
│                geometric, georeferencing, interpolate, aux_fixes, inversion,
│                slicing, bottom, farrange, outliers, segments,
│                navigation, profiles, range_correction, track, sample_axis
├── analysis/    characterize, nadir, beam, usable_range, tvg, navigation,
│                geometry, outliers (pure functions → dataclasses)
├── grids/       GridDefinition, resample, SideScanGriddedRaster,
│                recommend_resolution, geotiff
├── targets/     catalog I/O, correlate (target ↔ ping), waterfall boxes
├── export/      image (8/16-bit), windows/ (chip, swath, geo, background),
│                orientation, augment, target_window, ml/ (coco, yolo, dataset)
├── viz/         waterfall, gridded, track, targets, analysis, tvg plots
├── cli/         sidescan-convert / -correct / -export / -analyze
├── datasets.py  fetch() public example recordings (stdlib only, any layer may use)
└── log.py       configure_logging, log_timing
examples/        runnable scripts, smoke-tested in tests/examples;
                 gallery/ runs on fetched public data (`make gallery`, not in CI)
docs/            mkdocs site (guide + API reference)
```

### Data model

Backscatter is 3-D `(ping, channel, sample)`; `channel` is always size 2,
`0` = port, `1` = starboard. Never average across `channel`; port and
starboard samples are separate observations. `sample` is an absolute
integer coordinate, so slicing samples does not shift range. Sign
conventions: across-track port negative, starboard positive; along-track
forward positive; heading compass bearing clockwise from north.
`processing_level` is derived from history: `raw`, `tvg`, `geometric`,
`range_correction`, `georef`.

---

## Commands

```bash
uv sync --all-extras          # environment (the dev extra alone skips viz tests)
make test                     # unit tests; clears PYTHONPATH (see below)
make lint                     # ruff
make docs                     # mkdocs build --strict
make gallery                  # re-run examples/gallery, refresh docs/assets/gallery
PYTHONPATH="" uv run pytest tests/unit/grids/test_resample.py -v
PYTHONPATH="" uv run pytest tests/examples -q   # example smoke runs
```

ROS2 on this machine injects `PYTHONPATH=/opt/ros/...` into every shell,
which breaks pytest imports. Use `make` or prefix `PYTHONPATH=""`.

Integration tests in `tests/integration/` need real survey files that are
not in the repo; they skip without them. Small byte-for-byte cuts of
openly licensed public recordings (JSF, and XTF from three writers) are
listed in `tests/data/public/SOURCES.md` but also kept out of the repo:
set `SIDESCANPY_PUBLIC_DATA` to their directory to run
`test_public_fixtures.py`.

---

## Conventions

- **Tests first** for new behavior. Synthetic fixtures with real shapes
  (3-D, two channels, realistic sample counts). Never commit survey data,
  public or not; public test files are listed (source, licence,
  checksum) in `tests/data/public/SOURCES.md` and live outside the repo.
  `tests/unit/synthetic.py` has the shared builders.
- **Test names:** `test_<what>_<when>()`, grouped in classes by feature.
- **Docstrings:** numpydoc. They are the API reference (`make docs`
  fails on a malformed section). State behavior, parameters and caveats
  that are still true; no session history.
- **Schema first:** add a variable to `core/schema.py` before a step
  produces it.
- **No silent transformations:** a mask or skipped step logs at DEBUG;
  surprising data logs at WARNING.
- **Commits:** Conventional Commits with hyphenated scopes
  (`feat(io/sdf):`, `fix(grids/resample):`), one logical change each.
  Suggest commits; commit only when the user asks.
- **Versioning:** SemVer, bumped per release, not per commit. Deprecation
  shims added in 0.6.0 are removed in 0.8.0.

### Gates (all in the unit suite)

| Test | Enforces |
|---|---|
| `test_layering.py` | import direction; no matplotlib below `viz` |
| `test_logging_policy.py` | no INFO/print/basicConfig in library code |
| `test_gates.py` | no "stage" wording in `src/` |
| `test_import_all.py` | every module imports |
| `test_optional_matplotlib.py` | non-viz layers and convert/correct run without matplotlib |
| `tests/examples/` | every example runs on synthetic data |

---

## Known issues

- **Hardware gain state is unknown.** JSF files carry a per-ping
  `_gain_factor` that tracks the signal like an AGC; whether "raw" samples
  are already gain-compensated is unresolved. Some recordings behave as if
  spreading or both TVG terms were applied in hardware. Measure per survey
  with `analysis.fit_range_decay` and turn TVG terms off accordingly; do
  not assume.
- **TVG spreading coefficient is unsettled.** The default is 30 log R
  (insonified area grows with range); 40 log R (two-way spherical) is the
  physics alternative. `spreading_loss_db` sets it; compare per survey.
- **Klein SDF depth is NaN.** Depth is a sensor voltage; the calibration
  lives in companion INI files that are not parsed.
- **Sound velocity.** Zero or missing values fall back to 1500 m/s
  (always the case for SDF). One value per ping; no water-column profile.
- **SDF layout** matches the Klein 3500 series (no fields 93/94); bit-17
  masking of samples is empirical. Pings of different lengths within a
  file are NaN-padded and `valid_samples` records the true length; a
  change of sample interval (range setting) is only handled by
  `read_and_convert_segments`, which splits the file.
- **DVL validity flags** reject everything on the devices seen so far;
  embedded per-ping values are used instead. JSF attitude (type 2020)
  flags are honoured: pitch, roll and heading are valid on the devices
  seen so far, heave is flagged invalid and stays NaN.
- **griddata resampling** (`resample_method="griddata"`, the dataset
  default) interpolates most pixels at resolutions finer than the ping
  spacing, is much slower than `kdtree`/`binned`, and dominates peak
  memory for 1024 px swath windows on dense surveys.
- **Re-correcting a georeferenced file is slow.** Adding a step to a file
  that already has `latitude_per_sample` etc. spends ~250 s per 2500-ping
  file in h5py writes (one cheap custom step), against ~15 s for all
  steps in one pass on a freshly converted file. Not yet diagnosed
  (chunk-size alignment to `HDF5_CHUNK_PING` is not the cause). Until
  then, run every step in one pass from a fresh conversion.
- **Gridding:** a UTM grid uses the zone of the bbox centre;
  `bbox_from_channel_dataset` pads by a fixed amount in degrees.
- **Analysis assumptions:** `analyze_nadir` detects on values as given
  (not inverted); `estimate_beam_geometry` assumes two channels;
  `fit_range_decay` picks one named candidate and does not fit absorption
  freely.
- **`fathometer_reading_m`** is not attitude-corrected.
- **Missing altitude.** Some recordings have none (some EdgeTech Discover
  files write 0 in JSF and nothing in XTF). The opt-in `bottom_track` step
  fills it from the first bottom return; it reads slightly high on sloping
  or very shallow seabed (near-edge return, not vertical).

---

## Lessons

- **Passing tests on unrealistic data is false confidence.** A gridding
  refactor passed with 2-D fixtures while real data is 3-D.
- **A good fit is a hypothesis.** Confirm a TVG recipe by applying it and
  checking the corrected data is flat; RMSE alone misled more than once.
- **Validate on the same window the fit used.** Recomputing the window
  independently tested the wrong thing.
- **A smooth boundary can be a stuck detector.** Compare sides or
  frequencies when a result looks too clean.
- **Diagnostic views are not data products.** Judge a GeoTIFF in a GIS,
  not by a coverage-alpha plot.
- **Aux streams are sample-and-hold.** In every supported format, nav and
  attitude (aux records and per-ping fields) repeat the latest sensor value
  until it updates. Count repeats before trusting a track, and interpolate
  between real fixes (`held_value_fixes`), not between logged entries.
- **Documentation drifts.** Check claims in this file against the code
  before relying on them.
