# Processing

`sidescanpy.processing` corrects, slices and measures frequency-channel
datasets. Every function takes and returns an `xr.Dataset`; the input is
never modified.

## Corrections

One API applies an ordered list of steps:

| Function | Applies steps to |
|---|---|
| `correct(ds, steps, params, aux=None, force=False)` | one frequency channel |
| `correct_survey(survey, steps, params, force=False)` | every channel of a survey (feeds the survey's navigation/attitude records to `interpolate_aux`) |
| `correct_file(path, steps, params, out_path=None, chunk_pings=None)` | a `.ss.h5` file, written atomically (in place by default), optionally in ping chunks to bound memory |

```python
from sidescanpy.processing import correct

ds = correct(ds, ["tvg", "geometric"], params={"tvg": {"spreading_loss_db": 30.0}})
```

### Built-in steps

In canonical order. `DEFAULT_STEPS` is `interpolate_aux → tvg → geometric → georef`.

| Step | Function | Parameters |
|---|---|---|
| `crop_farrange` | `crop_far_range` | `cutoff_sample`, `n_bins`, `min_valid_fraction`, `rise_threshold_db`, `margin_bins` |
| `interpolate_aux` | `interpolate_aux_onto_dataset` | `fields`, `method`, `extrapolation`, `drop_invalid` |
| `tvg` | `apply_tvg` | `spreading_loss_db`, `absorption_db_per_m`, `min_range_m`, `sound_velocity_ms`, `temperature_c`, `salinity_psu`, `depth_m`, `ph` |
| `geometric` | `compute_geometric_ranges` | `expose_unmasked_z` |
| `range_correction` | `apply_empirical_range_correction` | `table` (required) |
| `georef` | `compute_georeferencing` | `ellipsoid` |

Steps run in the order given; respecting dependencies is the caller's job
(`georef` and `range_correction` need `geometric`).

### Rules

- **Skip** a step by leaving it out. `params` may only name steps that
  are in `steps`; a typo raises `ValueError`, and an unknown parameter
  raises `TypeError`. Both are checked before anything runs.
- **Custom steps.** Any callable `fn(ds, **params) -> ds`, or
  `Step(name, fn)`, can be a step. It is recorded in history like a
  built-in. Give each one a distinct name; lambdas are rejected.
- **Idempotent.** A step already recorded in `processing_history` is
  skipped, so re-running is a no-op. `force=True` re-applies it; a
  radiometric step is inverted first, so new parameters replace the old
  ones instead of stacking.
- **History.** Each applied step appends
  `{"step", "params", "function", "version", "timestamp_utc"}` with the
  parameters actually used.
- **No raw copy.** Corrections modify `backscatter_intensity` in place.
  `invert(ds)` undoes recorded steps, and `uncorrected_backscatter(ds)`
  returns the signal with every radiometric step undone.
  `processing_level(ds)` reports the latest applied level (`raw`, `tvg`,
  `geometric`, `range_correction`, `georef`).

### TVG

`apply_tvg` adds the two-way propagation loss back:

- **Spreading:** `spreading_loss_db * log10(R)`. The default, 30, is the usual
  sidescan seafloor term: two-way spherical spreading (40 log R) less
  10 log R, because the insonified seafloor footprint grows linearly with
  range (e.g. [J. Mar. Sci. Eng. 2017, 5(4), 54](https://www.mdpi.com/2077-1312/5/4/54)).
  40 (point target or volume scattering) and 20 (one-way spreading) remain
  available; measure per survey with `analysis.fit_range_decay` and set the
  coefficient in the config. Use 0 when the hardware already applies TVG.
  Files record the coefficient they were corrected with in their history,
  so `invert()` still undoes them exactly whatever the default was.
- **Absorption:** `2 * alpha * R`. When `absorption_db_per_m` is not given,
  alpha comes from the Francois-Garrison model for the channel's
  frequency (`francois_garrison_absorption_db_per_m`), using
  `temperature_c`, `salinity_psu`, `depth_m` and `ph` (defaults 10 °C,
  35 psu, 10 m, pH 8). The resolved value is recorded in history.

Linear amplitude is converted to dB as `20 * log10(clip(x, 1e-6))` first,
so exact zeros map to about -120 dB. A zero or missing sound velocity
falls back to 1500 m/s. Setting a term to 0 turns it off, which is how to
avoid correcting a loss the hardware already compensated.

References:

- R. E. Francois and G. R. Garrison, "Sound absorption based on ocean
  measurements: Part I: Pure water and magnesium sulfate contributions,"
  J. Acoust. Soc. Am. 72(3), 896-907 (1982).
- R. E. Francois and G. R. Garrison, "Sound absorption based on ocean
  measurements: Part II: Boric acid contribution and equation for total
  absorption," J. Acoust. Soc. Am. 72(6), 1879-1890 (1982).

### Geometric ranges

`compute_geometric_ranges` adds per-sample `slant_range_m`,
`ground_range_m`, `across_track_m` (port negative, starboard positive,
pitch-corrected) and related variables:

- Slant range: `sample * sound_vel / (2 * sample_rate)`, from the absolute
  `sample` coordinate, so a dataset trimmed of leading samples keeps its
  true ranges.
- Ground range (flat bottom): `sqrt(slant² - altitude²)` where
  `slant > altitude`, NaN in the water column.

### Georeferencing

`compute_georeferencing` adds `latitude_per_sample` and
`longitude_per_sample` from the per-ping position, heading and across-track
offset on the WGS84 ellipsoid.

### Navigation interpolation

`interpolate_aux_onto_pings` (and the `interpolate_aux` step) replaces
per-ping latitude, longitude, heading, roll, pitch and heave with values
interpolated from the survey's navigation and attitude records
(`DEFAULT_FIELD_SOURCES`). Altitude, depth and sound velocity are live
sensor readings and are not interpolated by default; see
[JSF aux source selection](jsf_aux_source_selection.md).
`interpolate_onto_channel` takes any external `xr.Dataset` with a `time`
coordinate, such as a separate GPS log.

## Slicing

| Function | Keeps |
|---|---|
| `slice_by_pings(ds, start=, stop=)` | a ping index range |
| `slice_by_samples(ds, start=, stop=)` | a range of the absolute `sample` coordinate |
| `slice_by_range(ds, min_m, max_m)` | samples within a slant-range window |
| `slice_by_ground_range(ds, range_max_m=)` | samples within a ground range (needs `geometric`) |
| `slice_by_time(ds, start=, stop=)` | a time window |
| `slice_by_bbox(ds, lat_min=, lat_max=, lon_min=, lon_max=)` | pings (or, with `use_per_sample=True`, samples) inside a lat/lon box |
| `slice_survey(survey, fn, **kwargs)` | applies any slicer to every frequency of a survey |

Slicing keeps `processing_history`.

## Bottom, nadir and noise

| Function | What it does |
|---|---|
| `detect_first_bottom_return` | per-(ping, channel) sample where the bottom echo starts |
| `mask_nadir_gap` | NaN everything before the first bottom return |
| `detect_nadir_noise_extent`, `blank_nadir_noise` | find and zero an electronic noise burst at nadir |
| `smooth_boundary_across_pings` | rolling median of a per-ping boundary |
| `crop_far_range`, `detect_far_range_cutoff_sample` | find and crop far-range noise that rises above the floor |
| `reject_range_outliers` | per-range-bin median/MAD outlier rejection |
| `mask_bright_ping_rows` | mask pings much brighter than their neighbours |

## Segments

`find_segments` splits a file into contiguous segments and labels each one
clean or excluded, from gaps in time or position (`detect_gaps`), hardware
setting changes (`detect_setting_changes`), dropout or garbage runs
(`detect_garbage_runs`) and attitude instability (`detect_unstable_spans`).
`split_by_segments` returns one sub-survey per clean segment, and
`write_segments` writes each clean segment to its own `<label>.ss.h5`.

## Profiles and tracks

- `backscatter_vs_range` / `backscatter_vs_sample`: per-bin mean, median,
  std, percentiles and count.
- `along_track_distance_m(ds, fill_nav_stalls=False)`: cumulative
  geodesic distance along the track; `fill_nav_stalls` spreads pings that
  repeat a navigation fix.
- `build_survey_track(surveys, freq_key, scanline_stride=)`: nadir track
  and strided scan-line endpoints as a `SurveyTrack`.
