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
| `interpolate_aux` | `interpolate_aux_onto_dataset` | `fields`, `method`, `extrapolation`, `drop_invalid`, `held_values`, `max_hold_s`, `per_ping_fallback`, `groups`, `merge_window_s` |
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

- **Spreading:** `spreading_loss_db * log10(R)`. The default, 40, is
  two-way spherical spreading. 30 is a common sidescan alternative: the
  insonified seafloor patch grows in proportion to range, adding back
  10 log R (e.g. [J. Mar. Sci. Eng. 2017, 5(4), 54](https://www.mdpi.com/2077-1312/5/4/54)).
  Which fits real data is open; measure it per survey with
  `analysis.fit_range_decay` and set the coefficient in the config. 20
  (one-way spreading) was the default before 0.7.0; files corrected with
  it record 20 in their history, so `invert()` still undoes them exactly.
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

**Sample-and-hold aux streams.** Recorders commonly log the latest GPS fix
or attitude value at every ping or packet, so a value repeats until the
sensor updates (a ~1 Hz GPS under ~10 Hz pings repeats each position about
ten times; attitude streams repeat on their own schedules). Interpolating
those entries as logged keeps the stair-steps: consecutive pings stack on
one position, then jump, which shows up in ground-range and georeferenced
output. By default (`held_values="fixes"`) each aux record is first reduced
with `held_value_fixes`:

- every variable is its own group, latitude and longitude included, since
  some recorders log the two coordinates of one fix in separate entries
  (`groups=[POSITION_GROUP]` joins them);
- an entry is a fix when its group's value differs from the previous
  finite entry; NaN entries are never fixes;
- a fix that ends a held run is placed midway between the last held entry
  and the changed one (the update arrived in between); a fix right after
  another fix keeps its own time, so a stream with no repeats is unchanged;
- duplicate timestamps collapse to the last entry logged at that time;
- a value held for longer than `max_hold_s` (default 2 s) also keeps an
  anchor at its last entry, so the track stays flat across a real stop or
  a sensor dropout instead of ramping through it.

Some recorders also spread one sensor update over consecutive entries:
the bulk change, then a small follow-up (a few cm) one entry later, then
nothing until the next update. Every change then becomes a fix, two fixes
land one entry apart, and the interpolated speed swings between a fraction
of and several times the true speed. `merge_window_s` (default `None`, no
merging) collapses changes of one group that are at most that many
seconds apart (chained) into one update event:

- the event carries the group's value after its last change;
- it is timed like a fix at its *first* change (the midpoint after a
  hold): the update started reaching the log there, and the follow-ups
  complete it rather than report new motion. This gives every event the
  same latency, whereas timing at the last change would add a delay that
  depends on how many entries the update spread over;
- a `max_hold_s` hold is measured from the event's last change;
- groups merge independently, and the record's first entry is never
  merged.

Choose the window longer than the follow-up delay (one entry) and
shorter than the gap from the follow-up to the next update; about two log
intervals usually fits. Too long a window chains successive updates into
one event and is far worse than no merging. A float applies to every
group of every record, so a window sized for a ~3 Hz GPS also chains the
changes of a ~10 Hz roll sensor; a mapping by variable name, such as
`merge_window_s={"latitude": 0.15, "longitude": 0.15}`, merges only the
groups holding those variables (a group takes the smallest window of its
variables). A WARNING is logged when an event combines more than five
changes. `aux_update_events(record, groups=..., merge_window_s=...)`
reports, per group, the entries, raw changes, events after merging, the
percentiles (p5–p95) of the gaps between changes and between events, in
entries and seconds, and the fraction of changes that fall within the
window of the previous one. Split updates show as bimodal change gaps
(a peak at 1 entry and one at the update interval) that become unimodal
event gaps once merged. On a line running mostly north–south one
coordinate barely changes, so per-coordinate change counts say little
about how updates are split; check the joint group
(`groups=[POSITION_GROUP]`) too.

The step's `groups` parameter is passed to `held_value_fixes` for each
record, applying a group only where all its variables exist, so
`groups=[POSITION_GROUP]` joins latitude and longitude in the navigation
record and leaves attitude alone.

When a field has no usable aux source (record absent or empty, variable
missing or all NaN), `per_ping_fallback=True` (default) reduces the
channel's own per-ping values the same way and interpolates those. The
step's history record carries `held_values`, `max_hold_s`,
`per_ping_fallback`, `per_ping_fallback_fields`, `groups` and
`merge_window_s`. `held_values="raw"`
interpolates the entries as logged, as in 0.7.2 and earlier.

The step is skipped on files whose history already records it; re-convert,
or correct with `force=True`, to apply the reduction to a file corrected
before.

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
