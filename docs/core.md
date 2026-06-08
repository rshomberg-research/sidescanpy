# Core schema and survey

The `core` module defines the data contract used everywhere else in
SideScanPy and provides the top-level types user code interacts with.

## The two-layer model

```
SideScanSurvey                                     ← multi-frequency container
├── freq_channels: dict[str, SideScanFrequencyChannel]
│       │
│       └── dataset: xr.Dataset                    ← validated single-frequency
│               ├─ dims: ping × channel × sample
│               ├─ coords: ping_number, time, channel_id
│               └─ vars: backscatter_*, lat, lon, heading, ...
│
└── aux records (xr.Datasets)
    ├── navigation                                 ← ds.navigation_record
    ├── attitude                                   ← ds.attitude_record
    ├── dvl                                        ← ds.dvl_record
    └── sensors: dict[str, xr.Dataset]             ← ds.sensor_record
```

A frequency channel is a single ping-time-indexed dataset for one acoustic
frequency. A survey holds one or more channels plus auxiliary streams
(navigation, attitude, DVL, sensors) that are sampled on their own
timestamps — independent of pings — and shared across all frequencies.

Aux records live at the survey level (not the channel level) because
they describe the platform, not the sonar. The same GPS fixes apply to
every frequency the towfish was pinging at the same time.

## Schema validation

Every dataset is validated by an xarray accessor on construction or
assignment. A dataset that doesn't conform raises `ValueError` with a
message naming the missing piece.

| Accessor | Used on | Required vars |
|---|---|---|
| `ds.sidescan` | Single-frequency channel | `backscatter_raw`, `backscatter_intensity`, `latitude`, `longitude`, `heading_deg`, `depth_m`, `roll_deg`, `pitch_deg`, `heave_m`, `altitude_m`, `sound_vel_ms` |
| `ds.navigation_record` | Navigation aux | `latitude`, `longitude` |
| `ds.attitude_record` | Attitude aux | `roll_deg`, `pitch_deg`, `heave_m` |
| `ds.dvl_record` | DVL aux | `altitude_m` |
| `ds.sensor_record` | Generic sensor aux | (none — sensor-dependent) |

Required dims, coords, and attrs are listed in `schema.py`. The
sidescan schema enforces `channel` size 2 (port + starboard) and a
known `processing_level` value (`raw`, `radiometric`).

### Validation is eager

`SideScanFrequencyChannel.__init__` runs `ds.sidescan.validate()`.
Aux record setters on `SideScanSurvey` validate before assigning. Bad
data fails fast with a specific message — there's no "you can stuff
anything in and it'll fail later" mode.

### Empty-but-valid aux records

Aux records can be empty (zero `obs`) and still pass validation. This
matters for round-trip serialization: surveys without nav messages
still write a nav group on disk, which loads back as an empty record.
The `is_empty()` method on each aux accessor reports this. The
`SideScanSurvey.has_navigation` (etc.) shortcut returns `False` for
both `None` and empty.

The `make_empty_*_record()` helpers in `schema.py` build these
zero-obs datasets when needed.

## Survey container

`SideScanSurvey` is a thin container with a few useful conventions:

- **Frequencies** are added by string key (e.g. `"hz_1495000"`) via
  `add_frequency(key, dataset)`. Overwriting warns.
- **Aux records** are exposed as `@property` + `@setter` pairs:
  setting `survey.navigation = ds` validates and assigns; setting it to
  `None` clears.
- **Sensors** are keyed by `sensor_type` and accessed via
  `add_sensor(type, ds)` and `get_sensor(type)`.
- **Derived properties** (`time_range`, `bounding_box`,
  `frequency_hz_list`, etc.) compute on demand. The bounding box prefers
  the navigation aux record when populated, falling back to per-ping
  lat/lon arrays otherwise.

## Conventions

- All time coords are `datetime64[ns]`. Builders that produce time
  arrays from pandas may need `.astype("datetime64[ns]")` because
  pandas 2.x defaults to `[us]`.
- `processing_level` advances through the pipeline: raw → radiometric →
  geometric → georeferenced. Functions assert their expected input level
  and update the attribute on output.
- `processing_history` is a JSON-encoded list. Each entry has `step`,
  `timestamp_utc`, `params`, `function`, `version`. Use
  `ds.sidescan.append_to_history(...)` rather than building entries by
  hand.
- The `channel_id` coord is always `["port", "starboard"]` as native
  Python strings (the HDF5/NetCDF readers normalize byte-strings back
  to native strings).

## Future work

- Add tests for the aux stream accessors (currently only the main
  `sidescan` accessor has dedicated tests; aux validation is covered
  indirectly through round-trip and converter tests).
- Add a `SideScanSurvey.replace_frequency(key, ds)` method so callers
  don't have to reach into `_freq_channels` for in-place channel
  updates.
- Add a `SideScanSurvey.clone()` method for deep copying when needed
  (currently `interpolate.py::_shallow_copy_survey` does this manually).
- Optional: add `processing_level` validation that prevents going
  backwards (e.g. raw → radiometric is allowed; radiometric → raw is
  not).
- Document the `HDF5_*` constants in their own section once `docs/io_hdf5.md`
  exists, then remove the duplication here.