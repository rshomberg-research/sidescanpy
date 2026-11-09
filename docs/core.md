# Data model

Every reader produces the same structure, and every correction, grid
and export function works on it. This page describes that structure and
its conventions. [HDF5](io_hdf5.md) lists the variables as they are
stored in a file.

## Survey and frequency channels

```
SideScanSurvey
├── frequencies: one SideScanFrequencyChannel per frequency, keyed "hz_<Hz>"
│   └── .dataset: xr.Dataset, dims (ping, channel, sample)
│       ├── backscatter_intensity (ping, channel, sample)
│       ├── per-ping variables: latitude, longitude, heading_deg, altitude_m, ...
│       └── attrs: sample_rate_hz, freq_hz_start, freq_hz_end, processing_history
├── navigation   xr.Dataset on its own time axis (obs)
├── attitude     xr.Dataset on its own time axis (obs)
├── dvl          xr.Dataset on its own time axis (obs)
└── sensors      {"pressure": xr.Dataset, ...}
```

A **frequency channel** is one acoustic frequency: an `xarray.Dataset`
indexed by ping, with both sides of the swath. A dual-frequency sonar
gives two channels, each with its own ping axis, because the two
frequencies need not ping together.

The **aux records** (navigation, attitude, DVL, sensors) hold the
platform's sensor streams on their own timestamps. They belong to the
survey rather than to a frequency because they describe the towfish or
vehicle, and the same GPS fixes apply to every frequency. The
`interpolate_aux` correction step interpolates them onto each channel's
ping times.

```python
survey.frequencies                        # ['hz_100000', 'hz_400000']
ds = survey.get_frequency("hz_400000").dataset
survey.navigation                         # xr.Dataset or None
survey.has_navigation                     # False when absent or empty
survey.get_sensor("pressure")              # KeyError if the file had none
survey.time_range                         # (first, last) ping time
survey.bounding_box                       # (lat_min, lat_max, lon_min, lon_max)
```

## Conventions

- **Channels.** `channel` always has size 2: index 0 is port and 1 is
  starboard (`channel_id` = `["port", "starboard"]`). Port and starboard
  samples are separate observations and are never averaged together.
- **Samples.** Sample 0 is nearest the sonar on both sides. `sample` is
  an absolute coordinate: a dataset sliced to drop near-range samples
  keeps its true sample numbers, so ranges computed from it stay right.
- **Signs.** Across-track distance is negative to port and positive to
  starboard. Along-track is positive forward. Heading is a compass
  bearing, degrees clockwise from north.
- **Units.** Per-ping variable names carry their unit: `_deg` degrees,
  `_m` metres, `_ms` metres per second. Positions are WGS84 degrees.
- **Time.** All `time` coordinates are `datetime64[ns]`.
- **Missing values** are NaN. An aux record can be empty (zero
  observations) and still be valid; it is written to file as an empty
  group.

## Validation

A dataset is checked when it becomes a channel or an aux record, and a
dataset that does not fit raises `ValueError` naming what is missing.
The checks are done by xarray accessors you can call yourself:

| Accessor | Checks | Required variables |
|---|---|---|
| `ds.sidescan` | a frequency channel | `backscatter_intensity`, `latitude`, `longitude`, `heading_deg`, `depth_m`, `roll_deg`, `pitch_deg`, `heave_m`, `altitude_m`, `sound_vel_ms`, `fathometer_reading_m` |
| `ds.navigation_record` | navigation | `latitude`, `longitude` |
| `ds.attitude_record` | attitude | `roll_deg`, `pitch_deg`, `heave_m` |
| `ds.dvl_record` | DVL | `altitude_m` |
| `ds.sensor_record` | any sensor | none (needs `time` and the attributes `source`, `sensor_type`) |

A channel also needs the dimensions `ping`, `channel` (size 2) and
`sample`, the coordinates `ping_number`, `time` and `channel_id`, and the
attributes `freq_hz_start`, `freq_hz_end`, `pulse_length_us` and
`sample_rate_hz`. `ds.sidescan.validate()` raises on the first problem;
`ds.sidescan.is_valid()` returns a bool.

To build a survey from your own data, make datasets that pass these
checks and add them:

```python
from pathlib import Path
from sidescanpy.core import SideScanSurvey

survey = SideScanSurvey(source_files=[Path("my_line.raw")], source_format="custom",
                        survey_name="my_line")
survey.add_frequency("hz_400000", ds)     # validates ds
survey.navigation = nav                   # validates nav; None clears it
```

## Processing history

Each correction step appends a record to the channel's
`processing_history` attribute (a JSON list) with the step name, the
parameters actually used, the function, the sidescanpy version and a UTC
timestamp. The processing level (`raw`, `tvg`, `geometric`,
`range_correction`, `georef`) is derived from that list by
`core.history.processing_level(ds)`; it is not stored separately. Steps
already in the history are skipped on a re-run, and radiometric steps
can be undone from their recorded parameters. See
[Processing](processing.md#rules).
