# HDF5 (.ss.h5)

sidescanpy stores a whole survey (every frequency and every aux record)
in one HDF5 file with the extension `.ss.h5`. Each group is a NetCDF-4
group written with h5netcdf, so any HDF5 or NetCDF tool can read it, and
xarray can open each group without sidescanpy.

:::{admonition} The format is not frozen
:class: warning
The layout below may change in any 0.x release before 1.0, and a
file written by one 0.x version may not read in a later one. Keep the
raw source files and re-convert after an upgrade. The root attribute
`sidescanpy_version` records which version wrote a file.
:::

```python
from sidescanpy.io.hdf5 import read_survey, read_survey_pings, write_survey

write_survey(survey, "line_001.ss.h5")
survey = read_survey("line_001.ss.h5")
survey = read_survey("line_001.ss.h5", chunks={"ping": 256})   # lazy, dask-backed
survey = read_survey_pings("line_001.ss.h5", start=0, stop=500) # a run of pings
```

`write_survey` refuses to overwrite an existing file unless
`overwrite=True`. `processing.correct_file` corrects a file in place,
writing to a temporary file and renaming it so an interrupted run never
leaves a half-written survey.

## Layout

```
line_001.ss.h5
├── /                        root attributes (below)
├── frequencies/
│   ├── hz_100000/           one group per frequency: a SideScanFrequencyChannel
│   └── hz_400000/
└── aux_streams/
    ├── navigation/          always written, empty if the file had none
    ├── attitude/            always written, empty if the file had none
    ├── dvl/                 always written, empty if the file had none
    └── sensors/             only if the survey has sensor records
        └── pressure/
```

Root attributes:

| Attribute | Content |
|---|---|
| `sidescanpy_version` | version that wrote the file (absent before 0.8.0) |
| `survey_name` | defaults to the first source file's stem |
| `source_format` | `jsf`, `sdf` or `xtf` |
| `source_files_json` | JSON list of the source file paths |

## Frequency groups

Each `frequencies/hz_<Hz>` group holds dimensions `ping`, `channel`
(always 2: `port`, `starboard`) and `sample`. A freshly converted file
has:

| Variable | Dimensions | Type | Meaning |
|---|---|---|---|
| `backscatter_intensity` | ping, channel, sample | float32 | amplitude as recorded (scaled by the format's weighting); corrected in place by radiometric steps |
| `valid_samples` | ping, channel | int32 | samples actually recorded in each ping; the rest is NaN padding |
| `time` | ping | datetime64 | ping time as recorded |
| `ping_number` | ping | int32 | ping number from the file |
| `sample` | sample | int64 | absolute sample index; slicing keeps it, so ranges stay right |
| `channel_id` | channel | string | `port`, `starboard` |
| `latitude`, `longitude` | ping | float64 | position, degrees (WGS84) |
| `heading_deg` | ping | float64 | heading, degrees clockwise from north |
| `roll_deg`, `pitch_deg` | ping | float64 | attitude, degrees |
| `heave_m` | ping | float64 | heave, metres |
| `depth_m` | ping | float32 | sensor depth below the surface, metres |
| `altitude_m` | ping | float32 | height above the seabed, metres |
| `fathometer_reading_m` | ping | float32 | altimeter reading as recorded, metres |
| `sound_vel_ms` | ping | float32 | sound velocity, m/s (NaN or 0 when not recorded) |

The `geometric` and `georef` correction steps add per-sample variables
on `(ping, channel, sample)`, each with `units` and `long_name`
attributes:

| Variable | Added by | Meaning |
|---|---|---|
| `slant_range_m` | `geometric` | slant range from the transducer |
| `ground_range_m` | `geometric` | range across the seabed; NaN in the water column |
| `across_track_m` | `geometric` | signed across-track distance, port negative |
| `along_track_m` | `geometric` | along-track offset from nadir (pitch) |
| `vertical_offset_m` | `geometric` | vertical distance from the transducer to the sample |
| `depth_per_sample_m` | `geometric` | water depth at the sample |
| `latitude_per_sample`, `longitude_per_sample` | `georef` | position of the sample, degrees (float64) |

Group attributes:

| Attribute | Content |
|---|---|
| `freq_hz_start`, `freq_hz_end` | transmit frequency band, Hz |
| `sample_rate_hz` | sample rate, Hz |
| `pulse_length_us` | pulse length, µs (0 when the format does not record it) |
| `processing_history` | JSON list of applied correction steps ([Processing](processing.md#rules)) |
| `xtf_*`, ... | reader provenance: recording program, sonar name, channel numbers, counts of skipped or repaired packets |

Per-ping variables carry their units in the name suffix (`_deg`, `_m`,
`_ms`) rather than a `units` attribute.

## Aux groups

Aux records have their own `time` on an `obs` dimension, independent of
the pings, because sensors update on their own schedules.

| Group | Variables |
|---|---|
| `navigation` | `latitude`, `longitude`, and where recorded `heading_deg`, `speed_knots` |
| `attitude` | `roll_deg`, `pitch_deg`, `heave_m`, and where recorded `heading_deg` |
| `dvl` | `altitude_m` |
| `sensors/<type>` | sensor-specific; `pressure` holds depth, sound speed and temperature |

Each group has a `source` attribute naming where its values came from
(for example `extracted_from_xtf_ping_headers`).

## Encoding

- Variables with two or more dimensions are gzip level 4 with shuffle,
  chunked 256 pings at a time. Per-ping variables are uncompressed.
- float64 is stored as float32, except latitude and longitude, where
  float32 resolves only about 0.4 m.
- `read_survey(path, chunks={"ping": 256})` matches the on-disk chunks;
  other chunk sizes work but read more than they need. Aux records
  always load into memory (they are small).
- Files from 0.5.0 may also carry `backscatter_raw`; they still read,
  and `correct_file` drops that variable when it rewrites the file.

## Reading without sidescanpy

```python
import xarray as xr

ds = xr.open_dataset("line_001.ss.h5", group="frequencies/hz_400000", engine="h5netcdf")
nav = xr.open_dataset("line_001.ss.h5", group="aux_streams/navigation", engine="h5netcdf")
```

From the shell, `h5ls -r line_001.ss.h5` lists every group and variable
and `h5dump -A line_001.ss.h5` prints the attributes.

For one frequency on its own, as a plain NetCDF file, see
[NetCDF](io_netcdf.md).
