# XTF

`.xtf` (eXtended Triton Format) is the most common vendor-neutral
sidescan format: Klein, EdgeTech and others can record or export it.
sidescanpy converts its sonar packets into the
[data format](../format/index.md). The reader follows the specification's revision 42 (Exail) and has been
checked against EdgeTech 4200 files recorded by Isis, and against public
files written by EdgeTech Discover 7.22, Isis 7.0.41 (Benthos SIS-1624)
and KleinXlt 4.0 (small cuts of them are the fixtures in
`tests/data/public/`). Only sonar packets are read; other packet types are
counted and skipped.

```python
from sidescanpy.io import read_and_convert, read_and_convert_segments
from sidescanpy.io.xtf import convert, read_xtf, split_by_sample_rate

# One survey per run of constant range (see "Range changes" below):
for stem, survey in read_and_convert_segments(path):
    ...
# or, step by step:
surveys = [convert(seg) for seg in split_by_sample_rate(read_xtf(path))]

survey = read_and_convert([path], "xtf")      # whole file, one survey
```

`read_xtf_header`, `iter_packets`, `count_packets` and `read_xtf_pings`
give the file header, the raw packets and the recorded pings (samples in
their recorded dtype) for inspection. `write_subset(src, dst, n_pings=,
start_ping=)` copies the header and a run of pings to a new file, byte for
byte (for making small test files).

## File structure

A 1024-byte file header (system, recording program, sonar name,
navigation units, and one 128-byte `ChanInfo` per channel; more than six
channels spill into further 1024-byte blocks), then packets. Every packet
starts with the magic `0xFACE`, a header type and its total byte count,
which the reader uses to step from packet to packet. A sonar packet is a
256-byte ping header followed, per channel, by a 64-byte channel header
and that channel's samples.

## Channels and data streams

The file header declares the channels (`ChanInfo`: type, frequency,
name). Every stream the reader converts comes from the sonar packets;
navigation and attitude are taken from each ping header, not from the
separate navigation or attitude packets some writers add.

| Stream | XTF source | In the data format | Rate |
|---|---|---|---|
| Sonar, port and starboard, per frequency | channels with `TypeOfChannel` 1 (port) and 2 (starboard), paired by `ChanInfo.Frequency` | one `frequencies/hz_<Hz>` per pair, each with its own ping axis | every ping |
| Position | ping header `SensorX/Ycoordinate` (`ShipX/Ycoordinate` in one case, see the table below) | per-ping `latitude`, `longitude`; `navigation` record | every ping (held between fixes) |
| Heading, pitch, roll, heave | ping header | per-ping values; `attitude` record | every ping |
| Depth, altitude | ping header `SensorDepth`, `SensorPrimaryAltitude` | `depth_m`, `altitude_m`, `fathometer_reading_m` | every ping |
| Sound velocity | ping header `SoundVelocity` | `sound_vel_ms` | every ping |
| Layback, cable out | ping header `Layback`, `CableOut` | `layback_m`, `cable_out_m`, `position_source` | every ping |
| Subbottom, bathymetry, other channels and packets | their own packets | skipped, counted in the reader's summary | |
| DVL, separate sensors | none | empty `dvl` record, no sensor records | |

Because every value is copied into each ping, it arrives at the ping
rate but changes only when its sensor updates; count the real changes
with `processing.aux_update_events` before trusting a track
([Navigation](../examples/plot_navigation.rst)).

## Mapping to the data format

| Survey | XTF source |
|---|---|
| frequency key `hz_<Hz>` | `ChanInfo.Frequency` (kHz) of a port/starboard pair |
| `backscatter_intensity` (ping, channel, sample) | port (`TypeOfChannel` 1) and starboard (2) samples, float32; sample 0 nearest the sonar on both sides (XTF stores port far -> near, so port is reversed) |
| `valid_samples` | per-ping `NumSamples`; padding beyond it is NaN |
| `latitude`, `longitude` | `SensorYcoordinate`, `SensorXcoordinate` (degrees with `NavUnits` 3; northing/easting converted from `nav_crs` with `NavUnits` 0) |
| `heading_deg`, `pitch_deg`, `roll_deg`, `heave_m` | `SensorHeading`, `SensorPitch` (nose up +), `SensorRoll` (port up +), `Heave` |
| `depth_m` | `SensorDepth` |
| `altitude_m`, `fathometer_reading_m` | `SensorPrimaryAltitude`; 0 is NaN |
| `sound_vel_ms` | `SoundVelocity`, doubled when below 1000 m/s (see below) |
| `sample_rate_hz` | `NumSamples / TimeDuration` |
| `time` | `Year` … `Second` + `HSeconds` (10 ms resolution) |
| `layback_m`, `cable_out_m` | `Layback`, `CableOut` + `CableOutHundredths` / 100 |
| `position_source` attribute | `towfish` when `SensorX/Y` sit away from `ShipX/Y`; else `unknown` (Isis files with no `ShipX/Y` and a recorded `Layback` are noted in `position_source_reason` as probably the ship). `ship` only when `SensorHeading` is 0 throughout and `SensorX/Y` sit a fixed east/north offset from `ShipX/Y` on a line away from north: the writer rotated its tow-point offset by heading 0, so the reader uses `ShipX/Y` and records `tow_point_aft_m` / `tow_point_starboard_m` (see processing → Layback) |
| navigation / attitude records | every ping header (`source = extracted_from_xtf_ping_headers`) |

`pulse_length_us` is not in the XTF headers and is 0.0. Gain codes are
not interpreted. Dataset attributes prefixed `xtf_` record the recording
program, sonar, channel numbers and names, and counts of reader
anomalies.

## Conventions and caveats

- **Sound velocity may be halved.** Some recorders store `c / 2` in
  `SoundVelocity` (750 for 1500 m/s). Values below 1000 m/s are doubled,
  and the result is cross-checked against `2 * SlantRange /
  TimeDuration`; a disagreement over 5 % is logged at WARNING.
- **Range changes.** The model has one `sample_rate_hz` per frequency,
  but some recorders change the range setting mid-file while keeping the
  sample count, so the sample interval changes. `split_by_sample_rate`
  cuts a file's pings into consecutive runs within which every
  port/starboard channel keeps its rate (`NumSamples / TimeDuration`,
  relative tolerance `rtol`, default 1 %, measured against the run's
  first ping); pings where a channel is absent never cause a cut. Runs
  shorter than `min_pings` (default 50) are dropped, since a change often
  leaves a few pings at an intermediate setting. Each run is a view of
  the file's arrays and converts to its own survey, with attributes
  `xtf_segment_index`, `xtf_segment_count`, `xtf_segment_ping_start`,
  `xtf_segment_ping_stop`, `xtf_file_pings`, `xtf_runs_dropped_short`
  and `xtf_pings_dropped_short_runs`. `read_and_convert_segments` and
  `sidescan-convert survey` do this for every XTF file and name the
  outputs `<stem>` (one run) or `<stem>_seg01`, `<stem>_seg02`, …
  (several). Converting an unsplit file with mixed rates still works:
  the most common rate is used, the other pings are counted in
  `xtf_pings_other_sample_rate` and a WARNING is logged, since their
  ranges are scaled wrongly by the ratio of the rates.
- **Pings with fewer channels than declared.** Some EdgeTech files begin
  with pings that hold only the low-frequency pair while declaring all
  four channels. The reader walks channel blocks by the packet byte
  count; those pings are left out of the high frequency (which keeps its
  own ping axis) and counted in `xtf_pings_absent`.
- **Projected coordinates** (`NavUnits` 0): `SensorXcoordinate` /
  `SensorYcoordinate` are easting / northing in metres, and the file does
  not say in which projection. Pass it as `nav_crs` (an EPSG code such as
  `32619`, a CRS string, or a `pyproj.CRS`) to `io.xtf.convert`,
  `io.read_and_convert` or `io.read_and_convert_segments`, or as
  `sidescan-convert survey --nav-crs EPSG:32619`; positions are converted
  to WGS84 longitude/latitude, (0, 0) becomes NaN, and the CRS is recorded
  in `xtf_nav_crs`. Without it such a file raises `ValueError`; the zone
  is never guessed.
- **Sample scaling.** Samples are scaled by `2**-Weight` from the channel
  header. Some older files carry meaningless `Weight` values (for example
  -8408, in some KleinXlt and GeoDAS files); blocks with `|Weight|` above
  30 are left unscaled, with a warning, and counted in the
  `xtf_weight_ignored` attribute (`xtf_weight_min` and `xtf_weight_max`
  give the range seen).
- **Older KleinXlt files.** Some KleinXlt files leave the channel
  `SampleFormat` at 255, which is not an XTF sample format. The reader
  treats it as unset, reads samples by `BytesPerSample`, and logs a
  warning.
- **No heading, and an offset that ignores the course.** Some GeoDAS
  files record `SensorHeading` 0 but offset Sensor from Ship by the tow
  point rotated by that heading, so the offset points south whatever the
  course. The reader detects this and uses the Ship positions (see the
  `position_source` row above). Their positions also jump back once a
  second at each GPS fix; `processing.smooth_navigation` removes that
  ([Navigation](../examples/plot_navigation.rst)).
- **Recorder gain.** Gain codes are read but not interpreted, and nothing
  in the file says whether the recorder applied TVG. Measure it with
  `analysis.fit_range_decay` before correcting
  ([Limitations](../limitations.md#radiometric-correction)).

## Reference

Exail (formerly Triton Imaging), *XTF File Format*, revision 42.
