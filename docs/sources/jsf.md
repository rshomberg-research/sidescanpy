# EdgeTech JSF

EdgeTech sonars and the Discover acquisition software record JSF
(`.jsf`). sidescanpy reads the sidescan sonar data and the navigation,
attitude and sensor messages around it, and converts them into the
[data format](../format/index.md).

```python
from sidescanpy.io import read_and_convert

survey = read_and_convert(["line_001.jsf"], fmt="jsf")
```

For lower-level work, `io.jsf.read_jsf(path)` returns every message as a
typed object, `read_jsf_iter(path)` yields them one at a time for large
files, and `io.jsf.convert(messages, source_files=...)` builds the survey
from a message list. `write_subset(src, dst, n_pings=, start_ping=)`
copies a run of pings, with the messages between them, into a new file
byte for byte.

## File structure

A stream of messages, each a 16-byte header (with the message type,
subsystem and channel) followed by a typed payload. Sonar data, NMEA
sentences and the sensor messages are interleaved in the order they
arrived, each with its own timestamp.

## Channels and data streams

A JSF file carries the sonar channels and several sensor streams, and
often the same quantity in more than one of them:

| Stream | JSF message | In the data format | Typical rate (one towfish) |
|---|---|---|---|
| Sonar, port and starboard, per frequency | sonar data (80), channel 0 port, 1 starboard | `frequencies/hz_<Hz>`: `backscatter_intensity` | every ping |
| Position, attitude, altitude, depth, sound velocity copied into each ping | header of message 80 | per-ping variables (`latitude`, `heading_deg`, `altitude_m`, ...) | arrives every ping, changes at the rate of its source |
| Layback and cable out | header of message 80 | `layback_m`, `cable_out_m`, `position_source` | every ping |
| Position | NMEA RMC, GLL, GGA (2002) | `navigation` record | about 0.5 Hz |
| Pitch, roll, heading, heave | pitch/roll (2020) | `attitude` record | about 10 Hz |
| Depth, sound speed, temperature | pressure sensor (2060) | `sensors["pressure"]` record | a few Hz |
| Doppler velocity log | DVL (2080) | `dvl` record | when fitted |

Sonar messages are grouped by their transmit start and end frequency,
one `hz_<Hz>` dataset per group, so a dual-frequency towfish gives two
frequencies, each with its own ping axis. A record with no messages is
empty but valid (`has_navigation`, `has_attitude` and `has_dvl` report
`False`); the converter does not fill one record from another source.

### Arrival rate is not update rate

Most of the copies are sample-and-hold. The sonar header repeats the
latest fix it has seen, so a value that arrives with every ping (tens of
times a second) may change only when the GPS updates (about once a
second or slower). Interpolating the repeated values as if each were a
new fix puts pings in stair-steps along the track. On one EdgeTech
towfish recording:

| Quantity | Source | Arrives | Changes |
|---|---|---|---|
| latitude, longitude | NMEA RMC | 0.5 Hz | 0.5 Hz |
| latitude, longitude | sonar header | 95 Hz | 0.5 Hz, one message after the NMEA fix |
| roll, pitch, heading | pitch/roll message | 10 Hz | 10 Hz |
| roll, pitch, heading | sonar header | 95 Hz | 10 Hz, the previous pitch/roll value |
| altitude | sonar header | 95 Hz | 38 Hz (the altimeter itself) |
| depth | sonar header | 95 Hz | 6 Hz (a live depth sensor) |

The sonar-header position and attitude are copies of the NMEA and
pitch/roll streams. The sonar-header altitude and depth change faster
than any other source and are the sensors' own readings.

### Which source each per-ping value uses

The `interpolate_aux` correction step replaces the per-ping copies with
values interpolated from the aux records at each ping time, after
reducing each record to its real updates (see
[Processing](../processing.md#navigation-interpolation)):

| Per-ping field | Interpolated from | When the record is empty |
|---|---|---|
| `latitude`, `longitude` | `navigation` | the sonar-header values, reduced the same way |
| `heading_deg`, `roll_deg`, `pitch_deg`, `heave_m` | `attitude` | the sonar-header values, reduced the same way |
| `altitude_m`, `depth_m`, `sound_vel_ms` | not interpolated | the sonar-header values are kept as recorded |

These are the defaults in `processing.DEFAULT_FIELD_SOURCES`. The step's
`fields` parameter interpolates a subset, or points a field at another
record; `interpolate_onto_channel` interpolates any dataset with a `time`
coordinate, such as a separate GPS log.

### Checking a new file

`processing.aux_update_events(record)` reports, for each variable in a
record, how many entries it has, how many of them change the value, and
the gaps between changes. Compare the change rate of a per-ping field
with the arrival rate of the aux streams: a field that changes exactly
when an aux stream arrives is a copy of it; a field that changes faster
than every aux stream is a live sensor.

Timing alone can mislead. On the recording above the sonar-header depth
changed at the same cadence as the pressure messages arrived, but the
pressure depth was 0 throughout while the sonar-header depth ranged from
3.5 to 6.0 m: a different sensor that happened to update on the same
schedule. Compare the values too
([What's in a file](../examples/plot_whats_in_a_file.rst) does this on a
public line).

## Mapping to the data format

- **Pairing.** Port and starboard messages are paired by ping number.
  A ping with only one side is kept with the other side NaN by default
  (`repair_mode="nan_pad"`); `"drop"` drops it and `"abort"` raises.
- **Samples.** Envelope samples are scaled by the message's weighting
  factor (2^-N) and stored as float32 `backscatter_intensity`. The
  per-ping gain factor is read but not applied (see
  [Limitations](../limitations.md)).
- **Time.** Ping time is the sonar message timestamp in nanoseconds;
  aux messages carry their own timestamps. All `time` coordinates are
  `datetime64[ns]`.
- **Invalid values.** Fields whose validity flag is clear become NaN.
- **Layback.** `layback_m` (bytes 228-231) and `cable_out_m` (bytes
  236-237, decimetres in the file) are carried per ping. Validity bit 13
  ("position interpolated") sets `position_source`: set, the position is
  the sonar's with layback applied by Discover; clear, it is the last
  navigation fix received, without layback. The cable-counter validity
  bit is clear in the Discover files seen so far although the cable out
  is plausible; the value is kept.

Other message types are either parsed into a typed object and not
converted (system information, navigation offsets, file timestamps,
situation, cable counter) or skipped, with one summary warning per file
listing the unsupported types.

## Known quirks

- **Heave is NaN.** On the devices seen so far, pitch/roll messages
  (2020) mark pitch, roll and heading valid and heave invalid, so
  `heave_m` is NaN in the `attitude` record.
- **Pressure fields can read 0.** Pressure messages (2060) have no
  validity flags. Devices that do not populate depth, sound speed or
  temperature write 0, which reads as a real value.
- **DVL validity flags reject everything** on the devices seen so far;
  the per-ping values embedded in the sonar messages are used instead.
- **No sound velocity or altitude.** Some Discover files write 0 for
  both. Sound velocity then falls back to 1500 m/s, and the
  `bottom_track` correction step measures altitude from the data
  ([Correct a line](../examples/plot_correct_a_line.rst)).
- **A recording stopped mid-ping.** Reading stops at a message cut off
  by the end of the file, with a warning, and up to two final pings
  whose sample count differs from the rest of the file are dropped.

## Reference

EdgeTech JSF file format specification, revision 2.x.
