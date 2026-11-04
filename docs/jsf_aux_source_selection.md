# JSF navigation and attitude sources

A JSF file often carries the same quantity in several messages.
Position comes in NMEA RMC, GLL and GGA sentences and is also copied into
the header of every sonar message. Roll and pitch come in pitch/roll
messages (type 2020) and again in the sonar header. Depth comes from the
pressure sensor, the DVL and the sonar header. This page says which
source sidescanpy uses for each value, and how to check that choice on
your own files.

## Arrival rate is not update rate

Most of the copies are sample-and-hold. The sonar header repeats the
latest fix it has seen, so a value that arrives with every ping (tens of
times a second) may change only when the GPS updates (about once a
second or slower). Interpolating the repeated values as if each were a
new fix puts pings in stair-steps along the track.

On one EdgeTech towfish recording, for example:

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

## Aux records

The converter fills the survey's aux records from these messages:

| Record | Source | Notes |
|---|---|---|
| `navigation` | NMEA RMC, GLL and GGA | positions from all three are merged |
| `attitude` | pitch/roll (2020) | heave is usually flagged invalid and is NaN |
| `dvl` | DVL (2080) | empty when the file has no DVL messages |
| `sensors["pressure"]` | pressure sensor (2060) | depth, sound speed, temperature; no validity flags, so unpopulated fields read 0 |

A record with no messages is empty but valid, and the survey's
`has_navigation`, `has_attitude` and `has_dvl` report `False`. The
converter does not fill one record from another source.

## Per-ping values

The `interpolate_aux` correction step replaces the per-ping copies with
values interpolated from the aux records at each ping time, after
reducing each record to its real updates (see
[Processing](processing.md#navigation-interpolation)):

| Per-ping field | Interpolated from | When the record is empty |
|---|---|---|
| `latitude`, `longitude` | `navigation` | the sonar-header values, reduced the same way |
| `heading_deg`, `roll_deg`, `pitch_deg`, `heave_m` | `attitude` | the sonar-header values, reduced the same way |
| `altitude_m`, `depth_m`, `sound_vel_ms` | not interpolated | the sonar-header values are kept as recorded |

These are the defaults in `processing.DEFAULT_FIELD_SOURCES`. The step's
`fields` parameter interpolates a subset, or points a field at another
record; `interpolate_onto_channel` interpolates any dataset with a `time`
coordinate, such as a separate GPS log.

## Checking a new file

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
schedule. Compare the values too.
