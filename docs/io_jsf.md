# JSF

EdgeTech sonars and the Discover acquisition software record JSF
(`.jsf`): a stream of messages, each a 16-byte header followed by a
typed payload. sidescanpy reads the sidescan sonar data messages and the
navigation, attitude and sensor messages around them, and converts them
into one `SideScanSurvey`.

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

## Messages read

| Message | Type | Becomes |
|---|---|---|
| Sonar data | 80 | backscatter for one channel of one ping, plus the per-ping position, attitude, altitude, depth and sound velocity embedded in its header |
| NMEA strings (RMC, GLL, GGA, VTG, ZDA, DBT, DPT) | 2002 | the `navigation` record (RMC, GLL and GGA positions) |
| Pitch/roll data | 2020 | the `attitude` record |
| Pressure sensor reading | 2060 | the `pressure` sensor record |
| Doppler velocity log | 2080 | the `dvl` record |

Other message types are either parsed into a typed object and not
converted (system information, navigation offsets, file timestamps,
situation, cable counter) or skipped, with one summary warning per file
listing the unsupported types.

## Mapping to the model

- **Frequencies.** Sonar messages are grouped by their transmit start
  and end frequency, one `hz_<Hz>` dataset per group. Channel 0 of a
  subsystem is port and channel 1 starboard.
- **Pairing.** Port and starboard messages are paired by ping number.
  A ping with only one side is kept with the other side NaN by default
  (`repair_mode="nan_pad"`); `"drop"` drops it and `"abort"` raises.
- **Samples.** Envelope samples are scaled by the message's weighting
  factor (2^-N) and stored as float32 `backscatter_intensity`. The per-ping gain
  factor is read but not applied (see [Limitations](limitations.md)).
- **Time.** Ping time is the sonar message timestamp in nanoseconds;
  aux messages carry their own timestamps. All `time` coordinates are
  `datetime64[ns]`.
- **Invalid values.** Fields whose validity flag is clear become NaN.

[JSF navigation and attitude sources](jsf_aux_source_selection.md)
explains which message each per-ping value is taken from and why.
- **Layback.** `layback_m` (bytes 228-231) and `cable_out_m` (bytes
  236-237, decimetres in the file) are carried per ping. Validity bit 13
  ("position interpolated") sets `position_source`: set, the position is
  the sonar's with layback applied by Discover; clear, it is the last
  navigation fix received, without layback. The cable-counter validity
  bit is clear in the Discover files seen so far although the cable out
  is plausible; the value is kept.

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
  ([Correct a line](examples/plot_correct_a_line.rst)).

## Reference

EdgeTech JSF file format specification, revision 2.x.
