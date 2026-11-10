# Klein SDF

Klein sidescan systems record SDF (`.sdf`). sidescanpy reads the sonar
pages of the UUV-3500 layout (page versions 3501 and 3502, low and high
frequency), as written by the Klein 4900, and converts them into the
[data format](../format/index.md).

```python
from sidescanpy.io import read_and_convert, read_and_convert_segments

survey = read_and_convert(["line_001.sdf"], fmt="sdf")
for stem, survey in read_and_convert_segments("line_001.sdf"):  # split at range changes
    ...
```

`io.sdf.read_sdf(path)` returns every page as a typed object
(`SDFSonarPage` for sonar pages, `SDFUnknownPage` for the rest),
`read_sdf_iter(path)` yields them one at a time, and `io.sdf.convert`
builds the survey from a page list.

There is no public SDF example yet. If you can share a short SDF file
under an open licence, please get in touch.

## File structure

A sequence of pages, each a 4-byte marker (`0xFFFFFFFF`), a 512-byte
header and a body of sonar samples. Each page is one ping at one
frequency, and its header carries everything recorded with that ping:
position, attitude, altitude, speed and the auxiliary sensors.

## Channels and data streams

Unlike JSF, SDF has no separate navigation or sensor messages: every
stream comes from the sonar page headers, one value per page.

| Stream | SDF source | In the data format | Rate |
|---|---|---|---|
| Sonar, port and starboard, low frequency | page 3501 body | `frequencies/hz_<Hz>` (from `sonar_freq_khz`) | every low-frequency ping |
| Sonar, port and starboard, high frequency | page 3502 body | a second `frequencies/hz_<Hz>` | every high-frequency ping |
| Towfish position | `fish_latitude`, `fish_longitude` (radians) | per-ping `latitude`, `longitude`; `navigation` record | every page (held between fixes) |
| Ship position | `ship_latitude`, `ship_longitude` | used only when the towfish position is 0 | every page |
| Heading, pitch, roll | towfish heading, pitch, roll | per-ping values; `attitude` record | every page |
| Altitude | altimeter reading | `altitude_m`, `fathometer_reading_m` | every page |
| Depth | a sensor voltage | `depth_m` is NaN (see below) | every page |
| Auxiliary pitch, roll, depth, altitude sensors | aux header fields | exposed on the page objects, not converted | every page |
| Cable out | field 50 | `cable_out_m` (no layback distance is recorded) | every page |
| Sound velocity, heave | not recorded | NaN; range at 1500 m/s | none |
| DVL, other sensors | none in SDF | empty `dvl` record, no sensor records | none |

Both frequencies keep their own ping axes. `position_source` is
`towfish` where the page's towfish position is set and used, `ship`
where the reader fell back to the ship position, and `mixed` if both
occur; `sdf_layback_method` records the header's layback method (0
Pythagorean, 1 hull mount).

## Page versions

| Version | Pages | Read |
|---|---|---|
| 3501 | UUV-3500 low frequency | yes |
| 3502 | UUV-3500 high frequency | yes |
| 3500 | UUV-3500 (obsolete) | no |
| 3000, 3001 | System 3000 | no |
| others | non-sonar pages | skipped |

## Mapping to the data format

| Survey | SDF header |
|---|---|
| frequency key `hz_<Hz>` | `sonar_freq_khz` (kHz) |
| `backscatter_intensity` | port and starboard samples, low 16 bits (see below), float32 |
| `latitude`, `longitude` | towfish position (`fish_latitude`, `fish_longitude`, radians); the ship position when the towfish position is 0 |
| `heading_deg`, `pitch_deg`, `roll_deg` | towfish heading, pitch, roll |
| `altitude_m`, `fathometer_reading_m` | altimeter reading |
| `depth_m` | NaN (see below) |
| `sound_vel_ms`, `heave_m` | NaN (not recorded) |
| `sample_rate_hz` | from the first page's sample count and range, at 1500 m/s |
| `time` | header date and time, with `fseconds` as the fractional part |
| navigation and attitude records | one entry per page |

The header stores speed and sound speed in cm/s and frequency in kHz;
the page objects expose them in m/s and Hz as well (`speed_ms`,
`sound_speed_ms`, `sonar_freq_hz`). Older firmware stores the year as an
offset from 1900 (126 for 2026); years below 200 are read that way.

## Known quirks

- **Depth is a voltage.** The header's depth field is a sensor voltage,
  and its calibration lives in a separate INI file that is not read.
  `depth_m` is NaN. If you have the calibration, set `depth_m` yourself
  after converting.
- **No sound velocity.** Range is computed at 1500 m/s.
- **Bit 17 in the samples.** Samples are stored as 32-bit values with
  16 bits of amplitude. On Klein 4900 data about 0.001 % of samples also
  have bit 17 set, which shows as isolated bright pixels. The reader
  keeps the low 16 bits. This masking is empirical; the specification
  does not document the bit.
- **Pings of different lengths.** Within one file, shorter pings are
  padded with NaN to the longest, and `valid_samples` records each ping's
  true length.
- **Range changes.** When the range setting changes, the sample interval
  can change too. `read_and_convert_segments` splits the file into runs
  of constant sample interval (runs under 50 pings are dropped);
  `read_and_convert` keeps it whole and uses the first page's rate.
- **Pitch sign.** The specification available here does not state the
  sign of the pitch field. The reader passes it through, and the
  geometric step treats positive as nose up, as for JSF and XTF. If a
  recording logs nose down as positive, its along-track offsets have the
  wrong sign.
- **Aux fields with variable units.** Header fields for auxiliary
  pitch, roll, depth and altitude sensors carry whatever units the
  sensor reports and are exposed without conversion.
- **A dead channel looks like noise.** If one frequency shows only a
  noise floor and sparse spikes while the other shows the seabed, the
  reader is working and the transducer is not; check the header's
  `error_flags`.

## Reference

Klein Associates, *SDF/SDFX Data Page Definitions*.
