# Klein SDF

Klein sidescan systems record SDF (`.sdf`): a sequence of pages, each a
4-byte marker (`0xFFFFFFFF`), a 512-byte header and a body of sonar
samples. sidescanpy reads the sonar pages of the UUV-3500 layout (page
versions 3501 and 3502, low and high frequency), as written by the
Klein 4900, and converts them into one `SideScanSurvey`.

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

## Page versions

| Version | Pages | Read |
|---|---|---|
| 3501 | UUV-3500 low frequency | yes |
| 3502 | UUV-3500 high frequency | yes |
| 3500 | UUV-3500 (obsolete) | no |
| 3000, 3001 | System 3000 | no |
| others | non-sonar pages | skipped |

## Mapping to the model

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
