# XTF format

`.xtf` (eXtended Triton Format) is the most common vendor-neutral
sidescan format: Klein, EdgeTech and others can record or export it. The
reader follows the specification's revision 42 (Exail) and has been
checked against EdgeTech 4200 files recorded by Isis. Only sonar packets
are read; other packet types are counted and skipped.

```python
from sidescanpy.io import read_and_convert
from sidescanpy.io.xtf import convert, read_xtf

survey = read_and_convert([path], "xtf")      # or, step by step:
survey = convert(read_xtf(path))
```

`read_xtf_header`, `iter_packets`, `count_packets` and `read_xtf_pings`
give the file header, the raw packets and the recorded pings (samples in
their recorded dtype) for inspection.

## File structure

A 1024-byte file header (system, recording program, sonar name,
navigation units, and one 128-byte `ChanInfo` per channel; more than six
channels spill into further 1024-byte blocks), then packets. Every packet
starts with the magic `0xFACE`, a header type and its total byte count,
which the reader uses to step from packet to packet. A sonar packet is a
256-byte ping header followed, per channel, by a 64-byte channel header
and that channel's samples.

## Mapping to the standard model

| Survey | XTF source |
|---|---|
| frequency key `hz_<Hz>` | `ChanInfo.Frequency` (kHz) of a port/starboard pair |
| `backscatter_intensity` (ping, channel, sample) | port (`TypeOfChannel` 1) and starboard (2) samples, float32 |
| `valid_samples` | per-ping `NumSamples`; padding beyond it is NaN |
| `latitude`, `longitude` | `SensorYcoordinate`, `SensorXcoordinate` (degrees, `NavUnits` 3) |
| `heading_deg`, `pitch_deg`, `roll_deg`, `heave_m` | `SensorHeading`, `SensorPitch` (nose up +), `SensorRoll` (port up +), `Heave` |
| `depth_m` | `SensorDepth` |
| `altitude_m`, `fathometer_reading_m` | `SensorPrimaryAltitude`; 0 is NaN |
| `sound_vel_ms` | `SoundVelocity`, doubled when below 1000 m/s (see below) |
| `sample_rate_hz` | `NumSamples / TimeDuration` |
| `time` | `Year` … `Second` + `HSeconds` (10 ms resolution) |
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
- **Range changes.** The model has one `sample_rate_hz` per frequency.
  When the range setting changes within a file while the sample count
  stays fixed, the most common rate is used, the other pings are counted
  in `xtf_pings_other_sample_rate` and a WARNING is logged: their ranges
  are scaled wrongly by the ratio of the rates.
- **Pings with fewer channels than declared.** Some EdgeTech files begin
  with pings that hold only the low-frequency pair while declaring all
  four channels. The reader walks channel blocks by the packet byte
  count; those pings are left out of the high frequency (which keeps its
  own ping axis) and counted in `xtf_pings_absent`.
- **Projected coordinates** (`NavUnits` 0) are not supported and raise
  `NotImplementedError`.
