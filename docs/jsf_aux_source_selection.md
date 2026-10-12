# JSF Auxiliary Stream Source Selection

When a JSF file is converted to a `SideScanSurvey`, the converter must
decide which message stream populates each auxiliary record (`navigation`,
`attitude`, `dvl`, `sensors`) and, separately, which stream is most
accurate for interpolating per-ping values (`latitude`, `heading_deg`,
etc.) at processing time.

This document describes the diagnostic methodology and the resulting
decisions, so that future readers (including future-you) can audit and
extend the logic without re-deriving it.

## Why this matters

Several physical quantities appear in multiple JSF message types. For
example:

- **Latitude/longitude:** carried in NMEA RMC, GLL, GGA, and embedded
  as a per-ping field on `SonarDataMessage`.
- **Roll/pitch:** carried in `PitchRollDataMessage`, embedded on
  `SonarDataMessage`, and partially in `DopplerVelocityLogDataMessage`.
- **Depth:** carried in `PressureSensorReadingMessage`,
  `DopplerVelocityLogDataMessage`, and embedded on `SonarDataMessage`.

These sources do not always agree. Some are direct sensor readings;
others are downstream copies that lag the original by one message. Using
the right source — and interpolating it correctly onto ping times —
materially affects positional and attitude accuracy.

## Diagnostic methodology

Run the auxiliary-stream diagnostic script against a representative JSF
file:

```bash
python scripts/plot_jsf_aux_streams.py path/to/file.jsf --no-plot
```

The script extracts every value of every quantity from every available
source and prints a table with these columns per (quantity × source):

| Column | Meaning |
|---|---|
| `N` | Number of message arrivals |
| `N_valid` | Number of non-NaN values |
| `mean / std / min / max` | Distribution of values |
| `span_s` | Time span covered |
| `rate_hz` | Message arrival rate (`N / span_s`) |
| `n_changes` | Number of value transitions |
| `change_rate_hz` | Effective update rate (`n_changes / span_s`) |
| `hold_ms_p50` | Median time the value remains constant |
| `mirror_of` | If the change pattern matches another source, names that source |

The key distinction: **arrival rate** vs **change rate**. A source can
arrive at 95 Hz but only carry new information at 0.5 Hz if 99% of its
values are repeats of the previous reading.

### Mirror detection

When source A's change times closely match source B's arrival times,
A is a **mirror** of B — A re-publishes B's values at A's own (faster)
arrival rate. The diagnostic script computes this automatically and
reports `mirror_of` for each (quantity, source) pair.

A typical mirror relationship: `SonarDataMessage.latitude` mirrors
`NMEARMCMessage.latitude`. Sonar messages arrive at ~95 Hz; latitude
within those messages updates at the NMEA rate (~0.5 Hz), one message
later. The sonar's per-ping latitude is, in effect, "the most recent
NMEA fix as of this ping" — useful for the embedded value but not an
independent source of new information.

## Decision matrix — auxiliary records

This is what `convert()` writes to each aux record on the survey.

| Aux record | Source | Rationale |
|---|---|---|
| `navigation` | NMEA RMC + GLL + GGA | All three provide identical lat/lon; we collect from all three to maximize coverage. NMEA is the upstream truth for position. |
| `attitude`   | `PitchRollDataMessage` | Only direct source for roll/pitch/heave. Sonar embedded values mirror this stream. |
| `dvl`        | `DopplerVelocityLogDataMessage` | Only direct source. May be absent in some files. |
| `sensors`    | `PressureSensorReadingMessage` | Pressure-derived depth, water sound speed, water temperature. Currently the only supported sensor; extensible by `sensor_type`. |

The converter does **not** fall back to alternate sources when the
primary is empty or invalid. A field whose validity bit is clear in a
`PitchRollDataMessage` is NaN in the `attitude_record` for that
observation (heave on devices without a heave sensor, for example).
With no attitude messages at all, the record is empty but schema-valid
and the survey's `has_attitude` property reports `False`.

## Decision matrix — per-ping interpolation

This is the Phase 3 work. It applies after the aux records are built,
when we want each `(ping, *)` to have an accurate `latitude`,
`heading_deg`, etc. — rather than the latched-previous-fix values that
sonar embedded provides.

The framework:

1. Classify each per-ping field's `sonar_embedded` value as either a
   **live sensor reading** or a **mirror of an aux stream**, using the
   diagnostic script's `mirror_of` column.
2. If it's a mirror, interpolate the upstream aux stream onto ping times.
3. If it's a live sensor, leave it alone.
4. If the upstream aux stream is empty or all-NaN, fall back to
   sonar_embedded with a warning.

### Per-field decisions (sample.jsf)

| Per-ping field | Sonar embedded class | Interpolation source | Fallback |
|---|---|---|---|
| `latitude` | mirror of NMEA | NMEA aux record | sonar_embedded |
| `longitude` | mirror of NMEA | NMEA aux record | sonar_embedded |
| `heading_deg` | mirror of attitude | attitude aux record | sonar_embedded |
| `roll_deg` | mirror of attitude | attitude aux record | sonar_embedded |
| `pitch_deg` | mirror of attitude | attitude aux record | sonar_embedded |
| `heave_m` | not in sonar | attitude aux record | NaN |
| `altitude_m` | live sensor (38 Hz) | sonar_embedded | DVL aux record (cross-check) |
| `depth_m` | live sensor (6 Hz) | sonar_embedded | pressure aux record (cross-check) |
| `sound_vel_ms` | live sensor | sonar_embedded | DVL or pressure |
| `water_temp_C` | live sensor | sonar_embedded | DVL or pressure |
| `course` | mirror of NMEA | NMEA RMC/VTG | sonar_embedded |
| `speed` | mirror of NMEA | NMEA RMC/VTG | sonar_embedded |

These are the *defaults*. Different surveys may need overrides — e.g.
if a particular file has bad NMEA fixes but good sonar embedded
positions, the user should be able to override the source for `latitude`
specifically.

## Worked example: sample.jsf

Running the diagnostic on `tests/data/jsf/sample.jsf` produces the
following key observations.

### Position is mirrored

Latitude and longitude appear in four sources. NMEA arrives at 0.5 Hz;
sonar_embedded arrives at 95 Hz. Both have effective change rates of
~0.5 Hz with hold times of ~2000 ms. **Sonar embedded position is just
NMEA latched.** Use NMEA for interpolation.

### Attitude is mirrored

Roll, pitch, and heading on sonar_embedded change at 10.3 Hz with
84 ms holds. `PitchRollDataMessage` arrives at 10 Hz. The match is
exact — sonar embedded attitude is the previous attitude reading.
**Use attitude for interpolation.**

### Altitude and depth are live

Sonar embedded altitude updates at 38 Hz with 41 ms holds. No DVL
altitude rate matches this — it's a true 38 Hz altimeter signal.
Sonar embedded depth updates at 6 Hz with 166 ms holds, also a real
sensor signal (likely fed by the pressure transducer at 7 Hz, but not
a direct mirror). **Use sonar_embedded for both.**

### Sound speed and temperature are unavailable

Stuck at 0 across all sources in this file. Either the device doesn't
populate these fields, or they're behind validity flags we don't check.
Survey-wide values would be required from external metadata
(CTD cast, climatology). **Per-ping values stay at 0 / NaN until a real
source is identified.**

## Known issues

### Attitude validity flags

`PitchRollDataMessage.parse_validity_flags()` follows the spec bit
layout. On the devices seen so far, pitch, roll and heading bits are
set and heave and yaw bits are clear, so `heave_m` is NaN.

### Timing vs value alignment

The `mirror_of` column identifies sources whose *change timing* aligns
with another source's *arrival times*. It does **not** verify that the
values agree.

In `sample.jsf`, for example, the diagnostic reports:

depth  sonar_embedded  →  mirror_of: pressure

But pressure depth is stuck at 0 across the file, while sonar_embedded
depth ranges from 3.5 to 6.0 m. The timing alignment is real — sonar
embedded depth happens to update at the same cadence pressure messages
arrive — but the values come from a different sensor entirely (the
sonar's internal depth fix, embedded into the ping message).

When choosing an interpolation source:
1. Check `mirror_of` for the timing relationship.
2. Cross-check `mean` and `std` between the candidate sources to confirm
   they agree on values.
3. If timing matches but values diverge, the apparent mirror is a
   coincidence of update cadence, not a real upstream source.
   
### Pressure sensor stuck at 0

`PressureSensorReadingMessage` has no validity flag check (unlike
sonar/attitude/DVL). If the device doesn't populate a field, it
arrives as 0 and is treated as a valid reading. Currently affects:

- `depth_m`
- `sound_speed_ms`
- `temperature_C`

Investigate whether the device intends 0 as "not provided" and add a
validity gate accordingly.

See `TODO` in `sidescanpy/io/jsf/messages.py`.

## Extending this analysis to new files

For each new survey or new device:

1. Run `python scripts/plot_jsf_aux_streams.py <file.jsf> --no-plot`.
2. For each quantity, check the `change_rate_hz` and `mirror_of` columns.
3. If a `sonar_embedded` row reports `mirror_of: <other_source>`, use
   that other source as the interpolation seed.
4. If `change_rate_hz` significantly exceeds any aux source's
   `rate_hz`, sonar_embedded is the live sensor — use it directly.
5. Update the per-field decisions table above if the file changes the
   default selections.

## References

- JSF format spec: Edgetech 2.x JSF File Format Specification
- `sidescanpy/io/jsf/messages.py` — message parsing
- `sidescanpy/io/jsf/converter.py` — `_build_*_record` functions
- `scripts/plot_jsf_aux_streams.py` — diagnostic script

---
*Last revised 2026 by rshomberg@uri.edu.*
