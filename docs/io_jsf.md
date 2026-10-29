# JSF IO

Parses Edgetech `.jsf` binary survey files. Produces typed message instances and
converts them into a `SideScanSurvey`.

## Entry points

| Function                              | Use case                                           |
| ------------------------------------- | -------------------------------------------------- |
| `read_jsf(path)`                      | Eager parse, returns full `list[JSFMessage]`.      |
| `read_jsf_iter(path)`                 | Generator over messages — for very large files.    |
| `convert(messages, source_files=...)` | Group + dispatch messages into a `SideScanSurvey`. |
| `write_subset(src, dst, n_pings=, start_ping=)` | Copy a run of pings to a new file, byte for byte. |

The reader-converter split is intentional: the reader knows about JSF binary,
the converter knows about the survey schema. Either can be swapped or tested
independently.

## Message hierarchy

```
JSFMessage (base)
├── SonarDataMessage              (per-ping sidescan data)
├── PitchRollDataMessage          (attitude: roll, pitch, heave)
├── DopplerVelocityLogDataMessage (DVL: bottom dist, water vel.)
├── PressureSensorReadingMessage  (pressure, temp, sound speed)
├── NMEAStringMessage             (parent for NMEA sentences)
│   ├── NMEARMCMessage            (lat/lon/speed/course)
│   ├── NMEAGLLMessage            (lat/lon)
│   ├── NMEAGGAMessage            (lat/lon/fix quality)
│   ├── NMEAVTGMessage            (course/speed)
│   ├── NMEAZDAMessage            (date/time)
│   ├── NMEADBTMessage            (depth below transducer)
│   └── NMEADPTMessage            (depth + offset)
└── (other types: parsed but not yet routed by converter)
```

Each subclass implements `from_bytes(header, payload)` for parsing and exposes
typed Python fields after `__post_init__`.

## Conventions

- **Timestamps.** Sonar messages: `timestamp_ns: int`. Aux messages: `timestamp:
  float` (epoch seconds). The converter handles both via type-aware dispatch.
  All output `time` coords are `datetime64[ns]`.
- **Invalid fields.** When validity flags reject a value, current message
  classes use `None`. The converter coerces to `np.nan`.
- **Field naming.** Schema-aligned suffixes (`_deg`, `_m`, `_ms`) are used on
  most newer fields. A few older fields still lack suffixes; see "Future work"
  below.

- **Layback.** `layback_m` (bytes 228-231) and `cable_out_m` (bytes 236-237,
  decimetres in the file) are carried per ping. Validity bit 13 ("position
  interpolated") sets `position_source`: set, the position is the sonar's
  with layback applied by Discover; clear, it is the last navigation fix
  received, without layback. The cable-counter validity bit is clear in the
  Discover files seen so far although the cable out is plausible; the value
  is kept.

## Known device-specific issues

- **Attitude validity flags.** `PitchRollDataMessage` (type 2020) flags follow
  the spec bit layout. Devices seen so far set the pitch, roll and heading bits
  and clear heave and yaw, so `heave_m` is NaN in the `attitude_record` while
  roll, pitch and heading are populated.
- **Pressure-stuck-at-zero.** `PressureSensorReadingMessage` has no validity
  gate; depth/sound_speed/temperature appear stuck at 0 in current sample data.
  Likely device-specific.

## Testing

- Unit tests use `Mock(spec=...)` instances. Decouples tests from binary
  parsing. See `tests/unit/io/jsf/conftest.py`.
- `tests/unit/io/test_public_fixtures.py` runs on a public EdgeTech Discover
  JSF cut (`tests/data/public/`, OGL v3) and checks it against the XTF files
  Discover wrote for the same pings.
- Integration tests run against a real fixture: `tests/data/jsf/sample.jsf`.
  Skipped by default; run with `pytest --slow`.
- Diagnostic script: `scripts/plot_jsf_aux_streams.py` for evaluating source
  quality and overlap on a real file.

## Future work

- Standardize on `timestamp_ns: int` across all message types (currently only
  `SonarDataMessage` and `PitchRollDataMessage`).
- Replace `Optional[float] = None` with `float = nan` for fields gated by
  validity flags. Removes the `_coerce` step in the converter.
- Add a validity gate to `PressureSensorReadingMessage`.
- Implement `to_bytes` for round-trip testing (currently `NotImplementedError`
  for most message types). Useful for synthesizing test fixtures.
- Add NMEA types as needed: HDT (true heading), GSV (satellite info).
- Integration test for `read_jsf_iter` once it's used in production.
