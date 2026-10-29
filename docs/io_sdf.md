# Klein SDF format

`.sdf` files are produced by Klein side-scan sonar systems (Klein 4900
and other UUV-3500-series). The format is binary, big-endian-aware, and
documented in the Klein "SDF/SDFX Data Page Definitions" spec.

## File structure

A Klein SDF file is a sequence of pages. Each page contains:

```
[4 bytes: marker (always 0xFFFFFFFF)]
[512 bytes: typed header]
[variable bytes: body — sonar samples or reserved data]
```

Pages can be of different types, identified by `page_version`:

| Version | Description |
|---|---|
| 3501 | UUV-3500 Low Frequency sonar data |
| 3502 | UUV-3500 High Frequency sonar data |
| 3000-3001 | System 3000 (older) |
| 3500 | UUV-3500 (obsolete) |
| Other | Various non-sonar page types |

The reader yields both sonar pages (`SDFSonarPage`) and non-sonar pages
(`SDFUnknownPage`). The converter filters to sonar pages only.

## Header structure

The 512-byte header has 128 uint32-equivalent slots covering 100+ named
fields plus 26 reserved uint32 slots. See `SDFPageHeader` in
`sidescanpy.io.sdf.pages` for the full field list.

Key fields:

| Field | Type | Notes |
|---|---|---|
| `page_size` | U32 | Bytes from after marker to end of page |
| `page_version` | U32 | Page type identifier |
| `ping_number` | U32 | Increments per ping |
| `num_samples` | U32 | Sample count per channel |
| `range_m` | U32 | Sonar range setting (meters) |
| `speed_cms` | U32 | Towfish speed (cm/s); use `.speed_ms` for m/s |
| `heading_deg`, `pitch_deg`, `roll_deg` | float32 | Towfish attitude |
| `altitude_m` | float32 | Altimeter reading (meters) |
| `depth_volts` | float32 | Depth sensor (Volts; needs calibration) |
| `ship_latitude_rad`, `ship_longitude_rad` | double | Ship GPS in radians |
| `fish_latitude_rad`, `fish_longitude_rad` | double | Towfish position in radians |
| `sonar_freq_khz` | U32 | Operating frequency (kHz); use `.sonar_freq_hz` for Hz |
| `_reserved` | tuple[U32; 26] | Forward-compatibility slots |

## Klein conventions

- **Latitude/longitude in radians**: All position fields are stored as
  float64 radians. Use the `.ship_latitude_deg`, `.fish_latitude_deg`
  etc. properties for degrees.

- **Speed in cm/s**: `speed_cms` field is centimeters per second.
  Use `.speed_ms` for meters per second.

- **Sound speed in cm/s**: Same conversion. Use `.sound_speed_ms`.

- **Frequency in kHz**: `sonar_freq_khz` is kilohertz. Use
  `.sonar_freq_hz` for hertz.

- **Year handling**: Older Klein firmware stores year as offset from
  1900 (e.g. 126 means 2026). Newer firmware uses 4-digit years (2026).
  The dataclass auto-detects via `_year < 200` check.

- **Time precision**: Klein records `hSecond` (hundredths) as field 24
  for backward compatibility, but `fseconds` (field 51) is the
  authoritative fractional component. The dataclass uses `fseconds`
  for timestamp computation.

## Sample data masking

Klein stores sample amplitudes as uint32 even though only 16 bits of
amplitude are used. Bit 17 is occasionally set (~0.001% of samples in
observed Klein 4900 data) as undocumented metadata.

The reader masks samples to the low 16 bits via `& 0xFFFF`. This
removes spurious bright pixels in waterfall visualizations. Masking is
logged at DEBUG level.

If you need access to the raw uint32 (including the metadata bits), use
`SDFPageHeader.from_bytes()` and `_parse_sonar_body()` directly to read
without masking.

## Known limitations

### Depth sensor calibration

The Klein `depth` field is stored in Volts and requires per-survey
calibration data from the INI file (which we don't parse). Surveys
converted from SDF have `depth_m` set to NaN. Users with calibration
data should override the `depth_m` variable manually.

### Sound velocity

Klein doesn't record sound velocity in the page header. Surveys have
`sound_vel_ms` set to NaN. The pipeline falls back to 1500 m/s in
`compute_geometric_ranges`.

### Pitch sign

The SDF notes available here do not state the sign of `pitch_deg` (field
29, from the towfish compass). The reader passes it through unchanged and
`compute_geometric_ranges` treats positive as nose up, as for JSF and XTF.
If a recording's convention is nose down positive, its along-track offsets
have the wrong sign.

### Aux fields with variable units

Fields 46-49 (`aux_pitch`, `aux_roll`, `aux_depth`, `aux_altitude`)
are described in the spec as "variable units" — they're carry-through
fields from auxiliary sensors that may report in degrees, meters, or
sensor-specific units. The dataclass exposes them as float32 values
without unit conversion.

### 455 kHz channel reliability

In one test survey, the 455 kHz transducer was non-functional;
data appeared as noise floor + sparse spikes. This is a hardware
condition, not a parsing issue. If you see similar patterns, check:
- The transducer hardware
- The `error_flags` field in the header for compass/sensor errors
- Whether 900 kHz data shows clear seafloor returns (confirming the
  reader is parsing correctly)

## See also

- Klein SDF/SDFX Data Page Definitions specification
- `tests/integration/test_sdf_to_survey.py` for end-to-end examples
- `scripts/batch_convert_and_plot.py` for batch processing workflow