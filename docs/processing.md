## File 6 — `docs/processing.md`

```markdown
# Processing

Processing functions transform validated datasets through the pipeline:

```
raw → radiometric → geometric → georeferenced
```

Each function asserts its expected input `processing_level` and updates
the attribute on output. `processing_history` is appended on every
transform. See [`docs/core.md`](core.md) for the schema conventions.

## Functions

| Function | Purpose | Input → Output level |
|---|---|---|
| `interpolate_aux_onto_pings` | Replace per-ping fields with interpolated aux values | (any) → (unchanged) |
| `interpolate_onto_channel` | Lower-level: interpolate any external dataset onto pings | (any) → (unchanged) |
| `apply_tvg` | Time-varying gain correction | raw → radiometric |
| `compute_geometric_ranges` | Compute per-sample slant_range_m and ground_range_m | (any) → (unchanged) |

`interpolate_*` functions don't change `processing_level` because
interpolation is metadata enrichment, not a processing step in the
radiometric/geometric sense.

## `interpolate_aux_onto_pings`

Replaces per-ping fields (latitude, longitude, heading, roll, pitch,
heave) with values interpolated from the survey's aux records onto each
ping's timestamp.

The default field-source mapping is in `DEFAULT_FIELD_SOURCES`:

| Per-ping field | Default source |
|---|---|
| `latitude` | `navigation.latitude` |
| `longitude` | `navigation.longitude` |
| `heading_deg` | `attitude.heading_deg` |
| `roll_deg` | `attitude.roll_deg` |
| `pitch_deg` | `attitude.pitch_deg` |
| `heave_m` | `attitude.heave_m` |

`altitude_m`, `depth_m`, `sound_vel_ms`, `water_temp_C` are *not*
interpolated by default — these are live sensor readings on
SonarDataMessage, not mirrors of an aux stream. See
[`docs/jsf_aux_source_selection.md`](jsf_aux_source_selection.md) for
the analysis behind these decisions.

### Key parameters

- `fields={...}` — override the default mapping.
- `method` — `"linear"` (default), `"nearest"`, etc.
- `extrapolation` — `"clamp"` (default, holds boundary values),
  `"linear"`, or `"nan"`.
- `drop_invalid` — drop NaN values from the source before interpolating
  (default `True`).
- `force` — interpolate even when the source aux is empty (default
  `False`; the field is left unchanged with a warning otherwise).
- `in_place` — mutate the survey in place (default `False`).

### External-source variant

Users with an externally captured GPS log (or any `xr.Dataset` with a
`time` coord) can use `interpolate_onto_channel` directly:

```python
my_external_nav = xr.open_dataset("external_nav.nc")
new_channel = interpolate_onto_channel(
    survey.get_frequency("hz_1495000"),
    source=my_external_nav,
    field_map={"latitude": "lat", "longitude": "lon"},
)
```

This bypasses the survey's aux records entirely.

## `apply_tvg`

Applies time-varying gain to compensate for two propagation losses:

1. **Spreading loss.** Sound spreads in 3D, losing energy with range (in dB: `20
   * log10(R)`).
2. **Absorption.** Water absorbs acoustic energy along the path (in dB: `2 *
   alpha * R` for two-way).

Raw backscatter is converted from linear amplitude (typically 0-100ish) to dB:
`20 * log10(clip(value, 1e-6))`. The 1e-6 floor prevents `log(0)` and maps zeros
to ~-120 dB. TVG gain is then added.

Output is in dB with no upper or lower clipping. Water column samples typically
land in the -100 to -50 dB range; strong returns near nadir in the +20 to +40 dB
range.

`backscatter_raw` is preserved unchanged across the call. Only
`backscatter_intensity` is updated.

### `TVGParams`

```python
@dataclass
class TVGParams:
    spreading_loss_db: float = 20.0
    absorption_db_per_m: float | None = None  # frequency default if None
    min_range_m: float = 0.5  # range clip for sample 0
```

### `absorption_for_frequency`

Frequency-dependent absorption defaults for typical coastal seawater (T=10°C,
S=35 ppt, 10 m depth):

| Frequency | Absorption |
|---|---|
| < 400 kHz | 0.05 dB/m |
| 400 kHz – 1 MHz | 0.10 dB/m |
| ≥ 1 MHz | 0.20 dB/m |

These are reasonable defaults from the Francois-Garrison formula. For precise
values, compute using full F-G with CTD data and pass via
`TVGParams(absorption_db_per_m=...)`.

## `compute_geometric_ranges`

Attaches per-sample `slant_range_m` and `ground_range_m` data variables
to the dataset. Lossless: backscatter is preserved at original sample
positions. Resampling onto a regular grid for plotting or mosaicking is
left to downstream tools (`xarray.interp`, `rioxarray`, etc.) on demand.

### Math

Slant range: `R_slant[ping, channel, sample] = sample_idx * sound_vel[ping] / (2 * sample_rate)`

Ground range (flat-bottom assumption):


$R_ground = sqrt(R_slant^2 - altitude^2)$   when R_slant > altitude
 = NaN                              when R_slant <= altitude

The water-column samples (where slant ≤ altitude — sound hasn't
reached the bottom yet) are NaN. This is the standard nadir gap.

### NaN handling

- NaN `altitude_m` for a ping → all `ground_range_m` for that ping is NaN.
- NaN `sound_vel_ms` for a ping → falls back to 1500 m/s with a warning.

### Notes

- Both port and starboard channels share the same per-ping altitude
  (one transducer assembly).
- Sound velocity is per-ping. Per-sample variation through the water
  column is not modeled.
- `processing_level` is not modified — the operation is additive.
- 
## Future work

- **Geometric correction.** Slant-range to ground-range calculation using per-ping
  altitude. The next major processing step. Will land as
  `processing/geometric.py::calculate_slant_to_ground_range`.
- **Per-ping sound velocity in TVG.** Currently `apply_tvg` uses the mean across
  all pings for the slant-range calculation. Per-ping is more accurate when
  crossing thermoclines. Defer until geometric correction needs the same
  per-ping range logic.
- **Linear-vs-dB confirmation for TVG input.** The current implementation
  assumes raw backscatter is in linear amplitude. A calibration test against
  known-good data would close this assumption.
- **Lazy/dask-backed processing.** Verify `apply_tvg` and
  `interpolate_aux_onto_pings` operate correctly on dask-backed inputs. The math
  is dask-friendly; needs explicit integration testing.