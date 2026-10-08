# Visualization

`sidescanpy.viz` draws one result per call and returns matplotlib
`Axes` / `Figure`; nothing is saved. Requires the `[viz]` extra.

## Waterfalls, grids and tracks

```python
import matplotlib.pyplot as plt
from sidescanpy.viz import plot_waterfall

fig, ax = plt.subplots(figsize=(12, 6))
plot_waterfall(ds, x_axis="across_track", y_axis="along_track", ax=ax, show_colorbar=True)
fig.savefig("waterfall.png")
```

| x_axis | Meaning | Needs |
|---|---|---|
| `sample` (default) | sample index, port mirrored | nothing |
| `ground_range` | across-bottom distance, signed by side | `geometric` |
| `across_track` | signed across-track distance, pitch-corrected | `geometric` |

| y_axis | Meaning |
|---|---|
| `ping` (default) | ping number (`ping_offset` keeps survey-wide numbering for a slice) |
| `time` | ping timestamp |
| `along_track` | cumulative geodesic distance |

Colour limits default to the 1st–99th percentile of finite values
(`vmin_pct` / `vmax_pct`, or fixed `vmin` / `vmax`); `gamma` applies a
power-law display stretch without changing the colorbar's units. The
default colormap is `copper`; NaN cells render in a neutral colour.

Also: `plot_survey_overview` (whole survey compressed),
`plot_gridded_raster`, `plot_track_and_scanlines`,
`plot_track_with_targets`, `plot_target_position_scatter`,
`contact_sheet`, `render_target_context`. Basemaps behind georeferenced
plots need the `[maps]` extra.

## Analysis plots

One per analysis result: `plot_value_histograms`, `plot_units_distribution`,
`plot_sample_ping`, `plot_outlier_comparison`, `plot_cross_track_profile`,
`plot_nadir_zoom`, `plot_nadir_waterfall`, `plot_beam_geometry`,
`plot_usable_range`, `plot_segments_overview`, `plot_navigation_summary`,
`plot_mean_vs_range`, `plot_tvg_gain_curve`, `plot_per_ping_profiles`,
`plot_multi_frequency_comparison`, `plot_altitude_bin_comparison`,
`plot_candidate_fit`, `plot_range_decay_fit`, `plot_range_decay_residuals`,
`set_fit_window_limits`.

Multi-panel figure layouts are in `examples/figures/`.
