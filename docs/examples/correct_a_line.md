# Correct a line

`correct_survey` applies an ordered list of correction steps. Each step
is optional, can be replaced by your own function, and is recorded with
its parameters in the dataset's `processing_history`, so a corrected
file says how it was made.

This JNCC line records no altitude above the seabed, which the
slant-to-ground-range step needs. The `bottom_track` step picks it from
the first seabed return in each ping (17.5 m here) before `geometric`
runs.

![Raw and corrected waterfall](../assets/gallery/correct_a_line.png)

Left: raw amplitude against slant range. Right: after the bottom track,
time-varying gain (spreading and absorption losses) and slant-to-ground
range, against across-track and along-track distance in metres. The
water column is gone, and the far range is no longer darker than the near
range.

```python
--8<-- "examples/gallery/correct_a_line.py"
```

The default TVG uses 30 log R spreading and Francois-Garrison
absorption. Whether a sonar already applied some gain in hardware varies;
`analysis.fit_range_decay` measures what a dataset still needs (see
`examples/tvg_calibration.py`).

Data: JNCC Wight-Barfleur Reef, Open Government Licence v3.0 (see
[the example data](index.md#sources-and-licences)).
