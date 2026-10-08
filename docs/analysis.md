# Analysis

`sidescanpy.analysis` measures a dataset. Each function takes data and
returns a dataclass with a `to_dict()`; none of them writes files, plots
or decides what to do next. Plots for each result are in
`sidescanpy.viz` ([Visualization](viz.md)); `sidescan-analyze investigate`
and `examples/investigate_dataset.py` run them all on one file and write
a report.

| Function | Answers |
|---|---|
| `characterize_values` | data type, shape and distribution of a variable |
| `infer_units` | raw counts, dB, volts or normalized, from the value range |
| `compare_outlier_rejection` | distribution before and after `processing.reject_range_outliers` |
| `analyze_nadir` | nadir type (gap, bright spike, noisy, clean), altitude sample and first bottom return |
| `estimate_beam_geometry` | near-edge and far-edge beam angles, beam width |
| `estimate_usable_range` | where the signal falls into the noise floor |
| `fit_range_decay` | which loss terms are already compensated in the data: `none`, `spreading_only`, `absorption_only`, `full`, or `ambiguous` |
| `fit_tvg_curve` | fits of the four candidate loss models to a binned dB-vs-range curve |
| `compare_before_after` | raw and TVG-corrected backscatter side by side |
| `bin_by_altitude` | pings split into equal-count altitude bins |
| `summarize_navigation` | per-field navigation/attitude statistics and a heading-convention check |
| `measure_ping_geometry` | median ping spacing, swath width and sample interval |

Analysis that needs the uncorrected signal recovers it with
`processing.uncorrected_backscatter`, so it works on corrected files.

## Reading the results

- A best fit in `fit_range_decay` is a hypothesis. Apply the winning
  correction and check that the corrected data is flat over the fit
  window before trusting it, especially when the window is short.
- A smooth per-ping boundary is not proof of a correct one: a detector
  stuck at a floor value is also smooth. Compare port and starboard, or
  two frequencies, when a result looks too clean.
