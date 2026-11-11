# Analysis

`sidescanpy.analysis` measures a dataset. Each function takes data and
returns a dataclass with a `to_dict()`; none of them writes files, plots
or decides what to do next. Plots for each result are in
`sidescanpy.viz` ([Visualization](viz.md)). `sidescan-analyze investigate`
runs them all on one file and writes a report, and
[Investigating an unknown survey](examples/plot_investigate_a_survey.rst)
walks through them on a public recording.

| Function | Answers |
|---|---|
| `characterize_values` | data type, shape and distribution of a variable |
| `infer_units` | raw counts, dB, volts or normalized, from the value range |
| `compare_outlier_rejection` | distribution before and after `processing.reject_range_outliers` |
| `analyze_nadir` | nadir type (gap, bright spike, noisy, clean), altitude sample and first bottom return |
| `estimate_beam_geometry` | near-edge and far-edge beam angles, beam width |
| `estimate_usable_range` | where the signal falls into the noise floor (the noise knee by default, or the SNR rule) |
| `find_noise_knee` | peak, noise floor and noise knee of a level-vs-range profile |
| `fit_tvg_models` | the TVG to apply: which spreading law (none, 20, 30 or 40 log R, or free) and whether absorption, fitted to the median level |
| `check_tvg` | how flat a TVG recipe leaves the median level over a window |
| `fit_range_decay`, `fit_tvg_curve` | deprecated (removal in 0.12): the earlier candidate classifier; use `fit_tvg_models` |
| `compare_before_after` | raw and TVG-corrected backscatter side by side |
| `bin_by_altitude` | pings split into equal-count altitude bins |
| `summarize_navigation` | per-field navigation/attitude statistics and a heading-convention check |
| `measure_ping_geometry` | median ping spacing, swath width and sample interval |
| `detect_shadow_proposals` | object proposal boxes (native ping, sample, side) from acoustic shadows and the highlight in front of them, with sizes, contrast, estimated height and a score |

Analysis that needs the uncorrected signal recovers it with
`processing.uncorrected_backscatter`, so it works on corrected files.

## Fitting the TVG

`fit_tvg_models(ds)` works on raw data (undo a recorded TVG with
`processing.invert` first) with slant ranges:

1. **Profile.** All samples past the median altitude, both sides pooled,
   binned by slant range in 1 m bins; the median of each bin's linear
   amplitude, in dB re 1 recorded unit. Integer samples (many recorders
   store whole numbers) are first spread over their unit interval by a
   fixed random offset, so the medians do not stick to whole numbers.
2. **Window.** From 5 m past the profile's peak (inside it the beam
   pattern dominates; the peak is searched in the near half of the
   range) to the noise knee, the first range past the peak within 3 dB
   of the lowest level. Beyond the knee the level is noise. When the
   level never falls to a floor, as on data the recorder has already
   levelled, there is no knee: the window runs to the end of the record
   and `window_source` says `"end_of_record"`. `window_m=(min, max)`
   sets it by hand.
3. **Models.** `L(R) = L0 - n log10 R - 2 alpha R` with `n` = 0, 20, 30
   or 40 and absorption (Francois-Garrison at the channel's center
   frequency) off or on: eight models, from "flat" (no TVG) and
   "absorption only" to "40 log R + abs". Each has its constant `L0`
   fitted by least squares; they are ranked by RMS residual in the
   window. A free `n` is fitted too, with and without absorption.
4. **Sides.** Each side is fitted alone on the same window;
   `sides_disagree` flags sides whose levels differ by more than 15 dB
   or whose shapes want different models.

The best model is the TVG to apply: `fit.tvg_params` holds its
`spreading_loss_db` and `absorption_db_per_m` for `apply_tvg` or the
`tvg` step. `fit.tvg_status` names what the recorder has already
removed, derived from that model: `none` (n and absorption still in the
data), `absorption_only` (n only), `spreading_only` (absorption only),
`full` (flat).

```python
from sidescanpy.analysis import check_tvg, fit_tvg_models
from sidescanpy.viz import plot_tvg_fit

fit = fit_tvg_models(ds)
check = check_tvg(ds, fit.tvg_params, fit.window_m)
print(fit.best, fit.best_model.rms_db, fit.free_n, check.spread_db, check.trend_db_per_10m)
fig, _ = plot_tvg_fit(fit, check)
```

`check_tvg` crops the data at the window's end, applies the recipe with
`apply_tvg`, and measures the corrected median level over the window:
its spread (standard deviation), trend (dB per 10 m) and end-to-end
change. Pass any recipe, such as a config's `tvg` block, to compare it
with the fit.

## Reading the results

- A best fit is a hypothesis. Apply it with `check_tvg` and check that
  the corrected level is flat over the same window before trusting it.
- Over short windows (under about 20 m) 20 log R with absorption, 30 log R
  and 40 log R differ by tenths of a dB, so the label means little; read
  the free `n` and the runner-up. For a survey, compare several lines (or
  take the median free `n`) and pick one model rather than one per line.
- `low_contrast` (the level falls less than 6 dB from peak to floor) and
  an `"end_of_record"` window mean there is little range dependence
  left: the recorder levelled it, or there is little seabed signal.
- The altitude is trusted: samples before it are dropped. A WARNING
  says when the profile peak lies far past the altitude; a recorded
  altitude that is too large hides the real peak and is not detected.
- A smooth per-ping boundary is not proof of a correct one: a detector
  stuck at a floor value is also smooth. Compare port and starboard, or
  two frequencies, when a result looks too clean.
