# Building a processing workflow

sidescanpy is a library: it converts, corrects, grids and exports, but it
does not decide what your survey needs. That decision is made by looking
at the data, so the workflow lives in your own project, as scripts or
notebooks that call the library. This page describes a layout that works
well for real surveys, and the template in
[`examples/project_template/`](https://github.com/rshomberg-research/sidescanpy/tree/main/examples/project_template)
implements it.

## One script per step

```
01_convert.py   raw .jsf/.sdf/.xtf -> .ss.h5, target catalog -> targets.json
02_analyze.py   measure the uncorrected data (usable range, TVG decay)
03_correct.py   apply the corrections chosen from step 2
04_export.py    waterfalls, track, grids, GeoTIFF
05_dataset.py   COCO/YOLO training dataset
```

Run them one at a time. Several steps take a long time on real data, and
each one's output decides the next one's settings: the decay fit in step
2 tells you which TVG terms to apply in step 3; the corrected overview
from step 3 tells you whether the export settings in step 4 make sense.
A single "run everything" driver hides exactly the moments you need to
look.

Each step:

- reads the same survey config (`config.yaml`);
- writes its outputs under one output root, in its own folder;
- writes QA figures next to them;
- writes a report (`reports/<step>.md` and `.json`) with the numbers it
  measured, the figures to look at, anything that needs checking, and the
  command for the next step;
- logs to the console and to `logs/<step>_<timestamp>.log`.

## One config per survey

Everything survey-specific lives in one YAML file: input and output
paths, the frequencies to process, the TVG recipe, export settings and
the dataset spec. A second survey gets its own copy, and nothing in the
scripts changes.

```yaml
correct:
  steps: [tvg, geometric, georef]
  chunk_pings: 2000
  params:
    tvg: {spreading_loss_db: 30.0}
  frequency_params:            # per-frequency overrides
    hz_1495000: {tvg: {absorption_db_per_m: 0.2}}
    hz_547500:  {tvg: {absorption_db_per_m: 0.1}}

dataset:                       # a DatasetSpec; unknown keys are an error
  styles: [chip, dual-swath]
  render_level: geometric
  size_px: 300
```

Recipes do not transfer between surveys, even from the same sonar. Run
the analysis step on every new survey and check its report before
correcting.

## What the library gives each step

| Step | Library calls |
|---|---|
| convert | `io.read_and_convert`, `processing.crop_far_range_noise`, `io.hdf5.write_survey`, `io.read_target_catalog`, `targets.correlate` (Klein positions by time) |
| analyze | `analysis.estimate_usable_range`, `analysis.fit_range_decay`, `processing.invert` (to analyze already-corrected files) |
| correct | `processing.correction_steps`, `processing.correct_file(..., frequency_params=, chunk_pings=)` |
| export | `io.hdf5.concat_survey_frequency`, `processing.ping_windows`, `targets.correlate.correlate_catalog`, `targets.waterfall.waterfall_boxes`, `grids.grid_channel`, `grids.to_geotiff`, `viz.*` |
| dataset | `export.ml.DatasetSpec`, `export.ml.build_target_dataset` |

## Iterating

Corrections are recorded in each file's `processing_history` and skipped
when already applied, so re-running step 3 only does what is missing. To
change a recipe, re-correct with `force=True`: radiometric steps are
inverted from their recorded parameters first, so new parameters replace
the old ones instead of stacking. `processing.invert(ds)` gives the
uncorrected signal back at any time; no raw copy is stored.

For work the steps don't cover (cutting out a bad stretch, a different
nadir mask, your own gain curve), pass your own function as a step:

```python
from sidescanpy.processing import Step, correct

def drop_near_range(ds, max_sample=50):
    return ds.isel(sample=slice(max_sample, None))

ds = correct(ds, [Step("drop_near_range", drop_near_range), "tvg"],
             params={"drop_near_range": {"max_sample": 80}})
```

It is recorded in history like a built-in step.

## Notebooks

The same calls work interactively. A notebook is a good place to decide
a recipe; the step scripts are a good place to apply it to a whole
survey once decided, repeatably.
