# Project template

A starting point for processing your own surveys with sidescanpy: one
script per step, one YAML config per survey, a report after every step.
Copy this directory into your project and edit it; it is yours, not part
of the library.

```
config.yaml        one survey's settings (copy per survey)
steplib.py         config loading, paths, logging, step reports
01_convert.py      raw .jsf/.sdf -> .ss.h5; target catalog -> targets.json
02_analyze.py      TVG model fit + after-TVG check on the uncorrected data
03_correct.py      TVG, geometric, georef (+ any configured steps), in place
04_export.py       track, waterfalls with targets boxed, grids + GeoTIFF
05_dataset.py      COCO/YOLO training dataset
```

Run one step, read `output/reports/<step>.md`, then run the next:

```bash
python 01_convert.py config.yaml
python 02_analyze.py config.yaml   # check the TVG fit against config.correct
python 03_correct.py config.yaml
python 04_export.py config.yaml
python 05_dataset.py config.yaml
```

See the [workflow guide](../../docs/workflow.md) for why it is shaped this
way and how to adapt it. `tests/examples/test_examples.py` runs all five
steps on a synthetic survey.
