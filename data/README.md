# data/

This directory holds survey data: raw sonar files, converted `.ss.h5`
files, and generated exports (waterfalls, gridded rasters, etc.).

**Nothing under `data/` except this file is committed to git.** Raw
JSF/SDF files routinely exceed GitHub's 100 MB limit, and converted/
exported artifacts are regenerable from raw data plus code — neither
belongs in version control. See `.gitignore`.

## Standard layout

Every survey gets one self-contained directory, everything about that
survey lives together underneath it:

```
data/<survey_name>/
├── raw/                              # original .jsf / .sdf files (untouched, source of truth)
│   ├── file1.jsf
│   └── file2.jsf
├── ss_h5/                             # converted .ss.h5 files — scripts/convert_survey.py output
│   ├── file1.ss.h5
│   └── file2.ss.h5
└── exports/                           # everything derived for viewing/analysis — scripts/visualization_demo.py output
    ├── waterfall_raw/
    │   └── hz_1495000/
    │       ├── segment_001.png
    │       └── ...
    ├── waterfall_radiometric/
    │   └── hz_1495000/...
    ├── waterfall_geometric/
    │   └── hz_1495000/...
    ├── waterfall_radiometric+geometric/
    │   └── hz_1495000/...
    ├── waterfall_full/
    │   └── hz_1495000/...
    ├── gridded_full/
    │   └── hz_1495000/...
    ├── track_full/
    │   └── hz_1495000/
    │       └── track.png
    └── manifest.json
```

| Level               | Convention                                                                                                     | Example                            |
| ------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Survey root         | `data/<survey_name>/` — one directory per physical survey/site/trip                                            | `data/site-b/`            |
| Raw data            | `raw/` — exactly the files as they came off the sonar, never written to by any script                          | `raw/20260615_line03.jsf`          |
| Converted data      | `ss_h5/` — one `.ss.h5` per raw file, filename stem preserved                                                  | `ss_h5/20260615_line03.ss.h5`      |
| Exports root        | `exports/` — anything generated for viewing/analysis; safe to delete and regenerate entirely                   | `exports/`                         |
| Export subtree      | `<output_type>_<correction_level>/`                                                                            | `waterfall_full/`, `gridded_full/` |
| Frequency subfolder | `hz_<frequency_in_hz>/`                                                                                        | `hz_1495000/`                      |
| Segment file        | `segment_<NNN>.png`, zero-padded, shared numbering across output types/correction levels for a given frequency | `segment_001.png`                  |
| Track file          | `track.png` under `track_full/hz_<freq>/` — one per survey per frequency, not segmented (`--output-types track`, opt-in) | `track_full/hz_1495000/track.png` |
| Manifest            | `manifest.json` at the root of `exports/`, one per survey                                                      | `exports/manifest.json`            |
| TVG diagnostics     | `tvg_diagnostics/hz_<freq>/` — sibling to the `<output_type>_<correction_level>` subtree, not suffixed by correction level since it inherently compares raw vs. corrected within one output (`project-a/scripts/analyze_tvg.py`, see `docs/planning/v0.8.0-tvg-diagnostics.md`) | `tvg_diagnostics/hz_1495000/mean_vs_range.png` |

This makes a survey directory self-describing and portable — copy
`data/site-b/` anywhere and everything about that survey
(source, converted, and every rendered view) travels with it.

## Populating a survey directory

```bash
# 1. Drop raw files under raw/
mkdir -p data/site-b/raw
cp /path/to/sonar/exports/*.jsf data/site-b/raw/

# 2. Convert
python scripts/convert_survey.py \
    --input-dir data/site-b/raw/ \
    --output-dir data/site-b/ss_h5/

# 3. Generate the visualization matrix
python scripts/visualization_demo.py \
    --input-dir data/site-b/ss_h5/ \
    --output-dir data/site-b/exports/
```

See `docs/planning/v0.7.0-visualization-demo.md` for the full CLI
reference of both scripts, and `CLAUDE.md` for project-wide context.

## Older, pre-existing survey directories

Some directories under `data/` predate this convention (e.g. loose
`.ss.h5`/raw files at the top level, not under `raw/`/`ss_h5/`/
`exports/`). They aren't retroactively reorganized — new work should
follow the standard layout above; old directories are left as-is
until someone has a reason to touch them.

## Test fixtures

Small fixtures used by `tests/unit/` live in `tests/data/` (checked
into git — kept intentionally tiny). Larger real-world files used by
`tests/integration/` (`--slow`) are expected locally under `data/`
but are not distributed with the repo; integration tests currently
don't skip gracefully when they're missing (see `CLAUDE.md` Known
Issues #1–2).
