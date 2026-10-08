# Command line

Four commands cover quick runs; each subcommand calls one library
function. Run any of them with `--help` for the full option list.

| Command | Subcommands |
|---|---|
| `sidescan-convert` | `survey` (JSF/SDF/XTF → `.ss.h5`, one file per input, except XTF files whose range setting changes, which give one `<stem>_seg01.ss.h5`, `<stem>_seg02.ss.h5`, … per constant-range run, runs shorter than `--min-segment-pings` (50) dropped; far-range noise cropped per output unless `--no-crop-farrange`), `targets` (catalog → `targets.json`/`.csv`; `--plot` adds a position scatter) |
| `sidescan-correct` | applies `--crop-farrange`, `--tvg`, `--geometric`, `--range-correction`, `--georef` to every `.ss.h5` in `--ss-h5-dir`, in place; TVG terms via `--spreading-loss-db`, `--absorption-db-per-m`, `--no-spreading`, `--no-absorption`, water conditions via `--temperature-c --salinity-psu --depth-m --ph`; `--ping-chunk-size` bounds memory |
| `sidescan-export` | `waterfall`, `grid` (`--geotiff`), `track`, `chip` (COCO/YOLO dataset, see [Targets and datasets](datasets.md)) |
| `sidescan-analyze` | `investigate` (every analysis + report), `tvg-fit`, `segment`, `nadir`, `beam-geometry`, `usable-range`, `units`, `nav-attitude` (alias `nav-summary`), `recommend-config` |

## Config files

Every subcommand takes `--config file.yaml` (or `.json`). Keys are the
option names with underscores (`spreading_loss_db: 30`); flags on the
command line override them. Keys the subcommand does not have log a
warning, so one config file can be shared across commands;
`--strict-config` turns the warning into an error.

## Logging

`-v` shows library DEBUG messages, `-q` shows warnings only, and
`--log-file PATH` also writes the log to a file.

## Without matplotlib

`sidescan-convert` and `sidescan-correct` run without the `[viz]` extra.
`sidescan-export` and `sidescan-analyze` need it for figures; analysis
subcommands with `--no-plots` and `recommend-config` do not.
