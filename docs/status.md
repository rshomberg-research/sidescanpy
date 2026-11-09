# Project status

sidescanpy is beta software. It is used on real surveys, but until
version 1.0, any 0.x release can change:

- **The `.ss.h5` file format.** Group layout, variable names, attributes
  and encodings are not frozen. A file written by one 0.x version may not
  read, or may read differently, in a later one. There is no migration
  tool yet.
- **The Python API and the CLI.** Functions, parameters, defaults, flags
  and config keys can be renamed or removed.
- **Default processing parameters**, such as the TVG spreading term, and
  so the numbers the default corrections produce.

Breaking changes are listed in [`CHANGELOG.md`](https://github.com/rshomberg-research/sidescanpy/blob/main/CHANGELOG.md)
under the release that makes them.

## Protecting your work

- **Keep the raw JSF, SDF and XTF files.** They are the archive. Treat
  `.ss.h5` files as derived products you can regenerate, and expect to
  re-convert after an upgrade.
- **Pin the version** your project uses (a release tag, see
  [Install](install.md)), and upgrade on purpose.
- **Check which version wrote a file.** Each `.ss.h5` records the layout
  version in the root attribute `format_version` (from 0.9.0) and the
  writer in `sidescanpy_version` (from 0.8.0), and every processing step
  records its version in `processing_history`:

```python
import h5netcdf

with h5netcdf.File("survey.ss.h5", "r") as f:
    print(f.attrs.get("format_version"), f.attrs.get("sidescanpy_version"))
```

## Documentation for a release

This site follows `main`. For the docs of a release, build them from a
clone of that tag (`make docs-serve`, then open <http://127.0.0.1:8000>).

## Test and example data

The repository holds code only, no survey data.

- **Examples.** The [worked examples](examples/index.rst) download public
  recordings with `sidescanpy.datasets.fetch()`; see
  [Example data](example_data.md). The other scripts in `examples/` run
  on synthetic data.
- **Test fixtures.** `tests/unit/io/test_public_fixtures.py` runs on five
  small cuts (4.1 MB) of the same kind of openly licensed recordings:
  EdgeTech JSF, and XTF from three writers. Their sources, licences and
  checksums are in [`tests/data/public/SOURCES.md`](https://github.com/rshomberg-research/sidescanpy/blob/main/tests/data/public/SOURCES.md).
  Until an archived copy is published, ask the maintainer for them. Point
  `SIDESCANPY_PUBLIC_DATA` at the folder to run those tests; without it
  they skip.

## Feedback

Open an issue on GitHub. For a reader problem, say which sonar and
acquisition software wrote the file and attach the shortest file that
shows it, or a cut made with `io.jsf.write_subset` /
`io.xtf.write_subset`, if you can share it.
