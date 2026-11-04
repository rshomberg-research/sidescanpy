# Pre-release status

sidescanpy's code is shared by invitation before its public release; this
documentation is public. This page covers what that means for you: what
may still change, how to get access and install it, where the test data
is, and how to report problems.

## What may change before 1.0

Until version 1.0, any 0.x release can change:

- **The `.ss.h5` file format.** Group layout, variable names, attributes
  and encodings are not frozen. A file written by one 0.x version may not
  read, or may read differently, in a later one. There is no migration
  tool yet.
- **The Python API and the CLI.** Functions, parameters, defaults, flags
  and config keys can be renamed or removed.
- **Default processing parameters**, such as the TVG spreading term, and
  so the numbers a default `correct_survey` run produces.

Breaking changes are listed in [`CHANGELOG.md`](https://github.com/rshomberg-research/sidescanpy/blob/main/CHANGELOG.md)
under the release that makes them.

To protect your own work:

- **Keep the raw JSF, SDF and XTF files.** They are the archive. Treat
  `.ss.h5` files as derived products you can regenerate, and expect to
  re-convert after an upgrade.
- **Pin the version** your project uses (a tag or a commit, see below),
  and upgrade on purpose.
- **Check which version wrote a file.** Each `.ss.h5` records the writer
  in the root attribute `sidescanpy_version` (files from 0.7.6 and
  earlier lack it), and every processing step records its version in
  `processing_history`:

```python
import h5netcdf

with h5netcdf.File("survey.ss.h5", "r") as f:
    print(f.attrs.get("sidescanpy_version", "0.7.6 or earlier"))
```

## Access

The repository is private. Invited people get read access: you can
clone it, install from it and open issues. Please don't pass the code or
the test data link on outside your group before the public release; ask
instead, and access can be added. This site and the example data are
public and can be shared freely.

## Install

sidescanpy is not on PyPI yet. [Install](install.md) gives the commands
to install from the repository, pinned to a release tag (or `main` for
the latest fixes). You need an SSH key or a `gh auth login` session that
GitHub accepts for your account.

## Documentation

This site, <https://rshomberg-research.github.io/sidescanpy/>, follows
`main`. For the docs of a release tag, or offline, either build them from
a clone (`make docs-serve`, then open <http://127.0.0.1:8000>) or
download the **sidescanpy-docs** artifact of that tag's **docs** run from
the repository's **Actions** tab (kept 90 days).

## Test and example data

The repository holds code only, no survey data.

- **Test fixtures.** Five small cuts (4.1 MB) of openly licensed public
  recordings: EdgeTech JSF, and XTF from three writers. Their sources,
  licences and checksums are in [`tests/data/public/SOURCES.md`](https://github.com/rshomberg-research/sidescanpy/blob/main/tests/data/public/SOURCES.md).
  They come as one zip, shared by link with invited people. Unzip it and
  point `SIDESCANPY_PUBLIC_DATA` at the folder:

    ```bash
    export SIDESCANPY_PUBLIC_DATA=/path/to/sidescanpy-public-fixtures
    uv run pytest tests/unit/io/test_public_fixtures.py
    ```

    Without the variable those tests skip. The fixtures are also handy
    as small real files to try the readers on.

- **Examples.** The [worked examples](examples/index.rst) download public
  recordings with `sidescanpy.datasets.fetch()`, which needs no access
  to anything private. The other scripts in `examples/` run on synthetic
  data. An archived copy of the example data with a DOI is planned for
  the public release.

Data you get from us keeps its original licence (UK Open Government
Licence v3, or U.S. public domain); credit the sources as `SOURCES.md`
asks.

## Feedback

Open an issue on GitHub. For a reader problem, say which sonar and
acquisition software wrote the file and attach the shortest file that
shows it, or a cut made with `io.jsf.write_subset` /
`io.xtf.write_subset`, if you can share it.
