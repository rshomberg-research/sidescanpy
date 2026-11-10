# Stability

sidescanpy is beta software. Until version 1.0, any release can change:

- **The [data format](format/index.md).** The `.ss.h5` layout, variable
  names, attributes and encodings are a draft (format version 0.1). A
  file written by one version may not read, or may read differently, in
  a later one.
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
- **Check which version wrote a file.** Each `.ss.h5` records the format
  version in the root attribute `format_version` and the writer in
  `sidescanpy_version`, and every processing step records its version in
  `processing_history`:

```python
import h5netcdf

with h5netcdf.File("survey.ss.h5", "r") as f:
    print(f.attrs.get("format_version"), f.attrs.get("sidescanpy_version"))
```

This site follows the latest code. For the documentation of the version
you pinned, build it from that release (`make docs-serve` in a clone of
the tag).
