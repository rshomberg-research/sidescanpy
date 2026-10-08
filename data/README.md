# data/

A local scratch area for survey data while developing. **Nothing here
except this file is committed**: raw JSF/SDF files are large, and
converted or exported files can be regenerated (see `.gitignore`).

The library does not read from this directory. Pass paths explicitly, or
lay out a project the way `examples/project_template/` does (one config
per survey, every step writing under one output root).

Integration tests use their own fixtures: place a real JSF file at
`tests/data/jsf/sample.jsf` and a Klein SDF file at
`tests/data/sdf/sample.sdf`; without them those tests skip.
