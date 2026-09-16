"""
Console-script CLIs consolidating the pipeline scripts that used to be
duplicated, near-identically, across every consumer project's own
``scripts/`` directory (see each command module's own docstring for
which old numbered scripts it replaces).

Entry points (registered in ``pyproject.toml``'s ``[project.scripts]``):

- ``sidescan-convert`` — raw survey / target-catalog conversion
- ``sidescan-correct`` — TVG / geometric / georef correction, far-range
  crop
- ``sidescan-export`` — waterfalls, grids, tracks, chips, dataset export
- ``sidescan-analyze`` — TVG/data-investigation diagnostics

2026 rshomberg@uri.edu
"""
