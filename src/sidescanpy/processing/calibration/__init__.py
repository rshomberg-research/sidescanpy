"""
Bundled calibration data for processing steps.

Currently holds `range_correction_default.json`, the fitted empirical
range-correction lookup used by
`processing.range_correction.RangeCorrectionTable.load_default()`. Not
Python source — data only, packaged via `[tool.setuptools.package-data]`
in `pyproject.toml` so it ships in built wheels, not just editable installs.
"""
