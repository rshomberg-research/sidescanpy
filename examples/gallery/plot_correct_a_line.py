"""
Correct a line
==============

:func:`~sidescanpy.processing.correct_survey` applies an ordered list of
correction steps. Each step is optional, can be replaced by your own
function, and is recorded with its parameters in the dataset's
``processing_history``, so a corrected file says how it was made.

This JNCC line records no altitude above the seabed, which the
slant-to-ground-range step needs. The ``bottom_track`` step picks it from
the first seabed return in each ping before ``geometric`` runs.

Data: JNCC Wight-Barfleur Reef, Open Government Licence v3.0 (see
:doc:`/example_data`).

Usage: ``python examples/gallery/plot_correct_a_line.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.core.units import linear_to_db
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import compute_geometric_ranges, correct_survey, slice_by_range
from sidescanpy.viz import plot_waterfall

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

survey = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
raw = compute_geometric_ranges(survey.get_frequency("hz_521000").dataset.copy(deep=True))
raw = slice_by_range(raw, max_m=150)

# %%
# Each step is optional and recorded in ``processing_history`` with its
# parameters. Leave a step out to skip it, or pass your own function.

survey = correct_survey(
    survey,
    steps=["interpolate_aux", "bottom_track", "tvg", "geometric", "georef"],
)
corrected = slice_by_range(survey.get_frequency("hz_521000").dataset, max_m=150)
print(f"altitude from bottom track: {float(corrected['altitude_m'].median()):.1f} m (median)")

# %%
# Left: raw amplitude against slant range. Right: after the bottom track,
# time-varying gain (spreading and absorption losses) and slant-to-ground
# range, against across-track and along-track distance in metres. The
# water column is gone, and the far range is no longer darker than the
# near range. After TVG most of the seabed sits within a few dB, so the
# colour scale stretches over the 10th to 99.5th percentile.

fig, (ax_raw, ax_cor) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
plot_waterfall(raw, x_axis="slant_range", gamma=0.5, ax=ax_raw, title="Raw (slant range)")
corrected = corrected.assign(backscatter_db=linear_to_db(corrected["backscatter_intensity"]))
plot_waterfall(corrected, x_axis="across_track", y_axis="along_track", value_var="backscatter_db",
               vmin_pct=10, vmax_pct=99.5, ax=ax_cor, title="Bottom track + TVG + ground range (dB)")
fig.savefig(out / "correct_a_line.png", dpi=110)

# %%
# The default TVG uses 30 log R spreading and Francois-Garrison
# absorption. Whether a sonar already applied some gain in hardware
# varies; :func:`~sidescanpy.analysis.fit_range_decay` measures what a dataset
# still needs (see ``examples/tvg_calibration.py``).
