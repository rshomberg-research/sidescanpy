"""
Correct a line
==============

Each correction is one function call on the survey. Every call is
recorded with its parameters in each frequency's ``processing_history``,
so a corrected file says how it was made; leave a call out to skip that
correction, or put your own function in the chain.

This JNCC line records no altitude above the seabed, which the
slant-to-ground-range step needs.
:func:`~sidescanpy.processing.altitude_from_bottom_track` picks it from
the first seabed return in each ping before
:func:`~sidescanpy.processing.compute_geometric_ranges` runs.

Data: JNCC Wight-Barfleur Reef, Open Government Licence v3.0 (see
:doc:`/example_data`).

Usage: ``python examples/gallery/plot_correct_a_line.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    altitude_from_bottom_track,
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
    slice_by_range,
)
from sidescanpy.viz import plot_waterfall

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_correct_a_line.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

survey = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
raw = compute_geometric_ranges(survey.get_frequency("hz_521000").dataset.copy(deep=True))
raw = slice_by_range(raw, max_m=150)

# %%
# The corrections, in order. Each returns a new survey and leaves its
# input unchanged.

survey = interpolate_aux_onto_pings(survey)   # navigation and attitude onto the pings
survey = altitude_from_bottom_track(survey)   # altitude from the first seabed return
survey = apply_tvg(survey)                    # spreading and absorption losses
survey = compute_geometric_ranges(survey)     # slant to ground range
survey = compute_georeferencing(survey)       # latitude and longitude per sample
corrected = slice_by_range(survey.get_frequency("hz_521000").dataset, max_m=150)
print(f"altitude from bottom track: {float(corrected['altitude_m'].median()):.1f} m (median)")

# %%
# Left: raw amplitude against slant range. Right: after the bottom track,
# time-varying gain (spreading and absorption losses) and slant-to-ground
# range, against across-track and along-track distance in metres. The
# water column is gone, and the far range is no longer darker than the
# near range. ``apply_tvg`` returns ``backscatter_intensity`` in dB, so the
# corrected panel plots it as it is. Both panels average samples down to
# about 1000 per side first, close to the figure's resolution, so fine
# texture is averaged rather than skipped.


def for_display(ds):
    """Average samples down to about 1000 per side."""
    return ds.coarsen(sample=max(1, ds.sizes["sample"] // 1000), boundary="trim").mean()


fig, (ax_raw, ax_cor) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
plot_waterfall(for_display(raw), x_axis="slant_range", gamma=0.5, ax=ax_raw, title="Raw (slant range)")
plot_waterfall(for_display(corrected), x_axis="across_track", y_axis="along_track",
               vmin_pct=2, vmax_pct=99.5, ax=ax_cor, title="Bottom track + TVG + ground range (dB)")
fig.savefig(out / "correct_a_line.png", dpi=110)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
plot_waterfall(for_display(corrected), x_axis="across_track", y_axis="along_track",
               vmin_pct=2, vmax_pct=99.5, ax=tax, title=None)
tax.set_facecolor("black")
tax.set_xticks([])
tax.set_yticks([])
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_correct_a_line.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# The default TVG uses 30 log R spreading and Francois-Garrison
# absorption. Whether a sonar already applied some gain in hardware
# varies; :func:`~sidescanpy.analysis.fit_range_decay` measures what a dataset
# still needs (see ``examples/tvg_calibration.py``).
