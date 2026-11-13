"""
Read a Humminbird recording
===========================

Recreational fishfinders with side imaging record far more sidescan than
survey sonars do. A Humminbird recording is a small ``.DAT`` header and a
directory of the same name with one ``.SON`` file per beam;
:func:`~sidescanpy.io.read_and_convert` takes the ``.DAT`` file and
converts the two side imaging beams into the same structure as a JSF,
SDF or XTF file, so every correction and export works on it unchanged.

The files record no sample interval, so the reader assumes one (see
:doc:`/sources/humminbird`). This example checks that assumption against
the depth the unit recorded, measures the gain the unit already applied,
then corrects the line and shows it in ground range.

Data: U.S. Fish and Wildlife Service, Chena River 2022, U.S. public
domain (see :doc:`/example_data`).

Usage: ``python examples/gallery/plot_read_a_humminbird_recording.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import analysis, datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    correct,
    interpolate_aux_onto_pings,
    smooth_navigation,
)
from sidescanpy.viz import plot_waterfall

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_read_a_humminbird_recording.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

# %%
# Download (once) and convert. Fetching the ``.DAT`` file also fetches its
# two side imaging ``.SON`` files into ``usfws_chena_R00013/`` beside it,
# where the reader looks for them.

path = datasets.fetch("usfws_chena_R00013.DAT")
survey = read_and_convert([path], fmt="humminbird")
ds = survey.get_frequency("hz_405000").dataset
print(dict(ds.sizes))
for name in ("humminbird_recording", "humminbird_water_type", "humminbird_sample_rate_source"):
    print(f"{name}: {ds.attrs[name]}")
print(f"sample_rate_hz: {ds.attrs['sample_rate_hz']:.0f}"
      f" -> {1500 / (2 * ds.attrs['sample_rate_hz']) * 100:.2f} cm per sample at 1500 m/s")

# %%
# The raw samples against slant range, port on the left. The dark band in
# the middle is the water column under the boat. The cyan lines are the
# depth the unit recorded, drawn at that slant range on each side: they
# follow the first seabed return, so the assumed sample interval puts the
# seabed where the unit's own depth sounder found it.

ds = compute_geometric_ranges(ds)
first = slice(0, 1500)
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
plot_waterfall(ds.isel(ping=first), x_axis="slant_range", gamma=0.7, ax=ax,
               title="Raw side imaging, slant range, with the recorded depth")
depth = ds["altitude_m"].values[first]
pings = np.arange(depth.size)
ax.plot(-depth, pings, color="c", lw=0.8)
ax.plot(depth, pings, color="c", lw=0.8, label="recorded depth")
ax.set_xlim(-12, 12)
ax.legend(loc="lower right")
fig.savefig(out / "humminbird_raw.png", dpi=110)

# %%
# The samples are the unit's 8-bit display values, after its own gain,
# which the file does not record. Measure what it applied before choosing
# a TVG: :func:`~sidescanpy.analysis.fit_tvg_models` fits each TVG model
# to the median level against range, and
# :func:`~sidescanpy.analysis.check_tvg` applies the best one and checks
# that the result is flat. On this recording the unit has already removed
# the spreading loss, so only absorption is left to correct; the default
# 30 log R would brighten the far range by several decibels.

fit = analysis.fit_tvg_models(ds)
check = analysis.check_tvg(ds, fit.tvg_params, fit.window_m)
rms = ", ".join(f"{name} {m.rms_db:.2f}" for name, m in fit.models.items())
print(f"best {fit.best} (TVG status {fit.tvg_status}); RMS in dB: {rms}")
print(f"after TVG: spread {check.spread_db:.2f} dB, trend {check.trend_db_per_10m:+.2f} dB per 10 m")

# %%
# Correct the line like any other: navigation onto the pings (the unit
# repeats each GPS fix until the next one), the measured TVG, and slant
# to ground range with the recorded depth as the altitude. The file has
# no sound velocity, attitude or pitch, so the steps log that they use
# 1500 m/s and a level boat. About 110 m of the line: the river bank
# rises out of the water on the port side, about 18 m out.

survey = smooth_navigation(interpolate_aux_onto_pings(survey), window_s=20.0)
corrected = apply_tvg(survey.get_frequency("hz_405000").dataset, fit.tvg_params)
corrected = correct(corrected, ["geometric"])
stretch = corrected.isel(ping=slice(2900, 3800))
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
plot_waterfall(stretch, x_axis="ground_range", y_axis="along_track", ax=ax,
               title=f"Measured TVG ({fit.best}), ground range")
fig.savefig(out / "humminbird_corrected.png", dpi=110)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
plot_waterfall(stretch, x_axis="ground_range", ax=tax, title=None)
tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_read_a_humminbird_recording.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore
