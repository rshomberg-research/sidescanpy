"""
A wreck's height from its shadow
================================

Anything standing on the seabed blocks the sound behind it, so the image
has a dark shadow beyond the object. The longer the shadow, the taller
the object. Over a flat seabed, with the towfish at altitude ``A`` and
the shadow starting at slant range ``R`` and running ``Ls`` further,
similar triangles give the height of the object's top:

.. math:: h = A \\, \\frac{L_s}{R + L_s}

This page measures the shadow of one part of a wreck ping by ping, with
plain numpy on a corrected dataset. For many small objects (boulders,
debris, mines) :func:`~sidescanpy.analysis.detect_shadow_proposals` does
the same per region and returns a table; its defaults are tuned for
objects under 3 m high and 8 m long, so a wreck's shadow is split into
many small pieces.

Data: BGS London Gateway, Klein 3000 at 400 kHz, Open Government Licence
v3.0, contains data supplied by UKRI (see :doc:`/example_data`).

Usage: ``python examples/gallery/plot_shadow_height.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, uniform_filter1d

from sidescanpy import datasets
from sidescanpy.export.windows.levels import with_db_var
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import altitude_from_bottom_track, compute_geometric_ranges

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_shadow_height.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("ERROR")

# %%
# The line and the wreck
# ----------------------
# One pass over the Dynamo wreck, starboard side. The recorder applied
# its own gain, so the samples are only converted to dB. Altitude comes
# from the file, with the bottom track filling any ping without it.

survey = read_and_convert([datasets.fetch("bgs_london_gateway_dynamo_88.xtf")], fmt="xtf", nav_crs="EPSG:32631")
ds = survey.get_frequency("hz_400000").dataset.load()
ds = with_db_var(compute_geometric_ranges(altitude_from_bottom_track(ds)))
STARBOARD = 1
level = ds["backscatter_db"].values[:, STARBOARD]
slant = ds["slant_range_m"].values[0, STARBOARD]
altitude = ds["altitude_m"].values
pings = slice(640, 720)
print(f"altitude over the wreck: {np.median(altitude[pings]):.1f} m; samples every {100 * slant[1]:.0f} cm")

# %%
# What counts as shadow
# ---------------------
# The seabed level falls with range, so a shadow is judged against the
# median level at the same range over the whole line. The samples are
# averaged over 1 m of range and 5 pings first, which keeps speckle from
# breaking the shadow up. A shadow is then the longest run at least 3 dB
# below that level beyond the first bottom return, with gaps under 0.5 m
# closed. 3 dB is low because the far part of this shadow sits close to
# the noise floor: past about 45 m the seabed itself is barely above it.

sample_m = float(slant[1] - slant[0])
smooth = uniform_filter1d(uniform_filter1d(level, int(round(1.0 / sample_m)), axis=1), 5, axis=0)
background = np.median(smooth, axis=0)


def shadow_run(ping):
    """Slant range (m) where the longest shadow run starts and its length (m)."""
    below = (smooth[ping] < background - 3.0) & (slant > altitude[ping] + 2.0) & (slant < 60.0)
    below = binary_closing(below, np.ones(int(round(0.5 / sample_m))))
    edges = np.diff(np.concatenate([[0], below.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return np.nan, 0.0
    k = int(np.argmax(stops - starts))
    return float(slant[starts[k]]), float(slant[stops[k] - 1] - slant[starts[k]])


# %%
# Height ping by ping
# -------------------

ping_index = np.arange(ds.sizes["ping"])[pings]
runs = np.array([shadow_run(p) for p in ping_index])
start_m, length_m = runs[:, 0], runs[:, 1]
height_m = altitude[pings] * length_m / (start_m + length_m)
tallest = int(np.nanargmax(height_m))
print(f"longest shadow at ping {ping_index[tallest]}: starts at {start_m[tallest]:.1f} m slant range, "
      f"{length_m[tallest]:.1f} m long -> top {height_m[tallest]:.1f} m above the seabed")

fig, (ax_img, ax_h) = plt.subplots(1, 2, figsize=(13, 5), width_ratios=[2.2, 1], sharey=True,
                                   constrained_layout=True)
crop = level[pings, : int(60.0 / sample_m)]
lo, hi = np.nanpercentile(crop, [1, 99.7])
ax_img.imshow(crop, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto",
              extent=[0, 60.0, ping_index[0], ping_index[-1]])
shown = length_m > 5.0
ax_img.plot(start_m[shown], ping_index[shown], "|", color="C1", ms=6, label="shadow start")
ax_img.plot((start_m + length_m)[shown], ping_index[shown], "|", color="C0", ms=6, label="shadow end")
ax_img.set_xlabel("starboard slant range (m)")
ax_img.set_ylabel("ping")
ax_img.legend(loc="upper right")
ax_h.plot(np.where(shown, height_m, np.nan), ping_index, "o-", ms=3)
ax_h.set_xlabel("height above the seabed (m)")
ax_h.set_title("shadows over 5 m long")
ax_h.grid(alpha=0.3)
fig.savefig(out / "shadow_height.png", dpi=200)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
tax.imshow(level[650:700, int(15 / sample_m): int(55 / sample_m)], cmap="gray", vmin=lo, vmax=hi,
           origin="lower", aspect="auto")
tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_shadow_height.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# How far to trust it
# -------------------
# The longest shadows put the top of this part of the wreck about 3.3 m
# above the seabed, and the shorter ones either side trace its rounded
# outline. The formula assumes a flat seabed between the object
# and the shadow's end, and an end that is real: where the seabed beyond
# the shadow is close to the noise floor, the shadow merges into it and
# reads too long. A shadow also gives the height of the object's highest
# point along each ping, not its shape. Compare passes from other
# directions (:doc:`plot_four_passes`) before trusting one number.
