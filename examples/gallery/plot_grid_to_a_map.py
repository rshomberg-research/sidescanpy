"""
Grid to a map
=============

Once every sample has a latitude and longitude (the ``georef`` step),
:func:`~sidescanpy.grids.grid_channel` resamples a channel onto a regular UTM
grid and :func:`~sidescanpy.grids.to_geotiff` writes it for a GIS.

The line is from a Benthos SIS-1624 survey in the Irish Sea (XTF written
by Isis), which records altitude and navigation, so the standard
corrections are enough. The GeoTIFF needs the ``geo`` extra
(``rioxarray``).

Data: JNCC mid-Irish Sea, Open Government Licence v3.0 (see
:doc:`/example_data`).

Usage: ``python examples/gallery/plot_grid_to_a_map.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.grids import grid_channel, to_geotiff
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    apply_tvg,
    compute_geometric_ranges,
    compute_georeferencing,
    interpolate_aux_onto_pings,
    slice_by_range,
)
from sidescanpy.viz import plot_gridded_raster

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

# %%
# Navigation in this file is sample-and-hold: each position repeats for
# several pings until the next fix arrives.
# :func:`~sidescanpy.processing.interpolate_aux_onto_pings` interpolates
# between the real fixes, not the repeated entries, which
# puts the pings about 0.65 m apart.

survey = read_and_convert([datasets.fetch("jncc_mid_irish_sea_MIS11_0.xtf")], fmt="xtf")
survey = interpolate_aux_onto_pings(survey)
survey = apply_tvg(survey)
survey = compute_geometric_ranges(survey)
survey = compute_georeferencing(survey)

# %%
# Grid each frequency at 1 m, leaving out the water column. Heading
# jitter and a few longer jumps between fixes leave empty cells;
# inverse-distance weighting with a 4 m search radius fills them from the
# nearest samples. North is up; the dark stripe is the nadir under the
# towfish.

fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
for ax, freq_key in zip(axes, ["hz_100000", "hz_400000"]):
    ds = slice_by_range(survey.get_frequency(freq_key).dataset, min_m=5)
    raster = grid_channel(ds, resolution_m=1.0, method="idw", max_search_radius_m=4.0)
    to_geotiff(raster, out / f"mid_irish_sea_{freq_key}.tif")
    plot_gridded_raster(raster, fig=fig, ax=ax, cmap="copper", vmin_pct=2, vmax_pct=98,
                        title=f"{freq_key}, 1 m UTM grid")
fig.savefig(out / "grid_to_a_map.png", dpi=110)
