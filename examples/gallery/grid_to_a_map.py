"""
Grid a corrected line onto a UTM map and write a GeoTIFF.

Uses a Benthos SIS-1624 line (XTF written by Isis) from the UK JNCC
mid-Irish Sea survey, which records altitude and navigation, so the
default correction steps are enough; an empirical range correction fitted
from the line evens out brightness across the swath. Needs the ``geo``
extra for the GeoTIFF.

Usage:
    python examples/gallery/grid_to_a_map.py [output_dir]

2026 rshomberg@uri.edu
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.grids import grid_channel, to_geotiff
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct_survey, slice_by_range
from sidescanpy.processing.range_correction import RangeCorrectionTable, fit_range_correction
from sidescanpy.viz import plot_gridded_raster

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

survey = read_and_convert([datasets.fetch("jncc_mid_irish_sea_MIS15_2.xtf")], fmt="xtf")
survey = correct_survey(survey)  # interpolate_aux, tvg, geometric, georef

table = RangeCorrectionTable()
for freq_key in survey.frequencies:
    ds = survey.get_frequency(freq_key).dataset
    table.add_curve(ds.attrs["freq_hz_start"], *fit_range_correction(ds))
survey = correct_survey(survey, steps=["range_correction"], params={"range_correction": {"table": table}})

# Grid each frequency at 1 m, leaving out the water column. Heading jitter
# and a few longer jumps between fixes leave empty cells between pings;
# inverse-distance weighting with a 4 m search radius fills them from the
# nearest samples.
fig, axes = plt.subplots(1, 2, figsize=(9, 7), constrained_layout=True)
for ax, freq_key in zip(axes, ["hz_100000", "hz_400000"]):
    ds = slice_by_range(survey.get_frequency(freq_key).dataset, min_m=5)
    raster = grid_channel(ds, resolution_m=1.0, method="idw", max_search_radius_m=4.0)
    to_geotiff(raster, out / f"mid_irish_sea_{freq_key}.tif")
    plot_gridded_raster(raster, fig=fig, ax=ax, cmap="copper", vmin_pct=2, vmax_pct=98,
                        title=f"{freq_key}, 1 m UTM grid")
fig.savefig(out / "grid_to_a_map.png", dpi=110)
