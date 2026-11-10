"""
Comparing results across sonars and file formats
=================================================

Recordings from different sonars, written in different file formats,
become the same data model, so one piece of code processes and compares
them all. Each line here is read with
:func:`~sidescanpy.io.read_and_convert`, TVG-corrected with
:func:`~sidescanpy.processing.apply_tvg` and plotted with
:func:`~sidescanpy.viz.plot_waterfall`; only the file name and format
string change. What still differs between the panels (texture, contrast,
speckle) comes from the sonar, its frequency and the seabed, not from
the processing.

The last part checks the readers against each other: one EdgeTech line
was written both as JSF and as XTF, and the two give identical samples.

Data: JNCC Wight-Barfleur Reef and mid-Irish Sea (Open Government
Licence v3.0) and USGS Grand Bay (U.S. public domain); see
:doc:`/example_data`.

Usage: ``python examples/gallery/plot_compare_sonars_and_formats.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import apply_tvg, compute_geometric_ranges
from sidescanpy.viz import plot_waterfall

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_compare_sonars_and_formats.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

# %%
# From left: EdgeTech (JSF written by Discover), Benthos SIS-1624 (XTF
# written by Isis) and Klein (XTF written by SonarPro), each in dB against
# slant range.

lines = [
    ("jncc_wight_barfleur_20130321174522.jsf", "jsf", "hz_218740", "EdgeTech JSF (Discover)"),
    ("jncc_mid_irish_sea_MIS11_0.xtf", "xtf", "hz_100000", "Benthos XTF (Isis)"),
    ("usgs_grand_bay_150602205800.xtf", "xtf", "hz_100000", "Klein XTF (SonarPro)"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
shown = []
for ax, (name, fmt, freq_key, label) in zip(axes, lines):
    survey = read_and_convert([datasets.fetch(name)], fmt=fmt)
    survey = apply_tvg(survey)
    ds = compute_geometric_ranges(survey.get_frequency(freq_key).dataset)
    # apply_tvg returns dB; average samples to about 1000 per side for display.
    ds = ds.coarsen(sample=max(1, ds.sizes["sample"] // 1000), boundary="trim").mean()
    shown.append(ds)
    plot_waterfall(ds, x_axis="slant_range", vmin_pct=2, vmax_pct=99.5, ax=ax,
                   title=f"{label}, {freq_key}")
fig.savefig(out / "compare_sonars_and_formats.png", dpi=150)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, taxes = plt.subplots(1, 3, figsize=(4, 2.8))
thumb.patch.set_facecolor("black")
for tax, shown_ds in zip(taxes, shown):
    plot_waterfall(shown_ds, x_axis="slant_range", vmin_pct=2, vmax_pct=99.5, ax=tax, title=None)
    tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_compare_sonars_and_formats.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# The same pings, written by EdgeTech Discover once as JSF and once as
# XTF. The two readers return identical samples.

jsf = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
xtf = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522H.xtf")], fmt="xtf")
a = jsf.get_frequency("hz_521000").dataset["backscatter_intensity"]
b = xtf.get_frequency(sorted(xtf.frequencies)[0]).dataset["backscatter_intensity"]
n_ping = min(a.sizes["ping"], b.sizes["ping"])
n_sample = min(a.sizes["sample"], b.sizes["sample"])
a = a.isel(ping=slice(0, n_ping), sample=slice(0, n_sample)).values
b = b.isel(ping=slice(0, n_ping), sample=slice(0, n_sample)).values
print(f"JSF vs XTF, {n_ping} pings: max |difference| = {np.nanmax(np.abs(a - b)):.3g}")
