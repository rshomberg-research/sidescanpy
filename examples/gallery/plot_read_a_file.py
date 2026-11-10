"""
Read a file
===========

Convert one raw EdgeTech JSF line to sidescanpy's data model, save it as
``.ss.h5``, and look at it. :func:`~sidescanpy.io.read_and_convert` reads JSF,
Klein SDF and XTF into the same structure: a
:class:`~sidescanpy.core.SideScanSurvey` holding one :class:`xarray.Dataset`
per frequency, with backscatter shaped ``(ping, channel, sample)``
(channel 0 is port, 1 starboard) plus the navigation and attitude
records.

Data: JNCC Wight-Barfleur Reef, Open Government Licence v3.0 (see
:doc:`/example_data`).

Usage: ``python examples/gallery/plot_read_a_file.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import compute_geometric_ranges
from sidescanpy.viz import plot_waterfall

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_read_a_file.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

# %%
# Download (once) and convert. The reader logs two warnings worth knowing
# about: this file records no sound velocity (1500 m/s is used) and no
# altitude. :ref:`sphx_glr_examples_plot_correct_a_line.py` fills the
# altitude in.

path = datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")
survey = read_and_convert([path], fmt="jsf")
write_survey(survey, out / "wight_barfleur.ss.h5", overwrite=True)

# %%
# One xarray Dataset per frequency: backscatter is (ping, channel, sample),
# channel 0 = port, 1 = starboard.

for freq_key in sorted(survey.frequencies):
    ds = survey.get_frequency(freq_key).dataset
    print(freq_key, dict(ds.sizes))

# %%
# The two frequencies of the same pings, raw amplitude against slant
# range. The dark band in the middle is the water column under the
# towfish; the seabed starts where it ends, about 18 m out on each side.
# A linear seabed feature runs along the starboard side at about 50 m.

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, freq_key in zip(axes, sorted(survey.frequencies)):
    ds = compute_geometric_ranges(survey.get_frequency(freq_key).dataset)
    plot_waterfall(ds, x_axis="slant_range", gamma=0.5, ax=ax,
                   title=f"{freq_key}: raw, slant range")
fig.savefig(out / "read_a_file.png", dpi=110)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
plot_waterfall(ds, x_axis="slant_range", gamma=0.5, ax=tax, title=None)
tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_read_a_file.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore
