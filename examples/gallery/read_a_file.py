"""
Read a raw sonar file and look at it.

Downloads one EdgeTech JSF line (40 MB) from the UK JNCC Wight-Barfleur
Reef survey, converts it to sidescanpy's data model, saves it as .ss.h5,
and plots the raw waterfall of both frequencies.

Usage:
    python examples/gallery/read_a_file.py [output_dir]

2026 rshomberg@uri.edu
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.io.hdf5 import write_survey
from sidescanpy.log import configure_logging
from sidescanpy.processing import compute_geometric_ranges
from sidescanpy.viz import plot_waterfall

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

# Download (once) and convert. read_and_convert takes JSF, SDF or XTF.
path = datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")
survey = read_and_convert([path], fmt="jsf")
write_survey(survey, out / "wight_barfleur.ss.h5", overwrite=True)

# One xarray Dataset per frequency: backscatter is (ping, channel, sample),
# channel 0 = port, 1 = starboard.
for freq_key in sorted(survey.frequencies):
    ds = survey.get_frequency(freq_key).dataset
    print(freq_key, dict(ds.sizes))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, freq_key in zip(axes, sorted(survey.frequencies)):
    ds = compute_geometric_ranges(survey.get_frequency(freq_key).dataset)
    plot_waterfall(ds, x_axis="slant_range", gamma=0.5, ax=ax,
                   title=f"{freq_key}: raw, slant range")
fig.savefig(out / "read_a_file.png", dpi=110)
