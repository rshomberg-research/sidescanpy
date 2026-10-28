"""
One data model, any format: the same code on three sonars and writers.

EdgeTech JSF (Discover), Benthos XTF (Isis) and Klein XTF (SonarPro),
each read, corrected to ground range and plotted with identical calls.
The Klein file's recorded altitude is far off (tens of metres in a
shallow estuary), so ``bottom_track`` also replaces altitudes that
disagree with the tracked one by more than 5 m. The JNCC line was also
written by Discover as XTF; both readers return the same samples.

Usage:
    python examples/gallery/three_formats.py [output_dir]

2026 rshomberg@uri.edu
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import correct_survey
from sidescanpy.processing.range_correction import RangeCorrectionTable, fit_range_correction
from sidescanpy.viz import plot_waterfall

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")

lines = [
    ("jncc_wight_barfleur_20130321174522.jsf", "jsf", "hz_521000", "EdgeTech JSF (Discover)"),
    ("jncc_mid_irish_sea_MIS15_2.xtf", "xtf", "hz_400000", "Benthos XTF (Isis)"),
    ("usgs_grand_bay_150602133300.xtf", "xtf", "hz_100000", "Klein XTF (SonarPro)"),
]
fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
for ax, (name, fmt, freq_key, label) in zip(axes, lines):
    survey = read_and_convert([datasets.fetch(name)], fmt=fmt)
    survey = correct_survey(
        survey,
        steps=["interpolate_aux", "bottom_track", "tvg", "geometric"],
        params={"bottom_track": {"max_diff_m": 5.0, "smooth_pings": 21}},
    )
    table = RangeCorrectionTable()
    for key in survey.frequencies:
        ds = survey.get_frequency(key).dataset
        table.add_curve(ds.attrs["freq_hz_start"], *fit_range_correction(ds))
    survey = correct_survey(survey, steps=["range_correction"],
                            params={"range_correction": {"table": table}})
    ds = survey.get_frequency(freq_key).dataset
    # Average samples down to about 1000 per side for display (values are dB).
    ds = ds.coarsen(sample=max(1, ds.sizes["sample"] // 1000), boundary="trim").mean()
    plot_waterfall(ds, x_axis="across_track", y_axis="along_track",
                   vmin_pct=2, vmax_pct=99.5, ax=ax, title=f"{label}, {freq_key}")
fig.savefig(out / "three_formats.png", dpi=110)

# The same pings, written by EdgeTech Discover once as JSF and once as XTF.
jsf = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
xtf = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522H.xtf")], fmt="xtf")
a = jsf.get_frequency("hz_521000").dataset["backscatter_intensity"]
b = xtf.get_frequency(sorted(xtf.frequencies)[0]).dataset["backscatter_intensity"]
n_ping = min(a.sizes["ping"], b.sizes["ping"])
n_sample = min(a.sizes["sample"], b.sizes["sample"])
a = a.isel(ping=slice(0, n_ping), sample=slice(0, n_sample)).values
b = b.isel(ping=slice(0, n_ping), sample=slice(0, n_sample)).values
print(f"JSF vs XTF, {n_ping} pings: max |difference| = {np.nanmax(np.abs(a - b)):.3g}")
