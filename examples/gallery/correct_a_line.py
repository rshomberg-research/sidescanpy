"""
Correct a line: altitude from the bottom track, TVG, ground range, and an
empirical range correction.

The JNCC Wight-Barfleur EdgeTech line records no altitude, so the
``bottom_track`` step picks it from the first seabed return before the
geometric step needs it. TVG leaves some brightness that still depends on
range; ``fit_range_correction`` measures it from the line itself and the
``range_correction`` step removes it. The figure shows the 521 kHz channel
raw, after TVG, and after the range correction.

Usage:
    python examples/gallery/correct_a_line.py [output_dir]

2026 rshomberg@uri.edu
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sidescanpy import datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import compute_geometric_ranges, correct_survey
from sidescanpy.processing.range_correction import RangeCorrectionTable, fit_range_correction
from sidescanpy.viz import plot_waterfall

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("INFO")
FREQ = "hz_521000"

survey = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
raw = compute_geometric_ranges(survey.get_frequency(FREQ).dataset.copy(deep=True))

# Each step is optional and recorded in processing_history with its
# parameters. Leave a step out to skip it, or pass your own function.
survey = correct_survey(
    survey,
    steps=["interpolate_aux", "bottom_track", "tvg", "geometric", "georef"],
)
tvg = survey.get_frequency(FREQ).dataset
print(f"altitude from bottom track: {float(tvg['altitude_m'].median()):.1f} m (median)")

# Fit the leftover range dependence per frequency, then remove it.
table = RangeCorrectionTable()
for freq_key in survey.frequencies:
    ds = survey.get_frequency(freq_key).dataset
    table.add_curve(ds.attrs["freq_hz_start"], *fit_range_correction(ds))
survey = correct_survey(survey, steps=["range_correction"], params={"range_correction": {"table": table}})
flat = survey.get_frequency(FREQ).dataset


def for_display(ds):
    """Average samples down to about 1000 per side, near screen resolution."""
    return ds.coarsen(sample=max(1, ds.sizes["sample"] // 1000), boundary="trim").mean()


# After TVG, backscatter_intensity is in dB.
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
plot_waterfall(for_display(raw), x_axis="slant_range", gamma=0.5, ax=axes[0],
               title="Raw (slant range)")
plot_waterfall(for_display(tvg), x_axis="across_track", y_axis="along_track",
               vmin_pct=2, vmax_pct=99.5, ax=axes[1], title="Bottom track + TVG + ground range")
plot_waterfall(for_display(flat), x_axis="across_track", y_axis="along_track",
               vmin_pct=2, vmax_pct=99.5, ax=axes[2], title="+ range correction")
fig.savefig(out / "correct_a_line.png", dpi=110)
