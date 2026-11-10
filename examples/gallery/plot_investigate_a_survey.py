"""
Investigating an unknown survey
===============================

A survey arrives with no notes. Before choosing corrections, measure
what the recording is: what the samples are, where the seabed starts
under the towfish, how far the sonar sees before the signal sinks into
noise, and whether the recorder already applied a time-varying gain.
Each answer comes from one :mod:`sidescanpy.analysis` function that
returns a result you can print, save and compare between surveys, and
each has a plot in :mod:`sidescanpy.viz.analysis`.

``sidescan-analyze investigate`` runs the same steps from the command
line, and ``examples/investigate_dataset.py`` writes them all to a
folder with a ``metrics.json``.

Data: JNCC Wight-Barfleur Reef, EdgeTech JSF, Open Government Licence
v3.0 (see :doc:`/example_data`).

Usage: ``python examples/gallery/plot_investigate_a_survey.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from sidescanpy import analysis, datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    altitude_from_bottom_track,
    blank_nadir_noise,
    compute_geometric_ranges,
    uniform_exclusion_boundary,
)
from sidescanpy.viz import analysis as plots
from sidescanpy.viz.tvg import plot_range_decay_fit

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_investigate_a_survey.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("WARNING")

# %%
# Load both frequencies as recorded: no corrections, the electronic noise
# burst at nadir zeroed, and slant and ground range added. This file
# records no altitude, so the bottom track supplies it first.

survey = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
channels = {}
for freq_key in survey.frequencies:
    ds = altitude_from_bottom_track(survey.get_frequency(freq_key).dataset.load())
    channels[freq_key] = compute_geometric_ranges(blank_nadir_noise(ds))
var = "backscatter_intensity"
ds = channels["hz_218740"]

# %%
# What are the samples?
# ---------------------
# Their type and range, and how many are zero or saturated. On a log
# scale, linear amplitude spreads over several decades; values already
# in dB would sit in a narrow band.

values = analysis.characterize_values(ds, value_var=var)
print(values)
fig, _ = plots.plot_value_histograms(ds, var)

# %%
# Where does the seabed start?
# ----------------------------
# Under the towfish the sonar hears only water until the first bottom
# return. :func:`~sidescanpy.analysis.analyze_nadir` finds that boundary
# per ping and classifies the nadir region: a clean ``gap`` here, the
# easy case (a bright spike or noise needs more care). Later steps
# exclude everything before it.

nadirs = {fk: analysis.analyze_nadir(c, value_var=var) for fk, c in channels.items()}
print({fk: n.nadir_type for fk, n in nadirs.items()})
fig, _ = plots.plot_nadir_waterfall(ds, var, nadir=nadirs["hz_218740"])

# %%
# How far does the sonar see?
# ---------------------------
# The level falls with range until it reaches the noise floor; beyond
# that the image holds no seabed.
# :func:`~sidescanpy.analysis.estimate_usable_range` finds where. Higher
# frequencies are absorbed faster, so the 521 kHz channel sees about half
# as far as the 219 kHz one.

usable = {
    fk: analysis.estimate_usable_range(c, value_var=var, bin_width_m=2.0,
                                       exclude_before_sample=nadirs[fk].first_bottom_return_sample)
    for fk, c in channels.items()
}
for fk, u in usable.items():
    print(f"{fk}: usable range {u.usable_range_m:.0f} m of {float(channels[fk]['slant_range_m'].max()):.0f} m recorded")
fig, _ = plots.plot_usable_range(usable["hz_218740"])

# %%
# Did the recorder already apply a gain?
# --------------------------------------
# Raw sidescan levels fall with range by spreading and absorption, unless
# the recorder compensated them. :func:`~sidescanpy.analysis.fit_range_decay`
# fits the level against range, within the usable range, with each
# combination of terms, and reports a TVG status: ``none`` (no gain
# applied: correct both terms), ``spreading_only`` or ``full`` (part or
# all of it already applied: turn those terms off), or ``ambiguous``.

decays = {}
for fk, c in channels.items():
    boundary = nadirs[fk].first_bottom_return_sample
    decays[fk] = analysis.fit_range_decay(
        c, value_var=var, bin_width_m=5.0, usable_range_m=usable[fk].usable_range_m,
        exclude_before_sample=uniform_exclusion_boundary(boundary) if boundary is not None else None,
    )
    d = decays[fk].to_dict()
    rmse = ", ".join(f"{name} {c['rmse_db']:.1f} dB" for name, c in d["candidates"].items())
    print(f"{fk}: TVG status {decays[fk].tvg_status} (fit RMS error: {rmse})")
units = analysis.infer_units(values, decays["hz_218740"])
print(f"units: {units.units}")
fig, _ = plot_range_decay_fit(decays["hz_218740"])
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
plot_range_decay_fit(decays["hz_218740"], ax=tax)
if tax.get_legend() is not None:
    tax.get_legend().remove()
tax.set_title("")
tax.set_xlabel("")
tax.set_ylabel("")
thumb.subplots_adjust(0.08, 0.08, 0.98, 0.98)
thumb.savefig(out / "thumb_investigate_a_survey.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# The answer
# ----------
# Raw counts, a clean nadir gap, a usable range of about 116 m at 219 kHz
# and 66 m at 521 kHz, and no gain applied by the recorder at either
# frequency: the level falls with range, and at 219 kHz the model with
# both spreading and absorption fits to within about 1 dB (at 521 kHz,
# with its shorter usable range, absorption alone fits slightly better). So this survey needs the full TVG
# (:func:`~sidescanpy.processing.apply_tvg` with its defaults), and
# anything past the usable range is noise to crop or mask, not seabed.
# Confirm on the corrected data that the level is flat with range: a good
# fit is a hypothesis, not a result.
