"""
How far a sidescan sees
=======================

The echo from the seabed gets weaker the farther it travels. The sound
spreads out on its way to the seabed and back, and seawater absorbs part
of it, more the higher the frequency. The usable range ends where the
echo sinks into the noise floor: the level the receiver records with no
seabed echo at all, from its own electronics, the sea and the towing
platform.

For a flat seabed the recorded echo level falls with slant range ``R``
roughly as

.. math:: L(R) = L_0 - 30 \\log_{10} R - 2 \\alpha R

where ``L_0`` collects the source level, the receiver's gain and the
seabed's backscatter strength, and ``alpha`` is the absorption
coefficient in dB per meter, counted twice for the way out and back.
The 30 is two-way spherical spreading (40 log R) less the 10 log R by
which the patch of seabed lit by the pulse grows with range (Tamsett,
2017, doi:10.3390/jmse5040054). Absorption rises steeply with frequency
(Francois and Garrison, 1982, J. Acoust. Soc. Am. 72(3) 896-907 and
72(6) 1879-1890), so a high frequency gives finer detail but a shorter
range.

Timing limits range too. The sonar pings every ``T`` seconds, a pulse
repetition frequency ``PRF = 1 / T``. Echoes have to arrive before the
next ping goes out, so the longest range the recorder can place without
doubt is

.. math:: R_{max} = \\frac{c}{2 \\, PRF} = \\frac{c \\, T}{2}

An echo from farther out arrives after the next ping and is drawn in
that ping at the wrong range (range ambiguity). At the near end, the
receiver is blanked or still ringing while the pulse goes out, so the
first samples of a ping hold zeros or the transmit pulse itself.

This page measures all three limits on one public line recorded at two
frequencies at once.

Data: JNCC Wight-Barfleur Reef, EdgeTech JSF, Open Government Licence
v3.0 (see :doc:`/example_data`).

Usage: ``python examples/theory/plot_range_limits.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import analysis, datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import altitude_from_bottom_track, compute_geometric_ranges
from sidescanpy.processing.radiometric import (
    center_frequency_hz,
    francois_garrison_absorption_db_per_m,
)

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_range_limits.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
# The JSF file records no sound velocity; ranges use 1500 m/s, the value
# the recorder wrote into its XTF copy of the same pings.
configure_logging("ERROR")
C = 1500.0

# %%
# The line, both frequencies
# --------------------------
# The sonar recorded a low and a high frequency on every ping. Each is a
# frequency sweep (chirp); :func:`~sidescanpy.processing.radiometric.center_frequency_hz`
# gives its center. The first ping is incomplete and is dropped. The
# altitude comes from the first seabed echo (``bottom_track``).

survey = read_and_convert([datasets.fetch("jncc_wight_barfleur_20130321174522.jsf")], fmt="jsf")
lines = {}
for key, name in zip(survey.frequencies, ("low", "high")):
    ds = survey.get_frequency(key).dataset.load()
    ds = ds.isel(ping=np.isfinite(ds["backscatter_intensity"].values).all(axis=(1, 2)))
    ds = compute_geometric_ranges(altitude_from_bottom_track(ds))
    lines[name] = ds
    fs = ds.attrs["sample_rate_hz"]
    print(f"{name}: {key}, sweep {ds.attrs['freq_hz_start'] / 1e3:.2f}-{ds.attrs['freq_hz_end'] / 1e3:.2f} kHz, "
          f"center {center_frequency_hz(ds) / 1e3:.2f} kHz; {ds.sizes['ping']} pings, {ds.sizes['sample']} samples "
          f"at {fs:.0f} Hz = {C * ds.sizes['sample'] / fs / 2:.1f} m recorded")

# %%
# Timing: the ping interval
# -------------------------
# The time between pings from the ping time stamps, and the longest
# range it allows.

seconds = (lines["high"]["time"].values - lines["high"]["time"].values[0]) / np.timedelta64(1, "s")
interval = np.diff(seconds)
T = float(np.median(interval))
r_max = C * T / 2
recorded = C * lines["high"].sizes["sample"] / lines["high"].attrs["sample_rate_hz"] / 2
print(f"ping interval: median {T:.3f} s (from {interval.min():.3f} to {interval.max():.3f} s), PRF {1 / T:.2f} Hz")
print(f"R_max = c / (2 PRF) = {C:.0f} x {T:.3f} / 2 = {r_max:.1f} m; recorded range {recorded:.1f} m "
      f"({recorded / r_max:.0%} of it)")
print(f"an echo from 250 m would arrive {2 * 250 / C - T:.3f} s after the next ping, drawn at "
      f"{250 - r_max:.1f} m in it")

# %%
# The near end: the first samples
# -------------------------------
# What each ping holds before the water column: count the samples that
# are exactly zero at the start of each ping, and average the first
# 12 ms over all pings and both sides to look for a transmit pulse.

WINDOW_S = 0.012
fig, (ax_mean, ax_hist) = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
for name, colour in (("low", "C0"), ("high", "C3")):
    ds = lines[name]
    fs = ds.attrs["sample_rate_hz"]
    amp = ds["backscatter_intensity"].values
    lead = np.argmax(amp > 0, axis=2)          # leading zeros per ping and side
    n = int(WINDOW_S * fs)
    mean_start = amp[:, :, :n].mean(axis=(0, 1))
    t_ms = 1000 * np.arange(n) / fs
    settled = mean_start[t_ms >= 10.0].mean()
    half = t_ms[np.argmax(mean_start >= settled / 2)]
    print(f"{name}: leading zeros {np.median(lead):.0f} samples (median; {lead.min()} to {lead.max()}) = "
          f"{1000 * np.median(lead) / fs:.2f} ms = {C * np.median(lead) / fs / 2:.2f} m of range")
    print(f"   mean level 10-12 ms after transmit {settled:.2f}; the mean reaches half of that at {half:.1f} ms; "
          f"largest mean in the first 12 ms {mean_start.max():.2f}")
    ax_mean.plot(t_ms, mean_start, color=colour, label=f"{name} ({center_frequency_hz(ds) / 1e3:.0f} kHz)")
    ax_hist.hist(1000 * lead.ravel() / fs, bins=40, color=colour, alpha=0.6, label=name)
ax_mean.set_xlabel("time since transmit (ms)")
ax_mean.set_ylabel("mean amplitude, all pings")
ax_mean.set_title("Start of each ping: zeros, then a slow rise", loc="left")
ax_mean.legend()
ax_hist.set_xlabel("length of the leading zeros (ms)")
ax_hist.set_ylabel("pings and sides")
ax_hist.set_title("The zero stretch varies from ping to ping", loc="left")
ax_hist.legend()
fig.savefig(out / "range_limits_first_samples.png", dpi=150)

# %%
# The far end: where the echo meets the noise
# -------------------------------------------
# :func:`~sidescanpy.analysis.estimate_usable_range` bins all samples
# beyond the first seabed echo by slant range (2 m bins, both sides),
# takes the median of the farthest tenth of the bins as the noise floor
# and reports the last bin still 3 noise standard deviations above it.
# The same binned medians, in dB, show the knee directly: the range
# where the level stops falling, taken here as the first bin within
# 1 dB of the lowest bin.

profiles = {}
for name in ("low", "high"):
    ds = lines[name]
    first = analysis.analyze_nadir(ds).first_bottom_return_sample
    usable = analysis.estimate_usable_range(ds, exclude_before_sample=first)
    binned = usable.binned
    centre = 0.5 * (binned["range_min_m"].values + binned["range_max_m"].values)
    level = 20 * np.log10(binned["median"].values)
    lowest = int(np.argmin(level))
    # No knee inside the record if the lowest bin is in the last tenth.
    knee = float(centre[np.argmax(level <= level[lowest] + 1.0)]) if lowest < 0.9 * len(level) else None
    profiles[name] = dict(range=centre, level=level, usable=usable.usable_range_m, knee=knee,
                          noise=20 * np.log10(usable.noise_floor), alpha=francois_garrison_absorption_db_per_m(
                              center_frequency_hz(ds)), freq=center_frequency_hz(ds))
    threshold = usable.noise_floor + usable.snr_threshold * usable.noise_std
    print(f"{name}: usable range (estimate_usable_range) {usable.usable_range_m:.0f} m; noise floor "
          f"{usable.noise_floor:.2f} ({20 * np.log10(usable.noise_floor):.1f} dB re 1), noise std {usable.noise_std:.2f}; "
          f"a bin counts as signal above {usable.noise_floor:.2f} + {usable.snr_threshold:.0f} x {usable.noise_std:.2f} = "
          f"{threshold:.2f} ({20 * np.log10(threshold):.1f} dB re 1)")
    knee_text = f"within 1 dB of it from {knee:.0f} m" if knee else "still falling at the end of the record: no knee"
    print(f"   lowest binned level {level[lowest]:.1f} dB at {centre[lowest]:.0f} m; {knee_text}; "
          f"level at the end of the record {level[-1]:.1f} dB")

# %%
# Absorption against frequency
# ----------------------------
# :func:`~sidescanpy.processing.radiometric.francois_garrison_absorption_db_per_m`
# with its default water (10 C, salinity 35, 10 m depth, pH 8; the file
# records no temperature or salinity). At the two center frequencies:

for name in ("low", "high"):
    p = profiles[name]
    print(f"{name} ({p['freq'] / 1e3:.0f} kHz): alpha = {1000 * p['alpha']:.1f} dB/km; two-way over 100 m: "
          f"2 x {p['alpha']:.4f} x 100 = {200 * p['alpha']:.1f} dB")

# %%
# The loss model against the data
# -------------------------------
# Anchor ``L(R) = L0 - 30 log10 R - 2 alpha R`` on the measured level at
# 40 m, then add the measured noise floor (as power), and compare with
# the binned medians farther out.

R0 = 40.0
for name in ("low", "high"):
    p = profiles[name]
    l0 = np.interp(R0, p["range"], p["level"])
    echo = l0 - 30 * np.log10(p["range"] / R0) - 2 * p["alpha"] * (p["range"] - R0)
    p["model"] = 10 * np.log10(10 ** (echo / 10) + 10 ** (p["noise"] / 10))
    p["echo"] = echo
    at = 90.0
    print(f"{name}: from {R0:.0f} to {at:.0f} m the model falls 30 log10({at / R0:.2f}) + 2 x {p['alpha']:.4f} x "
          f"{at - R0:.0f} = {30 * np.log10(at / R0) + 2 * p['alpha'] * (at - R0):.1f} dB; the data fall "
          f"{l0 - np.interp(at, p['range'], p['level']):.1f} dB")
    echo_250 = l0 - 30 * np.log10(250 / R0) - 2 * p["alpha"] * (250 - R0)
    print(f"   an echo from 250 m: model level {echo_250:.1f} dB re 1, against a noise floor of {p['noise']:.1f} dB")

# %%
# Absorption, levels and limits together
# --------------------------------------
# Left: absorption against frequency, with the two frequencies of this
# line. Right: the binned median level against slant range for both
# (solid), the spreading-plus-absorption model with the noise floor
# (dashed), the noise knee (circle), the usable range from
# :func:`~sidescanpy.analysis.estimate_usable_range` (triangle), and the
# end of the record and ``c / (2 PRF)`` (grey lines).

freqs = np.logspace(4, 6, 200)
fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(12.5, 5.2), width_ratios=[1, 1.5], constrained_layout=True)
ax_a.loglog(freqs / 1e3, 1000 * francois_garrison_absorption_db_per_m(freqs), color="0.2")
for name, colour in (("low", "C0"), ("high", "C3")):
    p = profiles[name]
    ax_a.plot(p["freq"] / 1e3, 1000 * p["alpha"], "o", color=colour, ms=8,
              label=f"{p['freq'] / 1e3:.0f} kHz: {1000 * p['alpha']:.0f} dB/km")
    ax_l.plot(p["range"], p["level"], color=colour, lw=2, label=f"{name}, {p['freq'] / 1e3:.0f} kHz, measured")
    ax_l.plot(p["range"], p["model"], "--", color=colour, lw=1.2, label="model: 30 log R + 2 alpha R, plus floor")
    if p["knee"]:
        ax_l.plot(p["knee"], np.interp(p["knee"], p["range"], p["level"]), "o", color=colour, ms=9, mfc="none", mew=2)
    ax_l.plot(p["usable"], np.interp(p["usable"], p["range"], p["level"]), "v", color=colour, ms=9)
ax_a.set_xlabel("frequency (kHz)")
ax_a.set_ylabel("absorption (dB/km, one way)")
ax_a.set_title("Francois-Garrison absorption", loc="left")
ax_a.grid(alpha=0.3, which="both")
ax_a.legend(loc="upper left")
for x, label in ((recorded, "end of record"), (r_max, "c / (2 PRF)")):
    ax_l.axvline(x, color="0.5", lw=1)
    ax_l.text(x - 1.5, 1.5, label, rotation=90, ha="right", va="bottom", fontsize=9, color="0.3")
ax_l.set_xlabel("slant range (m)")
ax_l.set_ylabel("median level (dB re 1 recorded unit)")
ax_l.plot([], [], "o", color="0.3", ms=8, mfc="none", mew=2, label="noise knee (within 1 dB of the lowest)")
ax_l.plot([], [], "v", color="0.3", ms=8, label="usable range (estimate_usable_range)")
ax_l.set_title("Echo level against range, both frequencies", loc="left")
ax_l.set_xlim(0, 220)
ax_l.set_ylim(0, 68)
ax_l.grid(alpha=0.3)
ax_l.legend(loc="upper right", fontsize=9, framealpha=0.95)
fig.savefig(out / "range_limits.png", dpi=150)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
for name, colour in (("low", "C0"), ("high", "C3")):
    tax.plot(profiles[name]["range"], profiles[name]["level"], color=colour, lw=2.5)
    tax.plot(profiles[name]["range"], profiles[name]["model"], "--", color=colour, lw=1.2)
tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_range_limits.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# Reading the result
# ------------------
# **Absorption sets the range at the high frequency.** Francois and
# Garrison give 59.8 dB/km at 230 kHz and 140.2 dB/km at 548 kHz in
# the default water, so over 100 m of slant range the round trip loses
# 12.0 dB to absorption at the low frequency and 28.0 dB at the high
# one. With 30 log R spreading added, the model predicts a fall of
# 16.5 dB (low) and 24.6 dB (high) from 40 to 90 m; the measured
# medians fall 18.4 and 25.5 dB. The seabed is not uniform, the water
# is not exactly the default and the recorder may add gain of its own,
# so this is a check of the shape, not a fit.
#
# **The noise knee and the usable range differ.** At the high
# frequency the median level stops falling at about 99 m (within 1 dB of
# its lowest, 8.0 dB re 1 at 115 m).
# :func:`~sidescanpy.analysis.estimate_usable_range` is more cautious
# and gives 66 m: it takes the farthest tenth of the record as the
# noise floor (3.81, 11.6 dB re 1) and asks for a median 3 noise standard
# deviations above it, 10.39 units or 20.3 dB re 1. Noise in sidescan
# is speckled, its standard deviation (2.19) a large share of its level,
# so that test asks for about 9 dB of signal over the floor. Beyond
# 115 m the high-frequency floor rises again, to 12.0 dB at the end of
# the record; something in the recorder or the water adds level with
# time, which the file does not explain. Both numbers are useful: 66 m
# is where the seabed is clearly above the noise, 99 m is where any
# seabed signal is gone.
#
# **The low frequency runs out of record, not of signal.** Its median is
# still falling at the end of the record (22.9 dB re 1 at 195 m, the
# lowest bin), so no knee lies inside it. The usable range of 116 m
# from :func:`~sidescanpy.analysis.estimate_usable_range` is then too
# short for a different reason: the "noise floor" it takes from the last
# tenth of the record (14.50, 23.2 dB re 1) is still partly seabed echo.
#
# **Timing.** The sonar pinged every 0.281 s (PRF 3.56 Hz), which allows
# ``c / (2 PRF) = 1500 x 0.281 / 2 = 210.8 m``. The recorder kept
# 199.9 m of each ping, 95 % of that. An echo from 250 m would arrive
# 0.052 s into the next ping and be drawn there at 39.2 m. The model
# puts such an echo at 6.4 dB re 1 at the low frequency and -47.0 dB at
# the high one, both below the measured floors (23.2 and 11.6 dB), so
# range ambiguity is not expected on this line; with a slower ping rate
# or a lower frequency it can be.
#
# **The near end holds zeros, not a transmit pulse.** Each ping starts
# with a run of exact zeros: a median 166 samples at the low frequency
# and 169 at the high one, about the same number of samples, so twice as
# long in time at the low frequency, which samples half as fast (3.82
# and 1.95 ms). The run varies from ping to ping (127 to 232 and 143 to
# 215 samples). After it the mean level rises smoothly, reaching half of
# its level 10 to 12 ms after transmit at 6.2 ms (low) and 4.1 ms
# (high), with no spike from the outgoing pulse. Whether that is the
# receiver's blanking ending, a gain ramp, or both, the file does not
# say. The seabed echo arrives far later, so on this line the near-end
# limit does not cost any seabed.
#
# *In sidescanpy:* :func:`~sidescanpy.analysis.estimate_usable_range`
# measures the usable range;
# :func:`~sidescanpy.processing.slice_by_range` cuts the record at it,
# and :func:`~sidescanpy.processing.apply_tvg` compensates spreading
# and absorption (with
# :func:`~sidescanpy.processing.radiometric.francois_garrison_absorption_db_per_m`
# for ``alpha``) up to the point where only noise is left to amplify.
# :doc:`plot_amplitude_and_db` explains the dB levels used here and
# :doc:`plot_how_sidescan_sees` the geometry.
