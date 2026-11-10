"""
The sonar equation and time-varying gain
========================================

The sonar equation keeps the books on an echo in decibels. The echo
level ``EL`` the receiver sees is the source level ``SL`` the
transducer sends, less the transmission loss ``TL`` on the way out and
again on the way back, plus the backscatter strength ``BS`` of whatever
returned the sound:

.. math:: EL = SL - 2 \\, TL + BS

One way, the sound spreads over a sphere and seawater absorbs part of
it, ``alpha`` dB per meter:

.. math:: TL = 20 \\log_{10} R + \\alpha R

so the echo of a single small target falls as ``40 log10 R + 2 alpha R``
with slant range ``R``.

The seabed is not a single target. Its backscatter strength is a
strength per square meter, ``S_b``, times the patch of seabed the pulse
lights at one time, ``A``:

.. math:: BS = S_b + 10 \\log_{10} A

Along track the patch is as wide as the beam, ``R theta`` for a beam of
horizontal width ``theta``; across track it is set by the pulse and, at
the low grazing angles of sidescan, hardly changes with range. So ``A``
grows in proportion to ``R``, which wins back ``10 log10 R``, and the
seabed echo falls about

.. math:: 30 \\log_{10} R + 2 \\alpha R

(Tamsett, 2017, doi:10.3390/jmse5040054). Fisheries echosounders, which
measure fish spread through a volume, use ``20 log10 R`` for the same
reason: the lit volume grows as ``R^2``.

The levels in a sidescan file are relative (:doc:`plot_amplitude_and_db`),
so ``SL``, the receiver gain and the absolute ``S_b`` are one unknown
constant. Only how the level changes with range matters. Time-varying
gain (TVG) adds back the modelled loss, ``n log10 R + 2 alpha R``, so a
uniform seabed comes out at the same level at all ranges. Many recorders
apply some TVG in hardware and do not record which, so the right model
is fitted to each recording rather than assumed. Absorption is computed
with Francois and Garrison (1982), J. Acoust. Soc. Am. 72(3) 896-907 and
72(6) 1879-1890.

TVG corrects the loss along the path only. What it leaves behind is the
transducer's beam pattern and the way the seabed scatters less at low
grazing angles (Lambert's law: ``S_b`` falls with the square of the
sine of the grazing angle). sidescanpy's empirical range correction
flattens those.

Data: JNCC Wight-Barfleur Reef, EdgeTech JSF (Open Government Licence
v3.0); USGS Grand Bay, Klein XTF (public domain); BGS London Gateway,
Klein 3000 XTF (Open Government Licence v3.0, contains data supplied by
UKRI) (see :doc:`/example_data`).

Usage: ``python examples/theory/plot_sonar_equation_and_tvg.py [output_dir]``
"""

# 2026 rshomberg@uri.edu

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sidescanpy import analysis, datasets
from sidescanpy.io import read_and_convert
from sidescanpy.log import configure_logging
from sidescanpy.processing import (
    altitude_from_bottom_track,
    apply_tvg,
    compute_geometric_ranges,
    slice_by_range,
)
from sidescanpy.processing.bottom import uniform_exclusion_boundary
from sidescanpy.processing.radiometric import (
    center_frequency_hz,
    francois_garrison_absorption_db_per_m,
)

# sphinx_gallery_thumbnail_path = '_build/gallery-output/thumb_sonar_equation_and_tvg.png'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out.mkdir(parents=True, exist_ok=True)
configure_logging("ERROR")

# %%
# Spreading laws and absorption
# -----------------------------
# Two-way loss relative to its value at 10 m, for the three spreading
# laws alone and for 30 log R with absorption at three frequencies
# (Francois-Garrison, default water: 10 C, salinity 35, 10 m depth, pH 8).
# The curves differ most at short range, where ``log10 R`` changes
# fastest; absorption takes over at long range and high frequency.

R = np.linspace(2, 200, 400)
R_REF = 10.0
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
for n, ls in ((20, ":"), (30, "-"), (40, "--")):
    ax.plot(R, n * np.log10(R / R_REF), color="k", ls=ls, lw=1.4, label=f"{n} log R")
for f_khz, colour in ((100, "C2"), (400, "C1"), (548, "C3")):
    a = francois_garrison_absorption_db_per_m(f_khz * 1e3)
    ax.plot(R, 30 * np.log10(R / R_REF) + 2 * a * (R - R_REF), color=colour, lw=2,
            label=f"30 log R + 2 alpha R, {f_khz} kHz (alpha {1000 * a:.0f} dB/km)")
    print(f"{f_khz} kHz: alpha = {a:.4f} dB/m; two-way absorption from 10 to 100 m: 2 x {a:.4f} x 90 = "
          f"{2 * a * 90:.1f} dB")
print(f"spreading from 10 to 100 m: 20 log R {20:.0f} dB, 30 log R {30:.0f} dB, 40 log R {40:.0f} dB")
ax.set_xlim(0, 200)
ax.set_ylim(-25, 110)
ax.set_xlabel("slant range R (m)")
ax.set_ylabel("two-way loss relative to 10 m (dB)")
ax.set_title("What TVG must add back: spreading plus absorption", loc="left")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.savefig(out / "sonar_equation_and_tvg_laws.png", dpi=150)

# %%
# Three recordings
# ----------------
# The high frequency of the Wight-Barfleur line (a chirp from 521 to
# 574 kHz, center 548 kHz; the JSF records no altitude, so it comes from
# the first seabed echo), a Klein line from the shallow Grand Bay
# estuary (100 kHz) and a London Gateway line (Klein 3000, 400 kHz) whose
# recorder applied its own TVG. Incomplete pings are dropped.

RECORDINGS = {
    "Wight-Barfleur, 548 kHz": ("jncc_wight_barfleur_20130321174522.jsf", {"fmt": "jsf"}, "hz_521000"),
    "Grand Bay, 100 kHz": ("usgs_grand_bay_150602205800.xtf", {"fmt": "xtf"}, "hz_100000"),
    "London Gateway, 400 kHz": ("bgs_london_gateway_dynamo_88.xtf", {"fmt": "xtf", "nav_crs": "EPSG:32631"},
                                "hz_400000"),
}
lines = {}
for label, (name, kwargs, key) in RECORDINGS.items():
    ds = read_and_convert([datasets.fetch(name)], **kwargs).get_frequency(key).dataset.load()
    ds = ds.isel(ping=np.isfinite(ds["backscatter_intensity"].values).all(axis=(1, 2)))
    ds = compute_geometric_ranges(altitude_from_bottom_track(ds))
    alpha = francois_garrison_absorption_db_per_m(center_frequency_hz(ds))
    lines[label] = dict(ds=ds, alpha=alpha, altitude=float(np.nanmedian(ds["altitude_m"].values)))
    print(f"{label}: {ds.sizes['ping']} pings, altitude {lines[label]['altitude']:.1f} m (median), "
          f"{float(ds['slant_range_m'].max()):.0f} m recorded, alpha {1000 * alpha:.1f} dB/km")

# %%
# The median level against range
# -------------------------------
# For each line: all samples beyond the altitude (past the water
# column), both sides, binned by slant range in 1 m bins; the median of
# each bin in dB re 1 recorded unit. The two Klein files store whole
# numbers, so each sample is first spread evenly over its unit interval
# (a fixed random offset between -0.5 and 0.5); without it the medians
# stick to whole numbers and the profile has steps.
#
# The fit window starts 5 m past the profile's peak (inside the peak the
# beam pattern dominates) and ends at the noise knee, taken as the first
# range past the peak where the level comes within 3 dB of its lowest
# value. Beyond the knee the level is noise, which no spreading law
# describes.

rng = np.random.default_rng(0)
EDGES = np.arange(0.0, 201.0, 1.0)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])


def median_profile(amp, slant, altitude):
    """Median level (dB re 1) per 1 m slant-range bin, samples past ``altitude`` only."""
    keep = np.isfinite(amp) & (slant > altitude)
    idx = np.digitize(slant[keep], EDGES) - 1
    vals = amp[keep]
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    bounds = np.searchsorted(idx, np.arange(len(CENTRES) + 1))
    med = np.array([np.median(vals[a:b]) if b - a > 50 else np.nan for a, b in zip(bounds[:-1], bounds[1:])])
    return 20 * np.log10(med)


for label, line in lines.items():
    ds = line["ds"]
    amp = ds["backscatter_intensity"].values.astype(np.float64)
    if np.all(amp == np.round(amp)):
        amp = amp + rng.uniform(-0.5, 0.5, amp.shape)
    slant = np.broadcast_to(ds["slant_range_m"].values, amp.shape)
    line["level"] = level = median_profile(amp, slant, line["altitude"])
    line["sides"] = [median_profile(amp[:, s], slant[:, s], line["altitude"]) for s in (0, 1)]
    peak = int(np.nanargmax(level))
    lowest = peak + int(np.nanargmin(level[peak:]))
    knee = CENTRES[peak + int(np.argmax(level[peak:] <= level[lowest] + 3.0))]
    line["window"] = (CENTRES[peak] + 5.0, float(knee))
    print(f"{label}: peak {level[peak]:.1f} dB at {CENTRES[peak]:.1f} m; lowest {level[lowest]:.1f} dB at "
          f"{CENTRES[lowest]:.1f} m; knee {knee:.1f} m; fit window {line['window'][0]:.1f} to {knee:.1f} m")

# %%
# Fitting the spreading law
# -------------------------
# Six fixed models, ``L(R) = L0 - n log10 R - k 2 alpha R`` with
# ``n`` = 20, 30 or 40 and absorption off (``k = 0``) or on (``k = 1``).
# Each has only ``L0`` free, the unknown constant of the sonar equation.
# A seventh lets ``n`` float too, with absorption on. The RMS residual
# over the fit window ranks them.

MODELS = [(n, k) for n in (20, 30, 40) for k in (0, 1)]


def model_name(n, k):
    return f"{n} log R" + (" + abs" if k else "")


def fit_models(level, alpha, window):
    sel = (CENTRES >= window[0]) & (CENTRES <= window[1]) & np.isfinite(level)
    r, y = CENTRES[sel], level[sel]
    fits = {}
    for n, k in MODELS:
        shape = -n * np.log10(r) - k * 2 * alpha * r
        l0 = float(np.mean(y - shape))
        fits[(n, k)] = (l0, float(np.sqrt(np.mean((y - shape - l0) ** 2))))
    design = np.c_[np.ones_like(r), -np.log10(r)]
    coef = np.linalg.lstsq(design, y + 2 * alpha * r, rcond=None)[0]
    free_rms = float(np.sqrt(np.mean((design @ coef - (y + 2 * alpha * r)) ** 2)))
    return fits, (float(coef[1]), float(coef[0]), free_rms), int(sel.sum())


for label, line in lines.items():
    fits, free, nbins = fit_models(line["level"], line["alpha"], line["window"])
    best = min(fits, key=lambda m: fits[m][1])
    line.update(fits=fits, free=free, best=best)
    table = ", ".join(f"{model_name(*m)} {fits[m][1]:.2f}" for m in MODELS)
    print(f"{label} ({nbins} bins): RMS residual (dB): {table}")
    ranked = sorted(fits, key=lambda m: fits[m][1])
    print(f"   best {model_name(*best)} ({fits[best][1]:.2f} dB), next {model_name(*ranked[1])} "
          f"({fits[ranked[1]][1]:.2f} dB); free fit with absorption: n = {free[0]:.1f}, RMS {free[2]:.2f} dB")
    per_side = []
    for side, name in ((0, "port"), (1, "starboard")):
        sf, _, _ = fit_models(line["sides"][side], line["alpha"], line["window"])
        sb = min(sf, key=lambda m: sf[m][1])
        per_side.append(f"{name} {model_name(*sb)} ({sf[sb][1]:.2f} dB)")
    print(f"   each side alone, same window: {'; '.join(per_side)}")
    w0, w1 = line["window"]
    fall = np.interp(w0, CENTRES, line["level"]) - np.interp(w1, CENTRES, line["level"])
    print(f"   the level falls {fall:.1f} dB from {w0:.1f} to {w1:.1f} m; 30 log R + abs predicts "
          f"30 log10({w1 / w0:.2f}) + 2 x {line['alpha']:.4f} x {w1 - w0:.0f} = "
          f"{30 * np.log10(w1 / w0) + 2 * line['alpha'] * (w1 - w0):.1f} dB")

# %%
# What :func:`~sidescanpy.analysis.fit_range_decay` says
# ------------------------------------------------------
# The library's classifier, first called as a survey investigation
# calls it: the usable range from
# :func:`~sidescanpy.analysis.estimate_usable_range` as the far edge, one
# boundary past the first seabed echo for all pings, 5 m bins. Then again
# on the window used above (``r_fit_min_m`` and ``usable_range_m`` set to
# it, 2 m bins, bin medians). It differs from the fit above in two ways:
# it anchors every model at one reference bin instead of fitting ``L0``
# (its "fixed-scale" RMSE), and it compares one spreading coefficient
# ``n`` at a time, with and without absorption, plus absorption alone and
# no loss. From the best candidate it names the TVG the recorder has
# already applied (``tvg_status``: ``none`` means the full model is
# still in the data). Run with ``n`` = 20, 30 and 40 in turn.

STATUS = "{0.tvg_status}, best {0.best_candidate} ({0.best_rmse_db:.2f} dB)"
for label, line in lines.items():
    ds = line["ds"]
    first = analysis.analyze_nadir(ds).first_bottom_return_sample
    usable = analysis.estimate_usable_range(ds, bin_width_m=2.0, exclude_before_sample=first)
    boundary = uniform_exclusion_boundary(first)
    spacing = float(np.nanmedian(np.diff(ds["slant_range_m"].values[0, 0])))
    usable_text = f"{usable.usable_range_m:.0f} m" if usable.usable_range_m else "none found"
    print(f"{label}: usable range {usable_text}; first seabed echo: median sample "
          f"{np.nanmedian(first):.0f} ({np.nanmedian(first) * spacing:.1f} m), boundary for all pings (90th "
          f"percentile) sample {int(np.nanmax(boundary))} ({np.nanmax(boundary) * spacing:.1f} m)")
    for n in (20, 30, 40):
        default = analysis.fit_range_decay(ds, bin_width_m=5.0, usable_range_m=usable.usable_range_m,
                                           exclude_before_sample=boundary, spreading_loss_db=n)
        same = analysis.fit_range_decay(ds, bin_width_m=2.0, usable_range_m=line["window"][1],
                                        r_fit_min_m=line["window"][0], r_ref_m="auto", fit_value_col="median",
                                        spreading_loss_db=n)
        print(f"   n = {n}: default window {default.r_fit_min_m:.1f}-{default.r_fit_max_m:.1f} m: "
              f"{STATUS.format(default)}; window above, reference {same.r_ref_m:.1f} m: {STATUS.format(same)}")

# %%
# Applying the TVG
# ----------------
# A good fit is a hypothesis. To test it, each line is cropped at its
# noise knee with :func:`~sidescanpy.processing.slice_by_range` (beyond
# it TVG would only amplify noise) and corrected with
# :func:`~sidescanpy.processing.apply_tvg` using its own best model. If
# the model is right the corrected median level is flat across the fit
# window.


def corrected_profile(line, n, k):
    ds = slice_by_range(line["ds"], max_m=line["window"][1])
    tvg = apply_tvg(ds, spreading_loss_db=float(n), absorption_db_per_m=k * line["alpha"])
    db = tvg["backscatter_intensity"].values.astype(np.float64)
    slant = np.broadcast_to(ds["slant_range_m"].values, db.shape)
    return median_profile(10 ** (db / 20), slant, line["altitude"]), tvg


def flatness(profile, window):
    sel = (CENTRES >= window[0]) & (CENTRES <= window[1]) & np.isfinite(profile)
    slope = np.polyfit(CENTRES[sel], profile[sel], 1)[0]
    return float(np.std(profile[sel])), float(10 * slope)


for label, line in lines.items():
    line["after"], line["tvg"] = corrected_profile(line, *line["best"])
    spread, slope = flatness(line["after"], line["window"])
    print(f"{label}: after TVG with {model_name(*line['best'])}: spread {spread:.2f} dB, "
          f"trend {slope:+.2f} dB per 10 m across the window")
for label, other in (("Wight-Barfleur, 548 kHz", (20, 1)), ("Grand Bay, 100 kHz", (30, 1))):
    line = lines[label]
    wrong, _ = corrected_profile(line, *other)
    spread, slope = flatness(wrong, line["window"])
    sel = (CENTRES >= line["window"][0]) & (CENTRES <= line["window"][1]) & np.isfinite(wrong)
    print(f"{label} with {model_name(*other)} instead: trend {slope:+.2f} dB per 10 m, "
          f"{wrong[sel][-1] - wrong[sel][0]:+.1f} dB from one end of the window to the other")

# %%
# The Wight-Barfleur swath before and after
# -----------------------------------------
# The raw level in dB and the same pings after
# :func:`~sidescanpy.processing.apply_tvg` with 30 log R and absorption,
# both sides, cropped at the noise knee.

wb = lines["Wight-Barfleur, 548 kHz"]
raw = slice_by_range(wb["ds"], max_m=wb["window"][1])["backscatter_intensity"].values
after = wb["tvg"]["backscatter_intensity"].values
knee = wb["window"][1]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
for ax, data, title in ((axes[0], 20 * np.log10(np.maximum(raw, 1e-6)), "Raw level (dB re 1)"),
                        (axes[1], after, "After TVG: 30 log R + 2 alpha R")):
    swath = np.concatenate([data[:, 0, ::-1], data[:, 1]], axis=1)
    lo, hi = np.percentile(swath[swath > -100], [2, 99.5])
    im = ax.imshow(swath, cmap="gray", vmin=lo, vmax=hi, aspect="auto", origin="lower",
                   extent=[-knee, knee, 0, swath.shape[0]])
    ax.set_title(title, loc="left")
    ax.set_xlabel("slant range, port (-) and starboard (+) (m)")
    fig.colorbar(im, ax=ax, label="dB", shrink=0.85)
axes[0].set_ylabel("ping")
fig.savefig(out / "sonar_equation_and_tvg_swath.png", dpi=150)

# %%
# Fit, apply, check
# -----------------
# Top: the median level against slant range (black) for each line, with
# the 20, 30 and 40 log R models with absorption anchored to it, the best
# model (thick), and the fit window (shaded). Bottom: the median level
# after :func:`~sidescanpy.processing.apply_tvg` with that best model,
# relative to its mean in the window. Flat means the model holds.

fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), height_ratios=[1.5, 1], sharex="col", constrained_layout=True)
for col, (label, line) in enumerate(lines.items()):
    ax, ax_b = axes[0, col], axes[1, col]
    lo_w, hi_w = line["window"]
    sel = (CENTRES >= lo_w) & (CENTRES <= hi_w)
    x_max = min(CENTRES[np.isfinite(line["level"])].max(), 2.2 * hi_w)
    for a in (ax, ax_b):
        a.axvspan(lo_w, hi_w, color="0.92", zorder=0)
        a.grid(alpha=0.3)
    ax.plot(CENTRES, line["level"], color="k", lw=1.5, label="measured median")
    r = CENTRES[(CENTRES >= 0.5 * lo_w) & (CENTRES <= x_max)]
    for (n, k), colour in (((20, 1), "C2"), ((30, 1), "C0"), ((40, 1), "C3"), ((20, 0), "C2"), ((30, 0), "C0"),
                           ((40, 0), "C3")):
        l0 = line["fits"][(n, k)][0]
        is_best = (n, k) == line["best"]
        if k == 1 or is_best:
            ax.plot(r, l0 - n * np.log10(r) - k * 2 * line["alpha"] * r, color=colour, ls="-" if k else "--",
                    lw=3 if is_best else 1, alpha=1 if is_best else 0.8,
                    label=f"{model_name(n, k)}: {line['fits'][(n, k)][1]:.2f} dB" + (" (best)" if is_best else ""))
    ax.set_xlim(0, x_max)
    finite = line["level"][CENTRES <= x_max]
    ax.set_ylim(np.nanmin(finite) - 3, np.nanmax(finite) + 4)
    ax.set_title(label, loc="left")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95, title="RMS in window", title_fontsize=8.5)
    rel = line["after"] - np.nanmean(line["after"][sel])
    spread, slope = flatness(line["after"], line["window"])
    shown = CENTRES >= lo_w - 3.0
    ax_b.plot(CENTRES[shown], rel[shown], color="k", lw=1.8)
    ax_b.axhline(0, color="0.4", lw=0.8)
    ax_b.set_ylim(-5, 5)
    ax_b.set_xlabel("slant range (m)")
    ax_b.text(0.02, 0.88, f"TVG {model_name(*line['best'])}: spread {spread:.2f} dB,\n"
              f"trend {slope:+.2f} dB per 10 m", transform=ax_b.transAxes, fontsize=8.5, va="top")
axes[0, 0].set_ylabel("median level (dB re 1 recorded unit)")
axes[1, 0].set_ylabel("after TVG, relative (dB)")
fig.savefig(out / "sonar_equation_and_tvg.png", dpi=150)
# sphinx_gallery_start_ignore
# A clean thumbnail for the examples page (hidden on this page).
thumb, tax = plt.subplots(figsize=(4, 2.8))
tax.plot(CENTRES, wb["level"], color="k", lw=2)
best_l0 = wb["fits"][wb["best"]][0]
r = CENTRES[(CENTRES > 15) & (CENTRES < 110)]
tax.plot(r, best_l0 - wb["best"][0] * np.log10(r) - 2 * wb["alpha"] * r, color="C0", lw=2.5)
tax.set_xlim(10, 130)
tax.set_axis_off()
thumb.subplots_adjust(0, 0, 1, 1)
thumb.savefig(out / "thumb_sonar_equation_and_tvg.png", dpi=100)
plt.close(thumb)
# sphinx_gallery_end_ignore

# %%
# Reading the result
# ------------------
# **Each line needs its own model.** Over its fit window (27.5 to
# 88.5 m) the Wight-Barfleur high frequency falls 32.6 dB, and
# 30 log R with absorption predicts 32.3 dB:
# ``30 log10(88.5 / 27.5) + 2 x 0.1402 x 61 = 32.3``. That model fits
# best (0.80 dB RMS), with 40 log R plus absorption close behind
# (1.02 dB); letting ``n`` float gives 34.0 at 0.55 dB, between the two.
# The Grand Bay line falls only 16.1 dB over 10.5 to 68.5 m, where
# 30 log R with absorption predicts 28.3 dB. Its best fixed model is
# 20 log R without absorption (1.03 dB), and a free ``n`` with
# absorption comes out at 12.0. Either the recorder applied part of the
# gain itself, or the estuary absorbs less than the default seawater
# assumed here; the file records neither. The London Gateway line falls
# 14.3 dB over 16.5 to 36.5 m, close to the 14.2 dB of 30 log R with
# absorption (0.42 dB), but 40 log R without absorption fits almost as
# well (0.45 dB). Its usable stretch is only 20 m long, and over so short
# a span the curves are too alike to tell apart. The archive says this
# recorder applied its own TVG; whatever gain that was, the stored
# levels still fall about as fast as an uncorrected seabed echo.
#
# **Applying the fitted TVG confirms it.** After
# :func:`~sidescanpy.processing.apply_tvg` with each line's best model,
# the median level across the window varies by 0.80, 1.01 and 0.42 dB
# (standard deviation) with trends of -0.32, +0.31 and -0.28 dB per
# 10 m: flat to within a fraction of a decibel. The wrong model shows at
# once: 20 log R with absorption on the Wight-Barfleur line leaves
# -5.5 dB from one end of the window to the other, and 30 log R with
# absorption on the Grand Bay line +12.8 dB. Pooling both sides can hide
# differences too: on their own, the Wight-Barfleur port side prefers
# 40 log R with absorption and the Grand Bay starboard side 20 log R
# with absorption, while the Grand Bay port side, which holds bright
# patches of seabed, fits no model well (4.00 dB).
#
# **The library classifier agrees once the window matches.**
# :func:`~sidescanpy.analysis.fit_range_decay` with its default window
# (from the first seabed echo to the usable range) starts at the beam's
# peak and, at Wight-Barfleur, stops at 66 m; at Grand Bay
# :func:`~sidescanpy.analysis.estimate_usable_range` found no usable
# range and the first-echo boundary landed at 37.4 m, although the
# median first echo is at 2.7 m (a tenth of the pings have a first echo
# detected far out), so the default fit looks only at 37.5 to 73.5 m.
# On those windows it ranks the models differently. Given the window
# used above, it names the same models: ``full_physics`` (``tvg_status``
# ``none``) at ``n = 30`` for Wight-Barfleur and London Gateway, and
# 20 log R without absorption (``spreading_only``, status
# ``absorption_only``) for Grand Bay. Its RMS values are higher (1.13
# against 0.80 dB at Wight-Barfleur) because it pins each model to one
# reference bin instead of fitting ``L0``, and at Wight-Barfleur it
# prefers ``n = 40`` (0.92 dB) to ``n = 30`` by that measure. The free
# fit's 34.0 explains the tie: the data sit between the two.
#
# **What TVG leaves.** Near the start of each window the corrected level
# still bends (lower panels, and the rise just outside the water column
# in the swath above): that is the beam pattern and the change of
# backscatter with grazing angle, which no range law removes.
# :func:`~sidescanpy.processing.range_correction.fit_range_correction`
# measures that leftover curve on TVG-corrected data and
# :func:`~sidescanpy.processing.range_correction.apply_empirical_range_correction`
# removes it. The workflow is the same each time: fit the decay within
# the usable range, apply the TVG, and check that the result is flat
# before trusting it. :doc:`plot_range_limits` shows where the usable
# range ends and :doc:`/examples/plot_correct_a_line` the whole
# correction chain.
