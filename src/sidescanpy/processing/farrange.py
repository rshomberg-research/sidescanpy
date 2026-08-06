"""
Far-range noise detection and cropping.

Some surveys record a channel's sample window well past where real
backscatter decays into noise — a per-survey acquisition setting, not
fixed by frequency (found 2026-08-06, see ``sidescanpy/CLAUDE.md`` Known
Issue #6 / ``docs/planning/handoff-hf-farrange-noise-amplification.md``).
Past that point, raw amplitude doesn't stay flat — it climbs back up,
new evidence for a hardware TVG/AGC artifact baked into the raw signal —
which ``apply_tvg()``'s uncapped, monotonically-increasing gain formula
then massively over-amplifies (a ~57 dB overshoot measured on real data),
saturating percentile-based color scales and burying the real near-nadir
signal. This module detects that decay-then-rise pattern directly from
raw data (candidate solution (a) in the handoff doc) and crops it out —
intentionally conservative: a channel with no detected rise is left
untouched, not cropped by a guessed constant, since the recorded window
length varies per survey.

Binning is done by raw sample index, not real range — this needs no
geometric computation (no aux interpolation, no altitude), so it can run
immediately after conversion, before any other correction step.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np
import xarray as xr

from sidescanpy.core.survey import SideScanSurvey

logger = logging.getLogger(__name__)

_MIN_LINEAR = 1e-6


def _linear_to_db(x: np.ndarray, min_linear: float = _MIN_LINEAR) -> np.ndarray:
    """Convert linear amplitude to dB, matching ``apply_tvg()``'s step-1
    arithmetic (``20*log10(clip(x, 1e-6))``) — deliberately duplicated
    rather than imported, same rationale as ``analysis/tvg_diagnostics.py``'s
    copy: this must not require modifying the module under investigation.
    """
    return 20.0 * np.log10(np.clip(np.asarray(x, dtype=np.float64), min_linear, None))


# Known Issue #5: a leading run of exact-zero raw amplitude (blanking/
# ring-down) at the *start* of every ping converts to exactly this dB
# floor via _linear_to_db's clip(x, 1e-6). Found 2026-08-06: left
# unmasked, this run's artificially low mean gets picked as "the floor"
# by the bin-minimum search below, and the real (much stronger)
# near-nadir signal in the very next bin then reads as a spurious "rise"
# — producing a false-positive crop right at the start of the array.
# Mirrors viz/waterfall.py's _mask_floor_clip (same value, same cause),
# duplicated rather than imported per this module's own precedent above.
_FLOOR_CLIP_DB = -120.0


def _mask_floor_clip(db: np.ndarray) -> np.ndarray:
    """Replace Known Issue #5's floor-clip artifact (exact -120dB) with
    NaN so it's excluded from bin statistics — it is not real low-
    amplitude signal, it's a blanking/ring-down artifact."""
    mask = db <= _FLOOR_CLIP_DB
    if not mask.any():
        return db
    return np.where(mask, np.nan, db)


def detect_far_range_cutoff_sample(
    ds: xr.Dataset,
    *,
    n_bins: int = 40,
    min_valid_fraction: float = 0.5,
    rise_threshold_db: float = 3.0,
    margin_bins: int = 1,
) -> int | None:
    """Detect the sample index beyond which raw backscatter climbs back
    up after decaying into its real noise floor.

    Bins ``backscatter_raw`` (converted to dB) by sample index across
    ``n_bins`` roughly-equal bins, finds the peak bin (the strongest
    mean return) among bins with enough finite samples to be
    statistically reliable, then finds the minimum-mean bin *after* that
    peak (the real far-range floor — searched only post-peak so a
    legitimate near-nadir ramp-up, ordinary geometry/beam-pattern
    behavior before reaching the real peak a few bins in, is never
    mistaken for the floor). If any later, equally-reliable bin's mean
    then rises ``rise_threshold_db`` or more above that floor, returns a
    cropped sample count (the floor bin's end, plus ``margin_bins`` of
    slack so real signal right at the boundary isn't clipped by bin
    granularity). If no such rise is found, returns ``None`` — this
    channel's recorded window doesn't show the artifact and nothing
    should be cropped; a fixed constant would not generalize across
    surveys with different recorded window lengths.

    Parameters
    ----------
    ds : xr.Dataset
        A single frequency channel dataset, raw (before TVG) —
        ``backscatter_raw`` and ``backscatter_intensity`` are identical
        at this point, so either works, but pass the raw dataset to be
        explicit.
    n_bins : int, default 40
        Number of sample-index bins to divide the sample axis into.
    min_valid_fraction : float, default 0.5
        A bin is "reliable" only if at least this fraction of its
        ``n_pings * n_channels * bin_size`` cells are finite.
    rise_threshold_db : float, default 3.0
        How far above the detected floor a later, reliable bin's mean
        must climb to count as "the noise climbing back up," not just
        ordinary scatter around the floor.
    margin_bins : int, default 1
        Extra bins kept past the detected floor bin.

    Returns
    -------
    int or None
        Sample count to crop to (``ds.isel(sample=slice(0, result))``),
        or ``None`` if no crop is warranted.
    """
    bs = ds["backscatter_raw"].values  # (ping, channel, sample)
    n_pings, n_channels, n_samples = bs.shape

    bin_size = max(1, n_samples // n_bins)
    n_bins_actual = n_samples // bin_size
    if n_bins_actual < 3:
        return None

    db = _mask_floor_clip(_linear_to_db(bs))
    min_count = int(n_pings * n_channels * bin_size * min_valid_fraction)

    bin_means = np.full(n_bins_actual, np.nan)
    bin_counts = np.zeros(n_bins_actual, dtype=np.int64)
    for i in range(n_bins_actual):
        lo, hi = i * bin_size, (i + 1) * bin_size
        chunk = db[:, :, lo:hi]
        finite = chunk[np.isfinite(chunk)]
        bin_counts[i] = finite.size
        if finite.size > 0:
            bin_means[i] = float(np.mean(finite))

    reliable = bin_counts >= min_count
    if not reliable.any():
        return None

    reliable_idx = np.where(reliable)[0]

    # The floor must be searched for only *after* the peak, not globally.
    # Real-data bug found 2026-08-06 (survey-a, hz_1495000): a
    # legitimate near-nadir ramp-up (weak returns very close to nadir,
    # before reaching the real peak a few bins in -- ordinary geometry/
    # beam-pattern behavior, not an artifact) can be *lower* than the
    # genuine far-range floor. A global-minimum search picks that
    # near-nadir ramp-up as "the floor," and the normal rise from there
    # up to the real peak then reads as a huge spurious "rise" -- cropping
    # everything after the first couple of bins on a channel with no
    # far-range problem at all.
    peak_pos = int(np.argmax(bin_means[reliable_idx]))
    peak_bin = int(reliable_idx[peak_pos])
    post_peak_idx = reliable_idx[reliable_idx > peak_bin]
    if post_peak_idx.size == 0:
        return None

    floor_pos = int(np.argmin(bin_means[post_peak_idx]))
    floor_bin = int(post_peak_idx[floor_pos])
    floor_mean = float(bin_means[floor_bin])

    later_reliable = reliable_idx[reliable_idx > floor_bin]
    if later_reliable.size == 0:
        return None

    rises = bin_means[later_reliable] >= floor_mean + rise_threshold_db
    if not rises.any():
        return None

    cutoff_bin = floor_bin + margin_bins
    cutoff_sample = min((cutoff_bin + 1) * bin_size, n_samples)
    return cutoff_sample


def crop_far_range_noise(
    survey: SideScanSurvey,
    **detect_kwargs,
) -> tuple[SideScanSurvey, dict[str, int | None]]:
    """For every frequency channel in ``survey``, detect and crop the
    far-range noise artifact (see ``detect_far_range_cutoff_sample``).
    Channels with no detected artifact are left untouched — this is a
    no-op safety valve per channel, not a guaranteed crop.

    Parameters
    ----------
    survey : SideScanSurvey
        Mutated in place (frequency channels replaced where cropped) and
        returned for chaining, matching ``run_correction_steps``'s
        convention.
    **detect_kwargs
        Passed through to ``detect_far_range_cutoff_sample``.

    Returns
    -------
    (survey, cutoffs)
        ``cutoffs``: ``freq_key -> cropped sample count``, or ``None``
        for channels left unchanged.
    """
    cutoffs: dict[str, int | None] = {}
    for freq_key, channel in list(survey.freq_channels.items()):
        n_samples = channel.n_samples
        cutoff = detect_far_range_cutoff_sample(channel.dataset, **detect_kwargs)
        cutoffs[freq_key] = cutoff
        if cutoff is not None and cutoff < n_samples:
            cropped_ds = channel.dataset.isel(sample=slice(0, cutoff))
            cropped_ds.sidescan.append_to_history(
                step_name="crop_far_range_noise",
                function_name="sidescanpy.processing.farrange.crop_far_range_noise",
                params={"cropped": True, "original_n_samples": n_samples, "cropped_n_samples": cutoff},
                version=_version(),
            )
            survey.add_frequency(freq_key, cropped_ds)
            logger.info(
                f"{freq_key}: cropped far-range noise, "
                f"{n_samples} -> {cutoff} samples"
            )
        else:
            # No crop applied, but still record that detection ran and
            # found nothing — without this, a channel that never needs
            # cropping would never gain a processing_history entry for
            # this step, and callers checking "has this file already
            # been through crop_far_range_noise?" (e.g. this pipeline's
            # 02_crop_farrange.py, to skip already-processed files on
            # rerun) would keep re-scanning it on every run.
            unchanged_ds = channel.dataset
            unchanged_ds.sidescan.append_to_history(
                step_name="crop_far_range_noise",
                function_name="sidescanpy.processing.farrange.crop_far_range_noise",
                params={"cropped": False},
                version=_version(),
            )
            survey.add_frequency(freq_key, unchanged_ds)
            logger.debug(f"{freq_key}: no far-range noise artifact detected, left unchanged")
    return survey, cutoffs


def _version() -> str:
    from sidescanpy import __version__
    return __version__
