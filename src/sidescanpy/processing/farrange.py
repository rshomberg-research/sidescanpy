"""
Far-range noise detection and cropping.

Some surveys record a channel's sample window well past where real
backscatter decays into noise — a per-survey acquisition setting, not
fixed by frequency. Past that point, raw amplitude doesn't stay flat — it climbs back up,
new evidence for a hardware TVG/AGC artifact baked into the raw signal —
which ``apply_tvg()``'s uncapped, monotonically-increasing gain formula
then massively over-amplifies (a ~57 dB overshoot measured on real data),
saturating percentile-based color scales and burying the real near-nadir
signal. This module detects that decay-then-rise pattern directly from
the uncorrected signal (and crops it out —
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

from sidescanpy._version import __version__
from sidescanpy.core.history import STEP_CROP_FARRANGE, append_history, read_history
from sidescanpy.core.survey import SideScanSurvey
from sidescanpy.core.units import linear_to_db, mask_floor_db
from sidescanpy.processing._per_frequency import per_frequency
from sidescanpy.processing.inversion import uncorrected_backscatter
from sidescanpy.processing.sample_axis import (
    invalid_sound_vel_warning_suppressed,
    sample_index,
    sample_interval_m,
)

logger = logging.getLogger(__name__)

# A leading run of exact-zero raw amplitude (blanking/ring-down) at the
# *start* of every ping converts to exactly the dB floor. Left unmasked,
# this run's artificially low mean gets picked as "the floor" by the
# bin-minimum search below, and the real (much stronger) near-nadir
# signal in the very next bin then reads as a spurious "rise" --
# producing a false-positive crop right at the start of the array.
# mask_floor_db() turns it into NaN so it is excluded from bin statistics.


_DETECT_PING_BLOCK = 1024
"""Pings per block when accumulating bin statistics (bounds memory; the
result does not depend on it beyond float summation order)."""


def _bin_stats(ds: xr.Dataset, bin_size: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin sum and count of finite, non-floor dB values of the uncorrected signal."""
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    n_pings = ds.sizes["ping"]
    for start in range(0, n_pings, _DETECT_PING_BLOCK):
        block = ds.isel(ping=slice(start, start + _DETECT_PING_BLOCK))
        values = np.asarray(uncorrected_backscatter(block).values, dtype=np.float64)
        db = mask_floor_db(linear_to_db(values[:, :, : n_bins * bin_size]))
        db = db.reshape(db.shape[0], db.shape[1], n_bins, bin_size)
        finite = np.isfinite(db)
        sums += np.where(finite, db, 0.0).sum(axis=(0, 1, 3))
        counts += finite.sum(axis=(0, 1, 3))
    return sums, counts


def detect_far_range_cutoff_sample(
    ds: xr.Dataset,
    *,
    n_bins: int = 40,
    min_valid_fraction: float = 0.5,
    rise_threshold_db: float = 3.0,
    margin_bins: int = 1,
) -> int | None:
    """Detect the sample index beyond which backscatter climbs back up
    after decaying into its real noise floor.

    Works on the *uncorrected* signal: any recorded radiometric steps are
    inverted first (:func:`sidescanpy.processing.inversion.uncorrected_backscatter`),
    so it can run before or after TVG. Bin statistics are accumulated in
    ping blocks, so memory stays bounded on large (or dask-backed) datasets.

    Bins the uncorrected signal (converted to dB) by sample index across
    ``n_bins`` roughly-equal bins, finds the peak bin (the strongest
    mean return) among bins with enough finite samples to be
    statistically reliable, then finds the minimum-mean bin *after* that
    peak (the real far-range floor — searched only post-peak so a
    legitimate near-nadir ramp-up is never mistaken for the floor). If any
    later, equally-reliable bin's mean then rises ``rise_threshold_db`` or
    more above that floor, returns a cropped sample count (the floor bin's
    end, plus ``margin_bins`` of slack). If no such rise is found, returns
    ``None`` — nothing should be cropped.

    A cutoff whose slant range is not beyond the median ``altitude_m`` is
    rejected with a WARNING and ``None`` is returned: on recordings that
    include the transmit pulse and the water column, the "floor" found is
    the water column and the "rise" is the seafloor return.

    Parameters
    ----------
    ds : xr.Dataset
        A single frequency channel dataset.
    n_bins : int, default 40
        Number of sample-index bins to divide the sample axis into.
    min_valid_fraction : float, default 0.5
        A bin is "reliable" only if at least this fraction of its
        ``n_pings * n_channels * bin_size`` cells are finite.
    rise_threshold_db : float, default 3.0
        How far above the detected floor a later, reliable bin's mean
        must climb to count as "the noise climbing back up".
    margin_bins : int, default 1
        Extra bins kept past the detected floor bin.

    Returns
    -------
    int or None
        Sample count to crop to (``ds.isel(sample=slice(0, result))``),
        or ``None`` if no crop is warranted.
    """
    n_pings = ds.sizes["ping"]
    n_channels = ds.sizes["channel"]
    n_samples = ds.sizes["sample"]

    bin_size = max(1, n_samples // n_bins)
    n_bins_actual = n_samples // bin_size
    if n_bins_actual < 3:
        return None

    sums, bin_counts = _bin_stats(ds, bin_size, n_bins_actual)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_means = np.where(bin_counts > 0, sums / np.maximum(bin_counts, 1), np.nan)
    min_count = int(n_pings * n_channels * bin_size * min_valid_fraction)

    reliable = bin_counts >= min_count
    if not reliable.any():
        return None

    reliable_idx = np.where(reliable)[0]

    # The floor must be searched for only *after* the peak, not globally.
    # A legitimate near-nadir ramp-up (weak returns very close to nadir,
    # before reaching the real peak a few bins in) can be *lower* than the
    # genuine far-range floor; a global-minimum search would pick it and
    # read the normal rise to the real peak as a huge spurious "rise".
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
    if _cutoff_inside_water_column(ds, cutoff_sample):
        return None
    return cutoff_sample


def _cutoff_inside_water_column(ds: xr.Dataset, cutoff_sample: int) -> bool:
    """True when the cutoff's slant range is not beyond the median altitude.

    A recording that starts at the transmit pulse has its strongest bin at
    the start and a quiet water column before the first bottom return; the
    detector then reads the water column as the floor and the seafloor as
    the noise rising back up. Such a cutoff is rejected (WARNING) rather
    than cropping away the whole swath.
    """
    if "altitude_m" not in ds or "sound_vel_ms" not in ds or "sample_rate_hz" not in ds.attrs:
        return False
    altitude = float(np.nanmedian(ds["altitude_m"].values)) if np.isfinite(ds["altitude_m"].values).any() else np.nan
    if not np.isfinite(altitude) or altitude <= 0:
        return False
    with invalid_sound_vel_warning_suppressed():  # georeferencing warns about it where it matters
        spacing = float(np.nanmedian(sample_interval_m(ds)))
    cutoff_range = spacing * sample_index(ds)[cutoff_sample - 1]
    if cutoff_range > altitude:
        return False
    logger.warning(
        "far-range cutoff at sample %d (%.1f m slant range) is inside the water column "
        "(median altitude %.1f m); not cropping",
        cutoff_sample, cutoff_range, altitude,
    )
    return True


@per_frequency
def crop_far_range(
    ds: xr.Dataset,
    *,
    cutoff_sample: int | None = None,
    n_bins: int = 40,
    min_valid_fraction: float = 0.5,
    rise_threshold_db: float = 3.0,
    margin_bins: int = 1,
) -> xr.Dataset:
    """Crop one channel's far-range noise (the ``"crop_farrange"`` step).

    Detects the cutoff with :func:`detect_far_range_cutoff_sample` (the
    detection keyword arguments are passed through) unless ``cutoff_sample``
    is given. A channel with no detected artifact is left untouched. Either
    way a ``"crop_farrange"`` history record is appended (with the resolved
    ``cutoff_sample``, ``None`` when nothing was cropped), so a later run
    knows detection already happened.

    Not invertible: cropped samples are gone (re-convert from the source
    file to get them back).

    ``ds`` may also be a :class:`~sidescanpy.core.survey.SideScanSurvey`:
    each frequency is cropped at its own detected cutoff and a new survey
    is returned (the input is not modified).
    """
    n_samples = ds.sizes["sample"]
    if cutoff_sample is None:
        cutoff_sample = detect_far_range_cutoff_sample(
            ds, n_bins=n_bins, min_valid_fraction=min_valid_fraction,
            rise_threshold_db=rise_threshold_db, margin_bins=margin_bins,
        )
    cropped = cutoff_sample is not None and cutoff_sample < n_samples
    if cropped:
        out = ds.isel(sample=slice(0, cutoff_sample))
        logger.debug("cropped far-range noise: %d -> %d samples", n_samples, cutoff_sample)
    else:
        out = ds
        logger.debug("no far-range noise artifact detected, left unchanged")
    return append_history(
        out,
        STEP_CROP_FARRANGE,
        {
            "n_bins": n_bins,
            "min_valid_fraction": min_valid_fraction,
            "rise_threshold_db": rise_threshold_db,
            "margin_bins": margin_bins,
            "cutoff_sample": int(cutoff_sample) if cropped else None,
            "original_n_samples": int(n_samples),
            "cropped": bool(cropped),
        },
        function="sidescanpy.processing.farrange.crop_far_range",
        version=__version__,
    )


def crop_far_range_noise(
    survey: SideScanSurvey,
    **detect_kwargs,
) -> tuple[SideScanSurvey, dict[str, int | None]]:
    """Apply :func:`crop_far_range` to every frequency channel in ``survey``.

    Parameters
    ----------
    survey : SideScanSurvey
        Mutated in place (frequency channels replaced) and returned.
    **detect_kwargs
        Passed through to :func:`crop_far_range`.

    Returns
    -------
    (survey, cutoffs)
        ``cutoffs``: ``freq_key -> cropped sample count``, or ``None``
        for channels left unchanged.
    """
    cutoffs: dict[str, int | None] = {}
    for freq_key, channel in list(survey.freq_channels.items()):
        ds = crop_far_range(channel.dataset, **detect_kwargs)
        cutoffs[freq_key] = read_history(ds)[-1]["params"]["cutoff_sample"]
        survey.add_frequency(freq_key, ds)
    return survey, cutoffs
