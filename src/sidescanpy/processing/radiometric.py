"""
Radiometric correction for sidescan sonar data.

This module implements Time-Varying Gain (TVG) correction to compensate for
transmission losses in the water column. The raw backscatter_intensity values
represent echo levels received by the transducer but do not account for:

1. Geometric spreading: sound spreads in 3D, losing energy proportional to 1/R²
   (in dB: 20*log10(R) dB loss)
2. Absorption: water absorbs acoustic energy proportional to distance traveled
   (in dB: 2*alpha*R dB loss for two-way path)

TVG correction adds these losses back, so equal backscatter at different ranges
produces equal corrected values. This is essential for radiometric calibration
and mosaic generation.

Raw backscatter from this sonar is in LINEAR AMPLITUDE (0-81 range, float32).
This module converts to dB, applies TVG, and outputs in dB.

2026 rshomberg@uri.edu
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass
class TVGParams:
    """
    Parameters for time-varying gain correction.

    Attributes
    ----------
    spreading_loss_db : float, default=20.0
        Spreading loss coefficient (dB per decade of range).
        Standard value is 20.0 for spherical spreading in 3D.
    absorption_db_per_m : float, optional
        Absorption coefficient (dB per meter).
        If None, uses frequency-dependent default from absorption_for_frequency().
    min_range_m : float, default=0.5
        Minimum range threshold (meters).
        Ranges below this are clipped to avoid log(0) singularity.
        Typical value 0.5 m accounts for transducer blanking.
    """

    spreading_loss_db: float = 20.0
    absorption_db_per_m: float = None
    min_range_m: float = 0.5


def absorption_for_frequency(freq_hz: float) -> float:
    """
    Return a reasonable default absorption coefficient (dB/m) by frequency.

    Uses simplified Francois-Garrison values for typical coastal seawater
    (T=10°C, S=35 ppt, pH=8, D=10 m). For precise values, compute using
    the full Francois-Garrison formula with CTD data.

    Parameters
    ----------
    freq_hz : float
        Sonar transmit frequency (Hz).

    Returns
    -------
    float
        Absorption coefficient (dB/m).

    Notes
    -----
    Francois-Garrison formula is implemented as:
        alpha = A1*P1*f1*f^2/(f1^2 + f^2) + A2*P2*f2*f^2/(f2^2 + f^2) + A3*P3*f^2
    Where:
        - First term: boric acid relaxation (~1 kHz)
        - Second term: MgSO4 relaxation (~100 kHz, dominant for 500-1500 kHz)
        - Third term: pure water viscosity

    Reference values (typical coastal seawater):
        - 100 kHz: ~0.01 dB/m
        - 547.5 kHz: ~0.10 dB/m
        - 1495 kHz: ~0.20 dB/m
    """
    # TODO: replace with more accurate function
    if freq_hz >= 1_000_000:
        # ~1.5 MHz and above
        return 0.20
    elif freq_hz >= 400_000:
        # ~547.5 kHz
        return 0.10
    else:
        # Lower frequencies
        return 0.05


def tvg_gain_db(range_m: np.ndarray, params: TVGParams) -> np.ndarray:
    """
    Per-range TVG gain (dB): spreading loss + absorption loss.

    ``spreading_loss (dB) = spreading_loss_db * log10(range_m)``
    ``absorption_loss (dB) = 2 * absorption_db_per_m * range_m`` (factor of
    2 for the two-way path)

    Shared by ``apply_tvg()`` (xarray survey pipeline) and
    ``apply_tvg_to_amplitude()`` (plain-array path, e.g. target crop
    windows) so both apply the identical formula rather than maintaining
    it twice. Caller is responsible for clipping ``range_m`` to
    ``params.min_range_m`` first (as both callers here do) — this
    function doesn't clip on its own since it has no opinion on what a
    caller's minimum range should be.
    """
    spreading = params.spreading_loss_db * np.log10(range_m)
    absorption = 2.0 * params.absorption_db_per_m * range_m
    return spreading + absorption


def apply_tvg_to_amplitude(
    amplitude: np.ndarray,
    meters_per_sample: float,
    *,
    params: TVGParams | None = None,
    freq_hz: float | None = None,
) -> np.ndarray:
    """
    Apply TVG correction to a plain linear-amplitude array — no xarray
    ``Dataset`` required.

    For use on data that isn't a full survey channel (e.g. a target's
    raw ping-sample window decoded straight from its catalog SQLite via
    ``sidescanpy.io.jsf.targets.decode_target_window``), where
    constructing a full ``SideScanSurvey``/``Dataset`` just to TVG-correct
    a small crop isn't warranted. Same formula and defaults as
    ``apply_tvg()`` — see that function for the physical justification.

    Parameters
    ----------
    amplitude : np.ndarray
        Linear amplitude, sample axis last (e.g. ``(n_pings, n_samples)``
        or ``(n_samples,)``).
    meters_per_sample : float
        Range covered by one sample (two-way), e.g. a target catalog's
        own ``PingMetersPerSample`` — the per-sample analog of
        ``apply_tvg()``'s ``sound_velocity / (2 * sample_rate)``.
    params : TVGParams, optional
        If ``None``, built from defaults with ``absorption_db_per_m``
        derived from ``freq_hz`` (required in that case, matching
        ``apply_tvg()``'s own default-resolution pattern).
    freq_hz : float, optional
        Used only when ``params`` is ``None`` (or has no
        ``absorption_db_per_m`` set), to derive a default via
        ``absorption_for_frequency()``.

    Returns
    -------
    np.ndarray
        TVG-corrected values in dB, same shape as ``amplitude``.
    """
    if params is None:
        if freq_hz is None:
            raise ValueError(
                "apply_tvg_to_amplitude requires either `params` with "
                "absorption_db_per_m set, or `freq_hz` to derive a default"
            )
        params = TVGParams(absorption_db_per_m=absorption_for_frequency(freq_hz))
    elif params.absorption_db_per_m is None:
        if freq_hz is None:
            raise ValueError(
                "params.absorption_db_per_m is None and no `freq_hz` was "
                "given to derive a default"
            )
        params = TVGParams(
            spreading_loss_db=params.spreading_loss_db,
            absorption_db_per_m=absorption_for_frequency(freq_hz),
            min_range_m=params.min_range_m,
        )

    n_samples = amplitude.shape[-1]
    amplitude_db = 20.0 * np.log10(np.clip(amplitude, 1e-6, None))
    range_m = np.arange(n_samples, dtype=np.float64) * meters_per_sample
    range_m = np.clip(range_m, params.min_range_m, None)
    gain_db = tvg_gain_db(range_m, params)
    return amplitude_db + gain_db


def apply_tvg(ds: xr.Dataset, params: TVGParams | None = None) -> xr.Dataset:
    """
    Apply time-varying gain (TVG) correction to raw backscatter_intensity.

    Compensates for geometric spreading and acoustic absorption losses.
    Converts raw linear amplitude to dB, applies TVG gain, and returns
    corrected dB values.

    **IMPORTANT**: Raw backscatter_intensity is assumed to be in LINEAR AMPLITUDE
    (0-81 range, float32). This function converts to dB internally.

    Conversion: 20 * log10(clip(linear_value, 1e-6))
    - Values < 1e-6 are clipped to prevent log(0)
    - Maps to approximately -120 dB (well below noise floor)
    - Preserves zero values appropriately (water column)

    **KNOWN ISSUE (unconfirmed root cause, under investigation as of
    2026-07-02):** the "water column" assumption above for the clipped-
    to-floor population is not yet confirmed. On real data, ~1% of
    samples hit this exact floor, clustered in a fixed leading run of
    ~50-60 samples at the *start* of every ping (not scattered through
    the range, not correlated with `altitude_m`), and — critically —
    `ground_range_m` at those samples does *not* collapse toward zero
    the way genuine water-column returns should (see
    `processing/geometric.py`'s `compute_geometric_ranges` masking).
    Whatever these samples are, the existing water-column mask isn't
    catching them, so they pass `resample_channel_to_grid()`'s
    finite-coordinate filter and get included in gridding — see
    `docs/planning/handoff-tvg-range-residual.md` and `CLAUDE.md` Known
    Issues #4-#5 for the full evidence and a related, separately
    confirmed ~37 dB range-dependent TVG residual found in the same
    investigation.

    Parameters
    ----------
    ds : xr.Dataset
        Valid frequency channel dataset with processing_level == "raw".
        Must have:
        - backscatter_intensity (ping, channel, sample) in linear amplitude
        - sound_vel_ms (ping) - sound velocity per ping
        - altitude_m (ping) - altitude above bottom
        Attributes must include:
        - freq_hz_start, sample_rate_hz (from schema)
        - processing_level == "raw"
        - processing_history (JSON string)

    params : TVGParams, optional
        TVG parameters. If None, uses frequency-appropriate defaults.
        If params.absorption_db_per_m is None, derives from freq_hz_start.

    Returns
    -------
    xr.Dataset
        New dataset with:
        - backscatter_intensity: TVG-corrected dB values
        - backscatter_raw: original linear values (for reference)
        - processing_level: "radiometric"
        - processing_history: appended with TVG step

    Raises
    ------
    ValueError
        If processing_level is not "raw" (prevents double-application).

    Examples
    --------
    >>> import xarray as xr
    >>> from sidescanpy.processing.radiometric import apply_tvg, TVGParams
    >>>
    >>> # Load raw data
    >>> ds_raw = xr.open_dataset("survey.ss.h5", group="frequencies/hz_1495000",
    ...                           engine="h5netcdf")
    >>>
    >>> # Apply TVG with defaults
    >>> ds_tvg = apply_tvg(ds_raw)
    >>>
    >>> # Or specify custom absorption
    >>> params = TVGParams(absorption_db_per_m=0.25)
    >>> ds_tvg = apply_tvg(ds_raw, params)
    >>>
    >>> # Save corrected data
    >>> ds_tvg.to_netcdf("survey.radiometric.nc")
    """
    # Validate input dataset
    ds.sidescan.validate()

    # Check processing level
    if ds.attrs["processing_level"] != "raw":
        raise ValueError(
            f"TVG expects processing_level 'raw', got '{ds.attrs['processing_level']}'. "
            "Has TVG already been applied?"
        )

    # Set default parameters if needed
    if params is None:
        freq_hz = ds.attrs["freq_hz_start"]
        alpha = absorption_for_frequency(freq_hz)
        params = TVGParams(absorption_db_per_m=alpha)
    elif params.absorption_db_per_m is None:
        params.absorption_db_per_m = absorption_for_frequency(
            ds.attrs["freq_hz_start"]
        )

    logger.info(
        "Applying TVG: freq=%s Hz, spreading=%.1f dB/dec, absorption=%.4f dB/m",
        ds.attrs["freq_hz_start"],
        params.spreading_loss_db,
        params.absorption_db_per_m,
    )

    # ========================================================================
    # Step 1: Convert raw linear amplitude to dB
    # ========================================================================
    # Raw values are in range 0-81 (float32)
    # Convert to dB: 20*log10(linear_value)
    # Clip minimum value to avoid log(0): values < 1e-6 map to ~-120 dB
    raw_linear = ds["backscatter_intensity"]
    raw_clipped = raw_linear.clip(min=1e-6)
    backscatter_db = 20.0 * np.log10(raw_clipped)

    logger.debug(
        "Linear→dB conversion: min=%.2f dB, max=%.2f dB, mean=%.2f dB",
        float(backscatter_db.min()),
        float(backscatter_db.max()),
        float(backscatter_db.mean()),
    )

    # ========================================================================
    # Step 2: Compute slant range array for each sample
    # ========================================================================
    # Range for sample i: R[i] = i * (sound_velocity / (2 * sample_rate))
    sample_rate = ds.attrs["sample_rate_hz"]
    n_samples = ds.sizes["sample"]
    sound_velocity = ds["sound_vel_ms"].mean().item()  # scalar, m/s
    # FIXME: sound_velocity should change at each ping
    # 2026-07-16: widened to also catch exactly-zero (not just NaN) —
    # confirmed on real JSF data that sound_vel_ms is exactly 0.0 for 100%
    # of pings in at least one survey, which silently pinned every
    # sample's slant range at min_range_m (see
    # docs/planning/handoff-tvg-sound-velocity-zero-bug.md). Matches
    # geometric.py's already-correct `invalid_sv` check.
    if np.isnan(sound_velocity) or sound_velocity <= 0:
        sound_velocity = 1500.0

    sample_interval_m = sound_velocity / (2.0 * sample_rate)
    slant_range = np.arange(n_samples, dtype=np.float32) * sample_interval_m
    slant_range = np.clip(slant_range, params.min_range_m, None)

    logger.debug(
        "Slant range: sample_interval=%.4f m, max_range=%.1f m",
        sample_interval_m,
        float(slant_range[-1]),
    )

    # ========================================================================
    # Step 3: Compute TVG gain (dB) for each sample
    # ========================================================================
    # TVG = spreading_loss + absorption_loss — see tvg_gain_db() docstring
    tvg_db = tvg_gain_db(slant_range, params)

    logger.debug(
        "TVG gains: min=%.2f dB, max=%.2f dB (at %.1f m)",
        float(tvg_db[0]),
        float(tvg_db[-1]),
        float(slant_range[-1]),
    )

    # FIXME: I am not 100% sure that the backscatter values are in intensity.
    # They could be in -db. The range is about 0-80, while db expectations would
    # be -80 to 0

    # ========================================================================
    # Step 4: Apply TVG correction
    # ========================================================================
    # corrected = backscatter_db + tvg_db
    # tvg_db has shape (n_samples,) and broadcasts over (ping, channel, sample)
    corrected_db = backscatter_db + xr.DataArray(tvg_db, dims=["sample"])
    # corrected_db = np.clip(corrected_db, 0, None)   # FIXME: fix the clipping

    logger.debug(
        "After TVG: min=%.2f dB, max=%.2f dB, mean=%.2f dB",
        float(corrected_db.min()),
        float(corrected_db.max()),
        float(corrected_db.mean()),
    )

    # ========================================================================
    # Step 5: Build output dataset
    # ========================================================================
    ds_out = ds.copy(deep=False)             # shallow — share refs
    ds_out.attrs = dict(ds.attrs)             # shallow but with our own attrs dict
    ds_out = ds_out.assign(
        backscatter_intensity=corrected_db,    # replaces ref to old array
    )    
    # ds_out["backscatter_intensity"] = corrected_db
    # ds_out["backscatter_raw"] = backscatter_db  # Store dB version for reference

    # ========================================================================
    # Step 6: Update processing provenance
    # ========================================================================
    history = json.loads(ds.attrs["processing_history"])
    history.append(
        {
            "step": "tvg_correction",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "params": {
                "spreading_loss_db": params.spreading_loss_db,
                "absorption_db_per_m": params.absorption_db_per_m,
                "min_range_m": params.min_range_m,
                "sound_velocity_source": "mean_of_per_ping_values",
                "sound_velocity_used_ms": float(sound_velocity),
            },
            "unit_conversion": {
                "raw_format": "linear_amplitude (0-81)",
                "conversion": "20*log10(clip(value, 1e-6))",
                "output_format": "dB",
            },
            "function": "sidescanpy.processing.radiometric.apply_tvg",
        }
    )
    ds_out.attrs["processing_level"] = "radiometric"
    ds_out.attrs["processing_history"] = json.dumps(history)

    logger.info(
        "TVG applied successfully. Output: processing_level='radiometric', "
        "processing_history appended with step 'tvg_correction'."
    )

    return ds_out