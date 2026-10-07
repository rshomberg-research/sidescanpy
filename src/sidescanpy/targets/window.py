"""
Target-window correction: TVG then display orientation.

Composes the decode-only reader in :mod:`sidescanpy.io.jsf.targets` with
:func:`~sidescanpy.processing.radiometric.apply_tvg_to_amplitude` and
:func:`~sidescanpy.targets.core.orient_for_display`.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np

from sidescanpy.io.jsf.targets import decode_target_window
from sidescanpy.processing.radiometric import apply_tvg_to_amplitude
from sidescanpy.targets.core import Target, orient_for_display

logger = logging.getLogger(__name__)


def decode_and_correct_window(target: Target, *, freq_hz: float) -> tuple[np.ndarray, Target]:
    """Decode ``target``'s window and run it through the same
    TVG -> orient-for-display chain used everywhere this project renders
    a target crop.

    Deliberately does not apply ``targets.core.detrend_by_range``: the
    per-column median swing it subtracts (3-12 dB on real crops) is
    present whether or not a target is there, so it mostly flattens real
    seafloor texture at crop scale rather than correcting a range residual.
    ``detrend_by_range`` remains available in
    ``targets.core`` for any caller that wants it.

    Lives here (not in ``io.jsf.targets``) because TVG is a processing
    step; ``io`` only decodes. Not re-exported from ``sidescanpy.targets``
    to avoid a circular import with ``io.jsf.targets``.

    Parameters
    ----------
    target : Target
        Must have ``extra_attributes["ping_meters_per_sample"]`` (set by
        ``read_discover_targets``).
    freq_hz : float
        Acquisition frequency for the TVG absorption term — this
        survey's target catalogs don't record it per-target, so callers
        pass their own known constant (see ``absorption_for_frequency``).

    Returns
    -------
    (window, target)
        TVG-corrected, display-oriented window, and a copy of ``target``
        with ``.sample``/``extra_attributes["window_row"]`` remapped to
        match ``orient_for_display``'s flips (see that function's
        docstring).
    """
    return correct_decoded_window(decode_target_window(target), target, freq_hz=freq_hz)


def correct_decoded_window(window: np.ndarray, target: Target, *, freq_hz: float) -> tuple[np.ndarray, Target]:
    """Apply TVG then display orientation to an already-decoded window.

    Second half of :func:`decode_and_correct_window`; see its docstring for
    the parameters and return value (``window`` is the raw array from
    ``io.jsf.targets.decode_target_window``).
    """
    logger.debug(f"Correcting target {target.target_id} window {window.shape} at {freq_hz} Hz")
    window = apply_tvg_to_amplitude(
        window, target.extra_attributes["ping_meters_per_sample"], freq_hz=freq_hz,
    )
    window, target = orient_for_display(window, target)
    return window, target
