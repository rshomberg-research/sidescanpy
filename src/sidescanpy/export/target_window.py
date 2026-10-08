"""
Target-window correction: TVG then display orientation.

Composes the decode-only reader in :mod:`sidescanpy.io.jsf.targets` with
:func:`~sidescanpy.processing.radiometric.apply_tvg_to_amplitude` and
:func:`~sidescanpy.export.orientation.orient_for_display`.

2026 rshomberg@uri.edu
"""

import logging

import numpy as np

from sidescanpy.core.target import Target
from sidescanpy.export.orientation import orient_for_display
from sidescanpy.io.jsf.targets import decode_target_window
from sidescanpy.processing.radiometric import apply_tvg_to_amplitude

logger = logging.getLogger(__name__)


def decode_and_correct_window(target: Target, *, freq_hz: float) -> tuple[np.ndarray, Target]:
    """Decode ``target``'s window and run it through the same
    TVG -> orient-for-display chain used everywhere this project renders
    a target crop.

    Does not apply :func:`~sidescanpy.export.orientation.detrend_by_range`:
    the per-column median it subtracts is present whether or not a target
    is there, so at crop scale it mostly flattens real seafloor texture.

    Lives in ``export`` (not ``io.jsf.targets``) because TVG is a
    processing step and orientation an export convention; ``io`` only
    decodes.

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
