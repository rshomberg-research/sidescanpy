"""
Export styles, render levels, and the backscatter variables each level reads.

A *style* is the window shape (``chip``: a tight single-channel crop;
``single-swath``: one full channel; ``dual-swath``: both channels side by
side). A *render level* is the processing state and coordinate system the
window is rendered in: ``uncorrected``/``db``/``radiometric`` crop native
sample/ping pixels; ``geometric``/``georef`` resample onto a real-meters
grid.

Neither ``uncorrected`` nor ``backscatter_db`` is stored on disk: they are
derived per loaded neighborhood by :func:`with_uncorrected_var` and
:func:`with_db_var`.

2026 rshomberg@uri.edu
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import xarray as xr

from sidescanpy.core.units import linear_to_db
from sidescanpy.processing.bottom import mask_nadir_gap
from sidescanpy.processing.inversion import UNCORRECTED_VAR, uncorrected_backscatter
from sidescanpy.processing.sample_axis import (
    invalid_sound_vel_warning_suppressed,
    sample_interval_m,
)

logger = logging.getLogger(__name__)

CHIP_STYLES = ("chip", "single-swath", "dual-swath")
"""Export window styles."""

RENDER_LEVELS = ("uncorrected", "db", "radiometric", "geometric", "georef")
"""Render levels, in correction order."""

PIXEL_LEVELS = ("uncorrected", "db", "radiometric")
"""Render levels cropped in native sample/ping pixels (no ``resolution_m``)."""

LEVEL_VARS = {"uncorrected": UNCORRECTED_VAR, "db": "backscatter_db", "radiometric": "backscatter_intensity"}
"""Value name (pixel render level, or geometric/georef ``value_var``) -> backscatter variable."""


def with_uncorrected_var(ds: xr.Dataset) -> xr.Dataset:
    """Add the ``uncorrected`` variable unless ``ds`` already has it.

    ``uncorrected`` is ``backscatter_intensity`` with every recorded
    radiometric step undone (see
    :func:`~sidescanpy.processing.inversion.uncorrected_backscatter`); lazy
    on dask-backed data.
    """
    if UNCORRECTED_VAR in ds:
        return ds
    return ds.assign({UNCORRECTED_VAR: uncorrected_backscatter(ds)})


def mask_nadir_gap_with_uncorrected(ds: xr.Dataset) -> xr.Dataset:
    """:func:`~sidescanpy.processing.bottom.mask_nadir_gap`, plus ``uncorrected``.

    ``uncorrected`` is derived from the masked intensity, so both carry the
    same NaN nadir gap.
    """
    return with_uncorrected_var(mask_nadir_gap(ds, value_vars=("backscatter_intensity", UNCORRECTED_VAR)))


def with_db_var(ds: xr.Dataset) -> xr.Dataset:
    """Add ``backscatter_db`` (``20*log10`` of the uncorrected signal, no TVG gain).

    The same raw-to-dB step :func:`~sidescanpy.processing.radiometric.apply_tvg_to_amplitude`
    takes before adding gain.
    """
    if "backscatter_db" in ds:
        return ds
    return ds.assign(backscatter_db=linear_to_db(with_uncorrected_var(ds)[UNCORRECTED_VAR]))


def nominal_sample_interval_m(ds: xr.Dataset) -> float:
    """Slant-range meters per sample at ``ds``'s first ping.

    Uses ``sample_rate_hz`` from ``ds.attrs`` (block-averaged neighborhoods
    divide it by their sample factor, so this reports the averaged cell
    size). Invalid sound velocity falls back to 1500 m/s as in
    :func:`~sidescanpy.processing.sample_axis.sample_interval_m`, without
    its per-call warning (this is called once per exported window).
    """
    with invalid_sound_vel_warning_suppressed():
        return float(sample_interval_m(ds.isel(ping=slice(0, 1)))[0])


@dataclass(frozen=True)
class WindowSpec:
    """How one export style is rendered.

    Attributes
    ----------
    style : str
        One of :data:`CHIP_STYLES`.
    level : str
        One of :data:`RENDER_LEVELS`.
    size_px : int
        Output size (both axes).
    jitter_px : int, default 0
        Uniform center jitter, in output pixels.
    resolution_m : float, optional
        Geometric/georef meters per output pixel; ``None`` chooses it per
        window (see :func:`~sidescanpy.export.windows.geo.extract_geo_style_window`).
    value_var : str, default "radiometric"
        Geometric/georef value (a key of :data:`LEVEL_VARS`). Pixel levels
        read the level's own variable.
    resample_method : str, default "griddata"
        Geometric resampler.
    across_axis : str, default "ground"
        Geometric across-track axis (``"ground"`` or ``"slant"``).
    swath_fill : bool, default False
        Geometric swath styles: frame the across-track extent the data fills.
    """

    style: str
    level: str
    size_px: int
    jitter_px: int = 0
    resolution_m: float | None = None
    value_var: str = "radiometric"
    resample_method: str = "griddata"
    across_axis: str = "ground"
    swath_fill: bool = False

    @property
    def is_pixel(self) -> bool:
        """Whether the level crops native sample/ping pixels."""
        return self.level in PIXEL_LEVELS

    @property
    def backscatter_var(self) -> str:
        """Variable a geometric/georef window resamples."""
        return LEVEL_VARS[self.value_var]

    @property
    def needs_db(self) -> bool:
        """Whether ``backscatter_db`` must be derived for this spec."""
        return "db" in (self.level, self.value_var)
